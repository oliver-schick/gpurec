#!/usr/bin/env python
"""Unified gpurec fit CLI — one general driver, no dataset-specific code.

Give it a species tree + gene families (``.ale`` CCPs OR gene-tree samples) and it
fits branch-wise DTL(+origination) rates by ML, with optional clade-grouping, an
origination anti-concentration barrier, and a Brownian rate prior. Multi-GPU via
``torchrun`` (family-sharded data parallel). Replaces run_williams/undine_branchwise.

  # .ale families, simplest:
  python -m experiments.gpurec_fit --species sp.nwk --genes 'ccps/*.ale' --out out.rates.txt

  # branch-wise + origination barrier + clade grouping straight from the tree:
  python -m experiments.gpurec_fit --species sp.nwk --genes genes_dir \\
      --origination optimize --barrier-kind simpson --barrier-c 30000 \\
      --clade-groups-from-tree --out out.rates.txt --per-family-out perfam.json

  # custom gene->genome map (else: species = leaf prefix before the first '_'):
  python -m experiments.gpurec_fit --species sp.nwk --genes 'g/*.ale' --gene-map map.tsv ...

  # 4 GPUs:
  torchrun --standalone --nproc_per_node=4 -m experiments.gpurec_fit --species ... --genes ...
"""
import argparse, glob, sys, math, json, time, os
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (
    _load_species_helpers, _build_wave_layout, _sp_helpers_for_uniform,
    _build_leaf_E, _parse_fraction_missing,
)
from gpurec import distributed as ddp

_LN2 = math.log(2.0)
_THETA_MIN = math.log2(1e-10)


# ── inputs ────────────────────────────────────────────────────────────────────
def _resolve_genes(specs):
    """globs / directories / explicit files -> sorted, de-._-ed file list."""
    out = []
    for s in specs:
        p = Path(s)
        if p.is_dir():
            out += [str(x) for x in p.iterdir() if not x.name.startswith("._")]
        elif any(c in s for c in "*?["):
            out += [x for x in glob.glob(s) if not Path(x).name.startswith("._")]
        else:
            out.append(s)
    return sorted(set(out))


def _make_species_of_leaf(gene_map_path, sep):
    """species(leaf) = map[leaf] if a --gene-map given else the prefix before `sep`."""
    table = {}
    if gene_map_path:
        for ln in Path(gene_map_path).read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.replace("\t", " ").split()
            if len(parts) >= 2:
                table[parts[0]] = parts[1]
    def species_of_leaf(leaf):
        if leaf in table:
            return table[leaf]
        return leaf.split(sep)[0] if sep else leaf
    return species_of_leaf


def _infer_format(paths, fmt):
    if fmt != "auto":
        return fmt
    exts = {Path(p).suffix.lower() for p in paths[:50]}
    return "ale" if ".ale" in exts else "genetrees"


def _load_ale(paths, species_name_to_index, species_of_leaf, min_species, dtype):
    """Load .ale CCP families with a CUSTOM species_of_leaf (so --gene-map / --species-sep
    work). Faithfully mirrors run_williams._load_families (incl. the ln->log2 conversion
    of log_split_probs and the GeneDataset-shaped family dict) + injectable mapping."""
    from gpurec.io.ale import parse_ale_file, build_family_from_ale
    inv_ln2 = 1.0 / math.log(2.0)
    fams, names, kept, drop_sp, drop_min, fail = [], [], 0, 0, 0, 0
    for path in paths:
        try:
            ale = parse_ale_file(path)
        except Exception:
            fail += 1; continue
        sp_set = {species_of_leaf(n) for n in ale.leaf_name_to_id}
        if any(s not in species_name_to_index for s in sp_set):
            drop_sp += 1; continue
        if len(sp_set) < min_species:
            drop_min += 1; continue
        try:
            fam = build_family_from_ale(ale, species_name_to_index,
                                        species_of_leaf=species_of_leaf, dtype=dtype)
        except Exception:
            fail += 1; continue
        ccp = fam["ccp"]
        ccp["log_split_probs_sorted"] = ccp["log_split_probs_sorted"] * inv_ln2  # ln -> log2 once
        fams.append({"ccp_helpers": ccp, "root_clade_id": int(fam["root_clade_id"]),
                     "leaf_row_index": fam["leaf_row_index"], "leaf_col_index": fam["leaf_col_index"],
                     "C": int(ccp["C"]), "N_splits": int(ccp["N_splits"])})
        names.append(Path(path).stem); kept += 1
    return fams, names, {"kept": kept, "dropped_min_species": drop_min,
                         "dropped_unknown_species": drop_sp, "parse_failures": fail}


def _load_genetrees(species_path, paths, dtype, limit):
    """Gene-tree samples -> CCP families via the C++ amalgamator (GeneDataset path).
    Species mapping is the C++ default (prefix before first '_')."""
    from gpurec.core.model import GeneDataset
    ds = GeneDataset(species_tree_path=species_path,
                     gene_tree_paths=paths if limit <= 0 else paths[:limit], dtype=dtype)
    names = list(getattr(ds, "family_names", [f"family_{i}" for i in range(len(ds.families))]))
    return list(ds.families), names, {"kept": len(ds.families)}


# ── model setup ─────────────────────────────────────────────────────────────────
def _build_groups(args, species_helpers, tree_path, S, device, dtype):
    """Return (group_index|None, omega_group_index|None, info)."""
    group_index = omega_group_index = None
    info = {}
    if args.clade_groups_from_tree:
        from clade_groups import (tree_clade_group_index, tree_origination_group_index,
                                  BIGTREE_DTL_CLADES)
        wl = BIGTREE_DTL_CLADES if args.dtl_clades == "paper" else None
        group_index, labels = tree_clade_group_index(species_helpers, str(tree_path), S,
                                                     clade_whitelist=wl)
        group_index = group_index.to(device)
        info["dtl_classes"] = int(group_index.max().item()) + 1
        info["dtl_labels"] = labels
        if args.origination == "optimize":
            omega_group_index, n_o = tree_origination_group_index(species_helpers, str(tree_path), S)
            omega_group_index = omega_group_index.to(device)
            info["o_classes"] = n_o
    elif args.clade_groups:
        from clade_groups import (group_index_for_species_helpers,
                                  omega_group_index_for_species_helpers)
        cg = Path(args.clade_groups)
        tr = cg / "species_trees" / "starting_species_tree.newick"
        mp = cg / "model_parameters" / "model_parameters.txt"
        group_index, _cat, _orig, _diag = group_index_for_species_helpers(species_helpers, str(tr), str(mp))
        group_index = group_index.to(device)
        info["dtl_classes"] = int(group_index.max().item()) + 1
        if args.origination == "optimize":
            omega_group_index, n_o = omega_group_index_for_species_helpers(species_helpers, str(tr), str(mp))
            if omega_group_index is not None:
                omega_group_index = omega_group_index.to(device); info["o_classes"] = n_o
    return group_index, omega_group_index, info


def _build_barrier_pi(args, species_helpers, parent_index, S, device, dtype):
    if args.origination != "optimize" or args.barrier_c <= 0:
        return None
    _par = parent_index.tolist()
    depth = [0] * S
    for e in range(S):
        d, cur, seen = 0, e, 0
        while _par[cur] >= 0 and _par[cur] != cur and seen <= S:
            d += 1; cur = _par[cur]; seen += 1
        depth[e] = d
    rho = float(args.barrier_rho)
    v = torch.tensor([float(args.barrier_decay) ** (-d) for d in depth], dtype=dtype, device=device)
    v = v / v.sum()
    return (1.0 - rho) / S + rho * v


# ── main ────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified gpurec branch-wise fit.")
    g = ap.add_argument_group("inputs")
    g.add_argument("--species", required=True, help="rooted species tree (newick)")
    g.add_argument("--genes", required=True, nargs="+", help="globs / dirs / files of .ale or gene trees")
    g.add_argument("--gene-format", choices=["auto", "ale", "genetrees"], default="auto")
    g.add_argument("--gene-map", default=None, help="TSV: gene_leaf<TAB>genome (else split on --species-sep)")
    g.add_argument("--species-sep", default="_", help="leaf->species separator (default '_', ALE style)")
    g.add_argument("--min-species", type=int, default=1)
    g.add_argument("--limit", type=int, default=0, help="use only the first N families (debug)")
    g.add_argument("--fraction-missing", default=None)
    g.add_argument("--fm-mode", choices=["off", "e-only", "both"], default="e-only")
    m = ap.add_argument_group("model")
    m.add_argument("--origination", choices=["uniform", "optimize"], default="uniform")
    m.add_argument("--barrier-kind", choices=["meanlog", "simpson", "renyi2"], default="simpson")
    m.add_argument("--barrier-c", type=float, default=0.0, help="O anti-concentration barrier strength (0=off)")
    m.add_argument("--barrier-rho", type=float, default=0.0)
    m.add_argument("--barrier-decay", type=float, default=2.0)
    m.add_argument("--clade-groups", default=None, help="AleRax 'branch wise' dir (DTL_br model)")
    m.add_argument("--clade-groups-from-tree", action="store_true", help="clade-group DTL + O from the tree")
    m.add_argument("--dtl-clades", choices=["all", "paper"], default="all",
                   help="with --clade-groups-from-tree: all named clades, or the paper DTL_br2 whitelist")
    m.add_argument("--prior", choices=["none", "brownian"], default="none")
    m.add_argument("--brownian-sigma", type=float, default=1.0)
    m.add_argument("--brownian-root-sigma", type=float, default=5.0)
    o = ap.add_argument_group("optimization / output")
    o.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "adam", "sgd"])
    o.add_argument("--steps", type=int, default=200)
    o.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    o.add_argument("--family-batch-size", type=int, default=0)
    o.add_argument("--init-rate", type=float, default=0.1)
    o.add_argument("--out", required=True)
    o.add_argument("--per-family-out", default=None, help="also write per-family logL (json)")
    args = ap.parse_args(argv)

    rank, world, local, device = ddp.maybe_init_distributed()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (run on a GPU; use torchrun for multi-GPU).")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    from gpurec.optimization.wave_optimizer import optimize_theta_wave
    from gpurec.core.tree_prior import species_parent_index

    gene_paths = _resolve_genes(args.genes)
    fmt = _infer_format(gene_paths, args.gene_format)
    log(f"[1/4] species={args.species}  genes={len(gene_paths)} ({fmt})")
    species_helpers = _load_species_helpers(args.species)
    S = int(species_helpers["S"]); names = list(species_helpers["names"])
    s2i = species_helpers["species_name_to_index"]
    if fmt == "ale":
        sol = _make_species_of_leaf(args.gene_map, args.species_sep)
        families, fam_names, stats = _load_ale(
            gene_paths if args.limit <= 0 else gene_paths[:args.limit], s2i, sol,
            args.min_species, dtype)
    else:
        families, fam_names, stats = _load_genetrees(args.species, gene_paths, dtype, args.limit)
    log(f"      S={S}  families kept={stats['kept']}  ({stats})")
    if not families:
        raise SystemExit("no usable families.")

    # family-shard for DDP; build the per-rank wave layout
    fam_use = ddp.shard_families(families, rank, world) if world > 1 else families
    if world > 1:
        log(f"[DDP] world={world}; rank {rank} got {len(fam_use)}/{len(families)} families")
    wave_layout, root_clade_ids = _build_wave_layout(fam_use, device, dtype)
    sp_gpu, _anc = _sp_helpers_for_uniform(species_helpers, device, dtype)
    unnorm_row_max = torch.log2(species_helpers["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)
    parent_index = species_parent_index(species_helpers).to(device)

    leaf_E = None
    if args.fm_mode != "off" and args.fraction_missing:
        fm, _n, _ = _parse_fraction_missing(Path(args.fraction_missing), s2i, S)
        leaf_E = _build_leaf_E(species_helpers, fm, S, dtype)[0].to(device=device, dtype=dtype)
    leaf_obs_log = leaf_E if args.fm_mode == "both" else None

    group_index, omega_group_index, ginfo = _build_groups(args, species_helpers, args.species, S, device, dtype)
    barrier_pi = _build_barrier_pi(args, species_helpers, parent_index, S, device, dtype)
    if ginfo:
        log(f"[2/4] clade-grouped: {ginfo.get('dtl_classes','-')} DTL classes, "
            f"{ginfo.get('o_classes','-')} O classes")
    brownian_sigma = float(args.brownian_sigma) if args.prior == "brownian" else None

    theta0 = math.log2(args.init_rate) * torch.ones(S, 3, dtype=dtype, device=device)
    omega_init = torch.zeros(S, dtype=dtype, device=device) if args.origination == "optimize" else None

    log(f"[3/4] optimizing (optimizer={args.optimizer} steps={args.steps} "
        f"origination={args.origination} barrier={args.barrier_kind} c={args.barrier_c}) ...")
    t0 = time.time()
    result = optimize_theta_wave(
        wave_layout=wave_layout, species_helpers=sp_gpu, root_clade_ids=root_clade_ids,
        unnorm_row_max=unnorm_row_max, theta_init=theta0, steps=args.steps,
        optimizer=args.optimizer, specieswise=True, pibar_mode="uniform",
        families=fam_use, family_batch_size=args.family_batch_size, device=device, dtype=dtype,
        leaf_E=leaf_E, leaf_obs_log=leaf_obs_log, verbose=(rank == 0),
        grad_reduction="sum", distributed=(world > 1), ddp_device=device,
        brownian_sigma=brownian_sigma, brownian_root_sigma=args.brownian_root_sigma,
        parent_index=parent_index, origination=args.origination, omega_init=omega_init,
        group_index=group_index, omega_group_index=omega_group_index,
        origination_dirichlet_c=args.barrier_c, origination_vertical_pi=barrier_pi,
        origination_barrier_kind=args.barrier_kind)
    data_logL_ln = (result.get("data_log_likelihood")     # result stores bits (log2)
                    or result.get("log_likelihood") or float("nan")) * _LN2
    log(f"      done in {time.time()-t0:.1f}s  data_logL_ln={data_logL_ln:.1f}")

    if rank == 0:
        rates = result["rates"]; om = result.get("origination")
        payload = {"command": " ".join(sys.argv), "species_tree": args.species, "S": S,
                   "names": names, "n_families": len(families), "gene_format": fmt,
                   "data_log_likelihood_ln": data_logL_ln,
                   "theta_log2": result["theta"].tolist(), "rates": rates.tolist(),
                   "clade_groups": ginfo}
        if om is not None:
            payload["origination"] = {"omega_log2": result["omega"].tolist(),
                                      "origination_prob": om.tolist()}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out if args.out.endswith(".json") else args.out + ".json").write_text(
            json.dumps(payload, indent=2))
        log(f"[4/4] wrote {args.out}")
        if args.per_family_out:
            # per-family logL at the fitted MAP, over ALL families (rank 0 recomputes whole)
            from gpurec.core.extract_parameters import extract_parameters_uniform
            from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood, origination_log_pO
            from gpurec.core.forward import Pi_wave_forward
            theta = result["theta"].to(device=device, dtype=dtype)
            log_pO = origination_log_pO(result["omega"].to(device=device, dtype=dtype)) if om is not None else None
            wl_all, rc_all = _build_wave_layout(families, device, dtype)
            lp = extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)
            E = E_fixed_point(species_helpers=sp_gpu, log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                              transfer_mat=lp[3], max_transfer_mat=lp[4], max_iters=4000, tolerance=1e-11,
                              warm_start_E=None, dtype=dtype, device=device, pibar_mode="uniform",
                              ancestors_T=_anc, leaf_E=leaf_E)
            Pi = Pi_wave_forward(wave_layout=wl_all, species_helpers=sp_gpu, E=E["E"], Ebar=E["E_bar"],
                                 E_s1=E["E_s1"], E_s2=E["E_s2"], log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                                 transfer_mat=lp[3], max_transfer_mat=lp[4], device=device, dtype=dtype,
                                 pibar_mode="uniform", leaf_obs_log=leaf_obs_log)
            nll = compute_log_likelihood(Pi["Pi"], E["E"], rc_all, log_pO=log_pO)
            per = [(-v) * _LN2 for v in nll.detach().cpu().tolist()]
            Path(args.per_family_out).write_text(json.dumps(
                {"names": fam_names, "logL_ln": per, "total": sum(per)}))
            log(f"      wrote per-family logL -> {args.per_family_out}")
    ddp.cleanup()


if __name__ == "__main__":
    main()
