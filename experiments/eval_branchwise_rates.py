"""Diagnostic: evaluate gpurec's likelihood at AleRax's EXACT branch-wise (D,L,T)
per-branch rates (uniform origination, e-only fm), for ALL candidate roots, and
rank the roots. This disentangles OPTIMIZER vs MODEL:

  - If gpurec at AleRax's branch-wise rates ranks Eury #1 (matching AleRax's
    branch-wise per_fam ranking up to the constant CCP offset), then gpurec's
    likelihood engine + the (D,L,T) heterogeneity ARE sufficient, and gpurec's
    clade-grouped OPTIMIZER simply failed to reach AleRax's optimum.
  - If gpurec at AleRax's exact (D,L,T) rates still ranks Cluster2 #1, then the
    (D,L,T) rates are NOT sufficient under uniform origination -> AleRax's
    per-clade ORIGINATION (DTLO for DPANN/Eury/TackA) is the missing ingredient.

Per-branch rates come from AleRax's branch-wise model_parameters via the same
leaf-set clade mapping used for the grouped model (clade_groups). Forward only.

Run on an A100. Loops all 10 roots in one process.
"""
from __future__ import annotations
import argparse
import glob
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _load_families, _build_wave_layout,
    _sp_helpers_for_uniform, _build_leaf_E, _parse_fraction_missing,
)
from clade_groups import branch_params_from_alerax  # noqa: E402

_LN2 = math.log(2.0)
DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")
ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury",
         "HaloThermoplas", "Kor", "TAC", "TackA"]


def eval_root(root, data_dir, device, dtype, min_species, fm_mode, o_mode, root_frac):
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.forward import Pi_wave_forward

    tree_path = data_dir / "rooted_phylogeny" / root
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"])
    sp_name_to_idx = sp["species_name_to_index"]

    bw = data_dir / "reconciliation models" / "branch wise" / root
    cg_tree = bw / "species_trees" / "starting_species_tree.newick"
    cg_mp = bw / "model_parameters" / "model_parameters.txt"
    rate_branch, orig_branch, diag = branch_params_from_alerax(
        sp, str(cg_tree), str(cg_mp))
    if diag["unmapped"]:
        raise SystemExit(f"{root}: unmapped nodes {diag['unmapped'][:3]}")
    theta = torch.log2(rate_branch.clamp_min(1e-10)).to(device=device, dtype=dtype)  # [S,3]
    # ORIGINATION p^O_e distribution -> log_pO = log2(p^O).
    #   'uniform' : log_pO=None (p^O=1/S)
    #   'alerax'  : AleRax's fitted per-branch O (the structured DTLO)
    #   'root'    : VERTICAL-EVOLUTION prior -- root_frac on the SINGLE root node
    #               (parent<0), (1-root_frac) spread uniformly over all branches.
    log_pO = None
    if o_mode == "alerax":
        if orig_branch is None:
            raise SystemExit(f"{root}: model_parameters has no O column")
        o = orig_branch.to(device=device, dtype=dtype).clamp_min(1e-12)
        o = o / o.sum()
        log_pO = torch.log2(o)                            # [S]
    elif o_mode == "root":
        from gpurec.core.tree_prior import species_parent_index
        par = species_parent_index(sp).tolist()
        root_nodes = [i for i in range(S) if par[i] < 0]
        if len(root_nodes) != 1:
            raise SystemExit(f"{root}: expected exactly 1 root branch, got {root_nodes}")
        rn = root_nodes[0]
        o = torch.full((S,), (1.0 - root_frac) / S, dtype=dtype, device=device)
        o[rn] += root_frac                                # spike on the one root branch
        o = o / o.sum()
        log_pO = torch.log2(o.clamp_min(1e-300))

    ale_paths = sorted(p for p in glob.glob(str(data_dir / "ccps" / "*.ale"))
                       if not Path(p).name.startswith("._"))
    fams, _ = _load_families(ale_paths, sp_name_to_idx,
                             min_species=min_species, dtype=dtype, limit=0)
    F = len(fams)
    wave_layout, root_clade_ids = _build_wave_layout(fams, device, dtype)
    sp_gpu, ancestors_T = _sp_helpers_for_uniform(sp, device, dtype)
    unnorm_row_max = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(
        device=device, dtype=dtype)

    leaf_E = None
    if fm_mode != "off":
        fm_path = data_dir / "fraction_missing"
        if fm_path.exists():
            fm, _n, _ = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
            leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
    leaf_obs_log = leaf_E if fm_mode == "both" else None

    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
        theta, unnorm_row_max, specieswise=True)
    E_out = E_fixed_point(
        species_helpers=sp_gpu, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt, max_iters=4000,
        tolerance=1e-10, warm_start_E=None, dtype=dtype, device=device,
        pibar_mode="uniform", ancestors_T=ancestors_T, leaf_E=leaf_E)
    Pi_out = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sp_gpu, E=E_out["E"],
        Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, transfer_mat=transfer_mat,
        max_transfer_mat=mt, device=device, dtype=dtype, pibar_mode="uniform",
        leaf_obs_log=leaf_obs_log)
    nll_log2 = compute_log_likelihood(Pi_out["Pi"], E_out["E"], root_clade_ids, log_pO=log_pO)
    total_logL_ln = float((-nll_log2).sum().item()) * _LN2
    n_groups = len({tuple(r.tolist()) for r in rate_branch})
    return total_logL_ln, F, n_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--min-species", type=int, default=1)
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--roots", default=None, help="comma-sep subset (default all 10)")
    ap.add_argument("--origination", default="uniform",
                    choices=["uniform", "alerax", "root"],
                    help="p^O: uniform 1/S | AleRax's fitted DTLO | 'root' "
                         "(VERTICAL-EVOLUTION prior: --root-frac on the single root "
                         "branch, rest uniform).")
    ap.add_argument("--root-frac", type=float, default=1.0,
                    help="for --origination root: fraction of origination mass on "
                         "the root branch (1.0=pure root; e.g. 0.5=root+uniform).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (A100).")
    device = torch.device("cuda")
    dtype = torch.float64
    data_dir = Path(args.data_dir)
    roots = args.roots.split(",") if args.roots else ROOTS

    results = {}
    for r in roots:
        ll, F, ng = eval_root(r, data_dir, device, dtype, args.min_species,
                              args.fm_mode, args.origination, args.root_frac)
        results[r] = (ll, F, ng)
        print(f"[done] {r:16s} logL_ln={ll:14.1f}  F={F}  groups={ng}", flush=True)

    print("\n" + "=" * 70)
    o_desc = {"uniform": "UNIFORM O", "alerax": "AleRax O (DTLO)",
              "root": f"ROOT-origination frac={args.root_frac}"}[args.origination]
    print("gpurec @ AleRax branch-wise (D,L,T) rates, %s, fm=%s" % (o_desc, args.fm_mode))
    print("ranking by total data logL (ln), best first:")
    best = max(results.values())[0]
    for r in sorted(results, key=lambda x: -results[x][0]):
        ll, F, ng = results[r]
        star = " <BEST" if ll == best else ""
        print(f"  {r:16s} {ll:14.1f}  d={ll-best:9.1f}  (F={F}, G={ng}){star}")
    if args.out:
        import json
        Path(args.out).write_text(json.dumps(
            {r: {"logL_ln": v[0], "F": v[1], "groups": v[2]} for r, v in results.items()},
            indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
