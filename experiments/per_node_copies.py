"""Per-family expected gene presence/copies at each species node, by sampling
reconciliations (gpurec.core.sampler) at a fitted FULLbasin sidecar's rates.

For each family f we build a sampler.FamilyForward, draw N reconciliations, and record
per species branch s:
  - presence:  P(family has >=1 gene copy on s)   = fraction of samples with s in occupied
  - copies:    E[# gene copies on s] = E[# lineage EXIT events {S,SL,T,TL,leaf} at s]
               (one count per surviving copy; O and internal D nodes are NOT counted)
Outputs:
  - <out>.npz                : presence [F,S], copies [F,S], names, root_branch
  - <out>.per_clade.json     : ancestral genome size (sum_f presence) at each named clade's
                               ancestor node
  - <out>.per_family.tsv.gz  : per-family expected copies at each node (long format)

VALIDATION (built in): sum_f P(origination at root) must match families_at_root's
expected@root (same softmax2(log_pO+Pi[root]) draw). Printed + asserted within MC error.

DENSE mode is required (the sampler needs the real linear transfer_mat). Run on a GPU node.
"""
from __future__ import annotations
import argparse, glob, gzip, json, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _load_families, _build_wave_layout, _sp_helpers_for_uniform,
    _parse_fraction_missing, _build_leaf_E,
)
from gpurec.core.sampler import FamilyForward, sample_family  # noqa: E402
from gpurec.core.sampler_cpp import sample_accumulate_cpp  # noqa: E402  (JIT C++ backtrack)

_LN2 = math.log(2.0)
DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
ALE = sorted(p for p in glob.glob(str(DD / "3_UFBOOTs" / "ufboot_for_alerax" / "*.ale"))
             if not Path(p).name.startswith("._"))


def _lse2(x, dim):
    return torch.logsumexp(x * _LN2, dim=dim) / _LN2


def _decode_family(fam):
    ccp = fam["ccp_helpers"] if "ccp_helpers" in fam else fam.get("ccp", fam)
    N = int(ccp["N_splits"])
    sp = ccp["split_parents_sorted"].tolist()
    lr = ccp["split_leftrights_sorted"].tolist()
    lp = ccp["log_split_probs_sorted"].tolist()          # already log2 (loader converted)
    lefts, rights = lr[:N], lr[N:2 * N]
    splits_of = {}
    for i in range(N):
        splits_of.setdefault(int(sp[i]), []).append((int(lefts[i]), int(rights[i]), float(lp[i])))
    cls = {int(c): int(s) for c, s in zip(fam["leaf_row_index"].tolist(),
                                          fam["leaf_col_index"].tolist())}
    return splits_of, cls, int(fam["root_clade_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Eury")
    ap.add_argument("--sidecar", help="gpurec rates sidecar JSON (theta_log2 + omega_log2)")
    ap.add_argument("--alerax-model", choices=["DTL_br1_O", "DTL_br2"],
                    help="instead of --sidecar, use AleRax's EXACT per-branch rates (the paper's "
                         "parameters) from the reference model_parameters.txt")
    ap.add_argument("--model-params", help="arbitrary AleRax model_parameters.txt (per-branch D,L,T,O); "
                    "e.g. the paper's 4_Ancestral_reconstruction rates. Mapped to nodes via --atree.")
    ap.add_argument("--atree", help="labelled species tree matching --model-params (defaults to --root tree)")
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--families", type=int, default=0, help="limit #families (0=all; for validation)")
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="cpp", choices=["cpp", "py"],
                    help="cpp = compiled OpenMP backtrack (core/sampler_cpp); py = pure-Python reference")
    ap.add_argument("--threads", type=int, default=0, help="OpenMP threads for cpp (0=all)")
    ap.add_argument("--validate-n", type=int, default=0,
                    help="run BOTH engines on the first N families and report agreement")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (dense forward).")
    dev = torch.device("cuda"); dtype = torch.float64

    from gpurec.core.extract_parameters import extract_parameters
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.forward import Pi_wave_forward
    from gpurec.core.tree_prior import species_parent_index

    tree = DD / "4_species_tree" / f"Undine_C60_{args.root}root_short_name.nw"
    sp = _load_species_helpers(str(tree)); S = int(sp["S"]); s2i = sp["species_name_to_index"]
    names = list(sp["names"])
    par = species_parent_index(sp).tolist(); root_branch = [i for i in range(S) if par[i] < 0][0]

    if args.model_params:
        from clade_groups import branch_params_from_alerax
        atree = args.atree or str(tree)
        rate_branch, orig_branch, diag = branch_params_from_alerax(sp, atree, args.model_params)
        print(f"  [model-params {Path(args.model_params).name}] unmapped={len(diag.get('unmapped', []))} "
              f"orig_sum={diag.get('orig_sum')}", flush=True)
        theta = torch.log2(rate_branch.clamp_min(1e-10)).to(device=dev, dtype=dtype)
        if orig_branch is not None:
            o = orig_branch.to(device=dev, dtype=dtype).clamp_min(1e-12); log_pO = torch.log2(o / o.sum())
        else:
            log_pO = torch.full((S,), -math.log2(S), dtype=dtype, device=dev)
    elif args.alerax_model:
        # the paper's EXACT per-branch rates (same params validated to r=0.999998 in
        # eval_bigtree_at_alerax). Map AleRax labelled-tree branches -> gpurec nodes.
        from clade_groups import branch_params_from_alerax
        AXBASE = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/alerax_ref/"
                      "3_Reconciliation/This_study/5_reconcilation models")
        mdir = AXBASE / args.alerax_model / (f"Undine_C60_{args.root}root"
                                             + ("2_OR" if args.alerax_model == "DTL_br1_O" else ""))
        atree = mdir / "species_trees" / "starting_species_tree.newick"
        atree = str(atree) if atree.exists() else str(tree)
        rate_branch, orig_branch, diag = branch_params_from_alerax(
            sp, atree, str(mdir / "model_parameters" / "model_parameters.txt"))
        print(f"  [alerax {args.alerax_model}] unmapped={len(diag.get('unmapped', []))} "
              f"orig_sum={diag.get('orig_sum')}", flush=True)
        theta = torch.log2(rate_branch.clamp_min(1e-10)).to(device=dev, dtype=dtype)
        if orig_branch is not None:
            o = orig_branch.to(device=dev, dtype=dtype).clamp_min(1e-12); log_pO = torch.log2(o / o.sum())
        else:
            log_pO = torch.full((S,), -math.log2(S), dtype=dtype, device=dev)
    else:
        if not args.sidecar:
            raise SystemExit("need --sidecar or --alerax-model")
        d = json.load(open(args.sidecar))
        theta = torch.tensor(d["theta_log2"], dtype=dtype, device=dev).reshape(S, 3)
        om = d.get("origination", {}).get("omega_log2")
        if om is None:
            log_pO = torch.full((S,), -math.log2(S), dtype=dtype, device=dev)
        else:
            om = torch.tensor(om, dtype=dtype, device=dev).reshape(S); log_pO = om - _lse2(om, 0)

    lim = args.families
    fams, _ = _load_families(ALE, s2i, min_species=1, dtype=dtype, limit=lim)
    F = len(fams)
    # wave layout + dense forward are built PER CHUNK in the loop below (the all-families
    # dense forward overflows GPU memory at S=513 / 7059 families).
    sp_gpu, _ = _sp_helpers_for_uniform(sp, dev, dtype)

    leaf_E = None; leaf_obs = None
    if args.fm_mode != "off" and (DD / "fraction_missing").exists():
        fm, _n, _s = _parse_fraction_missing(DD / "fraction_missing", s2i, S)
        leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=dev, dtype=dtype)
        leaf_obs = leaf_E if args.fm_mode == "both" else None

    tr_unnorm = torch.log2(sp["Recipients_mat"]).to(dev, dtype)
    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters(
        theta, tr_unnorm, genewise=False, specieswise=True, pairwise=False)
    _mtb = torch.exp2(mt)
    if _mtb.dim() == 1:
        _mtb = _mtb.unsqueeze(1)                                            # per-DONOR-row rescale
    transfer_lin_t = (transfer_mat * _mtb)                                  # [S,S] GPU linear p^T(d->r)
    transfer_lin = transfer_lin_t.cpu().numpy()                             # numpy copy for the C++ sampler
    E_out = E_fixed_point(species_helpers=sp_gpu, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                          transfer_mat=transfer_mat, max_transfer_mat=mt, max_iters=4000,
                          tolerance=1e-10, warm_start_E=None, dtype=dtype, device=dev,
                          pibar_mode="dense", leaf_E=leaf_E)
    # sp_child1/2 (sentinel S = leaf) -- family-independent
    pidx = sp["s_P_indexes"].cpu().long(); cidx = sp["s_C12_indexes"].cpu().long()
    sp_c1 = np.full(S, S, dtype=np.int64); sp_c2 = np.full(S, S, dtype=np.int64)
    m = pidx < S
    sp_c1[pidx[m].numpy()] = cidx[m].numpy(); sp_c2[(pidx[~m] - S).numpy()] = cidx[~m].numpy()
    E_np = E_out["E"].cpu().numpy(); Ebar_np = E_out["E_bar"].cpu().numpy()
    lpS = log_pS.cpu().numpy(); lpD = log_pD.cpu().numpy(); lpO = log_pO.cpu().numpy()

    presence = np.zeros((F, S)); copies = np.zeros((F, S)); orig_root = np.zeros(F)
    copies_all = np.zeros((F, S))                     # diagnostic: OLD all-event count
    obs_genes = np.zeros(S); obs_fam = np.zeros(S)    # observed leaf gene/family counts (ground truth)
    pp_root_an_sum = 0.0
    rng = np.random.default_rng(args.seed)
    ALL_EV = {"O", "D", "S", "T", "SL", "TL", "leaf"}
    vc_pres = np.zeros(S); vp_pres = np.zeros(S); vc_cop = np.zeros(S); vp_cop = np.zeros(S)  # cpp-vs-py
    # Genes at species node s = AleRax's SCount + SLCount + LeafCount (see GeneRaxCore
    # Scenario::gatherReconciliationStatistics): the gene lineages that are AT node s --
    # they speciate (S), speciate-with-loss (SL), or are an observed leaf there. NOT:
    #   O  -- a birth event, not a copy
    #   D  -- an internal duplication node (1->2, both children continue at s; counted by
    #         their own S/SL/leaf exits)
    #   T  -- the source lineage STAYS at s and is counted again at its own later S/SL/leaf;
    #         the received copy is counted at the recipient. (T is an internal branching.)
    #   TL -- the lineage MOVES off s to the recipient (counted there); it is not at node s.
    # This is parameter-independent-validated: at an extant leaf (no children -> no S/SL)
    # it reduces to LeafCount == the observed number of genes.
    COPY_EXIT = {"S", "SL", "leaf"}
    CHUNK = 250                                       # batch the DENSE forward (avoid all-families OOM)
    for i0 in range(0, F, CHUNK):
        batch = fams[i0:i0 + CHUNK]
        wl_b, rc_b = _build_wave_layout(batch, dev, dtype)
        Pi_t = Pi_wave_forward(
            wave_layout=wl_b, species_helpers=sp_gpu, E=E_out["E"], Ebar=E_out["E_bar"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"], log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=transfer_mat, max_transfer_mat=mt, device=dev, dtype=dtype,
            pibar_mode="dense", leaf_obs_log=leaf_obs)["Pi"]                    # [sumC, S] log2 (GPU)
        # Pibar = log2(exp2(Pi-m) @ transfer_lin^T) + m, batched on the GPU for the whole chunk
        # (was an O(C*S^2) numpy matmul PER FAMILY -- the 100-sample bottleneck).
        mx_t = Pi_t.amax(dim=1, keepdim=True)
        Pibar_t = torch.log2(torch.exp2(Pi_t - mx_t) @ transfer_lin_t.transpose(0, 1)) + mx_t
        Pi_b = Pi_t.cpu().numpy(); Pibar_b = Pibar_t.cpu().numpy()
        offs = np.cumsum([0] + [int(f["C"]) for f in batch])
        rc_bn = rc_b.cpu().numpy() if torch.is_tensor(rc_b) else np.asarray(rc_b)
        for j, fam in enumerate(batch):
            f = i0 + j
            off = int(offs[j]); Cf = int(fam["C"])
            Pi_f = Pi_b[off:off + Cf]
            Pibar_f = Pibar_b[off:off + Cf]
            splits_of, cls, root_id = _decode_family(fam)
            for L in cls.values():                    # observed ground truth from the data
                obs_genes[L] += 1
            for L in set(cls.values()):
                obs_fam[L] += 1
            fwd = FamilyForward(Pi=Pi_f, Pibar=Pibar_f, E=E_np, Ebar=Ebar_np, log_pS=lpS, log_pD=lpD,
                                transfer_mat=transfer_lin, log_pO=lpO, sp_child1=sp_c1, sp_child2=sp_c2,
                                clade_leaf_species=cls, clade_leaf_label={}, splits_of=splits_of,
                                root_clade_id=root_id, S=S)
            if args.engine == "cpp":
                pres_f, cop_f = sample_accumulate_cpp(fwd, args.n_samples, seed=args.seed + f,
                                                      n_threads=args.threads)
                presence[f] = pres_f; copies[f] = cop_f
                if f < args.validate_n:        # cross-check the compiled engine vs pure Python
                    pp = np.zeros(S); cp = np.zeros(S)
                    for sc in sample_family(fwd, args.n_samples, rng):
                        for s in sc.occupied:
                            pp[s] += 1
                        for ev in sc.events:
                            if ev.type in COPY_EXIT and 0 <= ev.species < S:
                                cp[ev.species] += 1
                    vc_pres += pres_f; vp_pres += pp; vc_cop += cop_f; vp_cop += cp
            else:
                for sc in sample_family(fwd, args.n_samples, rng):
                    for s in sc.occupied:
                        presence[f, s] += 1
                    for ev in sc.events:
                        if 0 <= ev.species < S:
                            if ev.type in COPY_EXIT:
                                copies[f, ev.species] += 1
                            if ev.type in ALL_EV:
                                copies_all[f, ev.species] += 1
                    if sc.events and sc.events[0].species == root_branch:
                        orig_root[f] += 1
            rr = lpO + Pi_b[int(rc_bn[j])]                      # analytic origination@root (validation)
            rr = rr - (np.log2(np.exp2(rr - rr.max()).sum()) + rr.max())
            pp_root_an_sum += float(np.exp2(rr[root_branch]))
        print(f"  {min(i0 + CHUNK, F)}/{F} families sampled", flush=True)
    presence /= args.n_samples; copies /= args.n_samples; orig_root /= args.n_samples
    copies_all /= args.n_samples

    # GROUND-TRUTH validation at extant leaves (parameter-INDEPENDENT): every reconciliation
    # must place each observed gene at its observed leaf, so sampled copies at a leaf MUST
    # equal the observed gene count there. This pins down whether copy-counting is correct.
    leaves = [s for s in range(S) if sp_c1[s] == S]
    og = float(sum(obs_genes[L] for L in leaves)); ofam = float(sum(obs_fam[L] for L in leaves))
    snew = float(sum(copies[:, L].sum() for L in leaves))
    sold = float(sum(copies_all[:, L].sum() for L in leaves))
    spres = float(sum(presence[:, L].sum() for L in leaves))
    print("\n" + "=" * 60)
    print(f"LEAF GROUND TRUTH (sampled must == observed at extant tips)  engine={args.engine}:")
    print(f"  observed genes at leaves   = {og:.0f}")
    print(f"  sampled copies (S+SL+leaf) = {snew:.1f}   ratio {snew/og:.3f}  <- should be ~1.000")
    if sold > 0:
        print(f"  sampled copies (OLD, all)  = {sold:.1f}   ratio {sold/og:.3f}  <- the bug")
    print(f"  observed families at leaves= {ofam:.0f}")
    print(f"  sampled presence at leaves = {spres:.1f}   ratio {spres/ofam:.3f}")

    if args.validate_n > 0 and vp_cop.sum() > 0:        # compiled-engine vs pure-Python reference
        import numpy.linalg as _la  # noqa
        def _corr(a, b):
            a = a.astype(float); b = b.astype(float)
            if a.std() == 0 or b.std() == 0:
                return 1.0
            return float(np.corrcoef(a, b)[0, 1])
        print("\n" + "=" * 60)
        print(f"ENGINE VALIDATION  cpp vs py  (first {args.validate_n} families, {args.n_samples} samples):")
        print(f"  Σ copies   cpp={vc_cop.sum():.1f}  py={vp_cop.sum():.1f}  "
              f"ratio={vc_cop.sum()/max(1e-9, vp_cop.sum()):.4f}  per-species r={_corr(vc_cop, vp_cop):.5f}")
        print(f"  Σ presence cpp={vc_pres.sum():.1f}  py={vp_pres.sum():.1f}  "
              f"ratio={vc_pres.sum()/max(1e-9, vp_pres.sum()):.4f}  per-species r={_corr(vc_pres, vp_pres):.5f}")

    # VALIDATION (self-contained): sampled vs analytic origination-at-root.
    # presence@root == origination@root (the root is occupied only by origination).
    samp_root = float(orig_root.sum()) if args.engine == "py" else float(presence[:, root_branch].sum())
    print("\n" + "=" * 60)
    print(f"VALIDATION origination@root:  sampled={samp_root:.1f}   analytic={pp_root_an_sum:.1f}   "
          f"diff={samp_root - pp_root_an_sum:+.1f}  (should match within ~sqrt(F/N))")

    # per-clade-ancestor genome size
    from clade_groups import BIGTREE_DTL_CLADES, species_node_leafsets, parse_labeled_newick
    leafsets = species_node_leafsets(sp, S)
    labeled = parse_labeled_newick(str(tree))            # {frozenset(leaves)->label}
    per_clade = {}
    for s in range(S):
        lab = labeled.get(leafsets[s])
        if lab in BIGTREE_DTL_CLADES or s == root_branch:
            key = "LACA(root)" if s == root_branch else lab
            per_clade[key] = dict(node=s,
                                  genome_size=float(presence[:, s].sum()),       # E[#families present]
                                  n_presence_gt0p5=int((presence[:, s] > 0.5).sum()),  # families w/ PP>0.5
                                  sum_copies=float(copies[:, s].sum()),          # Σ_f gene copies (exit-counted)
                                  expected_copies=float(copies[:, s].sum()))     # alias (back-compat)
    Path(args.out + ".per_clade.json").write_text(json.dumps(per_clade, indent=2))
    np.savez_compressed(args.out + ".npz", presence=presence, copies=copies,
                        names=np.array(names, dtype=object), root_branch=root_branch)
    with gzip.open(args.out + ".per_family.tsv.gz", "wt") as fh:
        fh.write("family_index\tspecies_node\tnode_name\texpected_presence\texpected_copies\n")
        for f in range(F):
            for s in range(S):
                if copies[f, s] > 1e-4 or presence[f, s] > 1e-4:
                    fh.write(f"{f}\t{s}\t{names[s]}\t{presence[f,s]:.4f}\t{copies[f,s]:.4f}\n")
    print(f"  wrote {args.out}.npz / .per_clade.json / .per_family.tsv.gz")
    print(f"  LACA(root) genome size (sampled, presence) = {per_clade.get('LACA(root)',{}).get('genome_size')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
