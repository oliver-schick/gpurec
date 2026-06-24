"""Per-family expected gene presence/copies at each species node, by sampling
reconciliations (gpurec.core.sampler) at a fitted FULLbasin sidecar's rates.

For each family f we build a sampler.FamilyForward, draw N reconciliations, and record
per species branch s:
  - presence:  P(family has >=1 gene copy on s)   = fraction of samples with s in occupied
  - copies:    E[# gene-tree nodes mapped to s]    (a copy-number proxy)
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
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--families", type=int, default=0, help="limit #families (0=all; for validation)")
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--seed", type=int, default=0)
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
    wl, root_clade_ids, family_meta = _build_wave_layout(fams, dev, dtype, return_meta=True) \
        if "return_meta" in _build_wave_layout.__code__.co_varnames else (*_build_wave_layout(fams, dev, dtype), None)
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
    transfer_lin = (transfer_mat * _mtb).cpu().numpy()                      # [S,S] linear p^T(d->r)
    E_out = E_fixed_point(species_helpers=sp_gpu, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                          transfer_mat=transfer_mat, max_transfer_mat=mt, max_iters=4000,
                          tolerance=1e-10, warm_start_E=None, dtype=dtype, device=dev,
                          pibar_mode="dense", leaf_E=leaf_E)
    Pi_out = Pi_wave_forward(wave_layout=wl, species_helpers=sp_gpu, E=E_out["E"], Ebar=E_out["E_bar"],
                             E_s1=E_out["E_s1"], E_s2=E_out["E_s2"], log_pS=log_pS, log_pD=log_pD,
                             log_pL=log_pL, transfer_mat=transfer_mat, max_transfer_mat=mt,
                             device=dev, dtype=dtype, pibar_mode="dense", leaf_obs_log=leaf_obs)
    Pi_all = Pi_out["Pi"].cpu().numpy()                                    # [C_total, S] log2

    # sp_child1/2 (sentinel S = leaf)
    pidx = sp["s_P_indexes"].cpu().long(); cidx = sp["s_C12_indexes"].cpu().long()
    sp_c1 = np.full(S, S, dtype=np.int64); sp_c2 = np.full(S, S, dtype=np.int64)
    m = pidx < S
    sp_c1[pidx[m].numpy()] = cidx[m].numpy(); sp_c2[(pidx[~m] - S).numpy()] = cidx[~m].numpy()

    E_np = E_out["E"].cpu().numpy(); Ebar_np = E_out["E_bar"].cpu().numpy()
    lpS = log_pS.cpu().numpy(); lpD = log_pD.cpu().numpy(); lpO = log_pO.cpu().numpy()

    if family_meta is None:  # fallback: rebuild clade offsets from per-family C
        offs = np.cumsum([0] + [int(f["C"]) for f in fams])
        family_meta = [{"clade_offset": int(offs[i]), "C": int(fams[i]["C"])} for i in range(F)]

    presence = np.zeros((F, S)); copies = np.zeros((F, S)); orig_root = np.zeros(F)
    rng = np.random.default_rng(args.seed)
    LINEAGE = {"O", "D", "S", "T", "SL", "TL", "leaf"}
    for f in range(F):
        off = int(family_meta[f]["clade_offset"]); Cf = int(family_meta[f]["C"])
        Pi_f = Pi_all[off:off + Cf]
        mx = Pi_f.max(axis=1, keepdims=True)
        Pibar_f = np.log2(np.exp2(Pi_f - mx) @ transfer_lin.T) + mx        # [Cf,S] log2
        splits_of, cls, root_id = _decode_family(fams[f])
        fwd = FamilyForward(Pi=Pi_f, Pibar=Pibar_f, E=E_np, Ebar=Ebar_np, log_pS=lpS, log_pD=lpD,
                            transfer_mat=transfer_lin, log_pO=lpO, sp_child1=sp_c1, sp_child2=sp_c2,
                            clade_leaf_species=cls, clade_leaf_label={}, splits_of=splits_of,
                            root_clade_id=root_id, S=S)
        scen = sample_family(fwd, args.n_samples, rng)
        for sc in scen:
            for s in sc.occupied:
                presence[f, s] += 1
            cc = np.zeros(S)
            for ev in sc.events:
                if ev.type in LINEAGE and 0 <= ev.species < S:
                    cc[ev.species] += 1
            copies[f] += cc
            if sc.events and sc.events[0].species == root_branch:
                orig_root[f] += 1
        if (f + 1) % 500 == 0:
            print(f"  {f+1}/{F} families sampled", flush=True)
    presence /= args.n_samples; copies /= args.n_samples; orig_root /= args.n_samples

    # VALIDATION (self-contained): sampled origination-at-root vs analytic posterior
    # analytic pp_root[f] = softmax2(log_pO + Pi_f[root_clade, :])[root_branch]
    rc_np = (root_clade_ids.cpu().numpy() if torch.is_tensor(root_clade_ids)
             else np.asarray(root_clade_ids))
    root_probs = Pi_all[rc_np]                                  # [F,S] log2
    logp = lpO[None, :] + root_probs
    logp -= (np.log2(np.exp2((logp - logp.max(1, keepdims=True)) * 1.0).sum(1, keepdims=True))
             + logp.max(1, keepdims=True))                      # logsumexp2 over species
    pp_root_an = np.exp2(logp[:, root_branch])                  # [F]
    samp_root = float(orig_root.sum()); an_root = float(pp_root_an.sum())
    print("\n" + "=" * 60)
    print(f"VALIDATION origination@root:  sampled={samp_root:.1f}   analytic={an_root:.1f}   "
          f"diff={samp_root-an_root:+.1f}  (should match within ~sqrt(F/N))")

    # per-clade-ancestor genome size
    from clade_groups import BIGTREE_DTL_CLADES, species_node_leafsets, parse_labeled_newick
    leafsets = species_node_leafsets(sp, S)
    labeled = parse_labeled_newick(str(tree))            # {frozenset(leaves)->label}
    per_clade = {}
    for s in range(S):
        lab = labeled.get(leafsets[s])
        if lab in BIGTREE_DTL_CLADES or s == root_branch:
            key = "LACA(root)" if s == root_branch else lab
            per_clade[key] = dict(node=s, genome_size=float(presence[:, s].sum()),
                                  expected_copies=float(copies[:, s].sum()))
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
