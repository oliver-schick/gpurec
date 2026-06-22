"""LACA (root) genome size + gene-content consistency: the set of gene families
originating at the root under (1) gpurec's FULL free per-branch DTL+O fit vs
(2) gpurec @ AleRax's EXACT DTL_br1_O per-branch rates (reproduces AleRax,
per-fam logL r=0.999998). Both use the SAME family list (index-aligned), so the
root sets join by index.

Per family f and rate set: pp_root[f] = softmax2(log_pO + Pi_f[root_clade,:])[root_branch]
  = posterior prob the family originates at the chosen root branch.
LACA genome size  : expected = sum_f pp_root[f] ; thresholded = #{f: pp_root>=0.5}.
Gene content      : root set R = {f : pp_root[f] >= 0.5}. Jaccard(R_full, R_alerax).

NB: AleRax saved only per_fam_likelihoods (no reconciliation/root content), so the
"AleRax" side here is gpurec evaluated at AleRax's exact rates -- a faithful proxy.

Run on a GPU node. Forward is chunked (CHUNK=500); E computed once per rate set.
"""
from __future__ import annotations
import argparse, glob, json, math, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _build_wave_layout, _sp_helpers_for_uniform,
    _parse_fraction_missing, _build_leaf_E,
)
from eval_at_alerax_rates import _load_families_named  # noqa: E402
from clade_groups import branch_params_from_alerax     # noqa: E402
from gpurec.core.tree_prior import species_parent_index  # noqa: E402

_LN2 = math.log(2.0)
DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/alerax_ref/"
          "3_Reconciliation/This_study")
BASE = DD / "5_reconcilation models"
ALE = sorted(p for p in glob.glob(str(DD / "3_UFBOOTs" / "ufboot_for_alerax" / "*.ale"))
             if not Path(p).name.startswith("._"))
CHUNK = 500


def _lse2(x, dim):
    return torch.logsumexp(x * _LN2, dim=dim) / _LN2


def _pp_root(theta, log_pO, root_branch, sp_gpu, anc, urm, fams, rc_all,
             leaf_E, leaf_obs_log, device, dtype):
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.forward import Pi_wave_forward
    lp = extract_parameters_uniform(theta, urm, specieswise=True)
    E = E_fixed_point(species_helpers=sp_gpu, log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                      transfer_mat=lp[3], max_transfer_mat=lp[4], max_iters=4000,
                      tolerance=1e-10, warm_start_E=None, dtype=dtype, device=device,
                      pibar_mode="uniform", ancestors_T=anc, leaf_E=leaf_E)
    pp = []
    for i in range(0, len(fams), CHUNK):
        chunk = fams[i:i + CHUNK]
        wl, rc = _build_wave_layout(chunk, device, dtype)
        Pi = Pi_wave_forward(wave_layout=wl, species_helpers=sp_gpu, E=E["E"], Ebar=E["E_bar"],
                             E_s1=E["E_s1"], E_s2=E["E_s2"], log_pS=lp[0], log_pD=lp[1],
                             log_pL=lp[2], transfer_mat=lp[3], max_transfer_mat=lp[4],
                             device=device, dtype=dtype, pibar_mode="uniform",
                             leaf_obs_log=leaf_obs_log)
        root_probs = Pi["Pi"][rc]                                   # [Fc, S]
        logp = log_pO.unsqueeze(0) + root_probs
        log_post = logp - _lse2(logp, dim=1).unsqueeze(1)
        pp.extend(torch.exp2(log_post[:, root_branch]).detach().cpu().tolist())
    return pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Eury")
    ap.add_argument("--sidecar", required=True, help="gpurec FULL free per-branch sidecar (.rates.txt.json)")
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda"); dtype = torch.float64
    root = args.root

    tree_path = DD / "4_species_tree" / f"Undine_C60_{root}root_short_name.nw"
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"]); s2i = sp["species_name_to_index"]
    par = species_parent_index(sp).tolist()
    root_branch = [i for i in range(S) if par[i] < 0][0]

    fams, names = _load_families_named(ALE, s2i, min_species=1, dtype=dtype)
    sp_gpu, anc = _sp_helpers_for_uniform(sp, device, dtype)
    urm = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)
    rc_all = None

    leaf_E = None
    if args.fm_mode != "off":
        fmp = DD / "fraction_missing"
        if fmp.exists():
            fm, _n, _s = _parse_fraction_missing(fmp, s2i, S)
            leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
    leaf_obs_log = leaf_E if args.fm_mode == "both" else None

    # (1) gpurec FULL free per-branch sidecar
    d = json.load(open(args.sidecar))
    theta_full = torch.tensor(d["theta_log2"], dtype=dtype, device=device).reshape(S, 3)
    om = d.get("origination", {}).get("omega_log2")
    if om is None:
        log_pO_full = torch.full((S,), -math.log2(S), dtype=dtype, device=device)
    else:
        om = torch.tensor(om, dtype=dtype, device=device).reshape(S)
        log_pO_full = om - _lse2(om, dim=0)
    print(f"[1/2] gpurec FULL free per-branch @ {root} ...", flush=True)
    pp_full = _pp_root(theta_full, log_pO_full, root_branch, sp_gpu, anc, urm, fams,
                       rc_all, leaf_E, leaf_obs_log, device, dtype)

    # (2) gpurec @ AleRax DTL_br1_O exact rates
    mdir = BASE / "DTL_br1_O" / f"Undine_C60_{root}root2_OR"
    mp = mdir / "model_parameters" / "model_parameters.txt"
    atree = mdir / "species_trees" / "starting_species_tree.newick"
    atree = str(atree) if atree.exists() else str(tree_path)
    rate_branch, orig_branch, diag = branch_params_from_alerax(sp, atree, str(mp))
    theta_ax = torch.log2(rate_branch.clamp_min(1e-10)).to(device=device, dtype=dtype)
    o = orig_branch.to(device=device, dtype=dtype).clamp_min(1e-12)
    log_pO_ax = torch.log2(o / o.sum())
    print(f"[2/2] gpurec @ AleRax DTL_br1_O rates @ {root} ...", flush=True)
    pp_ax = _pp_root(theta_ax, log_pO_ax, root_branch, sp_gpu, anc, urm, fams,
                     rc_all, leaf_E, leaf_obs_log, device, dtype)

    thr = args.thr
    R_full = {i for i, p in enumerate(pp_full) if p >= thr}
    R_ax = {i for i, p in enumerate(pp_ax) if p >= thr}
    inter = R_full & R_ax; union = R_full | R_ax
    jac = len(inter) / len(union) if union else float("nan")
    exp_full = sum(pp_full); exp_ax = sum(pp_ax)
    # pearson of per-family pp
    n = len(pp_full); mf = exp_full / n; ma = exp_ax / n
    cov = sum((pp_full[i] - mf) * (pp_ax[i] - ma) for i in range(n))
    vf = sum((pp_full[i] - mf) ** 2 for i in range(n)); va = sum((pp_ax[i] - ma) ** 2 for i in range(n))
    pear = cov / math.sqrt(vf * va) if vf > 0 and va > 0 else float("nan")

    print("\n" + "=" * 70)
    print(f"LACA (root={root}) genome size + content consistency  [F={n} families]")
    print(f"  gpurec FULL free per-branch : expected@root={exp_full:8.1f}  content(PP>={thr})={len(R_full)}")
    print(f"  gpurec @ AleRax DTL_br1_O    : expected@root={exp_ax:8.1f}  content(PP>={thr})={len(R_ax)}")
    print(f"  --- gene content overlap ---")
    print(f"  |R_full|={len(R_full)}  |R_alerax|={len(R_ax)}  inter={len(inter)}  union={len(union)}")
    print(f"  JACCARD(R_full, R_alerax) = {jac:.3f}")
    print(f"  per-family PP(origin@root) Pearson r = {pear:.4f}")
    if args.out:
        out = dict(root=root, F=n, expected_full=exp_full, expected_alerax=exp_ax,
                   content_full=len(R_full), content_alerax=len(R_ax),
                   inter=len(inter), union=len(union), jaccard=jac, pearson_pp=pear,
                   root_families_full=[names[i] for i in sorted(R_full)],
                   root_families_alerax=[names[i] for i in sorted(R_ax)])
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
