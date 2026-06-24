"""Fit the RECIPIENT ('transfer-to') transfer model on Williams via the legacy
dense forward + plain autograd (no engine/Triton/backward-VJP changes needed).

Transfer-to is realized by setting the recipient-logit matrix to a recipient-only
broadcast: transfer_mat_unnormalized[d,r] = omega_r at valid (non-ancestor) r, -inf
else. extract_parameters + the dense legacy forward then compute the w-weighted
recipient sum, and autograd through the unrolled fixed points gives grad w.r.t.
(theta, omega). This is the functional (if O(C*S^2), subset-scale) transfer-to fit;
the efficient wave/Triton path + implicit backward VJP is the separate optimization.

Modes:
  donor       : transfer_mat_unnormalized = log2(Recipients_mat) (current model), fit theta.
  transfer-to : transfer_mat_unnormalized = omega-broadcast, fit theta + omega (recipient).
                --transfer pure  -> global donor T rate (theta_T tied);  both -> per-branch T (a_d).

Reports converged data logL. Run on an A100 (autograd-unroll memory); use --families to subset.
"""
from __future__ import annotations
import argparse, glob, math, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _load_families, _build_leaf_E, _parse_fraction_missing,
)

_LN2 = math.log(2.0)
NEG = float("-inf")
DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")


def _dense_species_helpers(sp, device, dtype):
    return {
        "S": int(sp["S"]),
        "names": sp.get("names"),
        "s_P_indexes": sp["s_P_indexes"].to(device),
        "s_C12_indexes": sp["s_C12_indexes"].to(device),
        "Recipients_mat": sp["Recipients_mat"].to(dtype=dtype, device=device),
    }


def fit_root(root, data_dir, device, dtype, mode, transfer, n_fam, steps, lr, fm_mode, seed,
             tree=None, ale_dir=None, fm_path=None, warm_sidecar=None, fp_iters=400, out_sidecar=None):
    import json
    from gpurec.core.batching import collate_gene_families
    from gpurec.core.extract_parameters import extract_parameters
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.legacy import Pi_fixed_point

    tree_path = tree or str(data_dir / "rooted_phylogeny" / root)
    sp = _load_species_helpers(tree_path)
    S = int(sp["S"]); sp_name_to_idx = sp["species_name_to_index"]
    sh = _dense_species_helpers(sp, device, dtype)

    ale_glob = str(Path(ale_dir) / "*.ale") if ale_dir else str(data_dir / "ccps" / "*.ale")
    ale_paths = sorted(p for p in glob.glob(ale_glob) if not Path(p).name.startswith("._"))
    fams, _ = _load_families(ale_paths, sp_name_to_idx, min_species=1, dtype=dtype, limit=n_fam)
    items = [{"ccp": f["ccp_helpers"], "leaf_row_index": f["leaf_row_index"],
              "leaf_col_index": f["leaf_col_index"], "root_clade_id": int(f["root_clade_id"])}
             for f in fams]
    batched = collate_gene_families(items, dtype=dtype, device=device)
    ch = batched["ccp"]; li = batched["leaf_row_index"]; lc = batched["leaf_col_index"]
    root_clade_ids = batched["root_clade_ids"]
    F = len(fams)

    # fraction-missing leaf boundary (E-only, AleRax-faithful)
    leaf_E = leaf_mask = None
    if fm_mode != "off":
        fmp = Path(fm_path) if fm_path else (data_dir / "fraction_missing")
        if fmp.exists():
            fm, _n, _ = _parse_fraction_missing(fmp, sp_name_to_idx, S)
            leaf_E, leaf_mask = _build_leaf_E(sp, fm, S, dtype)
            leaf_E = leaf_E.to(device=device, dtype=dtype); leaf_mask = leaf_mask.to(device)

    recip_log = torch.log2(sh["Recipients_mat"].clamp_min(0))   # -inf where invalid (0)
    valid = torch.isfinite(recip_log)                            # [S,S] valid recipient mask

    g = torch.Generator(device="cpu").manual_seed(seed)
    warm = None
    if warm_sidecar:
        wd = json.load(open(warm_sidecar)); warm = torch.tensor(wd["theta_log2"], dtype=dtype).reshape(S, 3)
        print(f"  warm-started theta from {Path(warm_sidecar).name}", flush=True)
    if warm is not None:
        theta_DL = warm[:, :2].clone().to(device=device, dtype=dtype).requires_grad_(True)
    else:
        theta_DL = (0.1 * torch.randn(S, 2, generator=g)).to(device=device, dtype=dtype).requires_grad_(True)
    params = [theta_DL]
    if transfer == "both":
        if warm is not None:
            log_T = warm[:, 2:3].clone().to(device=device, dtype=dtype).requires_grad_(True)
        else:
            log_T = (0.1 * torch.randn(S, 1, generator=g)).to(device=device, dtype=dtype).requires_grad_(True)
        params.append(log_T)
    else:  # pure: single global donor transfer logit
        log_T = torch.zeros(1, device=device, dtype=dtype).requires_grad_(True)
        params.append(log_T)
    omega = torch.zeros(S, device=device, dtype=dtype)
    if mode == "transfer-to":
        omega = omega.clone().requires_grad_(True); params.append(omega)

    opt = torch.optim.Adam(params, lr=lr)

    def forward_nll():
        theta = torch.cat([theta_DL, log_T.expand(S, 1) if log_T.numel() == 1 else log_T], dim=1)  # [S,3]
        if mode == "donor":
            tmu = recip_log
        else:
            tmu = torch.where(valid, omega.unsqueeze(0).expand(S, S),
                              torch.full((S, S), NEG, device=device, dtype=dtype))
        pS, pD, pL, tf, mt = extract_parameters(theta, tmu, genewise=False, specieswise=True, pairwise=False)
        mv = mt.squeeze(-1) if mt.ndim == 2 else mt
        Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                           transfer_mat=tf, max_transfer_mat=mv, max_iters=fp_iters, tolerance=1e-8,
                           warm_start_E=None, dtype=dtype, device=device, pibar_mode="dense",
                           leaf_E=leaf_E)
        Po = Pi_fixed_point(ccp_helpers=ch, species_helpers=sh, leaf_row_index=li, leaf_col_index=lc,
                            E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
                            log_pS=pS, log_pD=pD, log_pL=pL, transfer_mat_T=tf.T.contiguous(),
                            max_transfer_mat=mv, max_iters=fp_iters, tolerance=1e-8, warm_start_Pi=None,
                            device=device, dtype=dtype,
                            leaf_obs_log=(leaf_E if fm_mode == "both" else None),
                            leaf_species_mask=(leaf_mask if fm_mode == "both" else None))
        nll = compute_log_likelihood(Po["Pi"], Eo["E"], root_clade_ids, log_pO=None)  # uniform O
        return nll.sum()

    last = None
    for it in range(steps):
        opt.zero_grad(); loss = forward_nll(); loss.backward(); opt.step()
        last = float(loss.item())
        if it % 10 == 0 or it == steps - 1:
            print(f"  [{root} {mode}/{transfer}] step {it:3d} data_nll(bits)={last:.1f} "
                  f"logL_ln={-last*_LN2:.1f}", flush=True)
    # fitted rates -> sidecar (theta_log2 [S,3] D,L,T + omega recipient weights)
    from gpurec.core.transfer_to import normalize_omega
    th_T = (log_T.expand(S, 1) if log_T.numel() == 1 else log_T).detach()
    theta_fit = torch.cat([theta_DL.detach(), th_T], dim=1)
    r_lin = torch.exp2(theta_fit)
    om = (normalize_omega(omega).detach() if mode == "transfer-to"
          else torch.zeros(S, device=device, dtype=dtype))
    if S == 513:                              # big-tree Eury: DPANN ancestor = node 510
        print(f"  [fit] DPANN(510) D,L,T = {r_lin[510,0]:.4f},{r_lin[510,1]:.4f},{r_lin[510,2]:.4f}  "
              f"(FULLbasin had T=2.90; paper T=0.71)", flush=True)
    if out_sidecar:
        json.dump(dict(theta_log2=theta_fit.cpu().tolist(), recipient_omega_log2=om.cpu().tolist(),
                       mode=mode, transfer=transfer, root=root, S=S, logL_ln=-last * _LN2),
                  open(out_sidecar, "w"))
        print(f"  wrote {out_sidecar}", flush=True)
    return dict(F=F, logL_ln=-last * _LN2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tree", help="explicit species tree (big tree); overrides --data-dir/rooted_phylogeny")
    ap.add_argument("--ale-dir", help="explicit dir of .ale files (big tree); overrides --data-dir/ccps")
    ap.add_argument("--fm-path", help="explicit fraction_missing file")
    ap.add_argument("--warm-sidecar", help="init theta (D,L,T) from a FULLbasin rates sidecar JSON")
    ap.add_argument("--fp-iters", type=int, default=400, help="fixed-point iters (lower = less autograd memory)")
    ap.add_argument("--out-sidecar", help="write fitted theta+omega to this JSON")
    ap.add_argument("--mode", default="transfer-to", choices=["donor", "transfer-to"])
    ap.add_argument("--transfer", default="pure", choices=["pure", "both"])
    ap.add_argument("--families", type=int, default=500)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    r = fit_root(args.root, Path(args.data_dir), device, dtype, args.mode, args.transfer,
                 args.families, args.steps, args.lr, args.fm_mode, args.seed,
                 tree=args.tree, ale_dir=args.ale_dir, fm_path=args.fm_path,
                 warm_sidecar=args.warm_sidecar, fp_iters=args.fp_iters, out_sidecar=args.out_sidecar)
    print(f"[result] root={args.root} mode={args.mode} transfer={args.transfer} "
          f"F={r['F']} logL_ln={r['logL_ln']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
