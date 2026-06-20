"""Sample reconciliations (origination step) and count families present at the
root for each candidate rooting, using a fitted rates sidecar.

Presence at the root node == origination at the root branch (in the undated
origination model nothing exists above the root). So per family we sample the
origination species e0 ~ softmax2(log_pO + Pi[root_clade, :]) -- the root of each
sampled reconciliation -- and count e0 == root_branch. Reports, per root:
  - expected #families at root  = sum_f PP_f(root)          (analytic, exact)
  - 1k-sampled #families at root = sum_f (sampled fraction)  (Monte Carlo)
  - thresholded gene content     = #{f : PP_f(root) >= 0.5}

Run on a GPU node (uses the wave forward). Loops the given roots.
"""
from __future__ import annotations
import argparse, glob, json, math, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _load_families, _build_wave_layout,
    _sp_helpers_for_uniform, _build_leaf_E, _parse_fraction_missing,
)

_LN2 = math.log(2.0)
DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")
ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury",
         "HaloThermoplas", "Kor", "TAC", "TackA"]


def _logsumexp2(x, dim):
    return torch.logsumexp(x * _LN2, dim=dim) / _LN2


def families_at_root(root, sidecar, data_dir, device, dtype, n_samples, seed, fm_mode):
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.forward import Pi_wave_forward
    from gpurec.core.tree_prior import species_parent_index

    sp = _load_species_helpers(str(data_dir / "rooted_phylogeny" / root))
    S = int(sp["S"])
    sp_name_to_idx = sp["species_name_to_index"]
    par = species_parent_index(sp).tolist()
    root_nodes = [i for i in range(S) if par[i] < 0]
    assert len(root_nodes) == 1, root_nodes
    root_branch = root_nodes[0]

    d = json.load(open(sidecar))
    theta = torch.tensor(d["theta_log2"], dtype=dtype, device=device).reshape(S, 3)
    omega = d.get("origination", {}).get("omega_log2")
    if omega is None:
        log_pO = torch.full((S,), -math.log2(S), dtype=dtype, device=device)
    else:
        om = torch.tensor(omega, dtype=dtype, device=device).reshape(S)
        log_pO = om - _logsumexp2(om, dim=0)

    ale_paths = sorted(p for p in glob.glob(str(data_dir / "ccps" / "*.ale"))
                       if not Path(p).name.startswith("._"))
    fams, _ = _load_families(ale_paths, sp_name_to_idx, min_species=1, dtype=dtype, limit=0)
    F = len(fams)
    wave_layout, root_clade_ids = _build_wave_layout(fams, device, dtype)
    sp_gpu, ancestors_T = _sp_helpers_for_uniform(sp, device, dtype)
    unnorm_row_max = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)

    leaf_E = None
    if fm_mode != "off":
        fm_path = data_dir / "fraction_missing"
        if fm_path.exists():
            fm, _n, _ = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
            leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
    leaf_obs_log = leaf_E if fm_mode == "both" else None

    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
        theta, unnorm_row_max, specieswise=True)
    E_out = E_fixed_point(species_helpers=sp_gpu, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                          transfer_mat=transfer_mat, max_transfer_mat=mt, max_iters=4000,
                          tolerance=1e-10, warm_start_E=None, dtype=dtype, device=device,
                          pibar_mode="uniform", ancestors_T=ancestors_T, leaf_E=leaf_E)
    Pi_out = Pi_wave_forward(wave_layout=wave_layout, species_helpers=sp_gpu, E=E_out["E"],
                             Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
                             log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, transfer_mat=transfer_mat,
                             max_transfer_mat=mt, device=device, dtype=dtype, pibar_mode="uniform",
                             leaf_obs_log=leaf_obs_log)
    root_probs = Pi_out["Pi"][root_clade_ids]                      # [F, S] = Pi_f[Gamma, :]
    logp = log_pO.unsqueeze(0) + root_probs                        # [F, S] origination posterior (unnorm, log2)
    log_post = logp - _logsumexp2(logp, dim=1, ).unsqueeze(1)      # normalized log2 posterior over e
    pp_root = torch.exp2(log_post[:, root_branch])                 # [F] PP_f(root) (analytic)

    # 1k-sample the origination per family; count e0 == root_branch
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    post = torch.exp2(log_post).cpu().clamp_min(0)                 # [F, S] linear
    post = post / post.sum(dim=1, keepdim=True)
    samp = torch.multinomial(post, n_samples, replacement=True, generator=g)  # [F, n_samples]
    sampled_frac = (samp == root_branch).double().mean(dim=1)      # [F]

    exp_at_root = float(pp_root.sum().item())
    sampled_at_root = float(sampled_frac.sum().item())
    content_05 = int((pp_root >= 0.5).sum().item())
    return dict(F=F, expected=exp_at_root, sampled=sampled_at_root,
                content_ge05=content_05, root_branch=root_branch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--sidecar-glob", required=True,
                    help="glob with {root} placeholder, e.g. .../cmlL2_100_{root}.rates.txt.json")
    ap.add_argument("--roots", default=None)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda"); dtype = torch.float64
    data_dir = Path(args.data_dir)
    roots = args.roots.split(",") if args.roots else ROOTS

    res = {}
    for r in roots:
        sc = args.sidecar_glob.format(root=r)
        if not Path(sc).exists():
            print(f"[skip] {r}: no sidecar {sc}"); continue
        res[r] = families_at_root(r, sc, data_dir, device, dtype, args.n_samples, args.seed, args.fm_mode)
        v = res[r]
        print(f"[done] {r:16s} F={v['F']}  expected@root={v['expected']:8.1f}  "
              f"sampled1k@root={v['sampled']:8.1f}  content(PP>=.5)={v['content_ge05']}", flush=True)

    print("\n" + "=" * 74)
    print("FAMILIES AT EACH ROOT (l2=100 rates) -- expected & 1k-sampled & thresholded")
    print(f"{'root':16s} {'exp@root':>10s} {'samp1k@root':>12s} {'content(PP>=.5)':>16s}")
    for r in sorted(res, key=lambda x: -res[x]['expected']):
        v = res[r]
        print(f"{r:16s} {v['expected']:10.1f} {v['sampled']:12.1f} {v['content_ge05']:16d}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
