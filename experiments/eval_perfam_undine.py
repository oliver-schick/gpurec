"""Forward-only per-family logL at a fitted sidecar's per-branch theta + p^O, for the
UNDINE big tree (This_study/Undine C60). Undine paths + reads origination.origination_prob
(the Stage B sidecars store the fixed rfx p^O there, not omega_log2). Output
{root, names, logL_ln, total} for the branchwise AU test (experiments/au_branchwise.py)."""
import sys, json, glob, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _build_wave_layout, _sp_helpers_for_uniform,
    _build_leaf_E, _parse_fraction_missing,
)
from eval_at_alerax_rates import _load_families_named, _norm_name  # noqa: E402
_LN2 = math.log(2.0)
DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
ALE = sorted(p for p in glob.glob(str(DD / "3_UFBOOTs" / "ufboot_for_alerax" / "*.ale"))
             if not Path(p).name.startswith("._"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--fm-mode", default="e-only")
    ap.add_argument("--ale-list", default=None,
                    help="Stage B: restrict to the .ale paths in this file (basename match).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    dev = torch.device("cuda"); dtype = torch.float64
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.forward import Pi_wave_forward
    tree = DD / "4_species_tree" / f"Undine_C60_{args.root}root_short_name.nw"
    sp = _load_species_helpers(str(tree)); S = int(sp["S"]); s2i = sp["species_name_to_index"]
    d = json.load(open(args.sidecar))
    theta = torch.tensor(d["theta_log2"], dtype=dtype, device=dev).reshape(S, 3)
    od = d.get("origination") or {}
    if od.get("origination_prob") is not None:
        p = torch.tensor(od["origination_prob"], dtype=dtype, device=dev).reshape(S)
        log_pO = torch.log2(p / p.sum())
    elif od.get("omega_log2") is not None:
        om = torch.tensor(od["omega_log2"], dtype=dtype, device=dev).reshape(S)
        log_pO = om - torch.logsumexp(om * _LN2, 0) / _LN2
    else:
        log_pO = torch.full((S,), -math.log2(S), dtype=dtype, device=dev)
    ale_paths = ALE
    if args.ale_list:
        want = {Path(ln.strip()).name for ln in Path(args.ale_list).read_text().splitlines() if ln.strip()}
        ale_paths = [p for p in ALE if Path(p).name in want]
        if not ale_paths:
            raise SystemExit(f"--ale-list {args.ale_list} matched 0 .ale files")
    fams, names = _load_families_named(ale_paths, s2i, min_species=1, dtype=dtype)
    wl, rc = _build_wave_layout(fams, dev, dtype)
    sp_gpu, anc = _sp_helpers_for_uniform(sp, dev, dtype)
    urm = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=dev, dtype=dtype)
    leaf_E = None; leaf_obs = None
    if args.fm_mode != "off" and (DD / "fraction_missing").exists():
        fm, _n, _s = _parse_fraction_missing(DD / "fraction_missing", s2i, S)
        leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=dev, dtype=dtype)
        leaf_obs = leaf_E if args.fm_mode == "both" else None
    lpS, lpD, lpL, tm, mt = extract_parameters_uniform(theta, urm, specieswise=True)
    E_out = E_fixed_point(species_helpers=sp_gpu, log_pS=lpS, log_pD=lpD, log_pL=lpL,
                          transfer_mat=tm, max_transfer_mat=mt, max_iters=4000, tolerance=1e-11,
                          warm_start_E=None, dtype=dtype, device=dev, pibar_mode="uniform",
                          ancestors_T=anc, leaf_E=leaf_E)
    Pi_out = Pi_wave_forward(wave_layout=wl, species_helpers=sp_gpu, E=E_out["E"], Ebar=E_out["E_bar"],
                             E_s1=E_out["E_s1"], E_s2=E_out["E_s2"], log_pS=lpS, log_pD=lpD, log_pL=lpL,
                             transfer_mat=tm, max_transfer_mat=mt, device=dev, dtype=dtype,
                             pibar_mode="uniform", leaf_obs_log=leaf_obs)
    nll_log2 = compute_log_likelihood(Pi_out["Pi"], E_out["E"], rc, log_pO=log_pO)
    logL_ln = [(-v) * _LN2 for v in nll_log2.detach().cpu().tolist()]
    json.dump({"root": args.root, "names": [_norm_name(x) for x in names],
               "logL_ln": logL_ln, "total": sum(logL_ln)}, open(args.out, "w"))
    print(f"root={args.root} fams={len(names)} total={sum(logL_ln):.1f} -> {args.out}", flush=True)


main()
