"""Forward-only: load a gpurec sidecar's FITTED per-branch theta (+omega), recompute
the per-family logL for one root. Output {root, names, logL_ln} for the AU test."""
import sys, json, glob, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (
    _load_species_helpers, _load_families_named, _build_wave_layout,
    _sp_helpers_for_uniform, _build_leaf_E, _parse_fraction_missing, _norm_name,
)
_LN2 = math.log(2.0)
DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--fm-mode", default="e-only")
    ap.add_argument("--min-species", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda"); dtype = torch.float64
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import (E_fixed_point, compute_log_likelihood,
                                        origination_log_pO)
    from gpurec.core.forward import Pi_wave_forward
    data_dir = Path(args.data_dir)
    sp = _load_species_helpers(str(data_dir / "rooted_phylogeny" / args.root))
    S = int(sp["S"]); sp_name_to_idx = sp["species_name_to_index"]
    d = json.load(open(args.sidecar))
    theta = torch.tensor(d["theta_log2"], dtype=dtype, device=device)        # [S,3]
    om = (d.get("origination") or {}).get("omega_log2")
    log_pO = origination_log_pO(torch.tensor(om, dtype=dtype, device=device)) if om else None
    ale_paths = sorted(p for p in glob.glob(str(data_dir / "ccps" / "*.ale"))
                       if not Path(p).name.startswith("._"))
    fams, names = _load_families_named(ale_paths, sp_name_to_idx,
                                       min_species=args.min_species, dtype=dtype)
    wave_layout, root_clade_ids = _build_wave_layout(fams, device, dtype)
    sp_gpu, ancestors_T = _sp_helpers_for_uniform(sp, device, dtype)
    unnorm_row_max = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)
    leaf_E = None
    if args.fm_mode != "off":
        fm_path = data_dir / "fraction_missing"
        if fm_path.exists():
            fm, _n, _ = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
            leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
    leaf_obs_log = leaf_E if args.fm_mode == "both" else None
    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
        theta, unnorm_row_max, specieswise=True)
    E_out = E_fixed_point(species_helpers=sp_gpu, log_pS=log_pS, log_pD=log_pD,
                          log_pL=log_pL, transfer_mat=transfer_mat, max_transfer_mat=mt,
                          max_iters=4000, tolerance=1e-11, warm_start_E=None, dtype=dtype,
                          device=device, pibar_mode="uniform", ancestors_T=ancestors_T,
                          leaf_E=leaf_E)
    Pi_out = Pi_wave_forward(wave_layout=wave_layout, species_helpers=sp_gpu, E=E_out["E"],
                             Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
                             log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                             transfer_mat=transfer_mat, max_transfer_mat=mt, device=device,
                             dtype=dtype, pibar_mode="uniform", leaf_obs_log=leaf_obs_log)
    nll_log2 = compute_log_likelihood(Pi_out["Pi"], E_out["E"], root_clade_ids, log_pO=log_pO)
    logL_ln = [(-v) * _LN2 for v in nll_log2.detach().cpu().tolist()]
    json.dump({"root": args.root, "names": [_norm_name(x) for x in names],
               "logL_ln": logL_ln, "total": sum(logL_ln)}, open(args.out, "w"))
    print(f"root={args.root} fams={len(names)} total={sum(logL_ln):.1f} -> {args.out}", flush=True)


main()
