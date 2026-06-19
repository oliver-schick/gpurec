"""EXACT-MATCH check: gpurec vs AleRax per-family log-likelihood at AleRax's OWN
fitted BRANCH-WISE rates -- the FULL structured model (per-branch D,L,T AND the
per-branch origination O distribution), not just the global rates.

Sets gpurec's theta to AleRax's per-branch (D,L,T) and log_pO to AleRax's per-
branch origination (the O column, renormalised to a distribution), runs the
gpurec forward (e-only fraction-missing), computes per-family logL, and compares
family-by-family to AleRax's `reconciliation models/branch wise/<root>/
per_fam_likelihoods.txt`. If gpurec's likelihood == AleRax's for the structured
DTLO model, the per-family logL matches up to the constant CCP-normalisation
offset (Pearson r ~ 1), confirming the exact match before we touch regularization.

Run on an A100.
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
    _load_species_helpers, _build_wave_layout, _sp_helpers_for_uniform,
    _build_leaf_E, _parse_fraction_missing,
)
from eval_at_alerax_rates import (  # noqa: E402
    _load_families_named, _parse_alerax_perfam, _norm_name,
)
from clade_groups import branch_params_from_alerax  # noqa: E402

_LN2 = math.log(2.0)
DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Eury")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--uniform-o", action="store_true",
                    help="ignore AleRax O (use uniform p^O) -- contrast run")
    ap.add_argument("--min-species", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (A100).")
    device = torch.device("cuda")
    dtype = torch.float64
    data_dir = Path(args.data_dir)

    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.forward import Pi_wave_forward

    tree_path = data_dir / "rooted_phylogeny" / args.root
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"]); sp_name_to_idx = sp["species_name_to_index"]

    bw = data_dir / "reconciliation models" / "branch wise" / args.root
    cg_tree = bw / "species_trees" / "starting_species_tree.newick"
    cg_mp = bw / "model_parameters" / "model_parameters.txt"
    rate_branch, orig_branch, diag = branch_params_from_alerax(sp, str(cg_tree), str(cg_mp))
    if diag["unmapped"]:
        raise SystemExit(f"unmapped nodes: {diag['unmapped'][:3]}")
    theta = torch.log2(rate_branch.clamp_min(1e-10)).to(device=device, dtype=dtype)
    log_pO = None
    if not args.uniform_o and orig_branch is not None:
        o = orig_branch.to(device=device, dtype=dtype).clamp_min(1e-12)
        log_pO = torch.log2(o / o.sum())
    print(f"[rates] root={args.root}  per-branch D,L,T + "
          f"{'AleRax O' if log_pO is not None else 'UNIFORM O'}", flush=True)

    ale_paths = sorted(p for p in glob.glob(str(data_dir / "ccps" / "*.ale"))
                       if not Path(p).name.startswith("._"))
    fams, names = _load_families_named(ale_paths, sp_name_to_idx,
                                       min_species=args.min_species, dtype=dtype)
    print(f"[load] S={S}  kept {len(fams)} families", flush=True)
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
                             log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, transfer_mat=transfer_mat,
                             max_transfer_mat=mt, device=device, dtype=dtype, pibar_mode="uniform",
                             leaf_obs_log=leaf_obs_log)
    nll_log2 = compute_log_likelihood(Pi_out["Pi"], E_out["E"], root_clade_ids, log_pO=log_pO)
    logL_ln = [(-v) * _LN2 for v in nll_log2.detach().cpu().tolist()]

    alerax = _parse_alerax_perfam(str(bw / "per_fam_likelihoods.txt"))
    alerax = {_norm_name(k): v for k, v in alerax.items()}
    g = {_norm_name(nm): lv for nm, lv in zip(names, logL_ln)}
    common = sorted(k for k in g if k in alerax)
    gv = [g[k] for k in common]; av = [alerax[k] for k in common]
    n = len(common)
    diffs = [x - y for x, y in zip(gv, av)]
    mean_d = sum(diffs) / n; rmse = math.sqrt(sum(d * d for d in diffs) / n)
    mg, ma = sum(gv) / n, sum(av) / n
    sxy = sum((x - mg) * (y - ma) for x, y in zip(gv, av))
    sxx = sum((x - mg) ** 2 for x in gv); syy = sum((y - ma) ** 2 for y in av)
    pear = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")
    out = []
    def emit(s=""): out.append(s); print(s, flush=True)
    emit("=" * 74)
    emit(f"EXACT MATCH: gpurec vs AleRax branch-wise per-family logL  root={args.root}  "
         f"({'AleRax O' if log_pO is not None else 'uniform O'}, fm={args.fm_mode})")
    emit("=" * 74)
    emit(f"  matched families : {n}")
    emit(f"  TOTAL logL  gpurec={sum(gv):.1f}  AleRax={sum(av):.1f}  "
         f"Delta={sum(gv)-sum(av):+.1f} ({(sum(gv)-sum(av))/n:+.4f}/fam)")
    emit(f"  per-family Delta: mean={mean_d:+.4f}  rmse={rmse:.4f}")
    emit(f"  per-family Pearson r = {pear:.6f}")
    emit("  (r~1 + small rmse => gpurec == AleRax for the full structured DTLO model)")
    if args.out:
        Path(args.out).write_text("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
