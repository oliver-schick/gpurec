"""Cross-validate gpurec vs AleRax LIKELIHOODS at AleRax's OWN fitted rates.

The definitive test of model agreement: take AleRax's fitted GLOBAL rates
(model_parameters.txt: one shared [D,L,T] for all branches), set gpurec's theta
to those EXACT rates (no optimization; broadcast to every branch == the global
model), run the gpurec forward with e-only fraction-missing and UNIFORM
origination (matching AleRax's global parametrization), and compute the
PER-FAMILY log-likelihood. Then compare family-by-family to AleRax's
per_fam_likelihoods.txt. If the two implementations' likelihood functions agree,
the per-family logL matches.

gpurec per-family logL (uniform origination, log2 units) is exactly
    logsumexp2(Pi[root,:]) - log2(S) - log2(1 - mean(exp2(E)))
(compute_log_likelihood returns its negative). We convert to ln (AleRax units).

Usage (cluster, A100):
    python experiments/eval_at_alerax_rates.py --root Eury \
        --rates 0.118554,0.211398,0.214623 --rate-order DLT \
        --alerax-perfam ".../global/Eury/per_fam_likelihoods.txt" --fm-mode e-only
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
    _build_leaf_E, _parse_fraction_missing, _INV_LN2,
)

_LN2 = math.log(2.0)
DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")


def _load_families_named(ale_paths, sp_name_to_idx, *, min_species, dtype):
    """Like run_williams_branchwise._load_families but also returns the per-family
    .ale basename (for aligning to AleRax per_fam_likelihoods)."""
    from gpurec.io.ale import parse_ale_file, build_family_from_ale, default_species_of_leaf
    fams, names = [], []
    for path in ale_paths:
        try:
            ale = parse_ale_file(path)
        except Exception:
            continue
        sp_set = {default_species_of_leaf(n) for n in ale.leaf_name_to_id}
        if any(s not in sp_name_to_idx for s in sp_set):
            continue
        if len(sp_set) < min_species:
            continue
        try:
            fam = build_family_from_ale(ale, sp_name_to_idx, dtype=dtype)
        except Exception:
            continue
        ccp = fam["ccp"]
        ccp["log_split_probs_sorted"] = ccp["log_split_probs_sorted"] * _INV_LN2
        fams.append({"ccp_helpers": ccp, "root_clade_id": int(fam["root_clade_id"]),
                     "leaf_row_index": fam["leaf_row_index"],
                     "leaf_col_index": fam["leaf_col_index"],
                     "C": int(ccp["C"]), "N_splits": int(ccp["N_splits"])})
        names.append(Path(path).name)
    return fams, names


def _parse_alerax_perfam(path):
    """name -> per-family logL (ln). AleRax strips the .ale extension; we match
    on the basename with common extensions removed."""
    out = {}
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                out[p[0]] = float(p[1])
            except ValueError:
                pass
    return out


def _norm_name(n):
    """Normalise a family name for matching (drop .ale/.txt extensions)."""
    for ext in (".ale", ".txt", ".ufboot"):
        if n.endswith(ext):
            n = n[: -len(ext)]
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Eury")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--rates", required=True, help="three AleRax global rates, comma-sep")
    ap.add_argument("--rate-order", default="DLT",
                    help="column order of --rates (permutation of D,L,T). gpurec "
                         "wants [D,L,T]; try DLT then DTL if the fit is poor.")
    ap.add_argument("--alerax-perfam", required=True)
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--min-species", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (run on A100).")
    device = torch.device("cuda")
    dtype = torch.float64
    data_dir = Path(args.data_dir)

    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.forward import Pi_wave_forward

    # ---- rates -> theta[S,3] (global broadcast, reordered to D,L,T) ----------
    vals = [float(x) for x in args.rates.split(",")]
    if len(vals) != 3:
        raise SystemExit("--rates needs exactly 3 numbers")
    order = args.rate_order.upper()
    if sorted(order) != ["D", "L", "T"]:
        raise SystemExit("--rate-order must be a permutation of D,L,T")
    rate_of = {order[i]: vals[i] for i in range(3)}
    D, L, T = rate_of["D"], rate_of["L"], rate_of["T"]
    print(f"[rates] interpreted as D={D} L={L} T={T} (input {args.rates} order {order})",
          flush=True)

    # ---- species tree + families (with names) -------------------------------
    tree_path = data_dir / "rooted_phylogeny" / args.root
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"]); sp_name_to_idx = sp["species_name_to_index"]
    ale_paths = sorted(p for p in glob.glob(str(data_dir / "ccps" / "*.ale"))
                       if not Path(p).name.startswith("._"))
    print(f"[load] S={S}, {len(ale_paths)} .ale; building families ...", flush=True)
    fams, names = _load_families_named(ale_paths, sp_name_to_idx,
                                       min_species=args.min_species, dtype=dtype)
    print(f"[load] kept {len(fams)} families", flush=True)
    wave_layout, root_clade_ids = _build_wave_layout(fams, device, dtype)
    sp_gpu, ancestors_T = _sp_helpers_for_uniform(sp, device, dtype)
    unnorm_row_max = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)

    # ---- fraction-missing (e-only: leaf_E set, leaf_obs_log=None) ------------
    leaf_E = None; leaf_obs_log = None
    fm_path = data_dir / "fraction_missing"
    if args.fm_mode != "off" and fm_path.exists():
        fm, n_set, _ = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
        leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
        leaf_obs_log = leaf_E if args.fm_mode == "both" else None
        print(f"[fm] {args.fm_mode}: {n_set} rows mapped", flush=True)

    # ---- theta (global) + forward -------------------------------------------
    theta = torch.tensor([math.log2(D), math.log2(L), math.log2(T)],
                         dtype=dtype, device=device).unsqueeze(0).expand(S, 3).contiguous()
    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
        theta, unnorm_row_max, specieswise=True)
    E_out = E_fixed_point(
        species_helpers=sp_gpu, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt, max_iters=4000, tolerance=1e-10,
        warm_start_E=None, dtype=dtype, device=device, pibar_mode='uniform',
        ancestors_T=ancestors_T, leaf_E=leaf_E)
    Pi_out = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sp_gpu, E=E_out['E'], Ebar=E_out['E_bar'],
        E_s1=E_out['E_s1'], E_s2=E_out['E_s2'], log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt, device=device, dtype=dtype,
        pibar_mode='uniform', leaf_obs_log=leaf_obs_log)
    nll_log2 = compute_log_likelihood(Pi_out['Pi'], E_out['E'], root_clade_ids, log_pO=None)
    logL_ln = (-nll_log2).detach().cpu().tolist()
    logL_ln = [v * _LN2 for v in logL_ln]

    # ---- align to AleRax + compare ------------------------------------------
    alerax = _parse_alerax_perfam(args.alerax_perfam)
    alerax_norm = {_norm_name(k): v for k, v in alerax.items()}
    g = {}
    for nm, lv in zip(names, logL_ln):
        g[_norm_name(nm)] = lv
    common = sorted(k for k in g if k in alerax_norm)
    print(f"\n[compare] gpurec families={len(g)}  AleRax families={len(alerax)}  "
          f"matched={len(common)}", flush=True)
    if not common:
        print("  [!] no name overlap — gpurec names e.g.:", list(g)[:3])
        print("      AleRax names e.g.:", list(alerax_norm)[:3])
        return 1

    gv = [g[k] for k in common]
    av = [alerax_norm[k] for k in common]
    diffs = [x - y for x, y in zip(gv, av)]
    n = len(common)
    sum_g, sum_a = sum(gv), sum(av)
    mean_d = sum(diffs) / n
    rmse = math.sqrt(sum(d * d for d in diffs) / n)
    max_d = max(diffs, key=abs)
    # Pearson
    mg, ma = sum_g / n, sum_a / n
    sxy = sum((x - mg) * (y - ma) for x, y in zip(gv, av))
    sxx = sum((x - mg) ** 2 for x in gv); syy = sum((y - ma) ** 2 for y in av)
    pear = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")

    lines = []
    def emit(s=""):
        lines.append(s); print(s, flush=True)
    emit("=" * 78)
    emit(f"gpurec vs AleRax per-family log-likelihood  (root={args.root}, "
         f"global rates, fm={args.fm_mode})")
    emit("=" * 78)
    emit(f"  matched families : {n}")
    emit(f"  TOTAL logL  gpurec = {sum_g:.2f}   AleRax = {sum_a:.2f}   "
         f"Δ = {sum_g - sum_a:+.2f} ({(sum_g - sum_a)/n:+.4f}/family)")
    emit(f"  per-family Δ (gpurec−AleRax):  mean={mean_d:+.4f}  rmse={rmse:.4f}  "
         f"max|Δ|={max_d:+.3f}")
    emit(f"  per-family Pearson r = {pear:.6f}")
    emit("")
    worst = sorted(common, key=lambda k: -abs(g[k] - alerax_norm[k]))[:8]
    emit(f"  largest |Δ| families:")
    emit(f"    {'family':<28}{'gpurec':>12}{'AleRax':>12}{'Δ':>10}")
    for k in worst:
        emit(f"    {k:<28}{g[k]:>12.3f}{alerax_norm[k]:>12.3f}{g[k]-alerax_norm[k]:>10.3f}")
    emit("")
    emit("  interpretation: mean Δ ~ a constant offset (normalisation/convention); "
         "small RMSE + r≈1 => the likelihood MODELS agree per family.")
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
