"""Diagnostic: gpurec's likelihood at AleRax's EXACT big-tree per-branch rates,
all 15 roots, per-family logL -> AU. Isolates OPTIMIZER vs ENGINE.

  DTL_br2   : AleRax's per-branch (D,L,T), UNIFORM origination (no O column).
  DTL_br1_O : AleRax's per-branch (D,L,T) + the fitted per-branch O column.

If gpurec @ AleRax rates reproduces the paper's Eury+MHH AU region (and per-family
logL ~ AleRax per_fam_likelihoods up to a constant CCP offset), gpurec's engine is
faithful on the big tree and only its OPTIMIZER missed AleRax's optimum (the BTroot
sweep landed on SGA). If gpurec @ AleRax rates STILL ranks SGA, there is an
engine/model gap on the big tree. Forward only; e-only fm. Run on an A100.

  python experiments/eval_bigtree_at_alerax.py --model DTL_br2   --out undine/evalAX_DTL_br2.json
  python experiments/eval_bigtree_at_alerax.py --model DTL_br1_O --out undine/evalAX_DTL_br1_O.json
"""
from __future__ import annotations
import argparse, glob, math, sys, json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402  (shared loaders)
    _load_species_helpers, _build_wave_layout, _sp_helpers_for_uniform,
    _build_leaf_E, _parse_fraction_missing)
from eval_at_alerax_rates import _load_families_named  # noqa: E402
from run_undine_branchwise import _fraction_missing_path  # noqa: E402
from clade_groups import branch_params_from_alerax  # noqa: E402

torch.set_default_dtype(torch.float64)
_LN2 = math.log(2.0)
DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
BASE = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/alerax_ref/"
            "3_Reconciliation/This_study/5_reconcilation models")
ALE = sorted(p for p in glob.glob(str(DD / "3_UFBOOTs" / "ufboot_for_alerax" / "*.ale"))
             if not Path(p).name.startswith("._"))
ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury", "HalobacThermopl",
         "Kor", "MHH", "Micra5", "MicraDia", "TACA", "TackA", "TAC", "UndineClu2"]
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}


def dirstem(model, r):
    return f"Undine_C60_{r}root" + ("2_OR" if model == "DTL_br1_O" else "")


def eval_root(model, root, device, dtype, fm_mode):
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.forward import Pi_wave_forward

    tree_path = DD / "4_species_tree" / f"Undine_C60_{root}root_short_name.nw"
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"]); s2i = sp["species_name_to_index"]
    mdir = BASE / model / dirstem(model, root)
    mp = mdir / "model_parameters" / "model_parameters.txt"
    atree = mdir / "species_trees" / "starting_species_tree.newick"
    atree = str(atree) if atree.exists() else str(tree_path)
    rate_branch, orig_branch, diag = branch_params_from_alerax(sp, atree, str(mp))
    nun = len(diag.get("unmapped", []))
    if nun:
        print(f"  [warn] {model} {root}: {nun} unmapped branches "
              f"(eval proceeds with mapped subset)", flush=True)
    theta = torch.log2(rate_branch.clamp_min(1e-10)).to(device=device, dtype=dtype)
    log_pO = None
    if model == "DTL_br1_O" and orig_branch is not None:
        o = orig_branch.to(device=device, dtype=dtype).clamp_min(1e-12)
        log_pO = torch.log2(o / o.sum())

    fams, names = _load_families_named(ALE, s2i, min_species=1, dtype=dtype)
    sp_gpu, anc = _sp_helpers_for_uniform(sp, device, dtype)
    urm = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)

    leaf_E = None
    if fm_mode != "off":
        fmp = _fraction_missing_path(DD)
        if fmp.exists():
            fm, _n, _s = _parse_fraction_missing(fmp, s2i, S)
            leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
    leaf_obs_log = leaf_E if fm_mode == "both" else None

    lp = extract_parameters_uniform(theta, urm, specieswise=True)
    # E is family-INDEPENDENT -> compute once and reuse across family batches.
    E = E_fixed_point(species_helpers=sp_gpu, log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                      transfer_mat=lp[3], max_transfer_mat=lp[4], max_iters=4000,
                      tolerance=1e-10, warm_start_E=None, dtype=dtype, device=device,
                      pibar_mode="uniform", ancestors_T=anc, leaf_E=leaf_E)
    # BATCH the forward over families: one giant wave layout for all 7059 families at
    # S=513 overflows the wave kernels (illegal memory access); BTroot used batch 500.
    CHUNK = 500
    per = []
    for i in range(0, len(fams), CHUNK):
        chunk = fams[i:i + CHUNK]
        wl, rc = _build_wave_layout(chunk, device, dtype)
        Pi = Pi_wave_forward(wave_layout=wl, species_helpers=sp_gpu, E=E["E"], Ebar=E["E_bar"],
                             E_s1=E["E_s1"], E_s2=E["E_s2"], log_pS=lp[0], log_pD=lp[1],
                             log_pL=lp[2], transfer_mat=lp[3], max_transfer_mat=lp[4],
                             device=device, dtype=dtype, pibar_mode="uniform",
                             leaf_obs_log=leaf_obs_log)
        nll = compute_log_likelihood(Pi["Pi"], E["E"], rc, log_pO=log_pO)
        per.extend([float(-v) * _LN2 for v in nll.detach().cpu().tolist()])
    n_groups = len({tuple(r.tolist()) for r in rate_branch})
    return names, per, n_groups


def au_rank(per_by_root, roots):
    common = sorted(set.intersection(*[set(d) for d in per_by_root.values()]))
    n, R = len(common), len(roots)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = torch.tensor([[per_by_root[r][f] for r in roots] for f in common], device=dev)
    tot = L.sum(0)
    gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]; B = 20000
    from torch.distributions import Multinomial
    probs = torch.full((n,), 1.0 / n, device=dev)
    BP = torch.zeros(len(gammas), R, device=dev)
    for k, g in enumerate(gammas):
        m = int(round(g * n))
        BP[k] = torch.bincount((Multinomial(m, probs).sample((B,)) @ L).argmax(1),
                               minlength=R).double() / B
    Phi = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    Phi_inv = lambda p: math.sqrt(2) * torch.erfinv(2 * p - 1)
    sig = torch.tensor([1 / math.sqrt(g) for g in gammas], device=dev)
    res = []
    for r in range(R):
        bp = BP[:, r].clamp(1.0 / (2 * B), 1 - 1.0 / (2 * B)); z = Phi_inv(1 - bp)
        phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        w = (B * phi * phi) / (bp * (1 - bp)).clamp_min(1e-12); sw = w.sqrt()
        X = torch.stack([1 / sig, sig], 1)
        beta = torch.linalg.lstsq((X * sw.unsqueeze(1)), (z * sw)).solution
        # AU edge cases. The multiscale regression needs BP to VARY across scales to
        # estimate the signed distance/curvature; at the saturated extremes it is
        # degenerate (z is constant across scales -> beta is meaningless, ~0.63).
        bpmin = float(BP[:, r].min()); bpmax = float(BP[:, r].max())
        if bpmax == 0.0:
            au = 0.0                      # never selected at any scale -> AU 0
        elif bpmin >= 1.0 - 1e-12:
            au = 1.0                      # selected in EVERY replicate at EVERY scale
                                          # (dominant tree) -> AU 1 by convention, not the
                                          # degenerate-regression ~0.63
        else:
            au = float(1 - Phi(beta[0] - beta[1]))
        res.append((roots[r], float(tot[r]), float(BP[5, r]), au))
    res.sort(key=lambda x: -x[1])
    return res, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DTL_br2", choices=["global", "DTL_br2", "DTL_br1_O"])
    ap.add_argument("--fm-mode", default="e-only", choices=["off", "e-only", "both"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (A100).")
    device = torch.device("cuda"); dtype = torch.float64

    per_by_root = {}
    for r in ROOTS:
        try:
            names, per, ng = eval_root(args.model, r, device, dtype, args.fm_mode)
        except Exception as e:
            print(f"[FAIL] {args.model} {r:16s}: {type(e).__name__}: {e}", flush=True)
            continue
        per_by_root[r] = dict(zip(names, per))
        print(f"[done] {args.model} {r:16s} total={sum(per):14.1f}  F={len(per)}  G={ng}",
              flush=True)
    roots = [r for r in ROOTS if r in per_by_root]
    if len(roots) < len(ROOTS):
        print(f"[warn] only {len(roots)}/{len(ROOTS)} roots evaluated; "
              f"AU over the available subset", flush=True)
    if len(roots) < 2:
        raise SystemExit("too few roots succeeded for AU")

    res, n = au_rank(per_by_root, roots)
    b0 = res[0][1]
    print(f"\n=== gpurec @ AleRax {args.model} rates -> AU  (n={n} fam, 15 roots) ===")
    print(f"{'root':16s} {'totalLogL':>13s} {'dTot':>9s} {'BP':>6s} {'AU':>7s}  verdict")
    for nm, t, bp, au in res:
        tag = "[deep]" if nm in DEEP else ("[SGA]" if nm in SGA else "")
        v = "BEST" if au >= 0.95 else ("in 95% set" if au >= 0.05 else "REJECTED")
        print(f"{nm:16s} {t:13.1f} {t - b0:9.1f} {bp:6.3f} {au:7.4f}  {v} {tag}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"model": args.model, "fm_mode": args.fm_mode,
             "au": [{"root": nm, "total": t, "BP": bp, "AU": au} for nm, t, bp, au in res],
             "per_family": per_by_root}))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
