"""Cross-validation verdict on overfitting: score HELD-OUT families under the free
per-branch vs the clade-grouped rates, both fit on the TRAIN half. If free beats
grouped on the held-out families, the extra ~2000 params generalize = NOT overfitting.

Train = first --train-n families (the slice the fits used); test = the rest.
Reports total + per-family-mean logL on train and test for both models.
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

_LN2 = math.log(2.0)
DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
ALE = sorted(p for p in glob.glob(str(DD / "3_UFBOOTs" / "ufboot_for_alerax" / "*.ale"))
             if not Path(p).name.startswith("._"))
CHUNK = 500


def _lse2(x, dim):
    return torch.logsumexp(x * _LN2, dim=dim) / _LN2


def per_fam_logl(theta, log_pO, fams, sp_gpu, anc, urm, leaf_E, leaf_obs_log, device, dtype):
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.forward import Pi_wave_forward
    lp = extract_parameters_uniform(theta, urm, specieswise=True)
    E = E_fixed_point(species_helpers=sp_gpu, log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                      transfer_mat=lp[3], max_transfer_mat=lp[4], max_iters=4000,
                      tolerance=1e-10, warm_start_E=None, dtype=dtype, device=device,
                      pibar_mode="uniform", ancestors_T=anc, leaf_E=leaf_E)
    out = []
    for i in range(0, len(fams), CHUNK):
        chunk = fams[i:i + CHUNK]
        wl, rc = _build_wave_layout(chunk, device, dtype)
        Pi = Pi_wave_forward(wave_layout=wl, species_helpers=sp_gpu, E=E["E"], Ebar=E["E_bar"],
                             E_s1=E["E_s1"], E_s2=E["E_s2"], log_pS=lp[0], log_pD=lp[1],
                             log_pL=lp[2], transfer_mat=lp[3], max_transfer_mat=lp[4],
                             device=device, dtype=dtype, pibar_mode="uniform",
                             leaf_obs_log=leaf_obs_log)
        nll = compute_log_likelihood(Pi["Pi"], E["E"], rc, log_pO=log_pO)
        out.extend([float(-v) * _LN2 for v in nll.detach().cpu().tolist()])
    return out


def load_rates(sidecar, S, device, dtype):
    d = json.load(open(sidecar))
    theta = torch.tensor(d["theta_log2"], dtype=dtype, device=device).reshape(S, 3)
    om = d.get("origination", {}).get("omega_log2")
    log_pO = None
    if om is not None:
        om = torch.tensor(om, dtype=dtype, device=device).reshape(S)
        log_pO = om - _lse2(om, dim=0)
    return theta, log_pO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Eury")
    ap.add_argument("--train-n", type=int, default=3530)
    ap.add_argument("--free", required=True)
    ap.add_argument("--grouped", required=True)
    ap.add_argument("--fm-mode", default="e-only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda"); dtype = torch.float64

    tree = DD / "4_species_tree" / f"Undine_C60_{args.root}root_short_name.nw"
    sp = _load_species_helpers(str(tree)); S = int(sp["S"]); s2i = sp["species_name_to_index"]
    fams, names = _load_families_named(ALE, s2i, min_species=1, dtype=dtype)
    sp_gpu, anc = _sp_helpers_for_uniform(sp, device, dtype)
    urm = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)
    leaf_E = None
    fmp = DD / "fraction_missing"
    if args.fm_mode != "off" and fmp.exists():
        fm, _n, _s = _parse_fraction_missing(fmp, s2i, S)
        leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
    leaf_obs_log = leaf_E if args.fm_mode == "both" else None

    res = {}
    for tag, sc in [("free", args.free), ("grouped", args.grouped)]:
        theta, log_pO = load_rates(sc, S, device, dtype)
        ll = per_fam_logl(theta, log_pO, fams, sp_gpu, anc, urm, leaf_E, leaf_obs_log, device, dtype)
        n = args.train_n
        tr = ll[:n]; te = ll[n:]
        res[tag] = dict(train_sum=sum(tr), test_sum=sum(te),
                        train_mean=sum(tr) / len(tr), test_mean=sum(te) / len(te),
                        n_train=len(tr), n_test=len(te))

    print("\n" + "=" * 64)
    print(f"CROSS-VALIDATION (root={args.root}, train_n={args.train_n})  -- logL (ln)")
    print(f"{'model':9s} {'train sum':>14s} {'TEST sum':>14s} {'train/fam':>10s} {'TEST/fam':>10s}")
    for tag in ("free", "grouped"):
        r = res[tag]
        print(f"{tag:9s} {r['train_sum']:14.1f} {r['test_sum']:14.1f} "
              f"{r['train_mean']:10.4f} {r['test_mean']:10.4f}")
    dtr = res["free"]["train_sum"] - res["grouped"]["train_sum"]
    dte = res["free"]["test_sum"] - res["grouped"]["test_sum"]
    print(f"\n  free - grouped  TRAIN = {dtr:+.1f}   HELD-OUT = {dte:+.1f}")
    print(f"  VERDICT: {'NOT overfitting (free generalizes: held-out gain > 0)' if dte > 0 else 'OVERFITTING (free loses out-of-sample)'}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
