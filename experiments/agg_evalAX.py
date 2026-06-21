"""Analyze eval_bigtree_at_alerax output: for each model (DTL_br2, DTL_br1_O) and
each root, correlate gpurec's per-family logL (@ AleRax's exact rates) against
AleRax's OWN per_fam_likelihoods. Pure python (login-node safe).

  Pearson r ~ 1 + a near-constant offset  => gpurec's ENGINE is faithful on the big
    tree, so the BTroot SGA result is purely an OPTIMIZER/grouping failure (gpurec
    can reach AleRax's Eury optimum, its default fit just doesn't).
  r < 1 / structured residual              => a forward-likelihood gap on the big tree.

Also prints the gpurec @ AleRax-rates AU ranking (from the json) next to AleRax's own
AU (auBT) so the rooting reproduction is visible side by side.
"""
import json, math, glob
from pathlib import Path

D = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine")
BASE = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/alerax_ref/"
            "3_Reconciliation/This_study/5_reconcilation models")
ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury", "HalobacThermopl",
         "Kor", "MHH", "Micra5", "MicraDia", "TACA", "TackA", "TAC", "UndineClu2"]


def dirstem(model, r):
    return f"Undine_C60_{r}root" + ("2_OR" if model == "DTL_br1_O" else "")


def read_alerax_perfam(model, r):
    f = BASE / model / dirstem(model, r) / "per_fam_likelihoods.txt"
    d = {}
    for ln in open(f):
        p = ln.split()
        if len(p) >= 2:
            try:
                d[p[0]] = float(p[1])
            except ValueError:
                pass
    return d


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


for model in ["DTL_br2", "DTL_br1_O"]:
    jf = D / f"evalAX_{model}.json"
    if not jf.exists():
        print(f"\n### {model}: evalAX json not present yet ###")
        continue
    js = json.load(open(jf))
    pf = js["per_family"]
    print(f"\n{'=' * 74}\n### {model}: gpurec @ AleRax rates vs AleRax per_fam_likelihoods ###")
    print(f"{'root':16s} {'n_common':>8s} {'Pearson_r':>10s} {'mean_off(g-a)':>13s} {'sd_off':>8s}")
    for r in ROOTS:
        if r not in pf:
            continue
        g = pf[r]
        a = read_alerax_perfam(model, r)
        # gpurec keys look like 'arCOG00001_01_PMSF.ufboot.ale'; AleRax strips to
        # 'arCOG00001_01_PMSF'. Normalize gpurec -> bare stem before matching.
        def _norm(k):
            return k.replace(".ufboot.ale", "").replace(".ale", "")
        gx, ay = [], []
        for k in g:
            ak = _norm(k)
            if ak in a:
                gx.append(g[k]); ay.append(a[ak])
        if len(gx) < 10:
            print(f"{r:16s} {len(gx):8d}   (too few common families to correlate)")
            continue
        r_p = pearson(gx, ay)
        offs = [x - y for x, y in zip(gx, ay)]
        mo = sum(offs) / len(offs)
        sd = math.sqrt(sum((o - mo) ** 2 for o in offs) / len(offs))
        print(f"{r:16s} {len(gx):8d} {r_p:10.6f} {mo:13.3f} {sd:8.3f}")
    print(f"\n--- gpurec @ AleRax {model} rates: AU ranking (from json) ---")
    print(f"{'root':16s} {'total':>13s} {'BP':>6s} {'AU':>7s}")
    for row in js["au"]:
        print(f"{row['root']:16s} {row['total']:13.1f} {row['BP']:6.3f} {row['AU']:7.4f}")
