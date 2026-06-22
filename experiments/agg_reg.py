"""SOTA regularized branch-wise DTL screen (REGtest): full per-branch DTL warm-started
in the deep basin, regularized by fused-lasso/TV or Brownian (vs FULLbasin = free,
unregularized). Per method: rank roots by data_logL_ln AND report the effective number
of distinct per-branch rate tuples (eff#) -- the test of whether the regularizer
auto-GROUPS (TV -> few distinct rates = data-driven clade grouping w/o overparam).
Pure python.
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}


def eff_groups(rates):
    # count distinct (D,L,T) tuples rounded to 3 sig digits -> ~ number of rate classes
    seen = set()
    for r in rates:
        seen.add(tuple(round(float(x), 3) for x in r))
    return len(seen)


methods = sorted({re.search(r"REGtest_([A-Za-z0-9.]+)_", f).group(1)
                  for f in glob.glob(f"{D}/REGtest_*_*.rates.txt.json")})
if not methods:
    print("(no REGtest outputs yet)")
for nm in methods:
    rows = []
    for f in glob.glob(f"{D}/REGtest_{nm}_*.rates.txt.json"):
        m = re.search(rf"REGtest_{nm}_(\w+)\.rates", f)
        if not m:
            continue
        d = json.load(open(f))
        dl = d.get("data_log_likelihood_ln")
        eff = eff_groups(d.get("rates", []))
        if dl is not None:
            rows.append((m.group(1), dl, eff))
    rows.sort(key=lambda x: -x[1])
    print(f"\n=== REGtest {nm}: regularized branch-wise DTL (deep-basin warm-start)  "
          f"[{len(rows)} roots] ===")
    if not rows:
        continue
    b0 = rows[0][1]
    for root, dl, eff in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- DEEP" if root in {"Eury", "TackA", "DPANN", "MHH"} else ""
        print(f"  {root:16s} {dl:15.1f}  d={dl - b0:9.1f}  eff#rates={eff:4d}/513  {tag}{flag}")
