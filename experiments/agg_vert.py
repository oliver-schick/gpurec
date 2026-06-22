"""Rank big-tree roots from VERTtest: DTL_br1_O + AleRax grouping + ROOT-MASS
origination prior (lambda*(1-p^O_root), favors root/Eury-region O concentration),
UNIFORM init. Tests the CORRECT-DIRECTION prior for the big tree (where BARtest's
anti-concentration barrier was directionally wrong: it pushed O->uniform and sank
the Eury region to the bottom). Per lambda. Pure python.
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}

lams = sorted({re.search(r"VERTtest_L(\d+)_", f).group(1)
               for f in glob.glob(f"{D}/VERTtest_L*_*.rates.txt.json")},
              key=lambda x: int(x))
if not lams:
    print("(no VERTtest outputs yet)")
for lam in lams:
    rows = []
    for f in glob.glob(f"{D}/VERTtest_L{lam}_*.rates.txt.json"):
        m = re.search(rf"VERTtest_L{lam}_(\w+)\.rates", f)
        if not m:
            continue
        d = json.load(open(f))
        dl = d.get("data_log_likelihood_ln")
        if dl is not None:
            rows.append((m.group(1), dl))
    rows.sort(key=lambda x: -x[1])
    print(f"\n=== VERTtest lambda={lam}: DTL_br1_O + root-mass prior, UNIFORM init  "
          f"[{len(rows)}/15 roots] ===")
    if not rows:
        print("  (none finished yet)")
        continue
    b0 = rows[0][1]
    for root, dl in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- DEEP" if root in {"Eury", "TackA", "DPANN", "MHH"} else ""
        print(f"  {root:16s} {dl:15.1f}  d={dl - b0:9.1f}  {tag}{flag}")
