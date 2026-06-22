"""Rank big-tree roots from BARtest: DTL_br1_O + AleRax grouping + Simpson O barrier,
UNIFORM init (no AleRax init). Per barrier-c. Tests whether the anti-concentration
barrier ALONE (no good-init "cheat") forbids the degenerate-O basin so gpurec recovers
Eury from uniform init = the init-independent principled cure.

  Eury/TackA/DPANN on top  => barrier works (gpurec-native principled rooting)
  SGA on top               => barrier too blunt OR optimizer stalled; need a smarter prior

Pure python (login-node ok). Usage: python3 agg_bartest.py
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}

cs = sorted({re.search(r"BARtest_c(\d+)_", f).group(1)
             for f in glob.glob(f"{D}/BARtest_c*_*.rates.txt.json")},
            key=lambda x: int(x))
if not cs:
    print("(no BARtest outputs yet)")
for c in cs:
    rows = []
    for f in glob.glob(f"{D}/BARtest_c{c}_*.rates.txt.json"):
        m = re.search(rf"BARtest_c{c}_(\w+)\.rates", f)
        if not m:
            continue
        d = json.load(open(f))
        dl = d.get("data_log_likelihood_ln")
        if dl is not None:
            rows.append((m.group(1), dl))
    rows.sort(key=lambda x: -x[1])
    print(f"\n=== BARtest c={c}: DTL_br1_O + barrier, UNIFORM init  [{len(rows)}/15 roots] ===")
    if not rows:
        print("  (none finished yet)")
        continue
    b0 = rows[0][1]
    for root, dl in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- DEEP" if root in {"Eury", "TackA", "DPANN", "MHH"} else ""
        print(f"  {root:16s} {dl:15.1f}  d={dl - b0:9.1f}  {tag}{flag}")
