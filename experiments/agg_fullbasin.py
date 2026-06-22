"""Q1: does FULL per-branch DTL (no clade grouping) HOLD the deep/Eury basin under
pure ML, or drift to the over-fit? Warm-started at each root's AXgrp deep-basin rates,
then free per-branch D,L,T + free O, NO prior. Rank by data_logL_ln.

  Eury/TackA/DPANN stay on top  => full branch-wise DTL works in the basin; the clade
                                   grouping was only a basin-finding crutch.
  drifts to Asgard/SGA          => free per-branch DTL over-fits even in the basin;
                                   the grouping is a genuine regularizer the model needs.
Pure python.
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}
rows = []
for f in glob.glob(f"{D}/FULLbasin_*.rates.txt.json"):
    m = re.search(r"FULLbasin_(\w+)\.rates", f)
    if m:
        d = json.load(open(f))
        dl = d.get("data_log_likelihood_ln")
        if dl is not None:
            rows.append((m.group(1), dl))
rows.sort(key=lambda x: -x[1])
print(f"\n=== FULL per-branch DTL, warm-started in deep basin, pure ML  [{len(rows)}/15] ===")
if not rows:
    print("  (none yet)")
else:
    b0 = rows[0][1]
    eury = dict(rows).get("Eury")
    print(f"  Eury={eury}  ({'HOLDS deep basin' if eury and eury > -1.70e6 else 'drifted'})")
    for root, dl in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- DEEP" if root in {"Eury", "TackA", "DPANN", "MHH"} else ""
        print(f"  {root:16s} {dl:15.1f}  d={dl - b0:9.1f}  {tag}{flag}")
