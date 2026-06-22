"""Rank big-tree roots by the FINAL (lambda=0, pure ML) stage of the annealing
chain -- the generic SOTA basin-finding result. If the depth-prior homotopy worked,
the cold-start ML now lands in the DEEP basin: Eury/TackA/DPANN on top (matching the
AleRax-init AXgrp result), instead of the SGA trap. Pure python.

Reference: deep basin (good) Eury ~ -1,642,425 ; SGA trap Eury ~ -1,737,101.
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}

rows = []
for f in glob.glob(f"{D}/ANNEAL_*_L0.rates.txt.json"):
    m = re.search(r"ANNEAL_(\w+)_L0\.rates", f)
    if not m:
        continue
    d = json.load(open(f))
    dl = d.get("data_log_likelihood_ln")
    if dl is not None:
        rows.append((m.group(1), dl))
rows.sort(key=lambda x: -x[1])
print(f"\n=== ANNEALED cold-start ML (lambda->0), final stage  [{len(rows)}/15 roots] ===")
if not rows:
    print("  (no final-stage outputs yet)")
else:
    b0 = rows[0][1]
    eury = dict(rows).get("Eury")
    deep_ok = eury is not None and eury > -1.70e6
    print(f"  basin: Eury={eury} -> {'DEEP (recovered!)' if deep_ok else 'SGA trap' if eury else '?'}")
    for root, dl in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- DEEP" if root in {"Eury", "TackA", "DPANN", "MHH"} else ""
        print(f"  {root:16s} {dl:15.1f}  d={dl - b0:9.1f}  {tag}{flag}")
