"""Rank big-tree roots by gpurec's fitted data_logL_ln under AleRax's EXACT grouping
seeded at AleRax's rates (AXgrp), per model. Compare to: AleRax's own AU ranking
(auBT -> Eury+MHH) and gpurec's default-init no-barrier fit (BTroot -> SGA).

  ranking stays Eury/MHH  => gpurec's default init just missed AleRax's basin (multimodal)
  ranking descends to SGA => AleRax's Eury is an early-stopping artifact (full ML = SGA)

Pure python (login-node ok). Usage: python3 agg_axgrp.py
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}

for model in ["DTL_br2", "DTL_br1_O"]:
    rows = []
    for f in glob.glob(f"{D}/AXgrp_{model}_*.rates.txt.json"):
        m = re.search(rf"AXgrp_{model}_(\w+)\.rates", f)
        if not m:
            continue
        d = json.load(open(f))
        dl = d.get("data_log_likelihood_ln")
        if dl is not None:
            rows.append((m.group(1), dl))
    rows.sort(key=lambda x: -x[1])
    print(f"\n=== AXgrp {model}: gpurec fit @ AleRax grouping + init-from-alerax  "
          f"[{len(rows)}/15 roots] ===")
    if not rows:
        print("  (none finished yet)")
        continue
    b0 = rows[0][1]
    for root, dl in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- Eury/MHH region" if root in {"Eury", "MHH"} else ""
        print(f"  {root:16s} {dl:15.1f}  d={dl - b0:9.1f}  {tag}{flag}")
