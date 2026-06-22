"""Williams cold-recovery + TV battery: full per-branch DTL + fused-lasso TV + free O,
COLD (global-rate warm-up), all 10 roots, with vs without the Simpson O barrier. Does
the TV-regularized branchwise recover Eury from cold on Williams (where it FAILED on the
big tree)? And does TV auto-group? Rank by data_logL_ln + report eff#rates. Pure python.
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/cleanml"
DEEP = {"Eury", "DPANN", "TackA"}
SGA = {"Cluster2", "AMD", "Asgard", "Alti"}


def eff(rates):
    return len({tuple(round(float(x), 3) for x in r) for r in rates})


for nm in ["tvfree", "tvbar"]:
    rows = []
    for f in glob.glob(f"{D}/WLbasin_{nm}_*.rates.txt.json"):
        m = re.search(rf"WLbasin_{nm}_(\w+)\.rates", f)
        if not m:
            continue
        d = json.load(open(f))
        dl = d.get("data_log_likelihood_ln")
        if dl is not None:
            rows.append((m.group(1), dl, eff(d.get("rates", []))))
    rows.sort(key=lambda x: -x[1])
    print(f"\n=== WLbasin {nm}: cold TV branchwise on Williams  [{len(rows)}/10 roots] ===")
    if not rows:
        print("  (none yet)")
        continue
    b0 = rows[0][1]
    eury_rank = next((i + 1 for i, (r, _, _) in enumerate(rows) if r == "Eury"), None)
    print(f"  Eury rank = #{eury_rank}  ({'RECOVERED' if eury_rank == 1 else 'not #1'})")
    for root, dl, e in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- Eury" if root == "Eury" else ""
        print(f"  {root:16s} {dl:13.1f}  d={dl - b0:8.1f}  eff#={e:4d}  {tag}{flag}")
