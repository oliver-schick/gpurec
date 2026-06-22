"""End-to-end TV-anneal pipeline result: cold global-rate warm-up -> strong->weak
fused-lasso/TV homotopy (complexity annealing) -> free per-branch ML. Reads the FINAL
stage per root (L0 = free per-branch, or the last TV stage) and ranks. Reports
eff#rates (auto-grouping). SUCCESS = Eury recovered COLD (deep cluster top), with
auto-grouped branch-wise rates and no AleRax-init cheat.
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}


def eff_groups(rates):
    return len({tuple(round(float(x), 3) for x in r) for r in rates})


# final stage = lowest-lambda file present per root (prefer L0)
finals = {}
for f in glob.glob(f"{D}/TVANNEAL_*_L*.rates.txt.json"):
    m = re.search(r"TVANNEAL_(\w+)_L(\d+)\.rates", f)
    if not m:
        continue
    root, lam = m.group(1), int(m.group(2))
    if root not in finals or lam < finals[root][0]:
        finals[root] = (lam, f)
rows = []
for root, (lam, f) in finals.items():
    d = json.load(open(f))
    dl = d.get("data_log_likelihood_ln")
    if dl is not None:
        rows.append((root, dl, eff_groups(d.get("rates", [])), lam))
rows.sort(key=lambda x: -x[1])
print(f"\n=== TV-ANNEAL cold-start result (final stage per root)  [{len(rows)}/15] ===")
if not rows:
    print("  (no outputs yet)")
else:
    b0 = rows[0][1]
    e = next(((dl, lam) for r, dl, eff, lam in rows if r == "Eury"), None)
    n_done = sum(1 for r, dl, eff, lam in rows if lam == 0)
    if e is None:
        verdict = "Eury not fitted yet"
    elif e[1] != 0:
        verdict = f"chains still annealing (Eury at L={e[1]}, not final) -> NO verdict yet"
    else:
        verdict = "DEEP (recovered COLD!)" if e[0] > -1.70e6 else "SGA/near-tie"
    print(f"  [{n_done}/15 chains reached L0]  Eury={e[0] if e else None} -> {verdict}")
    for root, dl, eff, lam in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- DEEP" if root in {"Eury", "TackA", "DPANN", "MHH"} else ""
        print(f"  {root:16s} {dl:15.1f}  d={dl - b0:9.1f}  eff#={eff:4d}  (Lfinal={lam}) {tag}{flag}")
