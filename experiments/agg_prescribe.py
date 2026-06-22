"""The derived basin-finding PRESCRIPTION on the big tree: full per-branch DTL + TV +
free O, global-rate DTL warm-up + STRUCTURED-O seed (omega=-depth*scale), multi-start
over scales, pick best logL per root. Does it reach the deep/Eury basin COLD?

Reference: deep/Eury basin Eury ~ -1,642,425 ; near-tie ~ -1,664,000 ; SGA ~ -1,712,000.
SUCCESS = Eury reaches the deep basin (> -1.655M) AND ranks #1 (deep cluster on top).
Pure python.
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}


def eff(rates):
    return len({tuple(round(float(x), 3) for x in r) for r in rates})


# gather all (root, scale) results
by = {}
for f in glob.glob(f"{D}/PRESCRIBE_s*_*.rates.txt.json"):
    m = re.search(r"PRESCRIBE_s([0-9.]+)_(\w+)\.rates", f)
    if not m:
        continue
    scale, root = m.group(1), m.group(2)
    d = json.load(open(f))
    dl = d.get("data_log_likelihood_ln")
    if dl is not None:
        by.setdefault(root, {})[scale] = (dl, eff(d.get("rates", [])))

# per-Eury: show the structured-O seed effect across scales
if "Eury" in by:
    print("=== Eury vs structured-O-seed scale (does the seed reach the deep basin?) ===")
    for s in sorted(by["Eury"], key=float):
        dl, e = by["Eury"][s]
        basin = "DEEP" if dl > -1.655e6 else ("near-tie" if dl > -1.69e6 else "SGA")
        print(f"  scale={s:>5}: logL={dl:13.1f}  eff#={e:4d}  [{basin}]")

# multi-start: best logL per root, then rank
print("\n=== multi-start (best logL over seeds) per root -> rooting ===")
best = {r: max(v.values()) for r, v in by.items()}  # (logL, eff)
rows = sorted(best.items(), key=lambda kv: -kv[1][0])
if rows:
    b0 = rows[0][1][0]
    er = next((i + 1 for i, (r, _) in enumerate(rows) if r == "Eury"), None)
    print(f"  Eury rank = #{er}  ({'RECOVERED COLD' if er == 1 else 'not #1'})")
    for root, (dl, e) in rows:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        flag = "  <-- Eury" if root == "Eury" else ""
        print(f"  {root:14s} {dl:13.1f}  d={dl - b0:8.1f}  eff#={e:4d}  {tag}{flag}")
