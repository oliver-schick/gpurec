"""Basin-finding screen: did a method drop the cold-start (uniform-init) optimizer
into the DEEP (Eury) basin instead of the SGA basin? Per method, prints the 3 probe
roots' data_logL_ln and flags success.

Reference (DTL_br1_O):                deep basin (good)     SGA basin (cold-start trap)
  Eury                                  ~ -1,642,425          ~ -1,737,101
  Cluster2                              ~ -1,648,352          ~ -1,712,542
SUCCESS = Eury reaches the deep basin (> -1.70M) AND Eury beats Cluster2 (deep rooting).
Pure python.
"""
import json, glob, re
D = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DEEP_REF = {"Eury": -1642425, "Cluster2": -1648352, "Micra5": -1648861}
SGA_REF = {"Eury": -1737101, "Cluster2": -1712542}

names = sorted({re.search(r"BASIN_([A-Za-z0-9]+)_", f).group(1)
                for f in glob.glob(f"{D}/BASIN_*_*.rates.txt.json")})
if not names:
    print("(no BASIN outputs yet)")
for nm in names:
    rows = {}
    for f in glob.glob(f"{D}/BASIN_{nm}_*.rates.txt.json"):
        m = re.search(rf"BASIN_{nm}_(\w+)\.rates", f)
        if m:
            d = json.load(open(f))
            dl = d.get("data_log_likelihood_ln")
            if dl is not None:
                rows[m.group(1)] = dl
    eury = rows.get("Eury")
    clus = rows.get("Cluster2")
    verdict = ""
    if eury is not None:
        in_deep = eury > -1.70e6
        beats = (clus is None) or (eury > clus)
        verdict = ("DEEP BASIN + Eury wins  <== SUCCESS" if (in_deep and beats)
                   else ("deep-ish but Eury<Cluster2" if in_deep else "SGA basin (trapped)"))
    print(f"\n=== method {nm}  [{len(rows)} probe roots] -> {verdict} ===")
    for r in ["Eury", "Cluster2", "Micra5"]:
        if r in rows:
            dref = rows[r] - DEEP_REF.get(r, rows[r])
            print(f"  {r:10s} {rows[r]:14.1f}   (deep-ref {DEEP_REF.get(r,0):>11}, "
                  f"d_from_deep={dref:+.0f})")
