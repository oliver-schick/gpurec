import json, os, math
U="/work/SzollosiU/gergely-szollosi/williams_run/undine"; LN2=math.log(2.0)
ROOTS=["Eury","Cluster2","MHH"]; CS=[300,1000,3000,10000]
def stats(om):
    mx=max(om); ex=[2.0**(w-mx) for w in om]; Z=sum(ex); p=[e/Z for e in ex]
    P2=sum(pe*pe for pe in p); return 1.0/P2, max(p)
print(f"{'c':>7s} {'root':10s} {'eff#':>7s} {'maxP^O':>7s} {'data_logL':>13s}  (target eff# ~50-150)")
for c in CS:
    for r in ROOTS:
        f=f"{U}/BSGcal_c{c}_{r}.rates.txt.json"
        if not os.path.exists(f): continue
        d=json.load(open(f)); om=(d.get("origination") or {}).get("omega_log2")
        if not om: continue
        eff,mp=stats(om); dL=d["data_log_likelihood_ln"]
        print(f"{c:>7d} {r:10s} {eff:7.1f} {mp:7.3f} {dL:13.1f}")
# per c, does a deep root (Eury/MHH) beat the SGA (Cluster2) on data total?
print("\nper-c: Eury - Cluster2 data_logL gap (>0 = Eury better):")
for c in CS:
    try:
        e=json.load(open(f"{U}/BSGcal_c{c}_Eury.rates.txt.json"))["data_log_likelihood_ln"]
        cl=json.load(open(f"{U}/BSGcal_c{c}_Cluster2.rates.txt.json"))["data_log_likelihood_ln"]
        print(f"  c={c:>6d}: Eury-Cluster2 = {e-cl:+.1f}")
    except: pass
