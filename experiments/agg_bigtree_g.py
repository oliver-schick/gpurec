"""Big-tree clade-grouped DTL_br2 + grouped O {root,2ch,DPANN} + Simpson. The paper model."""
import json, os, math
U="/work/SzollosiU/gergely-szollosi/williams_run/undine"; LN2=math.log(2.0)
ROOTS=["Alti","AMD","Asgard","Cluster2","DPANN","Eury","HalobacThermopl","Kor","MHH",
       "Micra5","MicraDia","TACA","TackA","TAC","UndineClu2"]
EXP={"Eury","MHH"}; c=30000
def stats(om):
    mx=max(om); ex=[2.0**(w-mx) for w in om]; Z=sum(ex); p=[e/Z for e in ex]
    return sum(pe*pe for pe in p), max(p)
rows=[]
for r in ROOTS:
    sc=f"{U}/BSG_c{c}_{r}.rates.txt.json"
    if not os.path.exists(sc): continue
    d=json.load(open(sc)); om=(d.get("origination") or {}).get("omega_log2")
    if not om: continue
    P2,mp=stats(om); dL=d["data_log_likelihood_ln"]
    rows.append((r, dL, dL-LN2*c*P2, mp, 1.0/P2))
print(f"=== BIG TREE clade-grouped DTL_br2 + grouped O + Simpson c={c}  ({len(rows)}/15) ===")
if rows:
    best=max(r[2] for r in rows)
    print(f"{'root':18s} {'data_logL':>13s} {'score':>13s} {'dScore':>9s} {'maxP^O':>7s} {'eff#':>6s}")
    for r,dL,sc_,mp,inv in sorted(rows,key=lambda x:-x[2]):
        tag="[Eury/MHH region]" if r in EXP else ""
        print(f"{r:18s} {dL:13.1f} {sc_:13.1f} {sc_-best:9.1f} {mp:7.3f} {inv:6.1f}  {tag}")
