"""Self-consistent point-MAP rooting under the SIMPSON anti-concentration barrier.

The MAP was fit minimizing  data_NLL(bits) + c*sum_e p^O_e^2  (Simpson, peak-weighted).
The cross-root score at fixed c is the MAP objective = data_logL - barrier, with the
barrier's simplex normalizer Z(c) constant across roots (same S) so it drops out:

    score_nats(root, c) = data_logL_ln  -  ln2 * c * sum_e (p^O_e)^2

Higher = better. Ranked WITHIN each c column (across-c needs Z(c); not done here).
Also shows maxP^O and 1/Simpson (effective #branches) per root to read the regime.
"""
import json, os, math
CM="/work/SzollosiU/gergely-szollosi/williams_run/cleanml"; LN2=math.log(2.0)
ROOTS=["Alti","AMD","Asgard","Cluster2","DPANN","Eury","HaloThermoplas","Kor","TAC","TackA"]
DEEP={"Eury","DPANN","TackA"}; SGA={"Cluster2","AMD","Asgard","Alti"}
CS=[3000,10000,30000]
def stats(om):
    mx=max(om); ex=[2.0**(w-mx) for w in om]; Z=sum(ex); p=[e/Z for e in ex]
    P2=sum(pe*pe for pe in p); return P2, max(p), 1.0/P2
for c in CS:
    rows=[]
    for r in ROOTS:
        sc=f"{CM}/cmlS_{c}_{r}.rates.txt.json"
        if not os.path.exists(sc): continue
        d=json.load(open(sc)); om=(d.get("origination") or {}).get("omega_log2")
        if not om: continue
        P2,mp,inv=stats(om); dL=d["data_log_likelihood_ln"]
        rows.append((r, dL, dL-LN2*c*P2, mp, inv))
    if not rows:
        print(f"\n=== c={c}: (no results yet) ==="); continue
    print(f"\n=== c={c} SIMPSON  ({len(rows)}/10 roots) ===")
    print(f"{'root':16s} {'data_logL':>11s} {'score':>11s} {'dScore':>8s} {'maxP^O':>7s} {'eff#':>6s}")
    best=max(r[2] for r in rows)
    for r,dL,sc_,mp,inv in sorted(rows,key=lambda x:-x[2]):
        tag="[deep]" if r in DEEP else ("[SGA]" if r in SGA else "")
        print(f"{r:16s} {dL:11.1f} {sc_:11.1f} {sc_-best:8.1f} {mp:7.3f} {inv:6.1f}  {tag}")
