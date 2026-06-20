"""EMPIRICAL-BAYES over the Simpson barrier strength c, with the prior normalizer.

  log Z(root,c) = data_logL_ln  -  ln2*c*Σp²(MAP)  -  log Z_prior(root,c)  +  occam

4-class barrier R(q)=Σ_j q_j²/n_j; prior on the simplex (flat base). The integrand is
sharply peaked (ln2*c≳2000) at q*∝n_j (R_min=1/S, root-INDEPENDENT), so Laplace gives,
with a=ln2*c, K classes, S=Σn_j, D=diag(1/n_j) restricted to {Σv=0}:
  log Z_prior ≈ -a/S + ((K-1)/2)·log(π/a) - ½·log det(BᵀD B)
and the restricted determinant has the closed form  det(BᵀDB) = S/(K·∏n_j), i.e.
  log det(BᵀDB) = log S - log K - Σ_j log n_j      (no linear algebra needed).
EB: c*(root)=argmax_c log Z(root,c); roots ranked by max-over-c evidence.
"""
import json, os, math
CM="/work/SzollosiU/gergely-szollosi/williams_run/cleanml"; LN2=math.log(2.0)
ROOTS=["Alti","AMD","Asgard","Cluster2","DPANN","Eury","HaloThermoplas","Kor","TAC","TackA"]
DEEP={"Eury","DPANN","TackA"}; SGA={"Cluster2","AMD","Asgard","Alti"}
CS=[3000,10000,30000]
def classes(om):
    mx=max(om); ex=[2.0**(w-mx) for w in om]; Z=sum(ex); p=[e/Z for e in ex]
    vals={}
    for pe in p: vals[round(pe,12)]=vals.get(round(pe,12),0)+1
    nj=[n for v,n in vals.items()]
    return nj, sum(pe*pe for pe in p)
def logZprior(nj, c):
    S=sum(nj); K=len(nj); a=LN2*c
    logdet = math.log(S) - math.log(K) - sum(math.log(n) for n in nj)
    return -a/S + 0.5*(K-1)*math.log(math.pi/a) - 0.5*logdet
allrows={}
for c in CS:
    rows=[]
    for r in ROOTS:
        sc=f"{CM}/cmlS_{c}_{r}.rates.txt.json"; ev=f"{CM}/EBevS_{c}_{r}.evidence.json"
        if not (os.path.exists(sc) and os.path.exists(ev)): continue
        d=json.load(open(sc)); e=json.load(open(ev))
        om=(d.get("origination") or {}).get("omega_log2"); occ=e.get("occam_factor")
        if not om or occ is None: continue
        nj,P2=classes(om); dL=d["data_log_likelihood_ln"]; lZp=logZprior(nj,c)
        logZ=dL - LN2*c*P2 - lZp + occ
        rows.append((r, dL, -LN2*c*P2, -lZp, occ, logZ))
        allrows.setdefault(r,{})[c]=logZ
    if not rows:
        print(f"\n=== c={c}: (incomplete) ==="); continue
    print(f"\n=== c={c}  ({len(rows)}/10)   logZ = data + barrier + (-logZprior) + occam ===")
    print(f"{'root':16s} {'data_logL':>11s} {'barrier':>9s} {'-logZpri':>9s} {'occam':>8s} {'logZ':>11s} {'dZ':>8s}")
    best=max(r[5] for r in rows)
    for r,dL,bar,nlz,occ,logZ in sorted(rows,key=lambda x:-x[5]):
        tag="[deep]" if r in DEEP else ("[SGA]" if r in SGA else "")
        print(f"{r:16s} {dL:11.1f} {bar:9.1f} {nlz:9.1f} {occ:8.1f} {logZ:11.1f} {logZ-best:8.1f} {tag}")
print("\n=== EMPIRICAL-BAYES rooting: c*(root)=argmax_c logZ, ranked by max evidence ===")
eb={r:max(cz.items(), key=lambda kv:kv[1]) for r,cz in allrows.items() if len(cz)==len(CS)}
if eb:
    best=max(v for _,v in eb.values())
    for r,(cstar,v) in sorted(eb.items(), key=lambda kv:-kv[1][1]):
        tag="[deep]" if r in DEEP else ("[SGA]" if r in SGA else "")
        print(f"  {r:16s} logZ={v:11.1f}  dZ={v-best:8.1f}  c*={cstar:6d}  {tag}")
    miss=[r for r in ROOTS if r not in eb]
    if miss: print(f"  (missing full c-grid: {miss})")
else:
    print("  (need all 3 c columns for every root)")
