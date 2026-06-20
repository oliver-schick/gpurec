"""AU test (multiscale bootstrap) per CONFIG, on gpurec per-family logL (perfam/*.json)."""
import json, math
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)
PF=Path("/work/SzollosiU/gergely-szollosi/williams_run/perfam")
ROOTS=["Alti","AMD","Asgard","Cluster2","DPANN","Eury","HaloThermoplas","Kor","TAC","TackA"]
DEEP={"Eury","DPANN","TackA"}; SGA={"Cluster2","AMD","Asgard","Alti"}
CONFIGS=["cmlS_3000","cmlS_10000","cmlS_30000","cml","cmlU","cmlBS_s1.0_c30000"]
dev="cuda" if torch.cuda.is_available() else "cpu"
from torch.distributions import Multinomial
def Phi(x): return 0.5*(1+torch.erf(x/math.sqrt(2)))
def Phi_inv(p): return math.sqrt(2)*torch.erfinv(2*p-1)
def au(L):
    n,R=L.shape; tot=L.sum(0)
    gammas=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]; B=20000
    probs=torch.full((n,),1.0/n,device=dev); BP=torch.zeros(len(gammas),R,device=dev)
    for k,g in enumerate(gammas):
        m=int(round(g*n)); sel=(Multinomial(m,probs).sample((B,))@L).argmax(1)
        BP[k]=torch.bincount(sel,minlength=R).double()/B
    sig=torch.tensor([1/math.sqrt(g) for g in gammas],device=dev); out=[]
    for r in range(R):
        bp=BP[:,r].clamp(1.0/(2*B),1-1.0/(2*B)); z=Phi_inv(1-bp)
        phi=torch.exp(-0.5*z*z)/math.sqrt(2*math.pi); w=(B*phi*phi)/(bp*(1-bp)).clamp_min(1e-12); sw=w.sqrt()
        X=torch.stack([1/sig,sig],1); beta=torch.linalg.lstsq((X*sw.unsqueeze(1)),(z*sw)).solution
        out.append((float(tot[r]),float(BP[5,r]),float(1-Phi(beta[0]-beta[1]))))
    return out
for cfg in CONFIGS:
    maps={}
    for r in ROOTS:
        f=PF/f"{cfg}_{r}.perfam.json"
        if not f.exists(): break
        d=json.load(open(f)); maps[r]=dict(zip(d["names"],d["logL_ln"]))
    if len(maps)<len(ROOTS): print(f"\n### {cfg}: incomplete ({len(maps)}/10) ###"); continue
    common=sorted(set.intersection(*[set(m) for m in maps.values()])); n=len(common)
    L=torch.tensor([[maps[r][f] for r in ROOTS] for f in common],device=dev)
    rows=sorted(zip(ROOTS,au(L)),key=lambda x:-x[1][0]); b0=rows[0][1][0]
    print(f"\n### {cfg}  (n={n}) ###")
    print(f"{'root':16s} {'totalLogL':>11s} {'dTot':>8s} {'BP':>6s} {'AU':>7s}  verdict")
    for nm,(t,bp,a) in rows:
        tag="[deep]" if nm in DEEP else ("[SGA]" if nm in SGA else "")
        v="BEST" if a>=0.95 else ("in95%" if a>=0.05 else "REJECT")
        print(f"{nm:16s} {t:11.1f} {t-b0:8.1f} {bp:6.3f} {a:7.4f}  {v} {tag}")
