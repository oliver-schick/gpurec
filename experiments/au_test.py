"""AU test (Shimodaira 2002, multiscale bootstrap) on a families x roots per-family
logL matrix. Reads AleRax per_fam_likelihoods.txt for each root. torch (CPU ok)."""
import math
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)
BW = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/Williams_et_al_2017/reconciliation models/branch wise")
ROOTS = ["Alti","AMD","Asgard","Cluster2","DPANN","Eury","HaloThermoplas","Kor","TAC","TackA"]
DEEP={"Eury","DPANN","TackA"}; SGA={"Cluster2","AMD","Asgard","Alti"}
def read(r):
    d={}
    for ln in open(BW/r/"per_fam_likelihoods.txt"):
        p=ln.split()
        if len(p)>=2:
            try: d[p[0]]=float(p[1])
            except ValueError: pass
    return d
maps={r:read(r) for r in ROOTS}
common=sorted(set.intersection(*[set(m) for m in maps.values()]))
n=len(common); R=len(ROOTS)
L=torch.tensor([[maps[r][f] for r in ROOTS] for f in common])   # [n,R]
dev="cuda" if torch.cuda.is_available() else "cpu"; L=L.to(dev)
tot=L.sum(0); best=int(tot.argmax())
gammas=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]; B=20000
probs=torch.full((n,),1.0/n,device=dev)
from torch.distributions import Multinomial
BP=torch.zeros(len(gammas),R,device=dev)
for k,g in enumerate(gammas):
    m=int(round(g*n)); cnt=Multinomial(m,probs).sample((B,))   # [B,n]
    sel=(cnt@L).argmax(1); BP[k]=torch.bincount(sel,minlength=R).double()/B
def Phi(x): return 0.5*(1+torch.erf(x/math.sqrt(2)))
def Phi_inv(p): return math.sqrt(2)*torch.erfinv(2*p-1)
sig=torch.tensor([1/math.sqrt(g) for g in gammas],device=dev)
res=[]
for r in range(R):
    bp=BP[:,r].clamp(1.0/(2*B),1-1.0/(2*B)); z=Phi_inv(1-bp)
    phi=torch.exp(-0.5*z*z)/math.sqrt(2*math.pi)
    w=(B*phi*phi)/(bp*(1-bp)).clamp_min(1e-12); sw=w.sqrt()
    X=torch.stack([1/sig,sig],1)                         # z = d/sig + c*sig
    beta=torch.linalg.lstsq((X*sw.unsqueeze(1)),(z*sw)).solution
    d,c=beta[0],beta[1]; au=float(1-Phi(d-c))
    res.append((ROOTS[r],float(tot[r]),float(BP[gammas.index(1.0),r]),au))
res.sort(key=lambda x:-x[1])
print(f"=== AU TEST (AleRax DTL_br2 per-family logL)  n={n} families, B={B}/scale ===")
print(f"{'root':16s} {'totalLogL':>11s} {'dTot':>8s} {'BP':>6s} {'AU':>7s}  verdict")
b0=res[0][1]
for nm,t,bp,au in res:
    tag="[deep]" if nm in DEEP else ("[SGA]" if nm in SGA else "")
    v="BEST" if au>=0.95 else ("in 95% set" if au>=0.05 else "REJECTED (P<0.05)")
    print(f"{nm:16s} {t:11.1f} {t-b0:8.1f} {bp:6.3f} {au:7.4f}  {v} {tag}")
