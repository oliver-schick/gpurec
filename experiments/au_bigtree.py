"""AU test (Shimodaira 2002, multiscale bootstrap) on the BIG TREE (This_study/Undine
C60), on AleRax's per-family logL for both model classes -- 'do what they did':
  DTL_br2   = BIC-best, 72-param PURE rich-clade DTL (no origination column).
  DTL_br1_O = DTL + {DPANN,Eury,TackA,rest} origination.
15 candidate roots, AU per root (the paper's rooting criterion, not argmax). Also
dumps the DTL_br1_O O-column distinct values = the origination class structure.
torch CPU ok, but run on a largegpu node (login/V100 fail the GLIBC torch import).
"""
import math
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)

BASE = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/alerax_ref/"
            "3_Reconciliation/This_study/5_reconcilation models")
ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury", "HalobacThermopl",
         "Kor", "MHH", "Micra5", "MicraDia", "TACA", "TackA", "TAC", "UndineClu2"]
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def dirstem(model, r):
    return f"Undine_C60_{r}root" + ("2_OR" if model == "DTL_br1_O" else "")


def read(model, r):
    f = BASE / model / dirstem(model, r) / "per_fam_likelihoods.txt"
    d = {}
    for ln in open(f):
        p = ln.split()
        if len(p) >= 2:
            try:
                d[p[0]] = float(p[1])
            except ValueError:
                pass
    return d


def Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def Phi_inv(p):
    return math.sqrt(2) * torch.erfinv(2 * p - 1)


def au_for(model):
    maps = {}
    for r in ROOTS:
        try:
            maps[r] = read(model, r)
        except FileNotFoundError:
            print(f"  MISSING {model} {r}")
            return
    common = sorted(set.intersection(*[set(m) for m in maps.values()]))
    n, R = len(common), len(ROOTS)
    L = torch.tensor([[maps[r][f] for r in ROOTS] for f in common]).to(dev)   # [n,R]
    tot = L.sum(0)
    gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    B = 20000
    probs = torch.full((n,), 1.0 / n, device=dev)
    from torch.distributions import Multinomial
    BP = torch.zeros(len(gammas), R, device=dev)
    for k, g in enumerate(gammas):
        m = int(round(g * n))
        sel = (Multinomial(m, probs).sample((B,)) @ L).argmax(1)
        BP[k] = torch.bincount(sel, minlength=R).double() / B
    sig = torch.tensor([1 / math.sqrt(g) for g in gammas], device=dev)
    res = []
    for r in range(R):
        bp = BP[:, r].clamp(1.0 / (2 * B), 1 - 1.0 / (2 * B))
        z = Phi_inv(1 - bp)
        phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        w = (B * phi * phi) / (bp * (1 - bp)).clamp_min(1e-12)
        sw = w.sqrt()
        X = torch.stack([1 / sig, sig], 1)                       # z = d/sig + c*sig
        beta = torch.linalg.lstsq((X * sw.unsqueeze(1)), (z * sw)).solution
        # Guard the saturated extremes: the multiscale regression is degenerate when BP
        # does not vary across scales (z constant -> beta meaningless, ~0.63). BP=1 at
        # every scale = dominant tree -> AU 1; BP=0 everywhere -> AU 0.
        bpmin = float(BP[:, r].min())
        if float(BP[:, r].max()) == 0.0:
            au = 0.0
        elif bpmin >= 1.0 - 1e-12:
            au = 1.0
        else:
            au = float(1 - Phi(beta[0] - beta[1]))
        res.append((ROOTS[r], float(tot[r]), float(BP[5, r]), au))
    res.sort(key=lambda x: -x[1])
    b0 = res[0][1]
    print(f"\n=== AU TEST: AleRax {model} (big tree)  n={n} fam, 15 roots, B={B}/scale ===")
    print(f"{'root':16s} {'totalLogL':>13s} {'dTot':>9s} {'BP':>6s} {'AU':>7s}  verdict")
    for nm, t, bp, au in res:
        tag = "[deep]" if nm in DEEP else ("[SGA]" if nm in SGA else "")
        v = "BEST" if au >= 0.95 else ("in 95% set" if au >= 0.05 else "REJECTED")
        print(f"{nm:16s} {t:13.1f} {t - b0:9.1f} {bp:6.3f} {au:7.4f}  {v} {tag}")


def dump_o_structure():
    mp = (BASE / "DTL_br1_O" / dirstem("DTL_br1_O", "Eury")
          / "model_parameters" / "model_parameters.txt")
    ovals = {}
    for ln in open(mp):
        p = ln.split()
        if len(p) >= 5:
            try:
                o = round(float(p[4]), 6)
                ovals[o] = ovals.get(o, 0) + 1
            except ValueError:
                pass
    print(f"\n=== DTL_br1_O O-column distinct values (Euryroot) = origination classes ===")
    for o, c in sorted(ovals.items(), key=lambda x: -x[1]):
        print(f"   O={o:.6f}  {c} branches")


if __name__ == "__main__":
    for model in ["DTL_br2", "DTL_br1_O"]:
        au_for(model)
    dump_o_structure()
