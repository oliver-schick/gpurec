"""Both-het (Stage B FULL) rooting: per root, the families are partitioned across rate
components, each with its OWN per-family logL (perfam_<j>.json from eval). Merge them ->
the both-het per-family logL for that root (each family appears in exactly one component).
Then point-rank the 15 roots by total logL AND run the multiscale-bootstrap AU.

Usage: stage_b_concat_au.py <parent-dir>   (expects <parent-dir>/<root>/perfam_*.json)
"""
import sys, json, glob, math
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)
ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury", "HalobacThermopl",
         "Kor", "MHH", "Micra5", "MicraDia", "TACA", "TackA", "TAC", "UndineClu2"]
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def Phi_inv(p):
    return math.sqrt(2) * torch.erfinv(2 * p - 1)


def au_test(L, B=20000):
    from torch.distributions import Multinomial
    n, R = L.shape
    gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    probs = torch.full((n,), 1.0 / n, device=L.device)
    BP = torch.zeros(len(gammas), R, device=L.device)
    for k, g in enumerate(gammas):
        m = int(round(g * n))
        sel = (Multinomial(m, probs).sample((B,)) @ L).argmax(1)
        BP[k] = torch.bincount(sel, minlength=R).double() / B
    sig = torch.tensor([1 / math.sqrt(g) for g in gammas], device=L.device)
    aus = []
    for r in range(R):
        bp = BP[:, r].clamp(1.0 / (2 * B), 1 - 1.0 / (2 * B))
        z = Phi_inv(1 - bp)
        phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        w = (B * phi * phi) / (bp * (1 - bp)).clamp_min(1e-12)
        sw = w.sqrt()
        X = torch.stack([1 / sig, sig], 1)
        beta = torch.linalg.lstsq((X * sw.unsqueeze(1)), (z * sw)).solution
        if float(BP[:, r].max()) == 0.0:
            au = 0.0
        elif float(BP[:, r].min()) >= 1.0 - 1e-12:
            au = 1.0
        else:
            au = float(1 - Phi(beta[0] - beta[1]))
        aus.append(au)
    return aus, BP[5].tolist()


def main():
    parent = Path(sys.argv[1])
    maps = {}
    for r in ROOTS:
        files = sorted(glob.glob(str(parent / r / "perfam_*.json")))
        if not files:
            print(f"  [skip] {r}: no perfam_*.json")
            continue
        m = {}
        for fp in files:
            d = json.load(open(fp))
            m.update(dict(zip(d["names"], d["logL_ln"])))   # each family in one component
        maps[r] = m
    roots = [r for r in ROOTS if r in maps]
    common = sorted(set.intersection(*[set(maps[r]) for r in roots]))
    n = len(common)
    print(f"  roots with data: {len(roots)}/15   common families: {n}")
    L = torch.tensor([[maps[r][f] for r in roots] for f in common]).to(dev)   # [n,R]
    tot = L.sum(0).tolist()
    aus, bp = au_test(L)
    rows = sorted(zip(roots, tot, bp, aus), key=lambda x: -x[1])
    b0 = rows[0][1]
    print(f"\n=== BOTH-HET (Stage B FULL) ROOTING  n={n} fam, {len(roots)} roots ===")
    print(f"{'rank':>4s} {'root':16s} {'totalLogL':>14s} {'dTot':>9s} {'BP':>6s} {'AU':>7s}  verdict")
    inset = []
    for i, (nm, t, b, au) in enumerate(rows, 1):
        tag = "[deep]" if nm in DEEP else ("[SGA]" if nm in SGA else "")
        v = "BEST" if au >= 0.95 else ("in 95% set" if au >= 0.05 else "REJECTED")
        if au >= 0.05:
            inset.append(nm)
        print(f"{i:4d} {nm:16s} {t:14.1f} {t - b0:9.1f} {b:6.3f} {au:7.4f}  {v:11s} {tag}")
    print(f"\n  95% confidence set: {', '.join(inset)}")
    print(f"  EURY: {'IN set' if 'Eury' in inset else 'REJECTED'} (rank {[r[0] for r in rows].index('Eury')+1 if 'Eury' in [r[0] for r in rows] else '?'})")
    json.dump([{"root": nm, "total": t, "BP": b, "AU": au} for nm, t, b, au in rows],
              open(parent / "bothhet_rooting.json", "w"), indent=2)
    print(f"  wrote {parent / 'bothhet_rooting.json'}")


main()
