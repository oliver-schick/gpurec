"""AU test (Shimodaira 2002 multiscale bootstrap) over the BRANCHWISE Stage-B model:
read eval_perfam_undine .json files (per-family logL at each root's fitted free-branchwise
DTL with origination pinned to the rfx no-dump p^O), align by family name, run the same
bootstrap as au_bigtree/au_rfx. Tests whether the branchwise model brings Eury into the
root confidence set (the rfx GLOBAL-rate model rejected it; the paper roots Eury via
clade/branchwise rates).

Usage: au_branchwise.py <dir-with-perfam_*.json>   (or pass explicit files)
"""
import sys, json, math, glob
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def Phi_inv(p):
    return math.sqrt(2) * torch.erfinv(2 * p - 1)


def au_test(L, roots, B=20000):
    from torch.distributions import Multinomial
    n, R = L.shape
    tot = L.sum(0)
    gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    probs = torch.full((n,), 1.0 / n, device=L.device)
    BP = torch.zeros(len(gammas), R, device=L.device)
    for k, g in enumerate(gammas):
        m = int(round(g * n))
        sel = (Multinomial(m, probs).sample((B,)) @ L).argmax(1)
        BP[k] = torch.bincount(sel, minlength=R).double() / B
    sig = torch.tensor([1 / math.sqrt(g) for g in gammas], device=L.device)
    res = []
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
        res.append((roots[r], float(tot[r]), float(BP[5, r]), au))
    return res


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(str(Path(arg) / "perfam_*.json"))) if Path(arg).is_dir() else sys.argv[1:]
    if not files:
        raise SystemExit(f"no perfam_*.json in {arg}")
    maps = {}
    for fp in files:
        d = json.load(open(fp))
        maps[d["root"]] = dict(zip(d["names"], d["logL_ln"]))
    roots = list(maps)
    common = sorted(set.intersection(*[set(m) for m in maps.values()]))
    n = len(common)
    print(f"  roots={roots}\n  common families={n}")
    L = torch.tensor([[maps[r][f] for r in roots] for f in common]).to(dev)   # [n,R]
    res = au_test(L, roots)
    res.sort(key=lambda x: -x[1])
    b0 = res[0][1]
    print(f"\n=== AU TEST: BRANCHWISE Stage-B model (big tree)  n={n} fam, {len(roots)} roots, B=20000/scale ===")
    print(f"{'root':16s} {'totalLogL':>14s} {'dTot':>9s} {'BP':>6s} {'AU':>7s}  verdict")
    out = []
    for nm, t, bp, au in res:
        tag = "[deep]" if nm in DEEP else ("[SGA]" if nm in SGA else "")
        v = "BEST" if au >= 0.95 else ("in 95% set" if au >= 0.05 else "REJECTED")
        print(f"{nm:16s} {t:14.1f} {t - b0:9.1f} {bp:6.3f} {au:7.4f}  {v:11s} {tag}")
        out.append(dict(root=nm, total=t, dTot=t - b0, BP=bp, AU=au, verdict=v))
    inset = [o["root"] for o in out if o["AU"] >= 0.05]
    print(f"\n  95% confidence set ({len(inset)}): {', '.join(inset)}")
    print(f"  EURY: {'IN the 95% set' if 'Eury' in inset else 'REJECTED'}")
    outp = Path(arg) / "au_branchwise.json" if Path(arg).is_dir() else Path("au_branchwise.json")
    outp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {outp}")


main()
