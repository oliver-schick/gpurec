"""AU test (Shimodaira 2002, multiscale bootstrap) on the per-family marginal logL of
the RANDOM-EFFECTS model -- the rooting criterion the paper uses (a confidence SET of
roots, not argmax). Reads rfx_<root>.npz (saved by em_origination.rooting_rfx), each
carrying perfam_logL[F] = log2 P(family | root, fitted w + shared p^O). Stacks them into
L[F, R] and runs the SAME multiscale bootstrap as au_bigtree.py (validated there on
AleRax's per_fam_likelihoods). AU is invariant to the log base (argmax-of-sum is scale-
invariant), so bits vs nats does not matter.

Verdict: AU >= 0.95 BEST, >= 0.05 in the 95% confidence set, else REJECTED.
"""
import json, math
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury", "HalobacThermopl",
         "Kor", "MHH", "Micra5", "MicraDia", "TACA", "TackA", "TAC", "UndineClu2"]
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}
RFX_DIR = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/rooting_rfx")
dev = "cuda" if torch.cuda.is_available() else "cpu"


def Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def Phi_inv(p):
    return math.sqrt(2) * torch.erfinv(2 * p - 1)


def au_test(L, B=20000):
    """L[n,R] per-family logL -> list of (root, totalLogL, BP, AU). Identical math to au_bigtree."""
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
        X = torch.stack([1 / sig, sig], 1)                       # z = d/sig + c*sig
        beta = torch.linalg.lstsq((X * sw.unsqueeze(1)), (z * sw)).solution
        if float(BP[:, r].max()) == 0.0:
            au = 0.0
        elif float(BP[:, r].min()) >= 1.0 - 1e-12:
            au = 1.0
        else:
            au = float(1 - Phi(beta[0] - beta[1]))
        res.append((ROOTS[r], float(tot[r]), float(BP[5, r]), au))
    return res


def main():
    cols = []
    F0 = None
    for r in ROOTS:
        fp = RFX_DIR / f"rfx_{r}.npz"
        if not fp.exists():
            raise SystemExit(f"MISSING {fp}")
        d = np.load(fp, allow_pickle=True)
        if "perfam_logL" not in d:
            raise SystemExit(f"{fp} has no perfam_logL -- re-run rooting_rfx with the updated code")
        pf = d["perfam_logL"].astype(np.float64)
        if F0 is None:
            F0 = pf.shape[0]
        elif pf.shape[0] != F0:
            raise SystemExit(f"family-count mismatch: {r} has {pf.shape[0]} vs {F0} "
                             "(roots loaded different family sets -> cannot align by index)")
        cols.append(pf)
    L = torch.tensor(np.stack(cols, 1)).to(dev)                  # [F, R]
    res = au_test(L)
    res.sort(key=lambda x: -x[1])
    b0 = res[0][1]
    print(f"\n=== AU TEST: RANDOM-EFFECTS model (big tree)  n={F0} fam, {len(ROOTS)} roots, B=20000/scale ===")
    print(f"{'root':16s} {'totalLogL':>14s} {'dTot':>9s} {'BP':>6s} {'AU':>7s}  verdict")
    out = []
    for nm, t, bp, au in res:
        tag = "[deep]" if nm in DEEP else ("[SGA]" if nm in SGA else "")
        v = "BEST" if au >= 0.95 else ("in 95% set" if au >= 0.05 else "REJECTED")
        print(f"{nm:16s} {t:14.1f} {t - b0:9.1f} {bp:6.3f} {au:7.4f}  {v:11s} {tag}")
        out.append(dict(root=nm, total=t, dTot=t - b0, BP=bp, AU=au, verdict=v))
    inset = [o["root"] for o in out if o["AU"] >= 0.05]
    print(f"\n  95% confidence set ({len(inset)}): {', '.join(inset)}")
    Path(RFX_DIR / "au_rfx.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote {RFX_DIR / 'au_rfx.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
