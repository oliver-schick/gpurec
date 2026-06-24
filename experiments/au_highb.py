"""High-replicate AU test (Shimodaira 2002 multiscale bootstrap), memory-safe.

Re-runs the per-root AU on a CACHED per-family logL matrix at an arbitrary number of
bootstrap replicates B (default 1,000,000) by CHUNKING over B, so it never builds the
[B, n] resample tensor that the 20k-replicate version held in memory. Same multiscale
weighting / AU regression as experiments/au_bigtree.py (and the same BP->{0,1} guard for
the saturated extremes), plus the EXACT per-scale win counts and a binomial standard
error on each bootstrap proportion -- so "BP = 1.0" can be read as "won every one of B
replicates at every scale", not a 20k-undersampling artifact.

Sources of the per-family logL matrix L [n_families, R_roots]:
  --perfam-dir DIR   : a tree of <model>/Undine_C60_<root>[2_OR]/per_fam_likelihoods.txt
                       (AleRax's cached per-family logL; the two clade models). Picks the
                       model by --model {DTL_br2, DTL_br1_O}.
  --npz FILE         : a cached gpurec matrix: arrays `L` [n,R] (ln) and `roots` [R].
                       (the branch-wise / FULLbasin model, dumped by au_fullbasin.py).

CPU is fine -- the bootstrap is pure resampling of a [~7000, 15] matrix; no forward.

  python experiments/au_highb.py --perfam-dir /tmp/perfam --model DTL_br2   -B 1000000
  python experiments/au_highb.py --perfam-dir /tmp/perfam --model DTL_br1_O -B 1000000
  python experiments/au_highb.py --npz FULLbasin_perfam.npz                 -B 1000000
"""
from __future__ import annotations
import argparse, glob, math, json
from pathlib import Path
import torch

ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury", "HalobacThermopl",
         "Kor", "MHH", "Micra5", "MicraDia", "TACA", "TackA", "TAC", "UndineClu2"]
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}
GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


def Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def Phi_inv(p):
    return math.sqrt(2) * torch.erfinv(2 * p - 1)


def _dirstem(model, r):
    return f"Undine_C60_{r}root" + ("2_OR" if model == "DTL_br1_O" else "")


def load_perfam_dir(perfam_dir, model):
    """Read AleRax per_fam_likelihoods.txt for all roots; intersect families -> L [n,R]."""
    maps = {}
    for r in ROOTS:
        f = Path(perfam_dir) / model / _dirstem(model, r) / "per_fam_likelihoods.txt"
        d = {}
        for ln in open(f):
            p = ln.split()
            if len(p) >= 2:
                try:
                    d[p[0]] = float(p[1])
                except ValueError:
                    pass
        maps[r] = d
    common = sorted(set.intersection(*[set(m) for m in maps.values()]))
    L = torch.tensor([[maps[r][fm] for r in ROOTS] for fm in common], dtype=torch.float64)
    return L, ROOTS


def load_npz(npz):
    import numpy as np
    z = np.load(npz, allow_pickle=True)
    L = torch.tensor(z["L"], dtype=torch.float64)
    roots = [str(x) for x in z["roots"]]
    return L, roots


def bootstrap_BP(L, B, chunk, seed, dev):
    """BP[k, r] = fraction of B replicates at scale gammas[k] for which root r wins,
    plus WINS[k, r] (exact integer counts). Fast resampler: uniform randint with
    replacement (probs are uniform) -> scatter_add counts -> counts @ L -> argmax."""
    n, R = L.shape
    L = L.to(dev)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    WINS = torch.zeros(len(GAMMAS), R, dtype=torch.float64, device=dev)
    for k, gamma in enumerate(GAMMAS):
        m = int(round(gamma * n))
        done = 0
        while done < B:
            c = min(chunk, B - done)
            idx = torch.randint(0, n, (c, m), generator=g, device=dev)        # [c, m]
            counts = torch.zeros(c, n, dtype=torch.float64, device=dev)
            counts.scatter_add_(1, idx, torch.ones(c, m, dtype=torch.float64, device=dev))
            win = (counts @ L).argmax(1)                                      # [c]
            WINS[k] += torch.bincount(win, minlength=R).double()
            done += c
    return WINS / B, WINS


def au_from_BP(BP, B):
    sig = torch.tensor([1 / math.sqrt(gm) for gm in GAMMAS], dtype=torch.float64)
    X = torch.stack([1 / sig, sig], 1)
    out = []
    for r in range(BP.shape[1]):
        col = BP[:, r]
        bp = col.clamp(1.0 / (2 * B), 1 - 1.0 / (2 * B))
        z = Phi_inv(1 - bp)
        phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        w = (B * phi * phi) / (bp * (1 - bp)).clamp_min(1e-12)
        sw = w.sqrt()
        beta = torch.linalg.lstsq((X * sw.unsqueeze(1)), (z * sw)).solution
        if float(col.max()) == 0.0:
            au = 0.0
        elif float(col.min()) >= 1.0 - 1e-12:
            au = 1.0
        else:
            au = float(1 - Phi(beta[0] - beta[1]))
        out.append(au)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perfam-dir")
    ap.add_argument("--model", choices=["DTL_br2", "DTL_br1_O"])
    ap.add_argument("--npz")
    ap.add_argument("-B", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.npz:
        L, roots = load_npz(args.npz); label = Path(args.npz).stem
    else:
        L, roots = load_perfam_dir(args.perfam_dir, args.model); label = args.model
    n, R = L.shape
    tot = L.sum(0)
    BP, WINS = bootstrap_BP(L, args.B, args.chunk, args.seed, dev)
    au = au_from_BP(BP.cpu(), args.B)
    bp1 = BP[GAMMAS.index(1.0)].cpu()                                          # standard scale
    se1 = (bp1 * (1 - bp1) / args.B).sqrt()                                    # binomial SE
    order = sorted(range(R), key=lambda r: -float(tot[r]))
    b0 = float(tot[order[0]])

    print(f"\n=== AU @ B={args.B:,}/scale  model={label}  n={n} families, {R} roots ===")
    print(f"{'root':16s} {'sumLogL':>13s} {'dTot':>9s} {'BP(g=1)':>9s} {'±SE':>8s} "
          f"{'AU':>7s}  {'BP@all scales (min..max)':>24s}")
    res = {}
    for r in order:
        nm = roots[r]
        tag = "[deep]" if nm in DEEP else ("[SGA]" if nm in SGA else "")
        bpmin = float(BP[:, r].min()); bpmax = float(BP[:, r].max())
        v = "BEST" if au[r] >= 0.95 else ("in95%" if au[r] >= 0.05 else "REJ")
        print(f"{nm:16s} {float(tot[r]):13.1f} {float(tot[r]) - b0:9.1f} "
              f"{float(bp1[r]):9.4f} {float(se1[r]):8.5f} {au[r]:7.4f}  "
              f"{bpmin:.4f}..{bpmax:.4f} {v} {tag}")
        res[nm] = dict(sumLogL=float(tot[r]), BP_g1=float(bp1[r]), SE_g1=float(se1[r]),
                       AU=au[r], BP_min=bpmin, BP_max=bpmax,
                       wins_per_scale=[int(x) for x in WINS[:, r].tolist()])
    # exact win counts for the top-2 roots, per scale (the saturation check)
    print("\n  exact win-counts per scale (top 2 roots):")
    for r in order[:2]:
        print(f"   {roots[r]:14s}: " + " ".join(f"{int(WINS[k, r]):>9d}" for k in range(len(GAMMAS)))
              + f"   /{args.B}")
    print("   gammas       : " + " ".join(f"{gm:>9.1f}" for gm in GAMMAS))
    conf = [roots[r] for r in order if au[r] >= 0.05]
    print(f"\n  AU 0.05 confidence set ({len(conf)}): {conf}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            dict(model=label, B=args.B, n=n, gammas=GAMMAS, conf_set=conf, roots=res), indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
