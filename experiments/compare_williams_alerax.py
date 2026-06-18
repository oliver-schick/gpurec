#!/usr/bin/env python
"""Compare gpurec per-branch DTL rates to the AleRax branch-wise reference.

Both files give per-branch (D, L, T) rates. gpurec writes "# node D L T";
AleRax model_parameters.txt is "node D L T [O]" (no header). We align on
species-tree LEAF names (unambiguous; internal-node labels differ between the
two tools and need the AleRax label map -- handled separately/optionally).

Per rate axis we report Pearson r and Spearman rho on log10(rate), the
log10 RMSE, and the median ratio gpurec/AleRax.

Usage:
  python experiments/compare_williams_alerax.py \
      --gpurec  williams_Eury_prior.rates.txt \
      --alerax  ".../branch wise/Eury/model_parameters/model_parameters.txt" \
      --tree    ".../rooted_phylogeny/Eury" \
      [--out report.txt]
"""
from __future__ import annotations
import argparse
import math
import re
import sys


def parse_rates(path, skip_header):
    """Return {node_name: (D, L, T)} from a whitespace table 'name D L T [..]'."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[0]
            try:
                D, L, T = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            out[name] = (D, L, T)
    return out


def leaf_names_from_newick(path):
    """Leaf labels = tokens immediately preceding ':' or ',' or ')' that are not
    after ')'. Simple + robust for these trees: a leaf is a name not preceded by ')'."""
    txt = open(path).read()
    leaves = set()
    # tokens: a name is [A-Za-z0-9._-]+ ; a leaf name is one NOT immediately
    # following ')'. Walk and collect names whose preceding non-space char is
    # '(' or ',' (i.e. start of a clade entry), then check it's a leaf (next
    # structural char is ':' ',' or ')', and the name itself wasn't a clade).
    for m in re.finditer(r"[(,]\s*([A-Za-z0-9._\-]+)\s*(?=[:,)])", txt):
        leaves.add(m.group(1))
    return leaves


def log10c(x, floor=1e-12):
    return math.log10(max(x, floor))


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpurec", required=True)
    ap.add_argument("--alerax", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = parse_rates(args.gpurec, skip_header=True)
    a = parse_rates(args.alerax, skip_header=False)
    leaves = leaf_names_from_newick(args.tree)

    common = sorted(n for n in leaves if n in g and n in a)
    lines = []
    def emit(s=""):
        lines.append(s)
        print(s)

    emit("=" * 70)
    emit("gpurec vs AleRax branch-wise rates  (leaf branches)")
    emit("=" * 70)
    emit(f"  gpurec nodes: {len(g)}   alerax nodes: {len(a)}   tree leaves: {len(leaves)}")
    emit(f"  aligned leaf branches: {len(common)}")
    if not common:
        emit("  [!] no overlap -- check node naming")
        return 1

    axes = [("D", 0), ("L", 1), ("T", 2)]
    emit("")
    emit(f"  {'axis':<4} {'Pearson(log)':>13} {'Spearman':>10} {'RMSE(log10)':>12} {'med ratio g/a':>14}")
    for name, idx in axes:
        gx = [log10c(g[n][idx]) for n in common]
        ax = [log10c(a[n][idx]) for n in common]
        r = pearson(gx, ax)
        rho = spearman(gx, ax)
        rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(gx, ax)) / len(gx))
        ratios = sorted(10 ** (x - y) for x, y in zip(gx, ax))
        med = ratios[len(ratios) // 2]
        emit(f"  {name:<4} {r:>13.4f} {rho:>10.4f} {rmse:>12.4f} {med:>14.3g}")

    # a few worst-disagreeing leaves on L (the hard axis)
    emit("")
    emit("  largest |log10 gpurec/alerax| on L (loss):")
    disag = sorted(common, key=lambda n: -abs(log10c(g[n][1]) - log10c(a[n][1])))[:8]
    emit(f"    {'leaf':<12} {'gpurec_L':>12} {'alerax_L':>12}")
    for n in disag:
        emit(f"    {n:<12} {g[n][1]:>12.4g} {a[n][1]:>12.4g}")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n  report -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
