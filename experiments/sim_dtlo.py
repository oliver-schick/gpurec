"""Forward DTL+O gene-tree simulator on a fixed species tree, for the S1/S2
identifiability test (gene-rich vs gene-poor LACA, matched present-day counts).

The point of the test: simulate two ground-truth worlds that produce the SAME present
-day gene-count distribution but DIFFERENT ancestral content, so leaf presence/absence
cannot tell them apart -- only the gene-tree TOPOLOGY can:
  S1 (gene-rich LACA): origination concentrated DEEP (root) + HIGH loss -> patchy present
     -day distributions arise by vertical-descent-and-loss (gene trees ~CONGRUENT, sparse).
  S2 (gene-poor LACA): origination SPREAD across clades + HIGH transfer -> patchiness
     arises by HGT (gene trees INCONGRUENT: transferred subtrees attach in the wrong clade).
A fit that uses topology should separate them; one that uses only presence cannot.

This module SIMULATES (gpurec computes likelihoods, it does not simulate). It builds the
actual gene tree (so transfers leave real incongruence), labels leaves by species, and
writes Newick gene trees + a present-day-count summary. Pure-python, runs on the Mac.

Event model (undated-DTL flavour, per species branch e, for a lineage entering e):
  - loss with prob L_e -> lineage dies;
  - else it passes to speciation (-> both children, or a leaf), and ABOVE that we layer
    Poisson(D_e) duplications (each a re-entry of branch e) and Poisson(T_e) transfers
    (each a copy entering a random recipient branch r != ancestors(e)).
Each duplication/transfer/speciation is a bifurcation in the gene tree.
"""
from __future__ import annotations
import argparse, math, sys, random
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MAX_LEAVES = 2000          # reject runaway families (real max is 1429)
MAX_WORK = 60000           # per-family event budget (abort -> reject the family)
sys.setrecursionlimit(200000)


def load_topology(tree_path):
    """Species tree topology from gpurec's C++ preprocessing (Triton-free)."""
    from gpurec.core.preprocess_cpp import _load_extension
    from gpurec.core.tree_prior import species_parent_index
    sp = _load_extension().preprocess_multiple_families(str(tree_path), {})["species"]
    S = int(sp["S"]); names = list(sp["names"])
    par = species_parent_index(sp).tolist()
    # derive children from the parent array (robust to s_C12 layout)
    cmap = [[] for _ in range(S)]
    for s in range(S):
        if par[s] is not None and par[s] >= 0 and par[s] != s:
            cmap[par[s]].append(s)
    children = [tuple(cmap[s]) for s in range(S)]            # () leaf, (c1,c2) internal
    is_leaf = [len(children[s]) == 0 for s in range(S)]
    root = [s for s in range(S) if par[s] < 0][0]
    # ancestors set per branch (to forbid transfer to own ancestor, as in the model)
    anc = [set() for _ in range(S)]
    for s in range(S):
        p = par[s]
        while p is not None and p >= 0 and p != s:
            anc[s].add(p); p = par[p]
    depth = [0] * S
    order = sorted(range(S), key=lambda s: len(anc[s]))
    for s in order:
        depth[s] = 0 if par[s] < 0 else depth[par[s]] + 1
    return dict(S=S, names=names, par=par, children=children, is_leaf=is_leaf,
                root=root, anc=anc, depth=depth)


class _Counter:
    __slots__ = ("n",)
    def __init__(self): self.n = 0


def sim_family(topo, s0, D, L, T, rng):
    """Simulate one family originating at branch s0. Returns (newick_or_None, leaf_species_list)."""
    S = topo["S"]; names = topo["names"]; children = topo["children"]
    is_leaf = topo["is_leaf"]; anc = topo["anc"]; allbr = range(S)
    leaves = []                                              # list of species indices (present-day genes)
    nl = _Counter(); work = _Counter()

    def newick_leaf(sp_idx):
        leaves.append(sp_idx)
        return f"{names[sp_idx]}__g{len(leaves)}"            # unique gene label = species + paralog idx

    def over():
        return nl.n > MAX_LEAVES or work.n > MAX_WORK

    def speciate(e):
        if over():
            return None
        if is_leaf[e]:
            nl.n += 1
            return newick_leaf(e)
        c1, c2 = children[e]
        a = process(c1); b = process(c2)
        if a and b:
            return f"({a},{b})"                              # speciation node
        return a or b                                        # SL: one child survived

    def process(e):
        work.n += 1
        if over():
            return None
        if rng.random() < L[e]:
            return None                                      # loss
        sub = speciate(e)
        if sub is None:
            return None
        # AT MOST one duplication + one transfer per branch entry (Bernoulli) -> bounded fan-out
        if rng.random() < D[e]:
            d = process(e)                                   # duplicate re-enters branch e
            if d:
                sub = f"({sub},{d})"
        if rng.random() < T[e]:
            r = int(rng.integers(0, S))                      # recipient: any non-ancestor branch
            if r != e and r not in anc[e]:
                tr = process(r)
                if tr:
                    sub = f"({sub},{tr})"                    # transfer node (donor continues + copy at r)
        return sub

    nw = process(s0)
    if work.n > MAX_WORK:
        return None, []
    if nw is None or len(leaves) < 4:                        # need >=4 leaves to carry topology (matches real data)
        return None, []
    if nl.n > MAX_LEAVES:
        return None, []
    return nw + ";", leaves


# Scenario presets. Rates are per-branch event intensities (Poisson means for D/T, prob for L).
# S1 and S2 are tuned (below, by --match) to give similar present-day family-size distributions.
def scenario_rates(topo, scenario, scale=1.0):
    S = topo["S"]; root = topo["root"]; depth = np.array(topo["depth"], float)
    D = np.full(S, 0.10); L = np.full(S, 0.0); T = np.full(S, 0.0)
    pO = np.zeros(S)
    if scenario == "S1":            # gene-rich LACA: deep origination + high loss, low transfer
        pO[:] = 1e-6; pO[root] = 1.0                       # ~all families originate at the root
        L[:] = 0.35 * scale; T[:] = 0.04
    elif scenario == "S2":          # gene-poor LACA: spread origination + high transfer, low loss
        pO[:] = 1.0                                         # originate ~uniformly across branches
        pO[root] = 0.2
        L[:] = 0.12; T[:] = 0.28 * scale
    else:
        raise SystemExit(f"unknown scenario {scenario}")
    pO = pO / pO.sum()
    return D, L, T, pO


def simulate(topo, scenario, n_fam, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    D, L, T, pO = scenario_rates(topo, scenario, scale)
    S = topo["S"]; names = topo["names"]
    trees = []; sizes = []; per_species = np.zeros(S)
    tries = 0
    while len(trees) < n_fam and tries < n_fam * 20:
        tries += 1
        s0 = int(rng.choice(S, p=pO))
        nw, leaves = sim_family(topo, s0, D, L, T, rng)
        if nw is None:
            continue
        trees.append(nw); sizes.append(len(leaves))
        for sp in set(leaves):
            per_species[sp] += 1
    return trees, np.array(sizes), per_species


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="species tree Newick")
    ap.add_argument("--scenario", choices=["S1", "S2"], required=True)
    ap.add_argument("--n-fam", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="rate scale (for matching)")
    ap.add_argument("--out", default=None, help="write gene trees (one Newick/line) here")
    args = ap.parse_args()
    topo = load_topology(args.tree)
    print(f"[topo] S={topo['S']} leaves={sum(topo['is_leaf'])} root={topo['root']} "
          f"max_depth={max(topo['depth'])}")
    trees, sizes, per_sp = simulate(topo, args.scenario, args.n_fam, args.seed, args.scale)
    print(f"[{args.scenario}] families={len(trees)}  leaf-count: mean={sizes.mean():.1f} "
          f"median={int(np.median(sizes))} min={sizes.min()} max={sizes.max()}")
    qs = np.percentile(sizes, [10, 25, 50, 75, 90]).astype(int)
    print(f"  leaf-count deciles [10,25,50,75,90] = {list(qs)}")
    print(f"  present-day occupancy: mean species/family = {(sizes>0).sum() and per_sp.sum()/len(trees):.1f}  "
          f"taxa-with-any-gene = {(per_sp>0).sum()}/{sum(topo['is_leaf'])}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(trees) + "\n")
        np.save(args.out + ".sizes.npy", sizes)
        print(f"  wrote {len(trees)} gene trees -> {args.out}")


if __name__ == "__main__":
    main()
