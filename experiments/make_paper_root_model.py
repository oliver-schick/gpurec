#!/usr/bin/env python3
"""Construct a per-root "paper-exact" clade-groups directory for the rooting test.

The paper (4_Ancestral_reconstruction) reconstructed rates ONLY at the Euryarchaeota
root: a clade-grouped DTL + structured-O model (model_parameters.txt over the
Eury-rooted species tree, 24 distinct (D,L,T) classes + 4 O classes incl. the lone
O=0.165 root origination mass). To refit THAT EXACT model at the other 14 candidate
roots (a comparable rooting test) we must express the same clade rate-classes on each
root's tree.

All 15 candidate roots are re-rootings of ONE unrooted topology, so the bipartition
set is shared: every node leaf-set L of root R's tree equals either a descendant set
of the paper (Eury) tree, or its complement (all_leaves - L) -- the same bipartition.
The root node (L = all leaves) maps to the paper ROOT row, so EVERY candidate root
symmetrically gets the root-origination class. Hence the mapping is just
"try L, else its complement"; no reference-taxon canonicalization is needed.

Output is the directory layout run_undine_branchwise.py --clade-groups expects:
  <out>/model_parameters/model_parameters.txt        (<label> D L T O per node)
  <out>/species_trees/starting_species_tree.newick   (R's tree; EVERY internal node
                                                       uniquely (re)labeled so each
                                                       row is referenceable by name)
--clade-groups re-derives the 24 DTL + 4 O classes by leaf-set on R's tree (0
unmapped) and --init-from-alerax seeds the fit at the paper (Eury-basin) rates.

Standalone: imports only the pure-Python parsers from clade_groups (no torch / no
gpurec), so it runs on a login node. Verify on Eury: complement_mapped must be 0
(Eury rooting == paper rooting) and DTL/O class counts must match the paper.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clade_groups import _Node, _tokenize_newick, load_model_parameters  # noqa: E402


def parse_tree(path: str) -> _Node:
    """Parse a Newick file into the root _Node (topology + leaf sets; brlen ignored)."""
    toks = _tokenize_newick(Path(path).read_text().strip())
    pos = 0

    def parse_clade() -> _Node:
        nonlocal pos
        node = _Node()
        if toks[pos] == "(":
            pos += 1
            while True:
                node.children.append(parse_clade())
                if toks[pos] == ",":
                    pos += 1
                    continue
                if toks[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"newick parse error near token {pos}: {toks[pos]!r}")
            if pos < len(toks) and toks[pos] not in (",", ")", ";"):
                node.name = toks[pos].split(":", 1)[0].strip()
                pos += 1
            leaves = set()
            for ch in node.children:
                leaves |= ch.leaves
            node.leaves = frozenset(leaves)
        else:
            node.name = toks[pos].split(":", 1)[0].strip()
            pos += 1
            node.leaves = frozenset([node.name])
        return node

    return parse_clade()


def paper_leafset_rates(paper_tree: str, paper_mp: str):
    """{descendant-leaf-set -> (D,L,T,O)} from the paper Eury tree + model_parameters."""
    root = parse_tree(paper_tree)
    dlt, orig = load_model_parameters(paper_mp)
    ls2name = {}

    def walk(n: _Node):
        ls2name[n.leaves] = n.name
        for c in n.children:
            walk(c)

    walk(root)
    ls2rate, miss = {}, 0
    for ls, name in ls2name.items():
        if name in dlt:
            d, l, t = dlt[name]
            ls2rate[ls] = (d, l, t, orig.get(name, 0.0))
        else:
            miss += 1
    return ls2rate, root.leaves, miss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-tree", required=True, help="paper Eury starting_species_tree.newick")
    ap.add_argument("--paper-mp", required=True, help="paper Eury model_parameters.txt")
    ap.add_argument("--fit-tree", required=True, help="root R's Undine_C60_<R>root tree")
    ap.add_argument("--out-dir", required=True, help="destination clade-groups dir for root R")
    ap.add_argument("--root", default="?")
    a = ap.parse_args()

    ls2rate, paper_all, miss_paper = paper_leafset_rates(a.paper_tree, a.paper_mp)
    if miss_paper:
        print(f"  [warn] {miss_paper} paper-tree nodes had no model_parameters row")

    fit_root = parse_tree(a.fit_tree)
    all_leaves = fit_root.leaves
    if all_leaves != paper_all:
        print(f"  [warn] leaf-set mismatch root={a.root}: "
              f"fit-only={sorted(all_leaves - paper_all)[:5]} "
              f"paper-only={sorted(paper_all - all_leaves)[:5]} "
              f"(|fit|={len(all_leaves)} |paper|={len(paper_all)})")

    rows = []           # (label, (D,L,T,O))
    comp = unmapped = 0
    ctr = [0]

    def assign(n: _Node):
        nonlocal comp, unmapped
        if n.children:                       # relabel every internal node uniquely
            n.name = f"PN{ctr[0]}"
            ctr[0] += 1
        L = n.leaves
        r = ls2rate.get(L)
        if r is None:
            r = ls2rate.get(all_leaves - L)  # same bipartition, complemented side
            if r is not None:
                comp += 1
        if r is None:
            unmapped += 1
            r = (1e-10, 1e-10, 1e-10, 0.0)
        rows.append((n.name, r))
        for c in n.children:
            assign(c)

    assign(fit_root)

    def to_nwk(n: _Node) -> str:
        if not n.children:
            return n.name
        return "(" + ",".join(to_nwk(c) for c in n.children) + ")" + n.name

    out = Path(a.out_dir)
    (out / "model_parameters").mkdir(parents=True, exist_ok=True)
    (out / "species_trees").mkdir(parents=True, exist_ok=True)
    with open(out / "model_parameters" / "model_parameters.txt", "w") as f:
        for name, (d, l, t, o) in rows:
            f.write(f"{name} {d:.10g} {l:.10g} {t:.10g} {o:.10g}\n")
    (out / "species_trees" / "starting_species_tree.newick").write_text(to_nwk(fit_root) + ";\n")

    dtl_classes = {tuple(round(x, 6) for x in r[:3]) for _, r in rows}
    o_classes = {round(r[3], 9) for _, r in rows}
    print(f"  root={a.root}: nodes={len(rows)} DTL_classes={len(dtl_classes)} "
          f"O_classes={len(o_classes)} complement_mapped={comp} unmapped={unmapped} -> {out}")
    if unmapped:
        raise SystemExit(f"  ERROR root={a.root}: {unmapped} unmapped nodes (topology mismatch)")


if __name__ == "__main__":
    main()
