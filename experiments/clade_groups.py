"""Build a gpurec clade-grouped (rate-category) ``group_index`` from AleRax's
"branch wise" model_parameters output.

AleRax's "branch wise" model is NOT 119 free per-branch rates: it is a
clade-grouped model with ~17 distinct (D,L,T) rate categories (defined by the
``all_others_sp_O`` named-clade parametrization). The model_parameters.txt
output tags every one of the 119 species-tree nodes with its category's rates,
so the grouping is recoverable by grouping nodes with identical (D,L,T) tuples.

To map AleRax's nodes onto gpurec's node indexing we use the DESCENDANT LEAF SET
as a topology-canonical key (independent of internal-node naming conventions):
AleRax labels every internal node in ``starting_species_tree.newick`` (leaves,
named clades like ``Eury``/``DPANN``, and auto-labels like ``Node_X_Y_0``); we
parse that tree to get each label's leaf set, attach the model_parameters rate
tuple, group identical tuples into categories, and key the result by leaf set.
gpurec then looks up each of its own nodes' leaf sets to get ``group_index[S]``.

Pure-Python / no gpurec import on the AleRax side, so it is Mac-testable. The
gpurec-side mapping (:func:`group_index_for_species_helpers`) needs only
``species_helpers`` (parent array + names), so it runs wherever gpurec runs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple


# ── labeled-Newick parser (extracts internal labels + leaf sets) ──────────────
class _Node:
    __slots__ = ("name", "children", "leaves")

    def __init__(self):
        self.name: str = ""
        self.children: List["_Node"] = []
        self.leaves: FrozenSet[str] = frozenset()


def _tokenize_newick(s: str) -> List[str]:
    # tokens: ( ) , ; and "label[:brlen]" chunks
    return re.findall(r"[(),;]|[^(),;]+", s.strip())


def parse_labeled_newick(path: str | Path) -> Dict[FrozenSet[str], str]:
    """Parse a labeled Newick; return {descendant-leaf-set -> node label}.

    Every node (leaf, named clade, or auto-label) contributes one entry. The
    leaf set is the canonical key; the label is informational. Branch lengths are
    ignored. Assumes uniquely-resolved leaf sets (a proper tree).
    """
    text = Path(path).read_text().strip()
    toks = _tokenize_newick(text)
    pos = 0

    def parse_clade() -> _Node:
        nonlocal pos
        node = _Node()
        if toks[pos] == "(":
            pos += 1  # consume '('
            while True:
                node.children.append(parse_clade())
                if toks[pos] == ",":
                    pos += 1
                    continue
                if toks[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"newick parse error near token {pos}: {toks[pos]!r}")
            # optional internal label[:brlen]
            label = ""
            if pos < len(toks) and toks[pos] not in (",", ")", ";"):
                label = toks[pos].split(":", 1)[0].strip()
                pos += 1
            node.name = label
            leaves = set()
            for ch in node.children:
                leaves |= ch.leaves
            node.leaves = frozenset(leaves)
        else:
            # leaf: "Name[:brlen]"
            chunk = toks[pos]
            pos += 1
            name = chunk.split(":", 1)[0].strip()
            node.name = name
            node.leaves = frozenset([name])
        return node

    root = parse_clade()
    if pos < len(toks) and toks[pos] == ";":
        pos += 1

    out: Dict[FrozenSet[str], str] = {}

    def walk(n: _Node):
        out[n.leaves] = n.name
        for ch in n.children:
            walk(ch)

    walk(root)
    return out


# ── AleRax model_parameters -> categories ─────────────────────────────────────
def load_model_parameters(path: str | Path) -> Tuple[Dict[str, Tuple[float, float, float]],
                                                      Dict[str, float]]:
    """Parse ``model_parameters.txt`` -> ({name -> (D,L,T)}, {name -> O}).

    Lines are ``<name> D L T [O]``. The optional 5th column is origination.
    """
    dlt: Dict[str, Tuple[float, float, float]] = {}
    orig: Dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        tok = line.split()
        if len(tok) < 4:
            continue
        name = tok[0]
        d, l, t = float(tok[1]), float(tok[2]), float(tok[3])
        dlt[name] = (d, l, t)
        if len(tok) >= 5:
            orig[name] = float(tok[4])
    return dlt, orig


def _categorize(dlt: Dict[str, Tuple[float, float, float]],
                round_sig: int = 6) -> Tuple[Dict[str, int], List[Tuple[float, float, float]]]:
    """Group names by identical (rounded) (D,L,T) tuple -> ({name -> cat}, [cat_rates])."""
    key_to_cat: Dict[Tuple[float, float, float], int] = {}
    cat_rates: List[Tuple[float, float, float]] = []
    name_to_cat: Dict[str, int] = {}
    for name, rate in dlt.items():
        key = tuple(round(x, round_sig) for x in rate)
        if key not in key_to_cat:
            key_to_cat[key] = len(cat_rates)
            cat_rates.append(rate)
        name_to_cat[name] = key_to_cat[key]
    return name_to_cat, cat_rates


def build_leafset_categories(tree_path: str | Path, model_params_path: str | Path,
                             round_sig: int = 6):
    """Return (leafset -> category id, cat_rates[G][3], cat_orig[G] or None, diag).

    Combines the labeled tree (leaf sets per node) with the model_parameters
    rate tuples (categories) keyed by leaf set.
    """
    leafset_to_label = parse_labeled_newick(tree_path)
    label_to_leafset = {lbl: ls for ls, lbl in leafset_to_label.items()}
    dlt, orig = load_model_parameters(model_params_path)
    name_to_cat, cat_rates = _categorize(dlt, round_sig)

    # representative origination per category (mean over members, if present)
    cat_orig: List[float] = [0.0] * len(cat_rates)
    if orig:
        sums = [0.0] * len(cat_rates)
        cnts = [0] * len(cat_rates)
        for name, o in orig.items():
            c = name_to_cat[name]
            sums[c] += o
            cnts[c] += 1
        cat_orig = [sums[c] / cnts[c] if cnts[c] else 0.0 for c in range(len(cat_rates))]

    leafset_to_cat: Dict[FrozenSet[str], int] = {}
    matched, unmatched = 0, []
    for name, cat in name_to_cat.items():
        ls = label_to_leafset.get(name)
        if ls is None:
            unmatched.append(name)
            continue
        leafset_to_cat[ls] = cat
        matched += 1

    diag = {
        "n_tree_nodes": len(leafset_to_label),
        "n_model_params": len(dlt),
        "n_categories": len(cat_rates),
        "matched": matched,
        "unmatched_names": unmatched,
        "has_origination": bool(orig),
    }
    return leafset_to_cat, cat_rates, (cat_orig if orig else None), diag


# ── gpurec-side mapping (needs species_helpers) ───────────────────────────────
def species_node_leafsets(species_helpers, S: int):
    """Compute each gpurec node's descendant-leaf-NAME set via the parent array.

    Returns list[frozenset[str]] of length S. Uses ``s_P_indexes`` (parent of
    each node; root maps to >=S or itself) and ``names`` (per-node labels; leaf
    labels are the species names).
    """
    import torch  # local import so the AleRax side stays import-light

    parent = species_helpers["s_P_indexes"]
    parent = parent.tolist() if hasattr(parent, "tolist") else list(parent)
    names = list(species_helpers["names"])

    # leaf mask: nodes that never appear as a parent value (< S)
    internal = {int(p) for p in parent if 0 <= int(p) < S}
    leaf_ids = [i for i in range(S) if i not in internal]

    node_leaves: List[set] = [set() for _ in range(S)]
    for l in leaf_ids:
        nm = names[l]
        cur = l
        seen = set()
        while True:
            node_leaves[cur].add(nm)
            if cur in seen:
                break
            seen.add(cur)
            p = int(parent[cur])
            if p < 0 or p >= S or p == cur:
                break
            cur = p
    return [frozenset(s) for s in node_leaves]


def group_index_for_species_helpers(species_helpers, tree_path, model_params_path,
                                     round_sig: int = 6):
    """Build ``group_index[S]`` (long) + category rate table for a gpurec run.

    Returns (group_index_tensor[S], cat_rates[G][3], cat_orig[G] or None, diag).
    Raises if any gpurec node's leaf set is not found in the AleRax categories
    (would indicate a tree/topology mismatch).
    """
    import torch

    S = int(species_helpers["S"])
    leafset_to_cat, cat_rates, cat_orig, diag = build_leafset_categories(
        tree_path, model_params_path, round_sig
    )
    node_leafsets = species_node_leafsets(species_helpers, S)

    gi = torch.empty(S, dtype=torch.long)
    missing = []
    for s in range(S):
        cat = leafset_to_cat.get(node_leafsets[s])
        if cat is None:
            missing.append((s, sorted(node_leafsets[s])[:4]))
            gi[s] = -1
        else:
            gi[s] = cat
    diag["gpurec_S"] = S
    diag["gpurec_unmapped"] = missing
    return gi, cat_rates, cat_orig, diag
