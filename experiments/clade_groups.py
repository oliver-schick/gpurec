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
    """Compute each gpurec node's descendant-leaf-NAME set.

    Returns list[frozenset[str]] of length S. Uses the canonical per-node parent
    array from :func:`gpurec.core.tree_prior.species_parent_index` (``s_P_indexes``
    itself is a 2K internal-node layout, NOT a per-node parent map), and
    ``names`` (per-node labels; leaf labels are the species names).
    """
    from gpurec.core.tree_prior import species_parent_index

    parent_t = species_parent_index(species_helpers)  # [S] long, root -> -1
    parent = parent_t.tolist()
    names = list(species_helpers["names"])

    # internal nodes = values that appear as some node's parent; leaves = rest.
    internal = {int(p) for p in parent if int(p) >= 0}
    leaf_ids = [i for i in range(S) if i not in internal]

    node_leaves: List[set] = [set() for _ in range(S)]
    for l in leaf_ids:
        nm = names[l]
        cur = l
        steps = 0
        while cur >= 0 and steps <= S:
            node_leaves[cur].add(nm)
            cur = int(parent[cur])
            steps += 1
    return [frozenset(s) for s in node_leaves]


def branch_params_from_alerax(species_helpers, tree_path, model_params_path):
    """Exact per-branch (D,L,T) and origination O for each gpurec node.

    Maps every gpurec node's descendant leaf set to the AleRax node (labeled
    tree) and reads that node's exact (D,L,T[,O]) from model_parameters.txt.
    Returns (rate_branch[S,3] float tensor, orig_branch[S] float tensor or None,
    diag). orig_branch is AleRax's per-branch origination (a distribution over
    branches summing to ~1); None if model_parameters has no O column.
    """
    import torch

    S = int(species_helpers["S"])
    leafset_to_label = parse_labeled_newick(tree_path)
    dlt, orig = load_model_parameters(model_params_path)
    # label -> leafset; then gpurec node -> leafset -> label -> rates
    label_to_leafset = {lbl: ls for ls, lbl in leafset_to_label.items()}
    node_leafsets = species_node_leafsets(species_helpers, S)
    # invert: leafset -> label
    rate_branch = torch.zeros(S, 3, dtype=torch.float64)
    orig_branch = torch.zeros(S, dtype=torch.float64) if orig else None
    missing = []
    for s in range(S):
        lbl = leafset_to_label.get(node_leafsets[s])
        if lbl is None or lbl not in dlt:
            missing.append((s, sorted(node_leafsets[s])[:3]))
            continue
        rate_branch[s] = torch.tensor(dlt[lbl], dtype=torch.float64)
        if orig_branch is not None:
            orig_branch[s] = orig.get(lbl, 0.0)
    diag = {"S": S, "n_model_params": len(dlt), "has_origination": bool(orig),
            "unmapped": missing,
            "orig_sum": float(orig_branch.sum()) if orig_branch is not None else None}
    return rate_branch, orig_branch, diag


def omega_group_index_for_species_helpers(species_helpers, tree_path, model_params_path,
                                          round_sig: int = 9):
    """Per-branch ORIGINATION category index [S] from AleRax's O column.

    Groups gpurec nodes by identical AleRax origination value O (e.g. AleRax's
    DTLO: DPANN/Eury/TackA each own O, all others share one). Returns
    (omega_group_index[S] long, n_groups). n_groups==0 / None if no O column.
    """
    import torch

    rate_branch, orig_branch, diag = branch_params_from_alerax(
        species_helpers, tree_path, model_params_path)
    if orig_branch is None:
        return None, 0
    vals = orig_branch.tolist()
    key_to_cat = {}
    gi = torch.empty(len(vals), dtype=torch.long)
    for s, v in enumerate(vals):
        key = round(float(v), round_sig)
        if key not in key_to_cat:
            key_to_cat[key] = len(key_to_cat)
        gi[s] = key_to_cat[key]
    return gi, len(key_to_cat)


_AUTO_LABEL = re.compile(r"^Node_.*_\d+$")

# Paper (Huang et al. 2025, DTL_br2) clade set mapped to the Undine_C60 tree labels:
# each gets its own D,T,L; everything else = a single global baseline. 22 clades
# (B1Sed10-29 absent from this tree) + baseline = 23 DTL classes ~ the 72-param BIC-best.
BIGTREE_DTL_CLADES = frozenset({
    "Halobacteriota", "Thermoplasmatota", "MHH",               # Euryarchaeota
    "Korarchaeota", "TAC", "Asgard",                            # TACK+Asgard
    "Micrarchaeota", "Iainarchaeota", "Altarchaeota", "Undinarchaeota",
    "Aenigmatarchaeota", "PWEA01", "EX4484_52", "Nanohaloarchaeota",
    "SpSt1190", "Parva_related", "UBA10117", "Pacearchaeota",
    "Woesearchaeota", "Nanoarchaeota", "DPANN", "Cluster1",     # DPANN + LCAs
})


def _is_named_clade(label: str) -> bool:
    """A clade-defining label: a non-empty internal name that is not an auto-label."""
    return bool(label) and _AUTO_LABEL.match(label) is None


def tree_clade_group_index(species_helpers, tree_path, S: int | None = None,
                           clade_whitelist=None):
    """Topology-only clade-grouped DTL ``group_index[S]`` from a LABELED species tree.

    No AleRax output needed. Each branch is assigned to its SMALLEST ENCLOSING
    NAMED clade (the paper's per-clade branch-wise DTL_br model); branches above
    all named clades fall in a 'ROOT' class. This is the constrained channel-1
    control (clade-wise, NOT free per-branch). Returns (gi[S] long, labels[G]).
    """
    import torch
    if S is None:
        S = int(species_helpers["S"])
    leafset_to_label = parse_labeled_newick(tree_path)
    if clade_whitelist is not None:                  # paper's explicit DTL_br clade set
        named = [(ls, lbl) for ls, lbl in leafset_to_label.items()
                 if lbl in clade_whitelist]
    else:                                            # all named internal clades (>=2 leaves;
        named = [(ls, lbl) for ls, lbl in leafset_to_label.items()   # excludes leaf accessions)
                 if _is_named_clade(lbl) and len(ls) >= 2]
    named.sort(key=lambda kv: len(kv[0]))            # smallest clade first
    node_leafsets = species_node_leafsets(species_helpers, S)
    lab_of: List[str] = []
    for s in range(S):
        vs = node_leafsets[s]
        chosen = "ROOT"
        for ls, lbl in named:                        # first (smallest) enclosing
            if vs <= ls:
                chosen = lbl
                break
        lab_of.append(chosen)
    labels = sorted(set(lab_of))
    lab2id = {l: i for i, l in enumerate(labels)}
    gi = torch.tensor([lab2id[l] for l in lab_of], dtype=torch.long)
    return gi, labels


# Origination clade classes mirroring AleRax's DTL_br1_O on the big tree: DPANN /
# Eury / TACK+Asgard each own an origination rate; everything else shares a single
# baseline (4 classes). These are ROOT-INDEPENDENT major clades, so the same O
# structure transfers across all candidate roots and CARRIES THE ROOTING SIGNAL.
# (The earlier {root, 2 children, DPANN} structure is the paper's spec for ancestral
# reconstruction at a FIXED best root; for ROOTING it leaves the rooting-relevant
# Eury and TACK+Asgard LCAs lumped in the undifferentiated 'rest' class -> O flattens
# to uniform -> SGA artifact. cf. omega_group_index_for_species_helpers, which reads
# exactly this {DPANN, Eury, TackA, rest} grouping off AleRax's Williams O column.)
BIGTREE_ORIGINATION_CLADES = ("Eury", "TackA", "DPANN")


def tree_origination_group_index(species_helpers, tree_path, S: int | None = None,
                                 clade_labels=BIGTREE_ORIGINATION_CLADES):
    """Topology-only origination ``omega_group_index[S]``: each branch -> the SMALLEST
    enclosing major-clade in ``clade_labels`` (DPANN / Eury / TACK+Asgard by default),
    else a shared 'rest' baseline. Mirrors AleRax's DTL_br1_O origination structure
    (DPANN/Eury/TackA each own O). Returns (gi[S] long, n_groups). Absent labels (e.g.
    a clade-internal rooting that breaks a clade's monophyly) are skipped -> folded
    into 'rest'."""
    import torch
    if S is None:
        S = int(species_helpers["S"])
    leafset_to_label = parse_labeled_newick(tree_path)
    label_to_leafset = {lbl: ls for ls, lbl in leafset_to_label.items()}
    named = [(label_to_leafset[l], l) for l in clade_labels if l in label_to_leafset]
    named.sort(key=lambda kv: len(kv[0]))            # smallest clade first
    node_leafsets = species_node_leafsets(species_helpers, S)
    lab_of: List[str] = []
    for s in range(S):
        vs = node_leafsets[s]
        chosen = "rest"
        for ls, lbl in named:                        # first (smallest) enclosing
            if vs <= ls:
                chosen = lbl
                break
        lab_of.append(chosen)
    labels = sorted(set(lab_of))
    lab2id = {l: i for i, l in enumerate(labels)}
    gi = torch.tensor([lab2id[l] for l in lab_of], dtype=torch.long)
    return gi, len(labels)


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
