"""Load classic ALEobserve ``.ale`` files into gpurec per-family CCP arrays.

Motivation
----------
gpurec normally builds conditional clade probabilities (CCPs) from Newick gene
trees via the C++ amalgamator (``core/cpp/preprocess.cpp``).  A classic ALE
``.ale`` file (written by ALEobserve, see ``ALE/src/ALE.cpp::save_state``)
already stores the CCP distribution over a sample of gene trees, so the original
trees are not needed -- but ``.ale`` is not a native gpurec input.  This module
bridges the gap: it parses a ``.ale`` file and emits the exact per-family dict
that ``preprocess_multiple_families`` produces, so the existing optimizer runs
on ``.ale`` data unchanged.

The ``.ale`` format (text)
--------------------------
Sections, in order (``save_state``):

* ``#constructor_string``  : a representative Newick string (one line)
* ``#observations``        : number of sampled trees (float)
* ``#Bip_counts``          : ``set_id <tab> count`` -- #trees containing clade ``set_id``
* ``#Bip_bls``             : ``set_id <tab> branch_length`` (ignored here)
* ``#Dip_counts``          : ``parent_set_id <tab> g1 <tab> g2 <tab> count``
                             -- #trees where clade ``parent`` splits into ``g1``/``g2``
* ``#last_leafset_id``     : largest set id (int)
* ``#leaf-id``             : ``leaf_name <tab> leaf_id`` (1-based)
* ``#set-id``              : ``set_id <tab>:<tab> leaf_id <tab> leaf_id ...``
* ``#END``

Clades ("leafsets") are subsets of leaves identified by ``set_id``.  The full
leaf set Gamma is left *implicit* (no set id, no counts).

Mapping ``.ale`` -> gpurec CCP
------------------------------
gpurec's amalgamator builds a *rooted* CCP: it creates the full-leaf root clade
Gamma and, for every observed bipartition ``{g, Gamma\g}``, a *root split*
``Gamma -> (g, Gamma\g)``; for every observed clade split it creates an
*internal split* ``g -> (g1, g2)``.  The conditional probability of a split is
``weight / sum_of_sibling_weights`` (== ``Dip / Bip``).  The classic ``.ale``
stores exactly this information:

* ``Dip_counts``  ->  gpurec **internal splits**  ``g -> (g1, g2)``, weight = count
* ``Bip_counts``  ->  gpurec **root splits**      ``Gamma -> (g, Gamma\g)``, weight = count
* Gamma is synthesized here (the ``.ale`` leaves it implicit).

This was verified by hand on ``(a,(b,c))`` and is validated numerically against
the C++ amalgamator on the same tree sample (see
``experiments/validate_ale_parser.py``).

The segment/pointer layout (``split_*_sorted``, ``seg_*``, ``ptr_ge2`` ...) is a
faithful port of ``build_ccp_arrays``; wave scheduling is left to the Python BFS
fallback in :mod:`gpurec.core.scheduling`, which only needs the split arrays
(so ``phased_waves`` is intentionally *not* emitted).

``log_split_probs_sorted`` is in **natural log** (matching the C++ output);
``GeneDataset`` converts it to log2 on load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class AleData:
    """Raw contents of a classic ALEobserve ``.ale`` file."""

    constructor_string: str
    observations: float
    # ALE set-id (clade id) -> frozenset of 1-based ALE leaf ids
    set_leaves: dict[int, frozenset[int]]
    # ALE set-id -> count of trees containing this clade (bipartition side)
    bip_counts: dict[int, float]
    # (parent_set_id, child1_set_id, child2_set_id, count)
    dip_counts: list[tuple[int, int, int, float]]
    leaf_name_to_id: dict[str, int]
    leaf_id_to_name: dict[int, str]
    last_leafset_id: int

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_name_to_id)


def parse_ale_file(path: str | Path) -> AleData:
    """Parse a classic ALEobserve ``.ale`` file into an :class:`AleData`.

    Mirrors ``approx_posterior::load_state`` (``ALE/src/ALE.cpp``): a line that
    contains ``#`` switches the active section; subsequent lines are parsed
    according to that section.
    """
    constructor_string = ""
    observations = 0.0
    bip_counts: dict[int, float] = {}
    dip_counts: list[tuple[int, int, int, float]] = []
    leaf_name_to_id: dict[str, int] = {}
    leaf_id_to_name: dict[int, str] = {}
    set_leaves: dict[int, frozenset[int]] = {}
    last_leafset_id = 0

    section = "#nothing"
    with open(path, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if "#" in line:
                # Section header (ALE checks for '#' anywhere in the line).
                section = line
                continue

            if section == "#constructor_string":
                constructor_string = line
                # one-line section; ALE resets to #nothing after reading it
                section = "#nothing"
            elif section == "#observations":
                observations = float(line)
            elif section == "#Bip_counts":
                tok = line.split()
                bip_counts[int(tok[0])] = float(tok[1])
            elif section == "#Bip_bls":
                pass  # branch lengths -- not needed for CCP
            elif section == "#Dip_counts":
                tok = line.split()
                dip_counts.append(
                    (int(tok[0]), int(tok[1]), int(tok[2]), float(tok[3]))
                )
            elif section == "#last_leafset_id":
                last_leafset_id = int(line)
            elif section == "#leaf-id":
                tok = line.split()
                name, leaf_id = tok[0], int(tok[1])
                leaf_name_to_id[name] = leaf_id
                leaf_id_to_name[leaf_id] = name
            elif section == "#set-id":
                # format: "<set_id> : <leaf_id> <leaf_id> ..."
                left, _, right = line.partition(":")
                set_id = int(left.split()[0])
                leaves = frozenset(int(x) for x in right.split())
                set_leaves[set_id] = leaves

    if not leaf_name_to_id:
        raise ValueError(f"No leaves parsed from .ale file {path!r}")

    return AleData(
        constructor_string=constructor_string,
        observations=observations,
        set_leaves=set_leaves,
        bip_counts=bip_counts,
        dip_counts=dip_counts,
        leaf_name_to_id=leaf_name_to_id,
        leaf_id_to_name=leaf_id_to_name,
        last_leafset_id=last_leafset_id,
    )


# ---------------------------------------------------------------------------
# Gene -> species mapping
# ---------------------------------------------------------------------------


def default_species_of_leaf(leaf_name: str) -> str:
    """Species name = prefix before the first ``_``.

    Matches ``extract_species_name`` in ``core/cpp/preprocess.cpp`` (the rule
    gpurec already uses for Newick gene leaves).
    """
    idx = leaf_name.find("_")
    return leaf_name[:idx] if idx != -1 else leaf_name


# ---------------------------------------------------------------------------
# .ale -> gpurec CCP arrays
# ---------------------------------------------------------------------------


class _CladeRegistry:
    """Assigns contiguous gpurec clade ids to leafsets (frozensets of ALE ids)."""

    def __init__(self) -> None:
        self._id_of: dict[frozenset[int], int] = {}
        self.leaves_of: list[frozenset[int]] = []  # clade id -> leafset

    def get_or_create(self, leafset: frozenset[int]) -> int:
        cid = self._id_of.get(leafset)
        if cid is None:
            cid = len(self.leaves_of)
            self._id_of[leafset] = cid
            self.leaves_of.append(leafset)
        return cid

    def get(self, leafset: frozenset[int]) -> int:
        return self._id_of[leafset]

    def __len__(self) -> int:
        return len(self.leaves_of)


def build_family_from_ale(
    ale: AleData,
    species_name_to_index: dict[str, int],
    *,
    species_of_leaf: Callable[[str], str] = default_species_of_leaf,
    dtype: torch.dtype = torch.float64,
    with_details: bool = False,
) -> dict:
    """Convert parsed :class:`AleData` into a gpurec per-family dict.

    Parameters
    ----------
    ale:
        Parsed ``.ale`` contents.
    species_name_to_index:
        ``{species_name -> column index}`` from the C++ species preprocessing
        (``species_helpers['species_name_to_index']``).
    species_of_leaf:
        Maps a gene-leaf name to its species name. Defaults to the
        prefix-before-first-``_`` rule gpurec uses for Newick input.
    dtype:
        Floating dtype for ``log_split_probs_sorted``.
    with_details:
        If True, also attach ``clade_leaf_names`` (list of frozenset[str]) to the
        ``ccp`` sub-dict for validation/debugging.

    Returns
    -------
    dict with keys ``ccp`` (the CCP arrays), ``root_clade_id``,
    ``leaf_row_index``, ``leaf_col_index`` -- the same shape
    ``preprocess_multiple_families`` returns per family.
    """
    reg = _CladeRegistry()

    # 1. Register leaf clades first (deterministic: sorted by name) so leaf
    #    clades get the low ids, then every .ale set-id leafset, then Gamma.
    leaf_ids_sorted = sorted(ale.leaf_name_to_id.values(),
                             key=lambda lid: ale.leaf_id_to_name[lid])
    for lid in leaf_ids_sorted:
        reg.get_or_create(frozenset((lid,)))

    for set_id in sorted(ale.set_leaves):
        reg.get_or_create(ale.set_leaves[set_id])

    gamma = frozenset(ale.leaf_name_to_id.values())
    root_clade_id = reg.get_or_create(gamma)

    def clade_of_setid(set_id: int) -> int:
        return reg.get(ale.set_leaves[set_id])

    # 2. Build splits: (parent_cid, left_cid, right_cid, weight).
    split_parents: list[int] = []
    split_lefts: list[int] = []
    split_rights: list[int] = []
    split_weights: list[float] = []

    # 2a. Internal splits from Dip_counts.
    for parent_sid, g1_sid, g2_sid, count in ale.dip_counts:
        split_parents.append(clade_of_setid(parent_sid))
        split_lefts.append(clade_of_setid(g1_sid))
        split_rights.append(clade_of_setid(g2_sid))
        split_weights.append(count)

    # 2b. Root splits from Bip_counts: one per *bipartition* {g, Gamma\g}.
    #     Both sides may carry a Bip_counts entry (same count); dedup so each
    #     rooting becomes a single Gamma-split, as the C++ amalgamator does.
    seen_biparts: set[frozenset[frozenset[int]]] = set()
    for set_id, count in ale.bip_counts.items():
        gset = ale.set_leaves[set_id]
        comp = gamma - gset
        if not comp:
            continue  # g == Gamma: not a real bipartition
        bipart_key = frozenset((gset, comp))
        if bipart_key in seen_biparts:
            continue
        seen_biparts.add(bipart_key)
        left_cid = reg.get_or_create(gset)
        right_cid = reg.get_or_create(comp)
        split_parents.append(root_clade_id)
        split_lefts.append(left_cid)
        split_rights.append(right_cid)
        split_weights.append(count)

    C = len(reg)
    ccp = _build_ccp_arrays(
        C=C,
        split_parents=split_parents,
        split_lefts=split_lefts,
        split_rights=split_rights,
        split_weights=split_weights,
        root_clade_id=root_clade_id,
        dtype=dtype,
    )

    # 3. Leaf -> species mapping (leaf_row_index / leaf_col_index), in clade-id
    #    order to match the C++ payload's convention.
    leaf_row_index: list[int] = []
    leaf_col_index: list[int] = []
    for cid, leafset in enumerate(reg.leaves_of):
        if len(leafset) != 1:
            continue
        (lid,) = tuple(leafset)
        leaf_name = ale.leaf_id_to_name[lid]
        species = species_of_leaf(leaf_name)
        if species not in species_name_to_index:
            raise KeyError(
                f"Species {species!r} (from gene leaf {leaf_name!r}) not found "
                f"in the species tree. Available example: "
                f"{sorted(species_name_to_index)[:5]}..."
            )
        leaf_row_index.append(cid)
        leaf_col_index.append(species_name_to_index[species])

    if with_details:
        ccp["clade_leaf_names"] = [
            frozenset(ale.leaf_id_to_name[i] for i in leafset)
            for leafset in reg.leaves_of
        ]

    return {
        "ccp": ccp,
        "root_clade_id": int(root_clade_id),
        "leaf_row_index": torch.tensor(leaf_row_index, dtype=torch.long),
        "leaf_col_index": torch.tensor(leaf_col_index, dtype=torch.long),
    }


def _build_ccp_arrays(
    *,
    C: int,
    split_parents: list[int],
    split_lefts: list[int],
    split_rights: list[int],
    split_weights: list[float],
    root_clade_id: int,
    dtype: torch.dtype,
) -> dict:
    """Port of ``build_ccp_arrays`` (preprocess.cpp): sorted split arrays +
    segment/pointer layout consumed by the wave pipeline."""
    N = len(split_parents)

    split_counts = [0] * C
    sum_weights = [0.0] * C
    for i in range(N):
        split_counts[split_parents[i]] += 1
        sum_weights[split_parents[i]] += split_weights[i]

    # Conditional split probability = weight / sum_of_sibling_weights, in ln.
    log_split_probs = [0.0] * N
    for i in range(N):
        denom = sum_weights[split_parents[i]]
        p = (split_weights[i] / denom) if denom > 0.0 else 0.0
        log_split_probs[i] = math.log(p) if p > 0.0 else float("-inf")

    # parents_sorted: clades by split_count desc, then id asc (matches C++).
    parents_sorted = sorted(range(C), key=lambda c: (-split_counts[c], c))
    parent_rank = [0] * C
    for rank, c in enumerate(parents_sorted):
        parent_rank[c] = rank

    # split_order: by parent rank, then original split index (stable).
    split_order = sorted(range(N), key=lambda i: (parent_rank[split_parents[i]], i))

    split_parents_sorted = [split_parents[i] for i in split_order]
    split_lefts_sorted = [split_lefts[i] for i in split_order]
    split_rights_sorted = [split_rights[i] for i in split_order]
    log_split_probs_sorted = [log_split_probs[i] for i in split_order]
    split_leftrights_sorted = split_lefts_sorted + split_rights_sorted  # [2N]

    seg_counts = [split_counts[parents_sorted[i]] for i in range(C)]

    ptr = [0] * (C + 1)
    for i in range(C):
        ptr[i + 1] = ptr[i] + seg_counts[i]

    num_segs_ge2 = sum(1 for c in seg_counts if c >= 2)
    num_segs_eq1 = sum(1 for c in seg_counts if c == 1)
    num_segs_eq0 = sum(1 for c in seg_counts if c == 0)

    stop_reduce_ptr_idx = num_segs_ge2
    end_rows_ge2 = ptr[stop_reduce_ptr_idx]
    ptr_ge2 = ptr[: stop_reduce_ptr_idx + 1]

    long = torch.long
    return {
        "C": int(C),
        "N_splits": int(N),
        "root_clade_id": int(root_clade_id),
        "split_counts": torch.tensor(split_counts, dtype=long),
        "split_order": torch.tensor(split_order, dtype=long),
        "split_parents_sorted": torch.tensor(split_parents_sorted, dtype=long),
        "split_leftrights_sorted": torch.tensor(split_leftrights_sorted, dtype=long),
        "log_split_probs_sorted": torch.tensor(log_split_probs_sorted, dtype=dtype),
        "parents_sorted": torch.tensor(parents_sorted, dtype=long),
        "seg_parent_ids": torch.tensor(parents_sorted, dtype=long),
        "seg_counts": torch.tensor(seg_counts, dtype=long),
        "ptr": torch.tensor(ptr, dtype=long),
        "ptr_ge2": torch.tensor(ptr_ge2, dtype=long),
        "num_segs_ge2": int(num_segs_ge2),
        "num_segs_eq1": int(num_segs_eq1),
        "num_segs_eq0": int(num_segs_eq0),
        "stop_reduce_ptr_idx": int(stop_reduce_ptr_idx),
        "end_rows_ge2": int(end_rows_ge2),
    }
