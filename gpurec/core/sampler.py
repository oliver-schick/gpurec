"""Stochastic reconciliation sampling (CPU, Triton-free).

Given gpurec's forward DP tables (Pi[clade,species] and the same rates/E/Pibar
the forward used), draw reconciliation scenarios by stochastic backtracking --
the sampling analogue of the undated-DTL fixed point. This replicates AleRax's
``MultiModel::backtrace`` (UndatedDTLMultiModel) exactly, resampling at each
(clade, species) cell an event proportional to its contribution to Pi, then
recursing.

Event decomposition (must match ``core/terms.py`` compute_DTS / compute_DTS_L,
all in log2/bits); at clade ``c`` on species branch ``s`` with species children
``s1, s2``, for each clade split ``c -> (L, R)`` weighted by the split prob:

    split terms (clade is split):
      D            : pD[s] * Pi[L,s] * Pi[R,s]                  -> D, recurse L@s, R@s
      T (R moves)  : Pi[L,s] * Pibar[R,s]                       -> T, L@s; sample recipient for R
      T (L moves)  : Pi[R,s] * Pibar[L,s]                       -> T, R@s; sample recipient for L
      S (orient 1) : pS[s] * Pi[L,s1] * Pi[R,s2]                -> S, recurse L@s1, R@s2
      S (orient 2) : pS[s] * Pi[L,s2] * Pi[R,s1]                -> S, recurse L@s2, R@s1
    non-split terms (clade stays whole):
      DL           : 2 * pD[s] * E[s] * Pi[c,s]                 -> self-loop: resample c@s (unrecorded)
      TL-lost      : Pi[c,s] * Ebar[s]                          -> self-loop: resample c@s (unrecorded)
      TL-move      : Pibar[c,s] * E[s]                          -> TL: c -> recipient r (recorded)
      SL (orient1) : pS[s] * E[s2] * Pi[c,s1]                   -> SL: c -> s1 (recorded)
      SL (orient2) : pS[s] * E[s1] * Pi[c,s2]                   -> SL: c -> s2 (recorded)
      leaf         : pS[s] * [c maps to s]                      -> leaf terminal

Sampling at a cell is a single categorical draw over all the above sub-terms
(in linear space, max-rescaled), so the DL / TL-lost self-loops carry their
correct probability and resampling realises the undated geometric series.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_LOG2E = 1.0 / math.log(2.0)
NEG_INF = float("-inf")

# event type codes (match the recorded events; DL / TL-lost are never recorded)
EV_S = "S"     # speciation
EV_D = "D"     # duplication
EV_T = "T"     # transfer (source stays, a copy moves to recipient)
EV_SL = "SL"   # speciation + loss (clade follows one child, sibling lost)
EV_TL = "TL"   # transfer + loss (source lost, clade follows the transferred copy)
EV_L = "L"     # loss (bookkeeping only; emitted for the lost sibling/branch)
EV_LEAF = "leaf"
EV_O = "O"     # origination (root of the gene family)


@dataclass
class Event:
    """One reconciliation event in a sampled scenario."""
    type: str
    species: int                       # species branch the event occurs on
    gene: int                          # reconciled gene-tree node index
    dest_species: int = -1             # recipient species for T / TL
    left_gene: int = -1
    right_gene: int = -1


@dataclass
class Scenario:
    """One sampled reconciliation: the event list + the gene nodes created."""
    events: List[Event] = field(default_factory=list)
    n_gene_nodes: int = 0
    # species branches the gene family occupies (lineage passes through) --
    # the union of every species touched by a surviving lineage. Used for the
    # family x branch presence table.
    occupied: set = field(default_factory=set)

    def _new_gene(self) -> int:
        i = self.n_gene_nodes
        self.n_gene_nodes += 1
        return i


@dataclass
class FamilyForward:
    """Everything the backtrack needs for one family (numpy, CPU)."""
    Pi: np.ndarray              # [C, S] log2
    Pibar: np.ndarray           # [C, S] log2
    E: np.ndarray               # [S] log2 (extinction)
    Ebar: np.ndarray            # [S] log2 (Pi-context transfer-to-extinct-recipient term)
    log_pS: np.ndarray          # [S] log2
    log_pD: np.ndarray          # [S] log2
    transfer_mat: np.ndarray    # [S, S] LINEAR transfer_mat[d, r]
    log_pO: np.ndarray          # [S] log2 origination distribution (sum exp2 = 1)
    sp_child1: np.ndarray       # [S] left species child, or S (sentinel) if leaf
    sp_child2: np.ndarray       # [S] right species child, or S if leaf
    clade_leaf_species: Dict[int, int]   # leaf clade id -> its leaf species branch
    clade_leaf_label: Dict[int, str]     # leaf clade id -> gene leaf label (optional)
    splits_of: Dict[int, List[Tuple[int, int, float]]]  # clade -> [(L, R, log2_split_prob)]
    root_clade_id: int
    S: int


def _categorical_log2(logw: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an index from unnormalized LOG2 weights (max-rescaled)."""
    m = logw.max()
    if not math.isfinite(m):
        # all -inf: shouldn't happen for a valid cell; fall back uniform
        return int(rng.integers(len(logw)))
    w = np.exp2(logw - m)
    tot = w.sum()
    r = rng.random() * tot
    acc = 0.0
    for i in range(len(w)):
        acc += w[i]
        if r <= acc:
            return i
    return len(w) - 1


def _sample_recipient(fwd: FamilyForward, donor: int, clade_at_recipient: int,
                      rng: np.random.Generator) -> int:
    """Sample a transfer recipient r ~ transfer_mat[donor, r] * Pi[clade, r]."""
    T = fwd.transfer_mat[donor]                      # [S] linear
    pir = np.exp2(fwd.Pi[clade_at_recipient] - fwd.Pi[clade_at_recipient].max())
    w = T * pir
    tot = w.sum()
    if tot <= 0.0 or not math.isfinite(tot):
        # degenerate; pick argmax as a safe fallback
        return int(np.argmax(w))
    r = rng.random() * tot
    acc = 0.0
    for i in range(len(w)):
        acc += w[i]
        if r <= acc:
            return i
    return len(w) - 1


def _backtrace(fwd: FamilyForward, cid: int, s: int, gene: int,
               sc: Scenario, rng: np.random.Generator, depth: int = 0) -> None:
    """Recursively sample one reconciliation of clade ``cid`` on species ``s``."""
    if depth > 200000:
        raise RuntimeError("backtrace recursion runaway (self-loop did not terminate)")
    sc.occupied.add(s)
    Pi = fwd.Pi
    Pibar = fwd.Pibar
    s1 = int(fwd.sp_child1[s])
    s2 = int(fwd.sp_child2[s])
    has_children = s1 != fwd.S  # internal species branch

    # Build the candidate sub-terms (log2) and a parallel list of "actions".
    logw: List[float] = []
    acts: List[tuple] = []

    splits = fwd.splits_of.get(cid, ())
    for (L, R, lsp) in splits:
        # D
        logw.append(lsp + fwd.log_pD[s] + Pi[L, s] + Pi[R, s]); acts.append((EV_D, L, R))
        # T: R moves (L stays at s)
        logw.append(lsp + Pi[L, s] + Pibar[R, s]); acts.append(("T_R", L, R))
        # T: L moves (R stays at s)
        logw.append(lsp + Pi[R, s] + Pibar[L, s]); acts.append(("T_L", L, R))
        if has_children:
            # S orient 1: L@s1, R@s2 ; orient 2: L@s2, R@s1
            logw.append(lsp + fwd.log_pS[s] + Pi[L, s1] + Pi[R, s2]); acts.append(("S", L, R, s1, s2))
            logw.append(lsp + fwd.log_pS[s] + Pi[L, s2] + Pi[R, s1]); acts.append(("S", L, R, s2, s1))

    # non-split terms
    # DL (self-loop, unrecorded): 2 * pD * E * Pi[c,s]
    logw.append(1.0 + fwd.log_pD[s] + fwd.E[s] + Pi[cid, s]); acts.append(("DL",))
    # TL-lost (self-loop, unrecorded): Pi[c,s] * Ebar[s]
    logw.append(Pi[cid, s] + fwd.Ebar[s]); acts.append(("TL_lost",))
    # TL-move (recorded): Pibar[c,s] * E[s]
    logw.append(Pibar[cid, s] + fwd.E[s]); acts.append(("TL_move",))
    if has_children:
        # SL orient1: c -> s1 (loss at s2): pS * E[s2] * Pi[c,s1]
        logw.append(fwd.log_pS[s] + fwd.E[s2] + Pi[cid, s1]); acts.append(("SL", s1, s2))
        # SL orient2: c -> s2 (loss at s1): pS * E[s1] * Pi[c,s2]
        logw.append(fwd.log_pS[s] + fwd.E[s1] + Pi[cid, s2]); acts.append(("SL", s2, s1))
    # leaf terminal: pS * [c maps to s]
    if fwd.clade_leaf_species.get(cid, -1) == s:
        logw.append(fwd.log_pS[s])  # clade_species_map contributes 0 (log2 1)
        acts.append(("leaf",))

    idx = _categorical_log2(np.asarray(logw), rng)
    act = acts[idx]
    kind = act[0]

    if kind == EV_D:
        _, L, R = act
        lg, rg = sc._new_gene(), sc._new_gene()
        sc.events.append(Event(EV_D, s, gene, left_gene=lg, right_gene=rg))
        _backtrace(fwd, L, s, lg, sc, rng, depth + 1)
        _backtrace(fwd, R, s, rg, sc, rng, depth + 1)

    elif kind == "S":
        _, L, R, sL, sR = act
        lg, rg = sc._new_gene(), sc._new_gene()
        sc.events.append(Event(EV_S, s, gene, left_gene=lg, right_gene=rg))
        _backtrace(fwd, L, sL, lg, sc, rng, depth + 1)
        _backtrace(fwd, R, sR, rg, sc, rng, depth + 1)

    elif kind in ("T_R", "T_L"):
        _, L, R = act
        # the "moving" clade and the "staying" clade
        if kind == "T_R":
            stay_c, move_c = L, R
        else:
            stay_c, move_c = R, L
        recipient = _sample_recipient(fwd, s, move_c, rng)
        lg, rg = sc._new_gene(), sc._new_gene()
        sc.events.append(Event(EV_T, s, gene, dest_species=recipient,
                               left_gene=lg, right_gene=rg))
        _backtrace(fwd, stay_c, s, lg, sc, rng, depth + 1)
        _backtrace(fwd, move_c, recipient, rg, sc, rng, depth + 1)

    elif kind == "SL":
        _, surv, lost = act
        sc.events.append(Event(EV_SL, s, gene, dest_species=surv))
        sc.events.append(Event(EV_L, lost, gene))  # bookkeeping loss on the sibling
        _backtrace(fwd, cid, surv, gene, sc, rng, depth + 1)

    elif kind == "TL_move":
        recipient = _sample_recipient(fwd, s, cid, rng)
        sc.events.append(Event(EV_TL, s, gene, dest_species=recipient))
        sc.events.append(Event(EV_L, s, gene))  # source lineage lost at s
        _backtrace(fwd, cid, recipient, gene, sc, rng, depth + 1)

    elif kind in ("DL", "TL_lost"):
        # geometric self-loop: a copy was lost; resample the same cell, unrecorded.
        _backtrace(fwd, cid, s, gene, sc, rng, depth + 1)

    elif kind == "leaf":
        sc.events.append(Event(EV_LEAF, s, gene))

    else:  # pragma: no cover
        raise RuntimeError(f"unknown sampled action {act!r}")


def sample_family(fwd: FamilyForward, n_samples: int,
                  rng: np.random.Generator) -> List[Scenario]:
    """Draw ``n_samples`` reconciliation scenarios for one family."""
    out: List[Scenario] = []
    root = fwd.root_clade_id
    # origination distribution over species: p^O_e * Pi[root, e]  (then condition).
    logw = fwd.log_pO + fwd.Pi[root]
    for _ in range(n_samples):
        sc = Scenario()
        g0 = sc._new_gene()
        e0 = _categorical_log2(logw, rng)
        sc.events.append(Event(EV_O, e0, g0))
        _backtrace(fwd, root, e0, g0, sc, rng)
        out.append(sc)
    return out
