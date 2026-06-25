"""Fast C++ backtrack reconciliation sampler (JIT-compiled).

Drop-in accelerator for the per-(family, species) presence/copies accumulation that
gpurec/core/sampler.py + experiments/per_node_copies.py do in pure Python. The C++
engine (core/cpp/recon_sampler.cpp) draws n reconciliations per family by the same
stochastic backtrack (AleRax/ALE) and returns the accumulated per-species
  presence[S]  (# samples in which s is occupied)
  copies[S]    (# of {S, SL, leaf} events at s  == ALE branch_counts["copies"])
OpenMP-parallel across samples on Linux; serial (still fast) on macOS.
"""
from __future__ import annotations

import pathlib
import sys
from functools import lru_cache
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.cpp_extension import load

_CPP = pathlib.Path(__file__).resolve().parent / "cpp" / "recon_sampler.cpp"


@lru_cache(maxsize=1)
def _ext() -> Any:
    # NB: NO -ffast-math -- the event weights use -inf for impossible events, and
    # fast-math has undefined inf/nan semantics. Apple clang rejects bare -fopenmp.
    if sys.platform == "darwin":
        cflags, ldflags = ["-O3"], []
    else:
        cflags, ldflags = ["-O3", "-fopenmp"], ["-fopenmp"]
    return load(name="recon_sampler", sources=[str(_CPP)],
                extra_cflags=cflags, extra_ldflags=ldflags, verbose=False)


def _csr_splits(splits_of, C):
    sptr = np.zeros(C + 1, dtype=np.int64)
    sL, sR, slp = [], [], []
    for c in range(C):
        for (L, R, lp) in splits_of.get(c, ()):  # noqa: E741
            sL.append(L); sR.append(R); slp.append(lp)
        sptr[c + 1] = len(sL)
    return (sptr,
            np.asarray(sL, dtype=np.int64), np.asarray(sR, dtype=np.int64),
            np.asarray(slp, dtype=np.float64))


def sample_accumulate_cpp(fwd, n_samples: int, seed: int = 0,
                          n_threads: int = 0):
    """Return per-species accumulators (SUMS over n_samples; divide by n_samples for E[·]):
    a dict with presence, copies, orig, dup, transfer, loss, spec, transfer_in -- each [S].
    presence = #samples occupying s; copies = #{S,SL,leaf}; orig/dup/transfer/loss/spec
    are per-branch event counts (O, D, T_out, L, S); transfer (=T_out) is counted at the
    DONOR, transfer_in at the RECIPIENT -- so a branch's genome arrivals are orig + inherited
    + transfer_in (+dup), and its T_out are donations OUT (do not add to its own genome)."""
    C = int(fwd.Pi.shape[0]); S = int(fwd.S)
    sptr, sL, sR, slp = _csr_splits(fwd.splits_of, C)
    leafsp = np.full(C, -1, dtype=np.int64)
    for c, sp in fwd.clade_leaf_species.items():
        leafsp[c] = sp
    t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt)  # noqa: E731
    f64, i64 = torch.float64, torch.int64
    res = _ext().sample_accumulate(
        t(fwd.Pi, f64), t(fwd.Pibar, f64), t(fwd.E, f64), t(fwd.Ebar, f64),
        t(fwd.log_pS, f64), t(fwd.log_pD, f64), t(fwd.log_pO, f64), t(fwd.transfer_mat, f64),
        t(sptr, i64), t(sL, i64), t(sR, i64), t(slp, f64),
        t(leafsp, i64), t(fwd.sp_child1, i64), t(fwd.sp_child2, i64),
        int(fwd.root_clade_id), C, S, int(n_samples), int(seed), int(n_threads))
    keys = ("presence", "copies", "orig", "dup", "transfer", "loss", "spec", "transfer_in")
    return {k: r.numpy() for k, r in zip(keys, res)}
