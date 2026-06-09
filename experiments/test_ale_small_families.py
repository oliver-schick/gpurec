"""Stress-test the .ale loader on tiny (1/2/3-leaf) families.

`archaea60/small_fams` is dominated by degenerate single-gene families:
  - 1 leaf  -> C=1, N_splits=0, root clade == the single leaf (no splits)
  - 2 leaves-> Gamma={a,b}, one root split Gamma->({a},{b})
  - 3 leaves-> the (a,(b,c)) structure (same as validated examples)

These are exactly the cases that break CCP/scheduling code. This script:
  1. parses + builds every (sampled) small_fams .ale, classifies by #leaves,
     and checks per-parent split probabilities normalize;
  2. runs the Triton-free data-prep path (collate_gene_families +
     compute_clade_waves + build_wave_layout) on a MIXED batch that includes
     the degenerate families, to confirm batching/scheduling survive them.

Run:
    PYTHONPATH=. python experiments/test_ale_small_families.py [small_fams_dir]
"""
from __future__ import annotations

import collections
import glob
import math
import sys
from pathlib import Path

import torch

from gpurec.io.ale import parse_ale_file, build_family_from_ale, default_species_of_leaf
from gpurec.core.batching import (
    collate_gene_families,
    collate_wave,
    build_wave_layout,
    split_phase_waves,
)
from gpurec.core.scheduling import compute_clade_waves

SMALL_FAMS = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("/Users/ssolo/src/archaea60/small_fams")


def n_leaves(ale) -> int:
    return len(ale.leaf_name_to_id)


def synth_species_index(files) -> dict[str, int]:
    species = set()
    for f in files:
        try:
            ale = parse_ale_file(f)
        except Exception:  # noqa: BLE001  (empty/malformed file -> skip)
            continue
        for name in ale.leaf_name_to_id:
            species.add(default_species_of_leaf(name))
    return {s: i for i, s in enumerate(sorted(species))}


def check_normalization(fam) -> float:
    """Max deviation from 1.0 of per-parent split-prob sums (parents with splits)."""
    ccp = fam["ccp"]
    N = int(ccp["N_splits"])
    if N == 0:
        return 0.0
    parents = ccp["split_parents_sorted"].tolist()
    logp = ccp["log_split_probs_sorted"].tolist()
    sums: dict[int, float] = collections.defaultdict(float)
    for p, lp in zip(parents, logp):
        sums[p] += math.exp(lp) if lp != float("-inf") else 0.0
    return max(abs(s - 1.0) for s in sums.values())


def main() -> int:
    if not SMALL_FAMS.is_dir():
        print(f"missing dir: {SMALL_FAMS}")
        return 1

    files = sorted(glob.glob(str(SMALL_FAMS / "*.ale")))
    print(f"found {len(files)} .ale files in {SMALL_FAMS}")

    # Sample for speed: every Kth file (plus we always keep the leaf-count reps).
    sample = files if len(files) <= 4000 else files[:: max(1, len(files) // 4000)]
    print(f"parsing {len(sample)} (sampled) ...")

    sp_index = synth_species_index(sample)
    print(f"synthetic species map: {len(sp_index)} distinct species")

    by_k = collections.Counter()          # n_leaves -> count
    cn_by_k: dict[int, tuple] = {}         # n_leaves -> (C, N_splits) example
    rep_file: dict[int, str] = {}          # n_leaves -> example file
    max_norm = 0.0
    fails = []

    for f in sample:
        try:
            ale = parse_ale_file(f)
            fam = build_family_from_ale(ale, sp_index)
            k = n_leaves(ale)
            by_k[k] += 1
            C = int(fam["ccp"]["C"])
            N = int(fam["ccp"]["N_splits"])
            cn_by_k.setdefault(k, (C, N))
            rep_file.setdefault(k, f)
            max_norm = max(max_norm, check_normalization(fam))
            # invariants
            assert 0 <= fam["root_clade_id"] < C, f"bad root in {f}"
            assert fam["leaf_row_index"].numel() == k, f"leaf count mismatch in {f}"
        except Exception as e:  # noqa: BLE001
            fails.append((Path(f).name, repr(e)))

    print("\n=== parse+build results by leaf count ===")
    print(f"{'#leaves':>7}  {'#families':>9}  {'C':>4}  {'N_splits':>8}")
    for k in sorted(by_k):
        C, N = cn_by_k[k]
        print(f"{k:>7}  {by_k[k]:>9}  {C:>4}  {N:>8}   e.g. {Path(rep_file[k]).name}")
    print(f"\nmax |per-parent prob sum - 1| = {max_norm:.2e}")
    print(f"parse/build failures: {len(fails)}")
    for name, err in fails[:5]:
        print(f"  {name}: {err}")

    # ---- Data-prep path on a mixed batch incl. degenerate families ----
    print("\n=== collate + wave-layout on a mixed batch (incl. 1-leaf) ===")
    picks = [rep_file[k] for k in sorted(rep_file)]  # one of each leaf count
    families = [build_family_from_ale(parse_ale_file(f), sp_index) for f in picks]
    ok = run_dataprep(families, [Path(p).name for p in picks])

    overall = (len(fails) == 0) and (max_norm < 1e-6) and ok
    print("\n" + ("PASS: small families handled (parse+build+collate+wave)"
                  if overall else "FAIL: see above"))
    return 0 if overall else 1


def run_dataprep(families, labels) -> bool:
    items = [{
        "ccp": fam["ccp"],
        "leaf_row_index": fam["leaf_row_index"],
        "leaf_col_index": fam["leaf_col_index"],
        "root_clade_id": int(fam["root_clade_id"]),
    } for fam in families]
    try:
        batched = collate_gene_families(items, dtype=torch.float64, device="cpu")
        fams_waves, fams_phases = [], []
        for fam in families:
            w, p = compute_clade_waves(fam["ccp"])
            fams_waves.append(w)
            fams_phases.append(p)
        offsets = [m["clade_offset"] for m in batched["family_meta"]]
        cross_waves = collate_wave(fams_waves, offsets)
        max_n = max(len(p) for p in fams_phases)
        cross_phases = [max((fp[k] for fp in fams_phases if k < len(fp)), default=1)
                        for k in range(max_n)]
        cross_waves, cross_phases = split_phase_waves(cross_waves, cross_phases,
                                                      phase=None, max_wave_size=32768)
        layout = build_wave_layout(
            waves=cross_waves, phases=cross_phases,
            ccp_helpers=batched["ccp"],
            leaf_row_index=batched["leaf_row_index"],
            leaf_col_index=batched["leaf_col_index"],
            root_clade_ids=batched["root_clade_ids"],
            device="cpu", dtype=torch.float64,
            family_clade_counts=[m["C"] for m in batched["family_meta"]],
            family_clade_offsets=[m["clade_offset"] for m in batched["family_meta"]],
        )
        total_C = int(batched["ccp"]["C"])
        roots = batched["root_clade_ids"].tolist()
        print(f"  batch: {len(families)} families ({', '.join(labels)})")
        print(f"  total clades C={total_C}, total splits N={int(batched['ccp']['N_splits'])}")
        print(f"  wave layout built: {len(cross_waves)} cross-family waves; "
              f"root_clade_ids={roots} (all < {total_C}: {all(0 <= r < total_C for r in roots)})")
        return all(0 <= r < total_C for r in roots)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] data-prep raised: {e!r}")
        return False


if __name__ == "__main__":
    sys.exit(main())
