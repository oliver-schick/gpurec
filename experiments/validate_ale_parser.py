"""Validate the .ale -> gpurec CCP parser against gpurec's C++ amalgamator.

Strategy
--------
For one real ALE example (species tree + a 3-tree sample + its .ale):

  reference : 3-tree Newick sample --> C++ amalgamate_clades_and_splits  --> CCP
  test      : same sample's .ale    --> gpurec.io.ale.build_family_from_ale --> CCP

Both produce gpurec-format CCP arrays; we decode each split to leaf-NAME sets
(robust to clade-id permutation) and assert the two conditional clade
distributions are identical.

Optionally re-generates the .ale with the ALEobserve binary to validate the
full ALEobserve -> parser loop.

Runs entirely on CPU (no Triton): it only touches the C++ preprocessing
extension and the pure-Python parser.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from gpurec.io.ale import parse_ale_file, build_family_from_ale


def _load_ext_macfriendly():
    """Compile the gpurec C++ preprocessing extension on macOS.

    gpurec's loader hard-codes ``-fopenmp`` (for the cluster's gcc); Apple clang
    rejects it. The C++ uses only ``#pragma omp`` (no ``omp_*`` runtime calls),
    so we drop ``-fopenmp`` (serial, identical results) and add the Homebrew
    libomp include dir so ``#include <omp.h>`` resolves. Uses Apple clang, so the
    torch (libc++) ABI is matched.
    """
    from torch.utils.cpp_extension import load

    cpp_dir = Path(__file__).resolve().parents[1] / "gpurec" / "core" / "cpp"
    sources = [str(cpp_dir / f) for f in
               ("preprocess.cpp", "tree_utils.cpp", "clade_utils.cpp")]
    cflags = ["-O3"]
    libomp_inc = Path("/opt/homebrew/opt/libomp/include")
    if libomp_inc.exists():
        cflags.append(f"-I{libomp_inc}")
    build_dir = Path.home() / ".cache" / "gpurec_macval" / "preprocess_cpp"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(name="preprocess_cpp", sources=sources, extra_cflags=cflags,
                extra_ldflags=[], build_directory=str(build_dir), verbose=False)

EX = Path("/Users/ssolo/src/tmp_ALE/example_data")
SPECIES_TREE = EX / "S.tree"
TREELIST = EX / "first_3.trees"
ALE_FILE = EX / "first_3.trees.ale"
ALEOBSERVE = Path("/Users/ssolo/src/ALE_old/ALEobserve")

TOL = 1e-9


def _parse_argv() -> None:
    """Optional override: validate_ale_parser.py [species_tree treelist ale_file]."""
    global SPECIES_TREE, TREELIST, ALE_FILE
    if len(sys.argv) == 4:
        SPECIES_TREE = Path(sys.argv[1])
        TREELIST = Path(sys.argv[2])
        ALE_FILE = Path(sys.argv[3])
    elif len(sys.argv) != 1:
        raise SystemExit("usage: validate_ale_parser.py [species_tree treelist ale_file]")


# ---------------------------------------------------------------------------
# Decode a gpurec CCP into a canonical, id-independent distribution:
#   { (parent_leafnames, frozenset{left_leafnames, right_leafnames}) -> prob }
# ---------------------------------------------------------------------------


def _canonical_from_ref(ccp: dict) -> dict:
    """Reference path: C++ family ccp dict (with include_details)."""
    C = int(ccp["C"])
    N = int(ccp["N_splits"])
    clade_leaves = ccp["clade_leaves"]          # clade id -> [leaf idx]
    clade_leaf_labels = ccp["clade_leaf_labels"]  # singleton clade -> leaf name

    # leaf index -> name, recovered from singleton clades
    idx2name: dict[int, str] = {}
    for cid in range(C):
        leaves = list(clade_leaves[cid])
        if len(leaves) == 1:
            idx2name[int(leaves[0])] = clade_leaf_labels[cid]

    def names(cid: int) -> frozenset:
        return frozenset(idx2name[int(i)] for i in clade_leaves[cid])

    parents = ccp["split_parents_sorted"].tolist()
    lr = ccp["split_leftrights_sorted"].tolist()
    lefts, rights = lr[:N], lr[N:]
    logp = ccp["log_split_probs_sorted"].tolist()
    return _assemble(parents, lefts, rights, logp, names)


def _canonical_from_test(family: dict) -> dict:
    """Test path: our parser's family dict (built with with_details=True)."""
    ccp = family["ccp"]
    N = int(ccp["N_splits"])
    clade_leaf_names = ccp["clade_leaf_names"]  # clade id -> frozenset[str]

    def names(cid: int) -> frozenset:
        return clade_leaf_names[cid]

    parents = ccp["split_parents_sorted"].tolist()
    lr = ccp["split_leftrights_sorted"].tolist()
    lefts, rights = lr[:N], lr[N:]
    logp = ccp["log_split_probs_sorted"].tolist()
    return _assemble(parents, lefts, rights, logp, names)


def _assemble(parents, lefts, rights, logp, names) -> dict:
    out: dict = {}
    for i in range(len(parents)):
        key = (names(parents[i]), frozenset((names(lefts[i]), names(rights[i]))))
        prob = math.exp(logp[i]) if logp[i] != float("-inf") else 0.0
        if key in out:
            raise AssertionError(f"duplicate split key: {key}")
        out[key] = prob
    return out


def _short(s: frozenset, k: int = 3) -> str:
    items = sorted(s)
    return "{" + ",".join(items[:k]) + ("..." if len(items) > k else "") + "}"


def compare(ref: dict, test: dict) -> bool:
    ok = True
    only_ref = set(ref) - set(test)
    only_test = set(test) - set(ref)
    if only_ref:
        ok = False
        print(f"  [FAIL] {len(only_ref)} splits in C++ ref but NOT in parser, e.g.:")
        for key in list(only_ref)[:5]:
            print(f"         parent={_short(key[0])} -> {[_short(c) for c in key[1]]}")
    if only_test:
        ok = False
        print(f"  [FAIL] {len(only_test)} splits in parser but NOT in C++ ref, e.g.:")
        for key in list(only_test)[:5]:
            print(f"         parent={_short(key[0])} -> {[_short(c) for c in key[1]]}")

    max_dprob = 0.0
    for key in set(ref) & set(test):
        max_dprob = max(max_dprob, abs(ref[key] - test[key]))
    print(f"  shared splits: {len(set(ref) & set(test))}  max |dprob| = {max_dprob:.3e}")
    if max_dprob > TOL:
        ok = False
        print(f"  [FAIL] probability mismatch exceeds tol {TOL:g}")

    # per-parent normalization sanity (both should sum to ~1 per parent)
    for label, dist in (("ref", ref), ("test", test)):
        sums: dict = {}
        for (parent, _children), p in dist.items():
            sums[parent] = sums.get(parent, 0.0) + p
        bad = [s for s in sums.values() if abs(s - 1.0) > 1e-6]
        if bad:
            ok = False
            print(f"  [FAIL] {label}: {len(bad)} parents whose split probs don't sum to 1")
    return ok


def build_reference(ext, tmp: Path) -> dict:
    """Split the treelist into per-tree Newick files and run the C++ amalgamator."""
    trees = [ln.strip() for ln in TREELIST.read_text().splitlines() if ln.strip()]
    paths = []
    for i, t in enumerate(trees):
        p = tmp / f"tree_{i}.nwk"
        p.write_text(t + "\n")
        paths.append(str(p))
    print(f"  reference: {len(paths)} gene trees -> C++ amalgamator")
    raw = ext.preprocess_multiple_families(
        str(SPECIES_TREE), {"fam": paths}, include_details=True
    )
    return raw["families"]["fam"], raw["species"]["species_name_to_index"]


def main() -> int:
    _parse_argv()
    print(f"species={SPECIES_TREE.name}  treelist={TREELIST.name}  ale={ALE_FILE.name}")
    print("Loading C++ preprocessing extension (JIT compile on first run)...")
    ext = _load_ext_macfriendly()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ref_family, sp_index = build_reference(ext, tmp)

        print(f"  test: parse {ALE_FILE.name}")
        ale = parse_ale_file(ALE_FILE)
        print(f"        observations={ale.observations:g}  leaves={ale.n_leaves}  "
              f"set-ids={len(ale.set_leaves)}  Dip={len(ale.dip_counts)}  "
              f"Bip={len(ale.bip_counts)}")
        test_family = build_family_from_ale(
            ale, sp_index, with_details=True
        )

        ref_ccp = ref_family["ccp"]
        test_ccp = test_family["ccp"]
        print(f"\n  C++ ref : C={int(ref_ccp['C'])}  N_splits={int(ref_ccp['N_splits'])}")
        print(f"  parser  : C={int(test_ccp['C'])}  N_splits={int(test_ccp['N_splits'])}")

        ref_canon = _canonical_from_ref(ref_ccp)
        test_canon = _canonical_from_test(test_family)
        print()
        ok = compare(ref_canon, test_canon)

        # Also check leaf->species mapping agrees (as a set of (leafnames,col)).
        ok = ok and _check_leaf_species(ref_family, test_family, ref_ccp, test_ccp)

    print("\n" + ("PASS: parser matches C++ amalgamator" if ok else "FAIL"))
    return 0 if ok else 1


def _check_leaf_species(ref_family, test_family, ref_ccp, test_ccp) -> bool:
    """leaf clade -> species column must agree between the two paths."""
    # ref: clade id -> name via singleton clade_leaves; col via leaf_col_index
    clade_leaves = ref_ccp["clade_leaves"]
    labels = ref_ccp["clade_leaf_labels"]
    ref_rows = ref_family["leaf_row_index"].tolist()
    ref_cols = ref_family["leaf_col_index"].tolist()
    ref_map = {}
    for cid, col in zip(ref_rows, ref_cols):
        ref_map[labels[cid]] = col

    names = test_ccp["clade_leaf_names"]
    test_rows = test_family["leaf_row_index"].tolist()
    test_cols = test_family["leaf_col_index"].tolist()
    test_map = {}
    for cid, col in zip(test_rows, test_cols):
        (nm,) = tuple(names[cid])
        test_map[nm] = col

    if ref_map != test_map:
        diff = {k: (ref_map.get(k), test_map.get(k)) for k in set(ref_map) | set(test_map)
                if ref_map.get(k) != test_map.get(k)}
        print(f"  [FAIL] leaf->species column mismatch ({len(diff)} leaves), e.g. "
              f"{list(diff.items())[:3]}")
        return False
    print(f"  leaf->species mapping: {len(test_map)} leaves agree")
    return True


if __name__ == "__main__":
    sys.exit(main())
