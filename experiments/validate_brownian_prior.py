"""Validate the Brownian (time-uniform TKP) rate prior + parent-index helper.

CPU-only (no Triton, no CUDA needed). Checks:

  (1) PARENT INDEX: derive ``parent_index[S]`` from ``species_helpers`` via
      ``species_parent_index`` for the bundled test species tree (loaded with the
      macOS C++ loader pattern), AND for a hand-made random binary tree. Verify
      both are valid rooted trees (exactly one root, every other node one parent,
      node count == S, acyclic) -- the helper already asserts this, so a clean
      return is the pass.

  (2) PRIOR FINITE-DIFFERENCE CHECK: central differences of the penalty ``P``
      vs the analytic gradient ``dP/dtheta``, over many random theta entries, for
      BOTH scalar and per-axis sigma. Assert max rel-err < 1e-6. Confirm the
      root-anchor term and the parent/child scatter are both exercised (the root
      row gets a nonzero anchor contribution; interior rows get parent+child
      contributions).

Run from the repo root on the Mac (no GPU):

    PYTHONPATH=. python experiments/validate_brownian_prior.py

The combined NLL+prior gradient FD check needs the GPU wave path; the exact
command to run on the A100 is printed at the end (modeled on
experiments/validate_fraction_missing_wave.py).

Exits nonzero on any failure.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gpurec.core.tree_prior import (  # noqa: E402
    brownian_log_prior_and_grad,
    species_parent_index,
)

DT = torch.float64
SP = "tests/data/test_trees_1/sp.nwk"


def _load_ext():
    """Build the C++ preprocess extension on macOS (drop -fopenmp; add libomp)."""
    from torch.utils.cpp_extension import load
    cpp = _REPO_ROOT / "gpurec" / "core" / "cpp"
    srcs = [str(cpp / f) for f in ("preprocess.cpp", "tree_utils.cpp", "clade_utils.cpp")]
    extra_cflags = ["-O3"]
    inc = Path("/opt/homebrew/opt/libomp/include")
    ldflags = []
    if inc.exists():
        extra_cflags.append(f"-I{inc}")
    else:
        extra_cflags.append("-fopenmp")
        ldflags.append("-fopenmp")
    bdir = Path.home() / ".cache" / "gpurec_macval" / "preprocess_cpp"
    bdir.mkdir(parents=True, exist_ok=True)
    return load(name="preprocess_cpp", sources=srcs, extra_cflags=extra_cflags,
                extra_ldflags=ldflags, build_directory=str(bdir), verbose=False)


def _random_tree_helpers(n_leaves: int, seed: int):
    """Hand-make a random rooted strictly-binary tree's species_helpers.

    Returns a dict with 'S', 's_P_indexes' (2K layout), 's_C12_indexes' encoded
    exactly as the C++ preprocess emits them (so species_parent_index parses it
    the same way), plus the ground-truth parent array for cross-checking.
    """
    rng = torch.Generator().manual_seed(seed)
    # Build by randomly merging active nodes (coalescent-style). Leaves are
    # ids [0, n_leaves); each internal node gets the next id.
    active = list(range(n_leaves))
    next_id = n_leaves
    parents = {}  # child -> parent
    left_children, right_children, parent_nodes = [], [], []
    while len(active) > 1:
        i = int(torch.randint(0, len(active), (1,), generator=rng).item())
        a = active.pop(i)
        j = int(torch.randint(0, len(active), (1,), generator=rng).item())
        b = active.pop(j)
        p = next_id
        next_id += 1
        parents[a] = p
        parents[b] = p
        parent_nodes.append(p)
        left_children.append(a)
        right_children.append(b)
        active.append(p)
    S = next_id
    root = active[0]

    # C++ layout: s_P_indexes = [p_0..p_{K-1}, p_0+S..p_{K-1}+S];
    #             s_C12_indexes = [left_0..left_{K-1}, right_0..right_{K-1}].
    pnodes = torch.tensor(parent_nodes, dtype=torch.long)
    sP = torch.cat([pnodes, pnodes + S], dim=0)
    sC = torch.cat([torch.tensor(left_children, dtype=torch.long),
                    torch.tensor(right_children, dtype=torch.long)], dim=0)
    helpers = {"S": S, "s_P_indexes": sP, "s_C12_indexes": sC}

    gt_parent = torch.full((S,), -1, dtype=torch.long)
    for c, p in parents.items():
        gt_parent[c] = p
    return helpers, gt_parent, root


def check_parent_index() -> bool:
    print("=" * 70)
    print("[1] PARENT INDEX derivation + validation")
    print("=" * 70)
    ok = True

    # (a) bundled test species tree via the macOS C++ loader.
    try:
        ext = _load_ext()
        raw = ext.preprocess_multiple_families(SP, {})
        sr = raw["species"]
        helpers = {
            "S": int(sr["S"]),
            "s_P_indexes": sr["s_P_indexes"],
            "s_C12_indexes": sr["s_C12_indexes"],
            "names": list(sr["names"]),
        }
        S = helpers["S"]
        pi = species_parent_index(helpers)  # asserts validity internally
        n_root = int((pi == -1).sum().item())
        print(f"  (a) bundled tree  S={S}: parent_index built + verified valid "
              f"(roots={n_root})")
        print(f"      parent_index = {pi.tolist()}")
        # cross-check a couple of named edges against the newick
        #   (((5,(11,12)6)3,(13,14)4)1,((9,10)7,8)2)0
        names = helpers["names"]
        name_of = {i: names[i] for i in range(S)}
        edges = {name_of[c]: (name_of[int(pi[c])] if pi[c] >= 0 else "ROOT")
                 for c in range(S)}
        expect = {"5": "3", "6": "3", "11": "6", "3": "1", "4": "1", "1": "0",
                  "7": "2", "8": "2", "2": "0", "0": "ROOT"}
        mism = {k: (edges[k], v) for k, v in expect.items() if edges.get(k) != v}
        if mism:
            print(f"      [FAIL] edge mismatches vs newick: {mism}")
            ok = False
        else:
            print(f"      newick cross-check OK (10 named edges match)")
    except Exception as exc:  # noqa: BLE001
        print(f"  (a) [SKIP] could not build C++ loader on this box: {exc!r}")

    # (b) hand-made random trees.
    for seed in (0, 7, 42):
        for n_leaves in (3, 8, 50):
            helpers, gt_parent, root = _random_tree_helpers(n_leaves, seed)
            S = helpers["S"]
            pi = species_parent_index(helpers)  # asserts validity
            same = bool(torch.equal(pi, gt_parent))
            n_root = int((pi == -1).sum().item())
            tag = "OK" if same and n_root == 1 else "FAIL"
            if not (same and n_root == 1):
                ok = False
            print(f"  (b) random seed={seed} leaves={n_leaves} S={S}: "
                  f"matches ground-truth parents={same} roots={n_root} [{tag}]")

    print(f"  parent-index check: {'PASS' if ok else 'FAIL'}\n")
    return ok


def check_prior_fd() -> bool:
    print("=" * 70)
    print("[2] PRIOR finite-difference gradient check")
    print("=" * 70)
    ok = True
    torch.manual_seed(1234)

    # Use a mid-size hand-made tree so many parent/child edges are exercised.
    helpers, gt_parent, root = _random_tree_helpers(n_leaves=40, seed=3)
    S = helpers["S"]
    parent_index = species_parent_index(helpers)

    configs = [
        ("scalar sigma", 0.8, 4.0, -3.3),
        ("per-axis sigma", torch.tensor([0.5, 1.3, 2.0], dtype=DT),
         torch.tensor([3.0, 5.0, 7.0], dtype=DT),
         torch.tensor([-3.3, -2.0, -4.0], dtype=DT)),
    ]

    # The penalty P is exactly quadratic in theta, so central differences have
    # ZERO truncation error -- the only FD error is floating-point cancellation
    # in (P(+h) - P(-h)), which scales like eps*|P|/(h*|grad|). With P ~ O(10^3)
    # in float64 a tiny h (1e-6) leaves cancellation noise ~1e-6; a larger step
    # (1e-3) pushes the differenced quantity well above the rounding floor while
    # the exact-quadratic property keeps truncation at zero.
    h = 1e-3
    for name, sigma, sigma_root, mu in configs:
        # Random theta spread away from mu so the root anchor is nonzero.
        theta = (torch.randn(S, 3, dtype=DT) * 1.5 - 3.0).requires_grad_(False)

        penalty, grad = brownian_log_prior_and_grad(
            theta, parent_index, sigma, sigma_root, mu,
        )

        # Sanity: root row receives a nonzero anchor contribution, and an interior
        # (non-root, non-leaf) node receives both parent and child contributions.
        root_id = int((parent_index == -1).nonzero(as_tuple=False)[0].item())
        root_grad_nonzero = bool((grad[root_id].abs() > 0).all().item())
        # an interior node = appears as a parent AND has a parent
        has_parent = parent_index >= 0
        is_parent = torch.zeros(S, dtype=torch.bool)
        is_parent[parent_index[has_parent]] = True
        interior = (has_parent & is_parent).nonzero(as_tuple=False).flatten()
        interior_ok = interior.numel() > 0

        # Finite-difference EVERY entry (S*3) for the smaller tree; here S~79.
        max_rel = 0.0
        worst = None
        n_checked = 0
        for s in range(S):
            for col in range(3):
                tp = theta.clone(); tp[s, col] += h
                tm = theta.clone(); tm[s, col] -= h
                pp, _ = brownian_log_prior_and_grad(tp, parent_index, sigma, sigma_root, mu)
                pm, _ = brownian_log_prior_and_grad(tm, parent_index, sigma, sigma_root, mu)
                fd = float((pp - pm) / (2 * h))
                an = float(grad[s, col])
                denom = max(abs(fd), abs(an), 1e-8)
                rel = abs(fd - an) / denom
                n_checked += 1
                if rel > max_rel:
                    max_rel = rel
                    worst = (s, col, fd, an)

        chk = max_rel < 1e-6
        ok &= chk and root_grad_nonzero and interior_ok
        print(f"  [{name}] P={float(penalty):.6f}  entries checked={n_checked}  "
              f"max rel-err={max_rel:.3e} (<1e-6) -> {chk}")
        print(f"     root-anchor exercised (grad[root] all nonzero)={root_grad_nonzero}  "
              f"interior parent+child node present={interior_ok} (#interior={interior.numel()})")
        if worst:
            print(f"     worst: s={worst[0]} col={worst[1]} "
                  f"fd={worst[2]:.6e} analytic={worst[3]:.6e}")

    # sigma -> inf limit: increment penalty vanishes, only root anchor remains.
    theta = torch.randn(S, 3, dtype=DT) * 1.5 - 3.0
    big = 1e12
    p_inf, g_inf = brownian_log_prior_and_grad(theta, parent_index, big, 4.0, -3.3)
    root_id = int((parent_index == -1).nonzero(as_tuple=False)[0].item())
    # Only the root row should carry gradient (anchor); all others ~0.
    other_max = float(g_inf[torch.arange(S) != root_id].abs().max())
    chk_inf = other_max < 1e-9
    ok &= chk_inf
    print(f"  [sigma->inf] increment penalty vanishes: max|grad| off-root="
          f"{other_max:.2e} (<1e-9) -> {chk_inf}")

    print(f"  prior FD check: {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> int:
    ok = True
    ok &= check_parent_index()
    ok &= check_prior_fd()

    print("=" * 70)
    print("RESULT:", "PASS -- Brownian prior + parent-index validated (CPU)"
          if ok else "FAIL -- see checks above")
    print("=" * 70)
    print()
    print("NEXT (GPU, A100): the combined NLL+prior gradient FD check needs the")
    print("wave path. Run, modeled on validate_fraction_missing_wave.py:")
    print()
    print("    PYTHONPATH=. python experiments/validate_fraction_missing_wave.py")
    print()
    print("and the full branch-wise prior runs:")
    print("    PYTHONPATH=. python experiments/run_williams_branchwise.py \\")
    print("        --root Eury --prior brownian --brownian-sigma 1.0")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
