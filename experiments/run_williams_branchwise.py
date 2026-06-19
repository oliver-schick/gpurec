"""Estimate BRANCH-WISE (per-species-branch) DTL rates from Williams 2017 .ale
gene families + a rooted species tree, by driving gpurec's specieswise wave
optimizer.

This is the .ale analogue of ``experiments/validate_three_modes.py`` for the
real Williams et al. 2017 archaeal dataset. It:

  1. loads species helpers from the rooted species tree ALONE via the C++
     extension (``preprocess_multiple_families(tree, {})``);
  2. parses each ``.ale`` gene family and builds gpurec CCP arrays
     (``build_family_from_ale``), mapping gene leaves to species by the
     prefix-before-first-``_`` rule (matches ALE / gpurec);
  3. builds a single cross-family wave layout (mirroring
     ``api/model.py::_build_static_state`` and
     ``experiments/test_ale_small_families.py``);
  4. runs ``optimize_theta_wave(..., specieswise=True, ...)`` to get a [S,3]
     theta -> per-branch (D, L, T) rates;
  5. (optionally) feeds a per-species ``fraction_missing`` boundary through the
     new ``leaf_E`` kwarg, exactly as ``core/model.py`` derives ``self.leaf_E``.

The output file is written to match AleRax ``model_parameters.txt`` so the
branch-wise rates are directly comparable:

    # node D L T
    <species_name> <D> <L> <T>
    ...

A JSON sidecar (``<out>.json``) dumps theta, the final NLL/logL, and run config.

Run on an A100 (CUDA required for the optimize step):

    cd /Users/ssolo/GALE
    PYTHONPATH=. python experiments/run_williams_branchwise.py --root Eury

Fast data-wiring sanity check (no GPU, no optimization):

    PYTHONPATH=. python experiments/run_williams_branchwise.py --root Eury --preflight
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = (
    "/work/SzollosiU/gergely-szollosi/williams_run/data/"
    "3_Reconciliation/Williams_et_al_2017"
)
KNOWN_ROOTS = (
    "Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury",
    "HaloThermoplas", "Kor", "TAC", "TackA",
)
_INV_LN2 = 1.0 / math.log(2.0)  # ln -> log2 (mirror core/model.py:115)


# ── Path resolution ──────────────────────────────────────────────────────────
def _glob_ales(d):
    """All *.ale under directory d, excluding macOS ._* AppleDouble stubs."""
    return sorted(p for p in glob.glob(str(Path(d) / "*.ale"))
                  if not Path(p).name.startswith("._"))


def _resolve_paths(data_dir: Path, root: str, extra_ale_dirs=None):
    """Return (species_tree_path, list_of_ale_paths, fraction_missing_path).

    extra_ale_dirs: optional list of additional dirs to glob *.ale from (e.g. the
    archaea60/small_fams complete-set small families). ._* stubs are excluded.
    """
    tree_path = data_dir / "rooted_phylogeny" / root
    ale_dir = data_dir / "ccps"
    fm_path = data_dir / "fraction_missing"
    ale_paths = _glob_ales(ale_dir)
    for d in (extra_ale_dirs or []):
        ale_paths += _glob_ales(d)
    return tree_path, ale_paths, fm_path


def _load_species_helpers(species_tree_path: str):
    """Load species-only helpers from the C++ extension (tree alone).

    Mirrors the snippet in the task brief / GeneDataset: an empty families dict
    makes ``preprocess_multiple_families`` emit only the ``species`` payload.
    Uses the DEFAULT loader (gcc 11 + -fopenmp on the cluster).
    """
    from gpurec.core.preprocess_cpp import _load_extension

    ext = _load_extension()
    raw = ext.preprocess_multiple_families(species_tree_path, {})
    return raw["species"]


def _parse_fraction_missing(fm_path: Path, species_name_to_index, S: int):
    """Parse the tab-separated fraction_missing file into a linear-space [S] vec.

    Each row is "<species>\t<fraction>" where fraction == 1 - p_obs. Species not
    present in the species tree are warned about and skipped. Returns
    (fm_tensor[S], n_set, skipped_names).
    """
    fm = torch.zeros(S, dtype=torch.float64)
    n_set = 0
    skipped: list[str] = []
    for raw_line in fm_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        if len(tok) < 2:
            continue
        species, frac = tok[0], float(tok[1])
        idx = species_name_to_index.get(species)
        if idx is None:
            skipped.append(species)
            continue
        fm[int(idx)] = frac
        n_set += 1
    return fm, n_set, skipped


def _build_leaf_E(species_helpers, fm: torch.Tensor, S: int, dtype: torch.dtype):
    """Derive the log2 leaf boundary leaf_E[S] from a linear fraction_missing[S].

    Byte-faithful port of ``GeneDataset.__init__`` (core/model.py:86-100):
      - species-tree leaf mask = nodes that are never internal parents
        (sP[sP < S] are the internal parents; everything else is a leaf);
      - leaf_E = log2(fraction_missing) at missing leaves, -inf elsewhere.
    """
    sP = species_helpers["s_P_indexes"]
    internal = sP[sP < S].unique()
    leaf_species_mask = torch.ones(S, dtype=torch.bool)
    leaf_species_mask[internal] = False

    fm = fm.to(dtype=dtype)
    missing = leaf_species_mask & (fm > 0)
    leaf_E = torch.where(
        missing,
        torch.log2(fm.clamp_min(torch.finfo(dtype).tiny)),
        torch.full_like(fm, float("-inf")),
    )
    return leaf_E, leaf_species_mask


# ── .ale family loading ──────────────────────────────────────────────────────
def _ale_species(ale, species_of_leaf) -> set[str]:
    """Distinct species names derived from an AleData's leaf labels."""
    return {species_of_leaf(name) for name in ale.leaf_name_to_id}


def _load_families(
    ale_paths,
    species_name_to_index,
    *,
    min_species: int,
    dtype: torch.dtype,
    limit: int = 0,
):
    """Parse + build all (or first ``limit``) .ale families.

    Returns (families, stats) where ``families`` is a list of dicts shaped like
    ``GeneDataset.families`` items (with key ``ccp_helpers``), and ``stats`` is a
    dict reporting kept/dropped counts and any species-mismatch info.

    Families are dropped (matching AleRax) when their distinct-species count is
    < ``min_species``, or when a derived species is absent from the species tree
    (counted + reported rather than crashing).
    """
    from gpurec.io.ale import (
        parse_ale_file,
        build_family_from_ale,
        default_species_of_leaf,
    )

    species_of_leaf = default_species_of_leaf
    paths = ale_paths if limit <= 0 else ale_paths[:limit]

    families = []
    kept = 0
    dropped_min_species = 0
    parse_failures: list[tuple[str, str]] = []
    missing_species_counter: Counter[str] = Counter()  # species -> #families affected
    families_with_missing = 0
    missing_examples: list[tuple[str, str]] = []  # (species, ale_file)

    for i, path in enumerate(paths):
        try:
            ale = parse_ale_file(path)
        except Exception as exc:  # noqa: BLE001  (malformed/empty .ale -> skip)
            parse_failures.append((Path(path).name, repr(exc)))
            continue

        sp_set = _ale_species(ale, species_of_leaf)

        # Detect species absent from the species tree BEFORE building (so a
        # mismatch is reported, not raised, and the family is dropped).
        unknown = [s for s in sp_set if s not in species_name_to_index]
        if unknown:
            families_with_missing += 1
            for s in unknown:
                missing_species_counter[s] += 1
                if len(missing_examples) < 20:
                    missing_examples.append((s, Path(path).name))
            continue

        if len(sp_set) < min_species:
            dropped_min_species += 1
            continue

        try:
            fam = build_family_from_ale(
                ale, species_name_to_index, dtype=dtype,
            )
        except Exception as exc:  # noqa: BLE001
            parse_failures.append((Path(path).name, repr(exc)))
            continue

        # ln -> log2 EXACTLY ONCE (mirror core/model.py:121).
        ccp = fam["ccp"]
        ccp["log_split_probs_sorted"] = ccp["log_split_probs_sorted"] * _INV_LN2

        families.append({
            "ccp_helpers": ccp,
            "root_clade_id": int(fam["root_clade_id"]),
            "leaf_row_index": fam["leaf_row_index"],
            "leaf_col_index": fam["leaf_col_index"],
            "C": int(ccp["C"]),
            "N_splits": int(ccp["N_splits"]),
        })
        kept += 1

    stats = {
        "n_considered": len(paths),
        "kept": kept,
        "dropped_min_species": dropped_min_species,
        "families_with_missing_species": families_with_missing,
        "missing_species_counter": missing_species_counter,
        "missing_examples": missing_examples,
        "parse_failures": parse_failures,
    }
    return families, stats


# ── Wave layout (mirror api/model.py::_build_static_state) ────────────────────
def _build_wave_layout(families, device, dtype, *, max_wave_size=32768):
    """Build a single cross-family wave layout for optimize_theta_wave.

    Mirrors ``api/model.py::_build_static_state`` (the canonical path) and the
    sequence validated in ``experiments/test_ale_small_families.py``:
    collate_gene_families -> compute_clade_waves -> collate_wave ->
    split_phase_waves -> build_wave_layout. Returns (wave_layout, root_clade_ids).
    """
    from gpurec.core.batching import (
        collate_gene_families,
        collate_wave,
        build_wave_layout,
        split_phase_waves,
    )
    from gpurec.core.scheduling import compute_clade_waves

    items = [
        {
            "ccp": fam["ccp_helpers"],
            "leaf_row_index": fam["leaf_row_index"],
            "leaf_col_index": fam["leaf_col_index"],
            "root_clade_id": int(fam["root_clade_id"]),
        }
        for fam in families
    ]
    batched = collate_gene_families(items, dtype=dtype, device=device)

    fams_waves, fams_phases = [], []
    for fam in families:
        w, p = compute_clade_waves(fam["ccp_helpers"])
        fams_waves.append(w)
        fams_phases.append(p)

    offsets = [m["clade_offset"] for m in batched["family_meta"]]
    cross_waves = collate_wave(fams_waves, offsets)

    max_n_waves = max(len(p) for p in fams_phases)
    cross_phases: list[int] = []
    for k in range(max_n_waves):
        phase_k = 1
        for fp in fams_phases:
            if k < len(fp):
                phase_k = max(phase_k, fp[k])
        cross_phases.append(phase_k)

    # Split oversized waves (mirror _build_static_state: phase=None pass).
    cross_waves, cross_phases = split_phase_waves(
        cross_waves, cross_phases, phase=None, max_wave_size=max_wave_size,
    )

    family_clade_counts = [m["C"] for m in batched["family_meta"]]
    family_clade_offsets = [m["clade_offset"] for m in batched["family_meta"]]

    wave_layout = build_wave_layout(
        waves=cross_waves,
        phases=cross_phases,
        ccp_helpers=batched["ccp"],
        leaf_row_index=batched["leaf_row_index"],
        leaf_col_index=batched["leaf_col_index"],
        root_clade_ids=batched["root_clade_ids"],
        device=device,
        dtype=dtype,
        family_clade_counts=family_clade_counts,
        family_clade_offsets=family_clade_offsets,
    )
    return wave_layout, batched["root_clade_ids"]


def _sp_helpers_for_uniform(species_helpers, device, dtype):
    """Move species helpers onto device (skip Recipients_mat for uniform mode);
    return (helpers, ancestors_T). Mirrors GeneDataset._species_helpers_for_mode.
    """
    skip = {"Recipients_mat"}
    out = {
        k: (
            v.to(device=device, dtype=dtype)
            if torch.is_tensor(v) and v.is_floating_point() and k not in skip
            else v.to(device=device) if torch.is_tensor(v) and k not in skip
            else v
        )
        for k, v in species_helpers.items()
    }
    ancestors_T = out["ancestors_dense"].T.to_sparse_coo()
    return out, ancestors_T


# ── Preflight ────────────────────────────────────────────────────────────────
def _preflight(args, data_dir: Path):
    """Parse the tree + a sample of .ale, report wiring, and EXIT.

    Triton-free: only touches the C++ species loader + .ale parser + CCP build.
    Robust to species mismatch (reports offenders rather than crashing).
    """
    tree_path, ale_paths, fm_path = _resolve_paths(data_dir, args.root, args.extra_ale_dir)

    print("=" * 70)
    print("PREFLIGHT  (data wiring sanity check, no optimization)")
    print("=" * 70)
    print(f"  data-dir     : {data_dir}")
    print(f"  root tree    : {tree_path}")
    print(f"  ccps dir     : {data_dir / 'ccps'}")
    print(f"  fraction_miss: {fm_path}")

    if args.root not in KNOWN_ROOTS:
        print(f"  [warn] root {args.root!r} not in known roots {KNOWN_ROOTS}")
    if not tree_path.exists():
        print(f"  [ERROR] species tree not found: {tree_path}")
        return 1
    if not ale_paths:
        print(f"  [ERROR] no .ale files found under {data_dir / 'ccps'}")
        return 1

    print(f"\n  found {len(ale_paths)} .ale files")

    # 1. Species tree alone.
    species_helpers = _load_species_helpers(str(tree_path))
    S = int(species_helpers["S"])
    names = list(species_helpers["names"])
    sp_name_to_idx = species_helpers["species_name_to_index"]
    tree_leaf_names = set(sp_name_to_idx.keys())

    sP = species_helpers["s_P_indexes"]
    internal = sP[sP < S].unique()
    leaf_mask = torch.ones(S, dtype=torch.bool)
    leaf_mask[internal] = False
    n_leaves = int(leaf_mask.sum().item())

    print(f"  species tree : S={S} nodes  ({n_leaves} leaves, "
          f"{S - n_leaves} internal)")
    print(f"  tree leaf labels (first 8): {sorted(tree_leaf_names)[:8]}")

    # 2. Sample of .ale families -> species coverage + mismatch + <min count.
    from gpurec.io.ale import parse_ale_file, default_species_of_leaf

    n_sample = len(ale_paths) if args.families <= 0 else min(args.families, len(ale_paths))
    sample = ale_paths[:n_sample]
    print(f"\n  parsing {len(sample)} .ale families (sample) ...")

    all_ale_species: set[str] = set()
    families_below_min = 0
    families_with_unknown = 0
    unknown_species: Counter[str] = Counter()
    unknown_examples: list[tuple[str, str]] = []
    parse_failures: list[tuple[str, str]] = []
    leaf_count_hist: Counter[int] = Counter()

    for path in sample:
        try:
            ale = parse_ale_file(path)
        except Exception as exc:  # noqa: BLE001
            parse_failures.append((Path(path).name, repr(exc)))
            continue
        sp_set = {default_species_of_leaf(n) for n in ale.leaf_name_to_id}
        all_ale_species |= sp_set
        leaf_count_hist[len(sp_set)] += 1
        unknown = [s for s in sp_set if s not in tree_leaf_names]
        if unknown:
            families_with_unknown += 1
            for s in unknown:
                unknown_species[s] += 1
                if len(unknown_examples) < 20:
                    unknown_examples.append((s, Path(path).name))
        if len(sp_set) < args.min_species:
            families_below_min += 1

    if parse_failures:
        print(f"  [warn] {len(parse_failures)} parse failures, e.g.:")
        for name, err in parse_failures[:5]:
            print(f"           {name}: {err}")

    # 3. Species-set match between .ale-derived species and tree leaves.
    ale_only = sorted(all_ale_species - tree_leaf_names)
    tree_only = sorted(tree_leaf_names - all_ale_species)
    print(f"\n  distinct species across sampled .ale : {len(all_ale_species)}")
    print(f"  species in .ale but NOT in tree       : {len(ale_only)}")
    if ale_only:
        print(f"    {ale_only[:15]}")
    print(f"  tree leaves never seen in .ale sample : {len(tree_only)}")
    if tree_only:
        print(f"    {tree_only[:15]}")

    if unknown_species:
        print(f"\n  [MISMATCH] {families_with_unknown} families reference species "
              f"absent from the tree.")
        print(f"  offending species -> #families affected (top 15):")
        for s, c in unknown_species.most_common(15):
            print(f"    {s:24s} {c}")
        print(f"  examples (species, .ale file):")
        for s, f in unknown_examples[:10]:
            print(f"    {s:24s} {f}")
    else:
        print(f"\n  [OK] every sampled .ale species is present in the species tree.")

    # 4. min-species filter preview (AleRax-style drop).
    print(f"\n  families with <{args.min_species} distinct species "
          f"(would be DROPPED): {families_below_min}")
    print(f"  leaf-count histogram (#species -> #families), low end:")
    for k in sorted(leaf_count_hist)[:8]:
        print(f"    {k:3d} species : {leaf_count_hist[k]} families")

    # 5. fraction_missing wiring.
    if fm_path.exists():
        fm, n_set, fm_skipped = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
        n_missing_leaves = int(((fm > 0) & leaf_mask).sum().item())
        print(f"\n  fraction_missing: {n_set} rows mapped to species nodes; "
              f"{n_missing_leaves} leaves with fraction>0")
        if fm_skipped:
            print(f"  [warn] {len(fm_skipped)} fraction_missing rows for species "
                  f"absent from tree: {fm_skipped[:10]}")
    else:
        print(f"\n  [warn] fraction_missing file not found: {fm_path}")

    print("\nPREFLIGHT complete. Re-run without --preflight to optimize.")
    return 0


# ── Full run ─────────────────────────────────────────────────────────────────
def _run(args, data_dir: Path):
    tree_path, ale_paths, fm_path = _resolve_paths(data_dir, args.root, args.extra_ale_dir)

    if args.root not in KNOWN_ROOTS:
        print(f"[warn] root {args.root!r} not in known roots {KNOWN_ROOTS}")
    if not tree_path.exists():
        raise SystemExit(f"species tree not found: {tree_path}")
    if not ale_paths:
        raise SystemExit(f"no .ale files found under {data_dir / 'ccps'}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA required for optimization (run on the A100). "
            "Use --preflight for the Triton-free data-wiring check on this Mac."
        )
    device = torch.device("cuda")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Import the optimizer LAST (it may pull Triton). Guarded so --preflight
    # never reaches here.
    from gpurec.optimization.wave_optimizer import optimize_theta_wave

    # 1. Species helpers (tree alone).
    print(f"[1/5] Loading species tree {tree_path.name} ...", flush=True)
    species_helpers = _load_species_helpers(str(tree_path))
    S = int(species_helpers["S"])
    names = list(species_helpers["names"])
    sp_name_to_idx = species_helpers["species_name_to_index"]
    print(f"      S={S} species nodes", flush=True)

    # Optional: CLADE-GROUPED ("branch wise") model from AleRax's output.
    group_index = None
    group_info = {"clade_groups": None}
    if args.clade_groups:
        from clade_groups import group_index_for_species_helpers
        cg_dir = Path(args.clade_groups)
        cg_tree = cg_dir / "species_trees" / "starting_species_tree.newick"
        cg_mp = cg_dir / "model_parameters" / "model_parameters.txt"
        if not cg_tree.exists() or not cg_mp.exists():
            raise SystemExit(
                f"--clade-groups dir must contain species_trees/"
                f"starting_species_tree.newick and model_parameters/"
                f"model_parameters.txt; got {cg_dir}"
            )
        group_index, cat_rates, cat_orig, cg_diag = group_index_for_species_helpers(
            species_helpers, str(cg_tree), str(cg_mp)
        )
        if cg_diag["gpurec_unmapped"]:
            raise SystemExit(
                f"clade-groups: {len(cg_diag['gpurec_unmapped'])} gpurec nodes "
                f"could not be mapped to an AleRax category by leaf set "
                f"(tree/topology mismatch?): {cg_diag['gpurec_unmapped'][:5]}"
            )
        n_groups = int(group_index.max().item()) + 1
        group_info = {
            "clade_groups": str(cg_dir),
            "n_groups": n_groups,
            "n_categories_alerax": cg_diag["n_categories"],
            "matched": cg_diag["matched"],
        }
        print(f"      CLADE-GROUPED model: {n_groups} rate categories over "
              f"S={S} branches (from AleRax branch-wise {cg_dir.name})", flush=True)

    # 2. Load families.
    print(f"[2/5] Parsing {len(ale_paths)} .ale families "
          f"(min-species={args.min_species}) ...", flush=True)
    families, stats = _load_families(
        ale_paths, sp_name_to_idx,
        min_species=args.min_species, dtype=dtype, limit=args.families,
    )
    print(f"      considered={stats['n_considered']}  kept={stats['kept']}  "
          f"dropped(<{args.min_species} species)={stats['dropped_min_species']}  "
          f"with-unknown-species={stats['families_with_missing_species']}",
          flush=True)
    if stats["missing_species_counter"]:
        top = stats["missing_species_counter"].most_common(10)
        print(f"      [warn] species absent from tree (top): {top}", flush=True)
        for s, f in stats["missing_examples"][:5]:
            print(f"             e.g. {s} in {f}", flush=True)
    if stats["parse_failures"]:
        print(f"      [warn] {len(stats['parse_failures'])} parse/build failures, "
              f"e.g. {stats['parse_failures'][:3]}", flush=True)
    if not families:
        raise SystemExit("no usable families after filtering; aborting.")

    # 3. Wave layout.
    print(f"[3/5] Building cross-family wave layout for {len(families)} "
          f"families ...", flush=True)
    t0 = time.time()
    wave_layout, root_clade_ids = _build_wave_layout(families, device, dtype)
    print(f"      wave layout built in {time.time() - t0:.1f}s", flush=True)

    sp_helpers_gpu, _ancestors_T = _sp_helpers_for_uniform(
        species_helpers, device, dtype,
    )
    unnorm_row_max = torch.log2(
        species_helpers["Recipients_mat"]
    ).max(dim=-1).values.to(device=device, dtype=dtype)

    # 4. fraction-missing boundary, decoupled into the E (extinction) side
    #    (leaf_E) and the Pi (reconciliation) leaf side (leaf_obs_log).
    #
    #    fm_mode:
    #      'off'    -> leaf_E=None, leaf_obs_log=None (no fraction-missing anywhere).
    #      'both'   -> leaf_E set, leaf_obs_log=leaf_E (AleRax *supplement*: extra
    #                  Pi baseline as well as the E extinction term).
    #      'e-only' -> leaf_E set, leaf_obs_log=None (AleRax *source*-faithful: _fm
    #                  in the extinction recursion only, NO extra Pi baseline).
    #
    #    leaf_obs_log uses the explicit-None sentinel of optimize_theta_wave: in
    #    'e-only' we pass leaf_obs_log=None to force the Pi boundary OFF; in 'both'
    #    we pass leaf_obs_log=leaf_E. (We never leave it unset, so behavior is
    #    explicit regardless of the kwarg default.)
    leaf_E = None
    leaf_obs_log = None
    fm_info = {"enabled": False, "fm_mode": args.fm_mode}
    if args.fm_mode == "off":
        print(f"[4/5] fraction-missing DISABLED (fm-mode=off)", flush=True)
    elif not fm_path.exists():
        print(f"[4/5] [warn] fraction_missing not found ({fm_path}); "
              f"proceeding WITHOUT it (fm-mode={args.fm_mode}).", flush=True)
        fm_info = {"enabled": False, "fm_mode": args.fm_mode,
                   "note": "fraction_missing file not found"}
    else:
        fm, n_set, fm_skipped = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
        leaf_E_cpu, leaf_mask = _build_leaf_E(species_helpers, fm, S, dtype)
        leaf_E = leaf_E_cpu.to(device=device, dtype=dtype)
        n_missing = int(torch.isfinite(leaf_E).sum().item())
        # Pi leaf boundary: same tensor in 'both', dropped in 'e-only'.
        leaf_obs_log = leaf_E if args.fm_mode == "both" else None
        fm_info = {
            "enabled": True,
            "fm_mode": args.fm_mode,
            "applied_to_E": True,
            "applied_to_Pi": args.fm_mode == "both",
            "rows_mapped": n_set,
            "leaves_with_fraction": n_missing,
            "skipped_rows": fm_skipped,
        }
        _pi_note = "E+Pi (supplement)" if args.fm_mode == "both" else "E only (AleRax-faithful)"
        print(f"[4/5] fraction-missing ENABLED [{_pi_note}]: {n_set} rows mapped, "
              f"{n_missing} leaves with fraction>0", flush=True)
        if fm_skipped:
            print(f"      [warn] {len(fm_skipped)} rows for species absent from "
                  f"tree: {fm_skipped[:10]}", flush=True)

    # 5. Optimize (specieswise -> [S,3] branch-wise rates).
    init_log2 = math.log2(args.init_rate)
    theta_init = init_log2 * torch.ones(S, 3, dtype=dtype, device=device)

    # --- Brownian (TKP) rate prior + AleRax-style box bounds ----------------
    from gpurec.core.tree_prior import species_parent_index

    parent_index = species_parent_index(species_helpers).to(device)
    prior_info = {"prior": args.prior}
    brownian_sigma = None
    brownian_root_sigma = args.brownian_root_sigma
    if args.prior == "brownian":
        brownian_sigma = float(args.brownian_sigma)
        prior_info.update({
            "brownian_sigma": brownian_sigma,
            "brownian_root_sigma": brownian_root_sigma,
            "units": "log2",
        })

    theta_bounds = None
    bounds_info = {"rate_bounds": None}
    if args.rate_bounds:
        try:
            lo_s, hi_s = args.rate_bounds.split(",")
            rate_lo, rate_hi = float(lo_s), float(hi_s)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"--rate-bounds must be 'MIN,MAX' (linear rates): {exc}")
        if not (rate_lo > 0 and rate_hi > rate_lo):
            raise SystemExit(f"--rate-bounds requires 0 < MIN < MAX, got {rate_lo},{rate_hi}")
        theta_lo, theta_hi = math.log2(rate_lo), math.log2(rate_hi)
        theta_bounds = (theta_lo, theta_hi)
        bounds_info = {
            "rate_bounds": [rate_lo, rate_hi],
            "theta_bounds_log2": [theta_lo, theta_hi],
        }

    # --- ORIGINATION (uniform | optimize) -----------------------------------
    # 'uniform' (default): p^O_e = 1/S for every branch (omega absent) ->
    #   byte-identical to the pre-origination driver run.
    # 'optimize': jointly optimise a free per-branch origination log2-weight
    #   omega [S] (init 0 = uniform); the reported "O" column is the normalised
    #   p^O_e = softmax(omega).
    omega_init = None
    origination_info = {"origination": args.origination}
    origination_l2 = 0.0
    if args.origination == "optimize":
        omega_init = torch.zeros(S, dtype=dtype, device=device)  # uniform init
        # p^O_e = softmax(omega) is a probability distribution (sum=1), so the
        # appropriate regularizer is L2/ridge on the logits (shrink toward
        # uniform), NOT a tree-structured Brownian prior. 0 -> free omega.
        origination_l2 = float(args.origination_l2)
        origination_info.update({
            "regularization": ("l2" if origination_l2 > 0 else "free"),
            "origination_l2": origination_l2,
        })

    print(f"[5/5] Optimizing specieswise theta [S={S},3]  "
          f"(init-rate={args.init_rate}, optimizer={args.optimizer}, "
          f"pibar={args.pibar_mode}, dtype={args.dtype}, steps={args.steps}, "
          f"fm-mode={args.fm_mode}, origination={args.origination}) ...",
          flush=True)
    if args.prior == "brownian":
        print(f"      prior=brownian  sigma={brownian_sigma} (all axes, log2)  "
              f"root_sigma={brownian_root_sigma} (log2)", flush=True)
    else:
        print(f"      prior=none (free per-branch rates)", flush=True)
    if args.origination == "optimize":
        if origination_l2 > 0:
            print(f"      origination=optimize  free omega + L2 ridge "
                  f"(lambda={origination_l2})", flush=True)
        else:
            print(f"      origination=optimize  (FREE per-branch omega, no reg)",
                  flush=True)
    if theta_bounds is not None:
        print(f"      rate-bounds (AleRax-style box) = [{bounds_info['rate_bounds'][0]:g}, "
              f"{bounds_info['rate_bounds'][1]:g}]  -> theta in "
              f"[{theta_bounds[0]:.4f}, {theta_bounds[1]:.4f}] (log2)", flush=True)

    t0 = time.time()
    result = optimize_theta_wave(
        wave_layout=wave_layout,
        species_helpers=sp_helpers_gpu,
        root_clade_ids=root_clade_ids,
        unnorm_row_max=unnorm_row_max,
        theta_init=theta_init,
        steps=args.steps,
        optimizer=args.optimizer,
        specieswise=True,
        pibar_mode=args.pibar_mode,
        families=families,
        family_batch_size=args.family_batch_size,
        device=device,
        dtype=dtype,
        leaf_E=leaf_E,
        leaf_obs_log=leaf_obs_log,
        verbose=True,
        brownian_sigma=brownian_sigma,
        brownian_root_sigma=brownian_root_sigma,
        parent_index=parent_index,
        theta_bounds=theta_bounds,
        origination=args.origination,
        omega_init=omega_init,
        origination_l2=origination_l2,
        group_index=group_index,
    )
    elapsed = time.time() - t0

    theta = result["theta"]              # [S,3] log2
    rates = result["rates"]              # [S,3] = 2**theta, cols [D,L,T]
    # ORIGINATION: per-branch normalised origination probability p^O_e (the "O"
    # column). For 'uniform' it is exactly 1/S on every branch.
    if args.origination == "optimize":
        origination_prob = result["origination"]  # [S] normalised p^O_e
        omega = result["omega"]                    # [S] log2-weights
        # Preserve the prior fields set during setup; add the fitted values.
        origination_info.update({
            "origination": "optimize",
            "omega_log2": omega.tolist(),
            "origination_prob": origination_prob.tolist(),
        })
        if "origination_prior" in result:
            origination_info["prior"] = result["origination_prior"]
    else:
        origination_prob = torch.full((S,), 1.0 / S, dtype=torch.float64)
        omega = None
    nll = float(result["negative_log_likelihood"])       # MAP objective (data + prior)
    logL = float(result["log_likelihood"])
    # DATA log-likelihood (prior excluded) -- the quantity for cross-root rooting
    # comparisons, since the Brownian penalty depends on the tree topology.
    data_nll = float(result.get("data_negative_log_likelihood", nll))
    data_logL = -data_nll

    # Rate summary across branches.
    def _mmm(col):
        c = rates[:, col]
        return float(c.min()), float(c.median()), float(c.max())

    print(f"\n{'=' * 60}")
    print(f"  DONE  root={args.root}  families={len(families)}  "
          f"time={elapsed:.1f}s")
    print(f"  NLL(log2)={nll:.4f}   logL(log2)={logL:.4f}   data_logL(log2)={data_logL:.4f}")
    for col, lab in ((0, "D"), (1, "L"), (2, "T")):
        lo, md, hi = _mmm(col)
        print(f"  {lab}: min={lo:.6g}  median={md:.6g}  max={hi:.6g}")
    print(f"{'=' * 60}")

    # Print origination summary.
    if args.origination == "optimize":
        o = origination_prob
        print(f"  O (p^O_e): min={float(o.min()):.6g}  median={float(o.median()):.6g}  "
              f"max={float(o.max()):.6g}  (sum={float(o.sum()):.6g})")
        print(f"{'=' * 60}")

    # Write rates (AleRax model_parameters-compatible) + JSON sidecar.
    # With origination='optimize' a 5th column O = p^O_e (normalised origination
    # probability) is appended; 'uniform' keeps the byte-identical D/L/T format.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        if args.origination == "optimize":
            fh.write("# node D L T O\n")
            for s in range(S):
                d, l, t = float(rates[s, 0]), float(rates[s, 1]), float(rates[s, 2])
                o = float(origination_prob[s])
                fh.write(f"{names[s]} {d:.10g} {l:.10g} {t:.10g} {o:.10g}\n")
        else:
            fh.write("# node D L T\n")
            for s in range(S):
                d, l, t = float(rates[s, 0]), float(rates[s, 1]), float(rates[s, 2])
                fh.write(f"{names[s]} {d:.10g} {l:.10g} {t:.10g}\n")
    print(f"  rates written -> {out_path}")

    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    payload = {
        "command": " ".join(sys.argv),
        "root": args.root,
        "data_dir": str(data_dir),
        "species_tree": str(tree_path),
        "S": S,
        "names": names,
        "n_families_kept": len(families),
        "n_families_dropped_min_species": stats["dropped_min_species"],
        "n_families_with_unknown_species": stats["families_with_missing_species"],
        "min_species": args.min_species,
        "init_rate": args.init_rate,
        "optimizer": args.optimizer,
        "pibar_mode": args.pibar_mode,
        "steps": args.steps,
        "dtype": args.dtype,
        "fm_mode": args.fm_mode,
        "fraction_missing": fm_info,
        "prior": prior_info,
        "bounds": bounds_info,
        "origination": origination_info,
        "clade_groups": group_info,
        "negative_log_likelihood_log2": nll,
        "log_likelihood_log2": logL,
        "negative_log_likelihood_ln": nll * math.log(2.0),
        "data_log_likelihood_log2": data_logL,
        "data_log_likelihood_ln": data_logL * math.log(2.0),
        "data_negative_log_likelihood_log2": data_nll,
        "theta_log2": theta.tolist(),
        "rates": rates.tolist(),
        "elapsed_s": elapsed,
        "n_steps": len(result.get("history", [])),
    }
    with open(sidecar, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  sidecar written -> {sidecar}")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("Estimate branch-wise (per-species-branch) DTL rates from "
                     "Williams 2017 .ale families via gpurec's specieswise "
                     "wave optimizer."),
    )
    p.add_argument("--root", default="Eury",
                   help=f"Rooted species tree filename under "
                        f"rooted_phylogeny/ (known: {', '.join(KNOWN_ROOTS)}). "
                        f"Default: Eury")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="Williams_et_al_2017 data dir (default: cluster path)")
    p.add_argument("--extra-ale-dir", action="append", default=None,
                   help="Additional dir(s) to glob *.ale from (repeatable), e.g. "
                        "archaea60/small_fams for the complete set. ._* stubs skipped.")
    p.add_argument("--families", type=int, default=0,
                   help="0 = all families; else use the first N (quick test)")
    p.add_argument("--min-species", type=int, default=4,
                   help="Drop families with < this many distinct species "
                        "(AleRax excludes <4). Default: 4")
    p.add_argument("--steps", type=int, default=200,
                   help="Max optimizer steps (default: 200)")
    p.add_argument("--optimizer", default="lbfgs",
                   choices=["lbfgs", "adam", "sgd"],
                   help="Optimizer (default: lbfgs)")
    p.add_argument("--pibar-mode", default="uniform",
                   help="pibar mode (default: uniform)")
    p.add_argument("--init-rate", type=float, default=0.1,
                   help="Initial rate for all D,L,T (natural space); theta_init "
                        "= log2(init_rate). Default: 0.1")
    p.add_argument("--no-fraction-missing", action="store_true",
                   help="Disable the fraction-missing leaf boundary entirely "
                        "(equivalent to --fm-mode off; overrides --fm-mode).")
    p.add_argument("--fm-mode", default="both", choices=["both", "e-only", "off"],
                   help="How to apply the fraction-missing boundary. "
                        "'both' (default): apply in BOTH the E (extinction) solve "
                        "and the Pi (reconciliation) leaf boundary -- the AleRax "
                        "*supplement* behavior (leaf_E set, leaf_obs_log=leaf_E). "
                        "'e-only': apply in the E solve ONLY, drop the extra Pi "
                        "baseline -- AleRax *source*-faithful (leaf_E set, "
                        "leaf_obs_log=None). 'off': disable everywhere "
                        "(leaf_E=None, leaf_obs_log=None). "
                        "--no-fraction-missing forces 'off'.")
    p.add_argument("--origination", default="uniform",
                   choices=["uniform", "optimize"],
                   help="ORIGINATION strategy (AleRax). 'uniform' (default): "
                        "p^O_e = 1/S on every branch (byte-identical to before). "
                        "'optimize': jointly optimise a free per-branch origination "
                        "log2-weight omega [S] (init uniform); the reported O column "
                        "is the normalised p^O_e = softmax(omega). Requires "
                        "optimizer=lbfgs.")
    p.add_argument("--prior", default="none", choices=["none", "brownian"],
                   help="Rate prior: 'none' (free per-branch) or 'brownian' "
                        "(time-uniform TKP relaxed clock coupling adjacent "
                        "branches' log-rates). Default: none")
    p.add_argument("--brownian-sigma", type=float, default=1.0,
                   help="Brownian increment std (log2 units), applied to all 3 "
                        "axes (D,L,T). Smaller -> stronger smoothing. Default: 1.0")
    p.add_argument("--brownian-root-sigma", type=float, default=5.0,
                   help="Root-anchor std (log2 units) for prior propriety. "
                        "Default: 5.0")
    p.add_argument("--origination-l2", type=float, default=0.0,
                   help="L2/ridge regularization strength on the origination "
                        "logits omega (penalty lambda*sum(omega^2), shrinking "
                        "p^O toward uniform). Only used with --origination "
                        "optimize. Default 0 = FREE omega. This is the correct "
                        "regularizer because p^O=softmax(omega) is a probability "
                        "distribution (sum=1), not independent per-branch rates "
                        "(a Brownian prior is NOT appropriate).")
    p.add_argument("--family-batch-size", type=int, default=0,
                   help="Process families in mini-batches of this size for the "
                        "forward/backward (gradient accumulated across batches), "
                        "bounding GPU memory. 0 (default) = all families at once. "
                        "Use a few hundred for large family sets / big trees to "
                        "avoid OOM.")
    p.add_argument("--rate-bounds", default=None,
                   help="AleRax-style box bounds on LINEAR rates as 'MIN,MAX' "
                        "(e.g. '1e-10,10'); converted to theta (log2) bounds and "
                        "passed to L-BFGS-B. Default: none (lower bound only).")
    p.add_argument("--clade-groups", default=None,
                   help="Replicate AleRax's CLADE-GROUPED 'branch wise' model: "
                        "path to an AleRax 'branch wise/<root>' dir containing "
                        "species_trees/starting_species_tree.newick and "
                        "model_parameters/model_parameters.txt. The ~17 distinct "
                        "(D,L,T) categories there define a group_index[S] (mapped "
                        "by descendant leaf set), and gpurec optimizes G grouped "
                        "rate rows instead of S free per-branch rates. Forces "
                        "specieswise + prior none. Default: None (free per-branch).")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"],
                   help="Precision (default: float64)")
    p.add_argument("--out", default=None,
                   help="Output rates file (default: "
                        "results/williams_<root>_branchwise.rates.txt)")
    p.add_argument("--preflight", action="store_true",
                   help="Parse tree + sample .ale, report wiring, and EXIT "
                        "(no optimization, runs without CUDA/Triton)")
    args = p.parse_args(argv)
    # --no-fraction-missing is a legacy alias for --fm-mode off (and overrides it).
    if args.no_fraction_missing:
        args.fm_mode = "off"
    if args.out is None:
        args.out = f"results/williams_{args.root}_branchwise.rates.txt"
    return args


def main(argv=None):
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    if args.preflight:
        return _preflight(args, data_dir)
    return _run(args, data_dir)


if __name__ == "__main__":
    sys.exit(main())
