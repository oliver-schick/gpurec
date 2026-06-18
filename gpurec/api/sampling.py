"""Sampling helper: feed gpurec-optimized DTL rates into AleRax for reconciliation sampling.

The public entry point is :meth:`GeneReconModel.sample_reconciliations`,
which delegates here. This module exists separately so the labelling
machinery (replicating ``PLLRootedTree::ensureUniqueLabels``) does not
clutter the model class.

Pure-Python / Rust-free: the species-tree Newick parsing and the AleRax
invocation are implemented here directly (the previous version delegated to
the ``rustree`` Rust crate). The AleRax binary itself is still required at
sampling time, but no Rust toolchain or extension is needed to build or import
gpurec.
"""
from __future__ import annotations

import pathlib
import subprocess
import tempfile
from typing import Any

import torch


# ──────────────────────────────────────────────────────────────────────
# Minimal Newick parser (species trees: rooted, binary, optional internal
# labels + branch lengths). Replaces rustree.parse_species_tree.
# ──────────────────────────────────────────────────────────────────────


class _SpNode:
    __slots__ = ("name", "parent", "left_child", "right_child")

    def __init__(self) -> None:
        self.name: str = ""
        self.parent: int | None = None
        self.left_child: int | None = None
        self.right_child: int | None = None


def _parse_newick_species_tree(path: str) -> tuple[list[_SpNode], int]:
    """Parse a Newick species tree into a flat node list.

    Returns ``(nodes, root_idx)`` where each node exposes ``name``, ``parent``,
    ``left_child``, ``right_child`` (indices into ``nodes``, or ``None``).
    Branch lengths and support values are ignored. The traversal order matches
    a recursive-descent parse, so leaf/internal node names are identical to what
    AleRax sees when it parses the same file.
    """
    text = pathlib.Path(path).read_text().strip()
    nodes: list[_SpNode] = []

    def _new() -> int:
        nodes.append(_SpNode())
        return len(nodes) - 1

    pos = 0  # cursor into text

    def parse_clade() -> int:
        nonlocal pos
        idx = _new()
        if pos < len(text) and text[pos] == "(":
            pos += 1  # consume '('
            children: list[int] = []
            while True:
                child = parse_clade()
                nodes[child].parent = idx
                children.append(child)
                if pos < len(text) and text[pos] == ",":
                    pos += 1
                    continue
                if pos < len(text) and text[pos] == ")":
                    pos += 1
                    break
                break
            if len(children) >= 1:
                nodes[idx].left_child = children[0]
            if len(children) >= 2:
                nodes[idx].right_child = children[1]
        # node label: up to ':' (branch length) or a structural char
        start = pos
        while pos < len(text) and text[pos] not in ":,();":
            pos += 1
        nodes[idx].name = text[start:pos].strip()
        # skip an optional ':branch_length'
        if pos < len(text) and text[pos] == ":":
            pos += 1
            while pos < len(text) and text[pos] not in ",();":
                pos += 1
        return idx

    root = parse_clade()
    return nodes, root


# ──────────────────────────────────────────────────────────────────────
# AleRax label replication
# ──────────────────────────────────────────────────────────────────────


def _is_numeric(s: str) -> bool:
    if not s:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _alerax_label_map(species_tree_path: str) -> dict[str, str]:
    """Build a ``{newick_name → alerax_label}`` mapping for every species
    tree node.

    Replicates ``PLLRootedTree::ensureUniqueLabels`` (see
    ``extra/AleRax_modified/ext/GeneRaxCore/src/trees/PLLRootedTree.cpp:309-339``):

    - Leaves keep their Newick label as-is.
    - Internal nodes: keep the Newick label only if it is non-empty,
      not duplicated, and not purely numeric. Otherwise the label is
      replaced with ``Node_<leftLeaf>_<rightLeaf>_<unique_id>``, where
      ``leftLeaf``/``rightLeaf`` are propagated from the left and right
      children in post-order.

    Pure-Python Newick parse (both gpurec and AleRax parse the same file, so
    node names match exactly for nodes that survive the renaming pass).
    """
    nodes, root = _parse_newick_species_tree(species_tree_path)
    n = len(nodes)

    # Iterative post-order to avoid recursion-depth issues on big trees.
    post_order: list[int] = []
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        idx, processed = stack.pop()
        if processed:
            post_order.append(idx)
            continue
        stack.append((idx, True))
        node = nodes[idx]
        if node.right_child is not None:
            stack.append((node.right_child, False))
        if node.left_child is not None:
            stack.append((node.left_child, False))

    any_leaf_label: list[str] = [""] * n
    seen: set[str] = set()
    name_to_label: dict[str, str] = {}

    def _get_unique(seen_set: set[str], base: str) -> str:
        i = 0
        while True:
            candidate = f"{base}_{i}"
            if candidate not in seen_set:
                return candidate
            i += 1

    for idx in post_order:
        node = nodes[idx]
        if node.left_child is None:
            # Leaf
            any_leaf_label[idx] = node.name
            name_to_label[node.name] = node.name
            seen.add(node.name)
            continue
        any_leaf_label[idx] = any_leaf_label[node.left_child]
        label = node.name or ""
        if (not label) or (label in seen) or _is_numeric(label):
            base = (
                f"Node_{any_leaf_label[node.left_child]}_"
                f"{any_leaf_label[node.right_child]}"
            )
            label = _get_unique(seen, base)
        # Map the *original* Newick name (which gpurec uses) to the
        # AleRax-renamed label. For leaves the original == renamed.
        original = node.name if node.name else label
        name_to_label[original] = label
        seen.add(label)

    return name_to_label


# ──────────────────────────────────────────────────────────────────────
# Rate-file writers
# ──────────────────────────────────────────────────────────────────────


def _write_specieswise_rates_dir(
    rates: torch.Tensor,
    species_names: list[str],
    species_tree_path: str,
    target_dir: pathlib.Path,
) -> None:
    """Write ``model_parameters.txt`` for AleRax PER-SPECIES mode.

    Parameters
    ----------
    rates : Tensor [S, 3]
        Natural-space ``[D, L, T]`` rates per species tree node, in the
        same order as ``species_names``.
    species_names : list[str]
        gpurec species names (from ``species_helpers['names']``).
    species_tree_path : str
        Path to the Newick file — used to compute AleRax-style labels.
    target_dir : Path
        Output directory; ``<target_dir>/model_parameters.txt`` is
        created.
    """
    if rates.ndim != 2 or rates.shape[1] != 3:
        raise ValueError(
            f"specieswise rates must have shape [S, 3], got {tuple(rates.shape)}"
        )
    if rates.shape[0] != len(species_names):
        raise ValueError(
            f"rates rows ({rates.shape[0]}) do not match species count "
            f"({len(species_names)})"
        )
    name_to_label = _alerax_label_map(species_tree_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "model_parameters.txt"
    rates_cpu = rates.detach().cpu()
    with open(out_path, "w") as f:
        f.write("# node D L T\n")
        for s, name in enumerate(species_names):
            label = name_to_label.get(name)
            if label is None:
                raise KeyError(
                    f"gpurec species name '{name}' has no matching AleRax "
                    f"label (the species tree at {species_tree_path} may "
                    f"have changed since this dataset was built)"
                )
            d, l, t = (float(rates_cpu[s, i]) for i in range(3))
            f.write(f"{label} {d:.10g} {l:.10g} {t:.10g}\n")


def _write_genewise_rates_dir(
    rates: torch.Tensor,
    gene_tree_paths: list[str],
    target_dir: pathlib.Path,
) -> None:
    """Write per-family ``<family>_rates.txt`` files for AleRax PER-FAMILY mode.

    Family names match the file stems of ``gene_tree_paths`` (which is how the
    families file below names them).
    """
    if rates.ndim != 2 or rates.shape[1] != 3:
        raise ValueError(
            f"genewise rates must have shape [G, 3], got {tuple(rates.shape)}"
        )
    if rates.shape[0] != len(gene_tree_paths):
        raise ValueError(
            f"rates rows ({rates.shape[0]}) do not match family count "
            f"({len(gene_tree_paths)})"
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    rates_cpu = rates.detach().cpu()
    for g, gpath in enumerate(gene_tree_paths):
        family_name = pathlib.Path(gpath).stem
        out_path = target_dir / f"{family_name}_rates.txt"
        d, l, t = (float(rates_cpu[g, i]) for i in range(3))
        with open(out_path, "w") as f:
            f.write("# D L T\n")
            f.write(f"{d:.10g} {l:.10g} {t:.10g}\n")


# ──────────────────────────────────────────────────────────────────────
# AleRax invocation (pure-Python subprocess; replaces rustree.reconcile_with_alerax)
# ──────────────────────────────────────────────────────────────────────


def _write_families_file(gene_tree_paths: list[str], path: pathlib.Path) -> list[str]:
    """Write an AleRax ``[FAMILIES]`` file. Returns the family names used."""
    names: list[str] = []
    with open(path, "w") as f:
        f.write("[FAMILIES]\n")
        for gpath in gene_tree_paths:
            name = pathlib.Path(gpath).stem
            names.append(name)
            f.write(f"- {name}\n")
            f.write(f"gene_tree = {pathlib.Path(gpath).resolve()}\n")
    return names


def _parse_rates_row(path: pathlib.Path) -> tuple[float, float, float] | None:
    """Parse the first numeric ``D L T`` (or ``label D L T``) row from an AleRax
    rates file. Mirrors ``parse_rates_file`` / ``parse_global_rates_file``."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        toks = t.split()
        # drop a leading non-numeric node label if present
        if toks and not _is_numeric(toks[0]):
            toks = toks[1:]
        nums = [float(x) for x in toks if _is_numeric(x)]
        if len(nums) >= 3:
            return (nums[0], nums[1], nums[2])
    return None


def _run_alerax(
    *,
    species_tree_path: str,
    gene_tree_paths: list[str],
    output_dir: pathlib.Path,
    model: str,
    num_samples: int,
    seed: int | None,
    fix_rates: bool,
    d: float | None = None,
    l: float | None = None,
    t: float | None = None,
    starting_rates_file: str | None = None,
    gene_tree_rooting: str | None = None,
    alerax_path: str = "alerax",
) -> dict[str, Any]:
    """Run AleRax in pure-sampling mode via subprocess.

    Faithful to the previous Rust invocation: writes a ``[FAMILIES]`` file and
    calls ``alerax -s <species> -f <families> -p <out> --model-parametrization
    <model> --gene-tree-samples <n> [--seed] [--d/--l/--t |
    --starting-rates-file] [--fix-rates] [--species-tree-search SKIP]``.

    Returns ``{"output_dir", "families": {name: {"rates": (d,l,t)|None}}}``. The
    sampled reconciliations are written by AleRax under ``output_dir`` (the rich
    tree-object parsing the Rust crate did is not reimplemented; read the files
    under ``output_dir`` directly if you need the scenarios).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    families_file = output_dir / "families.txt"
    family_names = _write_families_file(gene_tree_paths, families_file)

    cmd: list[str] = [
        alerax_path,
        "-s", str(pathlib.Path(species_tree_path).resolve()),
        "-f", str(families_file),
        "-p", str(output_dir),
        "--model-parametrization", model,
        "--gene-tree-samples", str(num_samples),
    ]
    if gene_tree_rooting is not None:
        cmd += ["--gene-tree-rooting", gene_tree_rooting]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if d is not None:
        cmd += ["--d", repr(d)]
    if l is not None:
        cmd += ["--l", repr(l)]
    if t is not None:
        cmd += ["--t", repr(t)]
    if starting_rates_file is not None:
        cmd += ["--starting-rates-file", str(starting_rates_file)]
    if fix_rates:
        cmd += ["--fix-rates"]
    # Fixed rates => never search the species tree topology (would invalidate
    # per-branch rates); matches the prior behaviour.
    if fix_rates or starting_rates_file is not None:
        cmd += ["--species-tree-search", "SKIP"]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:  # alerax not installed / not on PATH
        raise RuntimeError(
            f"AleRax binary {alerax_path!r} not found. Install it (e.g. "
            f"`conda install -c bioconda alerax`) or pass alerax_path=..."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"AleRax failed (exit {exc.returncode}).\n"
            f"cmd: {' '.join(cmd)}\nstderr:\n{exc.stderr}"
        ) from exc

    # Parse the rates AleRax wrote back (per-family or shared).
    model_params = output_dir / "model_parameters"
    families: dict[str, Any] = {}
    for name in family_names:
        per_fam = model_params / f"{name}_rates.txt"
        rates = _parse_rates_row(per_fam)
        if rates is None:
            rates = _parse_rates_row(model_params / "model_parameters.txt")
        families[name] = {"rates": rates}

    return {"output_dir": str(output_dir), "families": families}


# ──────────────────────────────────────────────────────────────────────
# Public entry point (called from GeneReconModel.sample_reconciliations)
# ──────────────────────────────────────────────────────────────────────


def sample_reconciliations(
    model: Any,
    *,
    num_samples: int = 100,
    output_dir: str | None = None,
    seed: int | None = None,
    keep_output: bool = False,
    alerax_path: str = "alerax",
) -> dict[str, Any]:
    """Run AleRax in pure-sampling mode using ``model``'s optimized rates.

    Supports the three single-axis modes:

    - ``global``      : passes ``--d/--l/--t --fix-rates``.
    - ``specieswise`` : writes ``model_parameters.txt`` with AleRax-style
      labels and passes ``--starting-rates-file --fix-rates``.
    - ``genewise``    : writes per-family ``<family>_rates.txt`` files
      and passes ``--starting-rates-file --fix-rates``.

    The combined ``genewise + specieswise`` mode and ``pairwise``
    transfers have no AleRax equivalent and raise
    :class:`NotImplementedError`.

    Returns ``{"output_dir", "families": {name: {"rates": (d,l,t)|None}}}``;
    the sampled reconciliation files live under the AleRax output directory.
    """
    ds = model._dataset
    if ds.pairwise:
        raise NotImplementedError(
            "sample_reconciliations does not support pairwise transfer mode."
        )
    if ds.genewise and ds.specieswise:
        raise NotImplementedError(
            "sample_reconciliations does not support combined genewise + "
            "specieswise mode (AleRax has no per-family-per-species rates "
            "storage)."
        )

    rates = model.rates  # natural space; layout matches theta

    # Resolve output directory: if user passed one, write rates files there
    # so they can be inspected after the run.
    tmp_holder: tempfile.TemporaryDirectory | None = None
    if output_dir is None:
        tmp_holder = tempfile.TemporaryDirectory(prefix="gpurec_alerax_")
        out_root = pathlib.Path(tmp_holder.name)
    else:
        out_root = pathlib.Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

    rates_dir = out_root / "starting_rates"
    alerax_run_dir = out_root / "alerax_run"

    common: dict[str, Any] = dict(
        species_tree_path=str(ds.species_tree_path),
        gene_tree_paths=[str(p) for p in ds.gene_tree_paths],
        output_dir=alerax_run_dir,
        num_samples=num_samples,
        seed=seed,
        alerax_path=alerax_path,
        fix_rates=True,
    )

    mode = model._mode
    if mode == "global":
        d, l_, t = (float(x) for x in rates.detach().cpu().tolist())
        common.update(model="GLOBAL", d=d, l=l_, t=t)
    elif mode == "specieswise":
        _write_specieswise_rates_dir(
            rates,
            list(ds.species_helpers["names"]),
            str(ds.species_tree_path),
            rates_dir,
        )
        common.update(model="PER-SPECIES", starting_rates_file=str(rates_dir))
    elif mode == "genewise":
        _write_genewise_rates_dir(
            rates,
            [str(p) for p in ds.gene_tree_paths],
            rates_dir,
        )
        common.update(model="PER-FAMILY", starting_rates_file=str(rates_dir))
    else:
        raise NotImplementedError(f"unsupported mode: {mode!r}")

    try:
        return _run_alerax(**common)
    finally:
        if tmp_holder is not None and not keep_output:
            tmp_holder.cleanup()
