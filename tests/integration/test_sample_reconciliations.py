"""End-to-end tests for ``GeneReconModel.sample_reconciliations``.

Verifies the gpurec → AleRax sampling pipeline for global, specieswise,
and genewise modes:

1. The Python helper writes the right files (rates dir layout, AleRax
   labels for specieswise).
2. AleRax accepts the rates via ``--d/--l/--t`` (global) or
   ``--starting-rates-file`` (specieswise / genewise) and runs sampling.
3. The output reconciliation samples carry the expected rates.

Tests skip if the modified AleRax binary is not available.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest
import torch

from gpurec import GeneReconModel


_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _ROOT.parent
DATA_DIR = _ROOT / "data" / "test_trees_20"
N_FAMILIES = 5

# Prefer the modified binary because the system alerax (if any) does not
# carry the --starting-rates-file flag this test exercises.
_MODIFIED = _PROJECT_ROOT / "extra" / "AleRax_modified" / "build" / "bin" / "alerax"
ALERAX_BINARY = str(_MODIFIED) if _MODIFIED.exists() else (shutil.which("alerax") or "")


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_rates(model: GeneReconModel, rates_natural: torch.Tensor) -> None:
    """Overwrite ``model.theta`` so ``model.rates == rates_natural``."""
    target = torch.log2(
        rates_natural.to(dtype=model.theta.dtype, device=model.theta.device)
    )
    with torch.no_grad():
        model.theta.copy_(target)


@pytest.fixture(scope="module")
def trees():
    if not DATA_DIR.exists():
        pytest.skip("test_trees_20 dataset not present")
    if not Path(ALERAX_BINARY).exists():
        pytest.skip(f"AleRax binary not found at {ALERAX_BINARY}")
    sp = str(DATA_DIR / "sp.nwk")
    genes = sorted(DATA_DIR.glob("g_*.nwk"))[:N_FAMILIES]
    if len(genes) < N_FAMILIES:
        pytest.skip(f"Need {N_FAMILIES} gene families")
    return sp, [str(p) for p in genes]


# ──────────────────────────────────────────────────────────────────────
# Global mode
# ──────────────────────────────────────────────────────────────────────


def test_sample_reconciliations_global(trees):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode="global",
        device=_device(),
    )
    _set_rates(
        model,
        torch.tensor([0.05, 0.07, 0.03], dtype=model.theta.dtype),
    )

    results = model.sample_reconciliations(
        num_samples=4, seed=42, alerax_path=ALERAX_BINARY
    )

    fams = results["families"]
    assert len(fams) == N_FAMILIES, f"got {len(fams)} families"
    # Every family should report the same scalar rates that gpurec set,
    # because --fix-rates disables AleRax's optimizer.
    for name, r in fams.items():
        d, l, t = r["rates"]
        assert math.isclose(d, 0.05, rel_tol=1e-3), f"{name}: D={d}"
        assert math.isclose(l, 0.07, rel_tol=1e-3), f"{name}: L={l}"
        assert math.isclose(t, 0.03, rel_tol=1e-3), f"{name}: T={t}"


# ──────────────────────────────────────────────────────────────────────
# Genewise mode
# ──────────────────────────────────────────────────────────────────────


def test_sample_reconciliations_genewise(trees):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode="genewise",
        device=_device(),
    )
    # Distinct (D, L, T) per family.
    rates = torch.tensor(
        [
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
            [0.07, 0.08, 0.09],
            [0.10, 0.11, 0.12],
            [0.13, 0.14, 0.15],
        ],
        dtype=model.theta.dtype,
    )
    _set_rates(model, rates)

    results = model.sample_reconciliations(
        num_samples=3, seed=42, alerax_path=ALERAX_BINARY
    )

    fams = results["families"]
    assert len(fams) == N_FAMILIES
    # Family names map to gene tree file stems (g_0000, g_0001, ...).
    for g, gpath in enumerate(genes):
        family_name = Path(gpath).stem
        rd, rl, rt = fams[family_name]["rates"]
        d, l, t = (float(rates[g, i]) for i in range(3))
        assert math.isclose(rd, d, rel_tol=1e-3), f"{family_name}: D={rd}, expected {d}"
        assert math.isclose(rl, l, rel_tol=1e-3), f"{family_name}: L={rl}, expected {l}"
        assert math.isclose(rt, t, rel_tol=1e-3), f"{family_name}: T={rt}, expected {t}"


# ──────────────────────────────────────────────────────────────────────
# Specieswise mode
# ──────────────────────────────────────────────────────────────────────


def test_sample_reconciliations_specieswise(trees, tmp_path):
    sp, genes = trees
    model = GeneReconModel.from_trees(
        species_tree=sp,
        gene_trees=genes,
        mode="specieswise",
        device=_device(),
    )
    S = model.n_species
    # Per-species rates: most are uniform, but row 0 is distinct so we
    # can verify that per-species variation actually flows through.
    base = torch.tensor([0.01, 0.02, 0.03], dtype=model.theta.dtype)
    rates = base.unsqueeze(0).expand(S, 3).clone()
    rates[0] = torch.tensor([0.50, 0.60, 0.70], dtype=model.theta.dtype)
    _set_rates(model, rates)

    out_dir = tmp_path / "specieswise_run"
    results = model.sample_reconciliations(
        num_samples=3,
        seed=42,
        alerax_path=ALERAX_BINARY,
        output_dir=str(out_dir),
        keep_output=True,
    )

    fams = results["families"]
    assert len(fams) == N_FAMILIES
    # AleRax writes per-species rates; we read the first numeric row, which
    # (in gpurec's species order) is index 0 — the one we set to (0.5,0.6,0.7).
    d, l, t = next(iter(fams.values()))["rates"]
    assert math.isclose(d, 0.5, rel_tol=1e-3)
    assert math.isclose(l, 0.6, rel_tol=1e-3)
    assert math.isclose(t, 0.7, rel_tol=1e-3)

    # Strong invariant: AleRax's output rates file (after sampling)
    # must equal the input rates file we wrote, modulo number formatting.
    in_path = out_dir / "starting_rates" / "model_parameters.txt"
    out_path = (
        out_dir / "alerax_run" / "alerax_output" / "model_parameters" / "model_parameters.txt"
    )
    assert in_path.exists()
    assert out_path.exists()

    def _row_map(path: Path) -> dict[str, tuple[float, float, float]]:
        out: dict[str, tuple[float, float, float]] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("node"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                out[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
        return out

    in_rows = _row_map(in_path)
    out_rows = _row_map(out_path)
    assert set(in_rows.keys()) == set(out_rows.keys()), (
        "AleRax output uses different node labels than the input file"
    )
    for label in in_rows:
        in_d, in_l, in_t = in_rows[label]
        out_d, out_l, out_t = out_rows[label]
        # AleRax prints rates with default iostream precision (~6 sig figs);
        # gpurec writes with 10 sig figs. Compare with a generous tolerance.
        assert math.isclose(in_d, out_d, rel_tol=1e-5, abs_tol=1e-12), (
            f"D differs at {label}: in={in_d}, out={out_d}"
        )
        assert math.isclose(in_l, out_l, rel_tol=1e-5, abs_tol=1e-12), (
            f"L differs at {label}: in={in_l}, out={out_l}"
        )
        assert math.isclose(in_t, out_t, rel_tol=1e-5, abs_tol=1e-12), (
            f"T differs at {label}: in={in_t}, out={out_t}"
        )


# ──────────────────────────────────────────────────────────────────────
# Mode-rejection guards
# ──────────────────────────────────────────────────────────────────────


def test_sample_reconciliations_rejects_pairwise(trees):
    sp, genes = trees
    # The high-level GeneReconModel.from_trees doesn't expose pairwise,
    # so we construct a pairwise GeneDataset and check the helper.
    from gpurec.core.model import GeneDataset

    ds = GeneDataset(
        species_tree_path=sp,
        gene_tree_paths=genes,
        genewise=False,
        specieswise=False,
        pairwise=True,
        dtype=torch.float32,
        device="cpu",
    )

    class _StubModel:
        _dataset = ds
        _mode = "global"
        rates = torch.tensor([0.05, 0.07, 0.03])

    from gpurec.api.sampling import sample_reconciliations

    with pytest.raises(NotImplementedError, match="pairwise"):
        sample_reconciliations(_StubModel(), num_samples=1, alerax_path=ALERAX_BINARY)
