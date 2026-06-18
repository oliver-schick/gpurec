"""High-level ``nn.Module`` for phylogenetic reconciliation.

Wraps a :class:`gpurec.core.model.GeneDataset` (used purely for preprocessing)
and exposes ``theta`` as an ``nn.Parameter`` so notebook users can use any
``torch.optim`` optimizer with the standard pattern::

    model = GeneReconModel.from_trees(
        species_tree="sp.nwk", gene_trees=["g1.nwk"], mode="global", device="cuda",
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        loss = model()              # NLL
        loss.backward()
        opt.step()
        model.clamp_theta_()
"""
from __future__ import annotations

import math
import os
from typing import Any, Optional

import torch

from gpurec.core.batching import (
    build_wave_layout,
    collate_gene_families,
    collate_wave,
    split_phase_waves,
)
from gpurec.core.model import GeneDataset
from gpurec.core.scheduling import compute_clade_waves

from .autograd import ReconStaticState, _GeneReconFunction, _apply_to_static
from .modes import _default_theta_init, _mode_to_flags


def _build_static_state(
    dataset: GeneDataset,
    *,
    pibar_mode: str,
    max_iters_E: int,
    tol_E: float,
    max_iters_Pi: int,
    tol_Pi: float,
    fixed_iters_Pi: Optional[int],
    neumann_terms: int,
    use_pruning: bool,
    pruning_threshold: float,
    cg_tol: float,
    cg_maxiter: int,
    gmres_restart: int,
    max_wave_size: Optional[int] = 32768,
    max_root_wave_size: Optional[int] = None,
) -> ReconStaticState:
    """Absorb the wave-layout boilerplate that lives in
    ``experiments/validate_three_modes.py:100-149`` and
    ``GeneDataset.compute_likelihood_batch:393-428``.

    Builds a single cross-family wave layout for the entire dataset and
    moves species helpers (and ``ancestors_T`` for uniform mode) onto the
    target device. The result is cached on the model and reused across
    every ``forward()`` call.
    """
    device = dataset.device
    dtype = dataset.dtype

    # 1. Cross-family wave layout
    items = [
        {
            "ccp": fam["ccp_helpers"],
            "leaf_row_index": fam["leaf_row_index"],
            "leaf_col_index": fam["leaf_col_index"],
            "root_clade_id": int(fam["root_clade_id"]),
        }
        for fam in dataset.families
    ]
    batched = collate_gene_families(items, dtype=dtype, device=device)

    fams_waves: list = []
    fams_phases: list = []
    for fam in dataset.families:
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

    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=None,
        max_wave_size=max_wave_size,
    )
    cross_waves, cross_phases = split_phase_waves(
        cross_waves,
        cross_phases,
        phase=3,
        max_wave_size=max_root_wave_size,
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

    # 2. Species helpers on device. Skip Recipients_mat for uniform mode
    #    (mirrors GeneDataset._species_helpers_for_mode lines 131-150).
    species_helpers, ancestors_T = dataset._species_helpers_for_mode(
        pibar_mode=pibar_mode, device=device, dtype=dtype,
    )

    # 3. Other static tensors
    unnorm_row_max = dataset.unnorm_row_max.to(device=device, dtype=dtype)
    transfer_mat_unnormalized = (
        dataset.tr_mat_unnormalized.to(device=device, dtype=dtype)
        if pibar_mode in ("dense", "topk")
        else None
    )

    return ReconStaticState(
        device=device,
        dtype=dtype,
        wave_layout=wave_layout,
        species_helpers=species_helpers,
        root_clade_ids=batched["root_clade_ids"],
        unnorm_row_max=unnorm_row_max,
        transfer_mat_unnormalized=transfer_mat_unnormalized,
        ancestors_T=ancestors_T,
        genewise=bool(dataset.genewise),
        specieswise=bool(dataset.specieswise),
        pibar_mode=pibar_mode,
        max_iters_E=max_iters_E,
        tol_E=tol_E,
        max_iters_Pi=max_iters_Pi,
        tol_Pi=tol_Pi,
        fixed_iters_Pi=fixed_iters_Pi,
        neumann_terms=neumann_terms,
        use_pruning=use_pruning,
        pruning_threshold=pruning_threshold,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        gmres_restart=gmres_restart,
    )


class GeneReconModel(torch.nn.Module):
    """A ``nn.Module`` view over a :class:`GeneDataset`.

    ``forward()`` returns the negative log-likelihood as a differentiable
    scalar. ``theta`` is registered as an ``nn.Parameter`` so any
    ``torch.optim`` optimizer can be used directly.
    """

    def __init__(
        self,
        *,
        dataset: GeneDataset,
        mode: str,
        pibar_mode: str = "uniform",
        max_iters_E: int = 2000,
        tol_E: float = 1e-8,
        max_iters_Pi: int = 2000,
        tol_Pi: float = 1e-6,
        fixed_iters_Pi: Optional[int] = 6,
        neumann_terms: int = 3,
        use_pruning: bool = True,
        pruning_threshold: float = 1e-6,
        cg_tol: float = 1e-8,
        cg_maxiter: int = 500,
        gmres_restart: int = 40,
        theta_init: Optional[torch.Tensor] = None,
        max_wave_size: Optional[int] = 32768,
        max_root_wave_size: Optional[int] = None,
    ):
        super().__init__()
        if dataset.pairwise:
            raise NotImplementedError(
                "GeneReconModel does not support pairwise transfer mode. "
                "Use optimize_theta_wave directly with a pairwise dataset."
            )
        # Validate mode early
        _mode_to_flags(mode)

        # Sanity check: dataset flags must be consistent with mode
        ds_g, ds_sw, _ = (dataset.genewise, dataset.specieswise, dataset.pairwise)
        expected_g, expected_sw, _ = _mode_to_flags(mode)
        if (ds_g, ds_sw) != (expected_g, expected_sw):
            raise ValueError(
                f"Dataset flags (genewise={ds_g}, specieswise={ds_sw}) do not "
                f"match requested mode {mode!r} "
                f"(expected genewise={expected_g}, specieswise={expected_sw}). "
                "Construct GeneDataset with matching flags or use "
                "GeneReconModel.from_trees()."
            )

        self._mode = mode
        self._dataset = dataset

        if theta_init is None:
            theta_init = _default_theta_init(dataset, mode)
        self.theta = torch.nn.Parameter(theta_init.clone())

        self._static = _build_static_state(
            dataset,
            pibar_mode=pibar_mode,
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            max_iters_Pi=max_iters_Pi,
            tol_Pi=tol_Pi,
            fixed_iters_Pi=fixed_iters_Pi,
            neumann_terms=neumann_terms,
            use_pruning=use_pruning,
            pruning_threshold=pruning_threshold,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter,
            gmres_restart=gmres_restart,
            max_wave_size=max_wave_size,
            max_root_wave_size=max_root_wave_size,
        )

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────
    @classmethod
    def from_trees(
        cls,
        species_tree: str,
        gene_trees: list[str],
        *,
        mode: str = "global",
        pibar_mode: str = "uniform",
        device: Any = "cuda",
        dtype: torch.dtype = torch.float32,
        theta_init_rates: Optional[tuple[float, float, float]] = None,
        preprocess_cache_dir: str | os.PathLike | None = None,
        refresh_preprocess_cache: bool = False,
        **solver_kwargs,
    ) -> "GeneReconModel":
        """One-liner: Newick paths → ready-to-optimize model.

        Parameters
        ----------
        species_tree : str
            Path to the species tree (Newick).
        gene_trees : list[str]
            Paths to gene trees (Newick).
        mode : str
            "global" | "specieswise" | "genewise".
        pibar_mode : str
            "uniform" (default, fast for nearly-uniform transfer matrices) or
            "dense" (full ``Pi @ T.T`` matmul).
        device : str | torch.device
            Target device. Defaults to ``"cuda"``.
        dtype : torch.dtype
            Floating-point dtype. ``torch.float32`` is the default; switch to
            ``torch.float64`` if optimization stalls due to precision.
        theta_init_rates : (D, L, T) | None
            Optional natural-space initial rates. If ``None``, the dataset
            default of ``log2(1e-10)`` is used (matching GeneDataset).
        preprocess_cache_dir : str | os.PathLike | None
            Optional directory for CPU preprocessing cache files. Reusing the
            same cache avoids reparsing/rebuilding unchanged gene trees.
        refresh_preprocess_cache : bool
            Ignore existing preprocessing cache entries and overwrite them.
        """
        genewise, specieswise, pairwise = _mode_to_flags(mode)
        if isinstance(device, str):
            device = torch.device(device)
        ds = GeneDataset(
            species_tree_path=species_tree,
            gene_tree_paths=gene_trees,
            genewise=genewise,
            specieswise=specieswise,
            pairwise=pairwise,
            dtype=dtype,
            device=device,
            preprocess_cache_dir=preprocess_cache_dir,
            refresh_preprocess_cache=refresh_preprocess_cache,
        )
        theta_init = None
        if theta_init_rates is not None:
            D, L, T = theta_init_rates
            base = torch.log2(
                torch.tensor([D, L, T], dtype=dtype, device=device)
            )
            if mode == "specieswise":
                theta_init = base.unsqueeze(0).expand(int(ds.S), -1).clone()
            elif mode == "genewise":
                theta_init = (
                    base.unsqueeze(0).expand(len(gene_trees), -1).clone()
                )
            else:
                theta_init = base
        return cls(
            dataset=ds,
            mode=mode,
            pibar_mode=pibar_mode,
            theta_init=theta_init,
            **solver_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────
    # Likelihood / loss
    # ──────────────────────────────────────────────────────────────────
    def forward(self, reduce: str = "sum") -> torch.Tensor:
        """Returns negative log-likelihood (a loss).

        ``reduce="sum"`` (default) returns a scalar (sum over families).
        ``reduce="per_family"`` returns a ``[G]`` vector and is only valid in
        genewise mode.
        """
        return _GeneReconFunction.apply(self.theta, self._static, reduce)

    def nll(self) -> torch.Tensor:
        """Alias for ``self()``."""
        return self.forward(reduce="sum")

    def nll_per_family(self) -> torch.Tensor:
        """Per-family NLL ``[G]``. Only valid in genewise mode."""
        if self._mode != "genewise":
            raise ValueError(
                "nll_per_family() is only valid in genewise mode; in global / "
                "specieswise mode all families share theta, so independent "
                "per-family gradients are not defined."
            )
        return self.forward(reduce="per_family")

    @torch.no_grad()
    def log_likelihood(self) -> float:
        """Inference helper: returns ``+log_likelihood`` (Python float)."""
        return float(-self.forward(reduce="sum").item())

    # ──────────────────────────────────────────────────────────────────
    # Reconciliation sampling via AleRax
    # ──────────────────────────────────────────────────────────────────
    def sample_reconciliations(
        self,
        *,
        num_samples: int = 100,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
        keep_output: bool = False,
        alerax_path: str = "alerax",
    ) -> dict:
        """Sample reconciliation scenarios from this model's optimized rates.

        Calls AleRax with ``--fix-rates`` so the optimized DTL rates flow
        through to the sampler unchanged. Supports ``global``,
        ``specieswise``, and ``genewise`` modes (combined
        ``genewise+specieswise`` and ``pairwise`` raise
        :class:`NotImplementedError`).

        Returns a ``dict`` ``{"output_dir": ..., "families": {name:
        {"rates": (D, L, T) | None}}}``; the sampled reconciliation files are
        written under the AleRax output directory. (Pure-Python AleRax bridge,
        no Rust.)
        """
        from .sampling import sample_reconciliations as _impl
        return _impl(
            self,
            num_samples=num_samples,
            output_dir=output_dir,
            seed=seed,
            keep_output=keep_output,
            alerax_path=alerax_path,
        )

    # ──────────────────────────────────────────────────────────────────
    # Parameter management
    # ──────────────────────────────────────────────────────────────────
    def clamp_theta_(self, min_rate: float = 1e-10) -> None:
        """In-place safety floor on theta to prevent rate underflow.

        Matches the clamp applied between Adam steps inside
        ``optimize_theta_wave`` (see ``wave_optimizer.py:531``).
        """
        with torch.no_grad():
            self.theta.clamp_(min=math.log2(min_rate))

    @property
    def rates(self) -> torch.Tensor:
        """Natural-space rates: ``2^theta``. Shape mirrors ``self.theta``."""
        return torch.exp2(self.theta.detach())

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def n_families(self) -> int:
        return len(self._dataset.families)

    @property
    def n_species(self) -> int:
        return int(self._dataset.S)

    @property
    def static(self) -> ReconStaticState:
        """Read-only access to the cached static state (for advanced use)."""
        return self._static

    # ──────────────────────────────────────────────────────────────────
    # Device / dtype handling
    # ──────────────────────────────────────────────────────────────────
    def _apply(self, fn):
        """Override so that ``.to(device)`` / ``.to(dtype)`` walks the
        non-Parameter tensors held inside ``self._static`` (wave_layout,
        species_helpers, etc.)."""
        super()._apply(fn)
        self._static = _apply_to_static(self._static, fn)
        # Reset warm-start cache when moving (its old device/dtype is stale)
        self._static.warm_E = None
        return self
