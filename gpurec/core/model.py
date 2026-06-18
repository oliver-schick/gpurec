import math
import os
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from .extract_parameters import extract_parameters, extract_parameters_uniform
from .likelihood import E_fixed_point, compute_log_likelihood
from .forward import Pi_wave_forward
from .batching import (
    build_wave_layout,
    collate_gene_families,
    collate_wave,
    split_phase_waves,
)
from .scheduling import compute_clade_waves
from .preprocess_cpp import _load_extension as _load_species_gene_ext


class GeneDataset(Dataset):
    def __init__(
        self,
        species_tree_path,
        gene_tree_paths,
        genewise, # whether to have a different theta for each gene family
        specieswise, # no need for genewise: it only appear when collating or gradient steps
        pairwise, # changes the size of theta
        dtype=torch.float32,
        device=None,
        preprocess_cache_dir: str | os.PathLike | None = None,
        refresh_preprocess_cache: bool = False,
        fraction_missing: torch.Tensor | None = None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.genewise = genewise
        self.specieswise = specieswise
        self.pairwise = pairwise
        self.device = device
        self.dtype = dtype
        ext = _load_species_gene_ext()

        use_single_preprocess = os.environ.get("GPUREC_PREPROCESS_MODE", "").lower() == "single"
        family_names = [f"family_{i:06d}" for i in range(len(gene_tree_paths))]
        if use_single_preprocess:
            raw_by_family = {
                name: ext.preprocess(species_tree_path, [str(path)])
                for name, path in zip(family_names, gene_tree_paths)
            }
            self.species_helpers = raw_by_family[family_names[0]]['species']
        elif preprocess_cache_dir is not None:
            self.species_helpers, raw_by_family = self._preprocess_with_cache(
                ext,
                species_tree_path,
                gene_tree_paths,
                family_names,
                preprocess_cache_dir=preprocess_cache_dir,
                refresh=refresh_preprocess_cache,
            )
        else:
            families_input = {
                name: [str(path)]
                for name, path in zip(family_names, gene_tree_paths)
            }
            raw_all = ext.preprocess_multiple_families(
                species_tree_path,
                families_input,
            )
            raw_by_family = raw_all['families']
            self.species_helpers = raw_all['species']
        self.tr_mat_unnormalized = torch.log2(self.species_helpers["Recipients_mat"])
        self.unnorm_row_max = self.tr_mat_unnormalized.max(dim=-1).values  # [S], precomputed
        self.S = int(self.species_helpers['S'])

        # Fraction-missing (per-leaf p_obs) leaf boundary. `fraction_missing` is an
        # optional linear-space [S] vector with fraction_missing_l per species node
        # (0 for fully-observed leaves and internal nodes). We derive a single log2
        # tensor self.leaf_E = log2(1 - p_obs_l) = log2(fraction_missing_l) at missing
        # species-leaves, -inf elsewhere. It serves BOTH the E extinction boundary and
        # the Pi leaf boundary (identical), and is what callers pass to
        # optimize_theta_wave(..., leaf_E=...). None when fraction_missing is None.
        # species-tree leaf mask: nodes that are never internal parents
        sP = self.species_helpers['s_P_indexes']
        internal = sP[sP < self.S].unique()
        leaf_species_mask = torch.ones(self.S, dtype=torch.bool, device=sP.device)
        leaf_species_mask[internal] = False
        self.leaf_species_mask = leaf_species_mask  # [S] bool
        if fraction_missing is None:
            self.leaf_E = None
        else:
            fm = fraction_missing.to(device=leaf_species_mask.device, dtype=dtype)
            missing = leaf_species_mask & (fm > 0)
            self.leaf_E = torch.where(
                missing,
                torch.log2(fm.clamp_min(torch.finfo(dtype).tiny)),
                torch.full_like(fm, float("-inf")),
            )

        # creating an initial theta (log2-space: rates = 2^theta)
        _THETA_INIT = math.log2(1e-10)
        if pairwise:
            if specieswise:
                raise ValueError("specieswise and pairwise are mutually exclusive")
            # Pairwise: theta has D and L only; T is implicit in the transfer matrix
            theta = _THETA_INIT * torch.ones(2, dtype=dtype, device=device)
            self.tr_mat_unnormalized = self.tr_mat_unnormalized - 10.0
        elif specieswise:
            theta = _THETA_INIT * torch.ones(self.S, 3, dtype=dtype, device=device)
        else:
            theta = _THETA_INIT * torch.ones(3, dtype=dtype, device=device)

        _INV_LN2 = 1.0 / math.log(2.0)
        families = []
        for i, (gpath, family_name) in enumerate(zip(gene_tree_paths, family_names)):
            raw = raw_by_family[family_name]
            ccp = raw['ccp']
            # Convert log_split_probs from ln (C++ output) to log2
            ccp['log_split_probs_sorted'] = ccp['log_split_probs_sorted'] * _INV_LN2
            families.append({
                'ccp_helpers': ccp,
                'root_clade_id': int(ccp['root_clade_id']),
                'leaf_row_index': raw['leaf_row_index'],
                'leaf_col_index': raw['leaf_col_index'],
                'C': int(ccp['C']),
                'N_splits': int(ccp['N_splits']),
                'theta': theta.clone(),
                'transfer_mat_unnormalized': self.tr_mat_unnormalized,
                'log_split_probs': ccp['log_split_probs_sorted'],
            })
        # stored on CPU. Only move when computing likelihood and optimizing.
        self.families = families
        self.gene_tree_paths = list(gene_tree_paths)
        self.species_tree_path = species_tree_path

        self.num_families = len(families)

    @staticmethod
    def _hash_file(path: str | os.PathLike) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def _preprocess_with_cache(
        cls,
        ext,
        species_tree_path,
        gene_tree_paths,
        family_names,
        *,
        preprocess_cache_dir: str | os.PathLike,
        refresh: bool,
    ):
        cache_dir = Path(preprocess_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        version = "light-v1"
        species_hash = cls._hash_file(species_tree_path)
        species_key = hashlib.sha256(
            f"{version}:species:{species_hash}".encode("utf-8")
        ).hexdigest()
        species_cache = cache_dir / f"species-{species_key}.pt"

        species_helpers = None
        if species_cache.exists() and not refresh:
            species_helpers = torch.load(
                species_cache,
                map_location="cpu",
                weights_only=False,
            )

        raw_by_family = {}
        missing = {}
        family_cache_paths = {}
        for name, path in zip(family_names, gene_tree_paths):
            gene_hash = cls._hash_file(path)
            family_key = hashlib.sha256(
                f"{version}:family:{species_hash}:{gene_hash}".encode("utf-8")
            ).hexdigest()
            cache_path = cache_dir / f"family-{family_key}.pt"
            family_cache_paths[name] = cache_path
            if cache_path.exists() and not refresh:
                raw_by_family[name] = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
            else:
                missing[name] = [str(path)]

        if missing:
            raw_all = ext.preprocess_multiple_families(
                species_tree_path,
                missing,
            )
            if species_helpers is None:
                species_helpers = raw_all["species"]
                torch.save(species_helpers, species_cache)

            for name, raw in raw_all["families"].items():
                raw_by_family[name] = raw
                torch.save(raw, family_cache_paths[name])

        if species_helpers is None:
            raw_species = ext.preprocess_multiple_families(
                species_tree_path,
                {},
            )
            species_helpers = raw_species["species"]
            torch.save(species_helpers, species_cache)

        return species_helpers, raw_by_family
    
    def __len__(self):
        return len(self.families)
    
    def __getitem__(self, idx):
        # Will work for a single sample, but will need
        # custom collate_fn to work with batches
        return self.families[idx]
    
    def change_dtype(self, dtype):
        """We may want to change dtype at the very end of optimization."""
        self.dtype = dtype
        for fam in self.families:
            fam['ccp_helpers'] = {k: v.to(dtype=dtype) if torch.is_tensor(v) else v for k, v in fam['ccp_helpers'].items()}
            fam['theta'] = fam['theta'].to(dtype=dtype)
        self.tr_mat_unnormalized = self.tr_mat_unnormalized.to(dtype=dtype)
        self.species_helpers = {k: v.to(dtype=dtype) if torch.is_tensor(v) else v for k, v in self.species_helpers.items()}

    def set_params(self, idx, D, T, L):
        # only if genewise
        theta = torch.log2(torch.tensor([D, L, T], device=self.device, dtype=self.dtype))
        self.families[idx]['theta'] = theta.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _normalize_max_transfer(max_transfer_mat: torch.Tensor) -> torch.Tensor:
        if max_transfer_mat.ndim >= 2 and max_transfer_mat.shape[-1] == 1:
            return max_transfer_mat.squeeze(-1)
        return max_transfer_mat

    def _resolve_device_dtype(
        self,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> tuple[torch.device, torch.dtype]:
        return (self.device if device is None else device,
                self.dtype if dtype is None else dtype)

    @staticmethod
    def _move_tensor(t: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if t.dtype.is_floating_point:
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)

    def _move_mapping(self, mapping: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        return {
            k: (self._move_tensor(v, device=device, dtype=dtype) if torch.is_tensor(v) else v)
            for k, v in mapping.items()
        }

    def _species_helpers_for_mode(
        self,
        *,
        pibar_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        skip_keys = set()
        if pibar_mode == 'uniform':
            skip_keys = {'Recipients_mat'}

        species_helpers = {
            k: (self._move_tensor(v, device=device, dtype=dtype) if torch.is_tensor(v) and k not in skip_keys else v)
            for k, v in self.species_helpers.items()
        }

        ancestors_T = None
        if pibar_mode == 'uniform':
            ancestors_T = species_helpers['ancestors_dense'].T.to_sparse_coo()
        return species_helpers, ancestors_T

    def _extract_single_params(
        self,
        fam: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        theta = fam['theta'].to(device=device, dtype=dtype)
        transfer_mat_unnorm = self.tr_mat_unnormalized.to(device=device, dtype=dtype)
        log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
            theta,
            transfer_mat_unnorm,
            genewise=self.genewise,
            specieswise=self.specieswise,
            pairwise=self.pairwise,
        )
        return log_pS, log_pD, log_pL, transfer_mat, self._normalize_max_transfer(max_transfer_mat)

    def _extract_batch_params(
        self,
        indices: list[int],
        *,
        pibar_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        use_uniform_extract = (pibar_mode == 'uniform') and not self.pairwise

        if use_uniform_extract and not self.genewise:
            unnorm_row_max = self.unnorm_row_max.to(device=device, dtype=dtype)
            theta0 = self.families[indices[0]]['theta'].to(device=device, dtype=dtype)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters_uniform(
                theta0, unnorm_row_max, specieswise=self.specieswise,
            )
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat

        if use_uniform_extract and self.genewise:
            unnorm_row_max = self.unnorm_row_max.to(device=device, dtype=dtype)
            theta_stack = torch.stack([
                self.families[i]['theta'].to(device=device, dtype=dtype) for i in indices
            ], dim=0)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters_uniform(
                theta_stack, unnorm_row_max, specieswise=self.specieswise, genewise=True,
            )
            return log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat

        transfer_mat_unnorm = self.tr_mat_unnormalized.to(device=device, dtype=dtype)
        if self.genewise:
            theta_stack = torch.stack([
                self.families[i]['theta'].to(device=device, dtype=dtype) for i in indices
            ], dim=0)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
                theta_stack, transfer_mat_unnorm,
                genewise=True, specieswise=self.specieswise, pairwise=self.pairwise,
            )
        else:
            theta0 = self.families[indices[0]]['theta'].to(device=device, dtype=dtype)
            log_pS, log_pD, log_pL, transfer_mat, max_transfer_mat = extract_parameters(
                theta0, transfer_mat_unnorm,
                genewise=False, specieswise=self.specieswise, pairwise=self.pairwise,
            )

        return log_pS, log_pD, log_pL, transfer_mat, self._normalize_max_transfer(max_transfer_mat)

    def _solve_e_fixed_point(
        self,
        *,
        species_helpers: dict[str, Any],
        log_pS: torch.Tensor,
        log_pD: torch.Tensor,
        log_pL: torch.Tensor,
        transfer_mat: torch.Tensor | None,
        max_transfer_vec: torch.Tensor,
        max_iters_E: int,
        tol_E: float,
        device: torch.device,
        dtype: torch.dtype,
        pibar_mode: str = 'dense',
        ancestors_T: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        return E_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            max_iters=max_iters_E,
            tolerance=tol_E,
            warm_start_E=None,
            dtype=dtype,
            device=device,
            pibar_mode=pibar_mode,
            ancestors_T=ancestors_T,
        )

    @torch.no_grad()
    def compute_likelihood(
        self,
        idx: int = 0,
        *,
        max_iters_E: int = 2000,
        tol_E: float = 1e-6,
        max_iters_Pi: int = 2000,
        tol_Pi: float = 1e-6,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        pibar_mode: str = 'dense',
    ) -> dict:
        """Compute log-likelihood for a single family via the batched pathway.

        A single-family evaluation is treated as a batch of size 1 to ensure
        consistency between `compute_likelihood` and `compute_likelihood_batch`.
        """

        logL = self.compute_likelihood_batch(
            indices=[idx],
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            max_iters_Pi=max_iters_Pi,
            tol_Pi=tol_Pi,
            device=device,
            dtype=dtype,
            pibar_mode=pibar_mode,
        )[0]

        return {
            'log_likelihood': float(logL),
            'Pi': None,
            'E': None,
            'Ebar': None,
            'E_s1': None,
            'E_s2': None,
        }

    @torch.no_grad()
    def compute_likelihood_batch(
        self,
        indices: list[int] | None = None,
        *,
        max_iters_E: int = 2000,
        tol_E: float = 1e-12,
        max_iters_Pi: int = 2000,
        tol_Pi: float = 1e-12,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        chunk_size: int | None = None,
        max_wave_size: int | None = 32768,
        max_root_wave_size: int | None = None,
        pibar_mode: str = 'dense',
    ) -> list[float]:
        """Compute log-likelihoods for a batch of gene families.

        Returns a list of per-family log-likelihoods, in the same order as `indices`.

        Uses wave-ordered forward pass for non-genewise families,
        falls back to fixed-point for genewise.

        Args:
            chunk_size: If set, process families in chunks of this size to avoid OOM.
                Recommended: 20 for S~2000.
            max_wave_size: If set, use fixed-size cross-family wave scheduling
                with at most this many clades per wave. If ``None``, merge
                families by their per-family wave index.
            max_root_wave_size: If set with index-merged scheduling, split
                only phase-3 root waves to cap DTS scratch memory.
            pibar_mode: 'dense' (cuBLAS matmul) or 'uniform' (O(W*S) exact for scalar/specieswise T).
                Use 'uniform' for large S where the transfer matrix is nearly uniform.
        """
        device, dtype = self._resolve_device_dtype(device, dtype)

        if indices is None:
            indices = list(range(len(self.families)))
        if len(indices) == 0:
            return []

        # Handle chunking: split large batches to avoid OOM
        if chunk_size is not None and len(indices) > chunk_size:
            all_logLs = []
            for start in range(0, len(indices), chunk_size):
                chunk_indices = indices[start:start + chunk_size]
                all_logLs.extend(self.compute_likelihood_batch(
                    chunk_indices,
                    max_iters_E=max_iters_E, tol_E=tol_E,
                    max_iters_Pi=max_iters_Pi, tol_Pi=tol_Pi,
                    device=device, dtype=dtype,
                    chunk_size=chunk_size,
                    max_wave_size=max_wave_size,
                    max_root_wave_size=max_root_wave_size,
                    pibar_mode=pibar_mode,
                ))
            return all_logLs

        # Build batch items compatible with collate_gene_families
        batch_items = []
        for idx in indices:
            fam = self.families[idx]
            batch_items.append({
                'ccp': fam['ccp_helpers'],
                'leaf_row_index': fam['leaf_row_index'],
                'leaf_col_index': fam['leaf_col_index'],
                'root_clade_id': int(fam['root_clade_id']),
            })

        batched = collate_gene_families(batch_items, dtype=dtype, device=device)
        ccp_helpers = batched['ccp']
        leaf_row_index = batched['leaf_row_index']
        leaf_col_index = batched['leaf_col_index']
        root_clade_ids = batched['root_clade_ids']  # Long[F]

        log_pS, log_pD, log_pL, transfer_mat, max_transfer_vec = self._extract_batch_params(
            indices,
            pibar_mode=pibar_mode,
            device=device,
            dtype=dtype,
        )

        species_helpers, ancestors_T = self._species_helpers_for_mode(
            pibar_mode=pibar_mode,
            device=device,
            dtype=dtype,
        )

        # E fixed point (vectorized across genes when parameters are batched)
        E_out = self._solve_e_fixed_point(
            species_helpers=species_helpers,
            log_pS=log_pS,
            log_pD=log_pD,
            log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_vec=max_transfer_vec,
            max_iters_E=max_iters_E,
            tol_E=tol_E,
            device=device,
            dtype=dtype,
            pibar_mode=pibar_mode,
            ancestors_T=ancestors_T,
        )
        E = E_out['E']
        E_s1 = E_out['E_s1']
        E_s2 = E_out['E_s2']
        Ebar = E_out['E_bar']

        # Free large [S,S] tensors after E_step when using uniform modes
        if pibar_mode == 'uniform':
            transfer_mat = None

        offsets = [m['clade_offset'] for m in batched['family_meta']]
        # Wave scheduling: merge all families into cross-family waves
        families_waves = []
        families_phases = []
        for idx in indices:
            fam = self.families[idx]
            waves_i, phases_i = compute_clade_waves(fam['ccp_helpers'])
            families_waves.append(waves_i)
            families_phases.append(phases_i)

        cross_waves = collate_wave(families_waves, offsets)

        max_n_waves = max(len(p) for p in families_phases)
        cross_phases = []
        for k in range(max_n_waves):
            phase_k = 1
            for fp in families_phases:
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

        family_clade_counts = [m['C'] for m in batched['family_meta']]
        family_clade_offsets = [m['clade_offset'] for m in batched['family_meta']]

        wave_layout = build_wave_layout(
            waves=cross_waves,
            phases=cross_phases,
            ccp_helpers=ccp_helpers,
            leaf_row_index=leaf_row_index,
            leaf_col_index=leaf_col_index,
            root_clade_ids=root_clade_ids,
            device=device,
            dtype=dtype,
            family_clade_counts=family_clade_counts,
            family_clade_offsets=family_clade_offsets,
        )

        Pi_out = Pi_wave_forward(
            wave_layout=wave_layout,
            species_helpers=species_helpers,
            E=E, Ebar=Ebar, E_s1=E_s1, E_s2=E_s2,
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=transfer_mat,
            max_transfer_mat=max_transfer_vec,
            device=device, dtype=dtype,
            local_iters=max_iters_Pi,
            local_tolerance=tol_Pi,
            pibar_mode=pibar_mode,
            family_idx=wave_layout.get('family_idx') if self.genewise else None,
            return_original=False,
        )

        logL_vec = compute_log_likelihood(
            Pi_out['Pi_wave_ordered'],
            E,
            wave_layout['root_clade_ids'],
        )
        return [float(x) for x in logL_vec.detach().cpu().tolist()]
