"""Wave-based optimizer (uses wave forward + wave backward)."""
from __future__ import annotations

import math
import time
from typing import List

import torch

from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
from gpurec.core.forward import Pi_wave_forward
from gpurec.core.extract_parameters import extract_parameters, extract_parameters_uniform

from .types import FixedPointInfo, LinearSolveStats, StepRecord
from .implicit_grad import implicit_grad_loglik_vjp_wave
from gpurec import distributed as _ddp


def _cuda_mem_diag(device) -> str:
    """Best-effort CUDA memory diagnostics string."""
    try:
        if not torch.cuda.is_available():
            return "cuda_unavailable"
        idx = device.index if hasattr(device, "index") and device.index is not None else torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(idx)
        alloc_b = torch.cuda.memory_allocated(idx)
        reserved_b = torch.cuda.memory_reserved(idx)
        return (
            f"free={free_b / (1024**3):.2f}GiB total={total_b / (1024**3):.2f}GiB "
            f"allocated={alloc_b / (1024**3):.2f}GiB reserved={reserved_b / (1024**3):.2f}GiB"
        )
    except Exception as exc:
        return f"mem_diag_unavailable({exc})"


class _PiLeafUnset:
    """Sentinel: caller did NOT pass ``leaf_obs_log`` at all.

    Lets ``optimize_theta_wave`` distinguish two cases that both want a Python
    ``None`` at the call boundary:

    * ``leaf_obs_log`` unset (this sentinel) -> default it to ``leaf_E`` so legacy
      callers that pass only ``leaf_E`` keep the "both" (supplement) behavior,
      byte-identical to before.
    * ``leaf_obs_log=None`` passed explicitly -> Pi leaf boundary OFF (the
      AleRax-faithful "e-only" mode), independent of ``leaf_E``.
    """


_PI_LEAF_UNSET = _PiLeafUnset()


def optimize_theta_wave(
    wave_layout,
    species_helpers,
    root_clade_ids,
    unnorm_row_max,
    theta_init,
    *,
    transfer_mat_unnormalized=None,
    steps: int = 200,
    lr: float = 0.2,
    tol_theta: float = 1e-3,
    e_max_iters: int = 2000,
    e_tol: float = 1e-8,
    neumann_terms: int = 4,
    use_pruning: bool = True,
    pruning_threshold: float = 1e-6,
    cg_tol: float = 1e-8,
    cg_maxiter: int = 500,
    gmres_restart: int = 40,
    specieswise: bool = False,
    device=None,
    dtype=torch.float64,
    pibar_mode: str = 'uniform',
    family_batch_size: int = 0,
    wave_layout_batches=None,
    families=None,
    stochastic_batches: bool = False,
    stochastic_seed: int = 0,
    optimizer: str = 'adam',
    momentum: float = 0.9,
    verbose: bool = False,
    leaf_E: torch.Tensor | None = None,
    leaf_obs_log=_PI_LEAF_UNSET,
    brownian_sigma=None,
    brownian_root_sigma: float = 5.0,
    brownian_mu=None,
    dtl_tv_lambda: float = 0.0,
    dtl_tv_eps: float = 1e-3,
    parent_index: torch.Tensor | None = None,
    theta_bounds=None,
    origination: str = 'uniform',
    omega_init: torch.Tensor | None = None,
    origination_sigma=None,
    origination_root_sigma: float = 5.0,
    origination_mu: float = 0.0,
    origination_decouple_root: bool = True,
    origination_l2: float = 0.0,
    group_index: torch.Tensor | None = None,
    omega_group_index: torch.Tensor | None = None,
    origination_depth: torch.Tensor | None = None,
    origination_depth_lambda: float = 0.0,
    origination_root_index: int | None = None,
    origination_root_lambda: float = 0.0,
    origination_dirichlet_c: float = 0.0,
    origination_vertical_pi: torch.Tensor | None = None,
    origination_barrier_kind: str = "meanlog",
    origination_floor: float = 0.0,
    distributed: bool = False,
    grad_reduction: str = "sum",   # Σ over families (correct for L-BFGS + DDP). "mean"
                                   # (legacy /n_batch_accum) is WRONG for L-BFGS — it makes
                                   # the grad inconsistent with the summed NLL -> line-search
                                   # stall. Only matters when family_batch_size>0.
    ddp_device=None,
):
    """Optimize theta using wave forward/backward + implicit gradient.

    Parameters
    ----------
    wave_layout : dict
        From build_wave_layout().
    species_helpers : dict
        Species tree helpers.
    root_clade_ids : Tensor
        Root clade IDs (original ordering) for likelihood computation.
    unnorm_row_max : Tensor [S]
        Row maxima of unnormalized transfer matrix.
    theta_init : Tensor [3] or [S, 3]
        Initial rate parameters (log-space).
    steps : int
        Maximum optimization iterations (ignored for 'lbfgs').
    lr : float
        Learning rate (ignored for 'lbfgs').
    optimizer : str
        'adam', 'sgd', or 'lbfgs'.
    family_batch_size : int
        Optional family mini-batch size for global/specieswise gradient accumulation.
        Effective when ``wave_layout_batches`` is provided by caller.
    wave_layout_batches : list[tuple[dict, Tensor]] | None
        Optional prebuilt batches of ``(wave_layout_batch, root_clade_ids_batch)``.
        When provided, forward/backward runs batch-by-batch and accumulates NLL/gradient.
    families : list[dict] | None
        Optional original family dicts. When provided with ``family_batch_size > 0``
        and ``wave_layout_batches is None``, batch wave layouts are built lazily per step.
    stochastic_batches : bool
        If True (and optimizer='sgd'), sample a random family batch each step
        instead of accumulating all family batches.
    stochastic_seed : int
        RNG seed used for stochastic family-batch sampling.
    momentum : float
        Momentum for SGD (default 0.9, ignored for adam/lbfgs).
    leaf_E : Tensor [S] | None
        Optional fraction-missing boundary, log2(1 - p_obs_l) = log2(fraction
        missing) at missing species-leaves and -inf elsewhere, applied to the E
        (extinction) solve ONLY: passed as leaf_E to E_fixed_point and to the E
        adjoint E_steps. This is the AleRax ``_fm`` term (the "S but not observed"
        extinction contribution). None (default) -> every gene observed in the E
        recursion (byte-identical to before).
    leaf_obs_log : Tensor [S] | None
        Optional fraction-missing boundary applied to the Pi (reconciliation) leaf
        boundary ONLY: passed as leaf_obs_log to Pi_wave_forward / Pi_wave_backward.
        This is the EXTRA ``(1-sigma)(1-p_obs)`` Pi baseline from the AleRax
        *supplement* (AleRaxSupp.tex L159) which the AleRax *source* does NOT apply.
        Decoupled from ``leaf_E`` so an AleRax-faithful run can keep fraction-missing
        in E while dropping it from Pi.

        Backward-compat: if ``leaf_obs_log`` is None but ``leaf_E`` is set, it
        defaults to ``leaf_E`` so callers passing only ``leaf_E`` get the previous
        "both" (supplement) behavior, byte-identical to before. To get the
        AleRax-faithful "E-only" mode, pass ``leaf_obs_log=None`` explicitly via the
        sentinel (see ``_PiLeafUnset`` below) — i.e. set ``leaf_E`` and leave the Pi
        boundary off. The driver exposes this through ``--fm-mode e-only``.
    brownian_sigma : float | Tensor [3] | None
        Std (log2 units) of the time-uniform TKP Brownian rate prior coupling
        adjacent branches' log-rates. None (default) disables the prior and keeps
        the path byte-identical to before. Scalar or per-axis [3] (D, L, T).
        Only applies in specieswise mode (theta [S,3]); ignored otherwise.
    brownian_root_sigma : float | Tensor [3]
        Std (log2 units) of the root-anchor term (propriety). Default 5.0.
    brownian_mu : float | Tensor [3] | None
        Root-anchor centre (log2 units). None (default) -> use the theta_init
        root row (i.e. log2(init_rate)) so the prior is centred at the init.
    parent_index : LongTensor [S] | None
        Parent of each species node (root maps to -1/self), from
        ``species_parent_index(species_helpers)``. Required when brownian_sigma
        is not None.
    theta_bounds : tuple | None
        Optional box constraints (log2 units) on theta for the L-BFGS-B path,
        AleRax-style. ``(lo, hi)`` scalars applied to every element, or a tuple
        of two arrays broadcastable to the flattened theta. None (default) keeps
        the current behavior (lower bound = log2(1e-10), no upper bound).
    origination : str
        ORIGINATION strategy (matching AleRax). 'uniform' (default): every species
        branch originates with equal probability p^O_e = 1/S (omega absent;
        byte-identical to before). 'optimize': a FREE per-branch origination
        log2-weight ``omega`` [S] is jointly optimised with theta. The per-branch
        origination probability is the softmax p^O_e = 2^omega_e / Σ_e 2^omega_e
        (log_pO = omega - logsumexp2(omega)), which enters compute_log_likelihood's
        numerator AND survival denominator. Only the L-BFGS path supports
        'optimize': the scipy vector becomes [theta_flat (3S or S*3); omega (S)],
        the implicit theta gradient uses the origination-weighted seeds, and the
        omega gradient is the DIRECT ∂(Σ NLL)/∂omega at fixed Pi, E.
    omega_init : Tensor [S] | None
        Initial origination log2-weights (only used when origination='optimize').
        None (default) -> omega = 0 (uniform p^O_e = 1/S).
    origination_l2 : float
        L2 / ridge regularization strength on the origination logits ``omega``:
        adds ``lambda * sum_e omega_e^2`` to the loss (grad ``2*lambda*omega``),
        shrinking omega toward 0 i.e. p^O toward UNIFORM (1/S). This is the
        APPROPRIATE regularizer for origination, since p^O_e = softmax(omega) is
        a probability DISTRIBUTION over branches (sum_e p^O_e = 1), not a set of
        independent per-branch rates. 0.0 (default) -> FREE omega (no reg). Only
        used when origination='optimize'.
    origination_sigma, origination_root_sigma, origination_mu,
    origination_decouple_root :
        DEPRECATED / no-ops. These configured a tree-structured Brownian prior on
        omega, which is inappropriate for a softmax probability distribution
        (it smooths adjacent log-weights as if they were independent rates). Use
        ``origination_l2`` instead. Kept only for call-signature stability.

    Returns
    -------
    dict with 'theta', 'rates', 'log_likelihood', 'negative_log_likelihood', 'history'.
    When origination='optimize', also 'omega' [S] (log2-weights) and 'origination'
    [S] (normalised p^O_e).
    """
    if device is None:
        device = theta_init.device

    # Resolve the Pi leaf boundary. Backward-compat: if the caller did NOT pass
    # ``leaf_obs_log`` at all, default it to ``leaf_E`` so the legacy single-tensor
    # path (fraction-missing in BOTH E and Pi, the supplement behavior) is
    # byte-identical to before. An explicit ``leaf_obs_log=None`` turns the Pi leaf
    # boundary OFF (AleRax-faithful "e-only"), independent of ``leaf_E``.
    if isinstance(leaf_obs_log, _PiLeafUnset):
        leaf_obs_log = leaf_E

    _THETA_MIN = math.log2(1e-10)

    # --- CLADE-GROUPED (rate-category) reparametrization --------------------
    # When ``group_index`` [S] (long, values 0..G-1) is given, theta is NOT a
    # free [S,3] per-branch tensor: instead G free rate-category rows theta_g
    # [G,3] are optimised and EXPANDED to per-branch via theta[s] = theta_g[
    # group_index[s]]. The gradient is REDUCED back per group by summing the
    # per-branch grad over each category (chain rule of the expand map). This
    # replicates AleRax's "branch wise" model, which is NOT 119 free branches
    # but ~17 named-clade rate categories (the model_parameters file). Only the
    # L-BFGS path supports grouping; group_index=None (default) is byte-identical
    # to the per-branch / global behavior. Requires specieswise theta [S,3].
    _use_groups = group_index is not None
    _group_index_t = None
    _n_groups = 0
    if _use_groups:
        if not (specieswise and theta_init.ndim == 2 and theta_init.shape[-1] == 3):
            raise ValueError(
                "group_index requires specieswise theta [S,3]; "
                f"got specieswise={specieswise}, theta_init.shape={tuple(theta_init.shape)}"
            )
        _group_index_t = group_index.to(device=device, dtype=torch.long).reshape(-1)
        if _group_index_t.numel() != theta_init.shape[0]:
            raise ValueError(
                f"group_index must have S={theta_init.shape[0]} entries, "
                f"got {_group_index_t.numel()}"
            )
        if int(_group_index_t.min().item()) < 0:
            raise ValueError("group_index entries must be >= 0")
        _n_groups = int(_group_index_t.max().item()) + 1

    def _expand_theta(theta_param):
        """[G,3] group rows -> [S,3] per-branch (no-op when not grouped)."""
        if not _use_groups:
            return theta_param
        return theta_param.index_select(0, _group_index_t)

    def _reduce_grad(grad_full):
        """[S,3] per-branch grad -> [G,3] per-group grad (no-op when not grouped)."""
        if not _use_groups:
            return grad_full
        g = torch.zeros((_n_groups, grad_full.shape[1]),
                        dtype=grad_full.dtype, device=grad_full.device)
        g.index_add_(0, _group_index_t, grad_full)
        return g

    # --- ORIGINATION setup --------------------------------------------------
    # 'uniform' (default): omega absent, log_pO=None everywhere -> byte-identical
    #   to the pre-origination path (uniform p^O_e = 1/S).
    # 'optimize': a free per-branch origination log2-weight omega [S] is jointly
    #   optimised; log_pO = omega - logsumexp2(omega) is the softmax log-prob.
    from gpurec.core.likelihood import origination_log_pO as _origination_log_pO
    if origination not in ('uniform', 'optimize', 'fixed'):
        raise ValueError(f"origination must be 'uniform', 'optimize' or 'fixed', got {origination!r}")
    _opt_origination = origination == 'optimize'
    # 'fixed': origination is held at a GIVEN per-branch distribution (e.g. the
    # paper's structured {DPANN,Eury,TackA,rest} O) while ONLY theta (D/L/T) is
    # optimised. omega is NOT a parameter; log_pO is a constant in the forward.
    # Tests whether the DPANN transfer inflation (T trading against a collapsed
    # free O) disappears once O can't move. omega_init supplies the fixed logits.
    _fixed_origination = origination == 'fixed'
    _fixed_log_pO = None
    if _fixed_origination:
        if omega_init is None:
            raise ValueError("origination='fixed' requires omega_init (the fixed "
                             "per-branch origination log2-weights)")
        _fixed_omega_t = omega_init.to(device=device, dtype=dtype).reshape(-1).clone()
        _fixed_log_pO = _origination_log_pO(_fixed_omega_t)   # constant softmax log-prob
    # ORIGINATION FLOOR (anti-collapse): mix a fraction alpha of the origination
    # mass back to UNIFORM, p^O = (1-alpha)*softmax(omega) + alpha/S, so no branch
    # can be starved to p^O~0 (which forces transfer to substitute for origination
    # and inflates deep-clade genomes, the FULLbasin/DPANN pathology). alpha=0 ->
    # pure softmax (byte-identical). Only meaningful with origination='optimize'.
    _orig_floor = float(origination_floor) if (origination == 'optimize' and origination_floor) else 0.0
    if not (0.0 <= _orig_floor < 1.0):
        raise ValueError(f"origination_floor must be in [0,1), got {origination_floor!r}")

    def _orig_log_pO(omega):
        """log2 p^O with the optional anti-collapse uniform floor (differentiable)."""
        lp = _origination_log_pO(omega)                 # log2 softmax(omega)
        if _orig_floor <= 0.0:
            return lp
        S_ = omega.shape[-1]
        p = (1.0 - _orig_floor) * torch.exp2(lp) + _orig_floor / S_
        return torch.log2(p)
    # Determine S (number of species branches) for the omega vector.
    _S_branches = int(species_helpers['S'])

    # --- GROUPED ORIGINATION (per-clade O category) -------------------------
    # When ``omega_group_index`` [S] (long, 0..G_O-1) is given with
    # origination='optimize', omega is NOT a free [S] per-branch logit vector:
    # instead G_O free origination-category logits are optimised and EXPANDED to
    # per-branch via omega[s] = omega_g[omega_group_index[s]]; the omega gradient
    # is REDUCED back per category by summation. This replicates AleRax's DTLO
    # parametrization (e.g. DPANN/Eury/TackA each own O, all others share one),
    # the per-branch origination DISTRIBUTION that lifts the true-root clades
    # above the small-genome attraction artefact. None (default) -> free omega.
    _use_omega_groups = (omega_group_index is not None) and _opt_origination
    _omega_group_index_t = None
    _n_omega_groups = 0
    if _use_omega_groups:
        _omega_group_index_t = omega_group_index.to(device=device, dtype=torch.long).reshape(-1)
        if _omega_group_index_t.numel() != _S_branches:
            raise ValueError(
                f"omega_group_index must have S={_S_branches} entries, "
                f"got {_omega_group_index_t.numel()}"
            )
        if int(_omega_group_index_t.min().item()) < 0:
            raise ValueError("omega_group_index entries must be >= 0")
        _n_omega_groups = int(_omega_group_index_t.max().item()) + 1

    def _expand_omega(omega_param):
        """[G_O] category logits -> [S] per-branch (no-op when not grouped)."""
        if not _use_omega_groups:
            return omega_param
        return omega_param.index_select(0, _omega_group_index_t)

    def _reduce_omega(grad_full):
        """[S] per-branch omega grad -> [G_O] per-category (no-op when not grouped)."""
        if not _use_omega_groups:
            return grad_full
        g = torch.zeros(_n_omega_groups, dtype=grad_full.dtype, device=grad_full.device)
        g.index_add_(0, _omega_group_index_t, grad_full)
        return g

    # Size of the omega block in the scipy vector (categories if grouped).
    _n_omega_param = _n_omega_groups if _use_omega_groups else _S_branches

    if _opt_origination:
        if omega_init is None:
            _omega_full0 = torch.zeros(_S_branches, dtype=dtype, device=device)
        else:
            _omega_full0 = omega_init.to(device=device, dtype=dtype).reshape(-1).clone()
        if _omega_full0.numel() != _S_branches:
            raise ValueError(
                f"omega_init must have S={_S_branches} entries, got {_omega_full0.numel()}"
            )
        if _use_omega_groups:
            # Reduce the per-branch init to per-category logits (mean over members).
            _cnt = torch.zeros(_n_omega_groups, dtype=dtype, device=device)
            _cnt.index_add_(0, _omega_group_index_t, torch.ones_like(_omega_full0))
            _osum = torch.zeros(_n_omega_groups, dtype=dtype, device=device)
            _osum.index_add_(0, _omega_group_index_t, _omega_full0)
            _omega0 = _osum / _cnt.clamp(min=1)
        else:
            _omega0 = _omega_full0
        if optimizer != 'lbfgs':
            raise NotImplementedError(
                "origination='optimize' is only supported with optimizer='lbfgs'"
            )
    else:
        _omega0 = None

    # --- Brownian (TKP) rate prior setup -----------------------------------
    # Only meaningful for specieswise theta [S,3]; skip otherwise so the
    # global/genewise paths stay byte-identical.
    _use_prior = brownian_sigma is not None
    _use_tv = float(dtl_tv_lambda) > 0.0          # fused-lasso / total-variation prior
    _specieswise_ok = specieswise and theta_init.ndim == 2 and theta_init.shape[-1] == 3
    if _use_prior and not _specieswise_ok:
        if verbose:
            print("  [brownian prior requested but theta is not specieswise [S,3]; "
                  "skipping prior]", flush=True)
        _use_prior = False
    if _use_tv and not _specieswise_ok:
        if verbose:
            print("  [TV prior requested but theta is not specieswise [S,3]; skipping]",
                  flush=True)
        _use_tv = False

    # Origination regularization: L2 (ridge) on the softmax logits.
    # The origination weights p^O_e = softmax(omega) form a PROBABILITY
    # DISTRIBUTION over branches (sum_e p^O_e = 1), so a tree-structured Brownian
    # prior on omega is NOT appropriate (it would smooth adjacent log-weights as
    # if they were independent per-branch rates). The right regularizer is L2 /
    # ridge on the logits: lambda * sum_e omega_e^2, which shrinks omega toward 0,
    # i.e. p^O toward UNIFORM (1/S). origination_l2 = 0 -> FREE omega (no reg).
    # (The legacy origination_sigma/_root_sigma/_mu/_decouple_root args are
    # accepted for signature stability but no longer used.)
    _orig_l2 = float(origination_l2) if (_opt_origination and origination_l2) else 0.0
    _use_orig_l2 = _orig_l2 > 0.0

    # VERTICAL-EVOLUTION origination prior: penalize the EXPECTED origination
    # DEPTH, E_{p^O}[depth] = sum_e p^O_e * depth_e, where depth_e is the number
    # of edges from the root to branch e (root = 0). This pulls origination mass
    # toward the root (vertical inheritance) and penalizes the below-root
    # loss-saving concentration that the small-genome-attraction artefact
    # exploits. Penalty lambda * E_{p^O}[depth]; gradient (base-2 softmax)
    # lambda * ln2 * p^O_e * (depth_e - E_{p^O}[depth]). Only for free per-branch
    # omega (not grouped). lambda selected by evidence (empirical Bayes).
    _depth = None
    _depth_lambda = float(origination_depth_lambda) if _opt_origination else 0.0
    _use_depth = (origination_depth is not None) and _depth_lambda > 0.0
    if _use_depth:
        if _use_omega_groups:
            raise ValueError(
                "origination_depth prior is for FREE per-branch omega; not "
                "compatible with omega_group_index (grouped origination).")
        _depth = origination_depth.to(device=device, dtype=dtype).reshape(-1)
        if _depth.numel() != _S_branches:
            raise ValueError(
                f"origination_depth must have S={_S_branches} entries, got {_depth.numel()}")

    # ROOT-MASS origination prior: penalize (1 - p^O_root), a FLAT tax on all
    # non-root origination mass (depth-agnostic), unlike the progressive depth
    # prior which taxes deep mass ever harder and collapses p^O onto the root.
    # This rewards SOME root origination (the vertical signal) while leaving the
    # shape of the remaining mass (e.g. AleRax's deep tail) free. pen = lambda *
    # (1 - p^O_root); grad (base-2 softmax) = lambda * ln2 * p_root * (p_e - 1{e=root}).
    _root_idx = None
    _root_lambda = float(origination_root_lambda) if _opt_origination else 0.0
    _use_root = (origination_root_index is not None) and _root_lambda > 0.0
    if _use_root:
        if _use_omega_groups:
            raise ValueError(
                "origination_root prior is for FREE per-branch omega; not "
                "compatible with omega_group_index (grouped origination).")
        _root_idx = int(origination_root_index)
        if not (0 <= _root_idx < _S_branches):
            raise ValueError(
                f"origination_root_index={_root_idx} out of range [0,{_S_branches})")

    # ASYMMETRIC DIRICHLET (vertical-evolution) prior on the origination
    # distribution p^O. Penalty = c * KL(pi || p^O) (up to a const) = -c * sum_e
    # pi_e log2 p^O_e, i.e. the Dirichlet(alpha_e = 1 + c*pi_e) log-density. pi is
    # the per-branch VERTICAL profile (root-concentrated, sum=1, strictly > 0): it
    # (i) shrinks p^O toward root-concentrated origination (strong biological prior),
    # and (ii) FORBIDS the degenerate vertex (KL -> inf as any p^O_e -> 0 where
    # pi_e > 0). Grad (base-2 logits) = c * (p^O_e - pi_e). Per-branch, applied
    # BEFORE the omega-group reduction, so it composes with grouped origination.
    _dir_c = float(origination_dirichlet_c) if _opt_origination else 0.0
    _dir_pi = None
    _barrier_kind = str(origination_barrier_kind).lower()
    if _barrier_kind not in ("meanlog", "simpson", "renyi2"):
        raise ValueError(f"origination_barrier_kind must be meanlog|simpson|renyi2, "
                         f"got {origination_barrier_kind!r}")
    _use_dirichlet = (origination_vertical_pi is not None) and _dir_c > 0.0
    if _use_dirichlet:
        _dir_pi = origination_vertical_pi.to(device=device, dtype=dtype).reshape(-1)
        if _dir_pi.numel() != _S_branches:
            raise ValueError(
                f"origination_vertical_pi must have S={_S_branches} entries, "
                f"got {_dir_pi.numel()}")
        if float(_dir_pi.min().item()) <= 0.0:
            raise ValueError("origination_vertical_pi must be strictly positive "
                             "(else the anti-degeneracy barrier vanishes there)")
        _dir_pi = _dir_pi / _dir_pi.sum()                 # normalize to a distribution

    _brownian_prior = None
    _tv_prior = None
    _prior_parent_index = None
    _prior_sigma = None
    _prior_root_sigma = None
    _prior_mu = None
    if _use_prior or _use_tv:
        from gpurec.core.tree_prior import species_parent_index as _species_parent_index
        if parent_index is None:
            _prior_parent_index = _species_parent_index(species_helpers).to(device)
        else:
            _prior_parent_index = parent_index.to(device=device, dtype=torch.long)
    if _use_tv:
        from gpurec.core.tree_prior import tv_log_prior_and_grad as _tv_prior
        if verbose:
            print(f"  [TV/fused-lasso DTL prior: lambda={dtl_tv_lambda} eps={dtl_tv_eps} "
                  f"-> piecewise-constant per-branch rates]", flush=True)
    if _use_prior:
        from gpurec.core.tree_prior import brownian_log_prior_and_grad as _brownian_prior
        _prior_sigma = brownian_sigma
        _prior_root_sigma = brownian_root_sigma
        # Default root-anchor centre = theta_init root row (== log2(init_rate)
        # for a uniform init), so the prior is centred at the initialization.
        if brownian_mu is None:
            _ti = theta_init.to(device=device, dtype=dtype)
            arange_S = torch.arange(_ti.shape[0], device=device)
            _is_root = (_prior_parent_index < 0) | (_prior_parent_index == arange_S)
            _root_ids = torch.nonzero(_is_root, as_tuple=False).flatten()
            _prior_mu = _ti[int(_root_ids[0].item())].clone()
        else:
            _prior_mu = brownian_mu

    def _apply_prior(theta_d, nll, grad_theta):
        """Add the DTL rate penalties (Brownian and/or TV/fused-lasso) to nll and
        their gradients to grad_theta."""
        if _use_prior:
            penalty, p_grad = _brownian_prior(
                theta_d, _prior_parent_index, _prior_sigma, _prior_root_sigma, _prior_mu,
            )
            nll, grad_theta = nll + float(penalty.item()), grad_theta + p_grad
        if _use_tv:
            pen_tv, g_tv = _tv_prior(
                theta_d, _prior_parent_index, dtl_tv_lambda, dtl_tv_eps,
            )
            nll, grad_theta = nll + float(pen_tv.item()), grad_theta + g_tv
        return nll, grad_theta

    def _apply_origination_prior(omega_d, nll, grad_omega):
        """L2 (ridge) penalty on the origination logits: lambda * sum(omega^2).

        Shrinks p^O toward uniform; lambda=0 leaves omega FREE. (Brownian is
        inappropriate -- omega is a softmax distribution, not per-branch rates.)"""
        if not _use_orig_l2 or omega_d is None or grad_omega is None:
            return nll, grad_omega
        pen = _orig_l2 * float((omega_d * omega_d).sum().item())
        return nll + pen, grad_omega + (2.0 * _orig_l2) * omega_d

    def _apply_origination_depth(omega_d, nll, grad_omega):
        """Vertical-evolution prior: penalize E_{p^O}[depth] (pull origination to
        the root). pen = lambda * sum_e p^O_e depth_e; grad (base-2 softmax) =
        lambda * ln2 * p^O_e (depth_e - E_{p^O}[depth])."""
        if not _use_depth or omega_d is None or grad_omega is None:
            return nll, grad_omega
        log_pO = _orig_log_pO(omega_d)          # omega - logsumexp2(omega)
        p = torch.exp2(log_pO)                         # [S] p^O
        Ed = (p * _depth).sum()
        pen = _depth_lambda * float(Ed.item())
        grad = (_depth_lambda * math.log(2.0)) * p * (_depth - Ed)
        return nll + pen, grad_omega + grad

    def _apply_origination_root(omega_d, nll, grad_omega):
        """Root-mass prior: penalize (1 - p^O_root). pen = lambda*(1 - p_root);
        grad (base-2 softmax) = lambda * ln2 * p_root * (p_e - 1{e=root}). Rewards
        root origination without taxing the depth-distribution of the rest."""
        if not _use_root or omega_d is None or grad_omega is None:
            return nll, grad_omega
        log_pO = _orig_log_pO(omega_d)          # [S]
        p = torch.exp2(log_pO)                         # [S] p^O
        p_root = p[_root_idx]
        pen = _root_lambda * float((1.0 - p_root).item())
        scale = _root_lambda * math.log(2.0) * p_root  # 0-dim tensor
        grad = scale * p                               # [S]
        grad[_root_idx] = grad[_root_idx] - scale
        return nll + pen, grad_omega + grad

    def _apply_origination_dirichlet(omega_d, nll, grad_omega):
        """Anti-concentration barrier on the origination distribution p^O. Three kinds:

        - 'meanlog' (reverse KL, default): pen = -c * sum_e pi_e log2 p^O_e
          (= Dirichlet(1+c*pi) log-density = c*KL(pi||p^O)). Grad = c (p^O - pi).
          NB this divergence is dominated by NEAR-ZERO p^O_e (it -> inf as any
          p^O_e -> 0), so it over-penalizes a *structured* p^O (real backbone mass,
          ~zero on tips) more than a bland near-uniform one -- it fights the wrong
          tail and rewards the SGA-friendly bland distribution.

        - 'simpson' (Renyi-2 / inverse-participation): pen = c * sum_e (p^O_e)^2.
          PEAK-weighted (dominated by the large masses), tolerant of structured
          zeros. Grad (base-2 logits): d pen/d omega_e = 2 c ln2 p_e (p_e - P2),
          P2 = sum_e p_e^2. This is the divergence whose inverse (1/P2 = effective
          #branches) cleanly separates the deep cluster from the SGA roots.

        - 'renyi2': pen = c * log2(sum_e p^O_e^2) = -c * H2 (Renyi-2 entropy, bits);
          scale-invariant version of simpson. Grad = 2 c p_e (p_e - P2)/P2.

        Operates on the EXPANDED per-branch omega [S]; the caller reduces to groups."""
        if not _use_dirichlet or omega_d is None or grad_omega is None:
            return nll, grad_omega
        log_pO = _orig_log_pO(omega_d)          # [S], log2 p^O
        p = torch.exp2(log_pO)                         # [S]
        if _barrier_kind == "meanlog":
            pen = -_dir_c * float((_dir_pi * log_pO).sum().item())
            grad = _dir_c * (p - _dir_pi)              # [S]
        else:
            P2 = (p * p).sum()                         # Simpson concentration (scalar)
            if _barrier_kind == "simpson":
                pen = _dir_c * float(P2.item())
                grad = (2.0 * _dir_c * math.log(2.0)) * p * (p - P2)
            else:  # renyi2
                pen = _dir_c * float(torch.log2(P2).item())
                grad = (2.0 * _dir_c) * p * (p - P2) / P2
        return nll + pen, grad_omega + grad

    # Precompute ancestors_T for uniform mode
    _ancestors_T = None
    if pibar_mode == 'uniform':
        anc_dense = species_helpers['ancestors_dense'].to(device=device, dtype=dtype)
        _ancestors_T = anc_dense.T.to_sparse_coo()

    _lazy_batch_ranges = None
    _lazy_batch_builder = None
    if wave_layout_batches is None and families is not None and family_batch_size and family_batch_size > 0:
        from gpurec.core.batching import collate_gene_families, collate_wave, build_wave_layout
        from gpurec.core.scheduling import compute_clade_waves

        fam_items = []
        fam_waves = []
        fam_phases = []
        for fam in families:
            fam_items.append({
                'ccp': fam['ccp_helpers'],
                'leaf_row_index': fam['leaf_row_index'],
                'leaf_col_index': fam['leaf_col_index'],
                'root_clade_id': int(fam['root_clade_id']),
            })
            w, p = compute_clade_waves(fam['ccp_helpers'])
            fam_waves.append(w)
            fam_phases.append(p)

        n_f = len(fam_items)
        _lazy_batch_ranges = [
            (i, min(i + int(family_batch_size), n_f))
            for i in range(0, n_f, int(family_batch_size))
        ]

        def _build_lazy_batch(batch_id: int):
            i, j = _lazy_batch_ranges[batch_id]
            sub_items = fam_items[i:j]
            sub_waves = fam_waves[i:j]
            sub_phases = fam_phases[i:j]

            batched = collate_gene_families(sub_items, dtype=dtype, device=device)
            offsets = [m['clade_offset'] for m in batched['family_meta']]
            cross_waves = collate_wave(sub_waves, offsets)

            max_n_waves = max(len(p) for p in sub_phases)
            cross_phases = []
            for k in range(max_n_waves):
                phase_k = 1
                for fp in sub_phases:
                    if k < len(fp):
                        phase_k = max(phase_k, fp[k])
                cross_phases.append(phase_k)

            family_clade_counts = [m['C'] for m in batched['family_meta']]
            family_clade_offsets = [m['clade_offset'] for m in batched['family_meta']]
            wl = build_wave_layout(
                waves=cross_waves,
                phases=cross_phases,
                ccp_helpers=batched['ccp'],
                leaf_row_index=batched['leaf_row_index'],
                leaf_col_index=batched['leaf_col_index'],
                root_clade_ids=batched['root_clade_ids'],
                device=device,
                dtype=dtype,
                family_clade_counts=family_clade_counts,
                family_clade_offsets=family_clade_offsets,
            )
            return wl, batched['root_clade_ids']

        _lazy_batch_builder = _build_lazy_batch

    def _num_layout_batches() -> int:
        if wave_layout_batches is not None:
            return len(wave_layout_batches)
        if _lazy_batch_ranges is not None:
            return len(_lazy_batch_ranges)
        return 1

    _rng = torch.Generator(device='cpu')
    _rng.manual_seed(int(stochastic_seed))

    def _select_batch_ids() -> list[int] | None:
        n_total = _num_layout_batches()
        use_stochastic = stochastic_batches and optimizer == 'sgd' and n_total > 1
        if not use_stochastic:
            return None
        sampled = int(torch.randint(0, n_total, (1,), generator=_rng).item())
        return [sampled]

    def _extract(theta_d):
        if pibar_mode in ('dense', 'topk') and transfer_mat_unnormalized is not None:
            log_pS, log_pD, log_pL, transfer_mat, mt_raw = extract_parameters(
                theta_d, transfer_mat_unnormalized,
                genewise=False, specieswise=specieswise, pairwise=False,
            )
            mt = mt_raw.squeeze(-1) if mt_raw.ndim == 2 else mt_raw
        else:
            log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
                theta_d, unnorm_row_max, specieswise=specieswise,
            )
        return log_pS, log_pD, log_pL, transfer_mat, mt

    # --- DDP: family-sharded data-parallel reduction -------------------------
    # Each rank runs the forward/backward on its family shard; the DATA nll+grad
    # are all-reduced(SUM) BEFORE the priors (which are identical per rank, so are
    # applied once on the reduced quantities). Loss/grad is a SUM over families and
    # the E-adjoint solve is linear with a shared operator => sum-of-shards == global.
    _distributed = bool(distributed) and _ddp.ddp_enabled()
    _ar_dev = ddp_device if ddp_device is not None else device

    def _reduce_data(nll, grad_theta, grad_omega):
        """In-place all-reduce(SUM) of per-shard DATA nll + grads. NaN-sanitize the
        grads LOCALLY first so one bad family-shard can't poison the cluster. Returns
        the reduced nll (float)."""
        if not _distributed:
            return nll
        if grad_theta is not None:
            grad_theta.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            _ddp.all_reduce_sum_(grad_theta)
        if grad_omega is not None:
            grad_omega.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            _ddp.all_reduce_sum_(grad_omega)
        _nllt = torch.tensor(float(nll), device=_ar_dev, dtype=dtype)
        _ddp.all_reduce_sum_(_nllt)
        return float(_nllt.item())

    def _forward_backward(theta_d, warm_E, selected_batch_ids=None, omega_d=None):
        """Full forward + backward.

        Returns (nll, grad_theta, statsG, E_out) when origination is uniform, and
        (nll, grad_theta, statsG, E_out, grad_omega) when omega_d is provided
        (origination='optimize'). ``grad_omega`` [S] is the DIRECT ∂(Σ NLL)/∂omega
        at FIXED Pi, E (no implicit solve): origination only enters
        compute_log_likelihood, so its omega gradient is a cheap autograd through
        the per-family numerators + the shared (E + omega) survival denominator.
        """
        # Origination log-prob log_pO = omega - logsumexp2(omega). None -> uniform.
        # 'fixed': a precomputed constant log_pO (omega is not a parameter).
        if _fixed_origination:
            log_pO = _fixed_log_pO
        else:
            log_pO = _orig_log_pO(omega_d) if omega_d is not None else None
        if wave_layout_batches is not None:
            if selected_batch_ids is None:
                layout_batches = wave_layout_batches
            else:
                layout_batches = [wave_layout_batches[i] for i in selected_batch_ids]
        elif _lazy_batch_builder is not None and _lazy_batch_ranges is not None:
            if selected_batch_ids is None:
                ids = range(len(_lazy_batch_ranges))
            else:
                ids = selected_batch_ids
            layout_batches = (_lazy_batch_builder(i) for i in ids)
        else:
            layout_batches = [(wave_layout, root_clade_ids)]

        torch.cuda.synchronize()
        t_e0 = time.perf_counter()
        try:
            with torch.no_grad():
                log_pS, log_pD, log_pL, transfer_mat, mt = _extract(theta_d)

                E_out = E_fixed_point(
                    species_helpers=species_helpers,
                    log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                    transfer_mat=transfer_mat, max_transfer_mat=mt,
                    max_iters=e_max_iters, tolerance=e_tol,
                    warm_start_E=warm_E,
                    dtype=dtype, device=device, pibar_mode=pibar_mode,
                    ancestors_T=_ancestors_T,
                    leaf_E=leaf_E,
                )
        except torch.OutOfMemoryError as exc:
            raise torch.OutOfMemoryError(
                f"OOM in optimize_theta_wave phase=E_fixed_point; "
                f"mode={'specieswise' if specieswise else 'global'}; "
                f"pibar_mode={pibar_mode}; {_cuda_mem_diag(device)}"
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise torch.OutOfMemoryError(
                    f"OOM in optimize_theta_wave phase=E_fixed_point; "
                    f"mode={'specieswise' if specieswise else 'global'}; "
                    f"pibar_mode={pibar_mode}; {_cuda_mem_diag(device)}"
                ) from exc
            raise

        torch.cuda.synchronize()
        t_pi0 = time.perf_counter()
        with torch.no_grad():
            nll = 0.0
            grad_theta = torch.zeros_like(theta_d)
            stats_list = []
            pi_bwd_t_total = 0.0
            cg_t_total = 0.0
            theta_t_total = 0.0
            pi_phase_time = 0.0
            grad_phase_time = 0.0
            n_batch_accum = 0
            # Origination: collect each family's root-clade Pi row [n_fam_b, S] so
            # the DIRECT omega gradient can be computed once (per-family numerators
            # + the shared survival denominator) after the wave loop.
            root_pi_rows = [] if log_pO is not None else None

            for _batch_idx, (wl_b, roots_b) in enumerate(layout_batches):
                t_pi_b0 = time.perf_counter()
                try:
                    Pi_out_b = Pi_wave_forward(
                        wave_layout=wl_b, species_helpers=species_helpers,
                        E=E_out['E'], Ebar=E_out['E_bar'],
                        E_s1=E_out['E_s1'], E_s2=E_out['E_s2'],
                        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                        transfer_mat=transfer_mat, max_transfer_mat=mt,
                        device=device, dtype=dtype, pibar_mode=pibar_mode,
                        leaf_obs_log=leaf_obs_log,
                    )
                    logL_b = compute_log_likelihood(
                        Pi_out_b['Pi'], E_out['E'], roots_b, log_pO=log_pO,
                    )
                    nll += float(logL_b.sum().item())
                    if root_pi_rows is not None:
                        # Detach: the omega gradient is evaluated at FIXED Pi.
                        root_pi_rows.append(Pi_out_b['Pi'][roots_b, :].detach())
                except torch.OutOfMemoryError as exc:
                    raise torch.OutOfMemoryError(
                        f"OOM in optimize_theta_wave phase=Pi_wave_forward; "
                        f"mode={'specieswise' if specieswise else 'global'}; "
                        f"pibar_mode={pibar_mode}; family_batch_size={family_batch_size}; {_cuda_mem_diag(device)}"
                    ) from exc
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise torch.OutOfMemoryError(
                            f"OOM in optimize_theta_wave phase=Pi_wave_forward; "
                            f"mode={'specieswise' if specieswise else 'global'}; "
                            f"pibar_mode={pibar_mode}; family_batch_size={family_batch_size}; {_cuda_mem_diag(device)}"
                        ) from exc
                    raise
                pi_phase_time += time.perf_counter() - t_pi_b0

                t_g_b0 = time.perf_counter()
                try:
                    grad_theta_b, statsG_b = implicit_grad_loglik_vjp_wave(
                        wl_b, species_helpers,
                        Pi_star_wave=Pi_out_b['Pi_wave_ordered'],
                        Pibar_star_wave=Pi_out_b['Pibar_wave_ordered'],
                        E_star=E_out['E'], E_s1=E_out['E_s1'],
                        E_s2=E_out['E_s2'], Ebar=E_out['E_bar'],
                        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
                        max_transfer_mat=mt,
                        root_clade_ids_perm=wl_b['root_clade_ids'],
                        theta=theta_d,
                        unnorm_row_max=unnorm_row_max,
                        specieswise=specieswise,
                        device=device, dtype=dtype,
                        neumann_terms=neumann_terms,
                        use_pruning=use_pruning,
                        pruning_threshold=pruning_threshold,
                        cg_tol=cg_tol, cg_maxiter=cg_maxiter,
                        gmres_restart=gmres_restart,
                        pibar_mode=pibar_mode,
                        transfer_mat=transfer_mat,
                        transfer_mat_unnormalized=transfer_mat_unnormalized,
                        ancestors_T=_ancestors_T,
                        leaf_E=leaf_E,
                        leaf_obs_log=leaf_obs_log,
                        log_pO=log_pO,
                    )
                except torch.OutOfMemoryError as exc:
                    raise torch.OutOfMemoryError(
                        f"OOM in optimize_theta_wave phase=implicit_grad_loglik_vjp_wave; "
                        f"mode={'specieswise' if specieswise else 'global'}; "
                        f"pibar_mode={pibar_mode}; family_batch_size={family_batch_size}; {_cuda_mem_diag(device)}"
                    ) from exc
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        raise torch.OutOfMemoryError(
                            f"OOM in optimize_theta_wave phase=implicit_grad_loglik_vjp_wave; "
                            f"mode={'specieswise' if specieswise else 'global'}; "
                            f"pibar_mode={pibar_mode}; family_batch_size={family_batch_size}; {_cuda_mem_diag(device)}"
                        ) from exc
                    raise
                grad_phase_time += time.perf_counter() - t_g_b0

                grad_theta = grad_theta + grad_theta_b
                stats_list.append(statsG_b)
                pi_bwd_t_total += float(getattr(statsG_b, 'pi_bwd_time', 0.0))
                cg_t_total += float(getattr(statsG_b, 'cg_time', 0.0))
                theta_t_total += float(getattr(statsG_b, 'theta_vjp_time', 0.0))
                n_batch_accum += 1

            # Gradient accumulation normalization: average over family batches.
            # grad_reduction="sum" skips it (pure Σ over families) — required for DDP
            # so summing per-shard grads == the single-GPU sum; nll is already a Σ.
            if grad_reduction != "sum" and n_batch_accum > 0:
                grad_theta = grad_theta / float(n_batch_accum)

            # --- DIRECT omega gradient (origination='optimize') --------------
            # omega enters ONLY compute_log_likelihood, so ∂(Σ NLL)/∂omega at FIXED
            # Pi, E is a direct autograd through the (origination-weighted)
            # per-family numerators and the shared survival denominator — no
            # implicit solve. Mirrors compute_log_likelihood EXACTLY:
            #   NLL_f = -(logsumexp2(root_Pi_f + log_pO) - logsumexp2(log2(1-exp2(E)) + log_pO))
            grad_omega = None
            if log_pO is not None and omega_d is not None:   # optimize-mode only; fixed/uniform skip
                from gpurec.core.log2_utils import logsumexp2 as _lse2
                from gpurec.core.log2_utils import _safe_log2_internal as _slog2
                root_pi_all = torch.cat(root_pi_rows, dim=0)  # [n_fam_total, S]
                n_fam_total = root_pi_all.shape[0]
                E_det = E_out['E'].detach()
                # log2(1 - exp2(E)) with E -> 0 guarded (matches compute_log_likelihood).
                log_one_minus_E = _slog2(1.0 - torch.exp2(E_det))
                with torch.enable_grad():
                    omega_leaf = omega_d.detach().clone().requires_grad_(True)
                    # match the forward's floored p^O so the omega gradient is consistent
                    log_pO_g = _orig_log_pO(omega_leaf)
                    num = _lse2(root_pi_all + log_pO_g, dim=-1)          # [n_fam_total]
                    denom = _lse2(log_one_minus_E + log_pO_g, dim=-1)    # scalar (shared E)
                    nll_omega = -(num - denom).sum()                     # Σ_f NLL_f
                    grad_omega = torch.autograd.grad(nll_omega, omega_leaf)[0].detach()
                # Match the theta gradient's family-batch averaging so the two
                # blocks of the joint scipy gradient are on the same scale.
                if grad_reduction != "sum" and n_batch_accum > 0:
                    grad_omega = grad_omega / float(n_batch_accum)

            if stats_list:
                method0 = stats_list[0].method
                method = method0 if all(s.method == method0 for s in stats_list) else "mixed"
                statsG = LinearSolveStats(
                    method=method,
                    iters=int(sum(s.iters for s in stats_list)),
                    rel_residual=float(max(s.rel_residual for s in stats_list)),
                    fallback_used=bool(any(s.fallback_used for s in stats_list)),
                )
                setattr(statsG, 'pi_bwd_time', pi_bwd_t_total)
                setattr(statsG, 'cg_time', cg_t_total)
                setattr(statsG, 'theta_vjp_time', theta_t_total)
            else:
                statsG = LinearSolveStats("none", 0, 0.0, False)
        torch.cuda.synchronize()
        if verbose:
            pi_bwd_t = getattr(statsG, 'pi_bwd_time', 0)
            cg_t = getattr(statsG, 'cg_time', 0)
            theta_t = getattr(statsG, 'theta_vjp_time', 0)
            print(f"    breakdown: E={t_pi0-t_e0:.3f}s  Pi={pi_phase_time:.3f}s  grad={grad_phase_time:.3f}s"
                  f"  [Pi_bwd={pi_bwd_t:.3f}s  CG={cg_t:.3f}s  theta_vjp={theta_t:.3f}s]",
                  flush=True)

        if omega_d is not None:
            return nll, grad_theta, statsG, E_out, grad_omega
        return nll, grad_theta, statsG, E_out

    # --- L-BFGS-B path (scipy) ---
    if optimizer == 'lbfgs':
        import numpy as np
        from scipy.optimize import minimize as scipy_minimize

        theta_t = theta_init.to(device=device, dtype=dtype).clone()
        if _use_groups:
            # Reduce the (typically uniform) per-branch init to per-group rows by
            # averaging members of each category -> [G,3]. This is the FREE param.
            _cnt = torch.zeros(_n_groups, dtype=dtype, device=device)
            _cnt.index_add_(0, _group_index_t, torch.ones_like(theta_t[:, 0]))
            _gsum = torch.zeros((_n_groups, 3), dtype=dtype, device=device)
            _gsum.index_add_(0, _group_index_t, theta_t)
            theta_param_t = _gsum / _cnt.clamp(min=1).unsqueeze(1)
        else:
            theta_param_t = theta_t
        if _distributed:
            _ddp.broadcast_(theta_param_t)   # float-identical init across ranks
        theta_shape = theta_param_t.shape   # OPTIMISED param: [G,3] grouped else theta_init.shape
        n_theta = theta_param_t.numel()
        history: List[StepRecord] = []
        warm_E_ref = [None]
        eval_count = [0]

        def forward_and_grad(x_flat_np):
            # JOINT vector layout: [theta_flat (n_theta); omega (S)] when origination
            # is optimised, else just theta_flat. Split, run, re-concat the gradient.
            x_flat = torch.from_numpy(x_flat_np).to(device=device, dtype=dtype)
            theta_flat = x_flat[:n_theta]
            theta_param = theta_flat.reshape(theta_shape).clamp(min=_THETA_MIN)
            theta_d = _expand_theta(theta_param)   # [G,3]->[S,3] (no-op when ungrouped)
            if _opt_origination:
                omega_param = x_flat[n_theta:].reshape(_n_omega_param)
                omega_d = _expand_omega(omega_param)   # [G_O]->[S] (no-op when ungrouped)
            else:
                omega_param = None
                omega_d = None

            t_start = time.perf_counter()
            if _opt_origination:
                nll, grad_theta, statsG, E_out, grad_omega = _forward_backward(
                    theta_d, warm_E_ref[0], omega_d=omega_d,
                )
                # DDP: all-reduce(SUM) the DATA nll + per-branch grads ACROSS shards
                # here, BEFORE the barrier/priors (those are identical per rank, so
                # are applied once on the reduced data). grad_omega is per-branch [S].
                nll = _reduce_data(nll, grad_theta, grad_omega)
                # Asymmetric Dirichlet (vertical) prior acts on per-branch p^O,
                # BEFORE the group reduction (so it composes with grouped omega).
                nll, grad_omega = _apply_origination_dirichlet(omega_d, nll, grad_omega)
                # Reduce the per-branch omega grad to per-category (chain rule).
                grad_omega = _reduce_omega(grad_omega)
            else:
                nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E_ref[0])
                nll = _reduce_data(nll, grad_theta, None)
                grad_omega = None
            nll, grad_theta = _apply_prior(theta_d, nll, grad_theta)
            # L2 ridge on the (possibly grouped) origination logits.
            nll, grad_omega = _apply_origination_prior(omega_param, nll, grad_omega)
            nll, grad_omega = _apply_origination_depth(omega_param, nll, grad_omega)
            nll, grad_omega = _apply_origination_root(omega_param, nll, grad_omega)
            # Reduce the per-branch [S,3] gradient to per-group [G,3] (chain rule
            # of the expand map); no-op when ungrouped.
            grad_param = _reduce_grad(grad_theta)
            step_time = time.perf_counter() - t_start
            warm_E_ref[0] = E_out['E'].detach()

            nll_is_nan = math.isnan(nll)
            eval_count[0] += 1
            if verbose:
                grad_inf_str = f"{float(grad_param.abs().max()):.3e}" if not nll_is_nan else "nan"
                nll_str = f"{nll:.4f}" if not nll_is_nan else "nan"
                e_it = int(E_out['iterations'])
                print(f"  step {eval_count[0]:3d}/{steps}  NLL={nll_str}  |g|={grad_inf_str}"
                      f"  E_iters={e_it}  t={step_time:.2f}s", flush=True)
            fp_info = FixedPointInfo(iterations_E=int(E_out['iterations']),
                                     iterations_Pi=0)
            history.append(StepRecord(
                iteration=eval_count[0], theta=theta_d.detach().cpu(),
                rates=torch.exp2(theta_d.detach()).cpu(),
                negative_log_likelihood=nll, log_likelihood=-nll,
                theta_step_inf=0.0,
                grad_infinity_norm=float(grad_param.abs().max().item()) if not nll_is_nan else float('nan'),
                fp_info=fp_info, gradient=grad_param.cpu(),
                solve_stats_F=LinearSolveStats("wave_neumann", neumann_terms, 0.0, False),
                solve_stats_G=statsG,
                step_time_s=step_time,
            ))

            grad_theta_np = grad_param.reshape(-1).cpu().to(torch.float64).numpy()
            if grad_omega is not None:
                grad_omega_np = grad_omega.reshape(-1).cpu().to(torch.float64).numpy()
                grad_np = np.concatenate([grad_theta_np, grad_omega_np])
            else:
                grad_np = grad_theta_np
            np.nan_to_num(grad_np, copy=False, nan=0.0)
            # Return +inf (not NaN) so scipy's line search backtracks instead of corrupting state
            return_nll = float('inf') if nll_is_nan else float(nll)
            return return_nll, grad_np

        theta_x0 = theta_param_t.reshape(-1).cpu().to(torch.float64).numpy()
        if theta_bounds is None:
            # Current behavior: lower bound = log2(1e-10), no upper bound.
            theta_bounds_list = [(_THETA_MIN, None)] * len(theta_x0)
        else:
            # AleRax-style box constraints (log2 units). Accept scalar (lo, hi)
            # applied to every element, or per-element arrays broadcastable to theta.
            lo_in, hi_in = theta_bounds
            lo_arr = np.broadcast_to(np.asarray(lo_in, dtype=np.float64), theta_x0.shape)
            hi_arr = np.broadcast_to(np.asarray(hi_in, dtype=np.float64), theta_x0.shape)
            theta_bounds_list = [(float(lo_arr[i]), float(hi_arr[i])) for i in range(len(theta_x0))]
        if _opt_origination:
            # Append omega block. omega is a softmax LOGIT (shift-invariant), so it
            # is UNBOUNDED — the normalisation in log_pO = omega - logsumexp2(omega)
            # makes any global shift a no-op. Leave omega unconstrained.
            omega_x0 = _omega0.reshape(-1).cpu().to(torch.float64).numpy()
            x0 = np.concatenate([theta_x0, omega_x0])
            bounds = theta_bounds_list + [(None, None)] * len(omega_x0)
        else:
            x0 = theta_x0
            bounds = theta_bounds_list
        result = scipy_minimize(
            forward_and_grad, x0, method='L-BFGS-B', jac=True,
            bounds=bounds,
            options={'maxiter': steps, 'maxfun': steps * 3, 'ftol': 1e-12, 'gtol': 1e-6},
        )

        x_final = torch.from_numpy(result.x).to(device=device, dtype=dtype)
        # Expand grouped [G,3] -> per-branch [S,3] for downstream eval/output
        # (no-op when ungrouped).
        theta_final = _expand_theta(x_final[:n_theta].reshape(theta_shape))
        if _opt_origination:
            # Expand grouped [G_O] -> per-branch [S] for eval/output (no-op ungrouped).
            omega_final = _expand_omega(x_final[n_theta:].reshape(_n_omega_param))
        else:
            omega_final = None

        # Detect float32 precision floor and retry in float64
        if dtype == torch.float32 and len(history) >= 5:
            recent_nlls = [h.negative_log_likelihood for h in history[-5:]]
            nll_range = max(recent_nlls) - min(recent_nlls)
            final_grad_inf = history[-1].grad_infinity_norm
            if final_grad_inf > 1e-2 and nll_range < 1e-4:
                if verbose:
                    print(f"  [float32 floor detected: grad_inf={final_grad_inf:.3e}, "
                          f"nll_range={nll_range:.2e}] retrying in float64 ...", flush=True)

                def _to64(x):
                    """Recursively convert floating-point tensors to float64.

                    Handles nested containers: dict, list, tuple.
                    Non-floating tensors (e.g. Long index tensors) and
                    non-tensor values (int, str, dict of strings) are left untouched.
                    """
                    if torch.is_tensor(x):
                        return x.to(torch.float64) if x.is_floating_point() else x
                    if isinstance(x, dict):
                        return {k: _to64(v) for k, v in x.items()}
                    if isinstance(x, list):
                        return [_to64(v) for v in x]
                    if isinstance(x, tuple):
                        return tuple(_to64(v) for v in x)
                    return x

                wl64 = _to64(wave_layout)
                sp64 = _to64(species_helpers)
                urm64 = _to64(unnorm_row_max)
                tm64 = _to64(transfer_mat_unnormalized) if transfer_mat_unnormalized is not None else None
                theta_init64 = theta_final.to(torch.float64)

                result64 = optimize_theta_wave(
                    wl64, sp64, root_clade_ids,
                    urm64, theta_init64,
                    transfer_mat_unnormalized=tm64,
                    steps=steps, lr=lr, tol_theta=tol_theta,
                    e_max_iters=e_max_iters, e_tol=e_tol,
                    neumann_terms=neumann_terms,
                    use_pruning=use_pruning, pruning_threshold=pruning_threshold,
                    cg_tol=cg_tol, cg_maxiter=cg_maxiter, gmres_restart=gmres_restart,
                    specieswise=specieswise, device=device,
                    dtype=torch.float64,
                    pibar_mode=pibar_mode,
                    family_batch_size=family_batch_size,
                    wave_layout_batches=wave_layout_batches,
                    families=families,
                    stochastic_batches=stochastic_batches,
                    stochastic_seed=stochastic_seed,
                    optimizer='lbfgs',
                    verbose=verbose,
                    leaf_E=leaf_E,
                    leaf_obs_log=leaf_obs_log,
                    brownian_sigma=brownian_sigma,
                    brownian_root_sigma=brownian_root_sigma,
                    # Pass the RESOLVED root-anchor centre (not None) so the
                    # float64 retry keeps the same prior centre as phase 1.
                    brownian_mu=_prior_mu if _use_prior else brownian_mu,
                    dtl_tv_lambda=dtl_tv_lambda,
                    dtl_tv_eps=dtl_tv_eps,
                    parent_index=(_prior_parent_index if (_use_prior or _use_tv)
                                  else parent_index),
                    theta_bounds=theta_bounds,
                    origination=origination,
                    omega_init=omega_final if _opt_origination else None,
                    origination_sigma=origination_sigma,
                    origination_root_sigma=origination_root_sigma,
                    origination_mu=origination_mu,
                    origination_decouple_root=origination_decouple_root,
                    origination_l2=origination_l2,
                    group_index=group_index,
                    omega_group_index=omega_group_index,
                    origination_depth=origination_depth,
                    origination_depth_lambda=origination_depth_lambda,
                    origination_root_index=origination_root_index,
                    origination_root_lambda=origination_root_lambda,
                    origination_dirichlet_c=origination_dirichlet_c,
                    origination_vertical_pi=origination_vertical_pi,
                    origination_barrier_kind=origination_barrier_kind,
                    distributed=distributed,
                    grad_reduction=grad_reduction,
                    ddp_device=ddp_device,
                )
                # Merge histories: float32 phase first, then float64 phase
                result64["history"] = history + result64["history"]
                return result64

        # Pure DATA negative-log-likelihood at the optimum, with the prior EXCLUDED.
        # `result.fun` is the MAP objective (data NLL + Brownian penalty); the penalty
        # depends on the species-tree topology, so it must NOT be in cross-root
        # likelihood (rooting) comparisons. Re-evaluate the forward once at the optimum
        # (with the optimised omega so the origination-weighted likelihood is used).
        _data_nll = float(_forward_backward(
            theta_final.to(device=device, dtype=dtype), None,
            omega_d=omega_final,
        )[0])

        result_dict = {
            "theta": theta_final.cpu(),
            "rates": torch.exp2(theta_final).cpu(),
            "negative_log_likelihood": result.fun,
            "log_likelihood": -result.fun,
            "data_negative_log_likelihood": _data_nll,
            "data_log_likelihood": -_data_nll,
            "history": history,
            "scipy_result": result,
        }
        if _opt_origination:
            # omega = origination log2-weights; origination = normalised p^O_e.
            result_dict["omega"] = omega_final.detach().cpu()
            result_dict["origination"] = torch.exp2(
                _orig_log_pO(omega_final)
            ).detach().cpu()
            result_dict["origination_strategy"] = origination
            if _orig_floor > 0.0:
                result_dict["origination_floor"] = _orig_floor
            if _use_orig_l2:
                result_dict["origination_prior"] = {
                    "type": "l2", "lambda": _orig_l2,
                }
        elif _fixed_origination:
            # Held-fixed origination: report the constant omega + p^O so the
            # sidecar records what O the theta fit was conditioned on.
            result_dict["omega"] = _fixed_omega_t.detach().cpu()
            result_dict["origination"] = torch.exp2(_fixed_log_pO).detach().cpu()
            result_dict["origination_strategy"] = "fixed"
        return result_dict

    # --- Iterative optimizers (adam, sgd) ---
    theta = torch.nn.Parameter(theta_init.to(device=device, dtype=dtype).clone())
    if _distributed:
        with torch.no_grad():
            _ddp.broadcast_(theta.data)          # float-identical init across ranks
    if optimizer == 'sgd':
        opt = torch.optim.SGD([theta], lr=lr, momentum=momentum, nesterov=False)
    else:
        opt = torch.optim.Adam([theta], lr=lr)

    history: List[StepRecord] = []
    prev_theta = theta.detach().clone()
    warm_E = None

    for it in range(1, steps + 1):
        theta_d = theta.detach()
        selected_batch_ids = _select_batch_ids()

        t_start = time.perf_counter()
        nll, grad_theta, statsG, E_out = _forward_backward(theta_d, warm_E, selected_batch_ids=selected_batch_ids)
        nll = _reduce_data(nll, grad_theta, None)   # DDP: all-reduce DATA before the prior
        nll, grad_theta = _apply_prior(theta_d, nll, grad_theta)
        warm_E = E_out['E'].detach()
        iters_E = int(E_out['iterations'])

        # Optimizer step (minimize NLL)
        opt.zero_grad(set_to_none=True)
        grad_clean = grad_theta.clone()
        grad_clean.nan_to_num_(nan=0.0)
        theta.grad = grad_clean
        opt.step()
        step_time = time.perf_counter() - t_start

        with torch.no_grad():
            theta.clamp_(min=_THETA_MIN)

        # Bookkeeping
        theta_detached = theta.detach()
        diff = float(torch.max(torch.abs(theta_detached - prev_theta)).item())
        prev_theta = theta_detached.clone()
        rates = torch.exp2(theta_detached)
        grad_inf = float(grad_theta.abs().max().item())

        if verbose:
            nll_str = f"{nll:.4f}" if not math.isnan(nll) else "nan"
            print(f"  step {it:3d}/{steps}  NLL={nll_str}  |g|={grad_inf:.3e}"
                  f"  E_iters={iters_E}  t={step_time:.2f}s", flush=True)

        fp_info = FixedPointInfo(iterations_E=iters_E, iterations_Pi=0)
        statsF_dummy = LinearSolveStats("wave_neumann", neumann_terms, 0.0, False)

        history.append(
            StepRecord(
                iteration=it,
                theta=theta_detached.cpu(),
                rates=rates.cpu(),
                negative_log_likelihood=nll,
                log_likelihood=-nll,
                theta_step_inf=diff,
                grad_infinity_norm=grad_inf,
                fp_info=fp_info,
                gradient=grad_theta.cpu(),
                solve_stats_F=statsF_dummy,
                solve_stats_G=statsG,
                step_time_s=step_time,
            )
        )

        if _distributed:                          # collective early-stop (avoid deadlock)
            _difft = torch.tensor(float(diff), device=_ar_dev, dtype=dtype)
            _ddp.all_reduce_max_(_difft); diff = float(_difft.item())
        if diff < tol_theta and it > 1:
            break

    return {
        "theta": theta.detach().cpu(),
        "rates": torch.exp2(theta.detach()).cpu(),
        "negative_log_likelihood": history[-1].negative_log_likelihood if history else float('nan'),
        "log_likelihood": history[-1].log_likelihood if history else float('nan'),
        "history": history,
    }
