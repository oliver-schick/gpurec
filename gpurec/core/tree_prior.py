"""Tree-structured (Thorne-Kishino-Painter) Brownian rate prior for the
species-wise DTL rate optimizer.

Free per-branch DTL rates make the marginal likelihood rate-unidentifiable, so
the loss rate explodes toward the boundary. The fix (recount project's
``brownian_prior.tex``, time-uniform TKP relaxed clock) is a tree-structured
prior that couples each branch's log-rate to its parent's log-rate, plus a soft
anchor on the root log-rate for propriety. The resulting MAP objective
(``NLL + P``) is coercive, ruling out the boundary explosion.

This module provides:

  * :func:`species_parent_index` -- derive ``parent_index[S]`` (parent of each
    species node; root maps to ``-1``) from ``species_helpers``;
  * :func:`brownian_log_prior_and_grad` -- the penalty ``P`` (a ``+scalar`` to
    add to the NLL) and its gradient ``dP/dtheta`` ([S,3]).

Everything is pure ``torch`` and works on CPU/CUDA in float32/float64.
"""
from __future__ import annotations

import torch


def species_parent_index(species_helpers) -> torch.Tensor:
    """Derive ``parent_index`` [S] (long) from ``species_helpers``.

    gpurec encodes the rooted, strictly-binary species tree via the two arrays
    consumed by :func:`gpurec.core.terms.gather_E_children` /
    :func:`gpurec.core.likelihood.E_step`:

      * ``s_P_indexes``  -- length ``2*K`` (K = #internal nodes). The first K
        entries are the internal-node ids (``p_j``); the next K are the same ids
        shifted by ``S`` (the child-2 slot of the ``2*S`` gather layout). So
        ``s_P_indexes[:K]`` is exactly the list of parent node ids.
      * ``s_C12_indexes`` -- length ``2*K``: ``[left_0..left_{K-1},
        right_0..right_{K-1}]`` -- the left children then the right children,
        aligned with ``s_P_indexes[:K]``.

    Hence for each internal node ``p_j = s_P_indexes[j]`` (``j in [0,K)``):
    ``parent[s_C12_indexes[j]]   = p_j``  (left child) and
    ``parent[s_C12_indexes[j+K]] = p_j``  (right child).
    The unique node never appearing as a child is the root; it maps to ``-1``.

    The result is verified to be a valid rooted tree (exactly one root, every
    other node exactly one parent, node count == S, acyclic).

    Returns
    -------
    parent_index : LongTensor [S]
        ``parent_index[v]`` is the species-node id of ``v``'s parent, or ``-1``
        if ``v`` is the root. On the same device as the helper arrays.
    """
    S = int(species_helpers["S"])
    sP = species_helpers["s_P_indexes"]
    sC = species_helpers["s_C12_indexes"]
    device = sP.device

    twoK = sP.numel()
    if twoK % 2 != 0:
        raise ValueError(
            f"s_P_indexes length must be even (2*K), got {twoK}"
        )
    K = twoK // 2
    if sC.numel() != twoK:
        raise ValueError(
            f"s_C12_indexes length ({sC.numel()}) must equal s_P_indexes "
            f"length ({twoK})"
        )

    parents = sP[:K].to(torch.long)        # internal-node ids [K]
    children = sC.to(torch.long)           # [left.., right..] (2K)
    parents_both = torch.cat([parents, parents], dim=0)  # [2K], aligned with children

    parent_index = torch.full((S,), -1, dtype=torch.long, device=device)
    # Each child appears exactly once across (left, right) lists, so this is an
    # unambiguous scatter (no two parents claim the same child in a valid tree).
    parent_index[children] = parents_both

    _verify_parent_index(parent_index, S)
    return parent_index


def _verify_parent_index(parent_index: torch.Tensor, S: int) -> None:
    """Assert ``parent_index`` is a valid rooted tree on ``S`` nodes.

    Checks: shape == [S]; exactly one root (parent == -1); every non-root has a
    parent in ``[0, S)``; no self-loops; acyclic (walking parents from every
    node reaches the root within < S steps).
    """
    if parent_index.shape != (S,):
        raise ValueError(f"parent_index must have shape [{S}], got {tuple(parent_index.shape)}")

    pi = parent_index.cpu()
    is_root = pi == -1
    n_roots = int(is_root.sum().item())
    if n_roots != 1:
        raise ValueError(f"expected exactly one root (parent == -1), found {n_roots}")

    nonroot = ~is_root
    if nonroot.any():
        p = pi[nonroot]
        if int(p.min().item()) < 0 or int(p.max().item()) >= S:
            raise ValueError("non-root parent indices must lie in [0, S)")
    # No self-loops.
    self_loop = (pi == torch.arange(S))
    if bool(self_loop.any().item()):
        bad = int(torch.nonzero(self_loop, as_tuple=False)[0].item())
        raise ValueError(f"node {bad} is its own parent (cycle)")

    # Acyclicity: from each node, walking to the root must terminate in < S hops.
    root = int(torch.nonzero(is_root, as_tuple=False)[0].item())
    pil = pi.tolist()
    for v in range(S):
        steps = 0
        cur = v
        while cur != root:
            cur = pil[cur]
            if cur == -1:
                # Reached a different sink than the unique root -> impossible
                # given the single-root check above, but guard anyway.
                raise ValueError(f"node {v} does not reach the root")
            steps += 1
            if steps > S:
                raise ValueError(f"cycle detected walking parents from node {v}")


def brownian_log_prior_and_grad(
    theta: torch.Tensor,
    parent_index: torch.Tensor,
    sigma,
    sigma_root,
    mu,
):
    """Time-uniform TKP (Brownian) log-rate prior penalty and its gradient.

    For each rate axis ``a in {D, L, T}`` (columns of ``theta``) and each
    non-root node ``v`` with parent ``pa(v)``, let
    ``Delta_a(v) = theta[v,a] - theta[pa(v),a]``. The penalty (``= -log p_B``,
    to be ADDED to the NLL) is::

        P = sum_a (1 / (2 sigma_a^2)) * sum_{v != root} Delta_a(v)^2
          + sum_a (1 / (2 sigma_root^2)) * (theta[root,a] - mu_a)^2

    with gradient ``dP/dtheta`` ([S,3])::

        for each non-root v, per axis a:
            d = (theta[v,a] - theta[pa(v),a]) / sigma_a^2
            grad[v,a]      += d
            grad[pa(v),a]  -= d
        grad[root,a] += (theta[root,a] - mu_a) / sigma_root^2

    ``sigma``, ``sigma_root`` and ``mu`` are in LOG2 units (since ``theta`` is
    log2). With ``sigma -> inf`` the increment penalty vanishes (recovers free
    per-branch rates); small ``sigma`` strongly smooths adjacent branches
    (approaches clade grouping).

    Parameters
    ----------
    theta : Tensor [S, 3]
        Species-wise log2 rates (columns ordered [D, L, T]).
    parent_index : LongTensor [S]
        Parent of each node; root maps to ``-1`` (or to itself -- both are
        treated as "root" and excluded from the increment sum).
    sigma : float or Tensor
        Brownian increment std (log2 units), scalar or per-axis [3].
    sigma_root : float or Tensor
        Root-anchor std (log2 units), scalar or per-axis [3].
    mu : float or Tensor
        Root-anchor centre (log2 units), scalar or per-axis [3].

    Returns
    -------
    penalty : Tensor (0-dim scalar)
        ``P`` above; add this to the NLL.
    grad : Tensor [S, 3]
        ``dP/dtheta``; add this to the NLL gradient.
    """
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError(f"theta must have shape [S, 3], got {tuple(theta.shape)}")
    S = theta.shape[0]
    device, dtype = theta.device, theta.dtype

    parent_index = parent_index.to(device=device, dtype=torch.long)
    if parent_index.shape != (S,):
        raise ValueError(
            f"parent_index must have shape [{S}], got {tuple(parent_index.shape)}"
        )

    # Per-axis precision (1/sigma^2). Broadcast scalar -> [3].
    def _as_axis_vec(x, name):
        t = torch.as_tensor(x, device=device, dtype=dtype)
        if t.ndim == 0:
            t = t.expand(3)
        elif t.shape != (3,):
            raise ValueError(f"{name} must be scalar or shape [3], got {tuple(t.shape)}")
        return t

    sigma_v = _as_axis_vec(sigma, "sigma")               # [3]
    sigma_root_v = _as_axis_vec(sigma_root, "sigma_root")  # [3]
    mu_v = _as_axis_vec(mu, "mu")                          # [3]

    inv_var = 1.0 / (sigma_v * sigma_v)                    # [3]
    inv_var_root = 1.0 / (sigma_root_v * sigma_root_v)     # [3]

    # Root / non-root masks (treat -1 OR self-parent as root).
    arange = torch.arange(S, device=device)
    is_root = (parent_index < 0) | (parent_index == arange)
    nonroot = ~is_root
    # Safe parent ids for gather/scatter (root rows are masked out of the sum).
    parent_safe = torch.where(nonroot, parent_index, arange)  # [S]

    grad = torch.zeros_like(theta)

    # ---- Brownian increment term over non-root edges -----------------------
    # Delta[v] = theta[v] - theta[pa(v)], zeroed at the root.
    delta = theta - theta.index_select(0, parent_safe)        # [S, 3]
    delta = torch.where(nonroot.unsqueeze(1), delta, torch.zeros_like(delta))

    penalty = 0.5 * (inv_var * (delta * delta).sum(dim=0)).sum()

    # d = Delta[v] / sigma_a^2 ; grad[v] += d, grad[pa(v)] -= d.
    d = delta * inv_var.unsqueeze(0)                          # [S, 3]
    grad = grad + d                                          # grad[v] += d (root rows are 0)
    grad.index_add_(0, parent_safe, -d)                      # grad[pa(v)] -= d

    # ---- Root anchor term --------------------------------------------------
    root_ids = torch.nonzero(is_root, as_tuple=False).flatten()
    if root_ids.numel() > 0:
        r = int(root_ids[0].item())
        root_dev = theta[r] - mu_v                            # [3]
        penalty = penalty + 0.5 * (inv_var_root * root_dev * root_dev).sum()
        grad[r] = grad[r] + inv_var_root * root_dev

    return penalty, grad
