#!/usr/bin/env python
"""Finite-difference check of the K-generalized Brownian prior.

`brownian_log_prior_and_grad` was generalized from a hard-coded [S,3] to any
[S,K] so the same TKP prior can regularize the per-branch origination weight
omega ([S,1]). This script verifies, on CPU (pure torch, no Triton):

  1. K=3 (D,L,T) is byte-unchanged vs a reference manual penalty/grad.
  2. The analytic gradient matches central finite differences for K=1,3,4.
  3. Per-column sigma (a [K] vector) works.
  4. The increment penalty is shift-invariant (omega -> omega+c), as required
     for the softmax-gauge origination use; only the root anchor moves.

Run: python3 experiments/validate_prior_kcols.py
"""
import sys
import torch

sys.path.insert(0, ".")
from gpurec.core.tree_prior import brownian_log_prior_and_grad

torch.manual_seed(0)
DT = torch.float64


def make_tree(S):
    """Random valid rooted binary-ish parent_index on S nodes (node 0 = root)."""
    parent = torch.full((S,), -1, dtype=torch.long)
    for v in range(1, S):
        parent[v] = torch.randint(0, v, (1,)).item()  # parent earlier than v
    return parent


def fd_grad(theta, parent, sigma, sigma_root, mu, eps=1e-6):
    g = torch.zeros_like(theta)
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            tp = theta.clone(); tp[i, j] += eps
            tm = theta.clone(); tm[i, j] -= eps
            pp = brownian_log_prior_and_grad(tp, parent, sigma, sigma_root, mu)[0]
            pm = brownian_log_prior_and_grad(tm, parent, sigma, sigma_root, mu)[0]
            g[i, j] = (pp - pm) / (2 * eps)
    return g


def check(S, K, sigma, sigma_root, mu, label):
    parent = make_tree(S)
    theta = torch.randn(S, K, dtype=DT)
    pen, grad = brownian_log_prior_and_grad(theta, parent, sigma, sigma_root, mu)
    g_fd = fd_grad(theta, parent, sigma, sigma_root, mu)
    err = (grad - g_fd).abs().max().item()
    rel = err / (g_fd.abs().max().item() + 1e-30)
    ok = rel < 1e-5
    print(f"  [{label}] S={S} K={K}  penalty={float(pen):.6f}  "
          f"max|grad-fd|={err:.2e}  rel={rel:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def check_shift_invariance():
    """Increment penalty must be invariant to a global shift of omega; with a
    LOOSE root anchor (large sigma_root) the total penalty barely moves."""
    S = 12
    parent = make_tree(S)
    omega = torch.randn(S, 1, dtype=DT)
    # Pure increment penalty: make the root anchor effectively off (huge sigma).
    p0 = brownian_log_prior_and_grad(omega, parent, 0.7, 1e8, 0.0)[0]
    p1 = brownian_log_prior_and_grad(omega + 3.14, parent, 0.7, 1e8, 0.0)[0]
    d = abs(float(p0) - float(p1))
    ok = d < 1e-6
    print(f"  [shift-invariance] |P(omega) - P(omega+c)| = {d:.2e}  "
          f"(increment term only) {'OK' if ok else 'FAIL'}")
    return ok


def check_k3_reference():
    """K=3 must equal an independent manual penalty/grad (no regression)."""
    S = 10
    parent = make_tree(S)
    theta = torch.randn(S, 3, dtype=DT)
    sigma, sigma_root, mu = 0.5, 4.0, 0.1
    pen, grad = brownian_log_prior_and_grad(theta, parent, sigma, sigma_root, mu)
    # Manual reference.
    inv = 1.0 / sigma ** 2
    invr = 1.0 / sigma_root ** 2
    P = torch.zeros((), dtype=DT)
    G = torch.zeros_like(theta)
    root = int((parent < 0).nonzero()[0])
    for v in range(S):
        if v == root:
            continue
        d = (theta[v] - theta[parent[v]]) * inv
        P = P + 0.5 * ((theta[v] - theta[parent[v]]) ** 2 * inv).sum()
        G[v] += d
        G[parent[v]] -= d
    rd = theta[root] - mu
    P = P + 0.5 * (invr * rd * rd).sum()
    G[root] += invr * rd
    ep = abs(float(pen) - float(P))
    eg = (grad - G).abs().max().item()
    ok = ep < 1e-9 and eg < 1e-9
    print(f"  [K=3 reference] |dP|={ep:.2e}  |dG|={eg:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def decouple_root(parent):
    """Mirror wave_optimizer: cut the root's incident edges (children -> -1)."""
    p = parent.clone()
    arange = torch.arange(p.shape[0])
    is_root = (p < 0) | (p == arange)
    root = int(is_root.nonzero()[0])
    p[p == root] = -1
    return p, root


def check_decouple_root():
    """Root decoupling: cut root edges + sigma_root=inf -> root is FREE.

    Verifies (a) the analytic gradient still matches finite differences on the
    decoupled graph, (b) the root row gradient is exactly 0 (no edge, no anchor),
    and (c) the penalty is invariant to perturbing ONLY the root weight (the root
    is not coupled to the rest of the tree)."""
    S = 14
    parent = make_tree(S)
    pdec, root = decouple_root(parent)
    omega = torch.randn(S, 1, dtype=DT)
    sigma, sigma_root, mu = 0.5, float("inf"), 0.0
    pen, grad = brownian_log_prior_and_grad(omega, pdec, sigma, sigma_root, mu)
    g_fd = fd_grad(omega, pdec, sigma, sigma_root, mu)
    err = (grad - g_fd).abs().max().item()
    root_g = abs(float(grad[root, 0]))
    # Perturb only the root weight; decoupled penalty must not change.
    o2 = omega.clone(); o2[root, 0] += 2.5
    pen2 = brownian_log_prior_and_grad(o2, pdec, sigma, sigma_root, mu)[0]
    d_root = abs(float(pen) - float(pen2))
    ok = err < 1e-5 and root_g < 1e-12 and d_root < 1e-12
    print(f"  [decouple-root] FD err={err:.2e}  |grad[root]|={root_g:.2e}  "
          f"|dP when root perturbed|={d_root:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


def main():
    print("Brownian prior K-generalization validation (CPU, float64)")
    results = []
    results.append(check_k3_reference())
    results.append(check(10, 3, 0.5, 4.0, 0.0, "K=3 scalar sigma"))
    results.append(check(10, 3, torch.tensor([0.3, 0.5, 0.9], dtype=DT),
                         torch.tensor([3., 4., 5.], dtype=DT),
                         torch.tensor([0., 0.1, -0.2], dtype=DT), "K=3 per-axis"))
    results.append(check(15, 1, 0.7, 5.0, 0.0, "K=1 omega scalar"))
    results.append(check(12, 4, 0.6, 4.0, 0.0, "K=4 (DLT+O) scalar"))
    results.append(check_shift_invariance())
    results.append(check_decouple_root())
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
