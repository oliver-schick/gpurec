"""Laplace marginal-likelihood (evidence) for gpurec -- Bayesian rooting + model
selection. Hessian via FINITE-DIFFERENCE HESSIAN-VECTOR PRODUCTS of the exact
batched gradient (full Hessian for small k; Lanczos+SLQ for large k).

Supports the JOINT parameter vector p = [theta_param ; omega_param]:
  - theta_param: DTL log2-rates in MODEL coords -- global (3), clade-grouped
    (G_dtl*3), or free-per-branch (S*3).
  - omega_param: ORIGINATION logits in model coords when the sidecar was fit with
    origination='optimize' -- clade-grouped (G_O) or free-per-branch (S). Absent
    for uniform-origination sidecars.
The Hessian-vector product Hv = (grad(p+eps v) - grad(p-eps v))/(2eps) is 2 grad
evals per Hv, INDEPENDENT of k. omega enters ONLY compute_log_likelihood, so the
omega gradient is a DIRECT autograd through the origination-weighted numerators +
shared survival denominator at fixed Pi,E (mirrors wave_optimizer); theta uses
the implicit (origination-weighted) gradient.

    log Z = data_logL_ln + (k/2)ln(tau) - 1/2 tau||p*-mu||^2 - 1/2 logdet(H_nats + tauI)

UNITS: H_nats = ln2 * H_bits (one ln2 -- true Hessian). BOUNDARY: floored theta
coords (== log2(1e-10)) with grad>0 are KKT-active -> dropped; omega logits are
unbounded (never floored). CUDA + float64 (A100).
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _load_families, _build_wave_layout,
    _sp_helpers_for_uniform, _build_leaf_E, _parse_fraction_missing,
)

_LN2 = math.log(2.0)
_THETA_MIN = math.log2(1e-10)
DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")


def _infer_dtl_model(theta_log2, cg_dir):
    if cg_dir:
        return "grouped"
    if torch.allclose(theta_log2, theta_log2[0:1].expand_as(theta_log2), atol=1e-9):
        return "global"
    return "free"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--prior-mu", type=float, default=None)
    ap.add_argument("--fd-eps", type=float, default=1e-3)
    ap.add_argument("--max-full-k", type=int, default=120)
    ap.add_argument("--lanczos-m", type=int, default=32)
    ap.add_argument("--lanczos-probes", type=int, default=3)
    ap.add_argument("--max-families", type=int, default=0)
    ap.add_argument("--extra-ale-dir", action="append", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (A100).")
    device = torch.device("cuda")
    dtype = torch.float64
    data_dir = Path(args.data_dir)

    meta = json.loads(Path(args.sidecar).read_text())
    root = meta["root"]
    fm_mode = meta.get("fm_mode", "e-only")
    if fm_mode == "both":
        raise SystemExit("fm_mode='both' inconsistent with the e-only forward; refit e-only.")
    _orig = meta.get("origination", {})
    _orig_strategy = _orig.get("origination", "uniform") if isinstance(_orig, dict) else "uniform"
    _opt_orig = _orig_strategy == "optimize"
    _prior_fit = meta.get("prior", {})
    _prior_kind = _prior_fit.get("prior", "none") if isinstance(_prior_fit, dict) else "none"

    theta_star = torch.tensor(meta["theta_log2"], dtype=dtype, device=device)  # [S,3]
    omega_star = None
    if _opt_orig:
        if "omega_log2" not in _orig:
            raise SystemExit("origination='optimize' sidecar lacks omega_log2.")
        omega_star = torch.tensor(_orig["omega_log2"], dtype=dtype, device=device)  # [S]
    cg = meta.get("clade_groups") or {}
    cg_dir = cg.get("clade_groups") if isinstance(cg, dict) else None
    min_species = int(meta.get("min_species", 1))

    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point, origination_log_pO
    from gpurec.core.forward import Pi_wave_forward
    from gpurec.optimization.implicit_grad import implicit_grad_loglik_vjp_wave
    from gpurec.core.log2_utils import logsumexp2 as _lse2, _safe_log2_internal as _slog2

    # ---- species tree + families ---------------------------------------------
    tree_path = data_dir / "rooted_phylogeny" / root
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"])
    sp_name_to_idx = sp["species_name_to_index"]
    ale_dirs = [str(data_dir / "ccps")] + list(args.extra_ale_dir or [])
    ale_paths = sorted(p for d in ale_dirs for p in glob.glob(str(Path(d) / "*.ale"))
                       if not Path(p).name.startswith("._"))
    fams, _ = _load_families(ale_paths, sp_name_to_idx, min_species=min_species,
                             dtype=dtype, limit=args.max_families)
    F = len(fams)
    wave_layout, root_clade_ids = _build_wave_layout(fams, device, dtype)
    sp_gpu, ancestors_T = _sp_helpers_for_uniform(sp, device, dtype)
    unnorm_row_max = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(
        device=device, dtype=dtype)
    leaf_E = None
    if fm_mode != "off":
        fm_path = data_dir / "fraction_missing"
        if fm_path.exists():
            fm, _n, _ = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
            leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
    dtl_model = _infer_dtl_model(theta_star, cg_dir)
    print(f"[load] root={root} S={S} F={F} dtl_model={dtl_model} "
          f"origination={_orig_strategy}", flush=True)

    # ---- groupings (DTL theta + origination omega) ---------------------------
    group_index = None
    omega_group_index = None
    if cg_dir:
        from clade_groups import (group_index_for_species_helpers,
                                  omega_group_index_for_species_helpers)
        cg_tree = str(Path(cg_dir) / "species_trees" / "starting_species_tree.newick")
        cg_mp = str(Path(cg_dir) / "model_parameters" / "model_parameters.txt")
        gi, _, _, _ = group_index_for_species_helpers(sp, cg_tree, cg_mp)
        group_index = gi.to(device)
        if _opt_orig:
            ogi, _no = omega_group_index_for_species_helpers(sp, cg_tree, cg_mp)
            omega_group_index = ogi.to(device) if ogi is not None else None

    # theta: model-coord <-> branch maps + initial param
    def expand_theta(tp):
        if dtl_model == "global":
            return tp.reshape(1, 3).expand(S, 3).contiguous()
        if dtl_model == "grouped":
            return tp.reshape(-1, 3).index_select(0, group_index)
        return tp.reshape(S, 3)

    def reduce_theta(gb):
        if dtl_model == "global":
            return gb.sum(0).reshape(-1)
        if dtl_model == "grouped":
            Gn = int(group_index.max().item()) + 1
            out = torch.zeros(Gn, 3, dtype=dtype, device=device)
            out.index_add_(0, group_index, gb)
            return out.reshape(-1)
        return gb.reshape(-1)

    def theta_param0():
        if dtl_model == "global":
            return theta_star[0].clone().reshape(-1)
        if dtl_model == "grouped":
            Gn = int(group_index.max().item()) + 1
            cnt = torch.zeros(Gn, dtype=dtype, device=device)
            cnt.index_add_(0, group_index, torch.ones_like(theta_star[:, 0]))
            gsum = torch.zeros(Gn, 3, dtype=dtype, device=device)
            gsum.index_add_(0, group_index, theta_star)
            return (gsum / cnt.clamp(min=1).unsqueeze(1)).reshape(-1)
        return theta_star.reshape(-1)

    # omega: grouped (G_O) / free (S) / absent
    _omega_grouped = _opt_orig and (omega_group_index is not None)

    def expand_omega(op):
        if not _opt_orig:
            return None
        if _omega_grouped:
            return op.index_select(0, omega_group_index)
        return op.reshape(S)

    def reduce_omega(gb):
        if _omega_grouped:
            Go = int(omega_group_index.max().item()) + 1
            out = torch.zeros(Go, dtype=dtype, device=device)
            out.index_add_(0, omega_group_index, gb)
            return out
        return gb.reshape(-1)

    def omega_param0():
        if not _opt_orig:
            return torch.zeros(0, dtype=dtype, device=device)
        if _omega_grouped:
            Go = int(omega_group_index.max().item()) + 1
            cnt = torch.zeros(Go, dtype=dtype, device=device)
            cnt.index_add_(0, omega_group_index, torch.ones_like(omega_star))
            osum = torch.zeros(Go, dtype=dtype, device=device)
            osum.index_add_(0, omega_group_index, omega_star)
            return osum / cnt.clamp(min=1)
        return omega_star.reshape(-1)

    tp0 = theta_param0()
    op0 = omega_param0()
    n_tp = tp0.numel()
    n_op = op0.numel()
    p0 = torch.cat([tp0, op0])
    k = p0.numel()
    warm = [None]

    def joint_grad(p):
        """Exact reduced gradient [k] (bits) of sum_f NLL_f at param p=[tp;op]."""
        tp = p[:n_tp]
        theta_branch = expand_theta(tp)
        omega_branch = expand_omega(p[n_tp:]) if _opt_orig else None
        log_pO = origination_log_pO(omega_branch) if omega_branch is not None else None

        log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
            theta_branch, unnorm_row_max, specieswise=True)
        E_out = E_fixed_point(
            species_helpers=sp_gpu, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
            transfer_mat=transfer_mat, max_transfer_mat=mt, max_iters=4000,
            tolerance=1e-11, warm_start_E=warm[0], dtype=dtype, device=device,
            pibar_mode="uniform", ancestors_T=ancestors_T, leaf_E=leaf_E)
        warm[0] = E_out["E"].detach()
        Pi_b = Pi_wave_forward(
            wave_layout=wave_layout, species_helpers=sp_gpu, E=E_out["E"],
            Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, transfer_mat=transfer_mat,
            max_transfer_mat=mt, device=device, dtype=dtype, pibar_mode="uniform",
            leaf_obs_log=None)
        g_theta, _ = implicit_grad_loglik_vjp_wave(
            wave_layout, sp_gpu, Pi_star_wave=Pi_b["Pi_wave_ordered"],
            Pibar_star_wave=Pi_b["Pibar_wave_ordered"], E_star=E_out["E"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"], Ebar=E_out["E_bar"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, max_transfer_mat=mt,
            root_clade_ids_perm=wave_layout["root_clade_ids"], theta=theta_branch,
            unnorm_row_max=unnorm_row_max, specieswise=True, device=device,
            dtype=dtype, pibar_mode="uniform", ancestors_T=ancestors_T,
            leaf_E=leaf_E, leaf_obs_log=None, log_pO=log_pO)
        g = reduce_theta(g_theta)
        if _opt_orig:
            # DIRECT omega gradient (mirrors wave_optimizer): autograd through the
            # origination-weighted numerators + shared survival denom at fixed Pi,E.
            root_pi_all = Pi_b["Pi"][wave_layout["root_clade_ids"]].detach()  # [n_fam,S]
            E_det = E_out["E"].detach()
            log_one_minus_E = _slog2(1.0 - torch.exp2(E_det))
            with torch.enable_grad():
                ob = omega_branch.detach().clone().requires_grad_(True)
                log_pO_g = ob - _lse2(ob, dim=-1, keepdim=True)
                num = _lse2(root_pi_all + log_pO_g, dim=-1)            # [n_fam]
                denom = _lse2(log_one_minus_E + log_pO_g, dim=-1)      # scalar
                nll_omega = -(num - denom).sum()
                g_omega_branch = torch.autograd.grad(nll_omega, ob)[0].detach()
            g = torch.cat([g, reduce_omega(g_omega_branch)])
        return g

    eps = float(args.fd_eps)
    tau = float(args.tau)
    g0 = joint_grad(p0)                                    # [k] bits
    grad_inf = float(g0.abs().max().item())

    # ---- boundary: theta floored coords (omega logits never floored) ---------
    floored = torch.zeros(k, dtype=torch.bool, device=device)
    floored[:n_tp] = p0[:n_tp] <= (_THETA_MIN + 1e-6)
    kkt_active = floored & (g0 > 1e-6)
    keep = ~kkt_active
    keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
    k_eff = int(keep.sum())
    p_U = p0[keep]
    n_floored = int(floored.sum())
    n_active = int(kkt_active.sum())
    print(f"[boundary] k={k} (theta {n_tp} + omega {n_op}) floored={n_floored} "
          f"dropped={n_active} k_eff={k_eff}; ||grad0||_inf={grad_inf:.3e}", flush=True)

    def hv_keep(v_keep):
        v_full = torch.zeros(k, dtype=dtype, device=device)
        v_full[keep_idx] = v_keep
        gp = joint_grad(p0 + eps * v_full)
        gm = joint_grad(p0 - eps * v_full)
        return _LN2 * ((gp - gm) / (2.0 * eps))[keep_idx]

    if k_eff <= args.max_full_k:
        print(f"[hess] FULL finite-diff Hessian: {2 * k_eff} solves", flush=True)
        H_U = torch.zeros(k_eff, k_eff, dtype=dtype, device=device)
        for j in range(k_eff):
            ej = torch.zeros(k_eff, dtype=dtype, device=device)
            ej[j] = 1.0
            H_U[:, j] = hv_keep(ej)
            if (j + 1) % 20 == 0:
                print(f"[hess]   col {j + 1}/{k_eff}", flush=True)
        H_U = 0.5 * (H_U + H_U.t())
        A = H_U + tau * torch.eye(k_eff, dtype=dtype, device=device)
        evA = torch.linalg.eigvalsh(0.5 * (A + A.t()))
        n_neg = int((evA <= 0).sum())
        logdet = float(torch.log(evA.clamp(min=tau * 1e-6)).sum())
        evH = torch.linalg.eigvalsh(H_U).clamp(min=0)
        p_eff = float((evH / (evH + tau)).sum())
        top5 = evH.flip(0)[:5].tolist()
        bot5 = evH[:5].tolist()
        null_dim = int((evH < 1e-8 * evH.max().clamp(min=1)).sum())
        hess_method = "full_fd"
        n_hv = k_eff
    else:
        m = int(args.lanczos_m)
        n_probe = int(args.lanczos_probes)
        print(f"[hess] LANCZOS+SLQ: m={m} probes={n_probe} -> ~{2 * m * n_probe} "
              f"solves (vs {2 * k_eff} full)", flush=True)

        def matvec_A(w):
            return hv_keep(w) + tau * w

        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260619)
        logdet_acc = 0.0
        peff_acc = 0.0
        ritz_top = []
        for _p in range(n_probe):
            v = (torch.randint(0, 2, (k_eff,), generator=gen).to(dtype) * 2 - 1).to(device)
            alphas, betas, Q = [], [], []
            q = v / torch.linalg.vector_norm(v)
            Q.append(q)
            w = matvec_A(q)
            a = float(torch.dot(w, q))
            alphas.append(a)
            w = w - a * q
            for _j in range(1, m):
                b = float(torch.linalg.vector_norm(w))
                if b < 1e-10:
                    break
                betas.append(b)
                qn = w / b
                for qi in Q:
                    qn = qn - torch.dot(qn, qi) * qi
                qn = qn / torch.linalg.vector_norm(qn)
                Q.append(qn)
                w = matvec_A(qn)
                a = float(torch.dot(w, qn))
                alphas.append(a)
                w = w - a * qn - b * Q[-2]
            mm = len(alphas)
            T = torch.zeros(mm, mm, dtype=dtype, device=device)
            for i in range(mm):
                T[i, i] = alphas[i]
            for i in range(len(betas)):
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]
            ev, evec = torch.linalg.eigh(T)
            wt = evec[0, :] ** 2
            evc = ev.clamp(min=tau * 1e-6)
            logdet_acc += k_eff * float((wt * torch.log(evc)).sum())
            peff_acc += k_eff * float((wt * ((ev - tau) / evc)).sum())
            ritz_top.append(float(ev.max()))
        logdet = logdet_acc / n_probe
        p_eff = peff_acc / n_probe
        n_neg = 0
        top5 = sorted(ritz_top, reverse=True)[:5]
        bot5 = []
        null_dim = None
        hess_method = "lanczos_slq"
        n_hv = m * n_probe

    mu = float(args.prior_mu) if args.prior_mu is not None else float(p_U.mean())
    data_logL_ln = float(meta["data_log_likelihood_ln"])
    prior_quad = 0.5 * tau * float(((p_U - mu) ** 2).sum())
    log_prior = 0.5 * k_eff * math.log(tau) - prior_quad
    occam = -0.5 * logdet
    log_Z = data_logL_ln + log_prior + occam

    out = {
        "sidecar": str(args.sidecar), "root": root, "dtl_model": dtl_model,
        "origination": _orig_strategy, "fm_mode": fm_mode, "hessian": hess_method,
        "n_hv": n_hv, "fd_eps": eps, "F_families": F,
        "n_families_kept_driver": meta.get("n_families_kept"),
        "k_nominal": int(k), "n_theta_param": n_tp, "n_omega_param": n_op,
        "k_eff": k_eff, "n_floored": n_floored, "n_kkt_active_dropped": n_active,
        "n_neg_eig_posterior": n_neg, "tau": tau, "prior_mu": mu,
        "data_log_likelihood_ln": data_logL_ln, "log_prior": log_prior,
        "occam_factor": occam, "logdet_posterior": logdet, "log_Z": log_Z,
        "p_eff": p_eff, "p_eff_minus_k": p_eff - k_eff, "null_dim": null_dim,
        "data_grad_inf_bits": grad_inf, "prior_fit": _prior_kind,
        "eig_H_top5": top5, "eig_H_bot5": bot5,
    }
    out_path = args.out or (str(args.sidecar).replace(".rates.txt.json", "") + ".evidence.json")
    Path(out_path).write_text(json.dumps(out, indent=2))
    print("=" * 70)
    for key in ["root", "dtl_model", "origination", "k_nominal", "k_eff",
                "data_log_likelihood_ln", "log_Z", "p_eff", "n_neg_eig_posterior"]:
        print(f"  {key:24s} {out[key]}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
