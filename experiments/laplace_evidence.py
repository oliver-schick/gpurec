"""Laplace marginal-likelihood (evidence) for gpurec -- Bayesian rooting + model
selection. Hessian via FINITE-DIFFERENCING THE EXACT BATCHED GRADIENT.

For a (root, model) MAP fit theta* (a driver sidecar *.rates.txt.json):

    log Z = log L_data(theta*) + log prior(theta*) + (k/2) ln(2pi) - 1/2 log det H

with H = posterior precision at theta*. The observed-information Hessian of the
NEGATIVE log-likelihood is built by central finite differences of the EXACT
gradient gpurec already computes (implicit differentiation): for each free param
j in the MODEL's coordinates,

    H[:,j] = ( grad(theta* + eps e_j) - grad(theta* - eps e_j) ) / (2 eps)

This sidesteps the per-family empirical-Fisher path (whose genewise reuse did
not validate) and is exact (no estimator bias) for the small models where the
rooting decision lives: global (k=3) and clade-grouped (k~51). Cost: 2k batched
forward+backward solves per root (k=3 -> 6; k=51 -> ~102). The free-per-branch
(k=357) / bwL2 (k~476, with origination) models need the Lanczos variant
(separate); full-Hessian finite diff does not scale there -- this script aborts
with a clear message if k exceeds --max-full-k.

UNITS (critical): gpurec NLL is in log2/bits; theta is log2-rate. The NAT
observed information is H_nats = ln2 * H_bits (ONE factor of ln2 -- the true
Hessian scales linearly with the objective base, unlike the empirical Fisher
outer product which scales as ln2^2). data_log_likelihood_ln is already in nats.

PRIOR: proper isotropic Gaussian ridge N(mu, tau^-1 I) on theta (log2-rate):
    log Z = data_logL_ln + (k/2)ln(tau) - 1/2 tau||theta*-mu||^2 - 1/2 logdet(H_nats + tau I).
Same tau across roots/models -> a FAIR proper Bayesian comparison. Occam factor
-1/2 logdet penalizes curvature/dimension; p_eff = tr(H(H+tauI)^-1) = effective
#params (clamped to the PSD part; FD Hessians can have small negative eigenvalues
from solver noise / saddle directions, which we floor).

BOUNDARY: rates at the floor (theta == _THETA_MIN) with a gradient pushing
further down are KKT-active -> dropped (reduced k_eff), deleting the rows/cols.

CUDA + float64 required (A100). Uniform origination only (asserts).
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


def _infer_model(theta_log2, cg_dir):
    if cg_dir:
        return "grouped"
    if torch.allclose(theta_log2, theta_log2[0:1].expand_as(theta_log2), atol=1e-9):
        return "global"
    return "free"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="ridge prior precision on theta (log2-rate); same across roots.")
    ap.add_argument("--prior-mu", type=float, default=None,
                    help="ridge prior centre (log2-rate). Default: theta* (kept-coord) mean.")
    ap.add_argument("--fd-eps", type=float, default=1e-3,
                    help="central finite-diff step (log2 units) for Hessian-vector products.")
    ap.add_argument("--max-full-k", type=int, default=120,
                    help="k_eff <= this -> exact full FD Hessian (2k Hv); else Lanczos+SLQ.")
    ap.add_argument("--lanczos-m", type=int, default=32,
                    help="Lanczos steps for the SLQ log-det/p_eff estimate (large-k path).")
    ap.add_argument("--lanczos-probes", type=int, default=3,
                    help="stochastic probe vectors for SLQ (large-k path).")
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
        raise SystemExit("fm_mode='both' inconsistent with the e-only forward here; refit e-only.")
    _orig = meta.get("origination", {})
    _orig_strategy = _orig.get("origination", "uniform") if isinstance(_orig, dict) else "uniform"
    if _orig_strategy != "uniform":
        raise SystemExit(
            f"origination='{_orig_strategy}': this full-Hessian script handles uniform O "
            "only (the rooting param is theta). Use the Lanczos/omega variant for bwL2.")
    _prior_fit = meta.get("prior", {})
    _prior_kind = _prior_fit.get("prior", "none") if isinstance(_prior_fit, dict) else "none"

    theta_star = torch.tensor(meta["theta_log2"], dtype=dtype, device=device)  # [S,3]
    cg = meta.get("clade_groups") or {}
    cg_dir = cg.get("clade_groups") if isinstance(cg, dict) else None
    min_species = int(meta.get("min_species", 1))

    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.forward import Pi_wave_forward
    from gpurec.optimization.implicit_grad import implicit_grad_loglik_vjp_wave

    # ---- species tree + families (same loader/min_species as the driver) -----
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
    print(f"[load] root={root} S={S} families={F} (driver {meta.get('n_families_kept','?')})",
          flush=True)

    leaf_E = None
    if fm_mode != "off":
        fm_path = data_dir / "fraction_missing"
        if fm_path.exists():
            fm, n_set, _ = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
            leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)

    # ---- grouping (for model coords) -----------------------------------------
    model = _infer_model(theta_star, cg_dir)
    group_index = None
    if cg_dir:
        from clade_groups import group_index_for_species_helpers
        gi, _, _, _ = group_index_for_species_helpers(
            sp, str(Path(cg_dir) / "species_trees" / "starting_species_tree.newick"),
            str(Path(cg_dir) / "model_parameters" / "model_parameters.txt"))
        group_index = gi.to(device)
    print(f"[model] {model}", flush=True)

    def expand(param):
        """model-coord param -> branch theta [S,3]."""
        if model == "global":
            return param.reshape(1, 3).expand(S, 3).contiguous()
        if model == "grouped":
            return param.reshape(-1, 3).index_select(0, group_index)
        return param.reshape(S, 3)

    def reduce_grad(g_branch):
        """branch grad [S,3] -> model-coord grad [k]."""
        if model == "global":
            return g_branch.sum(0).reshape(-1)
        if model == "grouped":
            Gn = int(group_index.max().item()) + 1
            out = torch.zeros(Gn, 3, dtype=dtype, device=device)
            out.index_add_(0, group_index, g_branch)
            return out.reshape(-1)
        return g_branch.reshape(-1)

    def param0():
        if model == "global":
            return theta_star[0].clone().reshape(-1)
        if model == "grouped":
            Gn = int(group_index.max().item()) + 1
            cnt = torch.zeros(Gn, dtype=dtype, device=device)
            cnt.index_add_(0, group_index, torch.ones_like(theta_star[:, 0]))
            gsum = torch.zeros(Gn, 3, dtype=dtype, device=device)
            gsum.index_add_(0, group_index, theta_star)
            return (gsum / cnt.clamp(min=1).unsqueeze(1)).reshape(-1)
        return theta_star.reshape(-1)

    tp0 = param0()
    k = tp0.numel()
    if k > args.max_full_k:
        raise SystemExit(
            f"k={k} > max-full-k={args.max_full_k}: full-Hessian finite diff is too "
            f"expensive ({2*k} solves). Use the Lanczos variant for this model.")

    warm = [None]

    def batched_grad(theta_branch):
        """Exact gradient d(sum_f NLL_f)/d(theta) [S,3] (bits) at theta_branch."""
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
        g, _ = implicit_grad_loglik_vjp_wave(
            wave_layout, sp_gpu, Pi_star_wave=Pi_b["Pi_wave_ordered"],
            Pibar_star_wave=Pi_b["Pibar_wave_ordered"], E_star=E_out["E"],
            E_s1=E_out["E_s1"], E_s2=E_out["E_s2"], Ebar=E_out["E_bar"],
            log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, max_transfer_mat=mt,
            root_clade_ids_perm=wave_layout["root_clade_ids"], theta=theta_branch,
            unnorm_row_max=unnorm_row_max, specieswise=True, device=device,
            dtype=dtype, pibar_mode="uniform", ancestors_T=ancestors_T,
            leaf_E=leaf_E, leaf_obs_log=None)
        return g

    # ---- finite-diff Hessian in model coords (central) -----------------------
    eps = float(args.fd_eps)
    tau = float(args.tau)
    g0 = reduce_grad(batched_grad(expand(tp0)))            # [k] bits
    grad_inf = float(g0.abs().max().item())

    # ---- boundary / floored params (drop KKT-active before curvature) --------
    floored = tp0 <= (_THETA_MIN + 1e-6)
    kkt_active = floored & (g0 > 1e-6)
    keep = ~kkt_active
    keep_idx = torch.nonzero(keep, as_tuple=False).flatten()
    k_eff = int(keep.sum())
    tp_U = tp0[keep]
    n_floored = int(floored.sum())
    n_active = int(kkt_active.sum())
    print(f"[boundary] k={k} floored={n_floored} kkt_active(dropped)={n_active} "
          f"k_eff={k_eff}; ||grad0||_inf={grad_inf:.3e}", flush=True)

    # Hessian-VECTOR product on kept coords (NAT units) via central FD of the
    # EXACT gradient: 2 grad evals per Hv, INDEPENDENT of k_eff.
    def hv_keep(v_keep):
        v_full = torch.zeros(k, dtype=dtype, device=device)
        v_full[keep_idx] = v_keep
        gp = reduce_grad(batched_grad(expand(tp0 + eps * v_full)))
        gm = reduce_grad(batched_grad(expand(tp0 - eps * v_full)))
        return _LN2 * ((gp - gm) / (2.0 * eps))[keep_idx]

    # ---- curvature: exact full Hessian (2k Hv) for small k, else Lanczos+SLQ --
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
        for p in range(n_probe):
            v = (torch.randint(0, 2, (k_eff,), generator=gen).to(dtype) * 2 - 1).to(device)
            alphas, betas, Q = [], [], []
            q = v / torch.linalg.vector_norm(v)
            Q.append(q)
            w = matvec_A(q)
            a = float(torch.dot(w, q))
            alphas.append(a)
            w = w - a * q
            for j in range(1, m):
                b = float(torch.linalg.vector_norm(w))
                if b < 1e-10:
                    break
                betas.append(b)
                qn = w / b
                for qi in Q:                       # full reorthogonalization
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
            wt = evec[0, :] ** 2                   # SLQ quadrature weights
            evc = ev.clamp(min=tau * 1e-6)         # eigenvalues of A=H+tauI (PD)
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

    # ---- evidence (ridge Gaussian prior) -------------------------------------
    mu = float(args.prior_mu) if args.prior_mu is not None else float(tp_U.mean())
    data_logL_ln = float(meta["data_log_likelihood_ln"])
    prior_quad = 0.5 * tau * float(((tp_U - mu) ** 2).sum())
    log_prior = 0.5 * k_eff * math.log(tau) - prior_quad
    occam = -0.5 * logdet
    log_Z = data_logL_ln + log_prior + occam

    out = {
        "sidecar": str(args.sidecar), "root": root, "model": model, "fm_mode": fm_mode,
        "hessian": hess_method, "n_hv": n_hv, "fd_eps": eps, "F_families": F,
        "n_families_kept_driver": meta.get("n_families_kept"),
        "k_nominal": int(k), "k_eff": k_eff, "n_floored": n_floored,
        "n_kkt_active_dropped": n_active, "n_neg_eig_posterior": n_neg,
        "tau": tau, "prior_mu": mu, "data_log_likelihood_ln": data_logL_ln,
        "log_prior": log_prior, "occam_factor": occam, "logdet_posterior": logdet,
        "log_Z": log_Z, "p_eff": p_eff, "p_eff_minus_k": p_eff - k_eff,
        "null_dim": null_dim, "data_grad_inf_bits": grad_inf, "prior_fit": _prior_kind,
        "eig_H_top5": top5, "eig_H_bot5": bot5,
    }
    out_path = args.out or (str(args.sidecar).replace(".rates.txt.json", "") + ".evidence.json")
    Path(out_path).write_text(json.dumps(out, indent=2))
    print("=" * 70)
    for key in ["root", "model", "k_nominal", "k_eff", "data_log_likelihood_ln",
                "log_Z", "p_eff", "n_neg_eig_posterior", "null_dim"]:
        print(f"  {key:24s} {out[key]}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
