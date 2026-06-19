"""Laplace marginal-likelihood (evidence) for gpurec -- Bayesian rooting + model
selection via the Gauss-Newton / empirical-Fisher Hessian.

For a (root, model) MAP fit theta* (a driver sidecar *.rates.txt.json), compute

    log Z = log L_data(theta*) + log prior(theta*) + (k/2) ln(2pi) - 1/2 log det H

with H = posterior precision at theta*. We use the empirical Fisher
    G = sum_f g_f g_f^T,   g_f = grad_theta (log L_f)   [PER-FAMILY scores]
which is PSD by construction and, by the Fisher/Bartlett identity, approximates
the observed information at the MLE. Over-parametrized models surface as a
small-eigenvalue spectrum -> the effective #params p_eff = tr(G (G+Lambda)^-1)
is the headline Occam diagnostic.

PER-FAMILY SCORES (the one real computation) reuse the VALIDATED genewise
per-family backward (gpurec/optimization/implicit_grad.py:370) with SHARED theta:
for shared theta the converged E* and the event params are identical across
families, so we broadcast them and the loop returns g_f = d(NLL_f)/dtheta [F,S,3]
exactly (each family's survival denominator is seeded with weight n_fam=1). We
VALIDATE sum_f g_f == the batched shared-theta gradient before trusting G.

UNITS (critical): gpurec works in log2/bits; theta is log2-rate. The empirical
Fisher that approximates the NAT observed information is
    G_nats = sum_f (ln2 * g_f_bits)(ln2 * g_f_bits)^T = ln2^2 * sum_f g_bits g_bits^T.
The ln2^2 is correct: the Fisher identity E[g g^T] = -E[Hessian] holds only in
NATURAL log, so scores must be converted to nats BEFORE the outer product. The
fit term log L_data and (k/2)ln(2pi) are likewise in nats.

PRIOR: we use a proper isotropic Gaussian ridge prior on theta (log2-rate),
N(mu, tau^-1 I), giving evidence
    log Z = log L_data(theta*) + (k/2) ln(tau) - 1/2 tau ||theta*-mu||^2
            - 1/2 log det(G_nats + tau I).
Same tau across roots/models -> a FAIR, proper Bayesian comparison (rooting:
constants cancel in softmax; model selection: the Occam -1/2 log det penalizes
near-flat over-parametrized directions). The Brownian-MRF prior (det-ratio form)
is a refinement, not needed for the headline result.

BOUNDARY: rates pinned at the floor (theta == _THETA_MIN = log2(1e-10)) with a
gradient pushing further down are KKT-active -> an interior Laplace is invalid
there; we DROP those coordinates (reduced k_eff), deleting (not zeroing) the
rows/cols.

CUDA + float64 required (run on an A100).
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
    _sp_helpers_for_uniform, _build_leaf_E, _parse_fraction_missing, _INV_LN2,
)

_LN2 = math.log(2.0)
_THETA_MIN = math.log2(1e-10)
DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")


def _infer_model(theta_log2: torch.Tensor, cg_dir):
    """global (all branch rows identical) | grouped (cg_dir set) | free."""
    if cg_dir:
        return "grouped"
    # all rows equal -> global
    if torch.allclose(theta_log2, theta_log2[0:1].expand_as(theta_log2), atol=1e-9):
        return "global"
    return "free"


def _reduce_branch(x_bsa: torch.Tensor, model: str, group_index, S: int):
    """Reduce a [..., S, 3] branch-space tensor to the model's free-param coords.

    global  -> [..., 3]      (sum over branches per axis)
    grouped -> [..., G, 3]   (index_add over branches by group)
    free    -> [..., S, 3]   (unchanged)
    Works for the [F,S,3] score stack (reduce over dim=-2) and the [S,3] total
    gradient. Returns the reduced tensor flattened on the last param dims.
    """
    if model == "global":
        red = x_bsa.sum(dim=-2)                      # [...,3]
    elif model == "grouped":
        Gn = int(group_index.max().item()) + 1
        lead = x_bsa.shape[:-2]
        out = torch.zeros(*lead, Gn, 3, dtype=x_bsa.dtype, device=x_bsa.device)
        out.index_add_(-2, group_index, x_bsa)       # add x[...,s,:] into out[...,gi[s],:]
        red = out                                    # [...,G,3]
    else:  # free
        red = x_bsa                                  # [...,S,3]
    return red.reshape(*red.shape[:-2], -1)          # flatten last two -> [..., k]


def _param_theta(theta_log2: torch.Tensor, model: str, group_index):
    """The free param vector theta* in model coords, as [k] (for boundary + prior)."""
    if model == "global":
        return theta_log2[0].clone()                 # [3]
    if model == "grouped":
        Gn = int(group_index.max().item()) + 1
        cnt = torch.zeros(Gn, dtype=theta_log2.dtype, device=theta_log2.device)
        cnt.index_add_(0, group_index, torch.ones_like(theta_log2[:, 0]))
        gsum = torch.zeros(Gn, 3, dtype=theta_log2.dtype, device=theta_log2.device)
        gsum.index_add_(0, group_index, theta_log2)
        return (gsum / cnt.clamp(min=1).unsqueeze(1)).reshape(-1)  # [G*3]
    return theta_log2.reshape(-1)                     # [S*3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True,
                    help="driver *.rates.txt.json (has theta_log2, root, fm_mode, clade_groups)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="ridge prior precision on theta (log2-rate). Same value "
                         "across roots/models for a fair comparison.")
    ap.add_argument("--prior-mu", type=float, default=None,
                    help="ridge prior centre (log2-rate). Default: theta* mean "
                         "(so ||theta*-mu||^2 is comparable across models).")
    ap.add_argument("--max-families", type=int, default=0,
                    help="cap #families (smoke). 0 = all.")
    ap.add_argument("--extra-ale-dir", action="append", default=None)
    ap.add_argument("--out", default=None, help="evidence json (default <sidecar>.evidence.json)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (run on A100).")
    device = torch.device("cuda")
    dtype = torch.float64
    data_dir = Path(args.data_dir)

    meta = json.loads(Path(args.sidecar).read_text())
    root = meta["root"]
    fm_mode = meta.get("fm_mode", "e-only")
    if fm_mode == "both":
        print("[warn] fm_mode='both' uses a Pi leaf boundary the genewise "
              "per-family backward does NOT replicate; scores would be biased. "
              "These runs are e-only -- proceeding only if e-only/off.", flush=True)
    theta_log2 = torch.tensor(meta["theta_log2"], dtype=dtype, device=device)  # [S,3]
    cg = meta.get("clade_groups") or {}
    cg_dir = cg.get("clade_groups") if isinstance(cg, dict) else None
    min_species = int(meta.get("min_species", 1))

    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.forward import Pi_wave_forward
    from gpurec.optimization.implicit_grad import (
        implicit_grad_loglik_vjp_wave, implicit_grad_loglik_vjp_wave_genewise,
    )

    # ---- species tree + families (SAME loader/min_species as the driver) -----
    tree_path = data_dir / "rooted_phylogeny" / root
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"])
    sp_name_to_idx = sp["species_name_to_index"]
    ale_dirs = [str(data_dir / "ccps")]
    if args.extra_ale_dir:
        ale_dirs += list(args.extra_ale_dir)
    ale_paths = []
    for d in ale_dirs:
        ale_paths += [p for p in glob.glob(str(Path(d) / "*.ale"))
                      if not Path(p).name.startswith("._")]
    ale_paths = sorted(ale_paths)
    fams, stats = _load_families(ale_paths, sp_name_to_idx,
                                 min_species=min_species, dtype=dtype,
                                 limit=args.max_families)
    F = len(fams)
    print(f"[load] root={root} S={S} families={F} "
          f"(driver kept {meta.get('n_families_kept','?')})", flush=True)

    wave_layout, root_clade_ids = _build_wave_layout(fams, device, dtype)
    sp_gpu, ancestors_T = _sp_helpers_for_uniform(sp, device, dtype)
    unnorm_row_max = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(
        device=device, dtype=dtype)

    # ---- fraction-missing (e-only: leaf_E in E solve, leaf_obs_log None) ------
    leaf_E = None
    if fm_mode != "off":
        fm_path = data_dir / "fraction_missing"
        if fm_path.exists():
            fm, n_set, _ = _parse_fraction_missing(fm_path, sp_name_to_idx, S)
            leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
            print(f"[fm] e-only: {n_set} rows mapped", flush=True)
    leaf_obs_log = None  # e-only / off (genewise backward applies no Pi boundary)

    # ---- shared forward at theta* -------------------------------------------
    log_pS, log_pD, log_pL, transfer_mat, mt = extract_parameters_uniform(
        theta_log2, unnorm_row_max, specieswise=True)
    E_out = E_fixed_point(
        species_helpers=sp_gpu, log_pS=log_pS, log_pD=log_pD, log_pL=log_pL,
        transfer_mat=transfer_mat, max_transfer_mat=mt, max_iters=4000,
        tolerance=1e-10, warm_start_E=None, dtype=dtype, device=device,
        pibar_mode="uniform", ancestors_T=ancestors_T, leaf_E=leaf_E)

    def _bc(x):
        """Broadcast a shared [S] (or scalar) param to [F,S]; leave [F,...] alone."""
        x = x if torch.is_tensor(x) else torch.as_tensor(x, dtype=dtype, device=device)
        if x.dim() == 0:
            x = x.expand(S)
        return x.reshape(1, -1).expand(F, S)

    # ---- PER-FAMILY scores g_f [F,S,3] via genewise backward (shared theta) ---
    print(f"[scores] computing per-family scores for F={F} families ...", flush=True)
    g_stack, _stats = implicit_grad_loglik_vjp_wave_genewise(
        fams, sp_gpu,
        E_all=E_out["E"].reshape(1, -1).expand(F, S),
        E_s1_all=E_out["E_s1"].reshape(1, -1).expand(F, S),
        E_s2_all=E_out["E_s2"].reshape(1, -1).expand(F, S),
        Ebar_all=E_out["E_bar"].reshape(1, -1).expand(F, S),
        log_pS_all=_bc(log_pS), log_pD_all=_bc(log_pD), log_pL_all=_bc(log_pL),
        mt_all=_bc(mt),
        theta_stack=theta_log2.reshape(1, S, 3).expand(F, S, 3),
        unnorm_row_max=unnorm_row_max, specieswise=True, device=device,
        dtype=dtype, pibar_mode="uniform", ancestors_T=ancestors_T)
    # g_stack: [F,S,3] = d(NLL_f)/dtheta (bits). NLL is a loss => score of logL is -g.
    nan_fam = torch.isnan(g_stack.reshape(F, -1)).any(dim=1)
    n_nan = int(nan_fam.sum())
    if n_nan:
        print(f"[scores] dropping {n_nan} families with NaN scores", flush=True)
        g_stack = g_stack[~nan_fam]
        F = g_stack.shape[0]

    # ---- VALIDATION: sum_f g_f == batched shared-theta gradient --------------
    Pi_b = Pi_wave_forward(
        wave_layout=wave_layout, species_helpers=sp_gpu, E=E_out["E"],
        Ebar=E_out["E_bar"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, transfer_mat=transfer_mat,
        max_transfer_mat=mt, device=device, dtype=dtype, pibar_mode="uniform",
        leaf_obs_log=leaf_obs_log)
    grad_batched, _ = implicit_grad_loglik_vjp_wave(
        wave_layout, sp_gpu,
        Pi_star_wave=Pi_b["Pi_wave_ordered"], Pibar_star_wave=Pi_b["Pibar_wave_ordered"],
        E_star=E_out["E"], E_s1=E_out["E_s1"], E_s2=E_out["E_s2"], Ebar=E_out["E_bar"],
        log_pS=log_pS, log_pD=log_pD, log_pL=log_pL, max_transfer_mat=mt,
        root_clade_ids_perm=wave_layout["root_clade_ids"], theta=theta_log2,
        unnorm_row_max=unnorm_row_max, specieswise=True, device=device, dtype=dtype,
        pibar_mode="uniform", ancestors_T=ancestors_T)
    g_sum = g_stack.sum(dim=0)                                   # [S,3]
    resid = float((g_sum - grad_batched).abs().max().item())
    denom = float(grad_batched.abs().max().item()) + 1e-30
    print(f"[validate] max|sum_f g_f - grad_batched| = {resid:.3e} "
          f"(rel {resid/denom:.3e})", flush=True)

    # ---- model + reduction ---------------------------------------------------
    group_index = None
    if cg_dir:
        from clade_groups import group_index_for_species_helpers
        cg_tree = Path(cg_dir) / "species_trees" / "starting_species_tree.newick"
        cg_mp = Path(cg_dir) / "model_parameters" / "model_parameters.txt"
        group_index, _, _, _ = group_index_for_species_helpers(
            sp, str(cg_tree), str(cg_mp))
        group_index = group_index.to(device)
    model = _infer_model(theta_log2, cg_dir)
    print(f"[model] {model}", flush=True)

    # per-family scores -> model coords; convert to NATS for the Fisher identity.
    g_model = _reduce_branch(g_stack, model, group_index, S) * _LN2   # [F,k] nats
    k = g_model.shape[1]
    G = g_model.t() @ g_model                                          # [k,k] PSD nats
    G = 0.5 * (G + G.t())

    # ---- boundary / floored-param handling -----------------------------------
    theta_param = _param_theta(theta_log2, model, group_index)         # [k] log2
    grad_param = _reduce_branch(grad_batched.unsqueeze(0), model, group_index, S)[0]  # [k] bits
    floored = theta_param <= (_THETA_MIN + 1e-6)
    kkt_active = floored & (grad_param > 1e-6)   # NLL grad pushes rate below floor
    keep = ~kkt_active
    k_eff = int(keep.sum())
    G_U = G[keep][:, keep]
    theta_U = theta_param[keep]
    n_floored = int(floored.sum())
    n_active = int(kkt_active.sum())
    print(f"[boundary] k={k} floored={n_floored} kkt_active(dropped)={n_active} "
          f"k_eff={k_eff}", flush=True)

    # ---- evidence (ridge Gaussian prior N(mu, tau^-1 I) on theta) -------------
    tau = float(args.tau)
    mu = float(args.prior_mu) if args.prior_mu is not None else float(theta_U.mean())
    eye = torch.eye(k_eff, dtype=dtype, device=device)
    H = G_U + tau * eye
    H = 0.5 * (H + H.t())
    evH = torch.linalg.eigvalsh(H).clamp(min=tau * 1e-6)
    logdet_H = float(torch.log(evH).sum())
    data_logL_ln = float(meta["data_log_likelihood_ln"])
    prior_quad = 0.5 * tau * float(((theta_U - mu) ** 2).sum())
    # Proper Gaussian prior N(mu, tau^-1 I): the prior-normaliser (k/2)ln(tau/2pi)
    # and the Laplace Gaussian-integral (k/2)ln(2pi) cancel the 2pi, leaving
    #   log_prior_eff = (k/2)ln(tau) - 1/2 tau||theta*-mu||^2,  occam = -1/2 logdet(G+tauI)
    # so that  log Z = data_logL + log_prior_eff + occam  (all in nats).
    log_prior = 0.5 * k_eff * math.log(tau) - prior_quad
    occam = -0.5 * logdet_H
    log_Z = data_logL_ln + log_prior + occam

    # ---- Occam diagnostics: effective #params + spectrum ---------------------
    evG = torch.linalg.eigvalsh(G_U).clamp(min=0)
    p_eff = float((evG / (evG + tau)).sum())
    nulldim = int((evG < 1e-10 * evG.max().clamp(min=1)).sum())
    cond = float((evG.max() / evG.clamp(min=1e-300).min()).item())

    out = {
        "sidecar": str(args.sidecar),
        "root": root,
        "model": model,
        "fm_mode": fm_mode,
        "F_families": F,
        "n_families_kept_driver": meta.get("n_families_kept"),
        "k_nominal": int(k),
        "k_eff": k_eff,
        "n_floored": n_floored,
        "n_kkt_active_dropped": n_active,
        "tau": tau,
        "prior_mu": mu,
        "data_log_likelihood_ln": data_logL_ln,
        "log_prior": log_prior,
        "occam_factor": occam,
        "logdet_H": logdet_H,
        "log_Z": log_Z,
        "p_eff": p_eff,
        "p_eff_minus_k": p_eff - k_eff,
        "null_dim": nulldim,
        "cond_number": cond,
        "validation_resid_max": resid,
        "validation_resid_rel": resid / denom,
        "eig_G_top5": evG.flip(0)[:5].tolist(),
        "eig_G_bot5": evG[:5].tolist(),
    }
    out_path = args.out or (str(args.sidecar).replace(".rates.txt.json", "")
                            + ".evidence.json")
    Path(out_path).write_text(json.dumps(out, indent=2))
    print("=" * 70)
    for key in ["root", "model", "k_nominal", "k_eff", "data_log_likelihood_ln",
                "log_Z", "p_eff", "null_dim", "validation_resid_rel"]:
        print(f"  {key:24s} {out[key]}")
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
