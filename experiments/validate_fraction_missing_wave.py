"""Validate the fraction-missing (per-leaf p_obs) feature on gpurec's PRODUCTION
GPU wave path against the CPU legacy reference.

This script MUST run on a CUDA device (A100). It mirrors the setup of
``experiments/test_fraction_missing.py`` for the bundled family
``tests/data/test_trees_1/{sp.nwk,g.nwk}`` and checks:

  (1) LEGACY reference NLL: E_fixed_point(leaf_E) + legacy Pi_fixed_point(leaf_obs_log,
      leaf_species_mask) + compute_log_likelihood, for fraction_missing in {0.0, 0.3}.
  (2) WAVE system-under-test NLL: E_fixed_point(leaf_E) +
      Pi_wave_forward(..., leaf_obs_log=leaf_E) + compute_log_likelihood, same fractions.
  (3) Equivalence asserts:
        - fm=0.0 : wave == wave-baseline(leaf_obs_log=None) == legacy-baseline (exact);
        - fm=0.3 : |wave - legacy| < 1e-9;
        - effect : |wave(0.3) - wave(0.0)| > 1e-6.
  (4) FINITE-DIFF GRADIENT check with fm=0.3, specieswise theta [S,3]:
        - analytic gradient through the wave backward (implicit_grad_loglik_vjp_wave
          with leaf_E) vs central finite differences of the wave-forward NLL,
          for pibar_mode in {'dense', 'uniform'}; assert max rel-error < 1e-4;
        - regression: with fm=0.0 the analytic gradient equals the no-fraction-missing
          gradient bitwise.

Run from the repo root on a GPU box:

    PYTHONPATH=. python experiments/validate_fraction_missing_wave.py

Exits nonzero on any failure.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

# Make `import gpurec...` work when run as `python experiments/...` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_INV = 1.0 / math.log(2.0)  # ln -> log2
DT = torch.float64
D, L, T = 0.1, 0.1, 0.1
SP = "tests/data/test_trees_1/sp.nwk"
G = "tests/data/test_trees_1/g.nwk"


def _load_ext():
    """Build the C++ preprocess extension (Linux/CUDA box has OpenMP)."""
    from torch.utils.cpp_extension import load
    cpp = _REPO_ROOT / "gpurec" / "core" / "cpp"
    srcs = [str(cpp / f) for f in ("preprocess.cpp", "tree_utils.cpp", "clade_utils.cpp")]
    cflags = ["-O3"]
    # macOS libomp include (harmless on Linux if missing)
    inc = Path("/opt/homebrew/opt/libomp/include")
    extra_cflags = list(cflags)
    if inc.exists():
        extra_cflags.append(f"-I{inc}")
    else:
        extra_cflags.append("-fopenmp")
    bdir = Path.home() / ".cache" / "gpurec_wave_val" / "preprocess_cpp"
    bdir.mkdir(parents=True, exist_ok=True)
    ldflags = [] if inc.exists() else ["-fopenmp"]
    return load(name="preprocess_cpp", sources=srcs, extra_cflags=extra_cflags,
                extra_ldflags=ldflags, build_directory=str(bdir), verbose=False)


def build_problem(device):
    """Load the bundled family; return helpers dicts and leaf index tensors."""
    ext = _load_ext()
    raw = ext.preprocess(SP, [G])
    sr, cr = raw["species"], raw["ccp"]
    S = int(sr["S"])
    C = int(cr["C"])
    sh = {
        "S": S, "names": sr["names"],
        "s_P_indexes": sr["s_P_indexes"].to(device),
        "s_C12_indexes": sr["s_C12_indexes"].to(device),
        "Recipients_mat": sr["Recipients_mat"].to(dtype=DT, device=device),
    }
    # C++ preprocess provides the exact ancestors_dense (includes-self convention)
    # used by uniform mode; prefer it over any reconstruction.
    if "ancestors_dense" in sr:
        sh["ancestors_dense"] = sr["ancestors_dense"].to(dtype=DT, device=device)
    ch = {
        "split_leftrights_sorted": cr["split_leftrights_sorted"].to(device),
        "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=DT, device=device) * _INV,
        "seg_parent_ids": cr["seg_parent_ids"].to(device),
        "ptr_ge2": cr["ptr_ge2"].to(device),
        "num_segs_ge2": int(cr["num_segs_ge2"]),
        "num_segs_eq1": int(cr["num_segs_eq1"]),
        "end_rows_ge2": int(cr["end_rows_ge2"]),
        "C": C, "N_splits": int(cr["N_splits"]),
        "split_parents_sorted": cr["split_parents_sorted"].to(device),
    }
    li = raw["leaf_row_index"].long().to(device)
    lc = raw["leaf_col_index"].long().to(device)
    root_id = torch.tensor([int(cr["root_clade_id"])], device=device)

    # species-tree leaf mask: nodes that are never internal parents
    sP = sh["s_P_indexes"]
    internal = sP[sP < S].unique()
    leaf_mask = torch.ones(S, dtype=torch.bool, device=device)
    leaf_mask[internal] = False

    return sh, ch, li, lc, root_id, leaf_mask, S, C


def leaf_E_of(frac, leaf_mask, S, device):
    """[S] log2(fraction_missing) at species-leaves with frac>0, -inf elsewhere."""
    fm = torch.zeros(S, dtype=DT, device=device)
    fm[leaf_mask] = frac
    return torch.where(fm > 0, torch.log2(fm.clamp_min(1e-300)),
                       torch.full_like(fm, float("-inf")))


def require_ancestors_dense(sh):
    """uniform mode needs species_helpers['ancestors_dense'] (provided by the
    C++ preprocess; includes-self convention). Fail loudly if it is missing."""
    if 'ancestors_dense' not in sh:
        raise RuntimeError(
            "species_helpers lacks 'ancestors_dense' (needed for pibar_mode='uniform'); "
            "the C++ preprocess extension should provide it."
        )


# ---------------------------------------------------------------------------
# Legacy reference
# ---------------------------------------------------------------------------

def legacy_nll(theta, sh, ch, li, lc, root_id, leaf_mask, leaf_E, leaf_obs_log,
               device, pibar_mode='dense'):
    from gpurec.core.extract_parameters import extract_parameters
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.legacy import Pi_fixed_point

    tm = torch.log2(sh["Recipients_mat"])
    pS, pD, pL, tf, mt = extract_parameters(theta, tm, genewise=False,
                                            specieswise=False, pairwise=False)
    mv = mt.squeeze(-1) if mt.ndim == 2 else mt
    Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                       transfer_mat=tf, max_transfer_mat=mv, max_iters=4000,
                       tolerance=1e-13, warm_start_E=None, dtype=DT, device=device,
                       pibar_mode=pibar_mode, leaf_E=leaf_E)
    fp = Pi_fixed_point(ccp_helpers=ch, species_helpers=sh,
                        leaf_row_index=li, leaf_col_index=lc,
                        E=Eo["E"], Ebar=Eo["E_bar"], E_s1=Eo["E_s1"], E_s2=Eo["E_s2"],
                        log_pS=pS, log_pD=pD, log_pL=pL,
                        transfer_mat_T=tf.T.contiguous(), max_transfer_mat=mv,
                        max_iters=4000, tolerance=1e-13, warm_start_Pi=None,
                        device=device, dtype=DT,
                        leaf_obs_log=leaf_obs_log, leaf_species_mask=leaf_mask)
    return float(compute_log_likelihood(fp["Pi"], Eo["E"], root_id))


# ---------------------------------------------------------------------------
# Wave system-under-test
# ---------------------------------------------------------------------------

def build_wave_layout_for(ch, li, lc, root_id, device):
    from gpurec.core.batching import (
        collate_gene_families, collate_wave, build_wave_layout, split_phase_waves,
    )
    from gpurec.core.scheduling import compute_clade_waves

    item = {
        'ccp': ch,
        'leaf_row_index': li,
        'leaf_col_index': lc,
        'root_clade_id': int(root_id.item()),
    }
    batched = collate_gene_families([item], dtype=DT, device=device)
    waves, phases = compute_clade_waves(ch)
    offsets = [m['clade_offset'] for m in batched['family_meta']]
    cross_waves = collate_wave([waves], offsets)
    cross_phases = list(phases)
    cross_waves, cross_phases = split_phase_waves(cross_waves, cross_phases,
                                                  phase=None, max_wave_size=32768)
    wl = build_wave_layout(
        waves=cross_waves, phases=cross_phases,
        ccp_helpers=batched['ccp'],
        leaf_row_index=batched['leaf_row_index'],
        leaf_col_index=batched['leaf_col_index'],
        root_clade_ids=batched['root_clade_ids'],
        device=device, dtype=DT,
        family_clade_counts=[m['C'] for m in batched['family_meta']],
        family_clade_offsets=[m['clade_offset'] for m in batched['family_meta']],
    )
    return wl, batched['root_clade_ids']


def _extract(theta, sh, pibar_mode, specieswise):
    from gpurec.core.extract_parameters import (
        extract_parameters, extract_parameters_uniform,
    )
    tm = torch.log2(sh["Recipients_mat"])
    urm = tm.max(dim=-1).values
    if pibar_mode == 'dense':
        pS, pD, pL, tf, mt_raw = extract_parameters(
            theta, tm, genewise=False, specieswise=specieswise, pairwise=False)
        mt = mt_raw.squeeze(-1) if mt_raw.ndim == 2 else mt_raw
        return pS, pD, pL, tf, mt, tm, urm
    else:
        pS, pD, pL, tf, mt = extract_parameters_uniform(
            theta, urm, specieswise=specieswise)
        return pS, pD, pL, tf, mt, tm, urm


def wave_forward_nll(theta, sh, wl, roots_orig, leaf_E, leaf_obs_log,
                     device, pibar_mode='dense', specieswise=False,
                     return_E=False, ancestors_T=None):
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.forward import Pi_wave_forward

    pS, pD, pL, tf, mt, tm, urm = _extract(theta, sh, pibar_mode, specieswise)
    Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                       transfer_mat=tf, max_transfer_mat=mt, max_iters=4000,
                       tolerance=1e-13, warm_start_E=None, dtype=DT, device=device,
                       pibar_mode=pibar_mode, leaf_E=leaf_E, ancestors_T=ancestors_T)
    Pi_out = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=Eo['E'], Ebar=Eo['E_bar'], E_s1=Eo['E_s1'], E_s2=Eo['E_s2'],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mt,
        device=device, dtype=DT, pibar_mode=pibar_mode,
        leaf_obs_log=leaf_obs_log,
        local_iters=4000, local_tolerance=1e-13,
    )
    nll = float(compute_log_likelihood(Pi_out['Pi'], Eo['E'], roots_orig).sum())
    if return_E:
        return nll, Eo, Pi_out, (pS, pD, pL, tf, mt, tm, urm)
    return nll


def wave_analytic_grad(theta, sh, wl, leaf_E, device, pibar_mode='dense',
                       ancestors_T=None):
    """Analytic dNLL/dtheta through the wave backward (specieswise theta [S,3])."""
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.forward import Pi_wave_forward
    from gpurec.optimization.implicit_grad import implicit_grad_loglik_vjp_wave

    pS, pD, pL, tf, mt, tm, urm = _extract(theta, sh, pibar_mode, specieswise=True)
    Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                       transfer_mat=tf, max_transfer_mat=mt, max_iters=4000,
                       tolerance=1e-13, warm_start_E=None, dtype=DT, device=device,
                       pibar_mode=pibar_mode, leaf_E=leaf_E, ancestors_T=ancestors_T)
    Pi_out = Pi_wave_forward(
        wave_layout=wl, species_helpers=sh,
        E=Eo['E'], Ebar=Eo['E_bar'], E_s1=Eo['E_s1'], E_s2=Eo['E_s2'],
        log_pS=pS, log_pD=pD, log_pL=pL,
        transfer_mat=tf, max_transfer_mat=mt,
        device=device, dtype=DT, pibar_mode=pibar_mode,
        leaf_obs_log=leaf_E,
        local_iters=4000, local_tolerance=1e-13,
    )
    grad_theta, _ = implicit_grad_loglik_vjp_wave(
        wl, sh,
        Pi_star_wave=Pi_out['Pi_wave_ordered'],
        Pibar_star_wave=Pi_out['Pibar_wave_ordered'],
        E_star=Eo['E'], E_s1=Eo['E_s1'], E_s2=Eo['E_s2'], Ebar=Eo['E_bar'],
        log_pS=pS, log_pD=pD, log_pL=pL,
        max_transfer_mat=mt,
        root_clade_ids_perm=wl['root_clade_ids'],
        theta=theta,
        unnorm_row_max=urm,
        specieswise=True,
        device=device, dtype=DT,
        # Dense uses a TRUNCATED Neumann series for the Pi self-loop adjoint.
        # Fraction-missing shrinks the magnitude of some gradient components
        # (e.g. the L/loss rate), so the fixed absolute Neumann truncation becomes a
        # larger *relative* error there: 4 terms -> ~1.8e-3, 8 -> ~9e-7, 12 -> ~3e-10.
        # Use 12 so the dense path is fully converged (well under the 1e-4 target);
        # uniform solves the self-loop with GMRES and is unaffected.
        neumann_terms=12, use_pruning=False,
        cg_tol=1e-12, cg_maxiter=2000,
        pibar_mode=pibar_mode,
        transfer_mat=tf if pibar_mode == 'dense' else None,
        transfer_mat_unnormalized=tm if pibar_mode == 'dense' else None,
        ancestors_T=ancestors_T,
        leaf_E=leaf_E,
    )
    return grad_theta.detach()


def _eadj_diagnostic(sh, ancestors_T, S, leaf_mask, device):
    """Probe the E-adjoint linear solve (I - G_E^T) w = q for fm=0 vs fm=0.3.

    Reports operator asymmetry and the residual that plain CG leaves, to expose
    why CG (SPD-only) silently mis-solves the non-symmetric fraction-missing E
    adjoint and the GMRES fallback is required.
    """
    import torch.func as tfunc
    from gpurec.core.likelihood import E_fixed_point, E_step
    from gpurec.optimization.linear_solvers import _cg, _gmres

    th = math.log2(0.1) * torch.ones(S, 3, dtype=DT, device=device)
    pS, pD, pL, tf, mt, tm, urm = _extract(th, sh, 'dense', specieswise=True)
    sp_P_idx = sh['s_P_indexes']
    sp_c12 = sh['s_C12_indexes']
    print("[3.5] E-adjoint solver diagnostic (dense):")
    for tag, frac in (("fm=0", 0.0), ("fm=0.3", 0.3)):
        le = leaf_E_of(frac, leaf_mask, S, device)
        Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                           transfer_mat=tf, max_transfer_mat=mt, max_iters=4000,
                           tolerance=1e-13, warm_start_E=None, dtype=DT, device=device,
                           pibar_mode='dense', leaf_E=le)
        E_star = Eo['E']

        def G_E_fun(E_in):
            return E_step(E_in, sp_P_idx, sp_c12, pS, pD, pL, tf, mt,
                          pibar_mode='dense', leaf_E=le)[0]

        E_g = E_star.detach().requires_grad_(True)
        with torch.enable_grad():
            _, vjpG = tfunc.vjp(G_E_fun, E_g)

        def A(w_flat):
            w = w_flat.view(E_star.shape).contiguous()
            gE, = vjpG(w.clone())
            return (w - gE).reshape(-1)

        q = torch.randn(S, dtype=DT, device=device)
        x_cg, st_cg, ok_cg = _cg(A, q, tol=1e-10, maxiter=2000)
        cg_res = float((A(x_cg) - q).norm()) / float(q.norm())
        x_gm, st_gm = _gmres(A, q, tol=1e-10, restart=40, maxiter=2000)
        gm_res = float((A(x_gm) - q).norm()) / float(q.norm())
        # operator asymmetry (small S only; skip the O(S^2) probe for big trees)
        asym = float('nan')
        if S <= 512:
            cols = []
            for i in range(S):
                e = torch.zeros(S, dtype=DT, device=device); e[i] = 1.0
                cols.append(A(e))
            Amat = torch.stack(cols, dim=1)
            asym = float((Amat - Amat.T).abs().max())
        print(f"     {tag:7s}: asym(max|A-A^T|)={asym:.3e}  "
              f"CG ok={ok_cg} res={cg_res:.2e}  GMRES res={gm_res:.2e}")


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA is required for this validation (run on an A100).")
        return 1
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sh, ch, li, lc, root_id, leaf_mask, S, C = build_problem(device)
    require_ancestors_dense(sh)
    ancestors_T = sh['ancestors_dense'].T.to_sparse_coo()
    wl, _roots_perm = build_wave_layout_for(ch, li, lc, root_id, device)
    # Pi_wave_forward returns Pi['Pi'] in ORIGINAL clade order (return_original=True),
    # and legacy Pi_fixed_point likewise. So compute_log_likelihood indexes with the
    # ORIGINAL root clade id.
    roots = root_id

    theta_glob = torch.log2(torch.tensor([D, L, T], dtype=DT, device=device))
    n_leaves = int(leaf_mask.sum())
    print(f"S={S} species ({n_leaves} leaves), C={C} clades, D=L=T={D}, device=cuda")

    ok = True

    # ---- (1)+(2)+(3) NLL equivalence (global theta, dense) ----
    le0 = leaf_E_of(0.0, leaf_mask, S, device)
    le3 = leaf_E_of(0.3, leaf_mask, S, device)

    legacy_base = legacy_nll(theta_glob, sh, ch, li, lc, roots, leaf_mask,
                             leaf_E=None, leaf_obs_log=None, device=device,
                             pibar_mode='dense')
    legacy_fm0 = legacy_nll(theta_glob, sh, ch, li, lc, roots, leaf_mask,
                            leaf_E=le0, leaf_obs_log=le0, device=device,
                            pibar_mode='dense')
    legacy_fm3 = legacy_nll(theta_glob, sh, ch, li, lc, roots, leaf_mask,
                            leaf_E=le3, leaf_obs_log=le3, device=device,
                            pibar_mode='dense')

    wave_base = wave_forward_nll(theta_glob, sh, wl, roots, leaf_E=None,
                                 leaf_obs_log=None, device=device, pibar_mode='dense')
    wave_fm0 = wave_forward_nll(theta_glob, sh, wl, roots, leaf_E=le0,
                                leaf_obs_log=le0, device=device, pibar_mode='dense')
    wave_fm3 = wave_forward_nll(theta_glob, sh, wl, roots, leaf_E=le3,
                                leaf_obs_log=le3, device=device, pibar_mode='dense')

    chk = (wave_fm0 == wave_base) and (wave_base == legacy_base)
    print(f"[1] fm=0.0 exact equality: wave={wave_fm0:.12f} "
          f"wave_base={wave_base:.12f} legacy_base={legacy_base:.12f} -> {chk}")
    ok &= chk

    d_lw = abs(wave_fm3 - legacy_fm3)
    chk = d_lw < 1e-9
    print(f"[2] fm=0.3 wave vs legacy: wave={wave_fm3:.12f} legacy={legacy_fm3:.12f} "
          f"|diff|={d_lw:.2e} (<1e-9) -> {chk}")
    ok &= chk

    d_eff = abs(wave_fm3 - wave_fm0)
    chk = d_eff > 1e-6
    print(f"[3] effect: |wave(0.3)-wave(0.0)|={d_eff:.6e} (>1e-6) -> {chk}")
    ok &= chk

    # also sanity: legacy fm=0 equals legacy baseline (mirrors test_fraction_missing)
    chk = (legacy_fm0 == legacy_base)
    print(f"    legacy fm=0 == legacy baseline: {chk}")
    ok &= chk

    # ---- (3.5) E-adjoint solver diagnostic (root-cause probe) ----------------
    # The implicit-gradient E adjoint solves (I - G_E^T) w = q. CG assumes the
    # operator is SPD, but the fraction-missing leaf boundary makes G_E strongly
    # NON-symmetric, so plain CG can exhaust maxiter while pAp stays positive and
    # silently return a large-residual (wrong) solve unless the success flag tracks
    # ACTUAL convergence and the GMRES fallback kicks in. This block reports the CG
    # residual + operator asymmetry at fm=0 vs fm=0.3 so the cause is visible on GPU.
    _eadj_diagnostic(sh, ancestors_T, S, leaf_mask, device)

    # ---- (4) Finite-difference gradient check (specieswise, fm=0.3) ----
    # Use a less-degenerate init (log2(0.1)) so the gradient is nonzero.
    theta_sw = math.log2(0.1) * torch.ones(S, 3, dtype=DT, device=device)
    le3_sw = leaf_E_of(0.3, leaf_mask, S, device)
    le0_sw = leaf_E_of(0.0, leaf_mask, S, device)
    h = 1e-4

    # Pick a handful of theta entries to perturb (spread across leaves/internals/cols).
    leaf_ids = torch.nonzero(leaf_mask, as_tuple=False).flatten().tolist()
    internal_ids = torch.nonzero(~leaf_mask, as_tuple=False).flatten().tolist()
    probe = []
    for s in (leaf_ids[:2] + internal_ids[:2]):
        for col in range(3):
            probe.append((int(s), col))
    probe = probe[:8]

    for pibar_mode in ('dense', 'uniform'):
        anc_T = ancestors_T if pibar_mode == 'uniform' else None
        g_analytic = wave_analytic_grad(theta_sw, sh, wl, le3_sw, device,
                                        pibar_mode=pibar_mode, ancestors_T=anc_T)

        max_rel = 0.0
        worst = None
        for (s, col) in probe:
            tp = theta_sw.clone(); tp[s, col] += h
            tm_ = theta_sw.clone(); tm_[s, col] -= h
            f_p = wave_forward_nll(tp, sh, wl, roots, leaf_E=le3_sw,
                                   leaf_obs_log=le3_sw, device=device,
                                   pibar_mode=pibar_mode, specieswise=True,
                                   ancestors_T=anc_T)
            f_m = wave_forward_nll(tm_, sh, wl, roots, leaf_E=le3_sw,
                                   leaf_obs_log=le3_sw, device=device,
                                   pibar_mode=pibar_mode, specieswise=True,
                                   ancestors_T=anc_T)
            fd = (f_p - f_m) / (2 * h)
            an = float(g_analytic[s, col])
            denom = max(abs(fd), abs(an), 1e-8)
            rel = abs(fd - an) / denom
            if rel > max_rel:
                max_rel = rel
                worst = (s, col, fd, an)
        chk = max_rel < 1e-4
        ws = (f"  worst entry s={worst[0]} col={worst[1]} fd={worst[2]:.6e} "
              f"analytic={worst[3]:.6e}") if worst else ""
        print(f"[4:{pibar_mode}] FD grad check (fm=0.3): max rel-err={max_rel:.3e} "
              f"(<1e-4) -> {chk}\n{ws}")
        ok &= chk

        # Regression: with fm=0.0 the analytic gradient equals the no-fraction-missing
        # (leaf_E=None) gradient. On the dense route this is bitwise identical (the
        # fraction-missing baseline writes nothing when every column mask is False).
        # On the uniform route, leaf_E=None may take the leaf-index fast path while
        # leaf_E=le0 (all -inf) is gated onto the dense leaf-term route, so the two
        # can differ by floating-point reassociation only; require a very tight match.
        g_fm0 = wave_analytic_grad(theta_sw, sh, wl, le0_sw, device,
                                   pibar_mode=pibar_mode, ancestors_T=anc_T)
        g_none = wave_analytic_grad(theta_sw, sh, wl, None, device,
                                    pibar_mode=pibar_mode, ancestors_T=anc_T)
        bitwise = bool(torch.equal(g_fm0, g_none))
        if pibar_mode == 'dense':
            chk = bitwise
            print(f"[4:{pibar_mode}] regression fm=0.0 grad == no-fm grad (bitwise) -> {chk}")
        else:
            tight = torch.allclose(g_fm0, g_none, rtol=0.0, atol=1e-12)
            chk = bitwise or tight
            print(f"[4:{pibar_mode}] regression fm=0.0 grad == no-fm grad "
                  f"(bitwise={bitwise}, atol1e-12={tight}) -> {chk}")
        ok &= chk

    print("\nRESULT:", "PASS -- fraction-missing wave path validated"
          if ok else "FAIL -- see checks above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
