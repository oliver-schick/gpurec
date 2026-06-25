"""EM for the origination distribution O (DTL held at global rates), as a cold-start
basin-finder. The reconciliation model is a latent-variable model; the EM M-step
re-assigns O in closed form -- O_s ∝ Σ_f P(family f originates at s) -- where the E-step
posterior P(originate at s) = softmax2(log O_s + Π_f[root, s]) is computed from the forward
Π. This is a LARGE move (re-assignment), unlike the gradient nudges that stall at diffuse O.

With DTL fixed, Π[root] is computed ONCE per root and the EM-O loop is pure-numpy/instant,
so we sweep all 15 roots and ask: does EM-O make Euryarchaeota win cold? Monotone in logL.
Run on a GPU node.
"""
from __future__ import annotations
import argparse, glob, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _build_wave_layout, _sp_helpers_for_uniform,
    _parse_fraction_missing, _build_leaf_E,
)
from eval_at_alerax_rates import _load_families_named  # noqa: E402

_LN2 = math.log(2.0)
DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
ALE = sorted(p for p in glob.glob(str(DD / "3_UFBOOTs" / "ufboot_for_alerax" / "*.ale"))
             if not Path(p).name.startswith("._"))
ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury", "HalobacThermopl",
         "Kor", "MHH", "Micra5", "MicraDia", "TACA", "TackA", "TAC", "UndineClu2"]
DEEP = {"Eury", "MHH", "TackA", "DPANN"}
SGA = {"Cluster2", "Micra5", "MicraDia", "AMD", "Asgard", "Alti", "UndineClu2"}
CHUNK = 500


def _load_root(root, fm_mode, dev, dtype):
    """Load families + wave layouts + species helpers ONCE for a root (the expensive ~138s
    part). Reused across all regime forwards so a 50-regime scan is one load + 50 cheap fwds."""
    from gpurec.core.tree_prior import species_parent_index
    tree = DD / "4_species_tree" / f"Undine_C60_{root}root_short_name.nw"
    sp = _load_species_helpers(str(tree)); S = int(sp["S"]); s2i = sp["species_name_to_index"]
    par = species_parent_index(sp).tolist(); root_branch = [i for i in range(S) if par[i] < 0][0]
    fams, _ = _load_families_named(ALE, s2i, min_species=1, dtype=dtype)
    sp_gpu, anc = _sp_helpers_for_uniform(sp, dev, dtype)
    urm = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=dev, dtype=dtype)
    leaf_E = None; leaf_obs = None
    if fm_mode != "off" and (DD / "fraction_missing").exists():
        fm, _n, _s = _parse_fraction_missing(DD / "fraction_missing", s2i, S)
        leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=dev, dtype=dtype)
        leaf_obs = leaf_E if fm_mode == "both" else None
    layouts = [_build_wave_layout(fams[i:i + CHUNK], dev, dtype) for i in range(0, len(fams), CHUNK)]
    names = list(sp["names"]); label2node = {}
    try:
        from clade_groups import parse_labeled_newick, species_node_leafsets
        leafsets = species_node_leafsets(sp, S); labeled = parse_labeled_newick(str(tree))
        for s in range(S):
            lab = labeled.get(leafsets[s])
            if lab:
                label2node[lab] = s
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] clade labels unavailable: {e}", flush=True)
    return dict(sp_gpu=sp_gpu, anc=anc, urm=urm, leaf_E=leaf_E, leaf_obs=leaf_obs, layouts=layouts,
                S=S, rb=root_branch, names=names, lab2n=label2node, nfam=len(fams))


def _forward_regime(Lo, D, L, T, dev, dtype):
    """Forward at GLOBAL (D,L,T) using a pre-loaded root -> root_pi[F,S] + E[S] (both numpy)."""
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.forward import Pi_wave_forward
    S = Lo["S"]
    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=dev)).reshape(1, 3).repeat(S, 1)
    lp = extract_parameters_uniform(theta, Lo["urm"], specieswise=True)
    E = E_fixed_point(species_helpers=Lo["sp_gpu"], log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                      transfer_mat=lp[3], max_transfer_mat=lp[4], max_iters=4000, tolerance=1e-10,
                      warm_start_E=None, dtype=dtype, device=dev, pibar_mode="uniform",
                      ancestors_T=Lo["anc"], leaf_E=Lo["leaf_E"])
    rows = []
    for (wl, rc) in Lo["layouts"]:
        Pi = Pi_wave_forward(wave_layout=wl, species_helpers=Lo["sp_gpu"], E=E["E"], Ebar=E["E_bar"],
                             E_s1=E["E_s1"], E_s2=E["E_s2"], log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                             transfer_mat=lp[3], max_transfer_mat=lp[4], device=dev, dtype=dtype,
                             pibar_mode="uniform", leaf_obs_log=Lo["leaf_obs"])["Pi"]
        rows.append(Pi[rc].cpu().numpy())
    return np.concatenate(rows, 0), E["E"].detach().cpu().numpy()


def root_pi_and_E(root, D, L, T, fm_mode, dev, dtype):
    """Forward at GLOBAL (D,L,T) -> per-family root-clade Pi [F,S] + mean E + helpers (thin wrapper)."""
    Lo = _load_root(root, fm_mode, dev, dtype)
    root_pi, E_vec = _forward_regime(Lo, D, L, T, dev, dtype)
    return (root_pi, Lo["S"], Lo["rb"], float(np.exp2(E_vec).mean()), Lo["names"], Lo["lab2n"], E_vec)


def _shared_o_em_torch(RP, omE, iters=300):
    """Shared-O rate random-effects EM on stacked [K,F,S] / [K,S] torch tensors.
    Returns (marginal_logL, gamma[K,F], log_pO[S])."""
    import torch, math as _m
    LN2 = _m.log(2.0); K, F, S = RP.shape
    lse2 = lambda x, d, kd=False: torch.logsumexp(x * LN2, dim=d, keepdim=kd) / LN2  # noqa: E731
    log_w = torch.full((K,), -_m.log2(K), device=RP.device)
    log_pO = torch.full((S,), -_m.log2(S), device=RP.device); prev = -1e30; gamma = None
    for _ in range(iters):
        m = log_pO[None, None, :] + RP; num = lse2(m, 2)
        surv = torch.log2((torch.exp2(log_pO)[None, :] * omE).sum(1).clamp_min(1e-30))
        joint = log_w[:, None] + (num - surv[:, None]); Pf = lse2(joint, 0); ll = float(Pf.sum().item())
        gamma = torch.exp2(joint - Pf[None, :]); log_w = torch.log2((gamma.mean(1)).clamp_min(1e-30))
        pbs = torch.exp2(m - lse2(m, 2, True)); Of = (gamma[:, :, None] * pbs).sum(0).sum(0)
        log_pO = torch.log2((Of / Of.sum()).clamp_min(1e-30))
        if abs(ll - prev) < 1e-2:
            break
        prev = ll
    # per-family marginal logL at the CONVERGED (w, p^O) -- for the AU test / model selection
    m = log_pO[None, None, :] + RP; num = lse2(m, 2)
    surv = torch.log2((torch.exp2(log_pO)[None, :] * omE).sum(1).clamp_min(1e-30))
    Pf = lse2(log_w[:, None] + (num - surv[:, None]), 0)
    return ll, gamma, log_pO, Pf


def _lse2(x, axis=-1, keepdims=False):
    mx = x.max(axis, keepdims=True)
    out = np.log2(np.exp2(x - mx).sum(axis, keepdims=True)) + mx
    return out if keepdims else out.squeeze(axis)


def em_mixture(root_pi, S, K, iters, root_branch, seed=0, tol=1e-2):
    """K-component MIXTURE over families for origination. Each family is a soft mixture
    of K origination 'types', each type k carrying its own p^O_k [S]. Breaks the pooling
    that collapses a single shared p^O onto the deep/root mode. Returns (log_pi[K],
    log_pO[K,S], data_ll, n_iters, gamma[F,K]).
      E: lc[f,k]=logsumexp2_s(log_pO[k,s]+Pi_f[s]); gamma=softmax_k(log_pi[k]+lc[f,k])
      M: pi_k=mean_f gamma; p^O_k ∝ Σ_f gamma[f,k]*softmax2_s(log_pO[k]+Pi_f)
    """
    F = root_pi.shape[0]
    rng = np.random.default_rng(seed)
    log_pO = np.full((K, S), -math.log2(S))
    log_pO[0, root_branch] += 12.0                    # component 0 seeded DEEP (root-concentrated)
    for k in range(K):
        if k > 0:
            log_pO[k] = -math.log2(S) + 0.5 * rng.standard_normal(S)  # spread, jittered
        log_pO[k] -= _lse2(log_pO[k])
    log_pi = np.full(K, -math.log2(K))
    ll_prev = -np.inf; nit = iters
    for it in range(iters):
        lc = np.empty((F, K))
        for k in range(K):
            lc[:, k] = _lse2(log_pO[k][None, :] + root_pi, axis=1)        # [F]
        joint = log_pi[None, :] + lc                                      # [F,K]
        ll = float(_lse2(joint, axis=1).sum())
        gamma = np.exp2(joint - _lse2(joint, axis=1, keepdims=True))      # [F,K]
        # M-step
        Nk = gamma.sum(0)                                                 # [K] soft counts
        log_pi = np.log2(np.maximum(Nk / F, 1e-300))
        for k in range(K):
            m = log_pO[k][None, :] + root_pi                             # [F,S]
            pbs = np.exp2(m - _lse2(m, axis=1, keepdims=True))          # p(s|f,k) [F,S]
            Ok = (gamma[:, k:k + 1] * pbs).sum(0)                        # expected orig at s in comp k
            log_pO[k] = np.log2(np.maximum(Ok / Ok.sum(), 1e-300))
        if abs(ll - ll_prev) < tol:
            nit = it + 1; break
        ll_prev = ll
    return log_pi, log_pO, ll, nit, gamma


def em_o(root_pi, S, iters):
    F = root_pi.shape[0]
    log_pO = np.full(S, -math.log2(S))
    ll_prev = -np.inf
    for it in range(iters):
        m = log_pO[None, :] + root_pi                          # [F,S]
        mx = m.max(1, keepdims=True)
        P = np.exp2(m - mx); Z = P.sum(1, keepdims=True)
        ll = float((np.log2(Z[:, 0]) + mx[:, 0]).sum())        # Σ_f logsumexp2(log_pO + Pi[root])
        P /= Z
        Of = P.sum(0)                                          # expected #originations per branch
        log_pO = np.log2(Of / Of.sum())
        if abs(ll - ll_prev) < 1e-3:
            break
        ll_prev = ll
    return log_pO, ll, it + 1


# ---- MIXTURE-OVER-REGIMES SCAN ------------------------------------------------
# A discrete mixture whose components are FIXED global-DTL "regimes" on a (D,L,T)
# grid (= the panel's "global per-class DTL", coarsely). Each regime's forward
# (root_pi[F,S] + extinction E[S]) is one GPU job (fan out across short-a100);
# then a pure-numpy EM lets families soft-assign to regimes WITH per-regime
# origination + per-regime survival. Tests whether the data wants >1 regime
# (core/accessory) and whether that de-concentrates origination off the root.
MIX_GRID = [(d, l, t)
            for d in (0.03, 0.08)
            for l in (0.1, 0.3, 0.6, 1.2, 2.4)
            for t in (0.05, 0.15, 0.4, 1.0, 2.5)]            # 2x5x5 = 50 regimes


def save_forward(root, idx, fm_mode, out_dir, dev, dtype):
    D, L, T = MIX_GRID[idx]
    root_pi, S, rb, mean_E, names, lab2n, E_vec = root_pi_and_E(root, D, L, T, fm_mode, dev, dtype)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = f"{out_dir}/regime_{idx:03d}.npz"
    np.savez_compressed(out, root_pi=root_pi.astype(np.float32), E=E_vec.astype(np.float32),
                        rb=rb, dlt=np.array([D, L, T]), names=np.array(names, dtype=object),
                        labels=np.array(list(lab2n.keys()), dtype=object),
                        label_nodes=np.array(list(lab2n.values()), dtype=np.int64))
    print(f"[save_forward] regime {idx:03d} D,L,T={D},{L},{T} mean_E={mean_E:.3f} -> {out}", flush=True)


def _single_regime_logL(root_pi, oneMinusE, iters=200):
    """Best per-regime data logL = single-component EM on this regime (with survival)."""
    F, S = root_pi.shape
    log_pO = np.full(S, -math.log2(S)); prev = -np.inf
    for _ in range(iters):
        m = log_pO[None, :] + root_pi
        num = _lse2(m, axis=1)
        surv = math.log2(max((np.exp2(log_pO) * oneMinusE).sum(), 1e-300))
        ll = float((num - surv).sum())
        pbs = np.exp2(m - _lse2(m, axis=1, keepdims=True))
        Of = pbs.sum(0); log_pO = np.log2(np.maximum(Of / Of.sum(), 1e-300))
        if abs(ll - prev) < 1e-2:
            break
        prev = ll
    return ll, log_pO


def regime_mixture_em(npz_dir, iters=400):
    files = sorted(glob.glob(f"{npz_dir}/regime_*.npz"))
    if not files:
        raise SystemExit(f"no regime_*.npz in {npz_dir}")
    rps, Es, dlts = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rps.append(d["root_pi"].astype(np.float32)); Es.append(d["E"].astype(np.float64))
        dlts.append(tuple(float(x) for x in d["dlt"]))
    rb = int(d["rb"]); names = list(d["names"])
    lab2n = {str(k): int(v) for k, v in zip(d["labels"], d["label_nodes"])}
    K = len(files); F, S = rps[0].shape

    # ---- GPU path (torch): the [K,F,S] logsumexp/softmax EM in seconds on the A100 ----
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        use_gpu = False
    if use_gpu:
        dev = torch.device("cuda"); LN2 = math.log(2.0)
        RP = torch.stack([torch.from_numpy(r) for r in rps]).to(dev, torch.float32)   # [K,F,S] log2
        omE = torch.stack([torch.from_numpy((np.maximum(1.0 - np.exp2(E), 1e-300)).astype(np.float32))
                           for E in Es]).to(dev)                                       # [K,S]
        lse2 = lambda x, dim, kd=False: torch.logsumexp(x * LN2, dim=dim, keepdim=kd) / LN2  # noqa: E731
        # single-regime baselines (all K in parallel, own origination each)
        lpO = torch.full((K, S), -math.log2(S), device=dev)
        for _ in range(200):
            m = lpO[:, None, :] + RP                                                   # [K,F,S]
            pbs = torch.exp2(m - lse2(m, 2, True))
            Of = pbs.sum(1); lpO = torch.log2((Of / Of.sum(1, keepdim=True)).clamp_min(1e-30))
        m = lpO[:, None, :] + RP
        surv = torch.log2((torch.exp2(lpO) * omE).sum(1).clamp_min(1e-30))             # [K]
        base_ll = (lse2(m, 2) - surv[:, None]).sum(1)                                  # [K]
        best_single = int(base_ll.argmax().item())
        base = [(float(base_ll[k]), None) for k in range(K)]
        # mixture EM
        log_pi = torch.full((K,), -math.log2(K), device=dev)
        log_pO = torch.full((K, S), -math.log2(S), device=dev); prev = -1e30; nit = iters
        for it in range(iters):
            m = log_pO[:, None, :] + RP                                                # [K,F,S]
            num = lse2(m, 2)                                                           # [K,F]
            surv = torch.log2((torch.exp2(log_pO) * omE).sum(1).clamp_min(1e-30))      # [K]
            Lk = (num - surv[:, None]).T                                              # [F,K]
            joint = log_pi[None, :] + Lk
            ll = float(lse2(joint, 1).sum().item())
            gamma = torch.exp2(joint - lse2(joint, 1, True))                          # [F,K]
            Nk = gamma.sum(0); log_pi = torch.log2((Nk / F).clamp_min(1e-30))
            pbs = torch.exp2(m - lse2(m, 2, True))                                    # [K,F,S]
            Ok = (gamma.T[:, :, None] * pbs).sum(1)                                   # [K,S]
            log_pO = torch.log2((Ok / Ok.sum(1, keepdim=True)).clamp_min(1e-30))
            if abs(ll - prev) < 1e-2:
                nit = it + 1; break
            prev = ll
        return dict(log_pi=log_pi.cpu().numpy(), log_pO=log_pO.cpu().numpy(),
                    gamma=gamma.cpu().numpy(), ll=ll, nit=nit, dlts=dlts, rb=rb, names=names,
                    lab2n=lab2n, base=base, best_single=best_single, K=K, F=F)

    # ---- CPU fallback (numpy) ----
    oneMinusE = [np.maximum(1.0 - np.exp2(E), 1e-300) for E in Es]
    base = [_single_regime_logL(rps[k], oneMinusE[k]) for k in range(K)]
    best_single = max(range(K), key=lambda k: base[k][0])
    log_pi = np.full(K, -math.log2(K))
    log_pO = np.full((K, S), -math.log2(S)); prev = -np.inf; nit = iters
    for it in range(iters):
        Lk = np.empty((F, K), dtype=np.float64)
        for k in range(K):
            m = log_pO[k][None, :] + rps[k]
            num = _lse2(m.astype(np.float64), axis=1)
            surv = math.log2(max((np.exp2(log_pO[k]) * oneMinusE[k]).sum(), 1e-300))
            Lk[:, k] = num - surv
        joint = log_pi[None, :] + Lk
        ll = float(_lse2(joint, axis=1).sum())
        gamma = np.exp2(joint - _lse2(joint, axis=1, keepdims=True))
        Nk = gamma.sum(0); log_pi = np.log2(np.maximum(Nk / F, 1e-300))
        for k in range(K):
            m = (log_pO[k][None, :] + rps[k]).astype(np.float64)
            pbs = np.exp2(m - _lse2(m, axis=1, keepdims=True))
            Ok = (gamma[:, k:k + 1] * pbs).sum(0)
            log_pO[k] = np.log2(np.maximum(Ok / Ok.sum(), 1e-300))
        if abs(ll - prev) < 1e-2:
            nit = it + 1; break
        prev = ll
    return dict(log_pi=log_pi, log_pO=log_pO, gamma=gamma, ll=ll, nit=nit, dlts=dlts,
                rb=rb, names=names, lab2n=lab2n, base=base, best_single=best_single, K=K, F=F)


def report_regime_mixture(R):
    pi = np.exp2(R["log_pi"]); pO = np.exp2(R["log_pO"]); K = R["K"]; rb = R["rb"]
    dlts = R["dlts"]; lab2n = R["lab2n"]; names = R["names"]
    marg = (pi[:, None] * pO).sum(0)
    hard = R["gamma"].argmax(1); cnt = np.bincount(hard, minlength=K)
    best_single_ll = R["base"][R["best_single"]][0]
    node2lab = {v: k for k, v in lab2n.items()}
    print(f"\n================ REGIME-MIXTURE ({K} regimes) ================")
    print(f"  mixture logL={R['ll']:.1f}   best-single logL={best_single_ll:.1f}   "
          f"gain={R['ll']-best_single_ll:.1f}  (best single regime DLT={dlts[R['best_single']]})  iters={R['nit']}")
    order = np.argsort(-pi)
    print(f"  ACTIVE regimes (pi>0.01):  [D,L,T]  pi  fams(hard)  O@root  top-orig-clades")
    for k in order:
        if pi[k] < 0.01:
            continue
        top = np.argsort(-pO[k])[:5]
        desc = ", ".join(f"{node2lab.get(int(s), names[int(s)][:10])}={pO[k][int(s)]:.3f}" for s in top)
        print(f"    D,L,T={str(tuple(round(x,3) for x in dlts[k])):22s} pi={pi[k]:.3f} "
              f"n={int(cnt[k]):5d} O@root={pO[k][rb]:.3f}  [{desc}]")
    print(f"  -- MARGINAL p^O (Σ_k pi_k p^O_k); single-regime collapses O@root high --")
    print(f"     root  marg={marg[rb]:.4f}")
    for lab in ("DPANN", "Undinarchaeota", "Nanoarchaeota", "Asgard", "TackA", "Eury", "Cluster1"):
        s = lab2n.get(lab)
        if s is not None:
            print(f"     {lab:16s} marg={marg[s]:.4f}")


def shared_o_rate_mixture_em(npz_dir, iters=300, node511=511):
    """STEP 1: per-family RATE random-effect + ONE SHARED origination (the model the
    panel/PI converged on). Each family marginalizes over the regime grid (= discretized
    continuous (D,L,T) random effect) but ALL families share a single p^O. Tests whether,
    once rate is a per-family random effect, the shared origination still piles onto the
    non-Eury ancestor 511 (the bloat driver) -- WITHOUT chasing DPANN's own O (which is
    correctly ~0). P(f)=Σ_s p^O_s Σ_k w_k 2^{rp_k[f,s]}/surv_k.  Reports shared p^O at
    root / 511 / DPANN, the rate-mixture w_k, and the gain vs the best single regime."""
    import glob as _g, torch, math as _m
    files = sorted(_g.glob(f"{npz_dir}/regime_*.npz"))
    rps, Es, dlts = [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rps.append(d["root_pi"].astype(np.float32)); Es.append(d["E"].astype(np.float64)); dlts.append(tuple(float(x) for x in d["dlt"]))
    rb = int(d["rb"]); names = list(d["names"]); lab2n = {str(k): int(v) for k, v in zip(d["labels"], d["label_nodes"])}
    K = len(files); F, S = rps[0].shape
    dev = torch.device("cuda"); LN2 = _m.log(2.0)
    RP = torch.stack([torch.from_numpy(r) for r in rps]).to(dev, torch.float32)              # [K,F,S]
    omE = torch.stack([torch.from_numpy(np.maximum(1.0 - np.exp2(E), 1e-300).astype(np.float32)) for E in Es]).to(dev)  # [K,S]
    lse2 = lambda x, dim, kd=False: torch.logsumexp(x * LN2, dim=dim, keepdim=kd) / LN2       # noqa: E731
    # best single-regime baseline (shared-O style, each regime its own pO)
    lpO_s = torch.full((K, S), -_m.log2(S), device=dev)
    for _ in range(200):
        ms = lpO_s[:, None, :] + RP; pbs = torch.exp2(ms - lse2(ms, 2, True)); Of = pbs.sum(1); lpO_s = torch.log2((Of / Of.sum(1, keepdim=True)).clamp_min(1e-30))
    ms = lpO_s[:, None, :] + RP; survs = torch.log2((torch.exp2(lpO_s) * omE).sum(1).clamp_min(1e-30)); base_ll = float((lse2(ms, 2) - survs[:, None]).sum(1).max())
    # shared-O rate mixture
    log_w = torch.full((K,), -_m.log2(K), device=dev); log_pO = torch.full((S,), -_m.log2(S), device=dev); prev = -1e30; nit = iters
    for it in range(iters):
        m = log_pO[None, None, :] + RP                                                        # [K,F,S]
        num = lse2(m, 2)                                                                       # [K,F]
        surv = torch.log2((torch.exp2(log_pO)[None, :] * omE).sum(1).clamp_min(1e-30))         # [K]
        joint = log_w[:, None] + (num - surv[:, None])                                         # [K,F]
        Pf = lse2(joint, 0); ll = float(Pf.sum().item())                                       # [F]
        gamma = torch.exp2(joint - Pf[None, :])                                                # [K,F] rate resp.
        log_w = torch.log2((gamma.mean(1)).clamp_min(1e-30))
        pbs = torch.exp2(m - lse2(m, 2, True))                                                 # [K,F,S]
        rho = (gamma[:, :, None] * pbs).sum(0)                                                 # [F,S] origination posterior
        Of = rho.sum(0); log_pO = torch.log2((Of / Of.sum()).clamp_min(1e-30))                 # SHARED p^O
        if abs(ll - prev) < 1e-2:
            nit = it + 1; break
        prev = ll
    pO = torch.exp2(log_pO).cpu().numpy(); w = torch.exp2(log_w).cpu().numpy()
    node2lab = {v: k for k, v in lab2n.items()}
    print(f"\n========= SHARED-O RATE RANDOM-EFFECTS (step 1) =========")
    print(f"  shared-O mixture logL={ll:.1f}   best-single={base_ll:.1f}   gain={ll-base_ll:.1f}  iters={nit}")
    print(f"  rate-mixture weights w_k (top 8 regimes): ")
    for k in np.argsort(-w)[:8]:
        print(f"    D,L,T={str(tuple(round(x,3) for x in dlts[k])):22s} w={w[k]:.3f}")
    print(f"  -- SHARED p^O (the origination profile high in the tree) --")
    print(f"     root(LACA)  p^O={pO[rb]:.4f}")
    print(f"     node511     p^O={pO[node511]:.4f}   <- the non-Eury ancestor (bloat driver)")
    for lab in ("DPANN", "Undinarchaeota", "Asgard", "TackA", "Eury", "Korarchaeota"):
        s = lab2n.get(lab)
        if s is not None:
            print(f"     {lab:16s} p^O={pO[s]:.4f}")
    return dict(log_pO=log_pO.cpu().numpy(), log_w=log_w.cpu().numpy(), rb=rb)


def rooting_rfx(root, fm_mode, dev, dtype, out_dir=None, iters=300):
    """GLOBAL-OPTIMUM-OVER-ROOTS under the per-family RANDOM-EFFECTS model: load the root
    once, forward all MIX_GRID regimes (per-family rate = the regime mixture), run the
    shared-O EM -> the marginal data logL = the root's score. Saves gamma + shared p^O for
    the genome read-out. Free of the per-branch DTL freedom that gave the SGA/Cluster2 root
    artifact -> tests whether the random-effects model recovers the Eury root region."""
    import torch
    Lo = _load_root(root, fm_mode, dev, dtype)
    RP_list, E_list = [], []
    for (D, L, T) in MIX_GRID:
        rp, Ev = _forward_regime(Lo, D, L, T, dev, dtype)
        RP_list.append(rp.astype(np.float32)); E_list.append(Ev)
    RP = torch.stack([torch.from_numpy(r) for r in RP_list]).to(dev, torch.float32)
    omE = torch.stack([torch.from_numpy(np.maximum(1.0 - np.exp2(E), 1e-300).astype(np.float32))
                       for E in E_list]).to(dev)
    ll, gamma, log_pO, Pf = _shared_o_em_torch(RP, omE, iters=iters)
    rb = Lo["rb"]; pO = torch.exp2(log_pO).cpu().numpy()
    print(f"ROOTSCORE\t{root}\tlogL={ll:.2f}\tnfam={Lo['nfam']}\tpO_root={pO[rb]:.4f}", flush=True)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"{out_dir}/rfx_{root}.npz", logL=ll, nfam=Lo["nfam"], rb=rb,
                            log_pO=log_pO.cpu().numpy(), gamma=gamma.cpu().numpy().astype(np.float16),
                            perfam_logL=Pf.cpu().numpy().astype(np.float32),
                            dlts=np.array(MIX_GRID), names=np.array(Lo["names"], dtype=object),
                            labels=np.array(list(Lo["lab2n"].keys()), dtype=object),
                            label_nodes=np.array(list(Lo["lab2n"].values()), dtype=np.int64))
    return ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlt", default="0.048,0.287,0.248", help="global D,L,T (AleRax-global default)")
    ap.add_argument("--em-iters", type=int, default=200)
    ap.add_argument("--fm-mode", default="e-only")
    ap.add_argument("--roots", default=None)
    ap.add_argument("--mixture-k", type=int, default=1,
                    help="K>1: fit a K-component MIXTURE over families for origination "
                         "(breaks the single-shared-p^O pooling that collapses to the root).")
    ap.add_argument("--mix-seed", type=int, default=0)
    ap.add_argument("--save-forward-idx", type=int, default=-1,
                    help="regime-mixture scan: compute+save forward for MIX_GRID[idx] (one GPU job).")
    ap.add_argument("--out-dir", default=None, help="dir for regime npz / mixture outputs.")
    ap.add_argument("--mixture-npz", default=None,
                    help="run the regime-mixture EM over regime_*.npz in this dir (CPU/numpy).")
    ap.add_argument("--shared-o", action="store_true",
                    help="with --mixture-npz: per-family rate random-effect + ONE SHARED p^O (step 1).")
    ap.add_argument("--rooting-rfx", action="store_true",
                    help="score the --roots root(s) under the per-family random-effects model "
                         "(marginal logL over the MIX_GRID rate mixture); --out-dir saves rfx_<root>.npz.")
    args = ap.parse_args()

    # regime-mixture / shared-O rate random-effects EM over precomputed forwards.
    if args.mixture_npz:
        if args.shared_o:
            shared_o_rate_mixture_em(args.mixture_npz, iters=args.em_iters * 2)
        else:
            R = regime_mixture_em(args.mixture_npz, iters=args.em_iters * 2)
            report_regime_mixture(R)
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    dev = torch.device("cuda"); dtype = torch.float64
    D, L, T = (float(x) for x in args.dlt.split(","))
    roots = args.roots.split(",") if args.roots else ROOTS

    # GPU path: save one regime's forward for the mixture scan.
    if args.save_forward_idx >= 0:
        r = roots[0]
        save_forward(r, args.save_forward_idx, args.fm_mode, args.out_dir, dev, dtype)
        return 0

    # GPU path: rooting under the per-family random-effects model (marginal logL per root).
    if args.rooting_rfx:
        for r in roots:
            rooting_rfx(r, args.fm_mode, dev, dtype, out_dir=args.out_dir, iters=args.em_iters * 2)
        return 0

    if args.mixture_k > 1:
        K = args.mixture_k
        for r in roots:
            root_pi, S, rb, mean_E, names, lab2n, _Ev = root_pi_and_E(r, D, L, T, args.fm_mode, dev, dtype)
            F = root_pi.shape[0]
            norm = F * math.log2(max(1e-300, 1.0 - mean_E))
            # baseline single-component
            log_pO1, ll1, _ = em_o(root_pi, S, args.em_iters)
            log_pi, log_pO, llK, nit, gamma = em_mixture(
                root_pi, S, K, args.em_iters, rb, seed=args.mix_seed)
            pi = np.exp2(log_pi); pO = np.exp2(log_pO)              # [K], [K,S]
            marg = (pi[:, None] * pO).sum(0)                        # [S] marginal p^O
            hard = gamma.argmax(1); cnt = np.bincount(hard, minlength=K)
            node2lab = {v: k for k, v in lab2n.items()}
            print(f"\n================ MIXTURE-{K} origination | root={r} ================")
            print(f"  data_logL(ln): single={ (ll1-norm)*_LN2:.1f}   mixture-{K}={ (llK-norm)*_LN2:.1f}   "
                  f"gain={ (llK-ll1)*_LN2:.1f} nats  (EM iters={nit})")
            for k in range(K):
                top = np.argsort(-pO[k])[:6]
                desc = ", ".join(f"{node2lab.get(int(s), names[int(s)] if int(s)<len(names) else s)}"
                                 f"={pO[k][int(s)]:.3f}" for s in top)
                print(f"  comp{k}: pi={pi[k]:.3f}  fams(hard)={int(cnt[k])}  O@root={pO[k][rb]:.3f}  "
                      f"top: {desc}")
            # deep clades: marginal + per-component origination
            print(f"  -- clade marginal p^O (single-comp DPANN was ~0); marg root_share={marg[rb]:.3f} --")
            for lab in ("DPANN", "Undinarchaeota", "Nanoarchaeota", "Asgard", "TackA", "Eury"):
                s = lab2n.get(lab)
                if s is None:
                    continue
                pc = " ".join(f"c{k}={pO[k][s]:.4f}" for k in range(K))
                print(f"     {lab:16s} marg={marg[s]:.4f}  [{pc}]")
        return 0

    out = {}
    for r in roots:
        root_pi, S, rb, mean_E, names, lab2n, _Ev = root_pi_and_E(r, D, L, T, args.fm_mode, dev, dtype)
        log_pO, ll, nit = em_o(root_pi, S, args.em_iters)
        data_logL = ll - root_pi.shape[0] * math.log2(max(1e-300, 1.0 - mean_E))
        pO = np.exp2(log_pO)
        top = float(pO.max()); eff = len({round(float(x), 3) for x in pO})
        out[r] = dict(logL_ln=data_logL * _LN2, O_top=top, O_eff=eff, O_root=float(pO[rb]), em_iters=nit)
        print(f"[EM-O] {r:16s} iters={nit:3d}  data_logL(ln)={out[r]['logL_ln']:13.1f}  "
              f"O_top={top:.3f} eff#={eff} O@root={float(pO[rb]):.3f}", flush=True)
    rows = sorted(out.items(), key=lambda kv: -kv[1]["logL_ln"])
    print("\n=== EM-O cold rooting (global DTL + EM origination) ===")
    b0 = rows[0][1]["logL_ln"]; er = next((i + 1 for i, (r, _) in enumerate(rows) if r == "Eury"), None)
    print(f"  Eury rank = #{er}")
    for i, (r, v) in enumerate(rows):
        tag = "[deep]" if r in DEEP else ("[SGA]" if r in SGA else "")
        fl = "  <-- Eury" if r == "Eury" else ""
        print(f"  #{i+1:2d} {r:16s} {v['logL_ln']:13.1f} d={v['logL_ln']-b0:8.1f} {tag}{fl}")
    print(f"  VERDICT: {'EM-O ROOTING WORKS (Eury #1)' if er == 1 else f'Eury #{er}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
