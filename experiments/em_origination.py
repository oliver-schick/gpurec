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


def root_pi_and_E(root, D, L, T, fm_mode, dev, dtype):
    """Forward at GLOBAL (D,L,T) -> per-family root-clade Pi [F,S] (numpy) + mean extinction."""
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point
    from gpurec.core.forward import Pi_wave_forward
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
    theta = torch.log2(torch.tensor([D, L, T], dtype=dtype, device=dev)).reshape(1, 3).repeat(S, 1)
    lp = extract_parameters_uniform(theta, urm, specieswise=True)
    E = E_fixed_point(species_helpers=sp_gpu, log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                      transfer_mat=lp[3], max_transfer_mat=lp[4], max_iters=4000, tolerance=1e-10,
                      warm_start_E=None, dtype=dtype, device=dev, pibar_mode="uniform",
                      ancestors_T=anc, leaf_E=leaf_E)
    rows = []
    for i in range(0, len(fams), CHUNK):
        wl, rc = _build_wave_layout(fams[i:i + CHUNK], dev, dtype)
        Pi = Pi_wave_forward(wave_layout=wl, species_helpers=sp_gpu, E=E["E"], Ebar=E["E_bar"],
                             E_s1=E["E_s1"], E_s2=E["E_s2"], log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                             transfer_mat=lp[3], max_transfer_mat=lp[4], device=dev, dtype=dtype,
                             pibar_mode="uniform", leaf_obs_log=leaf_obs)["Pi"]
        rows.append(Pi[rc].cpu().numpy())                      # [chunk, S] root-clade Pi (log2)
    root_pi = np.concatenate(rows, 0)                           # [F, S]
    mean_E = float(torch.exp2(E["E"]).mean().item())
    names = list(sp["names"])
    # map labelled clades -> gpurec node index (for reading mixture components)
    label2node = {}
    try:
        from clade_groups import parse_labeled_newick, species_node_leafsets
        leafsets = species_node_leafsets(sp, S)
        labeled = parse_labeled_newick(str(tree))
        for s in range(S):
            lab = labeled.get(leafsets[s])
            if lab:
                label2node[lab] = s
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] clade labels unavailable: {e}", flush=True)
    return root_pi, S, root_branch, mean_E, names, label2node


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
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    dev = torch.device("cuda"); dtype = torch.float64
    D, L, T = (float(x) for x in args.dlt.split(","))
    roots = args.roots.split(",") if args.roots else ROOTS

    if args.mixture_k > 1:
        K = args.mixture_k
        for r in roots:
            root_pi, S, rb, mean_E, names, lab2n = root_pi_and_E(r, D, L, T, args.fm_mode, dev, dtype)
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
        root_pi, S, rb, mean_E, names, lab2n = root_pi_and_E(r, D, L, T, args.fm_mode, dev, dtype)
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
