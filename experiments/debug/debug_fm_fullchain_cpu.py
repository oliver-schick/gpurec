"""Full-chain fraction-missing gradient test on pure CPU (NO Triton, NO CUDA).

Reproduces experiments/validate_fraction_missing_wave.py's FD-vs-analytic gradient
check WITHOUT GPU, by replacing the Triton Pi forward with the pure-PyTorch
``_self_loop_differentiable`` iteration, while using the REAL
``implicit_grad_loglik_vjp_wave`` (dense backward is pure torch) + ``E_fixed_point``.

It isolates the fraction-missing gradient bug:
  * (A) full chain  : FD of NLL(theta) vs analytic grad_theta, fm>0.
  * (B) Pi-only     : FD with E FROZEN  vs grad_theta_pi   (isolates H1).
  * (C) E-only      : FD(full) - FD(Efrozen) vs gtheta_E   (isolates H2).

Run:  PYTHONPATH=. python experiments/debug/debug_fm_fullchain_cpu.py
"""
import sys, types, math
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Pre-import torch internals that lazily touch a real triton, BEFORE we stub it.
import torch  # noqa: E402
import torch._dynamo  # noqa: F401,E402
from torch import func as _tfunc_preload  # noqa: F401,E402
_ = torch.autograd.grad  # force functorch eager_transforms import path

# ── Stub triton so module-level imports succeed; dense CPU path never calls it ──
_triton = types.ModuleType('triton')
_triton.__file__ = '<triton-stub>'
_triton.jit = lambda fn=None, **kw: ((lambda f: f)(fn) if fn else (lambda f: f))
_triton.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
_triton.cdiv = lambda a, b: -(-a // b)
_tl = types.ModuleType('triton.language')
_tl.__file__ = '<triton-language-stub>'
_tl.constexpr = int


class _TLDtype:  # torch._dynamo references triton.language.dtype as a type
    pass


_tl.dtype = _TLDtype
for _nm in ['program_id', 'load', 'store', 'arange', 'zeros', 'full', 'where',
            'maximum', 'minimum', 'exp2', 'log2', 'sum', 'atomic_add', 'dot',
            'cdiv', 'cos', 'sin', 'abs', 'where', 'static_assert', 'static_print']:
    setattr(_tl, _nm, lambda *a, **k: None)
_tl.float32 = _tl.float64 = _tl.int32 = _tl.int64 = int
_triton.language = _tl
sys.modules['triton'] = _triton
sys.modules['triton.language'] = _tl

import torch
from contextlib import contextmanager
import gpurec.core._helpers as _h
import gpurec.core.likelihood as _lk


@contextmanager
def _noop(*a, **k):
    yield


_h._nvtx_range = _noop
_lk._nvtx_range = _noop
# CPU: make cuda.synchronize a no-op (implicit_grad calls it for timing).
torch.cuda.synchronize = lambda *a, **k: None

from gpurec.core.likelihood import E_fixed_point
from gpurec.core.backward import _self_loop_differentiable, _dts_cross_differentiable, NEG_INF
from gpurec.core.log2_utils import _safe_log2_internal as _safe_log2, logsumexp2
from gpurec.core.scheduling import compute_clade_waves
from gpurec.core.batching import collate_wave, build_wave_layout, collate_gene_families
from gpurec.core.extract_parameters import extract_parameters, extract_parameters_uniform
from gpurec.optimization.implicit_grad import implicit_grad_loglik_vjp_wave

dtype = torch.float64
device = torch.device('cpu')
INV = 1.0 / math.log(2.0)
SP = str(_REPO / "tests/data/test_trees_1/sp.nwk")
G = str(_REPO / "tests/data/test_trees_1/g.nwk")
FM = 0.3


def _load_ext():
    from torch.utils.cpp_extension import load
    cpp = _REPO / "gpurec" / "core" / "cpp"
    srcs = [str(cpp / f) for f in ("preprocess.cpp", "tree_utils.cpp", "clade_utils.cpp")]
    inc = Path("/opt/homebrew/opt/libomp/include")
    lib = Path("/opt/homebrew/opt/libomp/lib")
    extra_cflags = ["-O3", "-Xpreprocessor", "-fopenmp", f"-I{inc}"]
    extra_ldflags = [f"-L{lib}", "-lomp"]
    bdir = Path.home() / ".cache" / "gpurec_fm_cpu" / "preprocess_cpp"
    bdir.mkdir(parents=True, exist_ok=True)
    return load(name="preprocess_cpp", sources=srcs, extra_cflags=extra_cflags,
                extra_ldflags=extra_ldflags, build_directory=str(bdir), verbose=False)


ext = _load_ext()
raw = ext.preprocess(SP, [G])
sr, cr = raw['species'], raw['ccp']
S = int(sr['S'])
ch = {
    "split_leftrights_sorted": cr["split_leftrights_sorted"],
    "log_split_probs_sorted": cr["log_split_probs_sorted"].to(dtype=dtype) * INV,
    "seg_parent_ids": cr["seg_parent_ids"],
    "ptr_ge2": cr["ptr_ge2"],
    "num_segs_ge2": int(cr["num_segs_ge2"]),
    "num_segs_eq1": int(cr["num_segs_eq1"]),
    "end_rows_ge2": int(cr["end_rows_ge2"]),
    "C": int(cr["C"]),
    "N_splits": int(cr["N_splits"]),
    "split_parents_sorted": cr["split_parents_sorted"],
}
batch_items = [{
    "ccp": ch,
    "leaf_row_index": raw["leaf_row_index"].long(),
    "leaf_col_index": raw["leaf_col_index"].long(),
    "root_clade_id": int(cr["root_clade_id"]),
}]
sh = {
    "S": S, "names": sr['names'],
    "s_P_indexes": sr["s_P_indexes"].to(device),
    "s_C12_indexes": sr["s_C12_indexes"].to(device),
    "Recipients_mat": sr["Recipients_mat"].to(dtype=dtype, device=device),
    "ancestors_dense": sr["ancestors_dense"].to(dtype=dtype, device=device),
}
ancestors_T = sh['ancestors_dense'].T.to_sparse_coo()
tm_unnorm = torch.log2(sh["Recipients_mat"])
unnorm_row_max = tm_unnorm.max(dim=-1).values

# species-leaf mask: nodes never internal parents
sP = sh['s_P_indexes']
internal = sP[sP < S].unique()
leaf_mask = torch.ones(S, dtype=torch.bool, device=device)
leaf_mask[internal] = False
fm_vec = torch.zeros(S, dtype=dtype, device=device)
fm_vec[leaf_mask] = FM
leaf_E = torch.where(fm_vec > 0, torch.log2(fm_vec.clamp_min(1e-300)),
                     torch.full_like(fm_vec, NEG_INF))

# wave layout
batched = collate_gene_families(batch_items, dtype=dtype, device=device)
fw, fp = [], []
for bi in batch_items:
    w, p = compute_clade_waves(bi['ccp'])
    fw.append(w); fp.append(p)
offsets = [m['clade_offset'] for m in batched['family_meta']]
cross_waves = collate_wave(fw, offsets)
max_n = max(len(p) for p in fp)
cross_phases = [max(x[k] if k < len(x) else 1 for x in fp) for k in range(max_n)]
wave_layout = build_wave_layout(
    waves=cross_waves, phases=cross_phases, ccp_helpers=batched['ccp'],
    leaf_row_index=batched['leaf_row_index'], leaf_col_index=batched['leaf_col_index'],
    root_clade_ids=batched['root_clade_ids'], device=device, dtype=dtype,
    family_clade_counts=[m['C'] for m in batched['family_meta']],
    family_clade_offsets=[m['clade_offset'] for m in batched['family_meta']],
)

p_cpu = sh['s_P_indexes'].long(); c_cpu = sh['s_C12_indexes'].long()
mask_c1 = p_cpu < S
sp_child1 = torch.full((S,), S, dtype=torch.long)
sp_child2 = torch.full((S,), S, dtype=torch.long)
sp_child1[p_cpu[mask_c1]] = c_cpu[mask_c1]
sp_child2[(p_cpu[~mask_c1] - S)] = c_cpu[~mask_c1]


def _extract(theta, pibar_mode):
    if pibar_mode == 'dense':
        pS, pD, pL, tf, mt_raw = extract_parameters(
            theta, tm_unnorm, genewise=False, specieswise=True, pairwise=False)
        mt = mt_raw.squeeze(-1) if mt_raw.ndim == 2 else mt_raw
        return pS, pD, pL, tf, mt
    pS, pD, pL, tf, mt = extract_parameters_uniform(theta, unnorm_row_max, specieswise=True)
    return pS, pD, pL, tf, mt


def full_forward(theta, pibar_mode, leaf_E_use, E_override=None):
    """Pure-torch forward. If E_override given, E is FROZEN to it (for Pi-only FD)."""
    pS, pD, pL, tf, mt = _extract(theta, pibar_mode)
    if E_override is None:
        Eo = E_fixed_point(species_helpers=sh, log_pS=pS, log_pD=pD, log_pL=pL,
                           transfer_mat=tf, max_transfer_mat=mt, max_iters=8000,
                           tolerance=1e-14, warm_start_E=None, dtype=dtype, device=device,
                           pibar_mode=pibar_mode, ancestors_T=ancestors_T, leaf_E=leaf_E_use)
    else:
        Eo = E_override
    E = Eo['E']; Ebar = Eo['E_bar']; E_s1 = Eo['E_s1']; E_s2 = Eo['E_s2']

    wave_metas = wave_layout['wave_metas']
    C_total = wave_layout['ccp_helpers']['C']
    Pi = torch.full((C_total, S), NEG_INF, dtype=dtype, device=device)
    lr = wave_layout['leaf_row_index'].long(); lc = wave_layout['leaf_col_index'].long()
    Pi[lr, lc] = 0.0
    Pibar = torch.full((C_total, S), NEG_INF, dtype=dtype, device=device)

    DL = 1.0 + pD + E
    SL1 = pS + E_s2
    SL2 = pS + E_s1
    tmT = None if pibar_mode == 'uniform' else tf.T.contiguous()

    # fraction-missing leaf baseline cols
    fm_cols = leaf_E_use > NEG_INF if leaf_E_use is not None else None

    for wi in range(len(wave_metas)):
        meta = wave_metas[wi]; ws = meta['start']; W = meta['W']; we = ws + W
        if meta['has_splits']:
            dts_r = _dts_cross_differentiable(Pi, Pibar, meta, sp_child1, sp_child2,
                                              pD, pS, S, device, dtype)
        else:
            dts_r = None
        leaf_wt = torch.full((W, S), NEG_INF, dtype=dtype, device=device)
        if leaf_E_use is not None:
            leaf_wt[:, fm_cols] = leaf_E_use[fm_cols]
        m = (lr >= ws) & (lr < we)
        if m.any():
            leaf_wt[lr[m] - ws, lc[m]] = 0.0
        leaf_wt = pS + leaf_wt

        Pi_W = Pi[ws:we].clone()
        for _ in range(2000):
            Pi_new = _self_loop_differentiable(
                Pi_W, mt, DL, Ebar, E, SL1, SL2, sp_child1, sp_child2,
                leaf_wt, dts_r, S, pibar_mode=pibar_mode,
                transfer_mat_T=tmT, ancestors_T=ancestors_T)
            sig = Pi_new > -100.0
            diff = torch.abs(Pi_new - Pi_W)[sig].max().item() if sig.any() else 0.0
            Pi_W = Pi_new
            if diff < 1e-13:
                break
        Pi[ws:we] = Pi_W.detach()
        Pi_max = Pi_W.max(dim=1, keepdim=True).values
        Pi_exp = torch.exp2(Pi_W - Pi_max)
        if pibar_mode == 'uniform':
            anc = Pi_exp @ ancestors_T
            Pibar[ws:we] = (_safe_log2(Pi_exp.sum(1, keepdim=True) - anc) + Pi_max + mt.unsqueeze(0)).detach()
        else:
            Pibar[ws:we] = (_safe_log2(Pi_exp @ tmT) + Pi_max + mt.unsqueeze(0)).detach()

    root_ids = wave_layout['root_clade_ids']
    lse = logsumexp2(Pi[root_ids], dim=-1)
    numerator = lse - math.log2(S)
    denom = torch.log2(1 - torch.exp2(E).mean(dim=-1))
    nll = -(numerator - denom).sum()
    return float(nll), Pi, Pibar, Eo, (pS, pD, pL, tf, mt)


def analytic(theta, pibar_mode, leaf_E_use):
    nll, Pi, Pibar, Eo, (pS, pD, pL, tf, mt) = full_forward(theta, pibar_mode, leaf_E_use)
    g, _ = implicit_grad_loglik_vjp_wave(
        wave_layout, sh,
        Pi_star_wave=Pi.detach(), Pibar_star_wave=Pibar.detach(),
        E_star=Eo['E'], E_s1=Eo['E_s1'], E_s2=Eo['E_s2'], Ebar=Eo['E_bar'],
        log_pS=pS, log_pD=pD, log_pL=pL, max_transfer_mat=mt,
        root_clade_ids_perm=wave_layout['root_clade_ids'],
        theta=theta, unnorm_row_max=unnorm_row_max, specieswise=True,
        device=device, dtype=dtype, neumann_terms=12, use_pruning=False,
        cg_tol=1e-13, cg_maxiter=2000, pibar_mode=pibar_mode,
        transfer_mat=tf if pibar_mode == 'dense' else None,
        transfer_mat_unnormalized=tm_unnorm if pibar_mode == 'dense' else None,
        ancestors_T=ancestors_T, leaf_E=leaf_E_use,
    )
    return g.detach(), Eo


theta0 = (math.log2(0.1) * torch.ones(S, 3, dtype=dtype, device=device)).clone()
leaf_ids = torch.nonzero(leaf_mask).flatten().tolist()
internal_ids = torch.nonzero(~leaf_mask).flatten().tolist()
probe = []
for s in (leaf_ids[:2] + internal_ids[:2]):
    for col in range(3):
        probe.append((int(s), col))
probe = probe[:8]
h = 1e-5

for pibar_mode in ('dense', 'uniform'):
    print(f"\n================ pibar_mode = {pibar_mode} (fm={FM}) ================")
    g, Eo0 = analytic(theta0, pibar_mode, leaf_E)
    worst = (0.0, None)
    worst_pi = (0.0, None)
    worst_e = (0.0, None)
    for (s, c) in probe:
        tp = theta0.clone(); tp[s, c] += h
        tm = theta0.clone(); tm[s, c] -= h
        # full FD
        fp, *_ = full_forward(tp, pibar_mode, leaf_E)
        fm_, *_ = full_forward(tm, pibar_mode, leaf_E)
        fd_full = (fp - fm_) / (2 * h)
        # E-frozen FD (Pi-only): freeze E to base E*, vary only theta in Pi/likelihood-params
        fp2, *_ = full_forward(tp, pibar_mode, leaf_E, E_override=Eo0)
        fm2, *_ = full_forward(tm, pibar_mode, leaf_E, E_override=Eo0)
        fd_pi = (fp2 - fm2) / (2 * h)
        fd_e = fd_full - fd_pi
        an = float(g[s, c])
        rel = abs(fd_full - an) / max(abs(fd_full), abs(an), 1e-9)
        tag = ' [L/pL]' if c == 1 else ''
        print(f"  s={s} c={c}{tag}: FD_full={fd_full:+.6e} analytic={an:+.6e} "
              f"rel={rel:.3e} | FD_pi={fd_pi:+.6e} FD_E={fd_e:+.6e}")
        if rel > worst[0]:
            worst = (rel, (s, c, fd_full, an))
    print(f"  WORST full rel-err = {worst[0]:.3e} at {worst[1]} -> "
          f"{'PASS' if worst[0] < 1e-4 else 'FAIL'}")
