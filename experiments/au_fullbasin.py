"""AU test (Shimodaira 2002, multiscale bootstrap) on gpurec's OWN full free
per-branch (FULLbasin) rooting: for each candidate root, compute per-family logL
under that root's tree at its FULLbasin-fitted per-branch DTL+O rates, then run the
AU test across roots (the paper's rooting criterion). Tells us whether gpurec's
sharp Eury (#1 by ~3.5k nats) yields a SINGLE-root confidence set or a region.

Reuses eval_bigtree_at_alerax.au_rank (same bootstrap), but feeds gpurec FULLbasin
per-fam logL instead of AleRax's per_fam_likelihoods. Run on a largegpu node.
"""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402
    _load_species_helpers, _build_wave_layout, _sp_helpers_for_uniform,
    _parse_fraction_missing, _build_leaf_E,
)
from eval_at_alerax_rates import _load_families_named            # noqa: E402
from eval_bigtree_at_alerax import au_rank, DD, ALE, ROOTS, DEEP, SGA  # noqa: E402

_LN2 = math.log(2.0)
SIDE = "/work/SzollosiU/gergely-szollosi/williams_run/undine/FULLbasin_{root}.rates.txt.json"
CHUNK = 500
FM = "e-only"


def _lse2(x, dim):
    return torch.logsumexp(x * _LN2, dim=dim) / _LN2


def per_root_logl(root, device, dtype):
    """Per-family logL (ln) for `root` at its FULLbasin per-branch rates."""
    from gpurec.core.extract_parameters import extract_parameters_uniform
    from gpurec.core.likelihood import E_fixed_point, compute_log_likelihood
    from gpurec.core.forward import Pi_wave_forward
    sc = SIDE.format(root=root)
    if not Path(sc).exists():
        return None
    tree_path = DD / "4_species_tree" / f"Undine_C60_{root}root_short_name.nw"
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"]); s2i = sp["species_name_to_index"]
    d = json.load(open(sc))
    theta = torch.tensor(d["theta_log2"], dtype=dtype, device=device).reshape(S, 3)
    om = d.get("origination", {}).get("omega_log2")
    log_pO = None
    if om is not None:
        om = torch.tensor(om, dtype=dtype, device=device).reshape(S)
        log_pO = om - _lse2(om, dim=0)

    fams, names = _load_families_named(ALE, s2i, min_species=1, dtype=dtype)
    sp_gpu, anc = _sp_helpers_for_uniform(sp, device, dtype)
    urm = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)
    leaf_E = None
    fmp = DD / "fraction_missing"
    if FM != "off" and fmp.exists():
        fm, _n, _s = _parse_fraction_missing(fmp, s2i, S)
        leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
    leaf_obs_log = leaf_E if FM == "both" else None

    lp = extract_parameters_uniform(theta, urm, specieswise=True)
    E = E_fixed_point(species_helpers=sp_gpu, log_pS=lp[0], log_pD=lp[1], log_pL=lp[2],
                      transfer_mat=lp[3], max_transfer_mat=lp[4], max_iters=4000,
                      tolerance=1e-10, warm_start_E=None, dtype=dtype, device=device,
                      pibar_mode="uniform", ancestors_T=anc, leaf_E=leaf_E)
    per = {}
    for i in range(0, len(fams), CHUNK):
        chunk = fams[i:i + CHUNK]; nm = names[i:i + CHUNK]
        wl, rc = _build_wave_layout(chunk, device, dtype)
        Pi = Pi_wave_forward(wave_layout=wl, species_helpers=sp_gpu, E=E["E"], Ebar=E["E_bar"],
                             E_s1=E["E_s1"], E_s2=E["E_s2"], log_pS=lp[0], log_pD=lp[1],
                             log_pL=lp[2], transfer_mat=lp[3], max_transfer_mat=lp[4],
                             device=device, dtype=dtype, pibar_mode="uniform",
                             leaf_obs_log=leaf_obs_log)
        nll = compute_log_likelihood(Pi["Pi"], E["E"], rc, log_pO=log_pO)
        for j, v in enumerate(nll.detach().cpu().tolist()):
            per[nm[j]] = float(-v) * _LN2
    return per


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda"); dtype = torch.float64
    per_by_root = {}
    for r in ROOTS:
        p = per_root_logl(r, device, dtype)
        if p is None:
            print(f"[skip] {r}: no FULLbasin sidecar", flush=True); continue
        per_by_root[r] = p
        print(f"[done] {r:16s} sum_logL={sum(p.values()):.1f}  (F={len(p)})", flush=True)
    roots = [r for r in ROOTS if r in per_by_root]
    res, n = au_rank(per_by_root, roots)
    print("\n" + "=" * 64)
    print(f"AU TEST on gpurec FULLbasin rooting  (n={n} families, {len(roots)} roots)")
    print(f"{'root':16s} {'sum_logL':>13s} {'BP':>7s} {'AU':>7s}  region")
    conf = []
    for root, tot, bp, au in res:
        tag = "[deep]" if root in DEEP else ("[SGA]" if root in SGA else "")
        star = "  <== in AU 0.05 set" if au >= 0.05 else ""
        if au >= 0.05:
            conf.append(root)
        print(f"{root:16s} {tot:13.1f} {bp:7.3f} {au:7.3f}  {tag}{star}")
    print(f"\nAU 0.05 confidence set ({len(conf)}): {conf}")
    Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/au_fullbasin.json").write_text(
        json.dumps({"roots": [r for r, *_ in res],
                    "au": {r: au for r, _, _, au in res},
                    "sum_logL": {r: t for r, t, _, _ in res},
                    "conf_set_0.05": conf, "n": n}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
