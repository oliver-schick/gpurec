#!/usr/bin/env python
"""Benchmark driver: gpurec DTL rate fit on Williams .ale, mode-selectable.

The bundled experiments/run_williams_branchwise.py and gpurec_fit.py both
hardcode specieswise=True. This thin driver adds --mode {global,specieswise}
and otherwise reuses their exact helpers + optimize_theta_wave, so the global
run matches AleRax's default (one shared D,L,T) and the specieswise run matches
what we already validated. Staged into <gpurec>/experiments/ (untracked) by the
kit and run with the repo venv; imports run_williams_branchwise for the helpers.

  python experiments/bench_gpurec_fit.py --mode global --root DPANN \
      --data-dir <Williams_et_al_2017> --families 200 --fm-mode e-only --out rates.txt
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import (  # noqa: E402  (same helpers the validated run used)
    _resolve_paths, _load_species_helpers, _load_families, _build_wave_layout,
    _sp_helpers_for_uniform, _parse_fraction_missing, _build_leaf_E, KNOWN_ROOTS,
)


# --- auto-batch memory model + self-calibrating cache --------------------------
# A batch's peak memory scales with its total clade count: peak ~= bytes_per_clade
# * sum(C), where bytes_per_clade = K * S * elt and K = the number of live [.,S]
# buffers at the BACKWARD peak (implicit-grad + CG -- the true peak, larger than the
# forward). K is a property of the ALGORITHM (mode/dtype/pibar_mode), ~independent of
# which families are fed, so it transfers across datasets. We cache K per-config: the
# first run of a config uses the seed below and writes back the measured K; every
# later run of that config sizes the first batch optimally ("best batch after the
# first run"). An OOM (now a clean, catchable error thanks to the int64 offsets)
# triggers an automatic shrink-and-retry, so a bad guess costs a retry, not a job.
_MEM_TARGET_FRAC = 0.80          # aim the realized TRANSIENT peak at this fraction of free memory.
                                 # Aggressive on purpose: an over-shoot is now a CLEAN, catchable
                                 # OOM (int64 fix) that the retry loop below shrinks and re-runs, so
                                 # we chase big/fast batches and let the backstop catch the rare miss
                                 # -- rather than permanently shrinking batches (slower) out of caution.
_DEFAULT_K = 12.0                # seed for an unmeasured config: intentionally > the small-batch measured
                                 # K (~6 at 240k clades). Per-clade memory RISES with batch size because
                                 # FFD packs the densest families (most splits/clade -> the [N_splits,S]
                                 # backward buffer) into the biggest batch, so a K measured at a sub-max
                                 # batch under-predicts the peak. 12 keeps the first (seeded) batch safe.
_K_SAFETY = 1.8                  # inflate a CACHED K on use. Per-clade memory is SUPER-linear here (measured
                                 # ~97k B/clade at a 350k batch -> 32 GiB, but ~137k at 481k -> 66 GiB), because
                                 # bigger batches pull in denser families (more splits/clade -> [N_splits,S]).
                                 # So a K measured at one size under-predicts a larger one; 1.8 keeps the next
                                 # batch in the proven-safe ~390k zone (cached K=6 -> eff 10.8 -> ~42 GiB). The
                                 # OOM-shrink-retry loop catches any residual miss.
_CALIB_PATH = os.environ.get(
    "GPUREC_MEMBATCH_CALIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".membatch_calib.json"))


def _calib_key(mode, dtype_str, pibar):
    return f"{mode}|{dtype_str}|{pibar}"


def _load_calib_K(key):
    try:
        with open(_CALIB_PATH) as fh:
            return float(json.load(fh)[key]["K"])
    except Exception:
        return None


def _save_calib_K(key, K, extra):
    try:
        d = {}
        if os.path.exists(_CALIB_PATH):
            with open(_CALIB_PATH) as fh:
                d = json.load(fh)
        d[key] = {"K": round(float(K), 3), **extra}
        with open(_CALIB_PATH, "w") as fh:
            json.dump(d, fh, indent=2)
        print(f"      [auto-batch] cached K={K:.2f} for '{key}' -> {_CALIB_PATH}", flush=True)
    except Exception as e:
        print(f"      [auto-batch] warn: could not update calib cache: {e}", flush=True)


def _pack_batches(families, clade_budget):
    """First-fit-decreasing pack of families into VARIABLE-size batches, each with
    sum(C) <= clade_budget. Small families pack together; a family larger than the
    whole budget gets its own solo batch (keeping the others <= budget). Correctness
    is partition-independent (the optimizer sums per-family gradients). Returns
    (groups, meta): groups = list of family-index lists."""
    Cs = [int(f["C"]) for f in families]
    C_max = max(Cs)
    order = sorted(range(len(families)), key=lambda i: Cs[i], reverse=True)
    batches = []                                                # each: [indices, total_C]
    for i in order:
        c = Cs[i]
        for b in batches:
            if b[1] + c <= clade_budget:
                b[0].append(i); b[1] += c; break
        else:
            batches.append([[i], float(c)])
    groups = [b[0] for b in batches]
    sizes = [len(g) for g in groups]
    csums = [b[1] for b in batches]
    meta = {"n_batches": len(groups), "clade_budget": clade_budget,
            "max_batch_clades": max(csums), "C_max": C_max,
            "min_families_per_batch": min(sizes), "max_families_per_batch": max(sizes)}
    return groups, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="gpurec rate fit (global or specieswise) on Williams .ale")
    ap.add_argument("--mode", choices=["global", "specieswise"], default="global")
    # Williams-style input: a data dir with rooted_phylogeny/<root> + ccps/.
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--root", default="DPANN")
    # OR explicit input (any rooted tree + a glob of .ale; e.g. Davin):
    ap.add_argument("--species-tree", default=None, help="rooted species tree (newick)")
    ap.add_argument("--ale-glob", default=None, help="glob for .ale files, e.g. '/…/ALEs/*.ale'")
    ap.add_argument("--fraction-missing", default=None, help="whitespace 'species fraction' file")
    ap.add_argument("--families", type=int, default=0, help="0 = all; else first N")
    ap.add_argument("--min-species", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--optimizer", default="lbfgs", choices=["lbfgs", "adam", "sgd"])
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--init-rate", type=float, default=0.1)
    ap.add_argument("--fm-mode", default="e-only", choices=["both", "e-only", "off"])
    ap.add_argument("--family-batch-size", default="auto",
                    help="'auto' (variable clade-budget batches from free GPU mem + family "
                         "clade counts), 0 (all families at once), or a positive int (fixed count)")
    ap.add_argument("--mem-safety", type=float, default=0.55,
                    help="auto-batch: fraction of free GPU memory to budget when --mem-factor is manual")
    ap.add_argument("--mem-factor", default="auto",
                    help="auto-batch: 'auto' (self-calibrating -- size the batch from a cached/seeded "
                         "per-clade memory estimate and refine it each run) or a float (manual: peak "
                         "[clades,S] buffers held at once; higher = smaller, safer batches).")
    ap.add_argument("--mem-retries", type=int, default=4,
                    help="auto-batch: on CUDA OOM, shrink the batch ~30%% and retry, up to N times")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    # --mem-factor accepts 'auto' (self-calibrating) or a manual float override.
    _mf = str(args.mem_factor).strip().lower()
    mem_factor_val = None if _mf == "auto" else float(_mf)
    # --family-batch-size accepts 'auto' or an int; resolved after families load.
    fbs_raw = str(args.family_batch_size).strip().lower()
    if fbs_raw not in ("auto",):
        try:
            fbs_raw = int(fbs_raw)
        except ValueError:
            raise SystemExit(f"--family-batch-size must be 'auto' or an int, got {args.family_batch_size!r}")

    import glob as _glob
    if args.species_tree:                       # explicit input (Davin etc.)
        if not args.ale_glob:
            raise SystemExit("--species-tree requires --ale-glob")
        tree_path = Path(args.species_tree)
        ale_paths = sorted(p for p in _glob.glob(args.ale_glob)
                           if not Path(p).name.startswith("._"))
        fm_path = Path(args.fraction_missing) if args.fraction_missing else Path("/nonexistent")
    else:                                       # Williams data-dir/root layout
        if not args.data_dir:
            raise SystemExit("need either --data-dir (+ --root) or --species-tree (+ --ale-glob)")
        data_dir = Path(args.data_dir)
        if args.root not in KNOWN_ROOTS:
            print(f"[warn] root {args.root!r} not in {KNOWN_ROOTS}", flush=True)
        tree_path, ale_paths, fm_path = _resolve_paths(data_dir, args.root, None)
    if not tree_path.exists():
        raise SystemExit(f"species tree not found: {tree_path}")
    if not ale_paths:
        raise SystemExit(f"no .ale files found (glob/data-dir empty)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (run on an A100).")
    device = torch.device("cuda")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    elt = torch.tensor([], dtype=dtype).element_size()          # 8 (f64) / 4 (f32)
    specieswise = (args.mode == "specieswise")
    # Measure free GPU memory now, while nothing is allocated yet, so auto-batch
    # sizing sees the true capacity (the wave layout + optimizer allocate later).
    free_bytes = torch.cuda.mem_get_info(device)[0]
    _gib = lambda b: b / 2**30
    _memnow = lambda: torch.cuda.memory_allocated(device)

    from gpurec.optimization.wave_optimizer import optimize_theta_wave

    print(f"[1/4] species tree {tree_path.name}", flush=True)
    sp = _load_species_helpers(str(tree_path))
    S = int(sp["S"]); names = list(sp["names"]); name2idx = sp["species_name_to_index"]

    print(f"[2/4] loading .ale families (min-species={args.min_species}) ...", flush=True)
    families, stats = _load_families(ale_paths, name2idx, min_species=args.min_species,
                                     dtype=dtype, limit=args.families)
    print(f"      kept {stats['kept']} / considered {stats['n_considered']}", flush=True)
    if not families:
        raise SystemExit("no usable families after filtering")
    total_C = sum(int(f["C"]) for f in families)
    print(f"      total clades ΣC = {total_C:,}  (mean {total_C/len(families):,.0f}/family)", flush=True)

    # --- Fixed per-run setup (built once, reused across any OOM retries) ---
    print(f"[3/4] cross-family wave layout for {len(families)} families ...", flush=True)
    _m_pre = _memnow()
    # Full layout: required positionally by the optimizer (fallback / f32->f64 paths);
    # the heavy compute uses the per-batch layouts (auto) built in the retry loop below.
    wave_layout, root_clade_ids = _build_wave_layout(families, device, dtype)
    print(f"      [mem] full wave_layout: +{_gib(_memnow()-_m_pre):.2f} GiB", flush=True)
    sp_gpu, _aT = _sp_helpers_for_uniform(sp, device, dtype)
    unnorm_row_max = torch.log2(sp["Recipients_mat"]).max(dim=-1).values.to(device=device, dtype=dtype)
    # fraction-missing: e-only matches AleRax (extinction term only).
    leaf_E = None; leaf_obs_log = None; fm_used = False
    if args.fm_mode != "off" and fm_path.exists():
        fm, _n, _sk = _parse_fraction_missing(fm_path, name2idx, S)
        leaf_E = _build_leaf_E(sp, fm, S, dtype)[0].to(device=device, dtype=dtype)
        leaf_obs_log = leaf_E if args.fm_mode == "both" else None
        fm_used = True
    init_log2 = math.log2(args.init_rate)
    theta_init = init_log2 * (torch.ones(S, 3, dtype=dtype, device=device) if specieswise
                              else torch.ones(3, dtype=dtype, device=device))

    def _run_optimize(wlb, fbs):
        # resident = fixed overhead (layouts+helpers+leaf_E) held across the optimize;
        # per-batch [C,S] buffers alloc/free inside the loop, so peak-resident = the
        # transient cost that scales with the batch's ΣC (used to back out bytes/clade).
        _resident = _memnow()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        print(f"[4/4] optimizing {args.mode} theta (steps={args.steps}, opt={args.optimizer}, "
              f"fm={args.fm_mode}; resident={_gib(_resident):.2f}GiB) ...", flush=True)
        _t = time.time()
        r = optimize_theta_wave(
            wave_layout=wave_layout, species_helpers=sp_gpu, root_clade_ids=root_clade_ids,
            unnorm_row_max=unnorm_row_max, theta_init=theta_init, steps=args.steps,
            optimizer=args.optimizer, specieswise=specieswise, pibar_mode="uniform",
            families=families, family_batch_size=fbs, wave_layout_batches=wlb,
            device=device, dtype=dtype, leaf_E=leaf_E, leaf_obs_log=leaf_obs_log, verbose=True)
        return r, _resident, time.time() - _t

    # --- Resolve batching + run. 'auto' self-calibrates: pick clade_budget from a
    # cached/seeded K, and on OOM shrink 40% and retry (int64 made OOM catchable). ---
    calib_key = _calib_key(args.mode, args.dtype, "uniform")
    batch_meta = None; family_batch_size = 0
    result = None; resident = 0; elapsed = 0.0
    if fbs_raw == "auto":
        if mem_factor_val is not None:                       # manual factor (back-compat)
            clade_budget = args.mem_safety * free_bytes / (S * elt * mem_factor_val)
            K_src = f"manual mem_factor={mem_factor_val:g}"
        else:                                                # cached (or seeded) K
            _cK = _load_calib_K(calib_key)
            _K0 = (_cK * _K_SAFETY) if _cK is not None else _DEFAULT_K
            K_src = (f"cached K={_cK:.2f}x{_K_SAFETY}={_K0:.2f}" if _cK is not None
                     else f"seed K={_K0:.2f} (config not yet calibrated)")
            clade_budget = _MEM_TARGET_FRAC * free_bytes / (_K0 * S * elt)
        wlb = None
        for attempt in range(max(1, args.mem_retries)):
            batch_groups, batch_meta = _pack_batches(families, clade_budget)
            print(f"      [auto-batch] try {attempt+1}/{args.mem_retries}: {K_src}  "
                  f"free={_gib(free_bytes):.1f}GiB target~{_MEM_TARGET_FRAC:.0%}  "
                  f"clade_budget={clade_budget:,.0f} -> {batch_meta['n_batches']} batches | "
                  f"fam/batch {batch_meta['min_families_per_batch']}-{batch_meta['max_families_per_batch']} | "
                  f"max {batch_meta['max_batch_clades']:,.0f} clades (C_max={batch_meta['C_max']})", flush=True)
            try:
                wlb = [_build_wave_layout([families[i] for i in idxs], device, dtype)
                       for idxs in batch_groups]
                result, resident, elapsed = _run_optimize(wlb, 0)
                break
            except torch.cuda.OutOfMemoryError as e:
                wlb = None
                torch.cuda.empty_cache()
                nb = clade_budget * 0.7          # gentle back-off (~30%); 0.7^4 still covers a 4x over-estimate
                print(f"      [auto-batch] OOM at clade_budget={clade_budget:,.0f} -> shrink to "
                      f"{nb:,.0f} & retry ({str(e).splitlines()[0][:90]})", flush=True)
                clade_budget = nb; K_src = "post-OOM shrink"
        if result is None:
            raise SystemExit(f"auto-batch: still OOM after {args.mem_retries} shrinks "
                             f"(clade_budget down to {clade_budget:,.0f}) -- GPU too small?")
    else:
        family_batch_size = fbs_raw
        print(f"      [batch] explicit --family-batch-size={family_batch_size}"
              f"{' (all families at once)' if family_batch_size == 0 else ''}", flush=True)
        result, resident, elapsed = _run_optimize(None, family_batch_size)

    peak_bytes = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    peak_gb = peak_bytes / 2**30

    rates = torch.as_tensor(result["rates"]).detach().cpu()
    nll = float(result["negative_log_likelihood"])
    logL = float(result.get("log_likelihood", -nll))
    print(f"\n  DONE  mode={args.mode}  root={args.root}  families={len(families)}  "
          f"time={elapsed:.1f}s  NLL(log2)={nll:.4f}", flush=True)

    # ---- Memory calibration: measure the real per-clade cost, cache it -------------
    # peak = resident (fixed) + transient (scales with the batch's ΣC). bytes_per_clade
    # = transient / max_batch_clades; K = bytes_per_clade/(S*elt) is the algorithm's
    # live-buffer count. Cache K per (mode,dtype,pibar) so the NEXT run of this config
    # sizes the first batch right -- "best batch after the first run".
    peak_batch_clades = (batch_meta["max_batch_clades"] if batch_meta is not None
                         else (total_C if family_batch_size == 0 else None))
    transient_bytes = max(0, peak_bytes - resident)
    calib = None
    if peak_batch_clades and peak_batch_clades > 0 and transient_bytes > 0:
        bytes_per_clade = transient_bytes / peak_batch_clades
        K = bytes_per_clade / (S * elt)
        calib = {"resident_gib": _gib(resident), "peak_gib": peak_gb,
                 "transient_gib": _gib(transient_bytes), "peak_batch_clades": peak_batch_clades,
                 "bytes_per_clade": bytes_per_clade, "K": K}
        print(f"  [mem] resident={_gib(resident):.2f}GiB peak={peak_gb:.2f}GiB "
              f"transient={_gib(transient_bytes):.2f}GiB over {peak_batch_clades:,.0f}-clade peak batch", flush=True)
        print(f"  [mem-calib] bytes/clade={bytes_per_clade:,.0f}  K={K:.2f} live [.,S] buffers (S={S}, elt={elt})",
              flush=True)
        # Persist K for this config -- but only from the self-calibrating path, and only
        # if we filled a healthy fraction of the budget (a tiny/OOM-shrunk batch would
        # under-measure the peak and cache too-optimistic a K).
        if fbs_raw == "auto" and mem_factor_val is None and batch_meta is not None \
                and peak_gb > 0.30 * _gib(free_bytes):
            _save_calib_K(calib_key, K, {"S": S, "dtype": args.dtype,
                                         "measured_peak_gib": round(peak_gb, 2),
                                         "measured_on_families": len(families)})
        else:
            print(f"  [mem-calib] (not caching K: manual/partial run — measured only "
                  f"{peak_gb:.1f} GiB of {_gib(free_bytes):.0f})", flush=True)
    else:
        print(f"  [mem] realized peak = {peak_gb:.2f} GiB (resident {_gib(resident):.2f} GiB; "
              f"bytes/clade not derivable for this batching mode)", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("# node D L T\n")
        if specieswise:
            r = rates.reshape(S, 3)
            for s in range(S):
                fh.write(f"{names[s]} {float(r[s,0]):.10g} {float(r[s,1]):.10g} {float(r[s,2]):.10g}\n")
        else:
            r = rates.flatten()
            fh.write(f"GLOBAL {float(r[0]):.10g} {float(r[1]):.10g} {float(r[2]):.10g}\n")
    sidecar = out.with_suffix(out.suffix + ".json")
    json.dump({
        "command": " ".join(sys.argv), "mode": args.mode, "root": args.root, "S": S,
        "n_families_kept": len(families), "n_families_considered": stats["n_considered"],
        "fraction_missing_used": fm_used, "fm_mode": args.fm_mode,
        "family_batch_size_arg": str(args.family_batch_size),
        "batching": ("clade_budget" if batch_meta is not None
                     else ("all_at_once" if family_batch_size == 0 else "fixed_count")),
        "family_batch_size_used": family_batch_size,
        "batch_plan": batch_meta,
        "total_clades": total_C,
        "mem_safety": args.mem_safety, "mem_factor": str(args.mem_factor),
        "calib_key": calib_key if fbs_raw == "auto" else None,
        "peak_gib": peak_gb, "mem_calibration": calib,
        "negative_log_likelihood_log2": nll, "log_likelihood_log2": logL,
        "negative_log_likelihood_ln": nll * math.log(2.0),
        "rates": rates.tolist(), "names": names if specieswise else None,
        "elapsed_s": elapsed, "n_steps": len(result.get("history", [])),
    }, open(sidecar, "w"), indent=2)
    print(f"  rates -> {out}\n  sidecar -> {sidecar}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
