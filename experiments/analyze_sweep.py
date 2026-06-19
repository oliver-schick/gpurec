#!/usr/bin/env python
"""Aggregate a Brownian-prior stiffness sweep and analyze the effect of sigma.

The sweep runs gpurec's branch-wise specieswise optimizer with a Brownian
(time-uniform TKP) rate prior at several prior stiffnesses ``sigma`` (log2),
over two datasets and ten roots. Each run writes a rate file

    <sweep-dir>/<dataset>_<root>_s<sigma>.rates.txt

(``# node D L T`` header then ``<species> <D> <L> <T>`` rows; 60 leaves + 59
internal nodes) and -- if the run finished cleanly -- a JSON sidecar
``<...>.rates.txt.json`` holding the config + final NLL/logL.

The prior stiffness sigma controls how strongly neighbouring branches are
pulled together along the species tree:

  * small sigma  -> STIFF: rates are smoothed / clade-grouped, low per-branch
                    dispersion (the rates look "un-smoothed-out");
  * large sigma  -> LOOSE: each branch free to fit its own rate, high
                    dispersion, more floor / runaway branches.

This script computes per-run dispersion / floor / runaway / median metrics,
the per-axis AleRax leaf-branch match (treedists dataset only, which is the
AleRax-comparable one), aggregates across roots at each sigma, and prints the
TREND of how sigma affects dispersion, AleRax match, and loss identifiability.

Parsing / correlation helpers are reused from
``experiments/compare_williams_alerax.py`` (parse_rates, leaf_names_from_newick,
pearson, spearman, log10c).

Pure stdlib; matplotlib is optional (plots are a bonus, text tables are the
deliverable).

Usage (on the cluster):

    python experiments/analyze_sweep.py \
        --sweep-dir /work/.../williams_run/sweep \
        --out-csv   sweep_per_run.csv \
        --out-dir   sweep_plots
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Reuse the parsing / correlation helpers from the comparison script.  Import by
# path so this works regardless of how the package is installed / PYTHONPATH.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from compare_williams_alerax import (  # noqa: E402
    parse_rates,
    leaf_names_from_newick,
    pearson,
    spearman,
    log10c,
)

# ── sweep grid ────────────────────────────────────────────────────────────────
DATASETS = ["treedists", "complete"]
ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN",
         "Eury", "HaloThermoplas", "Kor", "TAC", "TackA"]
SIGMAS = [0.1, 0.25, 0.5, 1.0, 2.0]
# treedists is the AleRax-comparable dataset; complete is not.
ALERAX_DATASET = "treedists"

DEFAULT_DATA_DIR = ("/work/SzollosiU/gergely-szollosi/williams_run/data/"
                    "3_Reconciliation/Williams_et_al_2017")

FLOOR_THRESH = 1e-9   # rate <= this == hit the floor
HI_THRESH = 10.0      # rate > this == runaway / poorly identified

AXES = [("D", 0), ("L", 1), ("T", 2)]


# ── small helpers ─────────────────────────────────────────────────────────────
def _fmt_sigma(sigma: float) -> str:
    """Render a sigma the same way the sweep filenames do (0.1, 0.25, ... 1.0)."""
    s = ("%g" % sigma)
    # "1" -> "1.0", "2" -> "2.0" to match s1.0 / s2.0 filenames; leave 0.1/0.25.
    if "." not in s and "e" not in s:
        s += ".0"
    return s


def _median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return float("nan")
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def _stdev(vals):
    """Population-ish sample standard deviation (n-1); nan for <2 points."""
    v = [x for x in vals if not math.isnan(x)]
    n = len(v)
    if n < 2:
        return float("nan")
    m = sum(v) / n
    return math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))


def _mean(vals):
    v = [x for x in vals if not math.isnan(x)]
    if not v:
        return float("nan")
    return sum(v) / len(v)


def _mean_sd(vals):
    return _mean(vals), _stdev(vals)


def _expected_runs():
    return len(DATASETS) * len(ROOTS) * len(SIGMAS)


def _g(x, dash="--"):
    """Format a float for a table cell, dash for nan."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return dash
    return f"{x:.4g}"


# ── per-run metrics ───────────────────────────────────────────────────────────
def _rate_file_name(dataset, root, sigma):
    return f"{dataset}_{root}_s{_fmt_sigma(sigma)}.rates.txt"


def _read_nll(rate_path: Path):
    """Read NLL from the JSON sidecar if present (ln units, falling back to log2).

    Returns (nll_value, units) or (nan, None).  The sweep writes both a log2 and
    an ln NLL; we prefer the natural-log NLL for comparability, but report
    whichever is available.
    """
    sidecar = rate_path.with_suffix(rate_path.suffix + ".json")
    if not sidecar.exists():
        return float("nan"), None
    try:
        with open(sidecar) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return float("nan"), None
    for key, units in (
        ("negative_log_likelihood_ln", "ln"),
        ("negative_log_likelihood_log2", "log2"),
        ("negative_log_likelihood", "?"),
        ("nll", "?"),
    ):
        if key in payload:
            try:
                return float(payload[key]), units
            except (TypeError, ValueError):
                pass
    return float("nan"), None


def _sidecar_sigma(rate_path: Path):
    """Sigma recorded in the sidecar (authoritative); None if unavailable."""
    sidecar = rate_path.with_suffix(rate_path.suffix + ".json")
    if not sidecar.exists():
        return None
    try:
        with open(sidecar) as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    prior = payload.get("prior") or {}
    val = prior.get("brownian_sigma", payload.get("brownian_sigma"))
    try:
        return None if val is None else float(val)
    except (TypeError, ValueError):
        return None


def _alerax_paths(data_dir: Path, root: str):
    """(model_parameters.txt, species tree) for a root in the AleRax reference."""
    mp = (data_dir / "reconciliation models" / "branch wise" / root /
          "model_parameters" / "model_parameters.txt")
    tree = data_dir / "rooted_phylogeny" / root
    return mp, tree


def compute_run_metrics(rate_path: Path, dataset: str, root: str, sigma: float,
                        data_dir: Path, alerax_cache: dict):
    """All per-run metrics for one rate file.  Returns a dict (or None if empty)."""
    if not rate_path.exists() or rate_path.stat().st_size == 0:
        return None
    rates = parse_rates(rate_path, skip_header=True)
    if not rates:
        return None

    rec = {
        "dataset": dataset,
        "root": root,
        "sigma": sigma,
        "n_nodes": len(rates),
        "path": str(rate_path),
    }
    nll, nll_units = _read_nll(rate_path)
    rec["NLL"] = nll
    rec["NLL_units"] = nll_units

    # Per-axis dispersion / median / floor / runaway across ALL branches.
    for name, idx in AXES:
        col = [v[idx] for v in rates.values()]
        logs = [log10c(x) for x in col]
        rec[f"disp_{name}"] = _stdev(logs)
        rec[f"median_{name}"] = _median(col)
        n = len(col)
        rec[f"frac_floor_{name}"] = sum(1 for x in col if x <= FLOOR_THRESH) / n
        rec[f"frac_hi_{name}"] = sum(1 for x in col if x > HI_THRESH) / n

    # AleRax leaf-branch match -- only meaningful for the treedists dataset.
    for name, _ in AXES:
        rec[f"pearson_{name}"] = float("nan")
        rec[f"spearman_{name}"] = float("nan")
        rec[f"ratio_{name}"] = float("nan")
    rec["n_aligned"] = 0
    rec["alerax_ok"] = False

    if dataset == ALERAX_DATASET:
        ref = alerax_cache.get(root)
        if ref is None:
            mp_path, tree_path = _alerax_paths(data_dir, root)
            ref_rates, leaves = None, None
            if mp_path.exists() and tree_path.exists():
                try:
                    ref_rates = parse_rates(mp_path, skip_header=False)
                    leaves = leaf_names_from_newick(tree_path)
                except OSError:
                    ref_rates, leaves = None, None
            ref = (ref_rates, leaves)
            alerax_cache[root] = ref
        ref_rates, leaves = ref
        if ref_rates and leaves:
            common = sorted(n for n in leaves if n in rates and n in ref_rates)
            rec["n_aligned"] = len(common)
            if common:
                rec["alerax_ok"] = True
                for name, idx in AXES:
                    gx = [log10c(rates[n][idx]) for n in common]
                    ax = [log10c(ref_rates[n][idx]) for n in common]
                    rec[f"pearson_{name}"] = pearson(gx, ax)
                    rec[f"spearman_{name}"] = spearman(gx, ax)
                    ratios = sorted(10 ** (x - y) for x, y in zip(gx, ax))
                    rec[f"ratio_{name}"] = ratios[len(ratios) // 2]
    return rec


# ── scanning the sweep ────────────────────────────────────────────────────────
def scan_sweep(sweep_dir: Path, data_dir: Path):
    """Return (records, present, expected, alerax_available)."""
    alerax_cache: dict = {}
    records = []
    present = 0
    expected = _expected_runs()
    for dataset in DATASETS:
        for root in ROOTS:
            for sigma in SIGMAS:
                path = sweep_dir / _rate_file_name(dataset, root, sigma)
                rec = compute_run_metrics(path, dataset, root, sigma,
                                          data_dir, alerax_cache)
                if rec is not None:
                    present += 1
                    records.append(rec)
    alerax_available = any(v[0] for v in alerax_cache.values())
    return records, present, expected, alerax_available


# ── output: per-run table ─────────────────────────────────────────────────────
PER_RUN_CSV_COLS = [
    "dataset", "root", "sigma", "NLL", "NLL_units",
    "disp_D", "disp_L", "disp_T",
    "median_D", "median_L", "median_T",
    "frac_floor_L", "frac_hi_L",
    "pearson_D", "pearson_L", "pearson_T",
    "spearman_D", "spearman_L", "spearman_T",
    "ratio_D", "ratio_L", "ratio_T",
    "n_nodes", "n_aligned", "path",
]


def write_csv(records, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PER_RUN_CSV_COLS)
        for r in _sorted_records(records):
            row = []
            for c in PER_RUN_CSV_COLS:
                v = r.get(c, "")
                if isinstance(v, float):
                    v = "" if math.isnan(v) else repr(v)
                row.append(v)
            w.writerow(row)


def _sorted_records(records):
    di = {d: i for i, d in enumerate(DATASETS)}
    ri = {r: i for i, r in enumerate(ROOTS)}
    return sorted(records, key=lambda r: (di.get(r["dataset"], 99),
                                          ri.get(r["root"], 99),
                                          r["sigma"]))


def print_per_run_table(records, emit):
    emit("=" * 110)
    emit("PER-RUN METRICS")
    emit("=" * 110)
    hdr = (f"{'dataset':<10} {'root':<14} {'sigma':>5} {'NLL':>12} "
           f"{'dispD':>7} {'dispL':>7} {'dispT':>7} "
           f"{'medD':>8} {'medL':>8} {'medT':>8} "
           f"{'flrL':>6} {'hiL':>6} "
           f"{'pearD':>7} {'pearL':>7} {'pearT':>7}")
    emit(hdr)
    emit("-" * len(hdr))
    for r in _sorted_records(records):
        emit(
            f"{r['dataset']:<10} {r['root']:<14} {r['sigma']:>5g} "
            f"{_g(r.get('NLL')):>12} "
            f"{_g(r['disp_D']):>7} {_g(r['disp_L']):>7} {_g(r['disp_T']):>7} "
            f"{_g(r['median_D']):>8} {_g(r['median_L']):>8} {_g(r['median_T']):>8} "
            f"{_g(r['frac_floor_L']):>6} {_g(r['frac_hi_L']):>6} "
            f"{_g(r['pearson_D']):>7} {_g(r['pearson_L']):>7} {_g(r['pearson_T']):>7}"
        )
    emit("")


# ── output: stiffness aggregation ─────────────────────────────────────────────
def aggregate_by_sigma(records):
    """{dataset: {sigma: {metric: (mean, sd), 'n': k}}} aggregated across roots."""
    agg = {}
    metrics = ["disp_D", "disp_L", "disp_T",
               "pearson_D", "pearson_L", "pearson_T",
               "median_D", "median_L", "median_T",
               "frac_hi_L", "frac_floor_L", "NLL"]
    for dataset in DATASETS:
        agg[dataset] = {}
        for sigma in SIGMAS:
            rs = [r for r in records
                  if r["dataset"] == dataset and r["sigma"] == sigma]
            if not rs:
                continue
            cell = {"n": len(rs)}
            for m in metrics:
                cell[m] = _mean_sd([r.get(m, float("nan")) for r in rs])
            agg[dataset][sigma] = cell
    return agg


def _ms(cell, metric):
    return cell.get(metric, (float("nan"), float("nan")))


def _ms_str(cell, metric):
    m, s = _ms(cell, metric)
    if math.isnan(m):
        return "    --      "
    if math.isnan(s):
        return f"{m:>7.3g}        "
    return f"{m:>7.3g}±{s:<5.2g}"


def print_stiffness_tables(agg, emit, alerax_available):
    emit("=" * 110)
    emit("STIFFNESS ANALYSIS  —  aggregated across roots (mean ± sd over the "
         "available roots at each sigma)")
    emit("=" * 110)
    for dataset in DATASETS:
        cells = agg.get(dataset, {})
        sigmas = [s for s in SIGMAS if s in cells]
        if not sigmas:
            emit(f"\n[{dataset}] no runs present.")
            continue
        is_alerax = (dataset == ALERAX_DATASET) and alerax_available
        emit("")
        emit(f"[{dataset}]   (n roots per sigma shown as 'n=')")
        cols = (f"  {'sigma':>5} {'n':>3} | "
                f"{'disp_D':>13} {'disp_L':>13} {'disp_T':>13} | "
                f"{'median_L':>13} {'frac_hi_L':>13}")
        if is_alerax:
            cols += (f" | {'pear_D':>13} {'pear_L':>13} {'pear_T':>13}")
        emit(cols)
        emit("  " + "-" * (len(cols) - 2))
        for sigma in sigmas:
            cell = cells[sigma]
            line = (f"  {sigma:>5g} {cell['n']:>3d} | "
                    f"{_ms_str(cell, 'disp_D')} {_ms_str(cell, 'disp_L')} "
                    f"{_ms_str(cell, 'disp_T')} | "
                    f"{_ms_str(cell, 'median_L')} {_ms_str(cell, 'frac_hi_L')}")
            if is_alerax:
                line += (f" | {_ms_str(cell, 'pearson_D')} "
                         f"{_ms_str(cell, 'pearson_L')} "
                         f"{_ms_str(cell, 'pearson_T')}")
            emit(line)
    emit("")


# ── output: written stiffness summary ─────────────────────────────────────────
def _trend_arrow(agg, dataset, metric):
    """Compare metric mean at smallest vs largest available sigma.

    Returns (lo_sigma, lo_val, hi_sigma, hi_val, arrow) or None.
    """
    cells = agg.get(dataset, {})
    sigmas = [s for s in SIGMAS if s in cells and not math.isnan(_ms(cells[s], metric)[0])]
    if len(sigmas) < 2:
        return None
    lo, hi = sigmas[0], sigmas[-1]
    lo_v = _ms(cells[lo], metric)[0]
    hi_v = _ms(cells[hi], metric)[0]
    if math.isnan(lo_v) or math.isnan(hi_v):
        return None
    if hi_v > lo_v * 1.02:
        arrow = "increases"
    elif hi_v < lo_v * 0.98:
        arrow = "decreases"
    else:
        arrow = "≈ flat"
    return lo, lo_v, hi, hi_v, arrow


def _peak_sigma(agg, dataset, metric):
    """sigma maximizing the mean of metric (for the AleRax-Pearson peak)."""
    cells = agg.get(dataset, {})
    best = None
    for s in SIGMAS:
        if s not in cells:
            continue
        m = _ms(cells[s], metric)[0]
        if math.isnan(m):
            continue
        if best is None or m > best[1]:
            best = (s, m)
    return best


def print_written_summary(agg, emit, alerax_available):
    emit("=" * 110)
    emit("STIFFNESS EFFECT — written summary")
    emit("=" * 110)
    emit("Recall: small sigma = STIFF prior (rates pulled toward neighbours, "
         "smoothed); large sigma = LOOSE (each branch free).")
    emit("")

    for dataset in DATASETS:
        if dataset not in agg or not agg[dataset]:
            continue
        emit(f"[{dataset}]")
        # (a) dispersion per axis
        for name, _ in AXES:
            t = _trend_arrow(agg, dataset, f"disp_{name}")
            if t:
                lo, lo_v, hi, hi_v, arrow = t
                emit(f"  - dispersion {name}: σ {lo:g}→{hi:g}  "
                     f"{lo_v:.3g} → {hi_v:.3g}  ({arrow} with σ)")
        # (c) loss identifiability
        t = _trend_arrow(agg, dataset, "frac_hi_L")
        if t:
            lo, lo_v, hi, hi_v, arrow = t
            emit(f"  - loss runaway frac_hi_L: σ {lo:g}→{hi:g}  "
                 f"{lo_v:.3g} → {hi_v:.3g}  ({arrow} with σ)")
        t = _trend_arrow(agg, dataset, "frac_floor_L")
        if t:
            lo, lo_v, hi, hi_v, arrow = t
            emit(f"  - loss floored frac_floor_L: σ {lo:g}→{hi:g}  "
                 f"{lo_v:.3g} → {hi_v:.3g}  ({arrow} with σ)")
        # (b) AleRax match peak
        if dataset == ALERAX_DATASET and alerax_available:
            for name, _ in AXES:
                pk = _peak_sigma(agg, dataset, f"pearson_{name}")
                t = _trend_arrow(agg, dataset, f"pearson_{name}")
                if pk:
                    note = ""
                    if t:
                        lo, lo_v, hi, hi_v, _arrow = t
                        # is the peak at an interior sigma (not an endpoint)?
                        if pk[0] not in (lo, hi):
                            note = "  [interior peak → intermediate stiffness best]"
                        elif pk[0] == lo:
                            note = "  [best at stiffest σ → AleRax ≈ stiff/clade-grouped]"
                    emit(f"  - AleRax Pearson {name}: peaks at σ={pk[0]:g} "
                         f"(r={pk[1]:.3f}){note}")
        emit("")

    # axis-level takeaway: which axis moves most with sigma (use treedists if
    # present, else first dataset with data).
    ds = ALERAX_DATASET if (ALERAX_DATASET in agg and agg[ALERAX_DATASET]) else None
    if ds is None:
        for d in DATASETS:
            if agg.get(d):
                ds = d
                break
    if ds is not None:
        spread = {}
        for name, _ in AXES:
            t = _trend_arrow(agg, ds, f"disp_{name}")
            if t:
                lo, lo_v, hi, hi_v, _a = t
                # relative change in dispersion across the sigma range
                spread[name] = abs(hi_v - lo_v) / (abs(lo_v) + 1e-9)
        emit("Cross-axis takeaway:")
        if spread:
            most = max(spread, key=spread.get)
            ranked = ", ".join(f"{k} (Δdisp/disp≈{v:.2f})"
                               for k, v in sorted(spread.items(),
                                                  key=lambda kv: -kv[1]))
            emit(f"  Dispersion sensitivity to σ by axis (most→least): {ranked}")
            emit(f"  → LOSS (L) is expected to be the axis most sensitive to σ; "
                 f"observed most-sensitive axis = {most}.")
            emit("  Loss rates are the least independently identifiable (loss "
                 "leaves few observable traces per branch), so the prior "
                 "stiffness σ dominates them: stiff σ smooths L hardest, loose "
                 "σ lets L disperse / run away (frac_hi_L↑) the most.")
    emit("")


# ── plots (optional) ──────────────────────────────────────────────────────────
def make_plots(agg, out_dir: Path, emit, alerax_available):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        emit(f"[plots] matplotlib unavailable ({e}); skipping plots.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # 1) dispersion vs sigma, one panel per axis, a line per dataset.
    fig, axesp = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax_i, (name, _) in enumerate(AXES):
        ax = axesp[ax_i]
        for dataset in DATASETS:
            cells = agg.get(dataset, {})
            xs = [s for s in SIGMAS if s in cells]
            if not xs:
                continue
            ys = [_ms(cells[s], f"disp_{name}")[0] for s in xs]
            es = [_ms(cells[s], f"disp_{name}")[1] for s in xs]
            es = [0.0 if math.isnan(e) else e for e in es]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=dataset)
        ax.set_title(f"dispersion {name}  (sd of log10 rate)")
        ax.set_xlabel("prior sigma (log2)")
        ax.set_xscale("log")
        if ax_i == 0:
            ax.set_ylabel("per-branch dispersion")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Per-branch dispersion vs prior stiffness σ "
                 "(stiff small-σ → low dispersion)")
    fig.tight_layout()
    p1 = out_dir / "dispersion_vs_sigma.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    written.append(p1)

    # 2) AleRax Pearson vs sigma (treedists), one line per axis.
    if alerax_available and agg.get(ALERAX_DATASET):
        cells = agg[ALERAX_DATASET]
        xs = [s for s in SIGMAS if s in cells]
        if xs:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
            for name, _ in AXES:
                ys = [_ms(cells[s], f"pearson_{name}")[0] for s in xs]
                es = [_ms(cells[s], f"pearson_{name}")[1] for s in xs]
                es = [0.0 if math.isnan(e) else e for e in es]
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=name)
            ax.set_title("AleRax leaf-branch match (treedists)\n"
                         "Pearson(log10) vs prior σ — expect peak at stiff/"
                         "intermediate σ")
            ax.set_xlabel("prior sigma (log2)")
            ax.set_ylabel("Pearson r (log10 rate, 60 leaf branches)")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(title="axis")
            fig.tight_layout()
            p2 = out_dir / "alerax_pearson_vs_sigma.png"
            fig.savefig(p2, dpi=130)
            plt.close(fig)
            written.append(p2)

    for p in written:
        emit(f"[plots] wrote {p}")
    return written


# ── main ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True,
                    help="dir of <dataset>_<root>_s<sigma>.rates.txt files")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="Williams data dir holding the AleRax reference + trees")
    ap.add_argument("--out-csv", default=None,
                    help="write the tidy per-run table here (CSV)")
    ap.add_argument("--out-dir", default=None,
                    help="write PNG plots here (skipped if matplotlib absent)")
    args = ap.parse_args(argv)

    sweep_dir = Path(args.sweep_dir)
    data_dir = Path(args.data_dir)
    if not sweep_dir.is_dir():
        print(f"[!] sweep dir not found: {sweep_dir}", file=sys.stderr)
        return 2

    out_lines = []

    def emit(s=""):
        out_lines.append(s)
        print(s)

    records, present, expected, alerax_available = scan_sweep(sweep_dir, data_dir)

    emit("=" * 110)
    emit("BROWNIAN-PRIOR STIFFNESS SWEEP — analysis")
    emit("=" * 110)
    emit(f"sweep dir : {sweep_dir}")
    emit(f"data dir  : {data_dir}")
    emit(f"runs present: {present} / {expected} expected "
         f"({len(DATASETS)} datasets × {len(ROOTS)} roots × {len(SIGMAS)} sigmas)")
    if present < expected:
        emit(f"  [note] {expected - present} run(s) missing/empty "
             f"(sweep may still be in progress) — analyzing what is present.")
    # how many per dataset / per sigma
    for dataset in DATASETS:
        n = sum(1 for r in records if r["dataset"] == dataset)
        emit(f"  {dataset:<10}: {n} run(s)")
    if not alerax_available:
        emit("  [note] AleRax reference not found under --data-dir "
             "(or no leaf overlap) — Pearson/Spearman columns will be N/A. "
             "This is expected off-cluster.")
    n_with_nll = sum(1 for r in records if not math.isnan(r.get("NLL", float("nan"))))
    emit(f"  NLL sidecars found for {n_with_nll}/{present} present run(s).")
    emit("")

    if not records:
        emit("[!] no non-empty rate files found — nothing to analyze.")
        if args.out_csv:
            write_csv([], Path(args.out_csv))
        return 1

    print_per_run_table(records, emit)

    agg = aggregate_by_sigma(records)
    print_stiffness_tables(agg, emit, alerax_available)
    print_written_summary(agg, emit, alerax_available)

    if args.out_csv:
        write_csv(records, Path(args.out_csv))
        emit(f"[csv] per-run table -> {args.out_csv}")

    if args.out_dir:
        make_plots(agg, Path(args.out_dir), emit, alerax_available)

    return 0


if __name__ == "__main__":
    sys.exit(main())
