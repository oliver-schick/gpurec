#!/usr/bin/env python
"""Aggregate a 2x2 rooting sweep: fm {off,e-only} x origination {uniform,optimize}.

The rooting test picks the root that maximises the marginal likelihood. gpurec's
per-root signal is the prior-free DATA log-likelihood (data_log_likelihood_ln in
each rooting run's sidecar); AleRax's is Sum_f per_fam_likelihoods.txt. The two
methods disagreed under gpurec's original setup (no-fm + uniform origination):
gpurec favoured Kor by a huge margin while AleRax favours Eury. This sweep tests
whether the two known confounds -- fraction-missing and optimized origination --
explain the disagreement, by running every root under each of the 4 conditions:

    offU = no-fm,  uniform origination      (baseline; reused from the first run)
    eU   = e-only fm, uniform origination
    offO = no-fm,  optimize origination (decoupled root)
    eO   = e-only fm, optimize origination (decoupled root)   <- fully corrected

For each condition it prints the per-root data-logL ranking, the best root, the
rank/linear correlation with AleRax, where Kor and Eury land, and whether one
root pathologically dominates (the artifact signature).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_williams_alerax import pearson, spearman  # noqa: E402

ROOTS = ["Alti", "AMD", "Asgard", "Cluster2", "DPANN", "Eury",
         "HaloThermoplas", "Kor", "TAC", "TackA"]

# condition tag -> (human label)
CONDS = [
    ("offU", "no-fm, uniform orig (baseline)"),
    ("eU", "e-only fm, uniform orig"),
    ("offO", "no-fm, optimize orig"),
    ("eO", "e-only fm, optimize orig (corrected)"),
]


def alerax_ll(rm: Path, root: str):
    f = rm / "branch wise" / root / "per_fam_likelihoods.txt"
    if not f.exists():
        return None
    s = 0.0
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                s += float(p[1])
            except ValueError:
                pass
    return s


def gpurec_ll(path: Path):
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d.get("data_log_likelihood_ln")


def cond_file(tag, root, sweep_dir, baseline_dir):
    if tag == "offU":
        return baseline_dir / f"rooting_{root}.rates.txt.json"
    return sweep_dir / f"rootsweep_{tag}_{root}.rates.txt.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True,
                    help="dir with rootsweep_<tag>_<root>.rates.txt[.json]")
    ap.add_argument("--baseline-dir", default=None,
                    help="dir with the offU baseline rooting_<root>.rates.txt.json "
                         "(default: <sweep-dir>/../sweep)")
    ap.add_argument("--data-dir", default="/work/SzollosiU/gergely-szollosi/williams_run/"
                                          "data/3_Reconciliation/Williams_et_al_2017")
    args = ap.parse_args()
    sweep = Path(args.sweep_dir)
    baseline = Path(args.baseline_dir) if args.baseline_dir else sweep.parent / "sweep"
    rm = Path(args.data_dir) / "reconciliation models"

    alerax = {r: alerax_ll(rm, r) for r in ROOTS}
    a_have = {r: v for r, v in alerax.items() if v is not None}
    a_best = max(a_have, key=a_have.get) if a_have else None

    print("=" * 92)
    print("ROOTING SWEEP — 2x2 (fraction-missing x origination), per-root data logL")
    print("=" * 92)
    if a_best:
        print(f"AleRax best root: {a_best}   (Sum per_fam logL; "
              f"{len(a_have)}/10 roots present)\n")

    summary = []
    for tag, label in CONDS:
        vals = {}
        for r in ROOTS:
            v = gpurec_ll(cond_file(tag, r, sweep, baseline))
            if v is not None:
                vals[r] = v
        if not vals:
            print(f"[{tag}] {label}: no runs present yet.\n")
            continue
        g_best = max(vals, key=vals.get)
        top = max(vals.values())
        ranked = sorted(vals, key=vals.get, reverse=True)
        # gap from best to 2nd (artifact signature: one root dominating)
        gap = (vals[ranked[0]] - vals[ranked[1]]) if len(ranked) >= 2 else float("nan")

        common = [r for r in ROOTS if r in vals and r in a_have]
        rP = pearson([vals[r] for r in common], [a_have[r] for r in common]) if len(common) >= 3 else float("nan")
        rS = spearman([vals[r] for r in common], [a_have[r] for r in common]) if len(common) >= 3 else float("nan")

        kor_rank = (ranked.index("Kor") + 1) if "Kor" in ranked else None
        eury_rank = (ranked.index("Eury") + 1) if "Eury" in ranked else None

        print(f"[{tag}]  {label}   ({len(vals)}/10 roots)")
        print(f"  ranking (Δ data-logL from this condition's best):")
        for r in ranked:
            mk = ""
            if r == g_best:
                mk += " <best"
            if r == a_best:
                mk += " <AleRax-best"
            print(f"    {r:<16}{vals[r] - top:>13.1f}{mk}")
        agree = "AGREE" if g_best == a_best else "DIFFER"
        print(f"  best={g_best} (AleRax={a_best} -> {agree})  "
              f"Pearson={rP:.3f} Spearman={rS:.3f}  "
              f"best->2nd gap={gap:.1f}  Kor rank={kor_rank}  Eury rank={eury_rank}")
        if gap > 200:
            print(f"  [!] one root dominates by {gap:.0f} ln-units -> likely "
                  f"optimisation artifact, not a clean rooting signal")
        print()
        summary.append((tag, label, g_best, agree, rP, rS, gap, kor_rank, eury_rank))

    if summary:
        print("=" * 92)
        print("SUMMARY  (does correcting the confounds recover AleRax's Eury?)")
        print("=" * 92)
        print(f"  {'cond':<6}{'best':<16}{'vs AleRax':<9}{'Pearson':>9}{'Spear':>8}"
              f"{'gap':>9}{'Kor#':>6}{'Eury#':>7}")
        for tag, label, gb, agree, rP, rS, gap, kr, er in summary:
            print(f"  {tag:<6}{gb:<16}{agree:<9}{rP:>9.3f}{rS:>8.3f}"
                  f"{gap:>9.1f}{str(kr):>6}{str(er):>7}")
        print("\n  Read: if 'best' moves to Eury (or Pearson turns strongly +) as fm/"
              "origination are corrected, the original Kor result was a confound/"
              "artifact; if Kor persists as best with a huge gap, it is a genuine "
              "(if surprising) data preference under gpurec's model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
