#!/usr/bin/env python
"""Rooting comparison: gpurec prior-free data logL per root vs AleRax sum per_fam logL.

The whole point of running differently-rooted species trees is the rooting test:
the best-supported root maximises the marginal likelihood. AleRax's per-root signal
is Sum_f per_fam_likelihoods.txt; gpurec's is the prior-free DATA log-likelihood at
the optimum (data_log_likelihood_ln in the rooting_<root>.rates.txt.json sidecar).

Absolute scales differ (normalisation / origination convention), so we compare the
RANKING and the relative pattern across roots.
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


def alerax_ll(rm: Path, root: str):
    f = rm / "branch wise" / root / "per_fam_likelihoods.txt"
    if not f.exists():
        return None, 0
    s, n = 0.0, 0
    for line in f.read_text().splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                s += float(p[1]); n += 1
            except ValueError:
                pass
    return s, n


def gpurec_ll(sweep: Path, root: str):
    j = sweep / f"rooting_{root}.rates.txt.json"
    if not j.exists():
        return None, None
    d = json.loads(j.read_text())
    return d.get("data_log_likelihood_ln"), d.get("n_families_kept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--data-dir", default="/work/SzollosiU/gergely-szollosi/williams_run/"
                                          "data/3_Reconciliation/Williams_et_al_2017")
    args = ap.parse_args()
    sweep = Path(args.sweep_dir)
    rm = Path(args.data_dir) / "reconciliation models"

    rows = []
    for r in ROOTS:
        g, gn = gpurec_ll(sweep, r)
        al, aln = alerax_ll(rm, r)
        rows.append([r, g, gn, al, aln])

    have = [x for x in rows if x[1] is not None]
    print(f"rooting runs present: {len(have)}/10")
    if not have:
        print("  (no rooting_*.json yet)"); return 0

    gbest = max(have, key=lambda x: x[1])[0]
    abest = max((x for x in rows if x[3] is not None), key=lambda x: x[3])[0]

    print(f"\n  {'root':<16}{'gpurec ΔdlogL':>15}{'AleRax ΔlogL':>15}   (Δ from each method's best)")
    g_top = max(x[1] for x in have)
    a_top = max(x[3] for x in rows if x[3] is not None)
    for r, g, gn, al, aln in sorted(rows, key=lambda x: (x[1] if x[1] is not None else -1e18),
                                    reverse=True):
        gd = f"{g - g_top:>14.1f}" if g is not None else f"{'--':>14}"
        ad = f"{al - a_top:>14.1f}" if al is not None else f"{'--':>14}"
        mark = ""
        if r == gbest:
            mark += " <gpurec-best"
        if r == abest:
            mark += " <AleRax-best"
        print(f"  {r:<16}{gd} {ad}{mark}")

    common = [x for x in have if x[3] is not None]
    if len(common) >= 3:
        gv = [x[1] for x in common]
        av = [x[3] for x in common]
        print(f"\n  per-root agreement (n={len(common)}): "
              f"Pearson={pearson(gv, av):.3f}  Spearman={spearman(gv, av):.3f}")
    print(f"  best root:  gpurec = {gbest}   AleRax = {abest}   "
          f"-> {'AGREE' if gbest == abest else 'DIFFER'}")
    print("\n  note: gpurec = prior-free data logL (no-fm, sigma=1, min-species=1 => all "
          "families); AleRax = Sum per_fam (5446). Absolute scales differ; ranking is the "
          "rooting signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
