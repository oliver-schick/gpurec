"""Collect the paperML campaign: free-fit-all (per-branch DTL + free O) warm-started
from the PAPER's 4_Ancestral rates, across all 15 roots. Reports, per root:
  - logL@paper : data logL AT the paper rates, unfit (eval_paperinit_<R>.json total)
  - logL_ML    : data logL after free-fitting ALL rates (fitML_<R>.rates.txt.json)
  - ML gain    : logL_ML - logL@paper  (how far below ML the paper rates sit)
  - AU         : approximately-unbiased rooting support at the fitted rates
Answers (1) are the paper rates ML? [Eury ML gain ~ 0 => yes] and (2) is Eury the ML
root? [AU(Eury)=1]. Mirrors au_fullbasin's au_rank bootstrap on the fitted per-fam logL.
"""
import sys, json, glob, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_bigtree_at_alerax import au_rank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    D = args.dir
    paper = {}                                  # logL at paper rates (unfit)
    for f in glob.glob(D + "/eval_paperinit_*.json"):
        d = json.load(open(f)); paper[d["root"]] = d["total"]
    ml = {}                                      # data logL after free-fit-all
    for f in glob.glob(D + "/fitML_*.rates.txt.json"):
        R = Path(f).name[len("fitML_"):-len(".rates.txt.json")]
        ml[R] = json.load(open(f)).get("data_log_likelihood_ln")
    per_by_root = {}                             # fitted per-fam logL -> AU
    for f in glob.glob(D + "/eval_fitML_*.json"):
        d = json.load(open(f)); per_by_root[d["root"]] = dict(zip(d["names"], d["logL_ln"]))
    roots = list(per_by_root)
    res, n = au_rank(per_by_root, roots) if roots else ([], 0)
    au = {r: a for r, _sl, _bp, a, *_ in res}
    best_ml = max([v for v in ml.values() if v is not None], default=None)

    print(f"=== paperML: free-fit-ALL from PAPER 4_Ancestral rates  (n={n} fam, {len(roots)} roots) ===")
    print(f"{'root':16s} {'logL@paper':>15s} {'logL_ML(fit)':>15s} {'ML gain':>10s} "
          f"{'Δ vs bestML':>12s} {'AU':>7s}")
    order = sorted(ml, key=lambda r: -(ml[r] if ml[r] is not None else -9e18))
    for r in order:
        pg, mg = paper.get(r), ml.get(r)
        gain = (mg - pg) if (pg is not None and mg is not None) else None
        dml = (mg - best_ml) if (mg is not None and best_ml is not None) else None
        print(f"{r:16s} {pg if pg is not None else 0:15.0f} {mg if mg is not None else 0:15.0f} "
              f"{gain if gain is not None else 0:10.0f} {dml if dml is not None else 0:12.0f} "
              f"{au.get(r, float('nan')):7.3f}")
    json.dump({"paper_logL": paper, "ml_logL": ml, "au": au,
               "ranking_by_ml": order, "au_ranking": [r for r, *_ in res], "n": n},
              open(D + "/paperML_summary.json", "w"), indent=1)
    if order:
        top = order[0]
        print(f"\nML root (best fitted logL) = {top}   AU={au.get(top, float('nan')):.3f}")
        if "Eury" in paper and "Eury" in ml:
            print(f"Eury: paper rates sit {ml['Eury'] - paper['Eury']:+.0f} nats below the free-fit ML "
                  f"(gain from fitting all rates).")


if __name__ == "__main__":
    main()
