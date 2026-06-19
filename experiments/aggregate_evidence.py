"""Aggregate per-(root, model) Laplace evidence JSONs into:
  (1) a Bayesian ROOTING posterior P(root | data) = softmax_root(log Z) per model,
  (2) a cross-MODEL comparison table (log Z, effective #params, k) per root.

Usage:
    python experiments/aggregate_evidence.py results/*.evidence.json
"""
from __future__ import annotations
import json
import math
import sys
from collections import defaultdict


def _softmax(d):
    m = max(d.values())
    ex = {k: math.exp(v - m) for k, v in d.items()}
    z = sum(ex.values())
    return {k: ex[k] / z for k in ex}


def main(paths):
    rows = []
    for p in paths:
        try:
            rows.append(json.loads(open(p).read()))
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {p}: {e}")
    if not rows:
        print("no evidence files")
        return 1

    by_model = defaultdict(dict)   # model -> {root: row}
    for r in rows:
        by_model[r["model"]][r["root"]] = r

    # family-set sanity (the CCP offset cancels only if F identical)
    Fs = {(r["model"], r["root"]): r["F_families"] for r in rows}
    print("=" * 84)
    print("LAPLACE EVIDENCE  (ridge Gaussian prior; log Z, p_eff in nats)")
    print("=" * 84)
    distinctF = sorted(set(Fs.values()))
    print(f"family counts across runs: {distinctF}  "
          f"{'(OK identical)' if len(distinctF) == 1 else '(!! DIFFER -> offset does NOT cancel)'}")
    maxresid = max(r.get("validation_resid_rel", 0.0) for r in rows)
    print(f"max validation rel-resid (sum_f g_f vs batched grad): {maxresid:.2e} "
          f"{'(OK)' if maxresid < 1e-3 else '(!! per-family scores suspect)'}")

    for model, d in sorted(by_model.items()):
        print()
        print(f"### MODEL = {model}   (k_eff per root)")
        keffs = sorted(set(r["k_eff"] for r in d.values()))
        if len(keffs) > 1:
            print(f"  [!] k_eff VARIES across roots {keffs}: the (k/2)ln(tau) and "
                  "logdet dimensionality do not fully cancel in the softmax -- "
                  "rooting log_Z differences are only approximately comparable.")
        post = _softmax({root: r["log_Z"] for root, r in d.items()})
        best = max(d, key=lambda root: d[root]["log_Z"])
        bestlz = d[best]["log_Z"]
        print(f"  {'root':16s} {'log_Z':>14s} {'dlogZ':>10s} {'P(root|data)':>13s} "
              f"{'k_eff':>6s} {'p_eff':>8s}")
        for root in sorted(d, key=lambda x: -d[x]["log_Z"]):
            r = d[root]
            star = " <BEST" if root == best else ""
            print(f"  {root:16s} {r['log_Z']:>14.1f} {r['log_Z']-bestlz:>10.1f} "
                  f"{post[root]:>13.4f} {r['k_eff']:>6d} {r['p_eff']:>8.1f}{star}")
        # Bayes factor best vs Cluster2 (the artefact root), if present
        if "Cluster2" in d and best != "Cluster2":
            lnbf = bestlz - d["Cluster2"]["log_Z"]
            print(f"  ln BF ({best} vs Cluster2) = {lnbf:.1f}  "
                  f"(= {lnbf/math.log(2):.1f} bits; |.|>5 ~ decisive)")

    # cross-model table per root
    roots = sorted({r["root"] for r in rows})
    models = sorted(by_model)
    print()
    print("### CROSS-MODEL evidence per root (log_Z ; p_eff/k_eff)")
    hdr = f"  {'root':16s}" + "".join(f"{m:>24s}" for m in models)
    print(hdr)
    for root in roots:
        cells = []
        for m in models:
            r = by_model[m].get(root)
            if r:
                cells.append(f"{r['log_Z']:>11.1f}|{r['p_eff']:>5.1f}/{r['k_eff']:<4d}")
            else:
                cells.append(f"{'--':>24s}")
        print(f"  {root:16s}" + "".join(f"{c:>24s}" for c in cells))

    # model ranking at a fixed root (best evidence model)
    print()
    print("### MODEL RANKING by evidence (per root): which model wins?")
    for root in roots:
        avail = {m: by_model[m][root] for m in models if root in by_model[m]}
        if len(avail) < 2:
            continue
        ranked = sorted(avail, key=lambda m: -avail[m]["log_Z"])
        s = "  ".join(f"{m}({avail[m]['log_Z']:.0f}, peff={avail[m]['p_eff']:.0f})"
                      for m in ranked)
        print(f"  {root:16s} {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
