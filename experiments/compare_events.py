"""Paper-vs-FULLbasin per-branch event comparison.

Joins the .perspecies_events.tsv files written by per_node_copies.py (one per rate
set) on the species node and tabulates per-branch O/D/T/L, highlighting where the
FULLbasin free-per-branch fit inflates transfer relative to the paper's rates
(the DPANN-T pathology). Triton-free; runs anywhere.

  python experiments/compare_events.py \
     --label paper_br2=pnc_paper_DTL_br2.perspecies_events.tsv \
     --label paper_br1O=pnc_paper_DTL_br1_O.perspecies_events.tsv \
     --label fullbasin=pnc_FULLbasin.perspecies_events.tsv \
     [--fixedO pnc_fixedO.perspecies_events.tsv]
"""
from __future__ import annotations
import argparse
from pathlib import Path


def _load(path):
    rows = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            node = int(p[ci["node"]])
            rows[node] = {"name": p[ci["name"]], "label": p[ci["label"]],
                          "spec": float(p[ci["spec"]]), "dup": float(p[ci["dup"]]),
                          "loss": float(p[ci["loss"]]), "transfer": float(p[ci["transfer"]]),
                          "orig": float(p[ci["orig"]]), "presence": float(p[ci["presence"]]),
                          "copies": float(p[ci["copies"]])}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", default=[],
                    help="name=path.perspecies_events.tsv (repeatable; first is the reference)")
    ap.add_argument("--top", type=int, default=25, help="rows for the T-inflation table")
    args = ap.parse_args()
    if len(args.label) < 2:
        raise SystemExit("need >=2 --label name=path")
    names, data = [], {}
    for spec in args.label:
        nm, path = spec.split("=", 1)
        names.append(nm); data[nm] = _load(path)
    ref = names[0]
    nodes = sorted(data[ref])

    def col(nm, node, field):
        return data[nm].get(node, {}).get(field, float("nan"))

    # 1. totals
    print("==== per-model event totals (summed over branches, E[events]·n_families) ====")
    print(f"{'model':>14s} {'orig':>10s} {'dup':>10s} {'transfer':>12s} {'loss':>12s} {'spec':>12s} {'copies':>12s}")
    for nm in names:
        tot = {f: sum(r[f] for r in data[nm].values()) for f in
               ("orig", "dup", "transfer", "loss", "spec", "copies")}
        print(f"{nm:>14s} {tot['orig']:>10.1f} {tot['dup']:>10.1f} {tot['transfer']:>12.1f} "
              f"{tot['loss']:>12.1f} {tot['spec']:>12.1f} {tot['copies']:>12.1f}")

    # 2. labeled-clade rows (transfer + orig + copies across models)
    other = [n for n in names if n != ref]
    print(f"\n==== labeled-clade branches: transfer / orig / copies across models ====")
    hdr = f"{'name':>22s} {'label':>12s}"
    for nm in names:
        hdr += f" | {nm[:10]:>10s} T  O   cop"
    print(hdr)
    for node in nodes:
        lab = col(ref, node, "label")
        if not lab:
            continue
        line = f"{col(ref,node,'name')[:22]:>22s} {lab[:12]:>12s}"
        for nm in names:
            line += f" | {col(nm,node,'transfer'):>5.2f} {col(nm,node,'orig'):>4.2f} {col(nm,node,'copies'):>5.0f}"
        print(line)

    # 3. biggest transfer inflation vs the reference (last model - ref)
    last = names[-1]
    print(f"\n==== top {args.top} branches by transfer inflation ({last} - {ref}) ====")
    print(f"{'name':>22s} {'label':>12s} {'T_'+ref[:8]:>12s} {'T_'+last[:8]:>12s} {'dT':>8s} "
          f"{'O_'+ref[:6]:>9s} {'O_'+last[:6]:>9s} {'cop_'+last[:6]:>9s}")
    deltas = sorted(nodes, key=lambda n: col(last, n, "transfer") - col(ref, n, "transfer"), reverse=True)
    for node in deltas[:args.top]:
        dT = col(last, node, "transfer") - col(ref, node, "transfer")
        print(f"{col(ref,node,'name')[:22]:>22s} {col(ref,node,'label')[:12]:>12s} "
              f"{col(ref,node,'transfer'):>12.3f} {col(last,node,'transfer'):>12.3f} {dT:>8.3f} "
              f"{col(ref,node,'orig'):>9.3f} {col(last,node,'orig'):>9.3f} {col(last,node,'copies'):>9.1f}")


if __name__ == "__main__":
    main()
