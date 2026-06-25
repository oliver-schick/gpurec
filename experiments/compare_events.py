"""Paper-vs-FULLbasin per-branch event comparison + genome-flow / lineage view.

Joins the .perspecies_events.tsv files written by per_node_copies.py (one per rate
set) on the species node. Transfer is a DONOR rate: `transfer` is counted at the
donor (genes leaving), `transfer_in` at the recipient (genes arriving). A branch's
genome (copies) is fed by origination + transfer_in + duplication + INHERITANCE from
its parent -- NOT by its own `transfer` (which is outflow). Triton-free.

  python experiments/compare_events.py \
     --label paper_br2=pnc_paper_DTL_br2.perspecies_events.tsv \
     --label fullbasin=pnc_FULLbasin.perspecies_events.tsv \
     [--lineage DPANN]     # trace a clade's flow up to the root
"""
from __future__ import annotations
import argparse

FIELDS = ("spec", "dup", "loss", "transfer", "transfer_in", "orig", "presence", "copies")


def _load(path):
    rows = {}
    with open(path) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            p = line.rstrip("\n").split("\t")
            d = {"name": p[ci["name"]], "label": p[ci["label"]],
                 "parent": int(p[ci["parent"]]) if "parent" in ci else -1}
            for f in FIELDS:
                d[f] = float(p[ci[f]]) if f in ci else 0.0
            rows[int(p[ci["node"]])] = d
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", default=[],
                    help="name=path.perspecies_events.tsv (repeatable; first is the reference)")
    ap.add_argument("--lineage", default=None, help="trace this clade label's flow up to the root")
    ap.add_argument("--top", type=int, default=14)
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
    print("==== per-model event totals (E[events]*n_families, summed over branches) ====")
    print(f"{'model':>12s} {'orig':>9s} {'dup':>9s} {'T_out':>10s} {'T_in':>10s} {'loss':>10s} {'copies':>11s}")
    for nm in names:
        t = {f: sum(r[f] for r in data[nm].values()) for f in
             ("orig", "dup", "transfer", "transfer_in", "loss", "copies")}
        print(f"{nm:>12s} {t['orig']:>9.0f} {t['dup']:>9.0f} {t['transfer']:>10.0f} "
              f"{t['transfer_in']:>10.0f} {t['loss']:>10.0f} {t['copies']:>11.0f}")

    # 2. genome arrival/departure budget for the labeled clades.
    #    arrivals = orig + transfer_in + dup (+ inherited, the remainder vs copies);
    #    departures from the present genome = transfer(out) + loss.
    print(f"\n==== genome flow at labeled clades (per model: orig | T_in | T_out | loss | copies) ====")
    hdr = f"{'clade':>16s}"
    for nm in names:
        hdr += f" |{nm[:9]:>9s}: O   Tin  Tout  L   cop"
    print(hdr)
    for node in nodes:
        lab = col(ref, node, "label")
        if not lab:
            continue
        line = f"{lab[:16]:>16s}"
        for nm in names:
            line += (f" | {col(nm,node,'orig'):>4.0f} {col(nm,node,'transfer_in'):>4.0f} "
                     f"{col(nm,node,'transfer'):>4.0f} {col(nm,node,'loss'):>4.0f} {col(nm,node,'copies'):>5.0f}")
        print(line)

    # 3. lineage trace: walk from the named clade up to the root, showing the flow
    #    so you can SEE whether a deep clade's genome is fed by inheritance from the
    #    backbone (deep origination descending) vs by transfer_in.
    if args.lineage:
        tgt = [n for n in nodes if col(ref, n, "label") == args.lineage]
        if not tgt:
            print(f"\n[lineage] no branch labelled {args.lineage!r}")
        else:
            node = tgt[0]
            print(f"\n==== lineage {args.lineage} -> root (orig / T_in / T_out / loss / copies per model) ====")
            chain, seen = [], set()
            while node >= 0 and node not in seen:
                seen.add(node); chain.append(node)
                par = col(ref, node, "parent")
                node = int(par) if par == par else -1
            for n in chain:
                lab = col(ref, n, "label") or col(ref, n, "name")[:16]
                line = f"  {lab[:18]:>18s}"
                for nm in names:
                    line += (f" | {col(nm,n,'orig'):>5.0f} {col(nm,n,'transfer_in'):>5.0f} "
                             f"{col(nm,n,'transfer'):>5.0f} {col(nm,n,'loss'):>5.0f} {col(nm,n,'copies'):>6.0f}")
                print(line)


if __name__ == "__main__":
    main()
