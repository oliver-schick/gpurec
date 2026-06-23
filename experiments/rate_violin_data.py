"""Dump per-branch D/L/T rates for the violin: BRANCHWISE (FULLbasin free per-branch)
vs SHARED (AXgrp clade-grouped fit), each branch tagged with its named clade. Tiny JSON
-> plotted on the Mac. log2-rate (theta_log2) so the spread is on a log scale.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from run_williams_branchwise import _load_species_helpers
from clade_groups import tree_clade_group_index, BIGTREE_DTL_CLADES

U = "/work/SzollosiU/gergely-szollosi/williams_run/undine"
DD = "/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study"
tree = f"{DD}/4_species_tree/Undine_C60_Euryroot_short_name.nw"

free = json.load(open(f"{U}/FULLbasin_Eury.rates.txt.json"))      # branchwise (free per-branch)
shared = json.load(open(f"{U}/AXgrp_DTL_br1_O_Eury.rates.txt.json"))  # shared (clade-grouped)
tf = torch.tensor(free["theta_log2"], dtype=torch.float64).reshape(-1, 3)
ts = torch.tensor(shared["theta_log2"], dtype=torch.float64).reshape(-1, 3)

sp = _load_species_helpers(tree); S = int(sp["S"])
gi, labels = tree_clade_group_index(sp, tree, S, clade_whitelist=BIGTREE_DTL_CLADES)


def lab(i):
    try:
        return str(labels[i])
    except Exception:
        return f"g{i}"


rows = []
for b in range(S):
    c = int(gi[b])
    rows.append(dict(clade=lab(c),
                     Df=float(tf[b, 0]), Lf=float(tf[b, 1]), Tf=float(tf[b, 2]),
                     Ds=float(ts[b, 0]), Ls=float(ts[b, 1]), Ts=float(ts[b, 2])))
out = f"{U}/rate_violin_data.json"
json.dump(rows, open(out, "w"))
print(f"wrote {len(rows)} branches, {len({r['clade'] for r in rows})} clades -> {out}")
