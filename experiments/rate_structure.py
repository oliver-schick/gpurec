"""How clade-structured are the FREE per-branch DTL rates fit in the deep (Eury) basin?
For the FULLbasin Eury fit (full free per-branch, no grouping), assign each branch to its
smallest enclosing named clade (the paper's DTL_br2 set) and report the fraction of rate
variance EXPLAINED by clade membership (ANOVA R^2) per axis, plus within-clade spread.
R^2 ~ 1 => rates are basically clade-constant; R^2 low => real within-clade heterogeneity."""
import json, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from run_williams_branchwise import _load_species_helpers
from clade_groups import tree_clade_group_index, BIGTREE_DTL_CLADES

DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
tree = str(DD / "4_species_tree" / "Undine_C60_Euryroot_short_name.nw")
sc = json.load(open("/work/SzollosiU/gergely-szollosi/williams_run/undine/FULLbasin_Eury.rates.txt.json"))
rates = torch.tensor(sc["rates"], dtype=torch.float64)            # [S,3] linear D,L,T
logr = torch.log2(rates.clamp_min(1e-10))                         # rates vary over orders of mag -> log
sp = _load_species_helpers(tree); S = int(sp["S"])
gi, labels = tree_clade_group_index(sp, tree, S, clade_whitelist=BIGTREE_DTL_CLADES)
G = int(gi.max()) + 1
print(f"S={S} branches, {G} clade classes (paper DTL_br2 set + ROOT baseline)")
for a, nm in enumerate(["D", "L", "T"]):
    x = logr[:, a]
    tot = ((x - x.mean()) ** 2).sum()
    within = 0.0
    for g in range(G):
        m = gi == g
        if m.sum() > 0:
            within += ((x[m] - x[m].mean()) ** 2).sum()
    r2 = 1 - float(within / tot)
    # median within-clade spread (in linear-rate fold-change = 2^std_log2)
    spreads = []
    for g in range(G):
        m = gi == g
        if m.sum() >= 3:
            spreads.append(float(x[m].std()))
    med_fold = 2 ** (sorted(spreads)[len(spreads)//2]) if spreads else float("nan")
    print(f"  {nm}: R^2(clade)={r2:.3f}   median within-clade rate spread ~ {med_fold:.2f}x")
print("\n(interpretation: R^2 near 1 = clade-constant rates; R^2 low + big within-clade fold = free rates are NOT mostly clade structure)")
