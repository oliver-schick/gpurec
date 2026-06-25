"""Build a branchwise theta warm-start sidecar from an AleRax clade model's per-branch rates
(e.g. DTL_br2 = the paper's UNIFORM-CLADE BIC-best model). Warm-starting the uniform-O branchwise
fit from these keeps it in the model's basin (Eury region) instead of the rfx centroid (TackA
basin) -- the rooting is bistable, so the init picks the basin. Mirrors per_node_copies'
--alerax-model rate extraction (branch_params_from_alerax). Output: {"theta_log2": [S,3]}.
"""
import sys, json, math, argparse
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import _load_species_helpers  # noqa: E402
from clade_groups import branch_params_from_alerax           # noqa: E402

DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
AXBASE = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/alerax_ref/"
              "3_Reconciliation/This_study/5_reconcilation models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--alerax-model", default="DTL_br2", choices=["DTL_br2", "DTL_br1_O"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tree = DD / "4_species_tree" / f"Undine_C60_{args.root}root_short_name.nw"
    sp = _load_species_helpers(str(tree)); S = int(sp["S"])
    mdir = AXBASE / args.alerax_model / (f"Undine_C60_{args.root}root"
                                         + ("2_OR" if args.alerax_model == "DTL_br1_O" else ""))
    atree = mdir / "species_trees" / "starting_species_tree.newick"
    atree = str(atree) if atree.exists() else str(tree)
    rate_branch, orig_branch, diag = branch_params_from_alerax(
        sp, atree, str(mdir / "model_parameters" / "model_parameters.txt"))
    theta = torch.log2(rate_branch.clamp_min(1e-10)).reshape(S, 3)   # [S,3] = log2(D,L,T) per branch
    json.dump({"theta_log2": theta.tolist()}, open(args.out, "w"))
    print(f"  [{args.alerax_model} {args.root}] warm-start init: S={S}, unmapped="
          f"{len(diag.get('unmapped', []))} -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
