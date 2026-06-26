"""Build a branchwise warm-start sidecar from the PAPER's 4_Ancestral reconstruction
rates (the 26-class clade DTL + structured O whose copy numbers we reproduced). Emits
BOTH per-branch theta (log2 D,L,T) AND the structured origination (omega_log2 / p^O), so
the sidecar is directly (a) eval-able by eval_perfam_undine (reads origination_prob) to get
logL AT the paper rates, and (b) usable as --init-from-rates for a free-fit-all run.

The paper model is Eury-rooted (GCA-labelled). For a non-Eury --root, branch_params_from_alerax
maps the paper clade-rates onto that root's branches by leafset (clades are root-invariant away
from the root); near-root branches that don't map fall back to the per-branch default and the
optimizer refines them. `unmapped` is reported so the transplant quality is visible.

Output: {"theta_log2":[S,3], "origination":{"omega_log2":[S], "origination_prob":[S]}}.
"""
import sys, json, argparse
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_williams_branchwise import _load_species_helpers  # noqa: E402
from clade_groups import branch_params_from_alerax           # noqa: E402

DD = Path("/work/SzollosiU/gergely-szollosi/williams_run/data/3_Reconciliation/This_study")
PAPER = Path("/work/SzollosiU/gergely-szollosi/williams_run/undine/paper_ancestral/"
             "4_Ancestral_reconstruction/Euryarchaeota_root")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model-params", default=str(PAPER / "model_parameters" / "model_parameters.txt"),
                    help="paper 4_Ancestral model_parameters.txt (26-class clade DTL + structured O)")
    ap.add_argument("--atree", default=str(PAPER / "species_trees" / "starting_species_tree.newick"),
                    help="labelled (GCA) species tree matching --model-params")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tree = DD / "4_species_tree" / f"Undine_C60_{args.root}root_short_name.nw"
    sp = _load_species_helpers(str(tree)); S = int(sp["S"])
    rate_branch, orig_branch, diag = branch_params_from_alerax(sp, args.atree, args.model_params)
    theta = torch.log2(rate_branch.clamp_min(1e-10)).reshape(S, 3)        # [S,3] log2(D,L,T)
    payload = {"theta_log2": theta.tolist()}
    if orig_branch is not None:
        o = orig_branch.reshape(-1).clamp_min(1e-12)
        pO = (o / o.sum())                                                # normalized origination prob
        payload["origination"] = {"omega_log2": torch.log2(pO).tolist(),
                                  "origination_prob": pO.tolist()}
        _r = int(torch.argmax(pO))
        print(f"  [paper {args.root}] p^O max branch={_r} p^O={float(pO[_r]):.4f}  "
              f"(root-mass origination)", flush=True)
    json.dump(payload, open(args.out, "w"))
    n_un = len(diag.get("unmapped", []))
    print(f"  [paper {args.root}] init: S={S}, unmapped={n_un}/{S} "
          f"({100.0*n_un/S:.1f}%) -> {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
