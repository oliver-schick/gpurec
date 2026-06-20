---
name: project-undine-bigtree-rooting
description: "The 'big tree' = This_study/Undine C60 dataset for the big-tree rooting test (after Williams): location, 15 candidate roots, ~257 taxa, ufboot families, AleRax ref"
metadata:
  node_type: memory
  type: project
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

"THE BIG TREE" (user, 2026-06-19: "after Williams also do the big tree rooting") = the **This_study / "Undine C60"** dataset, NOT a bigger archaea60 family set. It is a genuinely larger species tree (~257 leaves vs Williams' ~60 -- crude comma-token count of `Undine_C60_Euryroot_short_name.nw`, verify exactly after extraction). Distinct rooting test from Williams: own tree + own candidate roots.

LOCATION: inside `/work/SzollosiU/gergely-szollosi/3_Reconciliation.tar.gz`, dir `3_Reconciliation/This_study/` (only `Williams_et_al_2017/` was extracted before; This_study extraction started 2026-06-19 as intel job into `williams_run/data/3_Reconciliation/This_study/`). Subdirs:
- `4_species_tree/` -- **15 rooted candidate trees** `Undine_C60_<ROOT>root_short_name.nw`: Alti, AMD, Asgard, Cluster2, DPANN, Eury, HalobacThermopl, Kor, MHH, Micra5, MicraDia, TACA, TackA, TAC, UndineClu2. (Williams had 10; these add MHH, MicraDia, TACA, TAC, UndineClu2 and rename some.)
- `3_UFBOOTs/` -- ~7059 `.ufboot` files = per-family UFBoot **bootstrap tree samples** (~7059 gene families). These are the gene-family input.
- `2_phylogeny/` -- ML gene trees (.treefile/.iqtree/.aln). `1_protein_seqs/`,`0_protein_seqs_initial/` -- FASTA. `6_qmds/reconciliations_model_command.qmd` -- the AleRax run command.
- `5_reconcilation models/` -- the **AleRax reference** (106k txt incl per-root model_parameters + per_fam_likelihoods) to compare gpurec rooting/rates against.

KEY DIFFERENCE vs Williams: This_study has **no pre-built `.ale`** -- families are UFBoot samples. gpurec CAN amalgamate them directly (C++ `amalgamate_clades_and_splits`, preprocess.cpp:604/1212/2005 "handles single-tree or multi-tree CCP / gene tree samples"); NO ALEobserve needed. The Python entry is `GeneDataset(species_tree_path=..., gene_tree_paths=[*.ufboot])` (or `GeneReconModel.from_trees`). The Williams driver `experiments/run_williams_branchwise.py` is `.ale`-only -- the big-tree rooting needs a tree/ufboot-input driver (extend it with a `--gene-trees-dir`/ufboot path, or a new driver) that amalgamates via the C++ path then runs `optimize_theta_wave(specieswise=True)` exactly as Williams.

COMPUTE: S~257 species -> ~513 nodes (vs Williams ~119); ~7059 families. Heavier per-root than Williams but feasible on A100-80GB (S=19999 used ~18GB). 15 roots. Respect the **8 A100 SLURM cap** [[reference-saion-gpu-partition]]. Run AFTER the Williams rooting sweep. Apply the SAME corrected methodology: `--fm-mode e-only`, optimized origination with the root DECOUPLED (default), Brownian prior; report prior-free data-logL per root and compare to AleRax Sum per_fam_likelihoods (the [[project-williams-branchwise-replication]] rooting recipe + analyze_rooting_sweep.py).

Need to confirm a fraction_missing file for This_study (Williams had one). See [[project-williams-branchwise-replication]] for the rooting/fm/origination conventions.
