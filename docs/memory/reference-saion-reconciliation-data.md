---
name: reference-saion-reconciliation-data
description: Location/structure of the Williams 2017 + This_study reconciliation dataset on Saion (the 3_Reconciliation tarball) -- gpurec inputs
metadata: 
  node_type: memory
  type: reference
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

The reconciliation dataset for gpurec lives on Saion at `/work/SzollosiU/gergely-szollosi/3_Reconciliation/`.

- A 4.2 GB tarball `3_Reconciliation.tar.gz` (271,310 entries) holds the FULL dataset. As of 2026-06-18 only `Williams_et_al_2017/rooted_phylogeny/` was extracted to disk (88K); everything else is still packed in the tarball. macOS-made tar -- `.DS_Store` and `._*` AppleDouble junk throughout (ignore).
- Two top-level dirs inside: `Williams_et_al_2017/` and `This_study/`.

`Williams_et_al_2017/rooted_phylogeny/`: 10 rooted archaeal species trees (Newick, ~1.5 KB each), one per rooting hypothesis -- Alti, AMD, Asgard, Cluster2, DPANN, Eury, HaloThermoplas, Kor, TAC, TackA. These are the candidate roots from Williams, Szollosi et al. 2017 PNAS ("Integrative modeling of gene and genome evolution roots the archaeal tree of life").

`This_study/`: full phylogenomics->reconciliation pipeline:
- `0_protein_seqs_initial/`, `1_protein_seqs/` -- 21,177 `.faa`; alignments under `alignment/mafft/` + `alignment/bmge/`; 7,059 `.aln`.
- `2_phylogeny/` -- IQ-TREE: 10,070 `.treefile`, 7,059 `.iqtree`; `best_model/`, `pmsf/`, `iqtree_lg_guide/`.
- `3_UFBOOTs/` -- 7,059 `.ufboot` (gene-tree distributions); `ufboot_for_alerax/`.
- `4_species_tree/`.
- `5_reconcilation models/` -- e.g. `DPANN_DL/Undine_C60_<X>root/` per rooting hypothesis (Altiroot, AMDroot, Asgardroot, Cluster2root, DPANNroot, Euryroot, HalobacThermoplroot, Korroot, MHHroot, Micra5root, MicraDiaroot, TACAroot, TackAroot...), each with `model_parameters/` + `species_trees/`.
- 5,446 `.ale` files across the tree -- ALE-format gene-tree distributions, the classic reconciliation input. Also 330 `.newick`, 15 `.nw`, 160,651 `.txt` (per-family metadata/results).

See [[reference-saion-howto]], [[reference-saion-deploy-loop]].
