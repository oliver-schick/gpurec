---
name: reference-huang-archaea-rooting-paper
description: "The archaeal-rooting paper this project replicates — Huang…Szollosi,Williams,Spang 2025 bioRxiv 10.1101/2025.11.11.687807: AleRax DTLO reconciliation, Euryarchaeota root region, exact model specs"
metadata: 
  node_type: memory
  type: reference
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

Huang W-C, Dombrowski N, Mahendrarajah TA, Wahl N, Stamatakis A, **Szöllősi GJ** (the user), Williams TA, Spang A (2025). "Phylogenetic reconciliation supports a methanogenic ancestor of the Archaea and a derived origin for host-associated lineages." bioRxiv 2025.11.11.687807. (PDF body renders garbled via Read; the user pasted the full text + Methods this session.)

EXACT MODEL SPECS (what gpurec replicates):
- AleRax v1.1.1; ALL optimizations = limited-memory BFGS (ML, no prior, loose tolerance).
- Big tree (This_study/Undine C60): 257 taxa (gpurec S=513), 7059 arCOG gene families, 15 candidate roots. Best model by **BIC = DTL_br2 = 72 params** (lineage-specific D,T,L for each Eury/TACK+Asgard/DPANN sublineage + 2 internal DPANN nodes).
- Global 3-param model → DPANN Cluster2 root = EXPLICITLY the small-genome-attraction (SGA) artifact (their text). Adding DPANN-specific loss ameliorates it.
- Ancestral-reconstruction origination: `--origination OPTIMIZE` for {root, its 2 immediate descendants, DPANN} = 4 classes.
- Williams (60-taxon) reanalysis: 14-lineage DTL + 3-LCA DTLO {DPANN,Eury,TACK+Asgard} → Euryarchaeota root, AU test P=1.
- Result = a "narrow root REGION at/near base of Euryarchaeota" (Eury or MHH root non-rejected by AU) — NOT a sharp single root. LACA = complex free-living (hyper)thermophilic methanogen; LACA gene content ≈ 800–1219 families (Eury root).
- Rooting criterion = **AU test** (CONSEL, 1e6 bootstrap) on per-family likelihoods, NOT argmax-likelihood.

See [[project-rooting-two-channel-design]] [[project-undine-bigtree-rooting]].
