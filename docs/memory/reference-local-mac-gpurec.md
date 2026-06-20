---
name: reference-local-mac-gpurec
description: "What gpurec analysis runs LOCALLY on the M4 Max Mac (Triton-free preprocessing + topology/sidecar work) vs what needs Saion (GPU forward/backward)"
metadata:
  node_type: memory
  type: reference
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

The M4 Max Mac CAN run the Triton-free parts of gpurec locally -- do NOT round-trip
to Saion for topology/sidecar analysis. Local env: anaconda python3.13 + torch 2.11
(CPU) + ninja + libomp (/opt/homebrew/opt/libomp). Repo /Users/ssolo/GALE.

RUNS LOCALLY (no GPU/cluster):
- The C++ tree preprocessing: `_load_species_helpers` (gpurec.core.preprocess_cpp).
  REQUIRED FIX (committed b96b1fd): preprocess_cpp.py now drops the bare `-fopenmp`
  on `sys.platform=="darwin"` (Apple clang rejects it; the C++ only uses `#pragma
  omp`, no omp_* calls, so serial build is correct). First _load_extension() JIT-
  builds the .so under ~/.cache/torch_extensions (~1-2 min).
- experiments/clade_groups.py (leaf-set mapping, species_node_leafsets), JSON
  sidecar analysis, Newick parsing, origination/depth/rate analyses.
- RUN WITH: `PYTHONPATH=/Users/ssolo/GALE python3 script.py` (the `gpurec` package +
  `experiments/` must be importable; a script in /tmp won't find them otherwise).

NEEDS SAION (largegpu A100): anything importing core/forward.py or the optimizer
(Pi_wave_forward / wave_backward / optimize_theta_wave / implicit_grad) -> Triton
(no arm64-macOS wheel) AND the cluster .venv numpy/torch are GLIBC-built so they
CANNOT run on the login node either (only on largegpu compute nodes). So: fixed-
rate forward likelihood, gradients, Hessian/Laplace, optimization = Saion.

To fetch data for local analysis: `ssh -J oist saion 'tar -cf - <paths>' | tar -xf -`
(rooted_phylogeny trees + *.rates.txt.json sidecars are small).
