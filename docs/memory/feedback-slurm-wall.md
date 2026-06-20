---
name: feedback-slurm-wall
description: "OIST Saion largegpu caps jobs at 12h walltime, 8 GPUs; every slurm/*.sbatch needs --time <= 12:00:00 and trainers must be resumable"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

OIST Saion `largegpu` caps SLURM jobs at 12 h walltime, 8 GPUs. Every wrapper under `slurm/` must set `#SBATCH --time=HH:MM:SS` <= 12:00:00.

**Why:** user explicitly flagged this after jobs were rejected/queued; a `--time=24:00:00` script is rejected by SLURM. Example violation: a `slurm/run_set_transformer.sh` written with `--time=24:00:00` expecting an 8-stage curriculum to fit in one shot -- fixed to `12:00:00` relying on per-stage checkpoint resume. Comes up every time a new training pipeline is added.

**How to apply:** if a curriculum does not fit in 12 h, make the trainer resumable so re-submitting the same job picks up from the last completed stage:
- entry points call `load_progress(od)` and check for the most recent stage checkpoint before each stage;
- persist each stage via `save_ckpt(model, od, stage_name, prog, lf)` immediately after it completes;
- do not bury work in one mega-stage > 12 h; split into smaller resumable stages.

Caps are per-partition and independent and they ADD UP: 8 A100 (largegpu) + 4 V100 (gpu) = 12 GPUs at once. Extra array tasks sit `PENDING (AssocGrpGRES)` until a slot frees -- that is the cap, not an error. See [[reference-saion-gpu-partition]], [[reference-saion-howto]].
