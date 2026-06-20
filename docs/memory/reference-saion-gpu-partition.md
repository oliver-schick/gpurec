---
name: reference-saion-gpu-partition
description: "OIST Saion GPU hardware -- largegpu A100-80GB vs the MIXED gpu partition (V100 usable, P100 useless); pin :v100:, caps are per-partition"
metadata: 
  node_type: memory
  type: reference
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

On OIST Saion, `largegpu` (A100-80GB, `saion-gpu23..26`) is the main partition for big CUDA training + `.feather` work. The general `gpu` partition is MIXED, so do NOT write it off wholesale:

- `saion-gpu15..22`: 4x V100-32GB each (32 total). These WORK -- newer driver + libstdc++, so `torch.cuda.is_available()` is True and `import pyarrow`/`.feather` reads succeed. Pin with `-p gpu --gres=gpu:v100:1` (the szollosiuni assoc has `gpu` but NOT `gpu-v100`/`gpu-p100`). Use to offload eval/inference off a saturated largegpu; V100 has ~half the A100 HBM so shrink the batch (HO attention is `B*heads*N^2`).
- `saion-gpu07..14`: 4x P100 each. CUDA-11.1-era driver too old for the project's `torch 2.6.0+cu124` wheel -> `cuda.is_available()` False and `--device cuda` silently falls back to CPU; old libstdc++ also breaks `import pyarrow` (`GLIBCXX_3.4.21 not found`). A bare `--gres=gpu:1` on `gpu` can land one of these -- always request `:v100:` explicitly.

Caps are per-partition + independent: 8 A100 (largegpu) + 4 V100 (gpu) can run at once (= 12 GPUs); each hits `PENDING (AssocGrpGRES)` past its own cap. The `gpu` partition also has an `AssocGrpCpuLimit` (~32 cores) -- use `--cpus-per-task=2` for the GPU-bound eval so 4 tasks actually fit.

Quick interactive check: `srun --partition=largegpu --gres=gpu:1 --time=00:15:00 --immediate=120 bash runner.sh` then call `./.venv/bin/python3.11` (after `module load python/3.11.11`).

See [[reference-saion-deploy-loop]] and [[feedback-slurm-wall]].
