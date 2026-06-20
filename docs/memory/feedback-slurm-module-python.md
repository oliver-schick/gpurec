---
name: feedback-slurm-module-python
description: "Every Saion slurm/*.sbatch must `module load python/3.11.11`; admin warns the system python may change/vanish on OS upgrade"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

Every new SLURM wrapper under `slurm/` must `module load python/3.11.11` right after the `cd` into the repo and before any `python3.11` invocation.

**Why:** OIST admin (Jan) recommends the env module over the system `/usr/bin/python3.11`: "They're for system use, and we don't try to keep them consistent across updates" -- the system package can disappear or change version on an OS upgrade, silently breaking batch jobs with cryptic crashes (SLURM exit `0:53` was an early symptom on saion-gpu26, Apr 2026).

**How to apply:** put this right after the `cd`:
```
# Use the environment module rather than the system python -- admin
# recommendation (system python may change with OS upgrades).
module load python/3.11.11
```
Calling `python3.11` explicitly is still fine (the module exposes that binary); the change is purely which install of python3.11 wins. `.venv/bin/python` is self-contained on largegpu but still load the module per admin. See [[reference-saion-howto]], [[feedback-slurm-wall]].
