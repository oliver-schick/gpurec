---
name: reference-saion-howto
description: "Master operational how-to for OIST Saion -- connect, deploy loop, partitions, the .feather gotcha, fan-out arrays + per-partition GPU caps, preemption, reset --hard heal"
metadata: 
  node_type: memory
  type: reference
  originSessionId: b0dbaa9f-395a-4562-b1d3-98496728ab02
---

Master how-to for OIST Saion compute. Ties together the focused Saion notes and the operational lessons that keep biting. Cross-links in brackets.

## Connect + repo
- SSH is non-interactive (key-based, user `gergely-szollosi`). The `X11 forwarding request failed` / `bind 127.0.0.1:8888` warnings are harmless; filter with `grep -ivE "forwarding|bind on|channel"`.
- The bare `ssh saion` alias works ONLY on the OIST network/VPN -- `saion.oist.jp`/`deigo.oist.jp` are internal-only (NXDOMAIN off-network); only `login.oist.jp` (the `oist` alias) resolves publicly. OFF-NETWORK, jump through it: `ssh -J oist saion`. The config's `saion-ext` is broken (undefined `oist-ext` jump host). The saion login node's git is ancient (no `git -C`; use `cd <dir> && git ...`).
- SLURM (`sbatch`/`squeue`/`scancel`) exists ONLY on saion, never on the Mac.
- gpurec work dir on the cluster (FORK checkout, origin = `oliver-schick/gpurec`): `/work/SzollosiU/gergely-szollosi/gpurec-cpp`. The neighbouring `/work/SzollosiU/gergely-szollosi/gpurec` is upstream `SisyphusMountain/gpurec` main (old GMRES track, Rust `crates/`). Sibling project ising-denoiser: `/work/SzollosiU/gergely-szollosi/genecontext_mlm/ising-denoiser`. See [[reference-saion-deploy-loop]].

## Deploy loop (Mac -> Saion) -- see [[reference-saion-deploy-loop]]
1. Commit + push from the Mac. SLURM wrappers run the on-disk code and do NOT pull, so unpushed/unpulled fixes silently run stale. [[feedback-push]]
2. `ssh saion`, then heal the checkout, do NOT `git pull`: `git fetch origin && git reset --hard origin/<branch>` (gpurec branch is `cpp-rust-free`). Plain `git pull` can conflict/refuse because of the GBs of untracked artifacts (feathers, npz, parquet) in the worktree. `reset --hard` only rewrites tracked files and leaves untracked data intact. Never `git stash -u` -- it hangs for minutes trying to stash multi-GB untracked data.
3. `squeue --me` BEFORE relaunching (a fresh launch can collide with an in-flight job on the same outdir). Then `sbatch`.

## Partitions: largegpu (A100) vs gpu (MIXED: V100 usable, P100 useless) -- see [[reference-saion-gpu-partition]]
- largegpu: 4 nodes `saion-gpu23..26`, 8x A100-80GB each (32 A100 total). THE partition for big CUDA training AND for anything touching `.feather`.
- gpu is NOT all P100:
  - `saion-gpu15..22`: 4x V100-32GB each (32 V100). These WORK: `torch.cuda.is_available()` is True and pyarrow/`.feather` reads succeed (newer driver + libstdc++ than the P100s). Use them to offload inference/eval off a saturated largegpu. V100 has ~half the A100 HBM, so HO `B*heads*N^2` attention needs a smaller batch (batch 48 ~= 17.6 GB fits 32 GB; 64 is tight).
  - `saion-gpu07..14`: 4x P100 each. Driver too old for `torch 2.6.0+cu124` -> `cuda.is_available()` False, `--device cuda` silently runs on CPU; old libstdc++ also breaks pyarrow. Do NOT use.
  - Submit with `-p gpu --gres=gpu:v100:1` to PIN a V100 (a bare `gpu:1` may land a useless P100). The `gpu-v100`/`gpu-p100` partition names are NOT in the szollosiuni assoc -- use `-p gpu` + the `:v100:` gres type.
- See free GPUs: `sinfo -p largegpu -o "%n %G %t %C"` and `sinfo -p gpu -N -o "%N %G %t" | grep v100` (look for `idle`/`mix`).

## THE pyarrow / .feather gotcha (largegpu only)
Reading `.feather` (pandas -> pyarrow) works ONLY on largegpu. The login node and gpu/P100 nodes have a broken libarrow.so / old libstdc++ (`Unable to find a usable engine`, `GLIBCXX_... not found`). Consequence: ANY job that reads a feather MUST run on largegpu. Don't debug feather errors on the login node; move the work to `srun -p largegpu`.

## Python + downloads
- `.venv/bin/python` works standalone on largegpu (self-contained interpreter). Still put `module load python/3.11.11` in every `slurm/*.sbatch` after the `cd` -- admin says the system python may change/vanish on OS upgrade. [[feedback-slurm-module-python]]
- Cluster `urllib`/python SSL is broken for outbound HTTPS -> use `curl` for downloads, never python urllib.

## Go FAST: fan out with SLURM arrays
- Do NOT loop N items serially on one GPU. Use a job array: one A100 per task, e.g. `#SBATCH --array=0-19%8` (20 tasks, 8 concurrent), map `$SLURM_ARRAY_TASK_ID` -> item. Each task should be short and resume-safe (`[ -s "$out" ] && exit 0`), so the 12 h wall and preemption are non-issues.
- Association GPU caps are PER-PARTITION and independent, NOT one global 8. Extra array tasks sit `PENDING (AssocGrpGRES)` until a slot frees -- this is the cap, not an error.
  - `largegpu`: 8 A100 at once (the ceiling there).
  - `gpu`: 4 V100 at once. ALSO a CPU cap (`AssocGrpCpuLimit`, ~32 cores): the eval is GPU-bound, so pass `--cpus-per-task=2` or only ~4 tasks fit and you never reach the 4-GPU cap.
  - They ADD UP: hold 8 A100 + 4 V100 = 12 GPUs simultaneously (verified). When largegpu is full, push inference/eval to the V100s instead of waiting. [[feedback-slurm-wall]]

## Preemption + robustness
- largegpu jobs CAN be preempted mid-run (`srun: ... task N: Terminated`). Use `#SBATCH --requeue` + resume-safe launchers so a kill just re-runs the affected task from where it left off. `--open-mode=append` keeps the log.
- Wall cap is 12 h, 8 GPUs; split long curricula into resumable stages. [[feedback-slurm-wall]]
- Never let a launcher default to `rm -rf`/`scancel`; destructive ops behind an explicit `--clean`. [[feedback-no-destructive-defaults]]
- ASCII only in code, logs, and SLURM output. [[feedback-ascii-only]]

## Quick interactive largegpu shell
`srun -p largegpu --gres=gpu:a100:1 -c 8 --mem=32G -t 00:15:00 --pty bash` then `module load python/3.11.11 && .venv/bin/python ...`
