# gpurec vs AleRax — runtime benchmark kit

Drives a head-to-head **runtime** comparison of the two reconciliation tools on
OIST clusters, on the **Williams 2017** archaeal dataset:

| Tool | Cluster | Hardware | Job |
|------|---------|----------|-----|
| **AleRax** | **Deigo** | CPU / MPI (`compute`, AMD EPYC) | estimate DTL rates on a fixed species tree |
| **gpurec** | **Saion** | 1× A100 (`largegpu`) | same — branch-wise DTL rate estimation |

Both consume the **same** `.ale` CCP gene families and the **same** rooted
species tree, estimate per-branch DTL rates with the species tree **fixed**, and
are timed with `ruse` (wall, peak RSS) plus each tool's own elapsed timer.
"Pilot then full": a small subset first to prove the pipeline, then all 5,446
families.

> You run this kit **on your own computer**; it drives the clusters over your
> SSH aliases. It submits Slurm jobs and rsyncs data/results — nothing runs the
> heavy compute locally.

## New user? (e.g. a labmate in the same unit)

All cluster paths derive from your OIST id; the unit is shared. The **only**
required edit is your id:

```bash
# in config.sh (or export before running):
OIST_ID="your-firstname-lastname"      # -> /work,/flash,/bucket /<UNIT>/<OIST_ID>/...
UNIT="SzollosiU"                        # already the default
```

Since `/bucket` is **shared across the unit**, reuse an existing 4.2 GB dataset
download instead of fetching your own:

```bash
export SHARED_TARBALL=/bucket/SzollosiU/oliver-schick/3_Reconciliation.tar.gz
```

`bin/10_stage.sh` clones gpurec to `/work/<UNIT>/<OIST_ID>/gpurec` for you
(`GPUREC_CLONE_URL`, default `oliver-schick/gpurec`; point it at your own fork if
you like) and `bin/15_setup_gpurec.sh` builds its venv + C++ ext. Everything else
(partitions, modules, model) is unchanged. Off the OIST network,
`export SAION_SSH=saion-ext DEIGO_SSH=deigo-ext`. See `RESULTS.md` for the
reference numbers this pipeline produced.

## Why this is a fair comparison

* **Identical inputs.** AleRax reads each `.ale` directly; gpurec's validated
  `io/ale.py` loader turns the *same* `.ale` into the *same* CCP arrays
  (bit-exact vs its C++ amalgamator). Gene→species mapping is the
  prefix-before-first-`_` rule in **both** tools (verified against the species
  tree), so `families.txt` needs **no** mapping files.
* **Same task.** Fixed species tree (`--species-tree-search SKIP`), DTL model
  (`--rec-model UndatedDTL` ↔ gpurec DTL), branch-wise rates
  (`--per-species-rates` ↔ gpurec `specieswise`).
* **Same missing-data model.** gpurec runs `--fm-mode e-only` (the mode that
  matches AleRax, per `docs/oliver-handoff.md`) and AleRax gets the dataset's
  `fraction_missing` via `--fraction-missing-file` — so both apply the
  missing-gene factor in the extinction recursion. (Verified the file covers
  all 60 tree leaves; AleRax errors otherwise.)
* **Same GPU class.** gpurec is pinned to Saion `largegpu` A100s
  (`--gres=gpu:a100:1`) — homogeneous, recent-driver nodes — never the mixed
  `gpu` partition whose old-driver P100s silently fall back to CPU. The job logs
  `nvidia-smi` and hard-fails if CUDA isn't actually available.
* **Same families.** `families.txt` is the first-N sorted `.ale` — exactly the
  set gpurec's runner globs. (gpurec additionally drops <4-species families,
  matching AleRax's own exclusion; the report shows both kept counts.)

## One-time setup

Edit **`config.sh`** — at minimum the placeholders:

* `SAION_SSH` / `DEIGO_SSH` — your SSH aliases (`saion`/`saion-ext`, `deigo`/`deigo-ext`).
* `SAION_GPUREC_DIR` — your gpurec clone on Saion (branch `cpp-rust-free`).
* `*_BENCH_DIR`, `WILLIAMS_DIR_SAION` — your `/work` paths.
* `SAION_PARTITION` / `SAION_GRES` — A100 partition (here `largegpu`, `gpu:a100:1`).

The dataset is already on Saion. For Deigo the kit rsyncs it from
`LOCAL_WILLIAMS_DIR` (default `/data/17360806/.../Williams_et_al_2017`).

## Run it

```bash
cd benchmark

./bin/00_preflight.sh            # check SSH, data, gpurec checkout, modules
./bin/10_stage.sh                # fetch data + clone gpurec/AleRax on both clusters
./bin/15_setup_gpurec.sh         # ONE-TIME: gpurec .venv (torch+triton) + build C++ .so on Saion login

# ---- PILOT (default 200 families) ----
./bin/20_prepare_families.sh pilot
./bin/30_run_alerax.sh pilot     # submits the Deigo job, prints job id
./bin/40_run_gpurec.sh pilot     # submits the Saion job, prints job id
./bin/status.sh                  # check progress anytime (squeue + sacct + outputs); --logs to tail
#   ... wait for both to finish ...
./bin/50_collect.sh
python3 bin/60_report.py         # -> results/comparison.md + .csv

# ---- FULL (all 5,446 families) once the pilot looks right ----
./bin/20_prepare_families.sh full
./bin/30_run_alerax.sh full
./bin/40_run_gpurec.sh full
./bin/50_collect.sh && python3 bin/60_report.py
```

`results/comparison.md` ends with the headline:
`AleRax <wall>s vs gpurec <wall>s -> Nx`.

## Correctness check (same-optimum fidelity)

Before trusting the runtime numbers, confirm the two tools compute the **same
likelihood**. `bin/70_check_fidelity.sh` sets gpurec's rates to **AleRax's own
fitted values** (the dataset's shipped reference), runs gpurec **forward-only**
(no optimization) on all families, and correlates per-family logL with AleRax's
`per_fam_likelihoods.txt`. Agreement = **Pearson r ~ 1** (the handout reports
~0.999998). It follows the current `MODE`:

- `MODE=global` → `eval_at_alerax_rates.py`: gpurec at AleRax's shipped **global**
  D,L,T (`reconciliation models/global/<ROOT>/`).
- `MODE=specieswise` → `eval_branchwise_perfam.py`: gpurec at AleRax's **per-branch**
  rates + origination (`reconciliation models/branch wise/<ROOT>/`).

```bash
./bin/70_check_fidelity.sh     # extracts the ~3 MB AleRax reference for ROOT, submits a forward-only A100 job
./bin/50_collect.sh            # then read the Pearson r in results/saion/logs/gpurec_fidelity_*.out
```

It is self-contained (uses the shipped AleRax reference, not our own AleRax run)
and extracts only `branch wise/<ROOT>` + `global/<ROOT>` (~3 MB) from the shared
tarball — never the full 14 GB `reconciliation models/` tree.

## Knobs (`config.sh`)

`ROOT` (default `DPANN`; any of the 10 `rooted_phylogeny/` rootings),
`PILOT_N`, `FULL_N`, `DEIGO_NTASKS` (AleRax MPI ranks), `GPUREC_STEPS`,
`GPUREC_DTYPE`, `DEIGO_TIME`/`SAION_TIME`, `DEIGO_MODULES`, `SEED`.

## Files

```
config.sh                 all settings (edit this)
lib.sh                    ssh/rsync helpers
bin/00_preflight.sh       reachability + presence checks
bin/10_stage.sh           fetch data (Zenodo) + clone gpurec/AleRax on both clusters
bin/15_setup_gpurec.sh    one-time: gpurec .venv (torch+triton) + build C++ .so (Saion login)
bin/20_prepare_families.sh  build families.txt on Deigo (pilot|full)
bin/30_run_alerax.sh      submit AleRax on Deigo (pilot|full)
bin/40_run_gpurec.sh      submit gpurec on Saion (pilot|full)
bin/status.sh             check job progress on both clusters (squeue + sacct + outputs; --logs to tail)
bin/50_collect.sh         pull timings/logs/rates back to results/
bin/60_report.py          parse -> comparison.md + comparison.csv
bin/70_check_fidelity.sh  CORRECTNESS: gpurec logL at AleRax's own rates vs AleRax per-family logL
slurm/fidelity_gpurec_saion.sbatch  forward-only eval (eval_branchwise_perfam.py) under ruse
slurm/alerax_deigo.sbatch builds AleRax if needed; runs it under ruse + srun --mpi=pmix
slurm/bench_gpurec_saion.sbatch  runs experiments/run_williams_branchwise.py under ruse
```

## Notes & caveats

* **Branch-wise mode** is what the bundled gpurec runner
  (`experiments/run_williams_branchwise.py`) implements and what
  `--per-species-rates` matches. `global`/`genewise` need the corresponding
  gpurec driver (set `MODE` and point `bin/40` at it).
* **gpurec deploy** uses `git fetch && git reset --hard origin/cpp-rust-free`
  then `touch`es the cached `preprocess_cpp.so` so torch loads it instead of
  JIT-rebuilding (which fails on compute nodes lacking devtoolset-11) — per the
  project's Saion deploy memory. The gpurec sbatch is staged as an **untracked**
  file in the checkout, so `reset --hard` preserves it; nothing is pushed to
  your repo.
* **Convergence vs wall time.** Each tool runs to its own default stopping
  (AleRax converges internally; gpurec runs `GPUREC_STEPS`). The report gives
  raw wall time + ms/family; if you want iso-accuracy, match steps/tolerance and
  re-run.
* `ruse` is loaded via `module load ruse`; if absent the wrappers fall back to
  `/usr/bin/time -v` and the parser reads either.
```
