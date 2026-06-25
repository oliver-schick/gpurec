#!/usr/bin/env bash
# ======================================================================
# gpurec vs AleRax runtime benchmark -- central configuration
# ======================================================================
# Edit the values below for YOUR OIST account, then run the scripts in
# bin/ in order. Every script sources this file. Nothing here touches a
# cluster; it only sets variables.
#
# Comparison design
#   * AleRax  (CPU / MPI)  runs on  DEIGO   (no GPUs there)
#   * gpurec  (GPU/Triton) runs on  SAION   (largegpu A100 partition)
#   * Both estimate per-species-branch DTL rates on a FIXED rooted
#     species tree over the SAME Williams 2017 .ale gene families.
#   * Runtime is measured with `ruse` (wall time, peak RSS, CPU) plus the
#     tools' own elapsed timers.
#   * "Pilot then full": first a small subset (PILOT_N families) to prove
#     the pipeline end to end, then the full set.
# ======================================================================

# ---- Identity: the ONE thing a new user must set ---------------------
# Your OIST id (firstname-lastname). A colleague just sets OIST_ID=their-id and
# every /work, /flash, /bucket path below derives from it; the unit is shared.
# (Override any individual *_DIR below if your layout differs.)
OIST_ID="${OIST_ID:-oliver-schick}"
UNIT="${UNIT:-SzollosiU}"

# ---- SSH aliases (from your ~/.ssh/config) ---------------------------
# On the OIST network use the bare alias; off-network export SAION_SSH=saion-ext
# DEIGO_SSH=deigo-ext (one-time host-key accept may be needed; see README).
SAION_SSH="${SAION_SSH:-saion}"
DEIGO_SSH="${DEIGO_SSH:-deigo}"

# ---- Partitions / Slurm resources ------------------------------------
# Saion GPU partition for gpurec. On this account the A100 partition is
# "largegpu" (the cluster naming may change -- override SAION_PARTITION).
SAION_PARTITION="${SAION_PARTITION:-largegpu}"
SAION_GRES="${SAION_GRES:-gpu:a100:1}"   # if rejected, try: gpu:1
SAION_CPUS="${SAION_CPUS:-8}"
SAION_MEM="${SAION_MEM:-64G}"
SAION_TIME="${SAION_TIME:-12:00:00}"     # largegpu wall cap is 12h
SAION_PYTHON_MODULE="${SAION_PYTHON_MODULE:-python/3.11.11}"

# Deigo CPU partition for AleRax (MPI). compute = 4-day, 2000 cores/user.
DEIGO_PARTITION="${DEIGO_PARTITION:-compute}"
DEIGO_CONSTRAINT="${DEIGO_CONSTRAINT:-epyc}"   # AMD EPYC nodes (build + run)
# AleRax MPI ranks. NOTE: AleRax parallelizes per-family and logs a "Recommended
# maximum number of cores" (it was ~20 for the full Williams set, ~9 for the pilot)
# -- beyond that, extra ranks idle (load balance was ~0.32, a few huge families
# dominate). 64 over-provisions but ensures AleRax is never core-starved (so the
# comparison can't be accused of under-resourcing it); set lower to be frugal.
DEIGO_NTASKS="${DEIGO_NTASKS:-64}"
DEIGO_MEM_PER_CPU="${DEIGO_MEM_PER_CPU:-4G}"
DEIGO_TIME="${DEIGO_TIME:-1-00:00:00}"         # 1 day (raise for full set)
# Module names vary across cluster updates -- override if `module avail`
# shows different versions. These are loaded in the Deigo sbatch.
# Pinned to what `module avail` showed on Deigo (your preflight): OpenMPI is
# 'openmpi.gcc/5.0.3' (NOT a bare 'openmpi'), and cmake/gcc need explicit vers.
DEIGO_MODULES="${DEIGO_MODULES:-gcc/11.2.1 cmake/3.31.2 openmpi.gcc/5.0.3}"

# ---- Remote working directories (derived from UNIT/OIST_ID) ----------
# gpurec checkout on Saion /work (a clone of GPUREC_CLONE_URL, branch below).
SAION_GPUREC_DIR="${SAION_GPUREC_DIR:-/work/$UNIT/$OIST_ID/gpurec}"
GPUREC_BRANCH="${GPUREC_BRANCH:-cpp-rust-free}"
GPUREC_REMOTE="${GPUREC_REMOTE:-origin}"

# Where the kit stages inputs / outputs on each cluster. Saion = /work,
# Deigo = /flash (its fast scratch; Deigo /work is not where you write).
SAION_BENCH_DIR="${SAION_BENCH_DIR:-/work/$UNIT/$OIST_ID/bench_gpurec_alerax}"
DEIGO_BENCH_DIR="${DEIGO_BENCH_DIR:-/flash/$UNIT/$OIST_ID/bench_gpurec_alerax}"

# AleRax cloned + built on Deigo (under the bench dir).
DEIGO_ALERAX_DIR="${DEIGO_ALERAX_DIR:-${DEIGO_BENCH_DIR}/AleRax}"

# ---- Dataset locations (derived) -------------------------------------
# The Williams_et_al_2017 dir that CONTAINS ccps/ rooted_phylogeny/ fraction_missing.
WILLIAMS_DIR_SAION="${WILLIAMS_DIR_SAION:-/work/$UNIT/$OIST_ID/3_Reconciliation/Williams_et_al_2017}"
WILLIAMS_DIR_DEIGO="${WILLIAMS_DIR_DEIGO:-${DEIGO_BENCH_DIR}/Williams_et_al_2017}"

# ---- Where the kit FETCHES inputs (default: cluster-side, no local copy) ----
# Primary path: each cluster downloads/clones directly (fast cluster internet,
# no dependence on your local disk). Override DATA_SOURCE=local to upload from
# the LOCAL_* paths below instead.
DATA_SOURCE="${DATA_SOURCE:-cluster}"        # cluster | local

# Williams data = the 3_Reconciliation tarball on Zenodo record 17360806.
# The kit downloads it on each cluster and extracts only Williams_et_al_2017/.
ZENODO_TARBALL_URL="${ZENODO_TARBALL_URL:-https://zenodo.org/records/17360806/files/3_Reconciliation.tar.gz?download=1}"
# Internal tarball prefix to strip (3_Reconciliation/Williams_et_al_2017/ -> .).
WILLIAMS_TAR_STRIP="${WILLIAMS_TAR_STRIP:-2}"
WILLIAMS_TAR_MEMBERS="${WILLIAMS_TAR_MEMBERS:-3_Reconciliation/Williams_et_al_2017/ccps 3_Reconciliation/Williams_et_al_2017/rooted_phylogeny 3_Reconciliation/Williams_et_al_2017/fraction_missing}"
# /bucket and /home are shared across Saion+Deigo (/work and /flash are NOT), so
# the 4.2 GB tarball is downloaded ONCE to /bucket and both clusters extract from
# it. /bucket is unit-shared: a colleague can REUSE an existing download instead
# of fetching their own, e.g.
#   SHARED_TARBALL=/bucket/SzollosiU/oliver-schick/3_Reconciliation.tar.gz
SHARED_TARBALL="${SHARED_TARBALL:-/bucket/$UNIT/$OIST_ID/3_Reconciliation.tar.gz}"
SHARED_TARBALL_DLHOST="${SHARED_TARBALL_DLHOST:-$SAION_SSH}"   # login node that runs the one download
# Per-cluster tarball override (default: the shared one). Empty SHARED_TARBALL
# falls back to downloading separately on each cluster.
SAION_WILLIAMS_TARBALL="${SAION_WILLIAMS_TARBALL:-$SHARED_TARBALL}"
DEIGO_WILLIAMS_TARBALL="${DEIGO_WILLIAMS_TARBALL:-$SHARED_TARBALL}"

# Repos cloned cluster-side (HTTPS; public). gpurec on Saion, AleRax on Deigo.
GPUREC_CLONE_URL="${GPUREC_CLONE_URL:-https://github.com/oliver-schick/gpurec.git}"
ALERAX_CLONE_URL="${ALERAX_CLONE_URL:-https://github.com/BenoitMorel/AleRax.git}"
GPUREC_AUTOCLONE="${GPUREC_AUTOCLONE:-1}"    # 1 = clone gpurec on Saion if missing
ALERAX_AUTOCLONE="${ALERAX_AUTOCLONE:-1}"    # 1 = clone AleRax on Deigo if missing

# Local fallbacks, used only when DATA_SOURCE=local (your LaCie drive).
LOCAL_WILLIAMS_DIR="${LOCAL_WILLIAMS_DIR:-/media/oliver/LaCie/Lenovo_Extension/17360806/3_Reconciliation/Williams_et_al_2017}"
LOCAL_ALERAX_DIR="${LOCAL_ALERAX_DIR:-/media/oliver/LaCie/Lenovo_Extension/AleRax}"

# ---- Experiment parameters -------------------------------------------
ROOT="${ROOT:-DPANN}"                 # rooted_phylogeny/<ROOT> species tree
# Rate model. global = one shared D,L,T (fast in AleRax); specieswise = per-branch
# (AleRax --per-species-rates, much slower). Start with global.
MODE="${MODE:-global}"
REC_MODEL="${REC_MODEL:-UndatedDTL}"  # DTL model (matches gpurec)
SEED="${SEED:-42}"

PILOT_N="${PILOT_N:-200}"             # families in the pilot pass
FULL_N="${FULL_N:-0}"                 # 0 = ALL families (the full pass)

# gpurec optimizer controls (passed to the bench driver bench_gpurec_fit.py)
GPUREC_STEPS="${GPUREC_STEPS:-200}"
GPUREC_DTYPE="${GPUREC_DTYPE:-float64}"
GPUREC_MIN_SPECIES="${GPUREC_MIN_SPECIES:-4}"   # AleRax drops <4-species families
GPUREC_FAMILY_BATCH_SIZE="${GPUREC_FAMILY_BATCH_SIZE:-0}"  # 0 = all families at once; raise for full set if OOM
# Fraction-missing mode. 'e-only' matches AleRax (missing-data factor in the
# extinction recursion ONLY) -- per docs/oliver-handoff.md ("Always use it").
GPUREC_FM_MODE="${GPUREC_FM_MODE:-e-only}"
# torch version installed by bin/15_setup_gpurec.sh into the Saion .venv.
# Pinned to the validated stack (docs: torch 2.6.0+cu124 / triton 3.2.0). The
# Linux CUDA wheel bundles its own CUDA runtime + a matching triton. Set to
# just "torch" to take the latest (riskier: newer triton may break kernels).
GPUREC_TORCH_SPEC="${GPUREC_TORCH_SPEC:-torch==2.6.0}"
# numpy<2 installs a prebuilt manylinux2014 wheel (vs numpy 2.4.x which
# source-builds against gcc11 and then needs a CXXABI_1.3.9 libstdc++ the OIST
# login node lacks). <2 is also the safer ABI match for the older gpurec code.
GPUREC_NUMPY_SPEC="${GPUREC_NUMPY_SPEC:-numpy<2}"
# scipy (used by the L-BFGS optimizer; not in pyproject). <1.12 has a
# manylinux2014 wheel for the login node's glibc; the newest scipy ships only
# manylinux_2_28 wheels -> pip would source-build it and fail (no cython).
GPUREC_SCIPY_SPEC="${GPUREC_SCIPY_SPEC:-scipy<1.12}"

# ---- Local results dir (on your computer) ----------------------------
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}"

# ======================================================================
# Map gpurec MODE -> AleRax rate flag (do not edit unless adding a mode).
# ======================================================================
# Explicit, non-deprecated AleRax parametrization (the bare --per-species-rates
# form is deprecated, and relying on the empty-flag default is ambiguous).
case "$MODE" in
  global)       ALERAX_RATE_FLAG="--model-parametrization GLOBAL" ;;       # one shared D,T,L
  specieswise)  ALERAX_RATE_FLAG="--model-parametrization PER-SPECIES" ;;  # per-branch
  genewise)     ALERAX_RATE_FLAG="--model-parametrization PER-FAMILY" ;;   # per-family
  *) echo "config.sh: unknown MODE='$MODE' (use global|specieswise|genewise)" >&2; return 1 2>/dev/null || exit 1 ;;
esac

# The benchmark driver bench_gpurec_fit.py supports MODE = global | specieswise.
# (genewise would need AleRax --per-family-rates + a genewise gpurec call -- not wired.)
if [ "$MODE" = "genewise" ]; then
  echo "config.sh: NOTE -- MODE=genewise is not wired on the gpurec side yet" >&2
fi
