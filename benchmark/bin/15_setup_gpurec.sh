#!/usr/bin/env bash
# Set up gpurec's Python env on Saion + build the C++ extension ONCE.
# Resolves the preflight item "ensure a Python env with torch+triton+gpurec".
#
# Runs entirely on the Saion LOGIN node, which has: internet (for pip), a recent
# gcc (gcc/11.2.1 / devtoolset-11) for the C++ build, and a /work that is SHARED
# with the compute nodes -- so the .so built here is loaded by the GPU job.
#
# The .so is built by loading preprocess_cpp.py STANDALONE (it only needs torch),
# NOT by importing the gpurec package (whose __init__ pulls triton/forward). And
# TORCH_EXTENSIONS_DIR is pinned to <repo>/.torch_ext so the artifact lands where
# the sbatch wrappers expect it (and on shared /work).
#
# usage: bin/15_setup_gpurec.sh
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
start_log 15_setup_gpurec
require_not_placeholder "$SAION_GPUREC_DIR"

rsh "$SAION_SSH" "[ -d '$SAION_GPUREC_DIR/.git' ]" \
  || die "no gpurec checkout at $SAION_GPUREC_DIR -- run bin/10_stage.sh first"

TEXT_DIR="$SAION_GPUREC_DIR/.torch_ext"
log "Saion login node: venv + pip install ($GPUREC_TORCH_SPEC + gpurec[triton]) + build C++ .so"
log "  (one-time; torch+triton wheels are a few GB over the login node's internet)"

rsh "$SAION_SSH" "
  set -e
  cd '$SAION_GPUREC_DIR'
  module load $SAION_PYTHON_MODULE gcc/11.2.1
  # module load adds gcc11 to the COMPILE path but not the RUNTIME loader path,
  # so python keeps picking the old system /lib64/libstdc++.so.6 (no CXXABI_1.3.9).
  # Put gcc11's libstdc++ first so numpy + the freshly-built .so load here.
  GCCLIB=\$(dirname \"\$(gcc -print-file-name=libstdc++.so.6)\")
  export LD_LIBRARY_PATH=\"\$GCCLIB:\${LD_LIBRARY_PATH:-}\"
  echo '== gcc libstdc++ dir:' \$GCCLIB '=='
  echo '== python:' \$(python3.11 --version) '=='
  if [ ! -x .venv/bin/python ]; then python3.11 -m venv .venv; echo '== created .venv =='; else echo '== reusing .venv =='; fi
  ./.venv/bin/python -m pip install --upgrade pip wheel
  echo '== installing $GPUREC_TORCH_SPEC + $GPUREC_NUMPY_SPEC + ninja + $GPUREC_SCIPY_SPEC =='
  # --only-binary=:all: forbids source builds: pip must use a prebuilt wheel for
  # every package (so it auto-picks a glibc-compatible version instead of trying
  # to compile the newest one and failing on the old login-node toolchain).
  # scipy is used by optimize_theta_wave (L-BFGS-B) but is not declared in pyproject.
  ./.venv/bin/pip install --only-binary=:all: '$GPUREC_TORCH_SPEC' '$GPUREC_NUMPY_SPEC' ninja '$GPUREC_SCIPY_SPEC'
  echo '== installing gpurec (-e .[triton]) =='
  ./.venv/bin/pip install -e '.[triton]'
  echo '== building C++ preprocess extension (standalone load; gcc/11.2.1) =='
  # torch's cpp_extension builds via ninja (subprocess on PATH); we invoke python
  # by full path so the venv bin is not otherwise on PATH -- add it so ninja is found.
  export PATH=\"\$PWD/.venv/bin:\$PATH\"
  mkdir -p '$TEXT_DIR'
  PYTHONPATH=. TORCH_EXTENSIONS_DIR='$TEXT_DIR' ./.venv/bin/python - <<'PYEOF'
import importlib.util as u
spec = u.spec_from_file_location('pcpp', 'gpurec/core/preprocess_cpp.py')
mod = u.module_from_spec(spec); spec.loader.exec_module(mod)
mod._load_extension()
print('preprocess_cpp built OK')
PYEOF
  echo '== versions =='
  ./.venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, \"cuda-build\", torch.version.cuda)'
  ./.venv/bin/python -c 'import triton; print(\"triton\", triton.__version__)' || echo 'WARN: triton import failed on login node (OK if it imports on the A100)'
  ./.venv/bin/python -c 'import numpy, scipy; print(\"numpy\", numpy.__version__, \"scipy\", scipy.__version__)'
  echo '== cached artifact =='
  ls -la '$TEXT_DIR/preprocess_cpp/preprocess_cpp.so'
"
log "gpurec env ready. Re-run bin/00_preflight.sh (Saion .venv + .so now present), then bin/40_run_gpurec.sh pilot"
