"""C++-accelerated preprocessing pipeline for CCP construction."""

from __future__ import annotations

import pathlib
import sys
from functools import lru_cache
from typing import Any

from torch.utils.cpp_extension import load


_CPP_DIR = pathlib.Path(__file__).resolve().parent / "cpp"
_CPP_SRC = _CPP_DIR / "preprocess.cpp"


@lru_cache(maxsize=1)
def _load_extension() -> Any:
    sources = [
        str(_CPP_SRC),
        str(_CPP_DIR / "tree_utils.cpp"),
        str(_CPP_DIR / "clade_utils.cpp"),
    ]
    # The C++ only uses `#pragma omp` (no omp_* runtime calls), so OpenMP is
    # purely a parallelism hint and is OPTIONAL. Apple clang rejects a bare
    # `-fopenmp`, so on macOS we build serial (correct, just single-threaded) --
    # lets the Triton-free preprocessing run locally for analysis/tests.
    if sys.platform == "darwin":
        cflags, ldflags = ["-O3"], []
    else:
        cflags, ldflags = ["-O3", "-fopenmp"], ["-fopenmp"]
    return load(
        name="preprocess_cpp",
        sources=sources,
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        verbose=False,
    )
