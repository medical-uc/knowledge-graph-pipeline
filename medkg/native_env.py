"""Native threading guards, applied when `medkg` is imported.

The pipeline loads four libraries that each bring their own native threading
runtime into one process: **torch** (SapBERT, Stage 3), **thinc/blis** (spaCy's
backend, Stages 2 and 5), **faiss** (Stage 3), and **numpy** (Accelerate or
OpenBLAS, everywhere). On macOS, two of these initialising their own thread
pools in one process is a well-known cause of a bare `Segmentation fault: 11`
with no Python traceback — typically the moment the *second* one starts, or the
first time it executes after both are resident.

The tell is that the crash follows load order rather than any particular
library: load torch then run spaCy and it dies in spaCy; load spaCy then load
torch and it dies in torch. Each works perfectly on its own.

Thread counts have to be set **before** the libraries load, which is why this
runs at package import rather than inside a stage — by the time
`SapBertLinker.__init__` calls `torch.set_num_threads`, torch has already built
its pool. Values already present in the environment are never overwritten, so
you keep control.

    MEDKG_NATIVE_THREADS=1     default; 0 disables this module entirely
    MEDKG_ALLOW_DUPLICATE_OMP=1  additionally sets KMP_DUPLICATE_LIB_OK=TRUE

That second one is a diagnostic, not a fix: Intel warns it can produce silently
incorrect numerical results. For a medical pipeline a crash is much preferable
to quietly wrong embeddings, so it stays opt-in.

Caveat worth knowing: this only works if `medkg` is imported before torch,
spaCy or faiss. `run.py` guarantees that. If you `import torch` first in your
own script, set the variables yourself before you do.
"""
from __future__ import annotations

import os

# One variable per threading runtime the pipeline can pull in.
THREAD_VARS = (
    "OMP_NUM_THREADS",          # torch, faiss (libomp/libiomp5)
    "BLIS_NUM_THREADS",         # thinc's vendored BLAS -> spaCy
    "OPENBLAS_NUM_THREADS",     # numpy, if built against OpenBLAS
    "MKL_NUM_THREADS",          # numpy, if built against MKL
    "VECLIB_MAXIMUM_THREADS",   # numpy on macOS Accelerate
    "NUMEXPR_NUM_THREADS",
)


def pin_native_threads(threads=None, env=None) -> dict:
    """Set every native thread-count variable that isn't already set.

    Returns what was applied, so callers can report it. Pure apart from the
    mutation of `env`, and testable by passing a plain dict.
    """
    env = os.environ if env is None else env
    n = str(threads if threads is not None else env.get("MEDKG_NATIVE_THREADS", "1"))
    applied: dict[str, str] = {}
    if n == "0":                       # explicit opt-out; touch nothing
        return applied
    for var in THREAD_VARS:
        if var not in env:
            env[var] = n
            applied[var] = n
    if env.get("MEDKG_ALLOW_DUPLICATE_OMP") == "1" and "KMP_DUPLICATE_LIB_OK" not in env:
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        applied["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return applied


APPLIED = pin_native_threads()
