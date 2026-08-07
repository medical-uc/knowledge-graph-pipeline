"""medkg — a staged medical knowledge-graph construction pipeline (Stages 1-8).

Importing this package pins native thread counts (see `native_env`) BEFORE
torch, spaCy/thinc or faiss can load. That ordering is the whole point: those
libraries each build a threading pool at import time, and on macOS two of them
doing so in one process segfaults with no Python traceback. Setting the counts
afterwards is too late.
"""
from . import native_env as native_env  # noqa: F401  -- must be first

__all__ = [
    "ir", "config", "console", "native_env",
    "stage1_parse", "stage2_extract", "stage3_link", "stage4_postcoord",
    "stage5_images", "stage6_assert", "stage7_reason", "stage8_serve",
]
