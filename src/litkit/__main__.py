# src/litkit/__main__.py
"""LitKit entry point.

CRITICAL: Thread limits are set BEFORE any imports that could pull in FAISS
or other OpenMP-using C extensions. On macOS, FAISS + MPS threading causes
segfaults unless FAISS_NUM_THREADS=1 and OMP_NUM_THREADS=1.

These env vars MUST be set before the C extensions are loaded - setting them
after import has no effect because OpenMP initializes its thread pool at
library load time.
"""
import os
import sys

# Set thread limits BEFORE any heavy imports (faiss, numpy, torch, etc.)
# macOS: FAISS + MPS threading causes segfaults
if sys.platform == "darwin":
    os.environ.setdefault("FAISS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# Now safe to import cli (which eventually imports faiss)
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
