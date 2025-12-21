# /src/litkit/embeddings/devices.py

"""
Device and threading helpers for LitKit embeddings

This module centralizes two concerns that must be handled early in process
startup:

1) **Thread caps for BLAS/FAISS** — to avoid oversubscription, instability, or
   rare crashes on shared nodes. Call `configure_threads()` *before importing*
   libraries that spawn worker threads (notably `faiss`).

2) **Device selection for PyTorch/Transformers** — consistent, side-effect-free
   detection of a suitable compute device for embedding models with a simple
   override mechanism.

Environment variables
---------------------
- LITKIT_THREADS: If set (e.g., "16"), caps:
  VECLIB_MAXIMUM_THREADS, OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
  BLIS_NUM_THREADS, FAISS_NUM_THREADS. If unset, reasonable caps are derived
  from `os.cpu_count()` and applied (FAISS gets a stricter cap).
- LITKIT_FORCE_DEVICE: Force device selection: "cpu" | "cuda" | "mps".

Public API
----------
- configure_threads(default: str | None = None) -> dict[str, str]
    Apply thread caps (respecting existing env). Return the effective values.

- detect_device(force_env: str | None = None) -> str
    Return a PyTorch device string: "cuda" (if available), else "mps" (if
    available and usable), else "cpu". Honors LITKIT_FORCE_DEVICE.

- resolve_embed_devices(spec: str, force: bool = False) -> list[str]
    Normalize a device spec into a concrete list of devices. Supports:
    "auto", "cpu", "mps", "cuda:0", "cuda:0,cuda:1". In "auto" mode, selects
    all visible CUDA devices, otherwise "mps" or "cpu".

Notes
-----
- Imports of heavy dependencies (`torch`) are **lazy** and occur inside the
  functions to keep module import side effects minimal.
- Call `configure_threads()` as early as possible, ideally immediately after
  process start and **before** any `faiss` import.

Example
-------
    from litkit.embeddings.devices import configure_threads, detect_device, resolve_embed_devices

    configure_threads()          # must happen before importing faiss
    device = detect_device()     # e.g., "cuda" | "mps" | "cpu"
    devices = resolve_embed_devices("auto")  # e.g., ["cuda:0", "cuda:1"]
"""

from __future__ import annotations

import os
import sys


def configure_threads(default: str | None = None) -> dict[str, str]:
    """Set BLAS/FAISS thread caps (to avoid crashes) unless env already sets them.
    Respect the setting for LITKIT_THREADS if available.
    Return a dict of values.
    NB: This function must be called before importing faiss.
    
    Platform-specific behavior:
    - macOS: Forces FAISS_NUM_THREADS=1 and OMP_NUM_THREADS=1 by default
      to prevent segfaults from FAISS + MPS threading conflicts.
    - Linux/HPC: Uses CPU-count based heuristics for reasonable parallelism.
    
    Users can override with LITKIT_THREADS env var.
    """
    val = default or os.environ.get("LITKIT_THREADS", "").strip()
    if val:
        # Explicit override - respect user's choice
        for k in (
            "VECLIB_MAXIMUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "FAISS_NUM_THREADS",
        ):
            os.environ.setdefault(k, val)
    elif sys.platform == "darwin":
        # macOS: FAISS + MPS threading causes segfaults
        # Force single-threaded operation by default
        os.environ.setdefault("FAISS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        # Other thread vars can remain at reasonable defaults
        for k in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(k, "4")
    else:
        # Linux/HPC: use CPU-count based heuristics
        cpu = os.cpu_count() or 32  # size to the node
        polite = str(min(32, cpu))
        faiss_polite = str(min(16, max(4, cpu // 2)))
        os.environ.setdefault("FAISS_NUM_THREADS", faiss_polite)
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ.setdefault(k, polite)
    keys = [
        "VECLIB_MAXIMUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "FAISS_NUM_THREADS",
    ]
    return {k: os.environ.get(k, "") for k in keys}


# --- Internal helper ---
def _mps_ok() -> bool:
    try:
        import torch  # lazy import

        _ = torch.tensor([0.0]).to("mps")
        _ = (_ + 1).cpu()
        return True
    except Exception:
        return False


def detect_device(force_env: str | None = None) -> str:
    """Select the device type to be used for embeddings.
    Return a PyTorch device string: 'cuda' | 'mps' | 'cpu'
    Respect the setting for LITKIT_FORCE_DEVICE if force_env is None.
    """
    import torch  # lazy import

    forced = (
        force_env if force_env is not None else os.environ.get("LITKIT_FORCE_DEVICE", "")
    ).lower()
    if forced in {"cpu", "cuda", "mps"}:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and _mps_ok():
        return "mps"
    return "cpu"  # fallback: use CPU if a GPU is not available


def resolve_embed_devices(spec: str, force: bool = False) -> list[str]:
    """Normalize --embed-devices into a concrete list.
    Accepts: 'auto', 'cpu', 'mps', 'cuda:0,cuda:1', or 'cuda:0'.
    """
    import torch  # lazy import

    s = (spec or "auto").strip().lower()

    if s in {"cpu", "mps"}:
        return [s]
    if "," in s:
        items = [x.strip() for x in s.split(",") if x.strip()]
        bad = [tok for tok in items if tok.startswith("cuda") and ":" not in tok]
        if bad and not force:
            raise ValueError(
                "Invalid CUDA device tokens "
                f"{bad}; use 'cuda:0,cuda:1,...' or --force-embed-devices"
            )
        return items or ["cpu"]
    if s.startswith("cuda:"):
        return [s]
    if s == "auto":
        if torch.cuda.is_available():
            n = 1
            try:
                n = max(1, int(torch.cuda.device_count() or 1))
            except Exception:
                pass
            return [f"cuda:{i}" for i in range(n)]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and _mps_ok():
            return ["mps"]
        return ["cpu"]
    return [s]  # fallback: user input is treated as a single device token


__all__ = ["configure_threads", "detect_device", "resolve_embed_devices"]
