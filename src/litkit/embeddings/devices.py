# /src/litkit/embeddings/devices.py

from __future__ import annotations

import os


def configure_threads(default: str | None = None) -> dict:
    """Set BLAS/FAISS thread caps (to avoid crashes) unless env already sets them.
    Respect the setting for LITKIT_THREADS if available.
    Return a dict of values.
    NB: This function must be called before importing faiss.
    """
    val = default or os.environ.get("LITKIT_THREADS", "").strip()
    if val:
        for k in (
            "VECLIB_MAXIMUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "FAISS_NUM_THREADS",
        ):
            os.environ.setdefault(k, val)
    else:
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


def detect_device(force_env: str | None = None) -> dict[str, str]:
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
                f"Invalid CUDA device tokens {bad}; use 'cuda:0,cuda:1,...' or --force-embed-devices"
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
