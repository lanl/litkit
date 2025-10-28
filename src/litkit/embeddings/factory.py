# src/litkit/embeddings/factory.py

from __future__ import annotations

from typing import Any

from .base import Embedder  # protocol for type hints
from .devices import detect_device, resolve_embed_devices
from .sbert_mpnet import SBERT_ID, ChunkEmbedderSBERT
from .specter2 import PaperEmbedderSpecter2, SPECTER2_ID


def make_paper_embedder() -> tuple[Embedder, dict[str, Any]]:
    """
    Instantiate the paper-level embedder (SPECTER2).

    Returns
    -------
    (embedder, cfg)
        embedder : object implementing Embedder.encode(list[str]) -> np.ndarray
        cfg      : dict with metadata (model id, dim, device info if available)
    """
    e = PaperEmbedderSpecter2()  # Paper embedder handles its own device selection
    dim = getattr(e, "dim", 768)
    cfg = {
        "type": "paper",
        "impl": "PaperEmbedderSpecter2",
        "model_id": SPECTER2_ID,  # informational only
        "dim": int(dim),
        "device": getattr(e, "device", detect_device()),
    }
    return e, cfg


def make_chunk_embedder(
    devices: str | list[str] | None = "auto",
    *,
    workers: int = 1,
    force_devices: bool = False,
) -> tuple[Embedder, dict[str, Any]]:
    """
    Instantiate the chunk-level embedder (SBERT all-mpnet-base-v2).

    Parameters
    ----------
    devices : "auto" | list[str] | None
        Device spec (e.g., ["cuda:0","cuda:1"], ["cpu"], ["mps"]). "auto" picks sensibly.
    workers : int
        Cap on concurrent GPU workers when multi-GPU is used (pool spawns ≤ this).
    force_devices : bool
        If True, accept loose tokens without validation (e.g., "cuda:1").

    Returns
    -------
    (embedder, cfg)
        embedder : object implementing Embedder.encode(list[str]) -> np.ndarray
        cfg      : dict with metadata (model id, dim, devices, workers)
    """

    spec = ",".join(devices) if isinstance(devices, list) else (devices or "auto")
    dev_list = resolve_embed_devices(spec, force=force_devices)
    e = ChunkEmbedderSBERT(devices=dev_list, workers=workers)
    dim = getattr(e, "dim", 768)
    cfg = {
        "type": "chunk",
        "impl": "ChunkEmbedderSBERT",
        "model_id": SBERT_ID,
        "dim": int(dim),
        "devices": list(dev_list),
        "workers": int(workers),
    }
    return e, cfg


__all__ = ["make_paper_embedder", "make_chunk_embedder"]
