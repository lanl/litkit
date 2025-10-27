# /src/litkit/embeddings/sbert_mpnet.py

"""
SBERT (all-mpnet-base-v2) chunk embedder with optional multi-GPU worker pool.

Design goals
------------
- Fully offline-capable: loads model from local HF snapshot.
- Returns L2-normalized float32 embeddings (cosine ≡ inner product).
- Drop-in replacement for the inlined class that lived in cli.py.
"""

from __future__ import annotations

import numpy as np

from .base import inline_progress_renderer as _inline_progress_renderer
from .devices import detect_device
from .hf_local import local_snapshot_dir
from .pool import EmbeddingPool

# Model id used by SentenceTransformers
SBERT_ID = "sentence-transformers/all-mpnet-base-v2"


class ChunkEmbedderSBERT:
    """
    SentenceTransformers all-mpnet-base-v2, offline; returns L2-normalized float32 vectors.

    Parameters
    ----------
    devices : list[str] | None
        Devices like ["cuda:0","cuda:1"], ["cpu"], ["mps"]. Defaults to auto-detect.
    workers : int
        Max number of concurrent GPU workers (caps device fanout). Ignored for single device.

    Notes
    -----
    - Snapshot path is resolved via litkit.embeddings.hf_local.local_snapshot_dir(SBERT_ID).
    - If multiple CUDA devices requested, a background EmbeddingPool is used.
    """

    def __init__(self, devices: list[str] | None = None, workers: int = 1):
        from sentence_transformers import SentenceTransformer  # import lazily

        self.devices = list(devices) if devices else [detect_device()]
        self.workers = int(workers) if workers is not None else 1

        # Cap fanout by workers, if set > 0
        if self.workers > 0 and len(self.devices) > self.workers:
            self.devices = self.devices[: self.workers]

        model_path = local_snapshot_dir(SBERT_ID)  # must exist for offline runs

        # Multi-GPU only when >1 CUDA device
        multi_gpu = len(self.devices) > 1 and self.devices[0].startswith("cuda")
        if multi_gpu:
            self.pool = EmbeddingPool(model_path, self.devices)
            self.model = None
        else:
            self.pool = None
            try:
                self.model = SentenceTransformer(str(model_path), device=self.devices[0])
            except Exception as e:
                raise FileNotFoundError(
                    f"[offline] SBERT snapshot missing at {model_path}. "
                    f"Place '{SBERT_ID}' under $HF_HOME/hub/ (or set HF_HOME)."
                ) from e

        # Probe dimension robustly
        self.dim = 768
        try:
            _probe = self.encode(
                ["__dim_probe__"], progress_label=None, progress_done_summary=False
            )
            if isinstance(_probe, np.ndarray) and _probe.ndim == 2 and _probe.size:
                self.dim = int(_probe.shape[1])
        except Exception:
            pass

    def encode(
        self,
        texts: list[str],
        progress_label: str | None = "Embedding corpus chunks",
        batch_size: int | None = None,
        progress_done_summary: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized float32 embeddings of shape (N, dim).
        """
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        if self.pool is not None:
            # Multi-GPU path
            render = _inline_progress_renderer(
                progress_label or "Embedding corpus chunks (multi-GPU)",
                len(texts),
                done_summary=progress_done_summary,
            )
            render(0)
            arr = self.pool.encode(texts, batch_size or 64)
            render(len(texts), final=True)
            return arr

        # Single device path
        bs = int(batch_size or (64 if self.devices[0].startswith("cuda") else 16))
        total = len(texts)
        render = (
            _inline_progress_renderer(
                progress_label or "Embedding corpus chunks",
                total,
                done_summary=progress_done_summary,
            )
            if progress_label is not None
            else None
        )

        if render:
            render(0)

        out: list[np.ndarray] = []
        for i in range(0, total, bs):
            chunk = texts[i : i + bs]
            arr = self.model.encode(  # type: ignore[union-attr]
                chunk,
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            out.append(arr)
            if render:
                render(min(total, i + len(chunk)))

        if render:
            render(total, final=True)

        return np.vstack(out) if out else np.zeros((0, self.dim), dtype="float32")

    def close(self):
        if self.pool is not None:
            self.pool.close()


__all__ = ["ChunkEmbedderSBERT", "SBERT_ID"]
