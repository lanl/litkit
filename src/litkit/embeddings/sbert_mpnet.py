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

import atexit
import multiprocessing as mp
import signal
import threading
from pathlib import Path

import numpy as np

from .base import inline_progress_renderer as _inline_progress_renderer
from .devices import detect_device
from .hf_local import local_snapshot_dir

# Model id used by SentenceTransformers
SBERT_ID = "sentence-transformers/all-mpnet-base-v2"

__all__ = ["ChunkEmbedderSBERT", "SBERT_ID"]


class EmbeddingPool:
    """
    Lightweight multi-GPU pool for SentenceTransformers encode(). One process per device.

    Notes
    -----
    - Uses spawn() context for CUDA safety.
    - No sockets/NCCL; suitable for air-gapped/HPC environments.
    - Child workers perform encode() with normalize_embeddings=True and return float32.
    """

    def __init__(self, model_path: Path, devices: list[str]):
        self.model_path = str(model_path)
        self.devices = list(devices)
        self.ctx = mp.get_context("spawn")
        self.q_in = self.ctx.Queue()
        self.q_out = self.ctx.Queue()
        self.workers: list[mp.Process] = []
        self._closed = False
        self._prev_signals: tuple | None = None

        for rank, dev in enumerate(self.devices):
            p = self.ctx.Process(
                target=self._worker_main,
                args=(rank, dev, self.model_path, self.q_in, self.q_out),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        atexit.register(self.close)
        if threading.current_thread() is threading.main_thread():
            try:
                prev_int = signal.getsignal(signal.SIGINT)
                prev_term = signal.getsignal(signal.SIGTERM)
                signal.signal(signal.SIGINT, self._handle_signal)
                signal.signal(signal.SIGTERM, self._handle_signal)
                self._prev_signals = (prev_int, prev_term)
            except Exception:
                pass

    @staticmethod
    def _worker_main(rank: int, device: str, model_path: str, q_in, q_out):
        # Import inside worker to avoid CUDA init in parent.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_path, device=device)
        while True:
            task = q_in.get()
            if task is None:
                break
            task_id, texts, bs = task
            try:
                arr = model.encode(
                    texts,
                    batch_size=bs,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")
                q_out.put((task_id, arr))
            except Exception as e:  # propagate error to parent
                q_out.put((task_id, e))

        try:
            q_out.close()
            q_in.close()
        except Exception:
            pass

    def _handle_signal(self, signum, _frame):
        # Fast, clean shutdown on Ctrl-C/TERM
        try:
            self.close(force=True)
        finally:
            raise SystemExit(1)  # exit parent quickly

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Distribute encode() across devices and stitch results (float32, L2-normalized)."""
        if not texts:
            return np.zeros((0, 768), dtype="float32")

        # Partition by index to preserve original order.
        n = len(texts)
        splits = np.array_split(np.arange(n), max(1, len(self.devices)))
        submitted: list[int] = []
        for tid, idxs in enumerate(splits):
            if idxs.size == 0:
                continue
            shard = [texts[i] for i in idxs]
            self.q_in.put((tid, shard, int(batch_size)))
            submitted.append(tid)

        results: dict[int, np.ndarray | Exception] = {}
        for _ in submitted:
            tid, payload = self.q_out.get()
            results[tid] = payload

        # Raise the first error (after draining).
        for tid in submitted:
            if isinstance(results[tid], Exception):
                # best effort drain
                for t2 in submitted:
                    if t2 in results:
                        continue
                    try:
                        self.q_out.get(timeout=0.05)
                    except Exception:
                        pass
                raise results[tid]  # type: ignore[misc]

        out: list[np.ndarray] = []
        for tid, idxs in enumerate(splits):
            if idxs.size == 0:
                continue
            out.append(results[tid])  # type: ignore[arg-type]
        return np.vstack(out) if out else np.zeros((0, 768), dtype="float32")

    def close(self, force: bool = False):
        if self._closed:
            return
        self._closed = True
        try:
            for _ in self.workers:
                try:
                    self.q_in.put(None)
                except Exception:
                    pass
            if force:
                for p in self.workers:
                    if p.is_alive():
                        p.terminate()
            for p in self.workers:
                p.join(timeout=2.0)
        finally:
            try:
                self.q_in.close()
                self.q_out.close()
            except Exception:
                pass
            if self._prev_signals and threading.current_thread() is threading.main_thread():
                try:
                    signal.signal(signal.SIGINT, self._prev_signals[0])
                    signal.signal(signal.SIGTERM, self._prev_signals[1])
                except Exception:
                    pass


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
