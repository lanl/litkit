# src/litkit/embeddings/pool.py
"""
EmbeddingPool: one SentenceTransformers worker per device (CUDA/CPU/MPS).
Spawn-based, no sockets/NCCL; safe for air-gapped/HPC.

FIXED: Uses per-worker input queues to prevent race conditions where
the faster-initializing worker grabs all tasks.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import signal
import threading
from pathlib import Path

import numpy as np


class EmbeddingPool:
    """
    Lightweight multi-device pool for SentenceTransformers encode().
    Each worker loads the same model snapshot on its assigned device.

    Parameters
    ----------
    model_path : Path
        Local snapshot path for the SentenceTransformers model.
    devices : list[str]
        e.g., ["cuda:0", "cuda:1"] or ["cpu"] or ["mps"].

    Notes
    -----
    - Uses spawn() for CUDA hygiene.
    - Workers return L2-normalized float32 arrays.
    - Each worker has its own input queue to prevent race conditions.
    """

    def __init__(self, model_path: Path, devices: list[str]):
        self.model_path = str(model_path)
        self.devices = list(devices)
        if not self.devices:
            raise ValueError("EmbeddingPool requires at least one device")

        self.ctx = mp.get_context("spawn")
        
        # FIX: Per-worker input queues instead of shared queue
        # This prevents the race condition where the faster-initializing
        # worker grabs all tasks before slower workers are ready
        self.q_ins: list[mp.Queue] = [self.ctx.Queue() for _ in self.devices]
        self.q_out = self.ctx.Queue()
        
        self.workers: list[mp.Process] = []
        self._closed = False
        self._prev_signals: tuple | None = None

        for rank, dev in enumerate(self.devices):
            p = self.ctx.Process(
                target=self._worker_main,
                args=(rank, dev, self.model_path, self.q_ins[rank], self.q_out),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        # graceful shutdown hooks
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
            except Exception as e:
                q_out.put((task_id, e))

        # best-effort cleanup
        try:
            q_out.close()
            q_in.close()
        except Exception:
            pass

    def _handle_signal(self, signum, _frame):
        try:
            self.close(force=True)
        finally:
            raise SystemExit(1)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Distribute encode() across devices and return stacked results (float32)."""
        if not texts:
            return np.zeros((0, 768), dtype="float32")

        n = len(texts)
        num_workers = len(self.devices)
        
        # Split work evenly across workers
        splits = np.array_split(np.arange(n), max(1, num_workers))
        
        # FIX: Send each split directly to its designated worker's queue
        submitted: list[int] = []
        for worker_id, idxs in enumerate(splits):
            if idxs.size == 0:
                continue
            shard = [texts[i] for i in idxs]
            # Each worker gets work on its own queue - no race!
            self.q_ins[worker_id].put((worker_id, shard, int(batch_size)))
            submitted.append(worker_id)

        results: dict[int, np.ndarray | Exception] = {}
        for _ in submitted:
            tid, payload = self.q_out.get()
            results[tid] = payload

        # propagate first error (after draining)
        for tid in submitted:
            if isinstance(results[tid], Exception):
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
            # Send shutdown signal to each worker's queue
            for q_in in self.q_ins:
                try:
                    q_in.put(None)
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
                for q_in in self.q_ins:
                    q_in.close()
                self.q_out.close()
            except Exception:
                pass
            if self._prev_signals and threading.current_thread() is threading.main_thread():
                try:
                    signal.signal(signal.SIGINT, self._prev_signals[0])
                    signal.signal(signal.SIGTERM, self._prev_signals[1])
                except Exception:
                    pass


__all__ = ["EmbeddingPool"]
