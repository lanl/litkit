# src/litkit/embeddings/base.py

from __future__ import annotations

import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# -------------------- Public config & typing --------------------


@dataclass(slots=True)
class EmbedderConfig:
    """
    Minimal, backend-agnostic embedder configuration.
    """

    model_id: str
    dim: int = 768
    max_length: int = 512
    batch_size_cpu: int = 8
    batch_size_accel: int = 16
    normalize: bool = True  # return L2-normalized vectors
    return_dtype: str = "float32"  # always float32 for FAISS; keep configurable
    device: str | None = None  # e.g. "cpu", "mps", "cuda:0"
    offline: bool = True  # HF offline / air-gapped default

    def effective_batch_size(self, device: str | None, override: int | None) -> int:
        if override is not None:
            return int(override)
        d = (device or "").lower()
        is_accel = d.startswith("cuda") or d.startswith("mps")
        return self.batch_size_accel if is_accel else self.batch_size_cpu


class Embedder(Protocol):
    """
    Structural type that any embedder must satisfy.
    """

    dim: int

    def encode(
        self,
        texts: Sequence[str],
        progress_label: str | None = None,
        batch_size: int | None = None,
        progress_done_summary: bool = True,
    ) -> np.ndarray: ...


class BaseEmbedder(ABC):
    """
    Optional ABC base with a shared config and a no-op close().
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.dim = int(config.dim)

    @abstractmethod
    def encode(
        self,
        texts: Sequence[str],
        progress_label: str | None = None,
        batch_size: int | None = None,
        progress_done_summary: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        # Backends that spawn pools/processes/handles can override.
        return


# -------------------- Small, backend-agnostic utils --------------------


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    In-place safe L2 normalization to unit vectors (float32 out).
    """
    if x.dtype != np.float32:
        x = x.astype("float32", copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    x /= norms
    return x


def stack_or_empty(chunks: list[np.ndarray], dim: int, dtype: str = "float32") -> np.ndarray:
    """
    Vstack a list of 2D arrays or return (0, dim).
    """
    if not chunks:
        return np.zeros((0, int(dim)), dtype=dtype)
    return np.vstack(chunks).astype(dtype, copy=False)


# -------------------- Minimal, dependency-free progress line --------------------
# Shared single-line renderer used by embedders and the CLI.

PROGRESS_MODE = os.environ.get("LITKIT_PROGRESS_MODE", "auto").lower()
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_LAST_LEN = 0


def progress_is_append() -> bool:
    # "tty" => rewrite in place; "append" => always append; "auto" => TTY-aware
    if PROGRESS_MODE == "tty":
        return False
    if PROGRESS_MODE == "append":
        return True
    try:
        if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
            return False
    except Exception:
        pass
    try:
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            return False
    except Exception:
        pass
    return True


def progress_write(s: str, stream) -> None:
    global _PROGRESS_LAST_LEN
    with _PROGRESS_LOCK:
        if progress_is_append():
            stream.write(s + "\n")
            stream.flush()
            _PROGRESS_LAST_LEN = 0
        else:
            pad = max(0, _PROGRESS_LAST_LEN - len(s))
            stream.write("\r" + s + (" " * pad))
            stream.flush()
            _PROGRESS_LAST_LEN = len(s)


def progress_newline(stream) -> None:
    global _PROGRESS_LAST_LEN
    with _PROGRESS_LOCK:
        if not progress_is_append():
            stream.write("\n")
            stream.flush()
        _PROGRESS_LAST_LEN = 0


def inline_progress_renderer(label: str, total: int, stream=None, done_summary: bool = True):
    """
    Return a closure `render(done, final=False)` that updates a single progress line.
    On `final=True`, always prints a newline and, if enabled, a one-line [done] summary.
    """
    stream = stream or sys.stderr
    prev_len = 0
    t0 = time.time()

    def _render(done: int, final: bool = False):
        nonlocal prev_len
        elapsed = max(1e-3, time.time() - t0)
        rate = done / elapsed if elapsed > 0 else 0.0
        s = f"[progress] {label}: {done}/{total}"
        with _PROGRESS_LOCK:
            if progress_is_append():
                stream.write(s + "\n")
                stream.flush()
                prev_len = 0
                if final:
                    stream.write(
                        f"[done] {label}: completed in {int(elapsed)}s — "
                        f"{done}/{total}  ({(100.0*done/max(1,total)):.1f}%)  {rate:.1f}/s\n"
                    )
                    stream.flush()
            else:
                pad = max(0, prev_len - len(s))
                stream.write("\r" + s + (" " * pad))
                stream.flush()
                prev_len = len(s)
                if final:
                    stream.write("\n")
                    if done_summary:
                        stream.write(
                            f"[done] {label}: completed in {int(elapsed)}s — "
                            f"{done}/{total}  ({(100.0*done/max(1,total)):.1f}%)  {rate:.1f}/s\n"
                        )
                        stream.flush()

    return _render
