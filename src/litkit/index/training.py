# litkit/index/training.py
"""IVF-PQ index training for litkit."""

from __future__ import annotations

import random
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from litkit.index.constants import PQ_BITS
from litkit.index.factory import ivfpq_index, flat_ip_index, safe_pq_m


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


@dataclass
class TrainingConfig:
    """Configuration for IVF-PQ training."""
    
    dim: int
    """Embedding dimension."""
    
    target_samples: int = 100_000
    """Target number of training samples."""
    
    ivf_nlist: int = 16384
    """Number of IVF clusters (Voronoi cells)."""
    
    pq_m: int = 64
    """Number of PQ subquantizers."""
    
    pq_bits: int = PQ_BITS
    """Bits per subquantizer (usually 8)."""
    
    seed: int = 314159
    """Random seed for sampling."""
    
    batch_size: int = 512
    """Batch size for embedding training texts."""


@dataclass  
class TrainingResult:
    """Result of IVF-PQ training."""
    
    index: faiss.Index
    """Trained FAISS index (wrapped in IndexIDMap2)."""
    
    actual_nlist: int
    """Actual nlist used (may be reduced if insufficient samples)."""
    
    actual_m: int
    """Actual PQ m used (may differ from requested if dim not divisible)."""
    
    n_samples: int
    """Number of training samples used."""
    
    fell_back_to_flat: bool = False
    """True if training fell back to flat index due to insufficient samples."""


def compute_safe_ivfpq_params(
    n_samples: int,
    dim: int,
    requested_nlist: int,
    requested_m: int,
    pq_bits: int = PQ_BITS,
) -> tuple[int, int, bool]:
    """Compute safe IVF-PQ parameters given available samples.
    
    FAISS requires:
    - For IVF: n_samples >= nlist (at least 1 sample per centroid)
    - For PQ: n_samples >= 39 * 2^bits (FAISS guidance: ~39k for 8-bit PQ)
    - dim must be divisible by m
    
    Args:
        n_samples: Number of available training samples
        dim: Embedding dimension
        requested_nlist: Desired number of IVF clusters
        requested_m: Desired PQ subquantizer count
        pq_bits: Bits per subquantizer
    
    Returns:
        Tuple of (safe_nlist, safe_m, use_flat) where:
        - safe_nlist: Adjusted nlist value
        - safe_m: Adjusted m value (divisor of dim)
        - use_flat: True if should fall back to flat index
    """
    # PQ minimum samples: ~39 * codebook size
    k = 1 << pq_bits  # 256 for 8 bits
    min_for_pq = 39 * k  # ~10k for 8-bit PQ
    
    if n_samples < min_for_pq:
        # Not enough for any PQ training
        return 0, 0, True
    
    # Ensure m divides dim
    safe_m = safe_pq_m(dim, requested_m)
    
    # Adjust nlist based on samples
    # Rule of thumb: nlist should be <= n_samples / 39
    max_nlist = n_samples // 39
    safe_nlist = min(requested_nlist, max_nlist)
    safe_nlist = max(1, safe_nlist)  # At least 1
    
    # Final check: need at least nlist samples
    if n_samples < safe_nlist:
        safe_nlist = n_samples
    
    return safe_nlist, safe_m, False


def gather_training_samples(
    text_iterator: Iterator[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    target_samples: int,
    batch_size: int = 512,
    seed: int = 314159,
) -> np.ndarray:
    """Gather training samples by embedding texts.
    
    Samples texts via reservoir sampling to get a representative subset,
    then embeds them in batches.
    
    Args:
        text_iterator: Iterator yielding text strings to embed
        embed_fn: Function that embeds a batch of texts -> (n, dim) array
        target_samples: Maximum number of samples to collect
        batch_size: Batch size for embedding
        seed: Random seed for reservoir sampling
    
    Returns:
        Training embeddings as float32 array of shape (n, dim)
    """
    rng = random.Random(seed)
    
    # Reservoir sampling: collect up to target_samples texts
    reservoir: list[str] = []
    seen = 0
    
    for text in text_iterator:
        seen += 1
        if len(reservoir) < target_samples:
            reservoir.append(text)
        else:
            # Replace with decreasing probability
            j = rng.randint(0, seen - 1)
            if j < target_samples:
                reservoir[j] = text
    
    if not reservoir:
        return np.array([], dtype="float32").reshape(0, 0)
    
    _eprint(f"[train] Sampled {len(reservoir)} texts from {seen} total")
    
    # Embed in batches
    embeddings: list[np.ndarray] = []
    
    for i in range(0, len(reservoir), batch_size):
        batch = reservoir[i:i + batch_size]
        X = embed_fn(batch)
        embeddings.append(X)
        
        if (i + batch_size) % 5000 < batch_size:
            _eprint(f"[train] Embedded {min(i + batch_size, len(reservoir))}/{len(reservoir)}")
    
    X_train = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(X_train)
    
    return X_train


def train_ivfpq_index(
    X_train: np.ndarray,
    config: TrainingConfig,
) -> TrainingResult:
    """Train an IVF-PQ index on the provided training vectors.
    
    Automatically adjusts nlist and m based on available samples.
    Falls back to flat index if insufficient training data.
    
    Args:
        X_train: Training vectors, shape (n, dim), should be L2-normalized
        config: Training configuration
    
    Returns:
        TrainingResult with the trained index and parameters used
    """
    n_samples = X_train.shape[0]
    dim = X_train.shape[1] if n_samples > 0 else config.dim
    
    _eprint(
        f"[train] Training IVF-PQ: {n_samples} samples, dim={dim}, "
        f"target nlist={config.ivf_nlist}, m={config.pq_m}"
    )
    
    # Compute safe parameters
    safe_nlist, safe_m, use_flat = compute_safe_ivfpq_params(
        n_samples, dim, config.ivf_nlist, config.pq_m, config.pq_bits
    )
    
    if use_flat or n_samples == 0:
        _eprint(f"[train] Insufficient samples ({n_samples}); using flat index")
        base = flat_ip_index(dim)
        index = faiss.IndexIDMap2(base)
        return TrainingResult(
            index=index,
            actual_nlist=0,
            actual_m=0,
            n_samples=n_samples,
            fell_back_to_flat=True,
        )
    
    if safe_nlist != config.ivf_nlist:
        _eprint(f"[train] Reduced nlist: {config.ivf_nlist} -> {safe_nlist}")
    if safe_m != config.pq_m:
        _eprint(f"[train] Adjusted m: {config.pq_m} -> {safe_m}")
    
    # Create and train the index
    idx = ivfpq_index(dim, nlist=safe_nlist, m=safe_m, bits=config.pq_bits)
    
    # Suppress verbose output during training
    try:
        idx.verbose = False
        if hasattr(faiss, "cvar") and hasattr(faiss.cvar, "verbose"):
            faiss.cvar.verbose = False
    except Exception:
        pass
    
    _eprint(f"[train] Training with nlist={safe_nlist}, m={safe_m}...")
    
    try:
        idx.train(X_train)
    except Exception as e:
        _eprint(f"[train] Training failed: {e}; falling back to flat index")
        base = flat_ip_index(dim)
        index = faiss.IndexIDMap2(base)
        return TrainingResult(
            index=index,
            actual_nlist=0,
            actual_m=0,
            n_samples=n_samples,
            fell_back_to_flat=True,
        )
    
    _eprint("[train] Training complete")
    
    # Wrap in IndexIDMap2 for external ID support
    index = faiss.IndexIDMap2(idx)
    
    return TrainingResult(
        index=index,
        actual_nlist=safe_nlist,
        actual_m=safe_m,
        n_samples=n_samples,
        fell_back_to_flat=False,
    )
