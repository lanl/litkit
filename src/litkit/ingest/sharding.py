# litkit/ingest/sharding.py
"""Shard assignment for multi-node tar processing in litkit."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def shard_filter(
    paths: Iterable[Path],
    shard_id: int,
    num_shards: int,
) -> Iterator[Path]:
    """Deterministically assign tar files to shards using size-aware bin-packing.
    
    This algorithm distributes tar files across shards to balance total bytes
    processed by each shard, not just file count. This is important because
    tar files can vary significantly in size.
    
    Algorithm:
    1. Stat each tar file from the manifest to get sizes
    2. Sort by size (largest first) for better packing
    3. Greedily assign each file to the shard with lowest current load
    4. Yield only files assigned to this shard
    
    Args:
        paths: Iterable of paths to tar files
        shard_id: This process's shard ID (0 to num_shards-1)
        num_shards: Total number of shards
    
    Yields:
        Paths assigned to this shard
    
    Raises:
        RuntimeError: If a tar file cannot be stat'd
    """
    if num_shards <= 1:
        # Single shard: return all paths
        yield from paths
        return
    
    # Collect paths with sizes
    path_sizes: list[tuple[Path, int]] = []
    for p in paths:
        try:
            size = p.stat().st_size
            path_sizes.append((p, size))
        except Exception as e:
            # Fail loudly if a manifest file is unreadable
            raise RuntimeError(f"Cannot stat tar file {p}: {e}") from e
    
    if not path_sizes:
        return
    
    # Sort by size descending (largest first for better packing)
    path_sizes.sort(key=lambda x: -x[1])
    
    # Greedy bin-packing: assign each file to shard with lowest load
    shard_loads = [0] * num_shards
    assignments: list[list[Path]] = [[] for _ in range(num_shards)]
    
    for path, size in path_sizes:
        # Find shard with minimum current load
        min_shard = min(range(num_shards), key=lambda s: shard_loads[s])
        shard_loads[min_shard] += size
        assignments[min_shard].append(path)
    
    # Report load balance
    total_size = sum(shard_loads)
    ideal_size = total_size / num_shards if num_shards > 0 else 0
    my_size = shard_loads[shard_id]
    imbalance = (my_size / ideal_size - 1) * 100 if ideal_size > 0 else 0
    
    _eprint(
        f"[shard] Shard {shard_id}/{num_shards}: "
        f"{len(assignments[shard_id])} files, "
        f"{my_size / (1024**3):.2f} GB "
        f"({imbalance:+.1f}% vs ideal)"
    )
    
    # Yield paths assigned to this shard
    yield from assignments[shard_id]
