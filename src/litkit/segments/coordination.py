# litkit/segments/coordination.py
"""Producer/Consumer coordination for multi-node builds."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from litkit.segments.constants import PRODUCER_DONE_PATTERN


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


class ProducerCoordinator:
    """Manages producer completion signaling for multi-node coordination.
    
    Each producer writes a `.done` file when it finishes processing.
    The consumer waits for all producers to signal completion.
    """
    
    def __init__(self, seg_dir: Path, shard_id: int, num_shards: int):
        """Initialize the producer coordinator.
        
        Args:
            seg_dir: Segment directory
            shard_id: This producer's shard ID
            num_shards: Total number of producer shards
        """
        self.seg_dir = Path(seg_dir)
        self.shard_id = shard_id
        self.num_shards = num_shards
        self._done_file = self.seg_dir / PRODUCER_DONE_PATTERN.format(
            shard_id=shard_id
        )
    
    def mark_complete(self) -> None:
        """Signal that this producer has completed processing."""
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        
        # Write done file with metadata
        with open(self._done_file, "w") as f:
            f.write(f"shard_id={self.shard_id}\n")
            f.write(f"pid={os.getpid()}\n")
            f.write(f"timestamp={time.time()}\n")
        
        _eprint(f"[producer] Shard {self.shard_id} marked complete")
    
    def is_complete(self) -> bool:
        """Check if this producer has marked completion."""
        return self._done_file.exists()


class ConsumerCoordinator:
    """Manages consumer polling for producer completion.
    
    Waits for all producers to signal completion before ingesting
    segments into the main database and FAISS indices.
    """
    
    def __init__(
        self,
        seg_dir: Path,
        num_shards: int,
        poll_interval: float = 30.0,
    ):
        """Initialize the consumer coordinator.
        
        Args:
            seg_dir: Segment directory
            num_shards: Expected number of producer shards
            poll_interval: Seconds between completion checks
        """
        self.seg_dir = Path(seg_dir)
        self.num_shards = num_shards
        self.poll_interval = poll_interval
    
    def _done_file(self, shard_id: int) -> Path:
        """Get the done file path for a shard."""
        return self.seg_dir / PRODUCER_DONE_PATTERN.format(shard_id=shard_id)
    
    def completed_shards(self) -> list[int]:
        """Return list of shard IDs that have completed."""
        completed = []
        for shard_id in range(self.num_shards):
            if self._done_file(shard_id).exists():
                completed.append(shard_id)
        return completed
    
    def all_complete(self) -> bool:
        """Check if all producers have completed."""
        return len(self.completed_shards()) >= self.num_shards
    
    def wait_for_completion(
        self,
        timeout: float | None = None,
        progress_callback: callable = None,
        poll_interval: float | None = None,
    ) -> bool:
        """Wait for all producers to complete.
        
        Args:
            timeout: Maximum seconds to wait (None = wait forever)
            progress_callback: Optional function called with (completed, total)
            poll_interval: Seconds between checks (overrides instance default)
        
        Returns:
            True if all producers completed, False if timeout
        """
        interval = poll_interval if poll_interval is not None else self.poll_interval
        start = time.time()
        
        while True:
            completed = self.completed_shards()
            
            if progress_callback:
                progress_callback(len(completed), self.num_shards)
            
            if len(completed) >= self.num_shards:
                _eprint(
                    f"[consumer] All {self.num_shards} producers complete"
                )
                return True
            
            if timeout is not None and (time.time() - start) > timeout:
                _eprint(
                    f"[consumer] Timeout: {len(completed)}/{self.num_shards} "
                    "producers complete"
                )
                return False
            
            _eprint(
                f"[consumer] Waiting: {len(completed)}/{self.num_shards} "
                f"producers complete, polling in {interval}s"
            )
            time.sleep(interval)
    
    def cleanup_done_files(self) -> int:
        """Remove all producer done files after ingestion.
        
        Returns:
            Number of files removed
        """
        removed = 0
        for shard_id in range(self.num_shards):
            done_file = self._done_file(shard_id)
            if done_file.exists():
                try:
                    done_file.unlink()
                    removed += 1
                except OSError as e:
                    _eprint(f"[consumer] Failed to remove {done_file}: {e}")
        
        if removed:
            _eprint(f"[consumer] Cleaned up {removed} producer done files")
        return removed
