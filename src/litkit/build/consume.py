# litkit/build/consume.py
"""Consumer-only mode for multi-node builds.

This module handles the consumer role in producer/consumer multi-node builds:
1. Poll for producer completion markers
2. Merge shard databases into main DB
3. Ingest embedding segments from producers
4. Save consolidated FAISS indices
"""

from __future__ import annotations

from pathlib import Path
from sqlite3 import Connection
from typing import TYPE_CHECKING, Callable

from litkit.db import (
    merge_shard_databases,
    flush_pending_marks,
)
from litkit.index import faiss_load, faiss_save_force
from litkit.segments import (
    ConsumerCoordinator,
    ingest_paper_segments,
    ingest_chunk_segments,
)
from litkit.progress import eprint

if TYPE_CHECKING:
    import faiss


def run_consume_only_mode(
    conn: Connection,
    *,
    seg_dir: Path,
    num_shards: int,
    paper_index_path: Path,
    chunk_index_path: Path,
    faiss_lock_path: Path,
    db_lock_path: Path,
    FileLock: type,
    poll_interval: int = 30,
    timeout: int = 36000,
    progress_callback: Callable[[int, int], None] | None = None,
) -> bool:
    """Run consumer-only mode: wait for producers, merge DBs, ingest segments.
    
    This is the consumer role in multi-node builds. It:
    1. Waits for all producer completion markers
    2. Merges shard-specific SQLite DBs into the main DB
    3. Ingests paper/chunk embedding segments into FAISS
    4. Saves consolidated indices
    
    Args:
        conn: Connection to the main SQLite database
        seg_dir: Directory containing embedding segments and completion markers
        num_shards: Total number of producer shards to wait for
        paper_index_path: Path to the paper FAISS index
        chunk_index_path: Path to the chunk FAISS index
        faiss_lock_path: Path to FAISS lock file
        db_lock_path: Path to DB lock file
        FileLock: FileLock class for locking (injected to avoid circular imports)
        poll_interval: Seconds between polling for producer completion
        timeout: Maximum seconds to wait for producers
        progress_callback: Optional callback(complete, total) for progress updates
    
    Returns:
        True if consume completed successfully, False if timed out
    """
    def default_progress(complete: int, total: int) -> None:
        eprint(f"[consumer] Progress: {complete}/{total} producers complete")
    
    cb = progress_callback or default_progress
    
    eprint("[consumer] Starting consume-only mode")
    
    # Load existing indices
    paper_index = faiss_load(paper_index_path)
    chunk_index = faiss_load(chunk_index_path)
    
    # Create coordinator and wait for producers
    consumer_coordinator = ConsumerCoordinator(seg_dir, num_shards)
    
    eprint("[consumer] Waiting for producers to complete...")
    
    completed = consumer_coordinator.wait_for_completion(
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=cb,
    )
    
    if not completed:
        eprint("[consumer] TIMEOUT waiting for producers; aborting consume-only run.")
        return False
    
    eprint("[consumer] All producers have completed")
    
    # Merge shard databases FIRST (before segment ingestion)
    # This populates the main DB so doc_id → paper_id resolution works
    eprint("[consumer] Merging shard databases...")
    merge_stats = merge_shard_databases(conn, delete_after_merge=True)
    if merge_stats["shards"] > 0:
        eprint(
            f"[consumer] Merged {merge_stats['shards']} shard DB(s): "
            f"{merge_stats['papers']} papers, {merge_stats['chunks']} chunks, "
            f"{merge_stats['files']} files"
        )
    
    # NOW ingest segments - main DB has all data for doc_id resolution
    eprint("[consumer] Ingesting embedding segments...")
    paper_index, p_added = ingest_paper_segments(
        conn, paper_index, seg_dir, faiss_lock_path, paper_index_path, db_lock_path,
        FileLock=FileLock
    )
    chunk_index, c_added = ingest_chunk_segments(
        conn, chunk_index, seg_dir, faiss_lock_path, chunk_index_path, db_lock_path,
        FileLock=FileLock
    )
    eprint(f"[consumer] Ingested {p_added} paper vectors and {c_added} chunk vectors")
    
    # Force save indices after ingestion
    with FileLock(faiss_lock_path):
        faiss_save_force(paper_index, paper_index_path)
        faiss_save_force(chunk_index, chunk_index_path)
    
    with FileLock(db_lock_path):
        flush_pending_marks(conn.cursor())
        conn.commit()
    
    eprint("[consumer] Consume-only mode completed")
    return True
