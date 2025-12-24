# litkit/db/queries.py
"""Query helpers for papers, chunks, and files in litkit."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

# ---------------------------------------------------------------------------
# File Processing Queries
# ---------------------------------------------------------------------------


def already_processed(
    cur: sqlite3.Cursor,
    fpath: str,
    st,
) -> bool:
    """Check if file at `fpath` with current stat `st` was already ingested.
    
    Returns True if the file exists in the files table with matching
    size and mtime, indicating it doesn't need reprocessing.
    
    Args:
        cur: Database cursor
        fpath: File path to check
        st: SimpleNamespace or os.stat() result with st_size and st_mtime attributes
    
    Returns:
        True if file was already processed with same size/mtime
    """
    cur.execute(
        "SELECT size, mtime FROM files WHERE path = ?",
        (fpath,)
    )
    row = cur.fetchone()
    if row is None:
        return False
    size, mtime = row
    return size == st.st_size and abs(mtime - st.st_mtime) < 1e-6


def register_file(
    cur: sqlite3.Cursor,
    fpath: str,
    paper_id: int,
    st,
) -> None:
    """Insert/replace the (path, size, mtime, paper_id) record in files table.
    
    Args:
        cur: Database cursor
        fpath: File path
        paper_id: Associated paper ID
        st: SimpleNamespace or os.stat() result with st_size and st_mtime
    """
    cur.execute(
        "INSERT OR REPLACE INTO files(path, size, mtime, paper_id) "
        "VALUES (?, ?, ?, ?)",
        (fpath, st.st_size, st.st_mtime, paper_id),
    )


# ---------------------------------------------------------------------------
# Paper/Chunk ID Map Preloading
# ---------------------------------------------------------------------------


def preload_paper_id_map(conn: sqlite3.Connection) -> dict[str, int]:
    """Preload doc_id → paper_id mapping for O(1) lookups.
    
    Used during segment ingestion to quickly map document IDs to
    database primary keys without repeated queries.
    
    Args:
        conn: Database connection
    
    Returns:
        Dictionary mapping doc_id strings to paper_id integers
    """
    cur = conn.cursor()
    cur.execute("SELECT doc_id, id FROM papers")
    return {row[0]: row[1] for row in cur.fetchall()}


def preload_chunk_id_map(conn: sqlite3.Connection) -> dict[tuple[int, int], int]:
    """Preload (paper_id, ord) → chunk_id mapping for O(1) lookups.
    
    Used during segment ingestion to quickly map paper/chunk ordinals
    to database primary keys without repeated queries.
    
    Args:
        conn: Database connection
    
    Returns:
        Dictionary mapping (paper_id, ord) tuples to chunk_id integers
    """
    cur = conn.cursor()
    cur.execute("SELECT paper_id, ord, id FROM chunks")
    return {(row[0], row[1]): row[2] for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Chunk to Paper Resolution
# ---------------------------------------------------------------------------


def chunk_ids_to_paper_ids(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
) -> dict[int, int]:
    """Resolve chunk_id → paper_id mapping for a given set of chunk IDs.
    
    Batched to avoid SQLite variable limits (max ~999 params per query).
    
    Args:
        conn: Database connection
        chunk_ids: List of chunk IDs to resolve
    
    Returns:
        Dictionary mapping chunk_id to paper_id
    """
    if not chunk_ids:
        return {}
    
    result = {}
    cur = conn.cursor()
    
    # Batch in chunks of 500 to stay under SQLite limit
    batch_size = 500
    for i in range(0, len(chunk_ids), batch_size):
        batch = chunk_ids[i:i + batch_size]
        placeholders = ",".join("?" * len(batch))
        cur.execute(
            f"SELECT id, paper_id FROM chunks WHERE id IN ({placeholders})",
            batch,
        )
        for row in cur.fetchall():
            result[row[0]] = row[1]
    
    return result
