# litkit/db/sharding.py
"""Shard database management for distributed litkit builds."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Shard Path Utilities
# ---------------------------------------------------------------------------


def shard_db_path(sqlite_dir: Path, shard_id: int) -> Path:
    """Return path to shard-specific SQLite database for producer mode.
    
    Args:
        sqlite_dir: Directory containing SQLite databases
        shard_id: Shard identifier
    
    Returns:
        Path to the shard database file
    """
    return sqlite_dir / f"litkit_shard_{shard_id}.sqlite3"


def list_shard_dbs(sqlite_dir: Path) -> list[Path]:
    """List all shard SQLite databases in the sqlite directory.
    
    Args:
        sqlite_dir: Directory to search for shard databases
    
    Returns:
        List of paths to shard database files, sorted by name
    """
    if not sqlite_dir.exists():
        return []
    
    shards = sorted(sqlite_dir.glob("litkit_shard_*.sqlite3"))
    return shards


# ---------------------------------------------------------------------------
# Shard Merging
# ---------------------------------------------------------------------------


def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def merge_shard_databases(
    main_conn: sqlite3.Connection,
    sqlite_dir: Path,
    *,
    delete_after_merge: bool = True,
) -> dict[str, int]:
    """Merge all per-shard SQLite databases into the main database.
    
    Uses bulk SQL operations for efficient merging. Papers and chunks from
    shards are inserted using INSERT OR IGNORE to handle duplicates safely.
    
    Args:
        main_conn: Connection to the main database
        sqlite_dir: Directory containing shard databases
        delete_after_merge: If True, delete shard databases after merging
    
    Returns:
        Dictionary with merge statistics:
        - shards: Number of shards merged
        - papers: Number of papers merged
        - chunks: Number of chunks merged
        - files: Number of files merged
    """
    shard_dbs = list_shard_dbs(sqlite_dir)
    if not shard_dbs:
        return {"shards": 0, "papers": 0, "chunks": 0, "files": 0}
    
    stats = {"shards": 0, "papers": 0, "chunks": 0, "files": 0}
    cur = main_conn.cursor()
    
    for shard_path in shard_dbs:
        _eprint(f"[merge] Merging shard database: {shard_path.name}")
        
        # Attach shard database
        shard_alias = f"shard_{shard_path.stem}"
        cur.execute(f"ATTACH DATABASE ? AS {shard_alias}", (str(shard_path),))
        
        try:
            # Merge papers (INSERT OR IGNORE to handle duplicates)
            cur.execute(f"""
                INSERT OR IGNORE INTO papers 
                    (doc_id, title, abstract, authors, year, src_file, in_index)
                SELECT doc_id, title, abstract, authors, year, src_file, in_index
                FROM {shard_alias}.papers
            """)
            papers_merged = cur.rowcount
            stats["papers"] += papers_merged
            
            # Build doc_id to main paper_id mapping for chunk merging
            cur.execute(f"""
                INSERT OR IGNORE INTO chunks (paper_id, ord, text, in_index)
                SELECT 
                    main_papers.id,
                    shard_chunks.ord,
                    shard_chunks.text,
                    shard_chunks.in_index
                FROM {shard_alias}.chunks AS shard_chunks
                JOIN {shard_alias}.papers AS shard_papers 
                    ON shard_chunks.paper_id = shard_papers.id
                JOIN papers AS main_papers 
                    ON main_papers.doc_id = shard_papers.doc_id
            """)
            chunks_merged = cur.rowcount
            stats["chunks"] += chunks_merged
            
            # Merge files table
            cur.execute(f"""
                INSERT OR REPLACE INTO files (path, size, mtime_ns, paper_id)
                SELECT 
                    shard_files.path,
                    shard_files.size,
                    shard_files.mtime_ns,
                    main_papers.id
                FROM {shard_alias}.files AS shard_files
                LEFT JOIN {shard_alias}.papers AS shard_papers 
                    ON shard_files.paper_id = shard_papers.id
                LEFT JOIN papers AS main_papers 
                    ON main_papers.doc_id = shard_papers.doc_id
            """)
            files_merged = cur.rowcount
            stats["files"] += files_merged
            
            main_conn.commit()
            stats["shards"] += 1
            
            _eprint(
                f"[merge] Shard {shard_path.name}: "
                f"{papers_merged} papers, {chunks_merged} chunks, "
                f"{files_merged} files"
            )
            
        finally:
            # Detach shard database
            cur.execute(f"DETACH DATABASE {shard_alias}")
        
        # Delete shard after successful merge
        if delete_after_merge:
            try:
                os.remove(shard_path)
                _eprint(f"[merge] Deleted shard: {shard_path.name}")
            except OSError as e:
                _eprint(f"[merge] WARNING: Could not delete {shard_path}: {e}")
    
    _eprint(
        f"[merge] Complete: {stats['shards']} shards, "
        f"{stats['papers']} papers, {stats['chunks']} chunks"
    )
    return stats
