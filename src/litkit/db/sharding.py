# litkit/db/sharding.py
"""Shard database management for distributed litkit builds."""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path


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
    return sqlite_dir / f"litkit_shard_{shard_id:02d}.sqlite3"


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
    delete_after_merge: bool = True,
) -> dict[str, int]:
    """Merge all per-shard SQLite databases into the main database.
    
    Uses ATTACH DATABASE + INSERT...SELECT for efficient bulk merging.
    The merge is idempotent: duplicate papers (same doc_id) and chunks 
    (same paper_id+ord) are skipped using INSERT OR IGNORE.
    
    Note: This function discovers shard databases from SQLITE_DIR which
    should be derived from the main database path.
    
    Args:
        main_conn: Connection to the main litkit.sqlite3 database
        delete_after_merge: If True, delete shard DBs after successful merge
    
    Returns:
        Dict with merge statistics: {"papers": N, "chunks": M, "files": F, 
        "shards": S}
    """
    # Derive sqlite_dir from main database path
    # The main_conn is typically to sqlite_dir/litkit.sqlite3
    db_path = main_conn.execute("PRAGMA database_list").fetchone()[2]
    sqlite_dir = Path(db_path).parent if db_path else Path(".")
    
    shard_dbs = list_shard_dbs(sqlite_dir)
    if not shard_dbs:
        return {"papers": 0, "chunks": 0, "files": 0, "shards": 0}
    
    _eprint(
        f"[merge] Found {len(shard_dbs)} shard database(s) to merge "
        "(using bulk SQL)"
    )
    
    stats = {"papers": 0, "chunks": 0, "files": 0, "shards": 0}
    main_cur = main_conn.cursor()
    
    for shard_idx, shard_db in enumerate(shard_dbs):
        _eprint(f"[merge] Processing {shard_db.name}...")
        t0 = time.time()
        
        # Use unique alias per shard to avoid "database already in use"
        shard_alias = f"shard_db_{shard_idx}"
        # Use unique temp table name per shard for safety
        temp_map_table = f"_shard_paper_map_{shard_idx}"
        
        try:
            # ATTACH the shard database for bulk operations
            main_cur.execute(
                f"ATTACH DATABASE ? AS {shard_alias}", (str(shard_db),)
            )
            
            try:
                # Count rows before merge for statistics
                papers_before = main_cur.execute(
                    "SELECT COUNT(*) FROM papers"
                ).fetchone()[0]
                chunks_before = main_cur.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]
                files_before = main_cur.execute(
                    "SELECT COUNT(*) FROM files"
                ).fetchone()[0]
                
                # 1) BULK MERGE PAPERS
                # Insert papers that don't already exist (by doc_id)
                main_cur.execute(f"""
                    INSERT OR IGNORE INTO papers(
                        doc_id, pmid, pmcid, title, abstract, in_index
                    )
                    SELECT s.doc_id, s.pmid, s.pmcid, s.title, s.abstract, 0
                    FROM {shard_alias}.papers s
                    WHERE s.doc_id IS NOT NULL 
                      AND s.doc_id != ''
                      AND NOT EXISTS (
                          SELECT 1 FROM papers m WHERE m.doc_id = s.doc_id
                      )
                """)
                
                # 2) BULK MERGE CHUNKS
                # Create temp table to map shard paper_id -> main paper_id
                main_cur.execute(f"DROP TABLE IF EXISTS {temp_map_table}")
                main_cur.execute(f"""
                    CREATE TEMP TABLE {temp_map_table} AS
                    SELECT s.id AS shard_pid, m.id AS main_pid
                    FROM {shard_alias}.papers s
                    JOIN papers m ON m.doc_id = s.doc_id
                    WHERE s.doc_id IS NOT NULL AND s.doc_id != ''
                """)
                
                # Insert chunks with remapped paper_id
                main_cur.execute(f"""
                    INSERT OR IGNORE INTO chunks(paper_id, ord, text, in_index)
                    SELECT pm.main_pid, sc.ord, sc.text, 0
                    FROM {shard_alias}.chunks sc
                    JOIN {temp_map_table} pm ON pm.shard_pid = sc.paper_id
                    WHERE NOT EXISTS (
                        SELECT 1 FROM chunks mc 
                        WHERE mc.paper_id = pm.main_pid AND mc.ord = sc.ord
                    )
                """)
                
                # 3) BULK MERGE FILES
                # Insert/replace files with remapped paper_id
                main_cur.execute(f"""
                    INSERT OR REPLACE INTO files(path, size, mtime, paper_id)
                    SELECT sf.path, sf.size, sf.mtime, pm.main_pid
                    FROM {shard_alias}.files sf
                    JOIN {temp_map_table} pm ON pm.shard_pid = sf.paper_id
                """)
                
                # Drop temp table immediately after use
                main_cur.execute(f"DROP TABLE IF EXISTS {temp_map_table}")
                
                # Count rows after merge for statistics
                papers_after = main_cur.execute(
                    "SELECT COUNT(*) FROM papers"
                ).fetchone()[0]
                chunks_after = main_cur.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]
                files_after = main_cur.execute(
                    "SELECT COUNT(*) FROM files"
                ).fetchone()[0]
                
                papers_merged = papers_after - papers_before
                chunks_merged = chunks_after - chunks_before
                files_merged = files_after - files_before
                
                stats["papers"] += papers_merged
                stats["chunks"] += chunks_merged
                stats["files"] += files_merged
                stats["shards"] += 1
                
                elapsed = time.time() - t0
                _eprint(
                    f"[merge] {shard_db.name}: +{papers_merged} papers, "
                    f"+{chunks_merged} chunks, +{files_merged} files "
                    f"({elapsed:.1f}s)"
                )
                
            finally:
                # Always detach the shard database
                try:
                    main_cur.execute(f"DETACH DATABASE {shard_alias}")
                except Exception:
                    pass
            
            # Delete shard DB after successful merge
            if delete_after_merge:
                try:
                    shard_db.unlink()
                    _eprint(f"[merge] Deleted {shard_db.name}")
                except Exception as e:
                    _eprint(
                        f"[merge] WARNING: could not delete {shard_db.name}: {e}"
                    )
            
        except Exception as e:
            _eprint(
                f"[merge] ERROR processing {shard_db.name}: "
                f"{e.__class__.__name__}: {e}"
            )
            import traceback
            traceback.print_exc()
            # Continue with next shard
    
    # Commit all changes
    main_conn.commit()
    
    _eprint(
        f"[merge] Complete: {stats['papers']} papers, {stats['chunks']} chunks, "
        f"{stats['files']} files from {stats['shards']} shard(s)"
    )
    return stats
