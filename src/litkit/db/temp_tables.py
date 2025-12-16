# litkit/db/temp_tables.py
"""Temporary table helpers for retrieval operations in litkit."""

from __future__ import annotations

import sqlite3


def ensure_temp_candidates_table(conn: sqlite3.Connection) -> None:
    """Create the temporary candidates table if it doesn't exist.
    
    This table is used during retrieval to efficiently filter chunks
    by a set of candidate paper IDs.
    
    Args:
        conn: Database connection
    """
    conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS cand_papers "
        "(id INTEGER PRIMARY KEY)"
    )


def load_temp_candidates(conn: sqlite3.Connection, pids: list[int]) -> None:
    """Load paper IDs into the temporary candidates table.
    
    Clears any existing candidates and loads the new set. This is used
    to constrain chunk retrieval to only chunks from specific papers.
    
    Args:
        conn: Database connection
        pids: List of paper IDs to load as candidates
    """
    ensure_temp_candidates_table(conn)
    cur = conn.cursor()
    
    # Clear existing candidates
    cur.execute("DELETE FROM cand_papers")
    
    # Batch insert candidates
    if pids:
        cur.executemany(
            "INSERT OR IGNORE INTO cand_papers (id) VALUES (?)",
            [(p,) for p in pids],
        )
    
    conn.commit()
