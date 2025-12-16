# litkit/db/schema.py
"""SQLite schema definition and database initialization for litkit."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litkit.config.paths import WorkspacePaths

# ---------------------------------------------------------------------------
# Schema Definition
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT UNIQUE NOT NULL,
    title       TEXT,
    abstract    TEXT,
    authors     TEXT,
    year        INTEGER,
    src_file    TEXT,
    in_index    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_papers_doc_id ON papers(doc_id);
CREATE INDEX IF NOT EXISTS idx_papers_in_index ON papers(in_index);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id    INTEGER NOT NULL REFERENCES papers(id),
    ord         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    in_index    INTEGER DEFAULT 0,
    UNIQUE(paper_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper_id ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_in_index ON chunks(in_index);

CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    size        INTEGER,
    mtime_ns    INTEGER,
    paper_id    INTEGER REFERENCES papers(id)
);
"""


# ---------------------------------------------------------------------------
# Schema Migration
# ---------------------------------------------------------------------------

def ensure_in_index_columns(conn: sqlite3.Connection) -> None:
    """Idempotent migration: ensure `in_index` and `doc_id` columns exist.
    
    This handles databases created before these columns were added to the
    schema. Safe to call on every database open.
    """
    cur = conn.cursor()
    
    # Check papers table columns
    cur.execute("PRAGMA table_info(papers)")
    paper_cols = {row[1] for row in cur.fetchall()}
    
    if "in_index" not in paper_cols:
        cur.execute("ALTER TABLE papers ADD COLUMN in_index INTEGER DEFAULT 0")
    if "doc_id" not in paper_cols:
        # Add doc_id column with a temporary unique value based on id
        cur.execute("ALTER TABLE papers ADD COLUMN doc_id TEXT")
        cur.execute("UPDATE papers SET doc_id = 'legacy_' || id WHERE doc_id IS NULL")
        # Create index if not exists
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_papers_doc_id ON papers(doc_id)"
        )
    
    # Check chunks table columns
    cur.execute("PRAGMA table_info(chunks)")
    chunk_cols = {row[1] for row in cur.fetchall()}
    
    if "in_index" not in chunk_cols:
        cur.execute("ALTER TABLE chunks ADD COLUMN in_index INTEGER DEFAULT 0")
    
    # Ensure in_index indices exist
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_papers_in_index ON papers(in_index)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_in_index ON chunks(in_index)"
    )
    
    conn.commit()


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------

def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


def init_db(
    db_path: Path,
    journal_mode: str = "WAL",
    busy_timeout_ms: int = 30000,
) -> sqlite3.Connection:
    """Initialize (or open) the main SQLite database and ensure schema exists.
    
    Args:
        db_path: Path to the SQLite database file
        journal_mode: SQLite journal mode (WAL recommended for concurrent access)
        busy_timeout_ms: Timeout in milliseconds when waiting for locks
    
    Returns:
        Open database connection with schema initialized
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(f"PRAGMA journal_mode={journal_mode}")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA foreign_keys=ON")
    
    conn.executescript(SCHEMA)
    ensure_in_index_columns(conn)
    conn.commit()
    
    _eprint(f"[db] Initialized database at {db_path}")
    return conn


def init_shard_db(
    shard_path: Path,
    shard_id: int,
    journal_mode: str = "WAL",
    busy_timeout_ms: int = 30000,
) -> sqlite3.Connection:
    """Initialize a shard-specific SQLite database for producer mode.
    
    In distributed builds, each producer writes to its own shard database.
    These are later merged into the main database by the consumer.
    
    Args:
        shard_path: Path to the shard database file
        shard_id: Shard identifier (for logging)
        journal_mode: SQLite journal mode
        busy_timeout_ms: Timeout in milliseconds when waiting for locks
    
    Returns:
        Open database connection with schema initialized
    """
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    
    _eprint(f"[db] Producer shard {shard_id}: using {shard_path}")
    
    conn = sqlite3.connect(str(shard_path), check_same_thread=False)
    conn.execute(f"PRAGMA journal_mode={journal_mode}")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    conn.execute("PRAGMA foreign_keys=ON")
    
    conn.executescript(SCHEMA)
    ensure_in_index_columns(conn)
    conn.commit()
    
    return conn
