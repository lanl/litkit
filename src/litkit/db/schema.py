# litkit/db/schema.py
"""SQLite schema definition and database initialization for litkit."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema Definition (matches cli.py exactly)
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT,                      -- stable document identifier (pmcid or pmid or hash)
  pmid TEXT,
  pmcid TEXT,
  title TEXT,
  abstract TEXT,
  in_index INTEGER DEFAULT 0        -- 0=not yet in FAISS, 1=added
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id INTEGER NOT NULL,
  ord INTEGER NOT NULL,
  text TEXT NOT NULL,
  in_index INTEGER DEFAULT 0,       -- 0=not yet in FAISS, 1=added
  FOREIGN KEY(paper_id) REFERENCES papers(id)
);
CREATE INDEX IF NOT EXISTS chunks_paper_id ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS chunks_paper_ord ON chunks(paper_id, ord);
CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY,
  size INTEGER,
  mtime REAL,
  paper_id INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS papers_pmcid_uq ON papers(pmcid) WHERE pmcid <> '';
CREATE UNIQUE INDEX IF NOT EXISTS papers_pmid_uq  ON papers(pmid)  WHERE pmid  <> '';
CREATE UNIQUE INDEX IF NOT EXISTS papers_doc_id_uq ON papers(doc_id) WHERE doc_id <> '';
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
    for table in ("papers", "chunks"):
        cols = {row[1] for row in cur.execute(f"PRAGMA table_info({table})")}
        if "in_index" not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN in_index INTEGER DEFAULT 0")
    
    # Ensure doc_id column exists on papers table (for multi-producer deduplication)
    papers_cols = {row[1] for row in cur.execute("PRAGMA table_info(papers)")}
    if "doc_id" not in papers_cols:
        cur.execute("ALTER TABLE papers ADD COLUMN doc_id TEXT")
        # Create unique index for doc_id if it doesn't exist
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS papers_doc_id_uq ON papers(doc_id) WHERE doc_id <> ''"
        )
    
    conn.commit()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------

def init_db(
    db_path: Path,
    journal_mode: str = "TRUNCATE",
    busy_timeout_ms: int = 120000,
) -> sqlite3.Connection:
    """Initialize (or open) the main SQLite database and ensure schema exists.
    
    Args:
        db_path: Path to the SQLite database file
        journal_mode: SQLite journal mode (TRUNCATE recommended for NFS/Lustre)
        busy_timeout_ms: Timeout in milliseconds when waiting for locks
    
    Returns:
        Open database connection with schema initialized
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path), isolation_level="DEFERRED", timeout=busy_timeout_ms / 1000.0)
    
    mode = (journal_mode or "TRUNCATE").upper()
    supported = {"TRUNCATE", "WAL"}
    if mode not in supported:
        _eprint(f"[db] WARNING: unsupported journal_mode={mode!r} -> forcing TRUNCATE")
        mode = "TRUNCATE"
    
    conn.execute(f"PRAGMA journal_mode={mode};")
    
    if mode == "WAL":
        _eprint(
            "[db] WARNING: WAL selected. Ensure the DB is on node-local storage. "
            "On shared FS (NFS/Lustre), prefer --sqlite-journal-mode TRUNCATE."
        )
        conn.execute("PRAGMA wal_autocheckpoint=1000;")
    
    eff_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    _eprint(f"[db] journal_mode set to {eff_mode}")
    
    conn.execute("PRAGMA synchronous=FULL;")
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)};")
    conn.execute("PRAGMA temp_store=MEMORY;")
    
    conn.executescript(SCHEMA)
    ensure_in_index_columns(conn)
    conn.commit()
    
    return conn


def init_shard_db(
    shard_path: Path,
    shard_id: int,
    journal_mode: str = "TRUNCATE",
    busy_timeout_ms: int = 120000,
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
    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    
    _eprint(f"[db] Producer shard {shard_id}: using {shard_path}")
    
    conn = sqlite3.connect(str(shard_path), isolation_level="DEFERRED", timeout=busy_timeout_ms / 1000.0)
    
    mode = (journal_mode or "TRUNCATE").upper()
    conn.execute(f"PRAGMA journal_mode={mode};")
    conn.execute("PRAGMA synchronous=FULL;")
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)};")
    conn.execute("PRAGMA temp_store=MEMORY;")
    
    conn.executescript(SCHEMA)
    ensure_in_index_columns(conn)
    conn.commit()
    
    return conn
