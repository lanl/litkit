# litkit/db/connection.py
"""SQLite connection management for litkit."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litkit.config.paths import WorkspacePaths

# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

# Fallback default (30 seconds). The CLI can override via LITKIT_SQLITE_BUSY_TIMEOUT_MS
# environment variable, which is read at CALL TIME (not import time) to ensure
# the CLI's --sqlite-busy-timeout-ms flag is honored.
DEFAULT_BUSY_TIMEOUT_MS: int = 30000


def _get_busy_timeout() -> int:
    """Get busy timeout from environment (call-time) or return default.
    
    This function is called at connection time, not import time, so
    the CLI's os.environ["LITKIT_SQLITE_BUSY_TIMEOUT_MS"] = str(...)
    assignment in main() is honored.
    """
    return int(os.environ.get("LITKIT_SQLITE_BUSY_TIMEOUT_MS", str(DEFAULT_BUSY_TIMEOUT_MS)))


# ---------------------------------------------------------------------------
# Connection Functions
# ---------------------------------------------------------------------------

def connect_db(
    db_path: Path,
    busy_timeout_ms: int | None = None,
    *,
    autocommit: bool = False,
) -> sqlite3.Connection:
    """Open a connection to an existing SQLite database.
    
    This is for connecting to a database that has already been initialized.
    For creating/initializing a database, use init_db() or init_shard_db().
    
    Args:
        db_path: Path to the SQLite database file
        busy_timeout_ms: Timeout in ms when waiting for locks. If None, reads
                         from LITKIT_SQLITE_BUSY_TIMEOUT_MS env var (default: 30000)
        autocommit: If True, set isolation_level=None for autocommit mode
    
    Returns:
        Open database connection
    """
    if busy_timeout_ms is None:
        busy_timeout_ms = _get_busy_timeout()
    
    isolation = None if autocommit else ""
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
        isolation_level=isolation,
    )
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    
    return conn


def open_main_db(
    paths: "WorkspacePaths",
    journal_mode: str = "WAL",
    busy_timeout_ms: int = 30000,
) -> sqlite3.Connection:
    """Open or initialize the main database using WorkspacePaths.
    
    This is the preferred way to open the main database in new code.
    
    Args:
        paths: WorkspacePaths instance with db_path set
        journal_mode: SQLite journal mode
        busy_timeout_ms: Timeout in milliseconds
    
    Returns:
        Open database connection with schema initialized
    """
    from litkit.db.schema import init_db
    return init_db(paths.db_path, journal_mode, busy_timeout_ms)


def open_shard_db(
    paths: "WorkspacePaths",
    shard_id: int,
    journal_mode: str = "WAL",
    busy_timeout_ms: int = 30000,
) -> sqlite3.Connection:
    """Open or initialize a shard database using WorkspacePaths.
    
    This is the preferred way to open shard databases in new code.
    
    Args:
        paths: WorkspacePaths instance
        shard_id: Shard identifier
        journal_mode: SQLite journal mode
        busy_timeout_ms: Timeout in milliseconds
    
    Returns:
        Open database connection with schema initialized
    """
    from litkit.db.sharding import shard_db_path
    from litkit.db.schema import init_shard_db
    
    shard_path = shard_db_path(paths.sqlite_dir, shard_id)
    return init_shard_db(shard_path, shard_id, journal_mode, busy_timeout_ms)
