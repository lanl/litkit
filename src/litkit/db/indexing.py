# litkit/db/indexing.py
"""Index tracking for papers and chunks in litkit.

Manages the in_index column that tracks which papers/chunks have been
added to FAISS indices. Uses a pending marks buffer for batched updates.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict

# ---------------------------------------------------------------------------
# Pending Marks Buffer
# ---------------------------------------------------------------------------

# Module-level buffer for deferred in_index updates
# Key: table name ("papers" or "chunks")
# Value: list of IDs to mark as in_index=1
_PENDING_MARKS: dict[str, list[int]] = defaultdict(list)


def get_pending_marks() -> dict[str, list[int]]:
    """Get reference to the pending marks buffer.
    
    Returns:
        Dictionary mapping table names to lists of pending IDs
    """
    return _PENDING_MARKS


def clear_pending_marks() -> None:
    """Clear all pending marks (useful for testing)."""
    _PENDING_MARKS.clear()


def add_pending_marks(table: str, ids: list[int]) -> None:
    """Add IDs to the pending marks buffer for later flushing.
    
    Args:
        table: Table name ("papers" or "chunks")
        ids: List of IDs to mark as in_index=1
    """
    _PENDING_MARKS[table].extend(ids)


# ---------------------------------------------------------------------------
# Index Marking Functions
# ---------------------------------------------------------------------------


def mark_in_index(
    cur: sqlite3.Cursor,
    table: str,
    ids: list[int],
) -> None:
    """Mark the given IDs as in_index=1 in the specified table.
    
    This updates the database immediately. For batched/deferred updates,
    use add_pending_marks() followed by flush_pending_marks().
    
    Args:
        cur: Database cursor
        table: Table name ("papers" or "chunks")
        ids: List of IDs to mark
    """
    if not ids:
        return
    
    # Validate table name to prevent SQL injection
    if table not in ("papers", "chunks"):
        raise ValueError(f"Invalid table name: {table}")
    
    cur.executemany(
        f"UPDATE {table} SET in_index=1 WHERE id=?",
        [(i,) for i in ids],
    )


def flush_pending_marks(cur: sqlite3.Cursor) -> int:
    """Flush all pending marks to the database.
    
    Processes and clears the pending marks buffer, updating the database
    for all buffered IDs.
    
    Args:
        cur: Database cursor
    
    Returns:
        Total number of marks flushed
    """
    total = 0
    
    for table, ids in list(_PENDING_MARKS.items()):
        if not ids:
            continue
        mark_in_index(cur, table, ids)
        total += len(ids)
        _PENDING_MARKS[table].clear()
    
    return total
