# litkit/db/__init__.py
"""SQLite database management for litkit.

This module centralizes all SQLite-related code that was previously in cli.py:
- Schema definition and initialization
- Connection management  
- Shard database operations for distributed builds
- Query helpers for papers, chunks, and files
- Index tracking (mark_in_index, flush pending marks)
"""

from litkit.db.schema import (
    SCHEMA,
    init_db,
    init_shard_db,
)
from litkit.db.connection import (
    connect_db,
    open_main_db,
    open_shard_db,
    DEFAULT_BUSY_TIMEOUT_MS,
)
from litkit.db.sharding import (
    shard_db_path,
    list_shard_dbs,
    merge_shard_databases,
)
from litkit.db.queries import (
    already_processed,
    register_file,
    preload_paper_id_map,
    preload_chunk_id_map,
    chunk_ids_to_paper_ids,
)
from litkit.db.indexing import (
    mark_in_index,
    flush_pending_marks,
    get_pending_marks,
    clear_pending_marks,
    add_pending_marks,
)
from litkit.db.temp_tables import (
    ensure_temp_candidates_table,
    load_temp_candidates,
)

__all__ = [
    # Schema
    "SCHEMA",
    "init_db",
    "init_shard_db",
    # Connection
    "connect_db",
    "open_main_db",
    "open_shard_db",
    "DEFAULT_BUSY_TIMEOUT_MS",
    # Sharding
    "shard_db_path",
    "list_shard_dbs",
    "merge_shard_databases",
    # Queries
    "already_processed",
    "register_file",
    "preload_paper_id_map",
    "preload_chunk_id_map",
    "chunk_ids_to_paper_ids",
    # Indexing
    "mark_in_index",
    "flush_pending_marks",
    "get_pending_marks",
    "clear_pending_marks",
    "add_pending_marks",
    # Temp tables
    "ensure_temp_candidates_table",
    "load_temp_candidates",
]
