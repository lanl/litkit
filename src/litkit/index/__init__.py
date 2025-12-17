# litkit/index/__init__.py
"""FAISS index management for litkit.

This module centralizes all FAISS-related code that was previously in cli.py:
- Index creation (flat, HNSW, IVF-PQ)
- Index I/O (save, load, caching)
- Index introspection (type detection, unwrapping)
- ID operations (selectors, removal, dedup)
- Search operations with nprobe handling
"""

from litkit.index.constants import (
    PQ_BITS,
    USE_DOWNCAST_FALLBACK,
)
from litkit.index.factory import (
    flat_ip_index,
    hnsw_index,
    ivfpq_index,
    safe_pq_m,
)
from litkit.index.io import (
    faiss_save,
    faiss_save_force,
    faiss_load,
    faiss_load_cached,
    clear_faiss_cache,
)
from litkit.index.introspection import (
    unwrap_core_and_kind,
    kind_and_core,
    extract_ivf,
    report_faiss_index,
    faiss_present_ids,
)
from litkit.index.ids import (
    make_id_selector,
    safe_remove_ids,
)
from litkit.index.search import (
    pick_nprobe,
    auto_set_nprobe,
    faiss_search,
)
from litkit.index.dedup import (
    add_with_ids_dedup,
)

__all__ = [
    # Constants
    "PQ_BITS",
    "USE_DOWNCAST_FALLBACK",
    # Factory
    "flat_ip_index",
    "hnsw_index",
    "ivfpq_index",
    "safe_pq_m",
    # I/O
    "faiss_save",
    "faiss_save_force",
    "faiss_load",
    "faiss_load_cached",
    "clear_faiss_cache",
    # Introspection
    "unwrap_core_and_kind",
    "kind_and_core",
    "extract_ivf",
    "report_faiss_index",
    "faiss_present_ids",
    # IDs
    "make_id_selector",
    "safe_remove_ids",
    # Search
    "pick_nprobe",
    "auto_set_nprobe",
    "faiss_search",
    # Dedup
    "add_with_ids_dedup",
]
