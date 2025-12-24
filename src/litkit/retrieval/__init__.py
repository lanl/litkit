# src/litkit/retrieval/__init__.py
"""Retrieval module for litkit.

This module provides the two-stage RAG retrieval pipeline:
- Stage 1: Paper shortlisting via SPECTER2 + HNSW
- Stage 2: Chunk retrieval via SBERT + lexical boost

Functions are designed for library use - they accept explicit paths rather
than relying on cli.py globals.
"""

from litkit.retrieval.stages import (
    shortlist_papers,
    search_chunks_constrained,
    get_chunks,
)
from litkit.retrieval.helpers import (
    query_terms,
    sqlite_norm_expr,
    escape_like,
    normalize_for_search_py,
    avg_chunks_for_papers,
)
from litkit.retrieval.search import (
    faiss_search,
    temporary_search_params,
)
from litkit.retrieval.lexical import (
    LexicalConfig,
    LexicalResult,
    LexicalBackend,
    SqliteLexicalBackend,
    find_rare_terms,
    lexical_search,
    merge_lexical_and_ann,
)

__all__ = [
    # Main retrieval functions
    "shortlist_papers",
    "search_chunks_constrained",
    "get_chunks",
    # Helpers
    "query_terms",
    "sqlite_norm_expr",
    "escape_like",
    "normalize_for_search_py",
    "avg_chunks_for_papers",
    # FAISS search
    "faiss_search",
    "temporary_search_params",
    # Lexical
    "LexicalConfig",
    "LexicalResult",
    "LexicalBackend",
    "SqliteLexicalBackend",
    "find_rare_terms",
    "lexical_search",
    "merge_lexical_and_ann",
]
