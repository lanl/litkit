"""FAISS index creation and loading utilities for litkit.build.

This module provides functions for creating, loading, and managing FAISS indices
for the papers and chunks vector stores.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import faiss

from litkit.index import (
    hnsw_index,
    flat_ip_index,
    ivfpq_index,
    safe_pq_m,
    faiss_save_force,
)
from litkit.build.helpers import ensure_parent
from litkit.progress import eprint as _eprint

if TYPE_CHECKING:
    from litkit.build.config import BuildConfig


def init_empty_indices(
    cfg: "BuildConfig",
    *,
    FileLock: type,
    paper_dim: int = 768,
    chunk_dim: int = 768,
) -> None:
    """Create empty FAISS indices for bootstrap mode (--init-indices-only).
    
    This is used in multi-node setups where the consumer needs pre-existing
    indices before producers start. Creates both paper and chunk indices
    based on the configuration.
    
    Args:
        cfg: Build configuration with index parameters and paths
        FileLock: File lock class for thread-safe index writes
        paper_dim: Embedding dimension for papers (default: 768 for SPECTER2)
        chunk_dim: Embedding dimension for chunks (default: 768 for SBERT)
    
    Raises:
        ValueError: If cfg.faiss_writer is False
    
    Side effects:
        - Creates paper index at cfg.paper_index_path
        - Creates chunk index at cfg.chunk_index_path
        - Creates parent directories if needed
    """
    if not cfg.faiss_writer:
        raise ValueError("--init-indices-only requires --faiss-writer")
    
    _eprint("[bootstrap] Creating empty FAISS indices...")
    
    # Create paper index (HNSW or FLAT based on config)
    if cfg.papers_index == "flat":
        paper_base = flat_ip_index(paper_dim)
    else:  # hnsw (default)
        paper_base = hnsw_index(
            paper_dim,
            M=cfg.hnsw_m,
            ef_construction=cfg.efconstruction,
            ef_search=cfg.efsearch,
        )
    paper_index = faiss.IndexIDMap2(paper_base)
    
    ensure_parent(cfg.paper_index_path)
    with FileLock(cfg.faiss_lock):
        faiss_save_force(paper_index, cfg.paper_index_path)
    
    # Create chunk index (FLAT or IVF-PQ based on config)
    if cfg.chunks_index == "flat":
        chunk_base = flat_ip_index(chunk_dim)
    else:  # ivfpq (default)
        m_safe = safe_pq_m(chunk_dim, cfg.pq_m)
        chunk_base = ivfpq_index(chunk_dim, nlist=cfg.ivf_nlist, m=m_safe)
    chunk_index = faiss.IndexIDMap2(chunk_base)
    
    ensure_parent(cfg.chunk_index_path)
    with FileLock(cfg.faiss_lock):
        faiss_save_force(chunk_index, cfg.chunk_index_path)
    
    _eprint("[bootstrap] Empty indices created. Exiting.")
