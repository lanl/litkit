"""FAISS index creation and loading utilities for litkit.build.

This module provides functions for creating, loading, and managing FAISS indices
for the papers and chunks vector stores.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import faiss

from pathlib import Path

from litkit.index import (
    hnsw_index,
    flat_ip_index,
    ivfpq_index,
    safe_pq_m,
    faiss_save,
    faiss_save_force,
    faiss_load,
    unwrap_core_and_kind,
    kind_and_core,
    extract_ivf,
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


def load_or_create_paper_index(
    *,
    paper_index_path: Path,
    faiss_lock_path: Path,
    db_lock_path: Path,
    FileLock: type,
    paper_dim: int,
    papers_index: str,
    hnsw_m: int,
    efconstruction: int,
    efsearch: int,
    is_faiss_writer: bool,
) -> faiss.Index:
    """Load an existing paper index or create a new one.
    
    If the index exists:
      - Loads it from disk
      - Verifies metric is IP (inner product)
      - Wraps in IDMap2 if needed
      
    If the index doesn't exist:
      - Creates HNSW or FLAT index based on papers_index param
      - Wraps in IDMap2
      - Saves to disk (requires is_faiss_writer=True)
    
    Args:
        paper_index_path: Path to the paper index file
        faiss_lock_path: Path to FAISS lock file
        db_lock_path: Path to DB lock file
        FileLock: File lock class
        paper_dim: Embedding dimension
        papers_index: Index type ("hnsw" or "flat")
        hnsw_m: HNSW M parameter
        efconstruction: HNSW efConstruction parameter
        efsearch: HNSW efSearch parameter
        is_faiss_writer: Whether this process can write indices
    
    Returns:
        The loaded or created FAISS index (wrapped in IDMap2)
    
    Raises:
        RuntimeError: If metric is wrong or index missing without writer role
    """
    if paper_index_path.exists():
        paper_index = faiss_load(paper_index_path)
        
        # Verify metric type (must be IP for cosine-equivalent retrieval)
        kind, core, _ = unwrap_core_and_kind(paper_index)
        mt = getattr(core, "metric_type", None)
        
        if kind == "hnsw":
            if mt != faiss.METRIC_INNER_PRODUCT:
                raise RuntimeError(
                    "Papers HNSW index is L2; IP required for cosine-equivalent retrieval."
                )
        elif kind == "flat":
            if core.__class__.__name__.lower().endswith("flatl2"):
                raise RuntimeError("Papers FLAT index is L2; IP required.")
        # IVF not expected for papers; leave as-is
        
        # Ensure IDMap2 wrapper
        if not isinstance(paper_index, faiss.IndexIDMap2):
            paper_index = faiss.IndexIDMap2(paper_index)
            if is_faiss_writer:
                with FileLock(faiss_lock_path):
                    faiss_save(paper_index, paper_index_path)
    else:
        # Create new index
        if papers_index == "flat":
            base = flat_ip_index(paper_dim)
        else:
            base = hnsw_index(
                paper_dim,
                M=hnsw_m,
                ef_construction=efconstruction,
                ef_search=efsearch,
            )
            _eprint(
                f"[progress] Building HNSW (papers): started  M={hnsw_m}  "
                f"efConstruction={efconstruction}  efSearch={efsearch}"
            )
        
        paper_index = faiss.IndexIDMap2(base)
        ensure_parent(paper_index_path)
        
        if is_faiss_writer:
            with FileLock(db_lock_path), FileLock(faiss_lock_path):
                faiss_save(paper_index, paper_index_path)
        else:
            raise RuntimeError(
                "PAPER index does not exist. Start a writer with --faiss-writer "
                "or precreate the index."
            )
    
    return paper_index


def load_or_create_chunk_index(
    *,
    chunk_index_path: Path,
    chunk_trained_flag: Path,
    faiss_lock_path: Path,
    db_lock_path: Path,
    FileLock: type,
    chunk_dim: int,
    chunks_index: str,
    ivf_nlist: int,
    pq_m: int,
    is_faiss_writer: bool,
) -> tuple[faiss.Index, bool]:
    """Load an existing chunk index or create a new one.
    
    If the index exists:
      - Loads it from disk
      - Verifies metric is IP
      - Wraps in IDMap2 if needed
      - Checks if IVF-PQ needs training
      
    If the index doesn't exist:
      - Creates FLAT or placeholder IVF-PQ index
      - Wraps in IDMap2
      - Saves to disk (requires is_faiss_writer=True for FLAT)
    
    Args:
        chunk_index_path: Path to the chunk index file
        chunk_trained_flag: Path to the trained flag file
        faiss_lock_path: Path to FAISS lock file
        db_lock_path: Path to DB lock file
        FileLock: File lock class
        chunk_dim: Embedding dimension
        chunks_index: Index type ("ivfpq" or "flat")
        ivf_nlist: IVF nlist parameter
        pq_m: PQ m parameter
        is_faiss_writer: Whether this process can write indices
    
    Returns:
        (index, needs_training): The index and whether IVF-PQ training is needed
    
    Raises:
        RuntimeError: If metric is wrong or index missing without writer role
    """
    needs_training = False
    
    if chunk_index_path.exists():
        chunk_index = faiss_load(chunk_index_path)
        _, core = kind_and_core(chunk_index)
        mt = getattr(core, "metric_type", faiss.METRIC_INNER_PRODUCT)
        
        if mt != faiss.METRIC_INNER_PRODUCT:
            raise RuntimeError(
                "Chunks index metric is not IP; cosine/IP required for normalized SBERT."
            )
        
        # Ensure IDMap2 wrapper
        if not isinstance(chunk_index, faiss.IndexIDMap2):
            chunk_index = faiss.IndexIDMap2(chunk_index)
            if is_faiss_writer:
                with FileLock(faiss_lock_path):
                    faiss_save(chunk_index, chunk_index_path)
        
        # Check if IVF-PQ needs training
        ivf = extract_ivf(chunk_index)
        if isinstance(ivf, faiss.IndexIVFPQ) and not getattr(ivf, "is_trained", False):
            _eprint(
                "[train] WARNING: chunks index is IVFPQ but untrained; "
                "ignoring stale trained flag and retraining."
            )
            _clear_trained_flag(chunk_trained_flag)
            needs_training = True
    else:
        if chunks_index == "flat":
            base = flat_ip_index(chunk_dim)
            chunk_index = faiss.IndexIDMap2(base)
            _clear_trained_flag(chunk_trained_flag)
            ensure_parent(chunk_index_path)
            
            if is_faiss_writer:
                with FileLock(db_lock_path), FileLock(faiss_lock_path):
                    faiss_save(chunk_index, chunk_index_path)
            else:
                raise RuntimeError(
                    "CHUNK index does not exist. Start a writer with --faiss-writer "
                    "or precreate the index."
                )
        else:
            # Placeholder IVF-PQ (will be replaced after training)
            m_safe = safe_pq_m(chunk_dim, pq_m)
            if m_safe != pq_m:
                _eprint(
                    f"[train] note: adjusted pq_m {pq_m} -> {m_safe} to divide dim={chunk_dim}"
                )
            
            eff_nlist = 16  # placeholder; real nlist set during training
            chunk_index = ivfpq_index(chunk_dim, nlist=eff_nlist, m=m_safe)
            ensure_parent(chunk_index_path)
            
            if not is_faiss_writer:
                raise RuntimeError(
                    "CHUNK index does not exist. Start a writer with --faiss-writer "
                    "or precreate the index."
                )
            
            needs_training = True
    
    return chunk_index, needs_training


def _clear_trained_flag(flag_path: Path) -> None:
    """Remove the chunk trained flag file if it exists."""
    try:
        flag_path.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        _eprint(f"[train] WARNING: could not remove {flag_path}: {e}")
