"""IVF-PQ training utilities for litkit.build.

This module provides functions for training IVF-PQ indices for chunk vectors.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import faiss
import numpy as np

from litkit.index import (
    flat_ip_index,
    ivfpq_index,
    safe_pq_m,
    faiss_save,
    auto_set_nprobe,
    PQ_BITS,
)
from litkit.build.helpers import pack_paragraphs, ensure_parent, maybe_fsync_dir
from litkit.ingest import iter_tar_paths
from litkit.ingest.ingest import (
    count_tar_xml_members,
    iter_tar_xml_streams,
    parse_xml_fileobj,
)
from litkit.progress import eprint as _eprint, Progress as _Progress, Pulse as _Pulse, phase as _phase

if TYPE_CHECKING:
    from litkit.embeddings.base import Embedder


# Default training parameters (match cli.py constants)
TRAIN_CHUNK_SAMPLES = 150_000
BATCH_TRAIN_FLUSH = 4000


def _effective_nlist(
    n_train: int,
    requested_nlist: int,
    min_nlist: int = 16,
    *,
    user_forced: bool = False,
) -> int:
    """Calculate effective nlist for IVF training based on data size.
    
    Args:
        n_train: Number of training samples
        requested_nlist: Requested nlist value from user
        min_nlist: Minimum nlist floor
        user_forced: If True, user explicitly set --ivf-nlist
    
    Returns:
        Effective nlist value, capped by training data constraints
    """
    if n_train <= 0:
        return 0
    
    cap_by_data = min(requested_nlist, n_train)
    cap_by_heuristic = min(n_train, max(1, int(4 * math.sqrt(n_train))))
    dynamic_floor = 128 if (n_train >= 100_000 and not user_forced) else min_nlist
    floor = min(n_train, max(1, dynamic_floor))
    
    return max(floor, min(cap_by_data, cap_by_heuristic))


def _clear_trained_flag(flag_path: Path) -> None:
    """Remove the chunk trained flag file if it exists."""
    try:
        flag_path.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        _eprint(f"[train] WARNING: could not remove {flag_path}: {e}")


def train_ivfpq_index(
    *,
    chunk_dim: int,
    chunk_embedder: "Embedder",
    tar_dir: Path | None,
    tar_manifest: Path | None,
    # Index parameters
    ivf_nlist: int,
    pq_m: int,
    nprobe: int | None,
    ivf_nlist_forced: bool = False,
    # Chunking parameters
    chunk_target_chars: int = 1200,
    chunk_min_chars: int = 300,
    chunk_overlap: int = 200,
    chunk_embed_bs: int = 64,
    # Training budget
    train_samples: int = TRAIN_CHUNK_SAMPLES,
    batch_train_flush: int = BATCH_TRAIN_FLUSH,
    # Paths
    chunk_index_path: Path,
    chunk_trained_flag: Path,
    faiss_lock_path: Path,
    db_lock_path: Path,
    FileLock: type,
    # Control
    is_faiss_writer: bool = True,
) -> faiss.Index:
    """Train IVF-PQ chunk index or fall back to FLAT.
    
    Collects training samples by streaming chunks from tar files,
    embeds them, and trains an IVF-PQ index. Falls back to FLAT
    if insufficient training data or training fails.
    
    Args:
        chunk_dim: Embedding dimension for chunks
        chunk_embedder: Embedder instance for chunk encoding
        tar_dir: Directory containing tar shards
        tar_manifest: Manifest file with tar paths (overrides tar_dir)
        ivf_nlist: Requested IVF nlist parameter
        pq_m: PQ m parameter (subquantizers)
        nprobe: IVF probe count (None = auto)
        ivf_nlist_forced: True if user explicitly set --ivf-nlist
        chunk_target_chars: Target chars per chunk
        chunk_min_chars: Minimum chars per chunk
        chunk_overlap: Overlap chars between chunks
        chunk_embed_bs: Batch size for chunk embedding
        train_samples: Target number of training samples
        batch_train_flush: Batch size for training sample collection
        chunk_index_path: Path to save chunk index
        chunk_trained_flag: Path to trained flag file
        faiss_lock_path: Path to FAISS lock file
        db_lock_path: Path to DB lock file
        FileLock: File lock class
        is_faiss_writer: Whether this process can write indices
    
    Returns:
        Trained chunk index wrapped in IndexIDMap2
    """
    texts_buf: list[str] = []
    X_train_list: list[np.ndarray] = []
    
    # Progress tracking
    samples_collected = 0
    _phase("IVF-PQ: learn IVF centroids and PQ codebooks")
    samples_prog = _Progress(
        f"Current number of embeddings of randomly selected text chunks (desired number of vectors={train_samples})",
        total=train_samples,
        emit_final_line=False,
    )
    
    def _flush_train(buf: list[str]) -> np.ndarray:
        """Embed buffered texts and return their embeddings."""
        nonlocal samples_collected, samples_prog
        if not buf:
            return np.zeros((0, chunk_dim), dtype="float32")
        
        X = chunk_embedder.encode(
            buf,
            progress_label=f"Generating a batch of embeddings for use in IVF-PQ training (batch size is {len(buf)})",
            batch_size=chunk_embed_bs,
            progress_done_summary=False,
        )
        
        samples_collected += int(X.shape[0])
        samples_prog.done = min(train_samples, samples_collected)
        samples_prog.tick(inc=0, force=True)
        return X
    
    # Stream text chunks from tar files
    use_tar = tar_dir is not None or tar_manifest is not None
    
    if use_tar:
        tar_paths_list = list(iter_tar_paths(tar_dir, tar_manifest))
        rng = random.Random(int(os.environ.get("LITKIT_TRAIN_SEED", "314159")))
        rng.shuffle(tar_paths_list)
        
        _env_cap = int(os.environ.get("LITKIT_TRAIN_PER_PAPER", "0"))
        if _env_cap > 0:
            train_per_paper = _env_cap
        else:
            est_papers = 0
            for _p in tar_paths_list:
                try:
                    est_papers += count_tar_xml_members(_p)
                except Exception:
                    pass
            est_papers = max(1, est_papers)
            train_per_paper = max(8, min(64, math.ceil(train_samples / est_papers)))
        
        target_papers = max(1, train_samples // max(1, train_per_paper))
        papers_used = 0
        train_total = sum(int(x.shape[0]) for x in X_train_list)
        
        for tpath in tar_paths_list:
            for _member, fobj in iter_tar_xml_streams(tpath):
                try:
                    meta = parse_xml_fileobj(fobj)
                    if not meta:
                        continue
                    
                    paras = meta["paragraphs"] or (
                        [meta["abstract"]] if meta["abstract"] else []
                    )
                    chunks = (
                        pack_paragraphs(
                            paras,
                            max_chars=chunk_target_chars,
                            min_chars=chunk_min_chars,
                            overlap_chars=chunk_overlap,
                        )
                        if paras
                        else []
                    )
                    
                    if chunks and papers_used < target_papers:
                        sel = (
                            chunks
                            if len(chunks) <= train_per_paper
                            else rng.sample(chunks, train_per_paper)
                        )
                        for ch in sel:
                            texts_buf.append(ch)
                            if len(texts_buf) >= batch_train_flush:
                                Xb = _flush_train(texts_buf)
                                X_train_list.append(Xb)
                                train_total += int(Xb.shape[0])
                                texts_buf.clear()
                                if train_total >= train_samples:
                                    break
                        papers_used += 1
                    
                    if papers_used >= target_papers or train_total >= train_samples:
                        break
                finally:
                    try:
                        fobj.close()
                    except Exception:
                        pass
            
            if papers_used >= target_papers or train_total >= train_samples:
                break
    
    # Flush remaining buffer
    if texts_buf and sum(x.shape[0] for x in X_train_list) < train_samples:
        Xb = _flush_train(texts_buf)
        X_train_list.append(Xb)
        texts_buf.clear()
    
    samples_prog.finish()
    
    # Handle no training data case
    if not X_train_list:
        _eprint("[train] WARNING: no chunk texts found for training; falling back to FLAT index")
        base = flat_ip_index(chunk_dim)
        chunk_index = faiss.IndexIDMap2(base)
        _clear_trained_flag(chunk_trained_flag)
        if is_faiss_writer:
            faiss_save(chunk_index, chunk_index_path)
        return chunk_index
    
    # Prepare training data
    X_train = np.vstack(X_train_list)
    if X_train.shape[0] > train_samples:
        X_train = X_train[:train_samples]
    
    n_train = int(X_train.shape[0])
    _eprint(f"[train] chunk training samples: target={train_samples} collected={n_train}")
    ensure_parent(chunk_index_path)
    
    # Training decision tree
    pq_bits = PQ_BITS
    k = 1 << pq_bits
    min_for_micro = 256
    min_for_pq = 39 * k
    m_candidate = safe_pq_m(chunk_dim, pq_m)
    
    # Check if we have enough data for IVF-PQ
    if n_train < max(min_for_micro, min_for_pq, 100 * m_candidate):
        _eprint(f"[train] not enough samples for IVF-PQ (n={n_train}); using FLAT IP")
        base = flat_ip_index(chunk_dim)
        chunk_index = faiss.IndexIDMap2(base)
        if is_faiss_writer:
            with FileLock(db_lock_path), FileLock(faiss_lock_path):
                faiss_save(chunk_index, chunk_index_path)
        _clear_trained_flag(chunk_trained_flag)
        return chunk_index
    
    # Choose nlist with data-aware caps
    eff_nlist = _effective_nlist(n_train, ivf_nlist, user_forced=ivf_nlist_forced)
    max_by_samples = max(1, n_train // 40)
    if eff_nlist > max_by_samples:
        _eprint(f"[train] note: reducing nlist {eff_nlist} -> {max_by_samples} due to limited samples (n={n_train})")
        eff_nlist = max_by_samples
    
    # Re-check adequacy with final nlist
    if eff_nlist < 8 or n_train < max(min_for_pq, 50 * eff_nlist, 100 * m_candidate):
        _eprint(f"[train] nlist/m under-sampled (n={n_train}, nlist={eff_nlist}, m={m_candidate}); using FLAT IP")
        base = flat_ip_index(chunk_dim)
        chunk_index = faiss.IndexIDMap2(base)
        if is_faiss_writer:
            with FileLock(db_lock_path), FileLock(faiss_lock_path):
                faiss_save(chunk_index, chunk_index_path)
        _clear_trained_flag(chunk_trained_flag)
        return chunk_index
    
    # Train IVF-PQ
    m = m_candidate
    if m != pq_m:
        _eprint(f"[train] note: adjusted pq_m {pq_m} -> {m} to divide dim={chunk_dim}")
    _eprint(f"[train] training IVF-PQ: nlist={eff_nlist} m={m} (dim={chunk_dim})")
    
    try:
        new_chunk_index = ivfpq_index(chunk_dim, nlist=eff_nlist, m=m, bits=pq_bits)
        
        # Suppress FAISS verbosity
        try:
            if hasattr(new_chunk_index, "verbose"):
                new_chunk_index.verbose = False
            if hasattr(faiss, "cvar") and hasattr(faiss.cvar, "verbose"):
                faiss.cvar.verbose = False
        except Exception:
            pass
        
        _phase("IVF-PQ: train centroids and PQ codebooks using collected embeddings")
        pulse = _Pulse(
            f"[train] IVF-PQ (nlist={eff_nlist}, m={m}): k-means/codebook fitting",
            period=float(os.environ.get("LITKIT_TRAIN_HEARTBEAT_SEC", "0.5")),
        )
        try:
            faiss.normalize_L2(X_train)
            new_chunk_index.train(X_train)
        finally:
            pulse.stop()
        
        # Verify training succeeded
        if not getattr(new_chunk_index, "is_trained", False):
            raise RuntimeError("IVF-PQ index not trained (is_trained=False)")
        
        # Set nprobe
        auto_set_nprobe(new_chunk_index, nprobe)
        chunk_index = faiss.IndexIDMap2(new_chunk_index)
        
        # Save and write trained flag
        if is_faiss_writer:
            with FileLock(db_lock_path), FileLock(faiss_lock_path):
                faiss_save(chunk_index, chunk_index_path)
                
                _tf_tmp = chunk_trained_flag.with_suffix(".tmp")
                with open(_tf_tmp, "w") as fh:
                    fh.write(json.dumps({
                        "trained_on": int(time.time()),
                        "n": n_train,
                        "nlist": eff_nlist,
                        "m": m,
                    }, indent=2))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(_tf_tmp, chunk_trained_flag)
                maybe_fsync_dir(chunk_trained_flag)
        
        return chunk_index
        
    except Exception as e:
        _eprint(f"[train] WARNING: IVF-PQ training failed ({e}); falling back to FLAT")
        chunk_index = faiss.IndexIDMap2(flat_ip_index(chunk_dim))
        if is_faiss_writer:
            with FileLock(db_lock_path), FileLock(faiss_lock_path):
                faiss_save(chunk_index, chunk_index_path)
        _clear_trained_flag(chunk_trained_flag)
        return chunk_index
