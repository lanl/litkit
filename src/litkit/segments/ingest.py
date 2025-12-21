# litkit/segments/ingest.py
"""Segment ingestion into FAISS indices for litkit.

This module handles the consumer side of the producer-consumer workflow:
- Loads embedding segments written by producer nodes
- Resolves doc_ids to database IDs using preloaded maps
- Adds vectors to FAISS indices with deduplication
- Supports atomic file claiming to prevent race conditions
"""

from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from litkit.db.queries import preload_paper_id_map, preload_chunk_id_map
from litkit.index.dedup import add_with_ids_dedup
from litkit.index.io import faiss_save, faiss_save_force
from litkit.index.ids import make_id_selector, safe_remove_ids

if TYPE_CHECKING:
    from litkit.config.paths import WorkspacePaths




def _eprint(msg: str = "", *, end: str = "\n") -> None:
    """Print to stderr with flush."""
    sys.stderr.write(msg + end)
    try:
        sys.stderr.flush()
    except Exception:
        pass


@dataclass
class SegmentIngestConfig:
    """Configuration for segment ingestion."""
    
    segment_dir: Path
    save_every: int = 2  # Save FAISS index every N segments


def _glob_segment_files(outdir: Path, kind: str) -> list[Path]:
    """Find segment files matching both new and old naming patterns.
    
    Args:
        outdir: Directory to search
        kind: Either "papers" or "chunks"
    
    Returns:
        Sorted list of segment file paths (including .ingesting files)
    """
    cand = []
    
    if kind == "papers":
        # New style: papers_*.npz
        cand.extend(outdir.glob("papers_*.npz"))
        # Old style: papers.seg.*.npz
        cand.extend(outdir.glob("papers.seg.*.npz"))
        # In-progress files from previous interrupted runs
        cand.extend(outdir.glob("papers_*.npz.ingesting"))
        cand.extend(outdir.glob("papers.seg.*.npz.ingesting"))
    else:  # chunks
        # New style: chunks_*.npz
        cand.extend(outdir.glob("chunks_*.npz"))
        # Old style: chunks.seg.*.npz
        cand.extend(outdir.glob("chunks.seg.*.npz"))
        # In-progress files from previous interrupted runs
        cand.extend(outdir.glob("chunks_*.npz.ingesting"))
        cand.extend(outdir.glob("chunks.seg.*.npz.ingesting"))
    
    return sorted(cand)


def ingest_paper_segments(
    conn: sqlite3.Connection,
    paper_index: faiss.Index,
    outdir: Path,
    faiss_lock_path: Path,
    paper_index_path: Path,
    db_lock_path: Path | None = None,
    *,
    save_every: int = 2,
    paper_id_map: dict[str, int] | None = None,
    FileLock: type | None = None,
) -> tuple[faiss.Index, int]:
    """Ingest paper embedding segments from producer nodes.
    
    Segments contain doc_ids (globally unique file paths) and embeddings.
    At ingestion time, we resolve doc_id → paper_id using the merged main DB.
    
    Uses atomic file claiming (.ingesting rename) to prevent race conditions
    between multiple consumers.
    
    Args:
        conn: SQLite connection to main database
        paper_index: FAISS index for papers (may be wrapped in IndexIDMap2 if needed)
        outdir: Directory containing segment files
        faiss_lock_path: Path to FAISS lock file
        paper_index_path: Path to save the paper index
        db_lock_path: Path to DB lock file (optional, for commit locking)
        save_every: Save index every N segment files
        paper_id_map: Optional preloaded doc_id → paper_id mapping for O(1) lookups.
                      If None, will be loaded once at start.
        FileLock: File lock class to use (pass from caller to avoid circular import)
    
    Returns:
        (paper_index, added_count): The (possibly wrapped) index and number of vectors added.
        Caller should rebind their index reference to the returned value.
    """
    # Lazy import if not provided
    if FileLock is None:
        from litkit.concurrent.locking import FileLock as _FileLock
        FileLock = _FileLock
    
    outdir = Path(outdir)
    if not outdir.exists():
        return paper_index, 0
    
    cand = _glob_segment_files(outdir, "papers")
    if not cand:
        return paper_index, 0
    
    # Preload paper_id_map once if not provided (O(1) lookups vs O(N) queries)
    if paper_id_map is None:
        paper_id_map = preload_paper_id_map(conn)
        _eprint(f"[segments] preloaded {len(paper_id_map)} paper ID mappings")
    
    cur = conn.cursor()
    added_total = 0
    batch_counter = 0
    
    for p in cand:
        is_ingesting = p.name.endswith(".npz.ingesting")
        tmp = p if is_ingesting else p.with_suffix(p.suffix + ".ingesting")
        
        # Atomic claim: rename to .ingesting before reading
        if not is_ingesting:
            try:
                os.replace(p, tmp)
            except FileNotFoundError:
                continue  # Another process claimed it
            except Exception:
                continue
        
        try:
            with np.load(tmp, mmap_mode="r", allow_pickle=True) as z:
                # Reject wrong-kind files
                if "kind" in z.files and str(z["kind"].item()).strip() != "papers":
                    raise ValueError("wrong segment kind for paper ingester")
                
                # New format: doc_ids + vecs (content-addressed)
                if "doc_ids" in z.files and "vecs" in z.files:
                    doc_ids = z["doc_ids"]  # object array of strings
                    X = np.ascontiguousarray(z["vecs"].astype(np.float32))
                    
                    # Resolve doc_id → paper_id using preloaded map (O(1) lookups)
                    resolved_ids = []
                    valid_mask = []
                    for i, doc_id in enumerate(doc_ids):
                        doc_id_str = str(doc_id)
                        paper_id = paper_id_map.get(doc_id_str)
                        if paper_id is not None:
                            resolved_ids.append(paper_id)
                            valid_mask.append(True)
                        else:
                            valid_mask.append(False)
                    
                    # Log resolution failures for debugging canonicalization issues
                    missing = np.count_nonzero(~np.array(valid_mask, dtype=bool))
                    if missing:
                        _eprint(f"[segments] WARNING: skipped {missing}/{len(doc_ids)} paper embeddings in {p.name} "
                                "(doc_id not found in main DB - check doc_id canonicalization)")
                    
                    if not resolved_ids:
                        os.remove(tmp)
                        continue
                    
                    # Filter to only valid entries
                    valid_mask = np.array(valid_mask, dtype=bool)
                    ids = np.array(resolved_ids, dtype=np.int64)
                    X = X[valid_mask]
                
                # Legacy format: ids + vecs (shard-local IDs, deprecated)
                elif "ids" in z.files and ("vecs" in z.files or "emb" in z.files):
                    ids = np.ascontiguousarray(z["ids"].astype(np.int64))
                    X = np.ascontiguousarray(
                        (z["vecs"] if "vecs" in z.files else z["emb"]).astype(np.float32)
                    )
                else:
                    raise ValueError(f"Segment missing doc_ids/vecs or ids/vecs in {p.name}")
            
            if ids.size == 0:
                os.remove(tmp)
                continue
            
            # Ensure IDMap2 wrapper for external ID tracking
            if not isinstance(paper_index, faiss.IndexIDMap2):
                paper_index = faiss.IndexIDMap2(paper_index)
            
            prior_ntotal = int(getattr(paper_index, "ntotal", 0) or 0)
            added, ids_added, saved = 0, [], False
            
            with FileLock(faiss_lock_path):
                # Remove existing IDs to allow re-embedding (mirrors chunk behavior)
                sel = make_id_selector(ids)
                safe_remove_ids(paper_index, sel)
                
                added, ids_added = add_with_ids_dedup(paper_index, ids, X)
                if added:
                    # Force save on first vectors, throttled save otherwise
                    saved = faiss_save_force(paper_index, paper_index_path) if prior_ntotal == 0 \
                            else faiss_save(paper_index, paper_index_path)
            
            if added:
                if saved:
                    from litkit.db.indexing import mark_in_index
                    if db_lock_path:
                        with FileLock(db_lock_path):
                            if len(ids_added) > 0:
                                mark_in_index(cur, "papers", [int(i) for i in ids_added])
                            conn.commit()
                    else:
                        if len(ids_added) > 0:
                            mark_in_index(cur, "papers", [int(i) for i in ids_added])
                        conn.commit()
                # else: save failed, reconcile+backfill will repair
            
            added_total += int(added or 0)
            batch_counter += 1
            os.remove(tmp)
            
            # Periodic save
            if (batch_counter % max(1, int(save_every))) == 0:
                with FileLock(faiss_lock_path):
                    saved_now = faiss_save(paper_index, paper_index_path)
                if saved_now:
                    from litkit.db.indexing import flush_pending_marks
                    if db_lock_path:
                        with FileLock(db_lock_path):
                            flush_pending_marks(cur)
                            conn.commit()
                    else:
                        flush_pending_marks(cur)
                        conn.commit()
        
        except Exception as e:
            # On error, rename file back so it can be retried next run
            if not is_ingesting:
                try:
                    os.replace(tmp, p)
                except Exception:
                    pass
            _eprint(f"[segments] ERROR ingesting {p.name}: {e.__class__.__name__}: {e}")
    
    return paper_index, added_total


def ingest_chunk_segments(
    conn: sqlite3.Connection,
    chunk_index: faiss.Index,
    outdir: Path,
    faiss_lock_path: Path,
    chunk_index_path: Path,
    db_lock_path: Path | None = None,
    *,
    save_every: int = 2,
    paper_id_map: dict[str, int] | None = None,
    chunk_id_map: dict[tuple[int, int], int] | None = None,
    FileLock: type | None = None,
) -> tuple[faiss.Index, int]:
    """Ingest chunk embedding segments from producer nodes.
    
    Segments contain (paper_doc_id, ord) pairs and embeddings.
    At ingestion time, we resolve (paper_doc_id, ord) → chunk_id using the merged main DB.
    
    Uses atomic file claiming (.ingesting rename) to prevent race conditions.
    
    Safe file-handling:
      - rename "<file>.npz" -> "<file>.npz.ingesting" before reading (atomic)
      - if already ".npz.ingesting", read in place
      - on success, delete the .ingesting file
      - on failure, rename back so it can be retried next run

    Args:
        conn: SQLite connection to main database
        chunk_index: FAISS index for chunks (may be wrapped in IndexIDMap2 if needed)
        outdir: Directory containing segment files
        faiss_lock_path: Path to FAISS lock file
        chunk_index_path: Path to save the chunk index
        db_lock_path: Path to DB lock file (optional, for commit locking)
        save_every: Save index every N segment files
        paper_id_map: Optional preloaded doc_id → paper_id mapping for O(1) lookups.
                      If None, will be loaded once at start.
        chunk_id_map: Optional preloaded (paper_id, ord) → chunk_id mapping for O(1) lookups.
                      If None, will be loaded once at start.
        FileLock: File lock class to use (pass from caller to avoid circular import)

    Returns:
        (chunk_index, added_count): The (possibly wrapped) index and number of vectors added.
        Caller should rebind their index reference to the returned value.
    """
    # Lazy import if not provided
    if FileLock is None:
        from litkit.concurrent.locking import FileLock as _FileLock
        FileLock = _FileLock
    
    outdir = Path(outdir)
    if not outdir.exists():
        return chunk_index, 0
    
    # Ensure external ID mapping is present for robust remove and add
    if not isinstance(chunk_index, faiss.IndexIDMap2):
        chunk_index = faiss.IndexIDMap2(chunk_index)
    
    cand = _glob_segment_files(outdir, "chunks")
    if not cand:
        return chunk_index, 0
    
    # Preload ID maps once if not provided (O(1) lookups vs O(N) queries)
    if paper_id_map is None:
        paper_id_map = preload_paper_id_map(conn)
        _eprint(f"[segments] preloaded {len(paper_id_map)} paper ID mappings")
    if chunk_id_map is None:
        chunk_id_map = preload_chunk_id_map(conn)
        _eprint(f"[segments] preloaded {len(chunk_id_map)} chunk ID mappings")
    
    cur = conn.cursor()
    added_total = 0
    batch_counter = 0
    
    for p in cand:
        # Normalize to a working path "tmp" that we always read from:
        is_ingesting = p.name.endswith(".npz.ingesting")
        tmp = p if is_ingesting else p.with_suffix(p.suffix + ".ingesting")
        
        if not is_ingesting:
            try:
                # Claim atomically; another writer may race us.
                os.replace(p, tmp)
            except FileNotFoundError:
                continue
            except Exception:
                # Could not claim; skip
                continue
        
        try:
            with np.load(tmp, mmap_mode="r", allow_pickle=True) as z:
                # Reject wrong-kind files (old .seg has 'kind')
                if "kind" in z.files and str(z["kind"].item()).strip() != "chunks":
                    raise ValueError("wrong segment kind for chunk ingester")
                
                # New format: paper_doc_ids + ords + vecs (content-addressed)
                if "paper_doc_ids" in z.files and "ords" in z.files and "vecs" in z.files:
                    paper_doc_ids = z["paper_doc_ids"]  # object array of strings
                    ords = z["ords"]  # int32 array
                    X = np.ascontiguousarray(z["vecs"].astype(np.float32))
                    
                    # Resolve (paper_doc_id, ord) → chunk_id using preloaded maps (O(1) lookups)
                    resolved_ids = []
                    valid_mask = []
                    for i, (doc_id, ord_val) in enumerate(zip(paper_doc_ids, ords)):
                        doc_id_str = str(doc_id)
                        ord_int = int(ord_val)
                        
                        # Get paper_id from preloaded map
                        paper_id = paper_id_map.get(doc_id_str)
                        
                        if paper_id is not None:
                            # Get chunk_id from preloaded map
                            chunk_id = chunk_id_map.get((paper_id, ord_int))
                            
                            if chunk_id is not None:
                                resolved_ids.append(chunk_id)
                                valid_mask.append(True)
                            else:
                                valid_mask.append(False)
                        else:
                            valid_mask.append(False)
                    
                    # Log resolution failures for debugging canonicalization issues
                    missing = np.count_nonzero(~np.array(valid_mask, dtype=bool))
                    if missing:
                        _eprint(f"[segments] WARNING: skipped {missing}/{len(paper_doc_ids)} chunk embeddings in {p.name} "
                                "(doc_id/ord not found in main DB - check doc_id canonicalization)")
                    
                    if not resolved_ids:
                        os.remove(tmp)
                        continue
                    
                    # Filter to only valid entries
                    valid_mask = np.array(valid_mask, dtype=bool)
                    ids = np.array(resolved_ids, dtype=np.int64)
                    X = X[valid_mask]
                
                # Legacy format: ids + vecs (shard-local IDs, deprecated)
                elif "ids" in z.files and "vecs" in z.files:
                    ids = np.ascontiguousarray(z["ids"].astype(np.int64))
                    X = np.ascontiguousarray(z["vecs"].astype(np.float32))
                elif "ids" in z.files and "emb" in z.files:
                    ids = np.ascontiguousarray(z["ids"].astype(np.int64))
                    X = np.ascontiguousarray(z["emb"].astype(np.float32))
                else:
                    raise ValueError(f"Segment missing required keys: {z.files}")
            
            if ids.size == 0:
                os.remove(tmp)
                continue
            
            prior_ntotal = int(getattr(chunk_index, "ntotal", 0) or 0)
            added, ids_added, saved = 0, [], False
            
            with FileLock(faiss_lock_path):
                # Remove existing IDs to allow re-embedding
                sel = make_id_selector(ids)
                safe_remove_ids(chunk_index, sel)
                
                added, ids_added = add_with_ids_dedup(chunk_index, ids, X)
                if added:
                    saved = faiss_save_force(chunk_index, chunk_index_path) if prior_ntotal == 0 \
                            else faiss_save(chunk_index, chunk_index_path)
            
            if added:
                if saved:
                    from litkit.db.indexing import mark_in_index
                    if db_lock_path:
                        with FileLock(db_lock_path):
                            if len(ids_added) > 0:
                                mark_in_index(cur, "chunks", [int(i) for i in ids_added])
                            conn.commit()
                    else:
                        if len(ids_added) > 0:
                            mark_in_index(cur, "chunks", [int(i) for i in ids_added])
                        conn.commit()
                # else: save failed, reconcile+backfill will repair
            
            added_total += int(added)
            batch_counter += 1
            os.remove(tmp)
            
            # Periodic save
            if (batch_counter % max(1, int(save_every))) == 0:
                with FileLock(faiss_lock_path):
                    saved_now = faiss_save(chunk_index, chunk_index_path)
                if saved_now:
                    from litkit.db.indexing import flush_pending_marks
                    if db_lock_path:
                        with FileLock(db_lock_path):
                            flush_pending_marks(cur)
                            conn.commit()
                    else:
                        flush_pending_marks(cur)
                        conn.commit()
        
        except Exception as e:
            # If we claimed it, put it back so another run can retry.
            if not is_ingesting:
                try:
                    os.replace(tmp, p)
                except Exception:
                    pass
            _eprint(f"[segments] ERROR ingesting {p.name}: {e.__class__.__name__}: {e}")
    
    return chunk_index, added_total
