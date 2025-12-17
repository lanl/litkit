# litkit/segments/ingest.py
"""Segment ingestion into FAISS indices for litkit."""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np

from litkit.segments.constants import (
    PAPER_SEGMENT_PREFIX,
    CHUNK_SEGMENT_PREFIX,
    SEGMENT_EXTENSION,
)
from litkit.db.queries import preload_paper_id_map, preload_chunk_id_map
from litkit.db.indexing import add_pending_marks, flush_pending_marks
from litkit.index.dedup import add_with_ids_dedup
from litkit.index.io import faiss_save
from litkit.concurrent.locking import FileLock

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


def _load_segment(path: Path) -> dict:
    """Load a segment file and return its contents.
    
    Args:
        path: Path to .npz segment file
    
    Returns:
        Dictionary with arrays from the segment
    """
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def ingest_paper_segments(
    conn: sqlite3.Connection,
    paper_index: faiss.Index,
    outdir: Path,
    faiss_lock_path: Path,
    paper_index_path: Path,
    *,
    save_every: int = 2,
    paper_id_map: dict[str, int] | None = None,
) -> int:
    """Ingest paper embedding segments into FAISS index.
    
    Loads segment files, resolves doc_ids to paper IDs using the database,
    adds embeddings to the FAISS index, and marks papers as indexed.
    
    Args:
        conn: Database connection
        paper_index: FAISS index for papers
        outdir: Directory containing segment files
        faiss_lock_path: Path to FAISS lock file
        paper_index_path: Path to save the paper index
        save_every: Save index after every N segments
        paper_id_map: Optional pre-loaded doc_id -> paper_id mapping
    
    Returns:
        Total number of paper vectors added
    """
    outdir = Path(outdir)
    if not outdir.exists():
        return 0
    
    # Find paper segment files
    pattern = f"{PAPER_SEGMENT_PREFIX}_*{SEGMENT_EXTENSION}"
    seg_files = sorted(outdir.glob(pattern))
    
    if not seg_files:
        return 0
    
    # Load paper ID map if not provided
    if paper_id_map is None:
        paper_id_map = preload_paper_id_map(conn)
    
    total_added = 0
    cur = conn.cursor()
    
    for i, seg_path in enumerate(seg_files):
        try:
            data = _load_segment(seg_path)
            doc_ids = data["doc_ids"]
            embeddings = data["embeddings"].astype("float32")
            
            # Resolve doc_ids to paper IDs
            ids = []
            valid_rows = []
            for j, doc_id in enumerate(doc_ids):
                pid = paper_id_map.get(str(doc_id))
                if pid is not None:
                    ids.append(pid)
                    valid_rows.append(j)
                else:
                    _eprint(
                        f"[ingest] WARNING: unknown doc_id {doc_id}, skipping"
                    )
            
            if not ids:
                seg_path.unlink()
                continue
            
            # Filter to valid rows
            X = embeddings[valid_rows]
            
            # Add to FAISS with dedup
            with FileLock(faiss_lock_path):
                added, ids_added = add_with_ids_dedup(paper_index, ids, X)
                if added > 0:
                    add_pending_marks("papers", [int(i) for i in ids_added])
                    total_added += added
                
                # Periodic save
                if (i + 1) % save_every == 0:
                    faiss_save(paper_index, paper_index_path)
                    flush_pending_marks(cur)
                    conn.commit()
            
            # Remove consumed segment
            seg_path.unlink()
            _eprint(f"[ingest] Ingested {added} papers from {seg_path.name}")
            
        except Exception as e:
            _eprint(f"[ingest] ERROR processing {seg_path}: {e}")
            continue
    
    # Final flush
    with FileLock(faiss_lock_path):
        faiss_save(paper_index, paper_index_path)
    flush_pending_marks(cur)
    conn.commit()
    
    _eprint(f"[ingest] Total: {total_added} paper vectors ingested")
    return total_added


def ingest_chunk_segments(
    conn: sqlite3.Connection,
    chunk_index: faiss.Index,
    outdir: Path,
    faiss_lock_path: Path,
    chunk_index_path: Path,
    *,
    save_every: int = 2,
    paper_id_map: dict[str, int] | None = None,
    chunk_id_map: dict[tuple[int, int], int] | None = None,
) -> int:
    """Ingest chunk embedding segments into FAISS index.
    
    Loads segment files, resolves (doc_id, ord) to chunk IDs using the
    database, adds embeddings to the FAISS index, and marks chunks as indexed.
    
    Args:
        conn: Database connection
        chunk_index: FAISS index for chunks
        outdir: Directory containing segment files
        faiss_lock_path: Path to FAISS lock file
        chunk_index_path: Path to save the chunk index
        save_every: Save index after every N segments
        paper_id_map: Optional pre-loaded doc_id -> paper_id mapping
        chunk_id_map: Optional pre-loaded (paper_id, ord) -> chunk_id mapping
    
    Returns:
        Total number of chunk vectors added
    """
    outdir = Path(outdir)
    if not outdir.exists():
        return 0
    
    # Find chunk segment files
    pattern = f"{CHUNK_SEGMENT_PREFIX}_*{SEGMENT_EXTENSION}"
    seg_files = sorted(outdir.glob(pattern))
    
    if not seg_files:
        return 0
    
    # Load mappings if not provided
    if paper_id_map is None:
        paper_id_map = preload_paper_id_map(conn)
    if chunk_id_map is None:
        chunk_id_map = preload_chunk_id_map(conn)
    
    total_added = 0
    cur = conn.cursor()
    
    for i, seg_path in enumerate(seg_files):
        try:
            data = _load_segment(seg_path)
            doc_ids = data["doc_ids"]
            ords = data["ords"]
            embeddings = data["embeddings"].astype("float32")
            
            # Resolve (doc_id, ord) to chunk IDs
            ids = []
            valid_rows = []
            for j, (doc_id, ord_) in enumerate(zip(doc_ids, ords)):
                pid = paper_id_map.get(str(doc_id))
                if pid is None:
                    continue
                cid = chunk_id_map.get((pid, int(ord_)))
                if cid is not None:
                    ids.append(cid)
                    valid_rows.append(j)
            
            if not ids:
                seg_path.unlink()
                continue
            
            # Filter to valid rows
            X = embeddings[valid_rows]
            
            # Add to FAISS with dedup
            with FileLock(faiss_lock_path):
                added, ids_added = add_with_ids_dedup(chunk_index, ids, X)
                if added > 0:
                    add_pending_marks("chunks", [int(i) for i in ids_added])
                    total_added += added
                
                # Periodic save
                if (i + 1) % save_every == 0:
                    faiss_save(chunk_index, chunk_index_path)
                    flush_pending_marks(cur)
                    conn.commit()
            
            # Remove consumed segment
            seg_path.unlink()
            _eprint(f"[ingest] Ingested {added} chunks from {seg_path.name}")
            
        except Exception as e:
            _eprint(f"[ingest] ERROR processing {seg_path}: {e}")
            continue
    
    # Final flush
    with FileLock(faiss_lock_path):
        faiss_save(chunk_index, chunk_index_path)
    flush_pending_marks(cur)
    conn.commit()
    
    _eprint(f"[ingest] Total: {total_added} chunk vectors ingested")
    return total_added
