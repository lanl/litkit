# src/litkit/build/helpers.py
"""Helper functions for the build pipeline.

Functions:
- pack_paragraphs: Greedy text chunking with overlap
- dedupe_papers_with_doc_ids: Deduplicate paper buffers
- dedupe_chunks_with_doc_ids: Deduplicate chunk buffers
- ensure_parent: Create parent directory if needed
- maybe_fsync_dir: Optional directory fsync for NFS safety
"""

from __future__ import annotations

import os
from pathlib import Path

# Default chunking parameters (can be overridden via args)
CHUNK_TARGET_CHARS = 1200
BODY_MIN_CHARS = 300
CHUNK_OVERLAP_CHARS = 200


def pack_paragraphs(
    paras: list[str],
    max_chars: int = CHUNK_TARGET_CHARS,
    min_chars: int = BODY_MIN_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Greedy pack paragraphs, then add a small character overlap between
    consecutive chunks to reduce claim-splitting.
    
    Args:
        paras: List of paragraph strings to pack
        max_chars: Target maximum characters per chunk
        min_chars: Minimum characters per chunk (short chunks merge with previous)
        overlap_chars: Character overlap between consecutive chunks
    
    Returns:
        List of packed chunk strings
    """
    chunks: list[str] = []
    buf: list[str] = []
    total = 0

    def _flush_buf() -> None:
        nonlocal chunks, buf, total
        if not buf:
            return
        cur = " ".join(buf)
        if len(cur) < min_chars and chunks:
            chunks[-1] = chunks[-1] + " " + cur
        else:
            chunks.append(cur)
        buf, total = [], 0

    for p in paras:
        if buf and total + len(p) + 1 > max_chars:
            _flush_buf()
        buf.append(p)
        total += len(p) + 1
    _flush_buf()

    if overlap_chars > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap_chars:]
            out.append((tail + " " + chunks[i]).strip())
        chunks = out
    return chunks


def dedupe_papers_with_doc_ids(
    ids: list[int], texts: list[str], doc_ids: list[str]
) -> tuple[list[int], list[str], list[str]]:
    """Keep the first occurrence of each id; return aligned id/text/doc_id lists.
    
    Args:
        ids: Paper database IDs
        texts: Paper text (title + abstract)
        doc_ids: Content-addressed document IDs
    
    Returns:
        Tuple of (unique_ids, unique_texts, unique_doc_ids)
    """
    out_ids: list[int] = []
    out_texts: list[str] = []
    out_doc_ids: list[str] = []
    seen: set[int] = set()
    for i, t, d in zip(ids, texts, doc_ids, strict=False):
        if i in seen:
            continue
        seen.add(i)
        out_ids.append(i)
        out_texts.append(t)
        out_doc_ids.append(d)
    return out_ids, out_texts, out_doc_ids


def dedupe_chunks_with_doc_ids(
    ids: list[int], texts: list[str], paper_doc_ids: list[str], ords: list[int]
) -> tuple[list[int], list[str], list[str], list[int]]:
    """Keep the first occurrence of each id; return aligned id/text/paper_doc_id/ord lists.
    
    Args:
        ids: Chunk database IDs
        texts: Chunk text content
        paper_doc_ids: Parent paper document IDs
        ords: Chunk ordinals within papers
    
    Returns:
        Tuple of (unique_ids, unique_texts, unique_paper_doc_ids, unique_ords)
    """
    out_ids: list[int] = []
    out_texts: list[str] = []
    out_doc_ids: list[str] = []
    out_ords: list[int] = []
    seen: set[int] = set()
    for i, t, d, o in zip(ids, texts, paper_doc_ids, ords, strict=False):
        if i in seen:
            continue
        seen.add(i)
        out_ids.append(i)
        out_texts.append(t)
        out_doc_ids.append(d)
        out_ords.append(o)
    return out_ids, out_texts, out_doc_ids, out_ords


def ensure_parent(path: Path) -> None:
    """Ensure parent directory of `path` exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_fsync_dir(p: Path) -> None:
    """Optional: fsync the containing directory for extra safety on some NFS setups.
    
    Controlled by env var LITKIT_SEGMENT_FSYNC_DIR (default "1" = enabled).
    """
    if os.environ.get("LITKIT_SEGMENT_FSYNC_DIR", "1") != "1":
        return
    try:
        dfd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass
