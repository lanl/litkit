# litkit/pipeline/helpers.py
"""Helper utilities for document processing pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def dedupe_ids_and_texts(
    ids: list[int],
    texts: list[str],
) -> tuple[list[int], list[str]]:
    """Keep the first occurrence of each ID; return aligned id/text lists.
    
    Args:
        ids: List of integer IDs (may contain duplicates)
        texts: List of texts aligned with ids
    
    Returns:
        Tuple of (deduped_ids, deduped_texts) with duplicates removed
    """
    seen: set[int] = set()
    out_ids: list[int] = []
    out_texts: list[str] = []
    
    for i, text in zip(ids, texts):
        if i not in seen:
            seen.add(i)
            out_ids.append(i)
            out_texts.append(text)
    
    return out_ids, out_texts


def check_file_processed(cur: sqlite3.Cursor, file_path: str) -> bool:
    """Check if a file has already been processed.
    
    Args:
        cur: Database cursor
        file_path: Canonical file path to check
    
    Returns:
        True if the file has been processed, False otherwise
    """
    result = cur.execute(
        "SELECT 1 FROM files WHERE path=?",
        (file_path,)
    ).fetchone()
    return result is not None


def canonical_tar_path(tar_path: Path, member_name: str) -> str:
    """Build a canonical path string for a tar member.
    
    Format: "tar:///path/to/file.tar!/member.nxml"
    
    This path is globally unique across all shards and stable across
    DB merges, making it suitable for content-addressed segment storage.
    
    Args:
        tar_path: Path to the tar archive
        member_name: Name of the member within the tar
    
    Returns:
        Canonical path string
    """
    return f"tar://{tar_path}!/{member_name}"
