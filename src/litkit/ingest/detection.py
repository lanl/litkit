# litkit/ingest/detection.py
"""Tar file type detection for litkit."""

from __future__ import annotations

from pathlib import Path


def is_uncompressed_tar(tar_path: Path) -> bool:
    """Detect if a tar file is uncompressed (no .gz/.bz2/.xz suffix).
    
    Uncompressed tars can use parallel XML parsing because the archive
    format allows random access within the file.
    
    Args:
        tar_path: Path to the tar file
    
    Returns:
        True if the tar is uncompressed (plain .tar), False otherwise
    """
    name = tar_path.name.lower()
    return name.endswith(".tar") and not any(
        name.endswith(ext)
        for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz")
    )
