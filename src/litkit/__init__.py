# src/litkit/__init__.py
"""litkit package initialization"""

from __future__ import annotations

import logging
import sys
from importlib.metadata import PackageNotFoundError, version

from .core.types import Chunk, Document  # dataclasses

# Block unsupported platforms.
if sys.platform.startswith("win"):
    raise RuntimeError("litkit supports macOS and Linux only.")

# Be quiet by default; let host app configure handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())


# Single source of truth: version comes from package metadata.
try:

    __version__ = version("litkit")
except PackageNotFoundError:
    __version__ = "0+unknown"


__all__ = [
    "__version__",
    "Document",
    "Chunk",
]
