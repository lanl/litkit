# src/litkit/core/types.py
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Chunk:
    id: str
    paper_id: str
    text: str
    section: Optional[str] = None
    order: int = 0


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    abstract: Optional[str]
    chunks: Sequence[Chunk] = ()
