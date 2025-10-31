# src/litkit/core/types.py
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    paper_id: str
    text: str
    section: str | None = None
    order: int = 0


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    abstract: str | None
    chunks: Sequence[Chunk] = ()
