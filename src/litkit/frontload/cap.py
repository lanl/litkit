"""frontload.cap — helper for lexical front loading

What this module does
---------------------
- Imposes a limit on the number of text chunks used per paper in lexical front loading.

Public API
----------
- cap_chunks_per_paper(chunk_ids_in_rank_order, chunkid_to_paperid, max_per_paper=3) -> list[int]

Notes:
-----
- This module has no dependencies beyond the Python standard library.
"""

from __future__ import annotations

from typing import Any


def cap_chunks_per_paper(
    chunk_ids_in_rank_order: list[int], chunkid_to_paperid: dict[int, Any], max_per_paper: int = 3
) -> list[int]:
    """Helper for Stage-2: enforce a per-paper cap over a ranked list of chunk IDs.

    Returns a filtered list preserving original order.
    """
    count: dict[Any, int] = {}
    out: list[int] = []
    for cid in chunk_ids_in_rank_order:
        pid = chunkid_to_paperid.get(cid)
        c = count.get(pid, 0)
        if c < max_per_paper:
            out.append(cid)
            count[pid] = c + 1
    return out


# Public API
__all__ = [
    "cap_chunks_per_paper",
]
