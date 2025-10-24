"""formatting.answers — post-processing helpers for RAG answers.

What this module does
---------------------
- Collapse chunk-level numeric citations in an LLM answer to *document-level* citations.
- Build a plain-text "References" block from those document-level citations.

Public API
----------
- normalize_answer_and_build_refs(answer, selected_chunks) -> (normalized_answer, doc_refs)
- render_references(doc_refs, header=True) -> str

Inputs & conventions
--------------------
- ``selected_chunks`` is the Stage-2 list where the **1-based index** matches the in-text [n] citations.
- Each chunk dict may include: ``doi``, ``pmcid``, ``pmid``, ``paper_id``, ``paper_title``/``title``, ``year``.

Document de-duplication
-----------------------
Chunks are merged into documents using the first available key in this order:
``doi`` > ``pmcid`` > ``pmid`` > ``paper_id`` > ``paper_title``/``title`` (lower-cased).
If none is present, a per-row fallback is used to avoid collisions.

Recognition details
-------------------
- Citation blocks like ``[1]``, ``[1,2]``, ``[1-3]`` are supported, as well as the CJK-style ``【1】``.
- Ranges accept hyphen ``-``, en dash ``–``, and em dash ``—``; commas may be ASCII or common CJK variants.

Outputs
-------
- ``normalize_answer_and_build_refs`` returns:
  * ``normalized_answer``: the answer with citations renumbered to document IDs.
  * ``doc_refs``: a list of representative dicts in order of first in-text mention, each with:
    ``docnum``, ``pmcid``, ``pmid``, ``doi``, ``paper_title``, ``year``.

Notes:
-----
- This module has no dependencies beyond the Python standard library.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def _doc_key(ch: dict[str, Any]) -> str:
    doi = (ch.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    pmcid = (ch.get("pmcid") or "").strip().upper()
    if pmcid:
        return f"pmcid:{pmcid}"
    pmid = (ch.get("pmid") or "").strip()
    if pmid:
        return f"pmid:{pmid}"
    pid = ch.get("paper_id")  # unique in our DB; avoids title-collision merges
    if pid is not None:
        return f"paper:{pid}"
    title = (ch.get("paper_title") or ch.get("title") or "").strip().lower()
    return f"title:{title}" if title else f"rowid:{id(ch)}"


def _expand_ranges(nums: str) -> list[int]:
    """Turn strings like '1-3, 5, 7–8' into [1,2,3,5,7,8].
    Accepts hyphen '-', en dash '–', and em dash '—' as range separators.
    Accepts ASCII comma ',' and common CJK commas.
    """
    out: list[int] = []
    for part in re.split(r"[,\u3001\uFF0C]", nums):  # comma, CJK comma variants
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*[-–—]\s*(\d+)$", part)  # hyphen/en dash/em dash
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            out.extend(range(lo, hi + 1))
        else:
            if part.isdigit():
                out.append(int(part))
    return out


# Matches [1], [1,2], [1-3], and also 【1】/【1,2】 styles.
_CITATION_BLOCK = re.compile(
    r"(?:\[(?P<a>[0-9,\s\-–—\u3001\uFF0C]+)\])|(?:【(?P<b>[0-9,\s\-–—\u3001\uFF0C]+)】)"
)


def _unique_preserve(seq: Iterable[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_answer_and_build_refs(
    answer: str, selected_chunks: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Convert chunk-wise numeric citations in `answer` to doc-wise citations and
    return a doc-wise bibliography list (ordered by first mention).

    Parameters
    ----------
    answer : str
        The model's answer string containing bracketed numeric citations (e.g., "[1,2]").
    selected_chunks : List[dict]
        The chunks list where the 1-based index corresponds to the numbers used in `answer`.

    Returns:
    -------
    normalized_answer : str
        The answer with citations renumbered to refer to unique documents (papers).
    doc_refs : List[dict]
        One representative entry per unique paper, ordered by first in-text mention.
        Each dict includes: "docnum", "pmcid", "pmid", "doi", "paper_title", "year".
    """
    # Map chunk index -> document key
    idx2key: dict[int, str] = {i + 1: _doc_key(ch) for i, ch in enumerate(selected_chunks)}
    key2docnum: dict[str, int] = {}
    chunk2docnum: dict[int, int] = {}

    # First pass: discover order of first mentions (assign doc numbers)
    def _discover(match: re.Match) -> str:
        nums = match.group("a") or match.group("b") or ""
        for n in _expand_ranges(nums):
            if 1 <= n <= len(selected_chunks):
                k = idx2key[n]
                if k not in key2docnum:
                    key2docnum[k] = len(key2docnum) + 1
                chunk2docnum[n] = key2docnum[k]
        return match.group(0)  # no change in pass 1

    _ = _CITATION_BLOCK.sub(_discover, answer)

    # Second pass: replace with doc-wise unique numbers
    def _replace(match: re.Match) -> str:
        nums = match.group("a") or match.group("b") or ""
        mapped = []
        for n in _expand_ranges(nums):
            dn = chunk2docnum.get(n)
            if dn is not None:
                mapped.append(dn)
        mapped = _unique_preserve(mapped)
        if not mapped:
            return match.group(0)  # leave as-is if nothing mapped
        return "[" + ", ".join(str(x) for x in mapped) + "]"

    normalized_answer = _CITATION_BLOCK.sub(_replace, answer)

    # Build representative refs in docnum order
    rep_by_key: dict[str, dict[str, Any]] = {}
    for ch in selected_chunks:
        k = _doc_key(ch)
        if k not in rep_by_key:
            rep_by_key[k] = ch

    # Sort keys by docnum
    inv = sorted(((docnum, key) for key, docnum in key2docnum.items()), key=lambda x: x[0])
    doc_refs: list[dict[str, Any]] = []
    for docnum, key in inv:
        ch = rep_by_key.get(key, {})
        doc_refs.append(
            {
                "docnum": docnum,
                "pmcid": ch.get("pmcid"),
                "pmid": ch.get("pmid"),
                "doi": ch.get("doi"),
                "paper_title": ch.get("paper_title") or ch.get("title"),
                "year": ch.get("year"),
            }
        )
    return normalized_answer, doc_refs


def render_references(doc_refs: list[dict[str, Any]]) -> str:
    """Render a simple bibliography block from `doc_refs` returned by
    normalize_answer_and_build_refs(). Always includes a header.
    """
    # Accept list or dict-like (defensive)
    if not doc_refs:
        return "References\n"

    # If someone passes a dict keyed by docnum, normalize to a list of values
    if isinstance(doc_refs, dict):
        items = list(doc_refs.values())
    else:
        items = list(doc_refs)

    # Sort by docnum if present
    def _doc_key(d: dict[str, Any]) -> int:
        try:
            return int(d.get("docnum", 10**9))
        except Exception:
            return 10**9

    items.sort(key=_doc_key)

    lines: list[str] = []
    lines.append("References")

    for d in items:
        n = d.get("docnum", "?")
        title = (d.get("paper_title") or d.get("title") or "untitled").strip()
        year = d.get("year")
        year_str = f" ({year})" if year else ""

        id_parts: list[str] = []
        doi = (d.get("doi") or "").strip()
        pmcid = (d.get("pmcid") or "").strip()
        pmid = (d.get("pmid") or "").strip()
        if doi:
            id_parts.append(f"DOI: {doi}")
        if pmcid:
            id_parts.append(f"PMCID: {pmcid}")
        if pmid:
            id_parts.append(f"PMID: {pmid}")
        id_str = (". " + ", ".join(id_parts)) if id_parts else ""

        lines.append(f"[{n}] {title}{year_str}.{id_str}")

    return "\n".join(lines)


# Public API
__all__ = [
    "normalize_answer_and_build_refs",
    "render_references",
]
