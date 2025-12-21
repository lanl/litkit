# src/litkit/retrieval/helpers.py
"""Helper functions for retrieval operations.

Contains query term extraction, text normalization, and SQL helpers.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path


# Stopwords for query term filtering
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your",
    "about", "does", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "how", "are", "is", "was", "were", "be", "been",
    "being", "of", "on", "in", "to", "a", "an", "as", "by", "at", "it",
    "its", "their", "them", "we", "you", "i",
}


def query_terms(s: str) -> list[str]:
    """Extract meaningful query terms from a string.
    
    Extracts alphanumeric tokens, lowercases them, and filters out
    short/common words. Keeps tokens that are:
    - 5+ characters long
    - All uppercase and 3+ chars
    - Contain digits
    - Contain underscores or hyphens
    
    Args:
        s: Input string (typically a question)
    
    Returns:
        List of unique query terms in original order
    """
    words = re.findall(r"[A-Za-z0-9_-]{3,}", s.lower())
    out = []
    for w in words:
        if w in _STOPWORDS:
            continue
        if (
            len(w) >= 5
            or (w.isupper() and len(w) >= 3)
            or any(ch.isdigit() for ch in w)
            or "_" in w
            or "-" in w
        ):
            out.append(w)
    # Dedupe while preserving order
    seen = set()
    uniq = []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq


def sqlite_norm_expr(field: str = "text") -> str:
    """Build a SQL expression that normalizes unicode variants.
    
    Normalizes:
    - Hyphen/minus variants to ASCII '-'
    - Subscript digits to ASCII digits
    - Lowercase
    
    Args:
        field: SQL column name to wrap
    
    Returns:
        SQL expression string
    """
    f = f"lower({field})"
    # Hyphen/minus variants: U+2010..U+2014, U+2212, plus soft hyphen
    for ch, repl in [
        ("\u00ad", ""),      # soft hyphen (strip)
        ("\u2010", "-"),     # hyphen
        ("\u2011", "-"),     # non-breaking hyphen
        ("\u2012", "-"),     # figure dash
        ("\u2013", "-"),     # en dash
        ("\u2014", "-"),     # em dash
        ("\u2212", "-"),     # minus sign
    ]:
        f = f"replace({f}, '{ch}', '{repl}')"
    # Subscript digits → ASCII
    subs = "₀₁₂₃₄₅₆₇₈₉"
    for d_sub, d in zip(subs, "0123456789", strict=False):
        f = f"replace({f}, '{d_sub}', '{d}')"
    return f


def escape_like(s: str, esc: str = "\\") -> str:
    """Escape special characters for SQL LIKE patterns.
    
    Args:
        s: Input string
        esc: Escape character (default backslash)
    
    Returns:
        Escaped string safe for LIKE
    """
    # Order matters: escape the escape char first
    s = s.replace(esc, esc + esc)
    s = s.replace("%", esc + "%")
    s = s.replace("_", esc + "_")
    return s


def normalize_for_search_py(s: str) -> str:
    """Python-side text normalization matching SQLite normalization.
    
    Applies the same transformations as sqlite_norm_expr() but in Python,
    ensuring query terms match what's in the database.
    
    Args:
        s: Input string
    
    Returns:
        Normalized string
    """
    s = unicodedata.normalize("NFKC", s).lower()
    for ch, repl in [
        ("\u00ad", ""),
        ("\u2010", "-"),
        ("\u2011", "-"),
        ("\u2012", "-"),
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("\u2212", "-"),
    ]:
        s = s.replace(ch, repl)
    trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return s.translate(trans)


def avg_chunks_for_papers(
    pids: list[int],
    *,
    db_path: Path,
    connect_db,
    load_temp_candidates,
) -> float:
    """Calculate average chunks per paper for a list of paper IDs.
    
    Args:
        pids: List of paper IDs
        db_path: Path to SQLite database
        connect_db: Function to connect to DB
        load_temp_candidates: Function to load temp candidates table
    
    Returns:
        Average number of chunks per paper (0.0 if no papers)
    """
    if not pids:
        return 0.0
    conn = connect_db(db_path)
    try:
        load_temp_candidates(conn, pids)
        rows = conn.execute(
            """
            SELECT cp.id, COUNT(c.id)
            FROM cand_papers cp
            LEFT JOIN chunks c ON c.paper_id = cp.id
            GROUP BY cp.id
            """
        ).fetchall()
        return 0.0 if not rows else (sum(n for _, n in rows) / float(len(pids)))
    finally:
        conn.close()
