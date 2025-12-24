# src/litkit/retrieval/lexical.py
"""Lexical front-loading for RAG retrieval.

Lexical front-loading improves recall for queries containing:
- Technical terms with digits (e.g., "H1N1", "SARS-CoV-2", "IL-6")
- Hyphenated compound terms (e.g., "angiotensin-converting")
- Very long specific terms (e.g., "methyltransferase")

These terms often have exact matches in the corpus that ANN may miss.

Design:
- LexicalConfig: narrow, stable configuration bag
- LexicalResult: carries scores for future BM25-ish ranking
- LexicalBackend: protocol for swapping implementations (SQLite → FTS5 → etc)
- find_rare_terms: pure function for term extraction
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from litkit.retrieval.helpers import (
    query_terms,
    sqlite_norm_expr,
    escape_like,
    normalize_for_search_py,
)
from litkit.progress import eprint as _eprint

if TYPE_CHECKING:
    from sqlite3 import Connection


# Stopwords for query term filtering (shared with helpers.py)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your",
    "about", "does", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "how", "are", "is", "was", "were", "be", "been",
    "being", "of", "on", "in", "to", "a", "an", "as", "by", "at", "it",
    "its", "their", "them", "we", "you", "i",
}


@dataclass
class LexicalConfig:
    """Configuration for lexical front-loading behavior.
    
    This is a dumb configuration bag - no behavior lives here.
    The caller (search_chunks_constrained) decides:
    - Whether global lexical is allowed (based on Stage-1 sparsity)
    - The actual cap given k
    
    Attributes:
        enabled: Master kill-switch for lexical search
        cap: Max lexical results (None => caller computes from k)
        limit: SQL LIMIT for lexical scan
        allow_global: Force global scope (ignore candidate filter)
        min_term_length: Threshold for long-term fallback
        max_terms: Cap # of rare terms to use
        require_rare_terms: If False, allow long-only matches
    """
    enabled: bool = True
    cap: int | None = None
    limit: int = 200
    allow_global: bool = False
    min_term_length: int = 9
    max_terms: int = 8
    require_rare_terms: bool = True


@dataclass
class LexicalResult:
    """Result of lexical front-loading.
    
    Attributes:
        chunk_ids: Matching chunk IDs in score-descending order
        scores: chunk_id -> score (1.0 for now, BM25-ish later)
        terms_used: Which rare terms were searched
        scope: "candidates" or "global"
    """
    chunk_ids: list[int] = field(default_factory=list)
    scores: dict[int, float] = field(default_factory=dict)
    terms_used: list[str] = field(default_factory=list)
    scope: str = "candidates"


class LexicalBackend(Protocol):
    """Protocol for lexical search backends.
    
    Implement this to swap SQLite LIKE → FTS5 → Tantivy → etc.
    """
    
    def search(
        self,
        terms: list[str],
        candidate_papers: set[int] | None,
        config: LexicalConfig,
    ) -> LexicalResult:
        """Execute lexical search for rare terms.
        
        Args:
            terms: Rare terms to search for
            candidate_papers: Paper IDs to scope to (None = global)
            config: Lexical configuration
        
        Returns:
            LexicalResult with matching chunks, scores, and metadata
        """
        ...


def find_rare_terms(question: str, config: LexicalConfig) -> list[str]:
    """Extract rare/technical terms that benefit from exact match.
    
    This is a pure function - no DB dependencies, easy to test.
    
    Prioritizes:
    1. Terms with digits (H1N1, IL-6, CD4)
    2. Terms with LIKE-sensitive chars (hyphens, underscores, %, \\)
    3. Very long terms (config.min_term_length+) as fallback
    
    Args:
        question: The user's question
        config: LexicalConfig with min_term_length and max_terms
    
    Returns:
        List of rare terms, capped at config.max_terms
    """
    # Use shared query_terms for base extraction
    terms = query_terms(question)
    
    # Find "rare" terms: digits or LIKE-sensitive chars
    rare = [
        t for t in terms
        if any(ch.isdigit() for ch in t)
        or any(ch in "-_%\\" for ch in t)
    ]
    
    # Fallback: long terms if no rare terms and not requiring rare
    if not rare:
        if config.require_rare_terms:
            # Still try long terms as fallback
            rare = [t for t in terms if len(t) >= config.min_term_length]
        else:
            rare = [t for t in terms if len(t) >= config.min_term_length]
    
    return rare[:config.max_terms]


class SqliteLexicalBackend:
    """SQLite LIKE-based lexical search backend.
    
    This is the default backend. Uses normalized LIKE patterns
    for unicode-aware substring matching.
    
    Future backends might use FTS5, Tantivy, or external indices.
    """
    
    def __init__(
        self,
        db_conn: "Connection",
        load_temp_candidates,
    ):
        """Initialize with DB connection and temp table loader.
        
        Args:
            db_conn: SQLite connection
            load_temp_candidates: Function to populate cand_papers table
        """
        self.db_conn = db_conn
        self.load_temp_candidates = load_temp_candidates
        self._warn_once = False
    
    def search(
        self,
        terms: list[str],
        candidate_papers: set[int] | None,
        config: LexicalConfig,
    ) -> LexicalResult:
        """Execute LIKE-based lexical search.
        
        SQL ORDER BY priorities:
        1. Abstract chunks (ord = -1) first
        2. Title matches
        3. PMID presence (indexed in PubMed)
        4. PMCID presence
        5. Chunk ID (stable ordering)
        """
        if not terms or not config.enabled:
            return LexicalResult(scope="global" if candidate_papers is None else "candidates")
        
        scope = "global" if candidate_papers is None else "candidates"
        
        # Build LIKE clause with normalization
        norm = sqlite_norm_expr("text")
        like_parts = [f"{norm} LIKE ? ESCAPE '\\'"] * len(terms)
        like_clause = " OR ".join(like_parts)
        params = [
            f"%{escape_like(normalize_for_search_py(t))}%"
            for t in terms
        ]
        
        # Title hint for sorting (prefer chunks from papers with matching title)
        title_hint = normalize_for_search_py(terms[0] if terms else "")
        title_like_param = f"%{escape_like(title_hint)}%" if title_hint else "%"
        
        cur = self.db_conn.cursor()
        chunk_ids: list[int] = []
        
        try:
            if scope == "global":
                cur.execute(
                    f"""
                    SELECT c.id
                    FROM chunks c
                    JOIN papers p ON p.id = c.paper_id
                    WHERE ({like_clause.replace('text', 'c.text')})
                    ORDER BY (c.ord = -1) DESC,
                            ({sqlite_norm_expr('p.title')} LIKE ? ESCAPE '\\') DESC,
                            (p.pmid IS NOT NULL) DESC,
                            (p.pmcid IS NOT NULL) DESC,
                            c.id ASC
                    LIMIT ?
                    """,
                    params + [title_like_param, config.limit],
                )
            else:
                # Scope to candidate papers
                self.load_temp_candidates(self.db_conn, list(candidate_papers))
                cur.execute(
                    f"""
                    SELECT c.id
                    FROM chunks c
                    JOIN papers p ON p.id = c.paper_id
                    WHERE ({like_clause.replace('text', 'c.text')})
                    AND c.paper_id IN (SELECT id FROM cand_papers)
                    ORDER BY (c.ord = -1) DESC,
                            ({sqlite_norm_expr('p.title')} LIKE ? ESCAPE '\\') DESC,
                            (p.pmid IS NOT NULL) DESC,
                            (p.pmcid IS NOT NULL) DESC,
                            c.id ASC
                    LIMIT ?
                    """,
                    params + [title_like_param, config.limit],
                )
            chunk_ids = [row[0] for row in cur.fetchall()]
            
        except sqlite3.OperationalError as e:
            if not self._warn_once:
                self._warn_once = True
                logging.warning(
                    "[lexical] disabled: %s: %s (scope=%s)",
                    e.__class__.__name__, e, scope,
                )
            return LexicalResult(terms_used=terms, scope=scope)
        
        # Build scores dict (1.0 for all matches - placeholder for BM25)
        scores = {cid: 1.0 for cid in chunk_ids}
        
        return LexicalResult(
            chunk_ids=chunk_ids,
            scores=scores,
            terms_used=terms,
            scope=scope,
        )


def lexical_search(
    terms: list[str],
    *,
    config: LexicalConfig,
    backend: LexicalBackend,
    candidate_papers: set[int] | None = None,
) -> LexicalResult:
    """Main entry point for lexical search.
    
    Delegates to the backend implementation. This thin wrapper
    exists to provide a stable interface regardless of backend.
    
    Args:
        terms: Rare terms from find_rare_terms()
        config: LexicalConfig
        backend: LexicalBackend implementation
        candidate_papers: Paper IDs to scope to (None = global)
    
    Returns:
        LexicalResult from the backend
    """
    return backend.search(terms, candidate_papers, config)


def merge_lexical_and_ann(
    lexical: LexicalResult,
    ann_ranked: list[tuple[int, float]],
    k: int,
    lexical_cap: int,
) -> list[int]:
    """Interleave lexical and ANN results.
    
    Strategy: alternate lexical/ANN, dedupe, cap lexical contribution.
    Lexical results are already in score-descending order.
    
    Args:
        lexical: LexicalResult from lexical_search()
        ann_ranked: ANN results as (chunk_id, distance) tuples
        k: Final number of chunks to return
        lexical_cap: Max lexical items to include
    
    Returns:
        Merged list of chunk IDs, up to k items
    """
    lexical_ids = lexical.chunk_ids[:lexical_cap]
    
    seen: set[int] = set()
    merged: list[int] = []
    i = j = 0
    
    while len(merged) < k and (i < len(lexical_ids) or j < len(ann_ranked)):
        # Take from lexical
        if i < len(lexical_ids):
            cid = lexical_ids[i]
            i += 1
            if cid not in seen:
                seen.add(cid)
                merged.append(cid)
        
        if len(merged) >= k:
            break
        
        # Take from ANN
        if j < len(ann_ranked):
            cid, _ = ann_ranked[j]
            j += 1
            if cid not in seen:
                seen.add(cid)
                merged.append(cid)
    
    return merged[:k]
