"""Tests for retrieval helpers: query terms, text normalization, LIKE escaping,
lexical search, ANN/lexical merging, the per-paper cap and nprobe selection."""

import sqlite3

import pytest

from litkit.db import connect_db, init_db, load_temp_candidates
from litkit.frontload.cap import cap_chunks_per_paper
from litkit.index.search import pick_nprobe
from litkit.retrieval.helpers import (
    escape_like,
    normalize_for_search_py,
    query_terms,
    sqlite_norm_expr,
)
from litkit.retrieval.lexical import (
    LexicalConfig,
    LexicalResult,
    SqliteLexicalBackend,
    find_rare_terms,
    merge_lexical_and_ann,
)

# ------------------------------------------------------------------- query terms


def test_query_terms_keeps_long_digit_and_hyphen_terms_in_order():
    q = "What does IL-6 do to methyltransferase activity in CD4 cells, and il-6 again?"
    # 'what', 'does', 'do', 'to', 'in', 'and' are stopwords or too short; 'cells' and
    # 'again' are 5+ letters; duplicates ('il-6') are dropped.
    assert query_terms(q) == ["il-6", "methyltransferase", "activity", "cd4", "cells", "again"]


def test_query_terms_keeps_short_hyphen_and_underscore_terms():
    # 'p-gp' and 'a_b' are under 5 characters with no digits: kept for the '-' and '_'.
    assert query_terms("Does P-gp efflux a_b drugs?") == ["p-gp", "efflux", "a_b", "drugs"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#9: acronyms are dropped")
def test_query_terms_keeps_uppercase_acronyms():
    assert "hiv" in query_terms("Does HIV bind CD4?")


def test_find_rare_terms_prefers_digits_and_hyphens():
    cfg = LexicalConfig()
    assert find_rare_terms("Does IL-6 bind CD4 in methyltransferase pathways?", cfg) == [
        "il-6",
        "cd4",
    ]


def test_find_rare_terms_falls_back_to_long_terms_and_caps():
    # No digits or hyphens: terms of min_term_length (9) or more are used.
    assert find_rare_terms("explain methyltransferase pathways", LexicalConfig()) == [
        "methyltransferase"
    ]
    many = " ".join(f"t{i}x" for i in range(20))
    assert len(find_rare_terms(many, LexicalConfig(max_terms=3))) == 3


# -------------------------------------------------------------- normalization


@pytest.fixture
def sql_norm(tmp_path):
    """Evaluate sqlite_norm_expr on a connection opened the way retrieval opens it,
    so a fix that registers a SQL function on litkit's connections still works here."""
    db = tmp_path / "norm.sqlite3"
    init_db(db).close()
    conn = connect_db(db)

    def norm(s):
        return conn.execute(f"SELECT {sqlite_norm_expr('?')}", (s,)).fetchone()[0]

    yield norm
    conn.close()


@pytest.mark.parametrize(
    "s",
    [
        "IL‐6",  # hyphen
        "IL‑6",  # non-breaking hyphen
        "IL‒6",  # figure dash
        "IL–6",  # en dash
        "IL—6",  # em dash
        "A−B",  # minus sign
        "soft­hyphen",
        "CO₂ and H₂O",  # subscript digits
        "MiXeD CaSe",
    ],
)
def test_python_and_sql_normalization_agree(s, sql_norm):
    """The two implementations must normalize the same way, or terms never match."""
    assert normalize_for_search_py(s) == sql_norm(s)


def test_normalization_expected_values():
    assert normalize_for_search_py("IL–6 CO₂ soft­hyphen") == "il-6 co2 softhyphen"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#20: no NFKC/Unicode lower in SQL")
@pytest.mark.parametrize("s", ["ﬁbrosis", "CD４", "x²", "Épidémie"])
def test_python_and_sql_normalization_agree_beyond_ascii(s, sql_norm):
    assert normalize_for_search_py(s) == sql_norm(s)


@pytest.mark.parametrize(
    ("term", "matches", "non_matches"),
    [
        ("50%", ["dose 50% higher"], ["dose 500 higher"]),
        ("a_b", ["xa_bx"], ["xaXbx"]),
        ("c\\d", ["c\\d"], ["cd", "c/d"]),
    ],
)
def test_escape_like_matches_literally(term, matches, non_matches):
    """Oracle: SQLite itself. An escaped pattern must match only the literal text."""
    conn = sqlite3.connect(":memory:")
    pattern = f"%{escape_like(term)}%"

    def hit(text):
        return conn.execute("SELECT ? LIKE ? ESCAPE '\\'", (text, pattern)).fetchone()[0]

    assert all(hit(t) for t in matches)
    assert not any(hit(t) for t in non_matches)


# ------------------------------------------------------------- lexical backend


@pytest.fixture
def lexical_db(tmp_path):
    conn = init_db(tmp_path / "lex.sqlite3")
    conn.executemany(
        "INSERT INTO papers(id, doc_id, pmid, pmcid, title) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "d1", "1", "PMC1", "Unrelated title"),
            (2, "d2", "2", "PMC2", "IL-6 in fibrosis"),
            (3, "d3", "3", "PMC3", "Another paper"),
        ],
    )
    conn.executemany(
        "INSERT INTO chunks(id, paper_id, ord, text) VALUES (?, ?, ?, ?)",
        [
            (10, 1, 0, "body mentions IL–6 with a Unicode hyphen"),
            (11, 2, 0, "body mentions il-6 in plain ASCII"),
            (12, 2, -1, "abstract chunk about IL-6"),
            (13, 3, 0, "nothing relevant here"),
            (14, 3, 1, "IL-6 again, in paper 3"),
            (15, 1, 1, "IL-60 is a different term but contains the substring"),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


def test_lexical_search_ordering_and_normalization(lexical_db):
    backend = SqliteLexicalBackend(lexical_db, load_temp_candidates)
    res = backend.search(["il-6"], None, LexicalConfig())
    # Order: abstract chunk (ord -1) first, then chunks of the paper whose title
    # matches, then by chunk id. 13 has no match. 15 matches as a substring.
    assert res.chunk_ids == [12, 11, 10, 14, 15]
    assert res.scope == "global"


def test_lexical_search_scoped_to_candidates(lexical_db):
    backend = SqliteLexicalBackend(lexical_db, load_temp_candidates)
    res = backend.search(["il-6"], {3}, LexicalConfig())
    assert res.chunk_ids == [14]
    assert res.scope == "candidates"


def test_lexical_search_respects_limit_and_disable(lexical_db):
    backend = SqliteLexicalBackend(lexical_db, load_temp_candidates)
    assert len(backend.search(["il-6"], None, LexicalConfig(limit=2)).chunk_ids) == 2
    assert backend.search(["il-6"], None, LexicalConfig(enabled=False)).chunk_ids == []
    assert backend.search([], None, LexicalConfig()).chunk_ids == []


# ------------------------------------------------------------------- merging


def test_merge_alternates_dedupes_and_caps_lexical():
    lexical = LexicalResult(chunk_ids=[1, 2, 3, 4])
    ann = [(2, 0.9), (5, 0.8), (6, 0.7), (1, 0.6)]
    # Lexical capped to [1, 2]. Alternate: L1, A2, (L2 dup), A5, A6, (A1 dup).
    assert merge_lexical_and_ann(lexical, ann, k=5, lexical_cap=2) == [1, 2, 5, 6]


def test_merge_stops_at_k():
    lexical = LexicalResult(chunk_ids=[1, 2, 3])
    ann = [(7, 0.9), (8, 0.8), (9, 0.7)]
    assert merge_lexical_and_ann(lexical, ann, k=4, lexical_cap=3) == [1, 7, 2, 8]


def test_merge_with_no_lexical_is_ann_order():
    ann = [(7, 0.9), (8, 0.8), (9, 0.7)]
    assert merge_lexical_and_ann(LexicalResult(), ann, k=10, lexical_cap=5) == [7, 8, 9]


# ------------------------------------------------------------------ per-paper cap


def test_cap_chunks_per_paper_keeps_rank_order():
    ranked = [1, 2, 3, 4, 5, 6]
    paper = {1: "a", 2: "a", 3: "b", 4: "a", 5: "b", 6: "c"}
    assert cap_chunks_per_paper(ranked, paper, max_per_paper=2) == [1, 2, 3, 5, 6]
    assert cap_chunks_per_paper(ranked, paper, max_per_paper=1) == [1, 3, 6]
    assert cap_chunks_per_paper([], paper) == []


# -------------------------------------------------------------------- nprobe


@pytest.mark.parametrize(
    ("nlist", "user", "expected"),
    [
        (100, None, 10),  # sqrt(100)
        (64, None, 8),  # sqrt(64) = 8, the floor for small indexes
        (4, None, 4),  # never more than nlist
        (16384, None, 128),  # sqrt; floor rises to 32 at 16384
        (16384, 4, 32),  # user value below the floor is raised
        (65536, None, 256),
        (1_000_000, None, 1000),  # cap rises to 1024 at 65536
        (100, 1000, 100),  # user value above nlist is clamped
        (100, 3, 8),
    ],
)
def test_pick_nprobe(nlist, user, expected):
    assert pick_nprobe(nlist, user) == expected
