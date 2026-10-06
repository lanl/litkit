"""Tests for litkit.formatting: citation normalization, doc-level refs, references block.

Expected values are worked out by hand from the documented behavior in
src/litkit/formatting/answers.py and citations.py.
"""

import pytest

from litkit.formatting import (
    normalize_answer_and_build_refs,
    normalize_citations,
    render_references,
)


def _chunks(*dois):
    return [{"doi": d, "paper_title": d.upper(), "year": "2020"} for d in dois]


# ---------------------------------------------------------------- normalize_citations


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[1,2,3]", "[1, 2, 3]"),
        ("【1, 2, 1】", "[1, 2]"),  # fullwidth brackets, duplicate dropped
        ("[3, 1, 2, 1]", "[3, 1, 2]"),  # first-mention order kept
        ("[1，2、3]", "[1, 2, 3]"),  # fullwidth and ideographic commas
        ("[1†L1–L8]", "[1]"),  # line-location tail removed
        ("[2†L1-L8]", "[2]"),
        ("[1​]", "[1]"),  # zero-width space stripped
        ("text [Note] more", "text [Note] more"),  # not a citation
        ("【Note】", "[Note]"),  # brackets unified even when not a citation
    ],
)
def test_normalize_citations_canonical_form(raw, expected):
    assert normalize_citations(raw) == expected


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#8: ranges cut to first number")
@pytest.mark.parametrize("cite", ["[1-3]", "[1–3]", "[1, 2-3]"])
def test_range_citation_survives_the_answer_pipeline(cite):
    """As cli.py runs it: normalize_citations, then normalize_answer_and_build_refs.

    Checks the result, not normalize_citations' intermediate form, so either way of
    fixing #8 (expanding the range or passing it through) flips this test.
    """
    out, refs = normalize_answer_and_build_refs(
        normalize_citations(f"See {cite}."), _chunks("a", "b", "c")
    )
    assert out == "See [1, 2, 3]."
    assert [r["doi"] for r in refs] == ["a", "b", "c"]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#22: prose read as citation")
@pytest.mark.parametrize("prose", ["[3 patients]", "[95% CI 1.3–3.4]", "[2a]"])
def test_normalize_citations_leaves_prose_alone(prose):
    assert normalize_citations(f"x {prose} y") == f"x {prose} y"


@pytest.mark.parametrize("z", ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"])
def test_normalize_citations_strips_zero_width_characters(z):
    # Before the number, a zero-width character would stop the citation matching.
    assert normalize_citations(f"[{z}1, {z}2]") == "[1, 2]"


# ---------------------------------------------------- normalize_answer_and_build_refs


def test_docnums_follow_first_mention_not_chunk_order():
    # Chunk 3 (doi c) is cited first, so it becomes document 1; chunk 1 becomes 2.
    out, refs = normalize_answer_and_build_refs("A [3] then B [1].", _chunks("a", "b", "c"))
    assert out == "A [1] then B [2]."
    assert [r["doi"] for r in refs] == ["c", "a"]
    assert [r["docnum"] for r in refs] == [1, 2]


def test_chunks_of_one_paper_collapse_to_one_reference():
    # Chunks 1 and 3 are the same paper; [1, 2, 3] -> docs [1, 2, 1] -> [1, 2].
    chunks = _chunks("a", "b", "a")
    out, refs = normalize_answer_and_build_refs("See [1, 2, 3].", chunks)
    assert out == "See [1, 2]."
    assert [r["doi"] for r in refs] == ["a", "b"]


def test_ranges_expand_in_answer_normalization():
    out, refs = normalize_answer_and_build_refs("See [1-3].", _chunks("a", "b", "c"))
    assert out == "See [1, 2, 3]."
    assert [r["doi"] for r in refs] == ["a", "b", "c"]


def test_doi_match_ignores_case_and_pmcid_is_uppercased():
    chunks = [{"doi": "10.1/ABC"}, {"doi": "10.1/abc"}, {"pmcid": "pmc9"}, {"pmcid": "PMC9"}]
    out, refs = normalize_answer_and_build_refs("[1, 2] [3, 4]", chunks)
    assert out == "[1] [2]"
    assert len(refs) == 2


def test_chunks_without_identifiers_stay_separate():
    out, refs = normalize_answer_and_build_refs("[1, 2]", [{}, {}])
    assert out == "[1, 2]"
    assert len(refs) == 2


def test_paper_id_takes_precedence_over_title():
    # Same title, different papers in the DB: must not merge.
    chunks = [{"paper_id": 1, "paper_title": "Same"}, {"paper_id": 2, "paper_title": "Same"}]
    _, refs = normalize_answer_and_build_refs("[1, 2]", chunks)
    assert len(refs) == 2


@pytest.mark.parametrize("cite", ["[0]", "[3]"])
def test_out_of_range_citation_left_as_is_and_not_referenced(cite):
    out, refs = normalize_answer_and_build_refs(f"x {cite}", _chunks("a", "b"))
    assert out == f"x {cite}"
    assert refs == []


def test_uncited_chunks_are_not_referenced():
    _, refs = normalize_answer_and_build_refs("only [2]", _chunks("a", "b", "c"))
    assert [r["doi"] for r in refs] == ["b"]


# ------------------------------------------------------------------ render_references


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#23: two periods before ids")
def test_render_references_full_text():
    refs = [
        {"docnum": 2, "paper_title": "B", "year": None, "pmcid": "PMC2", "pmid": "22"},
        {"docnum": 1, "title": "A", "year": "2020", "doi": "10.1/a"},
    ]
    assert render_references(refs) == (
        "References\n" "[1] A (2020). DOI: 10.1/a\n" "[2] B. PMCID: PMC2, PMID: 22"
    )


def test_render_references_order_ids_and_title_fallback():
    """The parts of each line that #23 doesn't affect."""
    refs = [
        {"docnum": 2, "paper_title": "B", "year": None, "pmcid": "PMC2", "pmid": "22"},
        {"docnum": 1, "title": "A", "year": "2020", "doi": "10.1/a"},
    ]
    header, first, second = render_references(refs).split("\n")
    assert header == "References"
    assert first.startswith("[1] A (2020).")  # sorted by docnum; "title" fallback
    assert first.endswith("DOI: 10.1/a")
    assert second.startswith("[2] B.")
    assert second.endswith("PMCID: PMC2, PMID: 22")


def test_render_references_untitled_and_empty():
    assert render_references([{"docnum": 1}]) == "References\n[1] untitled."
    assert render_references([]) == "References\n"
