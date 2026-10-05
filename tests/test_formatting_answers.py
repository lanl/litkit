from litkit.formatting.answers import (
    _doc_key,
    _expand_ranges,
    _unique_preserve,
    normalize_answer_and_build_refs,
    render_references,
)


def test_expand_ranges():
    assert _expand_ranges("1-3") == [1, 2, 3]
    assert _expand_ranges("3-1") == [1, 2, 3]
    assert _expand_ranges("1, 3, 5–6") == [1, 3, 5, 6]
    assert _expand_ranges("") == []


def test_unique_preserve():
    assert _unique_preserve([1, 2, 1, 3, 2]) == [1, 2, 3]


def test_doc_key_priority():
    ch = {"doi": "10.1/a", "pmcid": "PMC1", "pmid": "123", "paper_id": "p1"}
    assert _doc_key(ch) == "doi:10.1/a"
    ch2 = {"pmcid": "PMC1", "pmid": "123"}
    assert _doc_key(ch2) == "pmcid:PMC1"
    ch3 = {"pmid": "123"}
    assert _doc_key(ch3) == "pmid:123"
    ch4 = {"paper_id": "p1"}
    assert _doc_key(ch4) == "paper:p1"
    ch5 = {"paper_title": "Title"}
    assert _doc_key(ch5) == "title:title"


def test_normalize_and_build_refs_simple():
    chunks = [
        {"doi": "10.1/a", "paper_title": "A", "year": "2020"},
        {"doi": "10.1/b", "paper_title": "B", "year": "2021"},
    ]
    answer = "Result [1] and [2]"
    norm, refs = normalize_answer_and_build_refs(answer, chunks)
    assert norm == "Result [1] and [2]"
    assert len(refs) == 2
    assert refs[0]["docnum"] == 1
    assert refs[0]["doi"] == "10.1/a"


def test_normalize_merges_same_doc():
    chunks = [
        {"doi": "10.1/a", "paper_title": "A"},
        {"doi": "10.1/a", "paper_title": "A"},
    ]
    answer = "See [1,2]"
    norm, refs = normalize_answer_and_build_refs(answer, chunks)
    # Both chunks map to same doc, so citations collapse to [1]
    assert norm == "See [1]"
    assert len(refs) == 1


def test_render_references_header():
    refs = [{"docnum": 1, "paper_title": "A", "year": "2020", "doi": "10.1/a"}]
    out = render_references(refs)
    assert out.startswith("References")
    assert "[1] A (2020)." in out
    assert "DOI: 10.1/a" in out


def test_render_empty():
    assert render_references([]) == "References\n"
