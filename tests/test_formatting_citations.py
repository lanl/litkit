from litkit.formatting.citations import normalize_citations


def test_normalize_simple_brackets():
    assert normalize_citations("[1]") == "[1]"
    assert normalize_citations("[1,2,3]") == "[1, 2, 3]"


def test_normalize_fullwidth_brackets():
    assert normalize_citations("【1, 2, 1】") == "[1, 2]"


def test_normalize_lineloc_tails():
    assert normalize_citations("[1†L1–L8]") == "[1]"
    assert normalize_citations("[2†L1-L8]") == "[2]"


def test_normalize_dedupe_preserve_order():
    assert normalize_citations("[3, 1, 2, 1]") == "[3, 1, 2]"


def test_normalize_mixed_commas():
    assert normalize_citations("[1，2，3]") == "[1, 2, 3]"


def test_non_numeric_brackets_untouched():
    assert normalize_citations("[Note]") == "[Note]"
    assert normalize_citations("text [Note] more") == "text [Note] more"


def test_normalize_zero_width():
    assert normalize_citations("[1\u200b]") == "[1]"
