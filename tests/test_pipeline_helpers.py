from litkit.pipeline.helpers import dedupe_ids_and_texts


def test_dedupe_preserves_first():
    ids = ["a", "b", "a", "c"]
    texts = ["1", "2", "1", "3"]
    out_ids, out_texts = dedupe_ids_and_texts(ids, texts)
    assert out_ids == ["a", "b", "c"]
    assert out_texts == ["1", "2", "3"]


def test_dedupe_empty():
    ids, texts = dedupe_ids_and_texts([], [])
    assert ids == []
    assert texts == []
