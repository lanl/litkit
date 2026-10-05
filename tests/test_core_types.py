import pytest

from litkit.core.types import Chunk, Document


def test_chunk_immutable():
    c = Chunk(id="c1", paper_id="p1", text="text")
    assert c.id == "c1"
    assert c.section is None
    assert c.order == 0
    with pytest.raises(AttributeError):
        c.id = "x"


def test_document_with_chunks():
    c1 = Chunk(id="c1", paper_id="p1", text="t")
    doc = Document(id="p1", title="T", abstract=None, chunks=[c1])
    assert doc.id == "p1"
    assert len(doc.chunks) == 1
    assert doc.chunks[0].id == "c1"


def test_document_empty_chunks():
    doc = Document(id="p1", title="T", abstract="a")
    assert doc.chunks == ()
