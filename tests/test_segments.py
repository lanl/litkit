"""Tests for litkit.segments: two-phase segment writers and build checkpoints."""

import json

import numpy as np
import pytest

from litkit.segments.checkpoint import (
    clear_shard_checkpoints,
    load_checkpoint,
    save_checkpoint,
    shard_ckpt_path,
)
from litkit.segments.ingest import _glob_segment_files
from litkit.segments.metadata import has_segment_files
from litkit.segments.writer import (
    ChunkSegmentWriter,
    SegmentWriter,
    cleanup_orphan_pending_files,
)


def _names(d):
    return sorted(p.name for p in d.iterdir())


@pytest.fixture
def seg_dir(tmp_path):
    d = tmp_path / "emb_segments"
    d.mkdir()
    return d


# --------------------------------------------------------------------- writers


def test_paper_segment_round_trip(seg_dir, unit_vectors):
    X = unit_vectors(3, 8)
    w = SegmentWriter(seg_dir, dtype="fp32", shard_id=12)
    pending = w.write(doc_ids=["tar://a!/1", "tar://a!/2", "tar://a!/3"], vecs=X)
    assert pending.name.startswith("paper_seg_12_0_")
    assert pending.name.endswith("_pending.npz")
    assert w.finalize() == 1
    (final,) = seg_dir.glob("paper_seg_12_0_*.npz")
    assert not final.name.endswith("_pending.npz")
    with np.load(final, allow_pickle=True) as z:
        assert list(z["doc_ids"]) == ["tar://a!/1", "tar://a!/2", "tar://a!/3"]
        assert str(z["kind"]) == "papers"
        np.testing.assert_array_equal(z["vecs"], X)  # fp32 stores the exact bits


def test_chunk_segment_round_trip_fp16(seg_dir, unit_vectors):
    X = unit_vectors(3, 8)
    w = ChunkSegmentWriter(seg_dir, dtype="fp16", shard_id=0)
    w.write(paper_doc_ids=["d1", "d1", "d2"], ords=[-1, 0, 0], vecs=X)
    w.finalize()
    (final,) = seg_dir.glob("chunk_seg_0_0_*.npz")
    with np.load(final, allow_pickle=True) as z:
        assert list(z["paper_doc_ids"]) == ["d1", "d1", "d2"]
        assert z["ords"].tolist() == [-1, 0, 0]
        assert z["vecs"].dtype == np.float16
        # fp16 has an 11-bit significand: relative error <= 2**-11 ~ 4.9e-4.
        np.testing.assert_allclose(z["vecs"].astype(np.float32), X, rtol=5e-4, atol=1e-7)


def test_add_flushes_every_segment_size(seg_dir):
    w = SegmentWriter(seg_dir, segment_size=3, dtype="fp32")
    for i in range(7):
        w.add(f"d{i}", np.ones((1, 4), dtype=np.float32))
    assert w.pending_count == 2  # 7 adds with size 3: two full segments, 1 buffered
    w.close()  # flushes the last one but does not finalize
    assert w.pending_count == 3
    assert w.finalize() == 3
    counts = []
    for p in sorted(seg_dir.glob("paper_seg_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            counts.append(len(z["doc_ids"]))
    assert sorted(counts) == [1, 3, 3]


def test_empty_write_writes_nothing(seg_dir):
    w = ChunkSegmentWriter(seg_dir)
    assert w.write(paper_doc_ids=[], ords=[], vecs=np.zeros((0, 4))) is None
    assert w.flush() is None
    assert _names(seg_dir) == []


def test_cleanup_pending_removes_only_this_writers_files(seg_dir, unit_vectors):
    papers = SegmentWriter(seg_dir, dtype="fp32")
    chunks = ChunkSegmentWriter(seg_dir, dtype="fp32")
    papers.write(doc_ids=["d"], vecs=unit_vectors(1, 4))
    papers.finalize()
    chunks.write(paper_doc_ids=["d"], ords=[0], vecs=unit_vectors(1, 4))
    assert chunks.cleanup_pending() == 1
    assert chunks.pending_count == 0
    assert [n.startswith("paper_seg_") for n in _names(seg_dir)] == [True]


def test_cleanup_orphan_pending_files(seg_dir, unit_vectors):
    w = ChunkSegmentWriter(seg_dir, dtype="fp32")
    w.write(paper_doc_ids=["d"], ords=[0], vecs=unit_vectors(1, 4))
    w.write(paper_doc_ids=["e"], ords=[0], vecs=unit_vectors(1, 4))
    w.finalize()
    w.write(paper_doc_ids=["f"], ords=[0], vecs=unit_vectors(1, 4))  # left pending
    assert cleanup_orphan_pending_files(seg_dir) == 1
    assert len(list(seg_dir.glob("chunk_seg_*.npz"))) == 2
    assert cleanup_orphan_pending_files(seg_dir / "missing") == 0


def _write_one(kind, seg_dir, vec):
    if kind == "papers":
        w = SegmentWriter(seg_dir, dtype="fp32")
        w.write(doc_ids=["d"], vecs=vec)
    else:
        w = ChunkSegmentWriter(seg_dir, dtype="fp32")
        w.write(paper_doc_ids=["d"], ords=[0], vecs=vec)
    return w


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#4: glob matches pending")
@pytest.mark.parametrize("kind", ["papers", "chunks"])
def test_consumer_does_not_see_pending_segments(seg_dir, unit_vectors, kind):
    _write_one(kind, seg_dir, unit_vectors(1, 4))  # pending, not finalized
    assert _glob_segment_files(seg_dir, kind) == []
    assert has_segment_files(seg_dir) is False


@pytest.mark.parametrize("kind", ["papers", "chunks"])
def test_consumer_sees_finalized_segments(seg_dir, unit_vectors, kind):
    """Control for the #4 test above."""
    _write_one(kind, seg_dir, unit_vectors(1, 4)).finalize()
    assert len(_glob_segment_files(seg_dir, kind)) == 1
    assert has_segment_files(seg_dir) is True


# ------------------------------------------------------------------ checkpoints


@pytest.fixture
def sqlite_dir(tmp_path):
    d = tmp_path / "sqlite"
    d.mkdir()
    return d


def test_checkpoint_round_trip_leaves_no_temp_file(sqlite_dir):
    ckpt = sqlite_dir / "build_checkpoint.json"
    data = {"build_stream": {"/path/to/a.tar": 1200, "/path/to/b.tar": 7}}
    save_checkpoint(data, ckpt, sqlite_dir / "ckpt.writer.lock")
    assert load_checkpoint(ckpt) == data
    assert json.loads(ckpt.read_text()) == data
    assert not list(sqlite_dir.glob("*.tmp"))


def test_checkpoint_save_replaces_previous(sqlite_dir):
    ckpt = sqlite_dir / "build_checkpoint.json"
    lock = sqlite_dir / "ckpt.writer.lock"
    save_checkpoint({"n": 1}, ckpt, lock)
    save_checkpoint({"n": 2}, ckpt, lock)
    assert load_checkpoint(ckpt) == {"n": 2}


def test_shard_checkpoints_are_separate_files(sqlite_dir):
    ckpt = sqlite_dir / "build_checkpoint.json"
    lock = sqlite_dir / "ckpt.writer.lock"
    save_checkpoint({"who": "shared"}, ckpt, lock)
    save_checkpoint({"who": 3}, ckpt, lock, shard_id=3)
    save_checkpoint({"who": 12}, ckpt, lock, shard_id=12)
    assert load_checkpoint(ckpt) == {"who": "shared"}
    assert load_checkpoint(ckpt, shard_id=3) == {"who": 3}
    assert load_checkpoint(ckpt, shard_id=12) == {"who": 12}
    assert load_checkpoint(ckpt, shard_id=4) == {}
    assert shard_ckpt_path(sqlite_dir, 3).name == "build_checkpoint_shard_03.json"
    assert shard_ckpt_path(sqlite_dir, 12).name == "build_checkpoint_shard_12.json"


def test_missing_or_corrupt_checkpoint_loads_empty(sqlite_dir):
    """A corrupt checkpoint restarts the scan from the beginning rather than crashing.

    That is safe only because already-ingested members are skipped through the
    files table; this pins the current behavior.
    """
    ckpt = sqlite_dir / "build_checkpoint.json"
    assert load_checkpoint(ckpt) == {}
    ckpt.write_text('{"build_stream": {"a": 1')  # truncated write
    assert load_checkpoint(ckpt) == {}


def test_clear_shard_checkpoints_removes_stale_higher_shards(sqlite_dir):
    for i in (0, 1, 2, 7, 15):
        shard_ckpt_path(sqlite_dir, i).write_text("{}")
    shared = sqlite_dir / "build_checkpoint.json"
    shared.write_text("{}")
    # The previous build had 3 shards, but files from an older 16-shard build remain.
    assert clear_shard_checkpoints(sqlite_dir, num_shards=3) == 5
    assert _names(sqlite_dir) == ["build_checkpoint.json"]
