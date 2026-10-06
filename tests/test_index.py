"""Tests for litkit.index: exact search, ID replacement, approximate recall,
index-type detection, and saving and loading."""

import os

import faiss
import numpy as np
import pytest

from litkit.concurrent.locking import FileLock
from litkit.index import io as index_io
from litkit.index.dedup import add_with_ids_dedup
from litkit.index.factory import flat_ip_index, hnsw_index, ivfpq_index, safe_pq_m
from litkit.index.ids import make_id_selector, safe_remove_ids
from litkit.index.introspection import faiss_present_ids, kind_and_core

DIM = 32


def _brute_force_top_k(base, queries, k):
    sims = queries @ base.T
    order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(sims, order, axis=1)


def _recall(index, base, queries, k, retrieved):
    truth, _ = _brute_force_top_k(base, queries, k)
    _, got = index.search(queries, retrieved)
    return float(np.mean([len(set(t) & set(g)) / k for t, g in zip(truth, got, strict=True)]))


# --------------------------------------------------------------------- exact


def test_flat_ip_index_matches_brute_force(unit_vectors):
    base, queries = unit_vectors(500, DIM), unit_vectors(20, DIM)
    index = flat_ip_index(DIM)
    index.add(base)
    scores, ids = index.search(queries, 10)
    order, sims = _brute_force_top_k(base, queries, 10)
    # Random float data has no ties, so the IDs must match exactly.
    np.testing.assert_array_equal(ids, order)
    # float32 dot products of unit vectors in 32 dimensions: error ~1e-7.
    np.testing.assert_allclose(scores, sims, rtol=0, atol=1e-5)


# ------------------------------------------------------------- add / dedup


def test_add_with_ids_dedup_replaces_vector_in_flat_index(unit_vectors):
    X = unit_vectors(4, DIM)
    index = faiss.IndexIDMap2(flat_ip_index(DIM))
    add_with_ids_dedup(index, [10, 20, 30], X[:3])
    added, ids = add_with_ids_dedup(index, [20], X[3:4])
    assert (added, ids.tolist()) == (1, [20])
    assert index.ntotal == 3
    assert faiss_present_ids(index) == {10, 20, 30}
    # ID 20 now holds the new vector, not the old one.
    np.testing.assert_allclose(index.reconstruct(20), X[3], atol=1e-6)


def test_add_with_ids_dedup_normalizes_without_touching_input(unit_vectors):
    index = faiss.IndexIDMap2(flat_ip_index(DIM))
    X = unit_vectors(1, DIM) * 5.0
    add_with_ids_dedup(index, [1], X)
    np.testing.assert_allclose(np.linalg.norm(index.reconstruct(1)), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(X), 5.0, rtol=1e-6)


def test_add_with_ids_dedup_skip_dedup_keeps_both_vectors(unit_vectors):
    """Control: skip_dedup is only safe when no ID repeats (the --rebuild case)."""
    X = unit_vectors(2, DIM)
    index = faiss.IndexIDMap2(flat_ip_index(DIM))
    add_with_ids_dedup(index, [5], X[:1])
    add_with_ids_dedup(index, [5], X[1:], skip_dedup=True)
    assert index.ntotal == 2


def test_add_with_ids_dedup_adds_to_hnsw(unit_vectors):
    """Control: new IDs go into the default paper index type and can be found."""
    X = unit_vectors(3, DIM)
    index = faiss.IndexIDMap2(hnsw_index(DIM))
    added, ids = add_with_ids_dedup(index, [10, 20, 30], X)
    assert (added, ids.tolist(), index.ntotal) == (3, [10, 20, 30], 3)
    _, found = index.search(X, 1)
    assert found.ravel().tolist() == [10, 20, 30]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#7: HNSW re-add duplicates")
def test_add_with_ids_dedup_does_not_duplicate_in_hnsw(unit_vectors):
    X = unit_vectors(4, DIM)
    index = faiss.IndexIDMap2(hnsw_index(DIM))
    add_with_ids_dedup(index, [1, 2, 3], X[:3])
    add_with_ids_dedup(index, [2], X[3:4])
    if faiss_present_ids(index) != {1, 2, 3}:
        raise RuntimeError(f"IDs changed: {faiss_present_ids(index)}")  # not the #7 symptom
    assert index.ntotal == 3


def test_safe_remove_ids(unit_vectors, capsys):
    X = unit_vectors(5, DIM)
    flat = faiss.IndexIDMap2(flat_ip_index(DIM))
    flat.add_with_ids(X, np.arange(5, dtype="int64"))
    assert safe_remove_ids(flat, make_id_selector([1, 3, 99])) == 2
    assert faiss_present_ids(flat) == {0, 2, 4}
    hnsw = faiss.IndexIDMap2(hnsw_index(DIM))
    hnsw.add_with_ids(X, np.arange(5, dtype="int64"))
    capsys.readouterr()
    assert safe_remove_ids(hnsw, make_id_selector([1])) == 0  # documented no-op
    assert hnsw.ntotal == 5
    assert capsys.readouterr().err == ""  # and silent


# ------------------------------------------------------------- approximate


def test_hnsw_recall_and_rises_with_ef_search(unit_vectors):
    base, queries = unit_vectors(3000, DIM), unit_vectors(50, DIM)
    index = hnsw_index(DIM, M=16, ef_construction=100, ef_search=8)
    index.add(base)
    low = _recall(index, base, queries, k=10, retrieved=10)
    index.hnsw.efSearch = 64
    high = _recall(index, base, queries, k=10, retrieved=10)
    # Measured over 5 seeds: efSearch=8 gives 0.66-0.67, efSearch=64 gives 0.99-1.00.
    # 0.9 still fails a broken graph or metric (random IDs give ~0.003).
    assert high >= 0.9
    assert high > low


def test_ivfpq_recall_and_rises_with_nprobe(unit_vectors):
    base, queries = unit_vectors(3000, DIM), unit_vectors(50, DIM)
    # nlist=16 keeps training valid (FAISS wants ~39 points per centroid).
    index = ivfpq_index(DIM, nlist=16, m=16, bits=8)
    index.train(base)
    index.add(base)
    index.nprobe = 1
    low = _recall(index, base, queries, k=10, retrieved=50)
    index.nprobe = 16
    high = _recall(index, base, queries, k=10, retrieved=50)
    # 10@50 recall measured over 5 seeds: nprobe=1 gives 0.25-0.30, nprobe=16
    # (all lists) gives 1.00. PQ distance error is the only loss at nprobe=16.
    assert high >= 0.9
    assert high > low + 0.3


def test_ivfpq_index_uses_inner_product():
    assert ivfpq_index(DIM, nlist=4, m=8).metric_type == faiss.METRIC_INNER_PRODUCT
    assert hnsw_index(DIM).metric_type == faiss.METRIC_INNER_PRODUCT


@pytest.mark.parametrize(("dim", "m", "expected"), [(768, 64, 64), (768, 100, 96), (30, 8, 6)])
def test_safe_pq_m_divides_dimension(dim, m, expected):
    assert safe_pq_m(dim, m) == expected
    assert dim % expected == 0


def test_kind_and_core_through_wrappers(unit_vectors):
    ivf = ivfpq_index(DIM, nlist=4, m=8)
    assert kind_and_core(faiss.IndexIDMap2(flat_ip_index(DIM)))[0] == "flat"
    assert kind_and_core(faiss.IndexIDMap2(hnsw_index(DIM)))[0] == "hnsw"
    kind, core = kind_and_core(faiss.IndexIDMap2(ivf))
    assert kind == "ivf"
    assert core.nlist == 4  # the IVF core, not its quantizer


# --------------------------------------------------------------- save / load


@pytest.fixture
def faiss_lock(tmp_path):
    path = tmp_path / "faiss.writer.lock"

    def lock():
        return FileLock(path, faiss_lock_path=path)

    return lock


@pytest.fixture
def fresh_throttle(monkeypatch):
    monkeypatch.setattr(index_io, "_last_save_ts", {"papers": 0.0, "chunks": 0.0})
    monkeypatch.setattr(index_io, "_SAVE_MIN_SEC", 120)


def test_faiss_save_requires_lock(tmp_path, fresh_throttle):
    with pytest.raises(AssertionError, match="FAISS_LOCK"):
        index_io.faiss_save(flat_ip_index(DIM), tmp_path / "papers.faiss")


def test_faiss_save_is_throttled_per_label(tmp_path, faiss_lock, fresh_throttle, unit_vectors):
    index = flat_ip_index(DIM)
    with faiss_lock():
        assert index_io.faiss_save(index, tmp_path / "papers.faiss") is True
        index.add(unit_vectors(1, DIM))
        # A second papers save within _SAVE_MIN_SEC is skipped; the file keeps ntotal=0.
        assert index_io.faiss_save(index, tmp_path / "papers.faiss") is False
        assert index_io.faiss_load(tmp_path / "papers.faiss").ntotal == 0
        # The throttle is per label, so a chunks save still goes through.
        assert index_io.faiss_save(index, tmp_path / "chunks.faiss") is True
        assert index_io.faiss_save_force(index, tmp_path / "papers.faiss") is True
    assert index_io.faiss_load(tmp_path / "papers.faiss").ntotal == 1
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("kind", ["flat", "hnsw", "ivfpq"])
def test_save_load_round_trip_gives_identical_search(
    tmp_path, faiss_lock, fresh_throttle, unit_vectors, make_index, kind
):
    base, queries = unit_vectors(1000, DIM), unit_vectors(10, DIM)
    ids = np.arange(1000, dtype="int64") * 7 + 3  # non-contiguous external IDs
    index = make_index(kind, base, ids)
    path = tmp_path / "chunks.faiss"
    with faiss_lock():
        index_io.faiss_save_force(index, path)
    loaded = index_io.faiss_load(path)
    scores0, ids0 = index.search(queries, 5)
    scores1, ids1 = loaded.search(queries, 5)
    np.testing.assert_array_equal(ids1, ids0)
    np.testing.assert_array_equal(scores1, scores0)  # same index, same bits
    assert set(ids1.ravel()) <= set(ids.tolist())


def test_faiss_load_cached_reloads_after_file_changes(tmp_path, faiss_lock, unit_vectors):
    path = tmp_path / "papers.faiss"
    a = flat_ip_index(DIM)
    with faiss_lock():
        index_io.faiss_save_force(a, path)
    index_io.clear_faiss_cache()
    assert index_io.faiss_load_cached(path).ntotal == 0
    assert index_io.faiss_load_cached(path) is index_io.faiss_load_cached(path)
    a.add(unit_vectors(3, DIM))
    with faiss_lock():
        index_io.faiss_save_force(a, path)
    # The cache key is (path, mtime_ns); bump mtime in case both saves share a tick.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert index_io.faiss_load_cached(path).ntotal == 3
    index_io.clear_faiss_cache()
