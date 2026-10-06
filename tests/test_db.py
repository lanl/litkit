"""Tests for litkit.db: schema creation, file registration, chunk-to-paper lookups."""

import sqlite3
from types import SimpleNamespace

import pytest

from litkit.db import already_processed, chunk_ids_to_paper_ids, init_db, register_file


@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "sqlite" / "litkit.sqlite3")
    yield c
    c.close()


def _names(conn, kind):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type=? AND name NOT LIKE 'sqlite_%'", (kind,)
    )
    return {r[0] for r in rows}


def test_init_db_creates_schema_and_pragmas(conn):
    assert _names(conn, "table") == {"papers", "chunks", "files"}
    assert {"papers_pmcid_uq", "papers_pmid_uq", "papers_doc_id_uq"} <= _names(conn, "index")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "truncate"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    cols = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
    assert {"doc_id", "pmid", "pmcid", "title", "abstract", "in_index"} <= cols


def test_init_db_is_idempotent(tmp_path):
    path = tmp_path / "db.sqlite3"
    c = init_db(path)
    c.execute("INSERT INTO papers(doc_id, pmid, pmcid) VALUES ('d1', '1', 'PMC1')")
    c.commit()
    c.close()
    c = init_db(path)
    assert c.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    c.close()


def test_unsupported_journal_mode_falls_back_to_truncate(tmp_path):
    c = init_db(tmp_path / "db.sqlite3", journal_mode="DELETE")
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "truncate"
    c.close()


def test_pmcid_is_unique_but_empty_ids_may_repeat(conn):
    conn.execute("INSERT INTO papers(doc_id, pmid, pmcid) VALUES ('a', '1', 'PMC1')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO papers(doc_id, pmid, pmcid) VALUES ('b', '2', 'PMC1')")
    # PubMed-only records have pmcid '' and must not collide with each other.
    conn.execute("INSERT INTO papers(doc_id, pmid, pmcid) VALUES ('c', '3', '')")
    conn.execute("INSERT INTO papers(doc_id, pmid, pmcid) VALUES ('d', '4', '')")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#10: missing doc_id column")
def test_init_db_upgrades_database_created_before_doc_id(tmp_path):
    path = tmp_path / "old.sqlite3"
    old = sqlite3.connect(path)
    old.executescript(
        "CREATE TABLE papers (id INTEGER PRIMARY KEY AUTOINCREMENT, pmid TEXT, pmcid TEXT,"
        " title TEXT, abstract TEXT);"
        "CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id INTEGER NOT NULL,"
        " ord INTEGER NOT NULL, text TEXT NOT NULL);"
        "CREATE TABLE files (path TEXT PRIMARY KEY, size INTEGER, mtime REAL, paper_id INTEGER);"
        "INSERT INTO papers(pmid, pmcid, title) VALUES ('1', 'PMC1', 'kept');"
    )
    old.commit()
    old.close()
    try:
        c = init_db(path)
    except sqlite3.OperationalError as e:
        # Only #10's own error counts as the expected failure; any other SQL error
        # in the migration propagates and fails the test.
        assert "no such column: doc_id" not in str(e), str(e)
        raise
    assert c.execute("SELECT title, in_index FROM papers").fetchall() == [("kept", 0)]
    c.close()


# ---------------------------------------------------------------- files table


def _st(size, mtime):
    return SimpleNamespace(st_size=size, st_mtime=mtime)


def test_register_then_already_processed(conn):
    cur = conn.cursor()
    path = "tar:///data/a.tar!/x.nxml"
    assert not already_processed(cur, path, _st(10, 1.5))
    register_file(cur, path, 7, _st(10, 1.5))
    assert already_processed(cur, path, _st(10, 1.5))
    assert not already_processed(cur, path, _st(11, 1.5))  # size changed
    assert not already_processed(cur, path, _st(10, 2.5))  # mtime changed
    assert not already_processed(cur, path + "x", _st(10, 1.5))


def test_register_file_replaces_previous_record(conn):
    cur = conn.cursor()
    register_file(cur, "p", 1, _st(10, 1.0))
    register_file(cur, "p", 2, _st(20, 2.0))
    assert cur.execute("SELECT size, mtime, paper_id FROM files").fetchall() == [(20, 2.0, 2)]


# ------------------------------------------------------- chunk_ids_to_paper_ids


def _add_chunks(conn, n):
    """n chunks, chunk i belongs to paper i % 7 + 1."""
    conn.executemany(
        "INSERT INTO chunks(id, paper_id, ord, text) VALUES (?, ?, 0, 't')",
        [(i, i % 7 + 1) for i in range(1, n + 1)],
    )


def test_chunk_ids_to_paper_ids_across_batches(conn):
    _add_chunks(conn, 1203)
    # 1203 ids plus 2 that don't exist: 3 batches of 500 (500, 500, 205).
    ids = list(range(1, 1204)) + [5000, 5001]
    got = chunk_ids_to_paper_ids(conn, ids)
    assert got == {i: i % 7 + 1 for i in range(1, 1204)}


class _CountingConn:
    """Wraps a connection and records the number of '?' in each SELECT ... IN query."""

    def __init__(self, conn):
        self._conn = conn
        self.params_per_query = []

    def cursor(self):
        outer = self

        class _Cur:
            def __init__(self, cur):
                self._cur = cur

            def execute(self, sql, params=()):
                outer.params_per_query.append(len(params))
                return self._cur.execute(sql, params)

            def fetchall(self):
                return self._cur.fetchall()

        return _Cur(self._conn.cursor())


def test_chunk_ids_to_paper_ids_batches_at_500(conn):
    # This SQLite build allows far more than 500 variables, so check batching directly.
    _add_chunks(conn, 1001)
    counting = _CountingConn(conn)
    got = chunk_ids_to_paper_ids(counting, list(range(1, 1002)))
    assert len(got) == 1001
    assert counting.params_per_query == [500, 500, 1]


def test_chunk_ids_to_paper_ids_empty(conn):
    assert chunk_ids_to_paper_ids(conn, []) == {}
