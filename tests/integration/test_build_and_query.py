"""End to end: build the tiny corpus with the real CLI, then query it.

The CLI runs in a subprocess (it keeps module-level state, installs signal
handlers and creates a writer guard), with fake embedders patched in by
run_litkit_fake.py, so no model, GPU or network is needed.
"""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import faiss
import pytest

from conftest import CORPUS, jats, write_tar

RUNNER = Path(__file__).with_name("run_litkit_fake.py")
FLAT = ["--chunks-index", "flat", "--papers-index", "flat"]


def litkit(workspace, *args):
    env = dict(os.environ, LITKIT_WORKSPACE=str(workspace))
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


class CliFailed(Exception):
    """The CLI exited nonzero.

    Deliberately not an AssertionError: the known-bug tests are
    xfail(raises=AssertionError), so a crash in the code they exercise must
    fail them instead of counting as the expected failure.
    """


def _ok(proc):
    if proc.returncode != 0:
        raise CliFailed(f"exit {proc.returncode}\n{proc.stderr[-2000:]}")
    return proc


def _db(workspace):
    return sqlite3.connect(workspace / "sqlite" / "litkit.sqlite3")


def _count(workspace, sql):
    with _db(workspace) as conn:
        return conn.execute(sql).fetchone()[0]


def _ntotal(workspace, name):
    return faiss.read_index(str(workspace / "indices" / name)).ntotal


def _context_titles(stdout):
    """Titles of the '[i] Title (PMCID:...)' block headers in --no-llm output."""
    return [
        line.split("] ", 1)[1].split(" (PMCID")[0]
        for line in stdout.splitlines()
        if line[:1] == "[" and "] " in line and "(PMC" in line
    ]


# Chunks per article: one title+abstract chunk (ord -1) plus the packed body.
# Each CORPUS paragraph is ~140 chars, so up to 8 fit one 1200-char chunk:
# hiv 6 paragraphs -> 1 body chunk, egfr 4 -> 1, malaria 2 -> 1.
EXPECTED_PAPERS = 3
EXPECTED_CHUNKS = 3 * 2


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "tars"
    d.mkdir()
    write_tar(d / "a.tar", {f"{k}/{v['pmcid']}.nxml": jats(**v) for k, v in CORPUS.items()})
    return d


@pytest.fixture
def built(isolated_env, corpus_dir):
    proc = _ok(
        litkit(
            isolated_env,
            "--build-only",
            "--faiss-writer",
            "--rebuild",
            "--tar-dir",
            str(corpus_dir),
            *FLAT,
        )
    )
    return isolated_env, proc


class WrongResult(Exception):
    """A result that is wrong in a way other than the bug an xfail test is about.

    Raised instead of failing an assert, so it fails the xfail test instead of
    counting as the expected failure.
    """


def _check(ok, message):
    if not ok:
        raise WrongResult(message)


@pytest.mark.parametrize("paper_index", ["flat", "hnsw"])
def test_build_and_query_with_each_paper_index(isolated_env, corpus_dir, paper_index):
    """Control for the xfails below: HNSW is the default paper index."""
    ws = isolated_env
    args = ["--build-only", "--faiss-writer", "--rebuild", "--tar-dir", str(corpus_dir)]
    _ok(litkit(ws, *args, "--chunks-index", "flat", "--papers-index", paper_index))
    assert _ntotal(ws, "papers.faiss") == EXPECTED_PAPERS
    assert _ntotal(ws, "chunks.faiss") == EXPECTED_CHUNKS
    from litkit.index.introspection import kind_and_core

    assert kind_and_core(faiss.read_index(str(ws / "indices" / "papers.faiss")))[0] == paper_index
    question, title = QUESTIONS[0]
    proc = _ok(litkit(ws, "--no-llm", "--top-papers", "1", "--per-paper-cap", "0", question))
    assert set(_context_titles(proc.stdout)) == {title}


def test_build_counts_match_between_db_and_indexes(built):
    ws, proc = built
    assert _count(ws, "SELECT COUNT(*) FROM papers") == EXPECTED_PAPERS
    assert _count(ws, "SELECT COUNT(*) FROM chunks") == EXPECTED_CHUNKS
    assert _count(ws, "SELECT COUNT(*) FROM papers WHERE in_index=0") == 0
    assert _count(ws, "SELECT COUNT(*) FROM chunks WHERE in_index=0") == 0
    assert _ntotal(ws, "papers.faiss") == EXPECTED_PAPERS
    assert _ntotal(ws, "chunks.faiss") == EXPECTED_CHUNKS
    assert f"indexed {EXPECTED_PAPERS} papers and {EXPECTED_CHUNKS} chunks" in proc.stderr
    with _db(ws) as conn:
        rows = conn.execute("SELECT pmcid, pmid, title FROM papers ORDER BY pmcid").fetchall()
    assert rows == [(v["pmcid"], v["pmid"], v["title"]) for v in CORPUS.values()]


def test_index_ids_are_database_ids(built):
    ws, _ = built
    from litkit.index.introspection import faiss_present_ids

    with _db(ws) as conn:
        paper_ids = {r[0] for r in conn.execute("SELECT id FROM papers")}
        chunk_ids = {r[0] for r in conn.execute("SELECT id FROM chunks")}
    assert faiss_present_ids(faiss.read_index(str(ws / "indices" / "papers.faiss"))) == paper_ids
    assert faiss_present_ids(faiss.read_index(str(ws / "indices" / "chunks.faiss"))) == chunk_ids


# Stage 1 ranks papers by title + abstract, so each question shares several words
# with its paper's title and abstract. test_questions_rank_their_paper_first_for_any_hash
# checks that this doesn't depend on how the fake embedder's hashes happen to fall.
QUESTIONS = [
    ("Does HIV entry require CD4 and CCR5?", CORPUS["hiv"]["title"]),
    ("EGFR signaling drives carcinoma proliferation", CORPUS["egfr"]["title"]),
    ("Plasmodium sporozoites invade the liver", CORPUS["malaria"]["title"]),
]


@pytest.mark.parametrize(("question", "title"), QUESTIONS)
def test_questions_rank_their_paper_first_for_any_hash(question, title):
    """Oracle check for the query tests: re-salt the token hashes (equivalent to any
    other choice of words) and count how often Stage 1 would rank the right paper first."""
    import hashlib

    import numpy as np

    from conftest import FakeEmbedder

    papers = [v["title"] + " " + v["abstract"] for v in CORPUS.values()]
    right = [v["title"] for v in CORPUS.values()].index(title)

    class Salted(FakeEmbedder):
        def __init__(self, dim, salt):
            super().__init__(dim)
            self.salt = salt

        def _token_vec(self, tok):
            h = hashlib.sha256(f"{self.salt}:{tok}".encode()).digest()[:8]
            v = np.random.default_rng(int.from_bytes(h, "little")).standard_normal(self.dim)
            return v / np.linalg.norm(v)

    wins = 0
    for salt in range(200):
        emb = Salted(32, salt)
        wins += int(np.argmax(emb.encode(papers) @ emb.encode([question])[0]) == right)
    # Measured over 1000 salts: >= 99.7% for each question at dim 32 (the wording
    # this replaced won 50.7%). 95% over 200 salts still fails a question that
    # shares no title/abstract words with its paper (~33%).
    assert wins >= 190


@pytest.mark.parametrize(("question", "title"), QUESTIONS)
def test_query_retrieves_the_matching_paper(built, question, title):
    ws, _ = built
    # Per-paper cap off: see #25 for what the default cap does to a 1-paper shortlist.
    proc = _ok(litkit(ws, "--no-llm", "--top-papers", "1", "--per-paper-cap", "0", question))
    assert set(_context_titles(proc.stdout)) == {title}


@pytest.mark.parametrize(("cap", "hiv_chunks"), [("3", 3), ("2", 2), ("0", 4)])
def test_per_paper_cap_limits_chunks_per_paper(isolated_env, corpus_dir, cap, hiv_chunks):
    """Every paper shortlisted, so #25 can't bite. With 300-character chunks, 2 of
    the ~140-character paragraphs fit per chunk: HIV has 6 paragraphs -> 3 body
    chunks + title/abstract = 4, EGFR 4 -> 2 + 1 = 3, malaria 2 -> 1 + 1 = 2.
    So a cap of 3 or 2 binds for HIV, and 0 means no cap."""
    ws = isolated_env
    small = ["--chunk-target-chars", "300", "--chunk-min-chars", "0"]
    args = ["--build-only", "--faiss-writer", "--rebuild", "--tar-dir", str(corpus_dir)]
    _ok(litkit(ws, *args, *FLAT, *small))
    with _db(ws) as conn:
        per_paper = dict(
            conn.execute(
                "SELECT p.pmcid, COUNT(*) FROM chunks c JOIN papers p ON p.id = c.paper_id"
                " GROUP BY p.id"
            ).fetchall()
        )
    assert per_paper == {"PMC100001": 4, "PMC100002": 3, "PMC100003": 2}
    question, title = QUESTIONS[0]
    proc = _ok(litkit(ws, "--no-llm", "--top-papers", "3", "--per-paper-cap", cap, question))
    titles = _context_titles(proc.stdout)
    assert titles[0] == title
    assert titles.count(title) == hiv_chunks


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#25: widening leaves shortlist")
def test_query_with_default_cap_stays_in_shortlist(built):
    ws, _ = built
    question, title = QUESTIONS[0]
    proc = _ok(litkit(ws, "--no-llm", "--top-papers", "1", question))
    titles = _context_titles(proc.stdout)
    _check(titles and titles[0] == title, f"shortlisted paper missing or not first: {titles}")
    assert set(titles) == {title}


def _chunk_rows(ws):
    with _db(ws) as conn:
        return conn.execute("SELECT id, paper_id, ord, text FROM chunks ORDER BY id").fetchall()


def test_update_skips_seen_members(built, corpus_dir):
    ws, _ = built
    before = _chunk_rows(ws)
    proc = _ok(
        litkit(
            ws, "--build-only", "--faiss-writer", "--update", "--tar-dir", str(corpus_dir), *FLAT
        )
    )
    assert "indexed 0 papers and 0 chunks" in proc.stderr
    # Unchanged members are skipped, not deleted and re-inserted under new ids.
    assert _chunk_rows(ws) == before
    assert _ntotal(ws, "chunks.faiss") == EXPECTED_CHUNKS


def test_update_without_checkpoint_skips_via_files_table(built, corpus_dir):
    """The checkpoint normally skips a finished tar; without it, the files table must."""
    ws, _ = built
    before = _chunk_rows(ws)
    (ws / "sqlite" / "build_checkpoint.json").unlink()
    proc = _ok(
        litkit(
            ws, "--build-only", "--faiss-writer", "--update", "--tar-dir", str(corpus_dir), *FLAT
        )
    )
    assert "resume=#0" in proc.stderr
    assert "indexed 0 papers and 0 chunks" in proc.stderr
    assert _chunk_rows(ws) == before


def test_update_adds_new_tar(built, corpus_dir):
    ws, _ = built
    extra = dict(
        pmcid="PMC100009",
        pmid="900009",
        title="Prion folding",
        abstract="Prions misfold.",
        paragraphs=[],
    )
    write_tar(corpus_dir / "b.tar", {"x/PMC100009.nxml": jats(**extra)})
    _ok(
        litkit(
            ws, "--build-only", "--faiss-writer", "--update", "--tar-dir", str(corpus_dir), *FLAT
        )
    )
    assert _count(ws, "SELECT COUNT(*) FROM papers") == EXPECTED_PAPERS + 1
    # No body paragraphs: the build packs the abstract as the body (ingest_loop.py,
    # `paras = meta["paragraphs"] or [ab]`), so 2 chunks: title+abstract and abstract.
    assert _count(ws, "SELECT COUNT(*) FROM chunks") == EXPECTED_CHUNKS + 2
    assert _ntotal(ws, "papers.faiss") == EXPECTED_PAPERS + 1
    assert _ntotal(ws, "chunks.faiss") == EXPECTED_CHUNKS + 2


# The paper index type matters for #7: HNSW, the default, can't remove vectors.
PAPER_INDEXES = ["flat", "hnsw"]


def _index_args(paper_index):
    return ["--chunks-index", "flat", "--papers-index", paper_index]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#7: reappearing PMCID duplicated")
@pytest.mark.parametrize("paper_index", PAPER_INDEXES)
def test_update_same_pmcid_in_new_tar_does_not_duplicate(isolated_env, corpus_dir, paper_index):
    ws = isolated_env
    build = ["--build-only", "--faiss-writer", "--tar-dir", str(corpus_dir)]
    _ok(litkit(ws, *build, "--rebuild", *_index_args(paper_index)))
    write_tar(corpus_dir / "b.tar", {"dup/PMC100001.nxml": jats(**CORPUS["hiv"])})
    _ok(litkit(ws, *build, "--update", *_index_args(paper_index)))
    _check_nothing_lost(ws)
    assert _count(ws, "SELECT COUNT(*) FROM chunks") == EXPECTED_CHUNKS
    _assert_db_matches_indexes(ws)


def _check_nothing_lost(ws):
    """Each CORPUS paper still has its 2 chunks and every row has a vector; losing
    them is a different bug from #7's duplicates, so it must not satisfy the xfail."""
    with _db(ws) as conn:
        n_chunks = dict(
            conn.execute(
                "SELECT p.pmcid, COUNT(c.id) FROM papers p"
                " LEFT JOIN chunks c ON c.paper_id = p.id GROUP BY p.id"
            ).fetchall()
        )
    short = {v["pmcid"]: n_chunks.get(v["pmcid"], 0) for v in CORPUS.values()}
    _check(all(n >= 2 for n in short.values()), f"a paper lost its chunks: {short}")
    _check(
        _ntotal(ws, "papers.faiss") >= _count(ws, "SELECT COUNT(*) FROM papers"),
        "paper vectors lost",
    )
    _check(
        _ntotal(ws, "chunks.faiss") >= _count(ws, "SELECT COUNT(*) FROM chunks"),
        "chunk vectors lost",
    )


def _assert_db_matches_indexes(ws):
    assert _ntotal(ws, "papers.faiss") == _count(ws, "SELECT COUNT(*) FROM papers")
    assert _ntotal(ws, "chunks.faiss") == _count(ws, "SELECT COUNT(*) FROM chunks")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#7: same PMCID in two tars")
@pytest.mark.parametrize(
    "batch",
    [
        # Batches of 1 flush the first copy before the second arrives: on main this
        # leaves a second paper vector (skip_dedup during --rebuild).
        "1",
        # The default batch still holds the first copy's chunks when the second
        # arrives, so a fix that only removes indexed vectors leaves orphans.
        "20000",
    ],
)
@pytest.mark.parametrize("paper_index", PAPER_INDEXES)
def test_rebuild_same_pmcid_in_two_tars_keeps_one_copy(
    isolated_env, tiny_corpus, batch, paper_index
):
    args = ["--build-only", "--faiss-writer", "--rebuild", "--tar-dir", str(tiny_corpus)]
    args += _index_args(paper_index)
    _ok(litkit(isolated_env, *args, "--paper-batch", batch, "--chunk-batch", batch))
    _check_nothing_lost(isolated_env)
    _assert_db_matches_indexes(isolated_env)
    with _db(isolated_env) as conn:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM chunks c JOIN papers p ON p.id = c.paper_id"
            " WHERE p.pmcid = 'PMC100001'"
        ).fetchone()
    assert n == 2  # title+abstract chunk and one body chunk, once


def test_compressed_tar_builds_the_same_as_uncompressed(isolated_env, tmp_path, built):
    """.tar.gz goes through the sequential parser; .tar through the parallel one."""
    ws_tar, _ = built
    gz = tmp_path / "gz"
    gz.mkdir()
    write_tar(gz / "a.tar.gz", {f"{k}/{v['pmcid']}.nxml": jats(**v) for k, v in CORPUS.items()})
    ws_gz = tmp_path / "ws_gz"
    _ok(litkit(ws_gz, "--build-only", "--faiss-writer", "--rebuild", "--tar-dir", str(gz), *FLAT))
    rows = "SELECT p.pmcid, c.ord, c.text FROM chunks c JOIN papers p ON p.id = c.paper_id ORDER BY 1, 2"
    with _db(ws_tar) as a, _db(ws_gz) as b:
        assert b.execute(rows).fetchall() == a.execute(rows).fetchall()
    _assert_db_matches_indexes(ws_gz)


def test_edge_case_members_are_skipped(isolated_env, tiny_corpus):
    """Empty <article>, no abstract, PubMed record and a non-XML member, in a .tar."""
    tar_dir = tiny_corpus.parent / "only_a"
    tar_dir.mkdir()
    (tiny_corpus / "corpus_a.tar").rename(tar_dir / "corpus_a.tar")
    _ok(
        litkit(
            isolated_env,
            "--build-only",
            "--faiss-writer",
            "--rebuild",
            "--tar-dir",
            str(tar_dir),
            *FLAT,
        )
    )
    with _db(isolated_env) as conn:
        titles = {r[0] for r in conn.execute("SELECT title FROM papers")}
    # The empty article yields no paper; the other five are ingested.
    assert titles == {v["title"] for v in CORPUS.values()} | {"No abstract here", "A PubMed record"}


def test_rebuild_without_yes_refuses_noninteractively(isolated_env, corpus_dir):
    env = dict(os.environ, LITKIT_WORKSPACE=str(isolated_env))
    env.pop("LITKIT_ASSUME_YES")
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--build-only",
            "--faiss-writer",
            "--rebuild",
            "--tar-dir",
            str(corpus_dir),
            *FLAT,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )
    assert proc.returncode == 2
    assert "Refusing to proceed non-interactively" in proc.stderr
