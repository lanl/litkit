"""Tests for litkit.ingest: JATS/PubMed parsing, chunk packing, tar and manifest handling."""

import io
import tarfile

import pytest

from conftest import CORPUS, jats, pubmed_xml, write_tar
from litkit.build.helpers import pack_paragraphs as build_pack_paragraphs
from litkit.ingest import (
    count_tar_xml_members,
    is_uncompressed_tar,
    iter_tar_paths,
    iter_tar_xml_streams,
    pack_paragraphs,
    parallel_iter_tar_articles,
    parse_xml_fileobj,
    shard_filter,
)


def _parse(text):
    return parse_xml_fileobj(io.BytesIO(text.encode("utf-8")))


# ---------------------------------------------------------------------------- parsing


def test_parse_jats_fields():
    meta = _parse(jats(**CORPUS["hiv"]))
    assert meta["pmcid"] == "PMC100001"  # not the pmcid-ver or pmcaid ids
    assert meta["pmid"] == "900001"
    assert meta["title"] == "HIV entry requires CD4"
    assert meta["abstract"] == "HIV binds CD4 and CCR5 on T cells."
    assert meta["paragraphs"] == CORPUS["hiv"]["paragraphs"]  # ref-list text excluded


def test_parse_jats_skips_tables_figures_captions_and_short_paragraphs():
    long = "x" * 40  # the parser keeps paragraphs of 40+ characters
    xml = (
        "<article><front><article-meta><title-group><article-title>T</article-title>"
        "</title-group></article-meta></front><body>"
        f"<p>{long}</p><p>{'y' * 39}</p>"
        f"<table-wrap><p>table {long}</p></table-wrap>"
        f"<fig><caption><p>figure {long}</p></caption></fig>"
        "</body></article>"
    )
    assert _parse(xml)["paragraphs"] == [long]


def test_parse_jats_whitespace_is_collapsed():
    xml = jats(title="A\n   split\ttitle", abstract="one\n\n two", pmcid="PMC1")
    meta = _parse(xml)
    assert meta["abstract"] == "one two"
    # The title comes from string(), which keeps internal whitespace as written.
    assert meta["title"] == "A\n   split\ttitle"


def test_parse_pubmed_record():
    meta = _parse(pubmed_xml("900005", "A PubMed record", "Only an abstract."))
    assert meta == {
        "pmid": "900005",
        "pmcid": "",
        "title": "A PubMed record",
        "abstract": "Only an abstract.",
        "paragraphs": [],
    }


@pytest.mark.parametrize("data", [b"", b"\x00\x01binary", b'<?xml version="1.0"?><article/>'])
def test_parse_returns_none_for_unusable_input(data):
    assert parse_xml_fileobj(io.BytesIO(data)) is None


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#24: raises instead of None")
@pytest.mark.parametrize("data", [b"   \n", b"not xml at all"])
def test_parse_returns_none_for_text_without_a_root(data):
    assert parse_xml_fileobj(io.BytesIO(data)) is None


# --------------------------------------------------------------------------- chunking


@pytest.fixture(params=["ingest", "build"])
def packer(request):
    """litkit has two copies of pack_paragraphs; the build loop uses the second."""
    return pack_paragraphs if request.param == "ingest" else build_pack_paragraphs


def test_pack_greedy_without_overlap(packer):
    a, b, c = "a" * 500, "b" * 500, "c" * 500
    # a+b = 1002 chars (two 501-char slots) fits 1200; adding c would make 1503.
    assert packer([a, b, c], max_chars=1200, min_chars=300, overlap_chars=0) == [f"{a} {b}", c]


def test_pack_boundary_at_exactly_max_chars(packer):
    # The packer counts len(p) + 1 per paragraph, so 599 + 599 counts 1200 and fits
    # max_chars=1200 (joined chunk 1199 chars); 600 + 600 counts 1202 and doesn't.
    a, b = "a" * 599, "b" * 599
    assert packer([a, b], max_chars=1200, min_chars=0, overlap_chars=0) == [f"{a} {b}"]
    a, b = "a" * 600, "b" * 600
    assert packer([a, b], max_chars=1200, min_chars=0, overlap_chars=0) == [a, b]


def test_pack_short_tail_merges_into_previous_chunk(packer):
    a, b, c = "a" * 500, "b" * 500, "c" * 100
    # a+b take 1002 of 1100; c would make 1103, so c starts a new chunk. That chunk
    # is 100 chars, under min_chars=300, so it joins the previous chunk instead.
    out = packer([a, b, c], max_chars=1100, min_chars=300, overlap_chars=0)
    assert out == [f"{a} {b} {c}"]


def test_pack_first_short_chunk_is_kept(packer):
    assert packer(["short"], max_chars=1200, min_chars=300, overlap_chars=0) == ["short"]


def test_pack_overlap_prefixes_tail_of_previous_chunk(packer):
    a, b, c = "a" * 500, "b" * 500, "c" * 500
    out = packer([a, b, c], max_chars=1200, min_chars=300, overlap_chars=10)
    assert out == [f"{a} {b}", "b" * 10 + " " + c]


def test_pack_chunks_never_exceed_max_chars_unless_one_paragraph_does(packer):
    paras = [
        c * n
        for c, n in zip("abcdefghij", [300, 899, 1, 450, 450, 1199, 600, 600, 2, 5], strict=True)
    ]
    for chunk in packer(paras, max_chars=1200, min_chars=0, overlap_chars=0):
        assert len(chunk) <= 1200 or " " not in chunk


def test_pack_oversized_paragraph_is_not_split(packer):
    big = "x" * 5000
    assert packer([big], max_chars=1200, min_chars=300, overlap_chars=0) == [big]


def test_pack_empty(packer):
    assert packer([]) == []


# ------------------------------------------------------------------------ tar members


def test_count_and_stream_only_xml_members(tiny_corpus):
    tar = tiny_corpus / "corpus_a.tar"
    names = [m.name for m, _ in iter_tar_xml_streams(tar)]
    # 3 CORPUS articles + empty + noabstract + pubmed.xml; readme.txt is skipped.
    assert names == [
        "hiv/PMC100001.nxml",
        "egfr/PMC100002.nxml",
        "malaria/PMC100003.nxml",
        "edge/empty.nxml",
        "edge/noabstract.nxml",
        "edge/pubmed.xml",
    ]
    assert count_tar_xml_members(tar) == 6


def test_extension_match_is_case_insensitive(tmp_path):
    tar = write_tar(tmp_path / "x.tar", {"A.NXML": jats(pmcid="PMC1", title="T")})
    assert count_tar_xml_members(tar) == 1


def test_unreadable_tar_yields_nothing(tmp_path):
    bad = tmp_path / "bad.tar"
    bad.write_bytes(b"not a tar" * 100)
    assert count_tar_xml_members(bad) == 0
    assert list(iter_tar_xml_streams(bad)) == []


def test_parallel_parse_matches_sequential_parse(tiny_corpus):
    """Two implementations of the same job: they must agree, in tar order."""
    tar = tiny_corpus / "corpus_a.tar"
    sequential = []
    for m, f in iter_tar_xml_streams(tar):
        meta = parse_xml_fileobj(f)
        if meta is not None:
            sequential.append((m.name, meta))
    parallel = [(m.name, meta) for m, meta in parallel_iter_tar_articles(tar, workers=4)]
    assert parallel == sequential
    # The empty <article> parses to nothing; the other five are kept.
    assert len(parallel) == 5


def test_parallel_member_meta_has_size_and_mtime(tiny_corpus):
    tar = tiny_corpus / "corpus_a.tar"
    with tarfile.open(tar) as tf:
        sizes = {m.name: m.size for m in tf.getmembers()}
    for m, _ in parallel_iter_tar_articles(tar, workers=2):
        assert m.size == sizes[m.name]
        assert m.mtime == 1_700_000_000


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.tar", True),
        ("A.TAR", True),
        ("a.tar.gz", False),
        ("a.tgz", False),
        ("a.tar.bz2", False),
        ("a.tbz2", False),
        ("a.tar.xz", False),
        ("a.txz", False),
        ("a.zip", False),
        ("notes.txt", False),
    ],
)
def test_is_uncompressed_tar(tmp_path, name, expected):
    assert is_uncompressed_tar(tmp_path / name) is expected


# ------------------------------------------------------------------ manifests and dirs


@pytest.fixture
def tar_dir(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    for name in ["b.tar", "a.tar.gz", "c.tgz", "notes.txt", "a.tar.gz.partial"]:
        (d / name).write_bytes(b"")
    (d / "sub").mkdir()
    (d / "sub" / "deep.tar").write_bytes(b"")
    return d


def test_directory_scan_is_sorted_top_level_tars_only(tar_dir):
    assert [p.name for p in iter_tar_paths(tar_dir)] == ["a.tar.gz", "b.tar", "c.tgz"]


def test_manifest_entries(tar_dir, tmp_path):
    manifest = tar_dir / "corpus.manifest"
    manifest.write_text(
        "# comment line\n"
        "\n"
        "b.tar   # inline comment\n"
        "a\n"  # no extension: tries .tar, .tar.gz, ...
        f"{tar_dir / 'c.tgz'}\n"  # absolute
        "./b.tar\n"  # same file again: de-duplicated
        "missing.tar\n"  # unresolved: skipped
        "notes.txt\n"  # not a tar: skipped
        "sub/deep.tar\n"  # relative with a directory
    )
    got = list(iter_tar_paths(None, manifest))
    expected = [tar_dir / n for n in ["b.tar", "a.tar.gz", "c.tgz", "sub/deep.tar"]]
    assert got == [p.resolve() for p in expected]


def test_manifest_relative_entries_use_tar_dir_when_given(tar_dir, tmp_path):
    manifest = tmp_path / "elsewhere.manifest"
    manifest.write_text("b.tar\n")
    assert [p.name for p in iter_tar_paths(tar_dir, manifest)] == ["b.tar"]
    assert list(iter_tar_paths(None, manifest)) == []  # resolved next to the manifest


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#19: dotted name, no extension")
def test_manifest_dotted_name_without_extension(tmp_path):
    (tmp_path / "oa_comm_xml.PMC000xxxxxx.baseline.2025-06-26.tar.gz").write_bytes(b"")
    manifest = tmp_path / "m.manifest"
    manifest.write_text("oa_comm_xml.PMC000xxxxxx.baseline.2025-06-26\n")
    assert len(list(iter_tar_paths(None, manifest))) == 1


def test_no_dir_and_no_manifest_is_an_error():
    with pytest.raises(ValueError, match="--tar-dir or --tar-manifest"):
        list(iter_tar_paths(None))


# ------------------------------------------------------------------------- sharding


def _sized_files(tmp_path, sizes):
    paths = []
    for i, n in enumerate(sizes):
        p = tmp_path / f"t{i:02d}.tar"
        p.write_bytes(b"\0" * n)
        paths.append(p)
    return paths


@pytest.mark.parametrize("num_shards", [2, 3, 7, 12])
def test_every_file_goes_to_exactly_one_shard(tmp_path, num_shards):
    # Repeated sizes exercise tie-breaking; 12 shards > 10 exercises two-digit ids.
    paths = _sized_files(tmp_path, [5, 5, 5, 1, 9, 9, 2, 7, 3, 3, 8, 4, 6, 1])
    shards = [list(shard_filter(paths, s, num_shards)) for s in range(num_shards)]
    flat = [p for shard in shards for p in shard]
    assert sorted(flat) == sorted(paths)
    assert len(flat) == len(set(flat))


def test_sharding_balances_bytes(tmp_path):
    # Shuffled, so the result depends on shard_filter sorting largest-first.
    sizes = [10, 100, 30, 90, 20, 80, 40, 70, 50, 60]
    paths = _sized_files(tmp_path, sizes)
    loads = [sum(p.stat().st_size for p in shard_filter(paths, s, 3)) for s in range(3)]
    assert sum(loads) == 550
    # Hand-run of greedy largest-first into the least-loaded shard (lowest index on ties):
    # 100->0, 90->1, 80->2, 70->2, 60->1, 50->0, 40->0, 30->1, 20->2, 10->2 = 190, 180, 180.
    # In the given order instead, it would be 200, 170, 180.
    assert loads == [190, 180, 180]


def test_single_shard_passes_paths_through_without_stat(tmp_path):
    missing = [tmp_path / "nope.tar"]
    assert list(shard_filter(missing, 0, 1)) == missing


def test_missing_file_fails_loudly_when_sharding(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot stat tar file"):
        list(shard_filter([tmp_path / "nope.tar"], 0, 2))
