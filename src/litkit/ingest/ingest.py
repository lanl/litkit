"""ingest.ingest — TAR-only ingest helpers for PubMed Central Open Access (PMC-OA) NXML.

This module provides small, dependency-light utilities for reading article
metadata and body text from JATS/NXML contained inside tar
archives (.tar, .tar.gz/.tgz, .tar.bz2/.tbz2, .tar.xz/.txz). It also includes
a simple paragraph packer for turning body text into chunk-sized strings.
XML documents are intentionally ignored: see setting for _XML_EXTS

Highlights
----------
- Defensive XML parsing via lxml (recovery on; DTD/network/entity resolution off).
- Streaming tar access that skips unreadable shards with a warning.
- Generators that yield member names or (TarInfo, fileobj) pairs; file objects
  are safely closed even on early loop exit.
- Normalized output structure described by `ArticleMeta` (TypedDict).

Public API
----------
Constants:
    CHUNK_TARGET_CHARS, BODY_MIN_CHARS, CHUNK_OVERLAP_CHARS

XML parsing:
    parse_jats(tree), parse_pubmed(tree), parse_xml_fileobj(fobj)

Chunking:
    pack_paragraphs(paras, max_chars=..., min_chars=..., overlap_chars=...)

TAR helpers:
    iter_tar_paths(tar_dir, manifest=None)
    iter_tar_xml_member_names(tar_path, exts=...)
    count_tar_xml_members(tar_path, exts=...)
    iter_tar_xml_streams(tar_path, exts=...)

Notes:
-----
- Unreadable archives are skipped and logged at WARNING level.
- These helpers do not mutate inputs and avoid loading entire archives into
  memory when possible.
- Intended to be composed by higher-level ingest pipelines; no global state.
"""

from __future__ import annotations

import logging
import tarfile
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO, Any, TypedDict

from lxml import etree

logger = logging.getLogger(__name__)

# Local defaults kept to avoid new coupling; callers usually pass explicit values.
CHUNK_TARGET_CHARS = 1200
BODY_MIN_CHARS = 300
CHUNK_OVERLAP_CHARS = 200

# Only ingest NXML files because we are processing the PMC-OA corpus (JATS/NXML files)
#   Ingestion can be modified to ingest only PubMed XML files or both XML and NXML.
#   Setting for XML only: _XML_EXTS = (".xml")
#   Setting for both XML and NXML: _XML_EXTS = (".xml", ".nxml")
_XML_EXTS = (".nxml",)
_TAR_EXTS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


class ArticleMeta(TypedDict):
    """Normalized metadata for a single article extracted from JATS or PubMed XML.

    Keys
    ----
    pmid : str
        PubMed identifier (empty string if unavailable).
    pmcid : str
        PubMed Central identifier (e.g., "PMC123456"; empty string if unavailable).
    title : str
        Article title; empty string if missing.
    abstract : str
        Abstract text with whitespace normalized; empty string if missing.
    paragraphs : list[str]
        Body paragraphs (whitespace-normalized), excluding figures/tables/captions/refs.
        Entries are typically >= 40 characters as filtered by the parser.
    """

    pmid: str
    pmcid: str
    title: str
    abstract: str
    paragraphs: list[str]


@contextmanager
def _open_tar_safely(tar_path: str | Path):
    """Open a tar (any compression). Yields None if unreadable and then logs a warning."""
    try:
        tf = tarfile.open(str(tar_path), mode="r:*")
    except (tarfile.TarError, OSError) as e:
        logger.warning("[tar] skipping unreadable shard %s: %s", tar_path, e)
        yield None
        return
    try:
        yield tf
    finally:
        try:
            tf.close()
        except Exception:
            pass


# -------------------------------
# Text + XML parsers
# -------------------------------
def _txt(node: Any) -> str:
    """Extract normalized visible text from an XML node (collapse whitespace)."""
    if node is None:
        return ""
    return " ".join(" ".join(node.itertext()).split())


def parse_jats(tree: etree._ElementTree) -> ArticleMeta:
    """Parse JATS (N)XML article into metadata and paragraph list.
    Returns: {pmid, pmcid, title, abstract, paragraphs}
    """
    xp = tree.xpath

    pmid = (
        xp(
            'string(//*[local-name()="article-id"]'
            '[translate(@pub-id-type,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz")="pmid"][1])'
        ).strip()
        or ""
    )

    pmcid = (
        xp(
            'string(//*[local-name()="article-id"]'
            '[translate(@pub-id-type,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz")="pmcid"][1])'
        ).strip()
        or ""
    )

    title = xp('string(//*[local-name()="title-group"]/*[local-name()="article-title"][1])').strip()
    if not title:
        title = xp('string(//*[local-name()="article-title"][1])').strip()

    abstract_nodes = xp('//*[local-name()="abstract"]')
    abstract = _txt(abstract_nodes[0]) if abstract_nodes else ""

    paras: list[str] = []
    # Body paragraphs but not in ref-list/table-wrap/fig/caption
    for p in xp(
        '//*[local-name()="body"]//*[local-name()="p" and '
        'not(ancestor::*[local-name()="ref-list"] or '
        'ancestor::*[local-name()="table-wrap"] or '
        'ancestor::*[local-name()="fig"] or '
        'ancestor::*[local-name()="caption"])]'
    ):
        t = _txt(p)
        if len(t) >= 40:
            paras.append(t)

    return {
        "pmid": pmid,
        "pmcid": pmcid,
        "title": title or "",
        "abstract": abstract or "",
        "paragraphs": paras,
    }


def parse_pubmed(tree: etree._ElementTree) -> ArticleMeta:
    """Parse PubMed XML (non-JATS) into a consistent structure."""
    xp = tree.xpath
    pmid = xp('string(//*[local-name()="PubmedArticle"]//*[local-name()="PMID"][1])').strip() or ""
    title = xp('string(//*[local-name()="Article"]/*[local-name()="ArticleTitle"][1])').strip()
    abs_nodes = xp(
        '//*[local-name()="Article"]/*[local-name()="Abstract"]/*[local-name()="AbstractText"]'
    )
    abstract = " ".join([_txt(n) for n in abs_nodes]).strip()
    return {
        "pmid": pmid,
        "pmcid": "",
        "title": title or "",
        "abstract": abstract or "",
        "paragraphs": [],
    }


def parse_xml_fileobj(fobj: IO[bytes]) -> ArticleMeta | None:
    """Parse a single XML/JATS/PubMed record from an open binary file-like object.

    This uses a defensive lxml parser (recovery enabled; entity resolution, DTD
    loading, and network access disabled) to tolerate imperfect inputs and avoid
    external resource fetches. Large documents are allowed via `huge_tree=True`.

    Parameters
    ----------
    fobj : IO[bytes]
        Readable binary file-like object positioned at the start of an XML
        document. The caller retains ownership and is responsible for closing it.

    Returns:
    -------
    Optional[ArticleMeta]
        Parsed metadata on success; `None` if parsing fails or if the document
        does not contain recognizable JATS/PubMed content.

    Notes:
    -----
    - Parse errors are swallowed and result in `None` rather than raising.
    - The returned structure is produced by `_parse_tree(...)`.
    """
    try:
        parser = etree.XMLParser(
            recover=True,
            huge_tree=True,
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
            dtd_validation=False,
        )
        tree = etree.parse(fobj, parser=parser)
    except Exception:
        return None
    return _parse_tree(tree)


def _parse_tree(tree: etree._ElementTree) -> ArticleMeta | None:
    root = tree.getroot()
    rootname = etree.QName(root).localname.lower() if root is not None else ""
    meta = parse_jats(tree) if rootname == "article" else parse_pubmed(tree)
    # Fallback: sometimes PubMed-ish roots still contain JATS sections
    if not (meta["title"] or meta["abstract"]):
        meta = parse_jats(tree)
    if not (meta["title"] or meta["abstract"] or meta["paragraphs"]):
        return None
    return meta


# -------------------------------
# Chunking helper
# -------------------------------
def pack_paragraphs(
    paras: list[str],
    max_chars: int = CHUNK_TARGET_CHARS,
    min_chars: int = BODY_MIN_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Greedy pack paragraphs, then add a small character overlap between
    consecutive chunks to reduce claim-splitting.

    The overlap is a character window from the previous chunk’s tail: for each
    chunk i > 0, we prefix it with the last `overlap_chars` characters of chunk i-1.
    """
    chunks: list[str] = []
    buf: list[str] = []
    total = 0

    def _flush_buf():
        nonlocal chunks, buf, total
        if not buf:
            return
        cur = " ".join(buf)
        if len(cur) < min_chars and chunks:
            chunks[-1] = chunks[-1] + " " + cur
        else:
            chunks.append(cur)
        buf, total = [], 0

    for p in paras:
        if buf and total + len(p) + 1 > max_chars:
            _flush_buf()
        buf.append(p)
        total += len(p) + 1
    _flush_buf()

    if overlap_chars > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap_chars:]
            out.append((tail + " " + chunks[i]).strip())
        chunks = out
    return chunks


# -------------------------------
# TAR helpers
# -------------------------------
def _is_tar_path(p: Path) -> bool:
    s = p.name.lower()
    return s.endswith(_TAR_EXTS)


def iter_tar_paths(tar_dir: str | Path, manifest: str | Path | None = None) -> Iterator[Path]:
    """Yield paths to tar-like archives under `tar_dir`, with optional manifest control.

    If `manifest` is provided, it is read line-by-line to select which shards to yield.
    Each non-empty, non-comment line may be:
      - an absolute or relative path to a tar file, or
      - a basename (optionally without an extension). If no extension is given,
        each known tar extension in `_TAR_EXTS` is tried in order.

    Lines starting with '#' and blank lines are ignored. Duplicate resolved paths
    are de-duplicated while preserving their first occurrence order.

    Without a manifest, the function lists *only the top level* of `tar_dir`
    (non-recursive) and yields files whose names end with any extension in
    `_TAR_EXTS`, in lexicographic order.

    Parameters
    ----------
    tar_dir : str | Path
        Root directory used to resolve relative manifest entries and, when no
        manifest is given, the directory to scan for tar files.
    manifest : str | Path, optional
        Optional path to a text file specifying shards (one per line).

    Yields:
    ------
    Path
        Resolved filesystem paths to existing files whose names end with a known
        tar extension. Unresolvable tokens (missing files / wrong suffix) are skipped.

    Notes:
    -----
    This function does not open or validate the files as tar archives; it only
    checks for existence and filename suffix via `_is_tar_path`.
    """
    root = Path(tar_dir)

    def _resolve_token(tok: str) -> Path | None:
        tok = tok.strip()
        if not tok:
            return None
        p = Path(tok)
        if p.is_absolute():
            return p if p.exists() and _is_tar_path(p) else None
        candidate = root / tok
        if candidate.exists() and _is_tar_path(candidate):
            return candidate
        if candidate.suffix == "":  # try common compressions when no suffix
            base = candidate  # no suffix to strip
            for ext in _TAR_EXTS:
                c3 = Path(str(base) + ext)
                if c3.exists() and _is_tar_path(c3):
                    return c3
        return None

    if manifest:
        seen: set[Path] = set()
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = _resolve_token(line)
            if p and p not in seen:
                seen.add(p)
                yield p  # preserve manifest order
    else:
        for p in sorted(root.iterdir()):  # not recursive, by design
            if p.is_file() and _is_tar_path(p):
                yield p


def iter_tar_xml_member_names(
    tar_path: str | Path,
    exts: Iterable[str] = _XML_EXTS,
) -> Iterator[str]:
    """Iterate member names inside a tar that look like XML, streaming safely."""
    lower_exts = tuple(e.lower() for e in exts)
    stack = ExitStack()
    try:
        tf = stack.enter_context(_open_tar_safely(tar_path))
        if tf is None:
            return  # unreadable: yield nothing

        # Stream through the archive without loading all headers into memory.
        # TarFile.next() returns the next TarInfo or None at EOF.
        while True:
            try:
                m = tf.next()  # type: ignore[attr-defined]
            except (tarfile.TarError, OSError) as e:
                logger.warning("[tar] error iterating %s: %s", tar_path, e)
                break
            if m is None:
                break
            if m.isfile() and m.name.lower().endswith(lower_exts):
                yield m.name
    finally:
        stack.close()


def count_tar_xml_members(tar_path: str | Path, exts: Iterable[str] = _XML_EXTS) -> int:
    """Count XML/NXML members in a tar/tgz."""
    lower_exts = tuple(e.lower() for e in exts)
    stack = ExitStack()
    try:
        tf = stack.enter_context(_open_tar_safely(tar_path))
        if tf is None:
            return 0
        count = 0
        while True:
            try:
                m = tf.next()  # stream headers without loading full table
            except (tarfile.TarError, OSError):
                break
            if m is None:
                break
            if m.isfile() and m.name.lower().endswith(lower_exts):
                count += 1
        return count
    finally:
        stack.close()


def iter_tar_xml_streams(
    tar_path: str | Path,
    exts: Iterable[str] = _XML_EXTS,
) -> Iterator[tuple[tarfile.TarInfo, IO[bytes]]]:
    lower_exts = tuple(e.lower() for e in exts)
    stack = ExitStack()
    try:
        tf = stack.enter_context(_open_tar_safely(tar_path))
        if tf is None:
            return
        while True:
            try:
                m = tf.next()
            except (tarfile.TarError, OSError) as e:
                logger.warning("[tar] error iterating %s: %s", tar_path, e)
                break
            if m is None:
                break
            if not (m.isfile() and m.name.lower().endswith(lower_exts)):
                continue
            try:
                f = tf.extractfile(m)
            except (tarfile.TarError, OSError) as e:
                logger.warning("[tar] cannot extract %s from %s: %s", m.name, tar_path, e)
                continue
            if f is None:
                continue
            try:
                yield m, f
            finally:
                try:
                    f.close()
                except Exception:
                    pass
    finally:
        stack.close()


# Public API
__all__ = [
    "ArticleMeta",
    # constants
    "CHUNK_TARGET_CHARS",
    "BODY_MIN_CHARS",
    "CHUNK_OVERLAP_CHARS",
    # parsers + helpers
    "parse_jats",
    "parse_pubmed",
    "parse_xml_fileobj",
    "pack_paragraphs",
    # tar helpers
    "iter_tar_paths",
    "iter_tar_xml_member_names",
    "count_tar_xml_members",
    "iter_tar_xml_streams",
]
