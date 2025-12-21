"""ingest.ingest — TAR-only ingest helpers for PubMed Central Open Access (PMC-OA) NXML.

This module provides small, dependency-light utilities for reading article
metadata and body text from JATS/NXML contained inside tar
archives (.tar, .tar.gz/.tgz, .tar.bz2/.tbz2, .tar.xz/.txz). It also includes
a simple paragraph packer for turning body text into chunk-sized strings.

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
    iter_tar_paths(tar_dir=None, manifest=None)
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

import io
import logging
import tarfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import IO, Any, TypedDict

from lxml import etree

logger = logging.getLogger(__name__)

# Local defaults kept to avoid new coupling; callers usually pass explicit values.
CHUNK_TARGET_CHARS = 1200
BODY_MIN_CHARS = 300
CHUNK_OVERLAP_CHARS = 200

# Ingest XML and JATS/NXML files
_XML_EXTS = (".nxml", ".xml")
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


def iter_tar_paths(
    tar_dir: str | Path | None,
    manifest: str | Path | None = None,
) -> Iterator[Path]:
    """Yield paths to tar-like archives, optionally driven by a manifest.

    If `manifest` is provided, it is read line-by-line to select which shards to yield.
    Each non-empty, non-comment line may be:
      - an absolute or relative path to a tar file, or
      - a basename (optionally without an extension). If no extension is given,
        each known tar extension in `_TAR_EXTS` is tried in order.

    When resolving relative entries from a manifest, the base directory is:
      * `Path(tar_dir)` if `tar_dir` is given, else
      * the directory containing the manifest file.

    Lines starting with '#' and blank lines are ignored. Duplicate resolved paths
    are de-duplicated while preserving their first occurrence order.

    Without a manifest, the function lists *only the top level* of `tar_dir`
    (non-recursive) and yields files whose names end with any extension in
    `_TAR_EXTS`, in lexicographic order.

    In a manifest file, inline comments after # are allowed.

    Parameters
    ----------
    tar_dir : str | Path | None
        Base directory used to resolve relative manifest entries and, when no
        manifest is given, the directory to scan for tar files. May be None when
        a manifest is provided. In that case, the manifest's parent is used.
    manifest : str | Path | None
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

    def _resolve_token(tok: str, base: Path) -> Path | None:
        tok = tok.strip()
        if not tok:
            return None
        p = Path(tok)
        if p.is_absolute():
            return p if p.exists() and _is_tar_path(p) else None
        candidate = base / tok
        if candidate.exists() and _is_tar_path(candidate):
            return candidate
        # If no suffix was provided, try common tar compressions.
        if candidate.suffix == "":
            for ext in _TAR_EXTS:
                c3 = Path(str(candidate) + ext)
                if c3.exists() and _is_tar_path(c3):
                    return c3
        return None


    if manifest is not None:
        mf = Path(manifest)
        base = Path(tar_dir) if tar_dir is not None else mf.parent
        seen: set[Path] = set()
        for line in mf.read_text(encoding="utf-8").splitlines():
            raw = line.split("#", 1)[0].strip()  # allow inline comments
            if not raw:
                continue
            p = _resolve_token(raw, base)
            if p:
                rp = p.resolve(strict=False)      # normalize for de-duplication
                if rp not in seen:
                    seen.add(rp)
                    yield rp                      # preserve manifest order
            else:
                logger.warning(
                    "[manifest] skipping unresolved entry %r (base=%s)", raw, base
                )
        return
    

    # No manifest: require a directory to scan.
    if tar_dir is None:
        raise ValueError("Either --tar-dir or --tar-manifest must be provided.")

    root = Path(tar_dir)
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


# -------------------------------
# Parallel XML parsing
# -------------------------------
def _parse_xml_bytes(data: bytes) -> ArticleMeta | None:
    """Parse XML from bytes. Thread-safe worker function."""
    try:
        parser = etree.XMLParser(
            recover=True,
            huge_tree=True,
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
            dtd_validation=False,
        )
        tree = etree.parse(io.BytesIO(data), parser=parser)
    except Exception:
        return None
    return _parse_tree(tree)


class TarMemberMeta:
    """Minimal file metadata from a tar member, used for checkpointing."""
    __slots__ = ("name", "size", "mtime")

    def __init__(self, name: str, size: int, mtime: float):
        self.name = name
        self.size = size
        self.mtime = mtime


def parallel_iter_tar_articles(
    tar_path: str | Path,
    workers: int = 8,
    exts: Iterable[str] = _XML_EXTS,
    yield_in_order: bool = True,
) -> Iterator[tuple[TarMemberMeta, ArticleMeta]]:
    """Iterate over articles in a tar file, parsing XML in parallel.

    This function reads tar members sequentially (tar format requires this),
    but parses the XML content in parallel using a thread pool.

    Parameters
    ----------
    tar_path : str | Path
        Path to tar file (compressed or uncompressed).
    workers : int
        Number of parallel parsing threads (default: 8).
    exts : Iterable[str]
        File extensions to consider as XML (default: .nxml, .xml).
    yield_in_order : bool
        If True (default), results are yielded in tar file order to support
        count-based checkpoint resume. If False, yields in completion order
        (faster but incompatible with checkpoint resume by count).

    Yields
    ------
    tuple[TarMemberMeta, ArticleMeta]
        (member_meta, parsed_article) for each successfully parsed XML file.
        member_meta contains name, size, mtime for checkpointing.

    Notes
    -----
    - Uncompressed .tar files allow faster I/O since no decompression is needed.
    - XML parsing (lxml) releases the GIL during C operations, so threading
      provides real parallelism.
    - With yield_in_order=True, uses a reorder buffer to preserve tar order.
    - Memory usage is bounded: O(workers * 4) results buffered at most.
    """
    import heapq
    
    lower_exts = tuple(e.lower() for e in exts)

    with _open_tar_safely(tar_path) as tf:
        if tf is None:
            return

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}  # future -> (sequence, meta)
            pending_count = 0
            max_pending = workers * 4  # limit memory: ~4 XMLs per worker in flight
            
            # For in-order yielding
            sequence = 0  # monotonically increasing submission order (tar order)
            next_yield_seq = 0  # next sequence number to yield
            result_heap: list[tuple[int, TarMemberMeta, ArticleMeta]] = []  # (seq, meta, article)
            eof_reached = False

            while True:
                # Submit new work while under the limit and not at EOF
                while not eof_reached and pending_count < max_pending:
                    try:
                        m = tf.next()
                    except (tarfile.TarError, OSError) as e:
                        logger.warning("[tar] error iterating %s: %s", tar_path, e)
                        m = None
                    if m is None:
                        eof_reached = True
                        break
                    if not (m.isfile() and m.name.lower().endswith(lower_exts)):
                        continue
                    try:
                        f = tf.extractfile(m)
                        if f is None:
                            continue
                        data = f.read()
                        f.close()
                    except (tarfile.TarError, OSError) as e:
                        logger.warning("[tar] cannot extract %s: %s", m.name, e)
                        continue

                    meta = TarMemberMeta(
                        name=m.name,
                        size=int(getattr(m, "size", 0)),
                        mtime=float(getattr(m, "mtime", 0.0) or 0.0),
                    )
                    fut = pool.submit(_parse_xml_bytes, data)
                    futures[fut] = (sequence, meta)
                    sequence += 1
                    pending_count += 1

                if not futures:
                    # Drain any remaining buffered results (in-order mode)
                    if yield_in_order:
                        while result_heap:
                            _, meta, article = heapq.heappop(result_heap)
                            yield (meta, article)
                    break  # no more work

                # Collect completed results
                done_futures = [f for f in futures if f.done()]
                if not done_futures:
                    # Wait for at least one to complete
                    import concurrent.futures
                    done, _ = concurrent.futures.wait(
                        futures.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    done_futures = list(done)

                for fut in done_futures:
                    seq, meta = futures.pop(fut)
                    pending_count -= 1
                    try:
                        result = fut.result()
                        if result is not None:
                            if yield_in_order:
                                # Buffer for in-order yielding
                                heapq.heappush(result_heap, (seq, meta, result))
                            else:
                                # Original behavior: yield immediately
                                yield (meta, result)
                    except Exception as e:
                        logger.warning("[parse] error parsing %s: %s", meta.name, e)
                        if yield_in_order:
                            # Must track failed parses to not block the sequence
                            # We increment next_yield_seq when we would have yielded this
                            # Since result is None, we skip it but advance the sequence
                            heapq.heappush(result_heap, (seq, meta, None))  # type: ignore
                
                # Yield buffered results in order
                if yield_in_order:
                    while result_heap and result_heap[0][0] == next_yield_seq:
                        _, meta, article = heapq.heappop(result_heap)
                        next_yield_seq += 1
                        if article is not None:
                            yield (meta, article)


# Public API
__all__ = [
    "ArticleMeta",
    "TarMemberMeta",
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
    "parallel_iter_tar_articles",
]
