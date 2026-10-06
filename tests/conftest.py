"""Shared fixtures for the litkit test suite.

Every test runs with litkit's environment variables cleared and a fresh
workspace under ``tmp_path``, so no test can read or write a real workspace,
model cache or API endpoint.
"""

import hashlib
import io
import os
import re
import sys
import tarfile
import tempfile

# macOS: torch, faiss-cpu and scikit-learn each ship libomp, and a process that
# initializes two of them aborts (#16). litkit survives only because importing
# sentence-transformers sets KMP_DUPLICATE_LIB_OK as a side effect. With two
# runtimes loaded, torch then segfaults in multi-threaded CPU kernels unless
# OpenMP runs one thread, which is what litkit's __main__ and cli.py set.
# The slow tier runs torch after FAISS in one process, so it needs both, set
# before faiss or torch is imported.
# Forced, not defaulted: OMP_NUM_THREADS=8 in the user's shell segfaults the slow tier.
if sys.platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["FAISS_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# Read before the variables below are cleared: the slow tests load real models
# from this Hugging Face cache (a directory containing hub/models--...).
REAL_HF_HOME = os.environ.get("LITKIT_TEST_HF_HOME")

# Variables that point litkit at real data, models or endpoints, or change its
# behavior.
_ENV_PREFIXES = ("LITKIT_", "HF_", "TRANSFORMERS_", "OPENAI_")
_ENV_NAMES = ("LLM_MODEL",)


def pytest_configure(config):
    """Clear the variables before pytest imports any test module.

    Several are read when a litkit module is imported (LITKIT_USE_DOWNCAST_FALLBACK,
    LITKIT_SAVE_EVERY_SEC, LITKIT_NO_LEXICAL, ...), which happens at collection,
    before any fixture runs; isolated_env clears them again for each test. This is
    a hook rather than import-time code because tests/integration/run_litkit_fake.py
    imports this module inside the CLI subprocess, where the variables are the
    test's own settings.
    """
    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES) or name in _ENV_NAMES:
            del os.environ[name]
    # Offline, with an empty cache, until isolated_env takes over for each test;
    # this covers module-scoped fixtures and imports that run before it.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = tempfile.mkdtemp(prefix="litkit-test-hf-")


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Clear litkit-related environment variables, point at a fresh workspace, and
    reset module-level state (FAISS save throttle and cache, pending in_index marks,
    cached workspace paths) before and after each test."""
    for name in list(os.environ):
        if name.startswith(_ENV_PREFIXES) or name in _ENV_NAMES:
            monkeypatch.delenv(name, raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("LITKIT_WORKSPACE", str(workspace))
    monkeypatch.setenv("LITKIT_ASSUME_YES", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    _reset_module_state()
    yield workspace
    _reset_module_state()


def _reset_module_state():
    """Reset litkit state that outlives a test, for modules that are already imported."""
    if "litkit.index.io" in sys.modules:
        io_mod = sys.modules["litkit.index.io"]
        io_mod._last_save_ts.update({"papers": 0.0, "chunks": 0.0})
        io_mod.clear_faiss_cache()
    if "litkit.db.indexing" in sys.modules:
        sys.modules["litkit.db.indexing"].clear_pending_marks()
    if "litkit.config.paths" in sys.modules:
        sys.modules["litkit.config.paths"].reset_default_paths()
    if "litkit.concurrent.locking" in sys.modules:
        sys.modules["litkit.concurrent.locking"].reset_default_lock_manager()


# --------------------------------------------------------------------------- fakes


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEmbedder:
    """Deterministic stand-in for SPECTER2 / SBERT with the same encode() contract.

    Each text becomes a bag-of-words vector: every token adds a fixed
    pseudo-random unit vector derived from sha256(token), and the sum is
    L2-normalized. Texts that share words therefore score higher than texts
    that don't, which is enough for retrieval tests to have a known answer.
    Returns float32 arrays of shape (len(texts), dim); empty input gives (0, dim).
    """

    def __init__(self, dim=32):
        self.dim = dim
        self.calls = []

    def _token_vec(self, tok):
        seed = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self.dim)
        return v / np.linalg.norm(v)

    def encode(self, texts, progress_label=None, batch_size=None, progress_done_summary=True):
        texts = list(texts)
        self.calls.append(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            v = np.zeros(self.dim)
            for tok in _tokens(text) or ["<empty>"]:
                v += self._token_vec(tok)
            out[i] = v / np.linalg.norm(v)
        return out

    def close(self):
        pass


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


# --------------------------------------------------------------------- tiny corpus


def jats(pmcid="", pmid="", title="", abstract="", paragraphs=()):
    """A minimal JATS article with the article-id types PMC uses."""
    ids = ""
    if pmcid:
        ids += f'<article-id pub-id-type="pmcid">{pmcid}</article-id>'
        ids += f'<article-id pub-id-type="pmcid-ver">{pmcid}.1</article-id>'
        ids += f'<article-id pub-id-type="pmcaid">{pmcid[3:]}</article-id>'
    if pmid:
        ids += f'<article-id pub-id-type="pmid">{pmid}</article-id>'
    abs_xml = f"<abstract><p>{abstract}</p></abstract>" if abstract else ""
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (
        '<?xml version="1.0"?>\n'
        '<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">'
        f"<front><article-meta>{ids}"
        f"<title-group><article-title>{title}</article-title></title-group>"
        f"{abs_xml}</article-meta></front>"
        f"<body><sec><title>Results</title>{body}</sec></body>"
        "<back><ref-list><ref><mixed-citation>Ref text that must not be a paragraph, "
        "long enough to pass the 40-character filter.</mixed-citation></ref></ref-list></back>"
        "</article>"
    )


def pubmed_xml(pmid, title, abstract):
    return (
        '<?xml version="1.0"?>\n<PubmedArticleSet><PubmedArticle><MedlineCitation>'
        f"<PMID>{pmid}</PMID><Article><ArticleTitle>{title}</ArticleTitle>"
        f"<Abstract><AbstractText>{abstract}</AbstractText></Abstract>"
        "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )


def _para(topic, i):
    return (
        f"{topic} paragraph {i} describes the {topic} experiments in enough detail "
        f"to form a body paragraph about {topic} for chunking tests."
    )


# Three topics with disjoint vocabularies, so a fake-embedder query about one
# topic retrieves that paper. Paragraph counts give known chunk counts.
CORPUS = {
    "hiv": dict(
        pmcid="PMC100001",
        pmid="900001",
        title="HIV entry requires CD4",
        abstract="HIV binds CD4 and CCR5 on T cells.",
        paragraphs=[_para("retrovirus", i) for i in range(6)],
    ),
    "egfr": dict(
        pmcid="PMC100002",
        pmid="900002",
        title="EGFR signaling in carcinoma",
        abstract="EGFR drives proliferation in carcinoma.",
        paragraphs=[_para("kinase", i) for i in range(4)],
    ),
    "malaria": dict(
        pmcid="PMC100003",
        pmid="",
        title="Plasmodium liver stage",
        abstract="Sporozoites invade hepatocytes.",
        paragraphs=[_para("parasite", i) for i in range(2)],
    ),
}


def write_tar(path, members):
    """Write {member_name: text} into an uncompressed (or .tar.gz) tar at path."""
    mode = "w:gz" if path.name.endswith(".tar.gz") else "w"
    with tarfile.open(path, mode) as tf:
        for name, text in members.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 1_700_000_000
            tf.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def tiny_corpus(tmp_path):
    """Two tars of JATS articles plus edge cases, under tmp_path/tars.

    corpus_a.tar: the three CORPUS articles, an empty <article>, an article
    with no abstract, a PubMed (non-JATS) record and a non-XML member.
    corpus_b.tar.gz: a second copy of the HIV article (same PMCID).
    """
    d = tmp_path / "tars"
    d.mkdir()
    members = {f"{k}/{v['pmcid']}.nxml": jats(**v) for k, v in CORPUS.items()}
    members["edge/empty.nxml"] = '<?xml version="1.0"?><article></article>'
    members["edge/noabstract.nxml"] = jats(
        pmcid="PMC100004", title="No abstract here", paragraphs=[_para("enzyme", 0)]
    )
    members["edge/pubmed.xml"] = pubmed_xml("900005", "A PubMed record", "Only an abstract.")
    members["edge/readme.txt"] = "not xml"
    write_tar(d / "corpus_a.tar", members)
    write_tar(d / "corpus_b.tar.gz", {"dup/PMC100001.nxml": jats(**CORPUS["hiv"])})
    return d


# ------------------------------------------------------------------- FAISS indexes


@pytest.fixture
def rng():
    """Seeded generator; tests that use it should pass for any seed."""
    return np.random.default_rng(20261006)


@pytest.fixture
def make_index():
    """Factory: an IndexIDMap2-wrapped flat, HNSW or IVF-PQ index holding X under ids.

    IVF-PQ uses nlist=8 and 8 sub-quantizers of 8 bits, which needs a few hundred
    training vectors (FAISS warns below ~39 per centroid); pass at least 400.
    """
    import faiss

    from litkit.index.factory import flat_ip_index, hnsw_index, ivfpq_index

    def make(kind, X, ids=None):
        dim = X.shape[1]
        if kind == "flat":
            core = flat_ip_index(dim)
        elif kind == "hnsw":
            core = hnsw_index(dim, M=16)
        elif kind == "ivfpq":
            core = ivfpq_index(dim, nlist=8, m=8)
            core.train(X)
        else:
            raise ValueError(kind)
        index = faiss.IndexIDMap2(core)
        ids = np.arange(len(X), dtype="int64") if ids is None else np.asarray(ids, "int64")
        index.add_with_ids(X, ids)
        return index

    return make


@pytest.fixture
def unit_vectors(rng):
    """Factory: n random L2-normalized float32 vectors of dimension dim."""

    def make(n, dim=32):
        x = rng.standard_normal((n, dim)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    return make
