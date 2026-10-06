"""Contract tests for the real SPECTER2 and SBERT embedders.

Marked slow (and gpu for the device test), so they are excluded by default.
They need the two model snapshots in a local Hugging Face cache:

    LITKIT_TEST_HF_HOME=/path/to/hf_cache uv run pytest -m slow
"""

from pathlib import Path

import numpy as np
import pytest

from conftest import REAL_HF_HOME

MODELS = ("allenai/specter2_base", "sentence-transformers/all-mpnet-base-v2")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not REAL_HF_HOME
        or not all(
            (Path(REAL_HF_HOME) / "hub" / f"models--{m.replace('/', '--')}").is_dir()
            for m in MODELS
        ),
        reason="set LITKIT_TEST_HF_HOME to a cache with SPECTER2 and all-mpnet-base-v2",
    ),
]

TEXTS = [
    "HIV binds CD4 and CCR5 to enter T cells.",
    "The human immunodeficiency virus uses the CD4 receptor for cell entry.",
    "Glaciers in the Alps retreated during the twentieth century.",
]


@pytest.fixture(scope="module", autouse=True)
def heavy_dependencies():
    # The lock has no torch wheel for Linux aarch64 (it comes from the container
    # build), so a plain `uv sync` there can't run this tier: skip, don't error.
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("transformers")


@pytest.fixture(autouse=True)
def real_model_cache(monkeypatch):
    """Runs after conftest's isolated_env, so it points HF_HOME back at the real cache."""
    monkeypatch.setenv("HF_HOME", REAL_HF_HOME)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module", params=["paper", "chunk"])
def embedder(request):
    mp = pytest.MonkeyPatch()
    mp.setenv("HF_HOME", REAL_HF_HOME)
    mp.setenv("HF_HUB_OFFLINE", "1")
    from litkit.embeddings.sbert_mpnet import ChunkEmbedderSBERT
    from litkit.embeddings.specter2 import PaperEmbedderSpecter2

    if request.param == "paper":
        emb = PaperEmbedderSpecter2(device="cpu")
    else:
        emb = ChunkEmbedderSBERT(devices=["cpu"])
    mp.undo()
    yield emb


def test_shape_dtype_and_unit_norm(embedder):
    X = embedder.encode(TEXTS)
    assert X.shape == (3, embedder.dim) == (3, 768)
    assert X.dtype == np.float32
    # float32 normalization of 768-dim vectors: |norm - 1| ~ 1e-7; 1e-5 leaves margin.
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)


def test_empty_input(embedder):
    assert embedder.encode([]).shape == (0, embedder.dim)


def test_related_texts_score_higher_than_unrelated(embedder):
    a, b, c = embedder.encode(TEXTS)
    assert a @ b > a @ c


def test_batching_does_not_change_vectors(embedder):
    together = embedder.encode(TEXTS, batch_size=3)
    alone = np.vstack([embedder.encode([t], batch_size=1) for t in TEXTS])
    # Padding changes low-order bits of batched kernels on CPU; observed ~1e-7.
    np.testing.assert_allclose(together, alone, atol=1e-5)


@pytest.mark.gpu
def test_accelerator_matches_cpu(embedder):
    import torch

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        pytest.skip("no CUDA or MPS device")
    from litkit.embeddings.sbert_mpnet import ChunkEmbedderSBERT
    from litkit.embeddings.specter2 import PaperEmbedderSpecter2

    other = (
        PaperEmbedderSpecter2(device=device)
        if isinstance(embedder, PaperEmbedderSpecter2)
        else ChunkEmbedderSBERT(devices=[device])
    )
    cpu = embedder.encode(TEXTS)
    acc = other.encode(TEXTS)
    # Different kernels and reduction order; cosine between the two stays near 1.
    np.testing.assert_allclose(np.sum(cpu * acc, axis=1), 1.0, atol=1e-3)
