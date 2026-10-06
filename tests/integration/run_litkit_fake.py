"""Run the litkit CLI with the fake embedders from conftest.py.

Usage: python run_litkit_fake.py [litkit arguments...]

Patches litkit.embeddings.factory before cli.main() loads it, so builds and
queries run end to end without model files, torch or a GPU.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import litkit.embeddings.factory as factory  # noqa: E402
from conftest import FakeEmbedder  # noqa: E402

DIM = int(os.environ.get("FAKE_EMBED_DIM", "32"))


def _make_paper_embedder():
    return FakeEmbedder(DIM), {"type": "paper", "impl": "fake", "dim": DIM}


def _make_chunk_embedder(devices="auto", *, workers=1, force_devices=False):
    return FakeEmbedder(DIM), {"type": "chunk", "impl": "fake", "dim": DIM}


factory.make_paper_embedder = _make_paper_embedder
factory.make_chunk_embedder = _make_chunk_embedder

from litkit.cli import main  # noqa: E402

sys.argv = ["litkit", *sys.argv[1:]]
sys.exit(main())
