# /src/embeddings/specter2.py

# Paper-level embedder using SPECTER2 (title+abstract → 768-dim)

from __future__ import annotations

import numpy as np
import torch

from .base import Embedder
from .base import inline_progress_renderer as _inline_progress_renderer
from .devices import detect_device
from .hf_local import load_auto_or_fallback, local_snapshot_dir

# Canonical HF id for offline snapshot layout
SPECTER2_ID = "allenai/specter2_base"


class PaperEmbedderSpecter2(Embedder):
    """
    Offline-friendly paper embedder that:
      • loads tokenizer+model from a local HF snapshot
      • runs on the resolved device (cpu/mps/cuda)
      • returns L2-normalized float32 vectors (N x D)
      • renders a single in-place progress line when asked

    Notes:
      - Context is truncated to 512 tokens (SPECTER2 default-safe).
      - Batch size heuristics: 16 on CUDA, 8 otherwise (override via encode()).
    """

    def __init__(
        self,
        model_id: str = SPECTER2_ID,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or detect_device()
        local_path = local_snapshot_dir(self.model_id)
        self.tok, self.model = load_auto_or_fallback(local_path, device=self.device)
        self.model.eval()

        # Default dimensionality; verify once to be safe
        self.dim = 768
        try:
            with torch.no_grad():
                toks = self.tok(["__probe__"], padding=True, truncation=True,
                                max_length=8, return_tensors="pt")
                toks = toks.to(next(self.model.parameters()).device)
                out = self.model(**toks)
                d = int(out.last_hidden_state.shape[-1])
                if d > 0:
                    self.dim = d
        except Exception:
            # fall back to 768 if probing fails; encode() will still work
            self.dim = 768

    def encode(
        self,
        texts: list[str],
        progress_label: str | None = None,
        batch_size: int | None = None,
        progress_done_summary: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into L2-normalized embeddings (float32, N x dim).
        If progress_label is provided, render a single compact line with updates,
        ending with a newline when finished.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        # Heuristic batch size unless caller specifies one
        try:
            model_device = next(self.model.parameters()).device
            on_cuda = str(model_device).startswith("cuda")
        except Exception:
            model_device, on_cuda = torch.device("cpu"), False

        bs = int(batch_size or (16 if on_cuda else 8))
        total = len(texts)

        render = (
            _inline_progress_renderer(
                progress_label or "Embedding papers", total, done_summary=progress_done_summary
            )
            if progress_label
            else None
        )
        if render:
            render(0)

        out_chunks = []
        with torch.no_grad():
            for i in range(0, total, bs):
                batch = texts[i : i + bs]
                toks = self.tok(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(model_device)
                out = self.model(**toks)
                # CLS/<s> token; normalize to unit length → cosine via IP
                cls = out.last_hidden_state[:, 0, :]
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                out_chunks.append(cls.detach().cpu().numpy().astype("float32"))
                if render:
                    render(min(total, i + len(batch)))

        if render:
            render(total, final=True)

        return np.vstack(out_chunks) if out_chunks else np.zeros((0, self.dim), dtype="float32")

    def __repr__(self) -> str:  # nice to have in logs
        return f"PaperEmbedderSpecter2(model_id={self.model_id!r}, device={self.device!r}, dim={self.dim})"
