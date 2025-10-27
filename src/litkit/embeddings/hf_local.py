# /src/litkit/embeddeings/hf_local.py

"""
Utilities for loading Hugging Face models from local snapshots only.

This module keeps LitKit fully usable on air-gapped machines by:
  • Resolving a repo's snapshot dir under HF's on-disk layout
    (<HF_HOME>/hub/models--ORG--REPO/snapshots/<commit>/).
  • Respecting HF_HOME from the environment (preferred), otherwise
    falling back to a module-local ./hf_cache path.
  • Providing a robust loader that first tries AutoModel/AutoTokenizer
    with local_files_only=True, then falls back to a concrete arch
    (BERT / RoBERTa / MPNet) based on files present in the snapshot.
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_offline_env(hf_home: str | Path | None = None) -> Path:
    """Set sensible HF offline defaults if the caller hasn't already.
    Returns the effective HF_HOME as a Path.
    """
    if hf_home is not None:
        os.environ.setdefault("HF_HOME", str(hf_home))
    effective = Path(os.getenv("HF_HOME", str(Path(__file__).parent / "hf_cache")))
    os.environ.setdefault("HF_HOME", str(effective))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return effective


def repo_root(repo_id: str, hf_home: str | Path | None = None) -> Path:
    """Return the local HF repo root directory for `repo_id`.

    Looks under: <HF_HOME>/hub/models--ORG--REPO/
    Falls back to ./hf_cache/hub/... next to this module.
    """
    sub = f"models--{repo_id.replace('/', '--')}"
    base = Path(os.getenv("HF_HOME", str(hf_home or "")) or "")
    if not base:
        base = Path(__file__).parent / "hf_cache"
    primary = base / "hub" / sub
    if (primary / "snapshots").exists() or (primary / "refs").exists():
        return primary
    return Path(__file__).parent / "hf_cache" / "hub" / sub


def local_snapshot_dir(
    repo_id: str, branch: str = "main", hf_home: str | Path | None = None
) -> Path:
    """Resolve a usable local snapshot directory for `repo_id`.

    Priority:
      1) refs/<branch> → commit → snapshots/<commit>
      2) If exactly one dir exists under snapshots/, use it
      3) Else raise FileNotFoundError with a clear message
    """
    root = repo_root(repo_id, hf_home=hf_home)
    snaps = root / "snapshots"
    ref = root / "refs" / branch
    if ref.exists():
        commit = ref.read_text().strip()
        p = snaps / commit
        if p.exists():
            return p
        raise FileNotFoundError(f"refs/{branch} → {commit} but snapshot missing under {snaps}")
    if snaps.exists():
        candidates = [d for d in snaps.iterdir() if d.is_dir()]
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            f"Multiple snapshots for {repo_id} but no refs/{branch}. "
            f"Create refs/{branch} or prune stale snapshots under {snaps}."
        )
    raise FileNotFoundError(f"No local snapshot found for {repo_id} under {root}")


def find_any(root: Path, names: list[str]) -> Path | None:
    """Return the first existing file among `names` under `root`, else None."""
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _detect_arch(local_path: Path) -> str | None:
    """Infer BERT / RoBERTa / MPNet by files present in `local_path`."""
    tok_json = local_path / "tokenizer.json"
    vocab_txt = local_path / "vocab.txt"
    merges_txt = local_path / "merges.txt"
    roberta_vocab = local_path / "vocab.json"
    spiece = find_any(local_path, ["spiece.model", "sentencepiece.bpe.model"])

    if tok_json.exists() and roberta_vocab.exists() and merges_txt.exists():
        return "roberta"
    if roberta_vocab.exists() and merges_txt.exists():
        return "roberta"
    if vocab_txt.exists():
        return "bert"
    if spiece is not None:
        return "mpnet"
    return None


def load_auto_or_fallback(local_path: Path, device: str = "cpu") -> tuple[object, object]:
    """Load (tokenizer, model) from a local snapshot directory.

    Strategy:
      1) Try AutoTokenizer/AutoModel with local_files_only=True
      2) Fallback to explicit arch based on snapshot files

    Returns:
      (tokenizer, model) — model already moved to `device` and in eval() mode.

    Raises:
      FileNotFoundError with a helpful message if no viable local snapshot is found.
    """
    # Lazy imports to keep import-time light
    from transformers import (
        AutoModel,
        AutoTokenizer,
        BertConfig,
        BertModel,
        BertTokenizerFast,
        MPNetConfig,
        MPNetModel,
        MPNetTokenizerFast,
        RobertaConfig,
        RobertaModel,
        RobertaTokenizerFast,
    )

    # 1) Auto* locally
    try:
        tok = AutoTokenizer.from_pretrained(
            str(local_path), local_files_only=True, trust_remote_code=False
        )
        model = AutoModel.from_pretrained(
            str(local_path), local_files_only=True, trust_remote_code=False
        ).to(device)
        model.eval()
        return tok, model
    except Exception:
        pass

    # 2) Detect arch and construct explicitly
    arch = _detect_arch(local_path)

    if arch == "bert":
        cfg = BertConfig.from_pretrained(str(local_path), local_files_only=True)
        tok_json = local_path / "tokenizer.json"
        if tok_json.exists():
            tok = BertTokenizerFast(tokenizer_file=str(tok_json))
        else:
            tok = BertTokenizerFast(vocab_file=str(local_path / "vocab.txt"))
        model = BertModel.from_pretrained(str(local_path), config=cfg, local_files_only=True).to(
            device
        )
        model.eval()
        return tok, model

    if arch == "roberta":
        cfg = RobertaConfig.from_pretrained(str(local_path), local_files_only=True)
        tok_json = local_path / "tokenizer.json"
        if tok_json.exists():
            tok = RobertaTokenizerFast(tokenizer_file=str(tok_json))
        else:
            tok = RobertaTokenizerFast(
                vocab_file=str(local_path / "vocab.json"),
                merges_file=str(local_path / "merges.txt"),
            )
        model = RobertaModel.from_pretrained(str(local_path), config=cfg, local_files_only=True).to(
            device
        )
        model.eval()
        return tok, model

    if arch == "mpnet":
        cfg = MPNetConfig.from_pretrained(str(local_path), local_files_only=True)
        tok = MPNetTokenizerFast.from_pretrained(str(local_path), local_files_only=True)
        model = MPNetModel.from_pretrained(str(local_path), config=cfg, local_files_only=True).to(
            device
        )
        model.eval()
        return tok, model

    raise FileNotFoundError(
        "[offline] Could not load model from local snapshot.\n"
        f"Checked: {local_path}\n"
        "Expected one of: tokenizer.json, vocab.txt (BERT), "
        "vocab.json + merges.txt (RoBERTa), or a sentencepiece model (MPNet)."
    )


__all__ = [
    "ensure_offline_env",
    "repo_root",
    "local_snapshot_dir",
    "find_any",
    "load_auto_or_fallback",
]
