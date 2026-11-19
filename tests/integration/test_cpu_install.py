"""Build and reopen a CPU index with the installed CLI and offline transformer weights."""

import os
import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from conftest import jats, write_tar


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_laptop_wheel_requires_torch_for_specter2_weights(platform):
    requirements = map(Requirement, requires("litkit"))
    (torch_requirement,) = [
        req
        for req in requirements
        if req.name == "torch"
        and (req.marker is None or req.marker.evaluate({"sys_platform": platform}))
    ]
    assert "2.5.1" not in torch_requirement.specifier


@pytest.mark.slow
def test_installed_cpu_build_and_search(tmp_path):
    if sys.platform.startswith("linux"):
        pytest.importorskip("torch")  # The ARM64 container supplies its own torch.
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    from litkit.embeddings.sbert_mpnet import SBERT_ID
    from litkit.embeddings.specter2 import SPECTER2_ID

    if os.name == "nt":
        assert sys.version_info[:2] == (3, 12)
        assert torch.version.cuda is None, "the Windows install must use the CPU wheel"

    # Both production loaders use local Hugging Face snapshots. Tiny weights
    # exercise their CPU kernels, NumPy conversions and FAISS without a download.
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "lunar", "rock", "basalt"]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    torch.manual_seed(0)
    model = BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    for model_id in (SPECTER2_ID, SBERT_ID):
        snapshot = (
            tmp_path
            / "hf"
            / "hub"
            / f"models--{model_id.replace('/', '--')}"
            / "snapshots"
            / "tiny"
        )
        # Distributed SPECTER2 weights use pytorch_model.bin. MPNet has safetensors.
        # The Linux container supplies its own converted SPECTER2 safetensors.
        use_bin_weights = model_id == SPECTER2_ID and sys.platform in ("win32", "darwin")
        model.save_pretrained(snapshot, safe_serialization=not use_bin_weights)
        tokenizer.save_pretrained(snapshot)

    corpus = tmp_path / "input articles"
    corpus.mkdir()
    passage = "The lunar rock sample contained basalt with a distinctive β-phase beneath the surface."
    article = jats(
        pmcid="PMC101", title="Lunar rock", abstract="A basalt sample.", paragraphs=[passage]
    )
    write_tar(corpus / "sample.tar", {"sample.nxml": article})

    workspace = tmp_path / "文献 index with spaces"
    env = dict(
        os.environ,
        LITKIT_WORKSPACE=str(workspace),
        HF_HOME=str(tmp_path / "hf"),
        LITKIT_FORCE_DEVICE="cpu",
    )
    if os.name == "nt":
        # Exercise ordinary Windows pipes even if the runner enabled Python's UTF-8 mode.
        env.pop("PYTHONIOENCODING", None)
        env["PYTHONUTF8"] = "0"
    cli = Path(sys.executable).with_name("litkit.exe" if os.name == "nt" else "litkit")

    def run(*args):
        result = subprocess.run(
            [cli, *args], cwd=tmp_path, env=env, capture_output=True, timeout=120
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="backslashreplace")
        return result.stdout.decode("utf-8")

    run("--build-only", "--faiss-writer", "--tar-dir", str(corpus), "--chunks-index", "flat")
    output = run("--no-llm", "lunar rock basalt")
    assert "PMC101" in output and passage in output
