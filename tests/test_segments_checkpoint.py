from pathlib import Path

from litkit.segments.checkpoint import (
    clear_shard_checkpoints,
    load_checkpoint,
    save_checkpoint,
    shard_ckpt_path,
)


def test_shard_ckpt_path():
    p = shard_ckpt_path(Path("/tmp/db"), 5)
    assert p.name == "build_checkpoint_shard_05.json"


def test_load_missing(tmp_path):
    ckpt = load_checkpoint(tmp_path / "nope.json")
    assert ckpt == {}


def test_save_and_load(tmp_path):
    ckpt_path = tmp_path / "ckpt.json"
    lock_path = tmp_path / "lock"
    data = {"done": 3, "files": ["a", "b"]}
    save_checkpoint(data, ckpt_path, lock_path)
    loaded = load_checkpoint(ckpt_path)
    assert loaded == data


def test_per_shard_save_load(tmp_path):
    base = tmp_path / "db"
    base.mkdir()
    ckpt_path = base / "build_checkpoint.json"
    lock_path = base / "lock"
    data = {"shard": 2}
    save_checkpoint(data, ckpt_path, lock_path, shard_id=2)
    loaded = load_checkpoint(ckpt_path, shard_id=2)
    assert loaded == data


def test_clear_shard_checkpoints(tmp_path):
    for i in range(3):
        p = shard_ckpt_path(tmp_path, i)
        p.write_text("{}")
    removed = clear_shard_checkpoints(tmp_path, num_shards=3)
    assert removed >= 3
    for i in range(3):
        assert not shard_ckpt_path(tmp_path, i).exists()
