"""Exercise writer ownership across independent processes using the same workspace."""

import os
import subprocess
import sys
import time

import pytest

from litkit.concurrent import FileLock

GUARD_OWNER = """
import sys
from pathlib import Path
from litkit.cli import _create_writer_guard_or_exit

_create_writer_guard_or_exit()
Path(sys.argv[1]).touch()
input()
"""


def wait_for_marker(marker, process):
    deadline = time.monotonic() + 30
    while not marker.exists():
        assert process.poll() is None, f"child exited early with {process.returncode}"
        if time.monotonic() >= deadline:
            pytest.fail(f"child did not create {marker.name}")
        time.sleep(0.02)


@pytest.mark.parametrize(
    "abrupt",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(os.name != "nt", reason="Windows delete-on-close guard")
        ),
    ],
    ids=["normal-exit", "terminated-owner"],
)
def test_second_writer_is_refused_and_can_restart(isolated_env, abrupt):
    workspace = isolated_env / "library with spaces"
    env = dict(os.environ, LITKIT_WORKSPACE=str(workspace))
    ready = isolated_env / "owner.ready"
    owner_cmd = [sys.executable, "-c", GUARD_OWNER, str(ready)]
    with subprocess.Popen(owner_cmd, env=env, stdin=subprocess.PIPE, text=True) as owner:
        try:
            wait_for_marker(ready, owner)
            for flags in (["--init-indices-only", "--faiss-writer"], ["--reconcile-only"]):
                contender = subprocess.run(
                    [sys.executable, "-m", "litkit", *flags],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert contender.returncode == 2, contender.stderr
                assert "Another FAISS writer" in contender.stderr
                assert owner.poll() is None
        finally:
            if abrupt:
                owner.kill()
                owner.wait(timeout=30)
            else:
                owner.communicate(input="\n", timeout=30)
    if not abrupt:
        assert owner.returncode == 0
    ready.unlink()
    subprocess.run(owner_cmd, env=env, input="\n", text=True, timeout=30, check=True)
    assert ready.exists()


def test_file_lock_serializes_processes(tmp_path):
    lock = tmp_path / "workspace with spaces" / "db.writer.lock"
    ready = tmp_path / "waiter.ready"
    script = """
from pathlib import Path
import sys
from litkit.concurrent import FileLock

input()
Path(sys.argv[2]).touch()
with FileLock(Path(sys.argv[1])):
    print("acquired", flush=True)
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(lock), str(ready)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as waiter:
        try:
            with FileLock(lock):
                waiter.stdin.write("\n")
                waiter.stdin.flush()
                wait_for_marker(ready, waiter)
                with pytest.raises(subprocess.TimeoutExpired):
                    waiter.wait(timeout=0.5)
            output, _ = waiter.communicate(timeout=30)
            assert waiter.returncode == 0 and output.strip() == "acquired"
        finally:
            if waiter.poll() is None:
                waiter.kill()
