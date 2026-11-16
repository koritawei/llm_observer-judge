"""Exercise the launcher with isolated runfiles and a stub server process."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("storage_available", [False, True])
def test_launcher_keeps_storage_optional(tmp_path: Path, storage_available: bool) -> None:
    workspace = tmp_path / "runfiles" / "_main"
    (workspace / "services" / "agent-eval").mkdir(parents=True)
    for dependency in ("agent_graph/src/agentgraph", "data_models/py/data_models", "message_queue"):
        (workspace / "libs" / dependency).mkdir(parents=True)
    storage_src = workspace / "libs" / "storage" / "src"
    if storage_available:
        (storage_src / "storage").mkdir(parents=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_stub = bin_dir / "python3"
    python_stub.write_text('#!/bin/sh\nprintf "%s\\n" "$PYTHONPATH"\nprintf "%s\\n" "$@"\n')
    python_stub.chmod(0o755)
    launcher = Path(__file__).resolve().parents[2] / "bazel_docker_server.sh"
    result = subprocess.run(
        ["/bin/bash", str(launcher)],
        env={
            "PATH": str(bin_dir) + os.pathsep + os.defpath,
            "HOME": str(tmp_path / "home"),
            "RUNFILES_DIR": str(workspace.parent),
            "BAZEL_WORKSPACE_HOME": str(workspace),
        },
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    python_path, *server_args = result.stdout.splitlines()
    assert server_args[:4] == ["-u", "-m", "uvicorn", "benchmark.service.main:create_app"]
    assert python_path.split(os.pathsep).count(str(storage_src)) == int(storage_available)
    assert ("WARN: libs/storage" in result.stderr) is not storage_available
