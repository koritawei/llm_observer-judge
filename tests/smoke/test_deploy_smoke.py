"""Deploy smoke — start a real uvicorn subprocess and check /health, the main routes, and a clean SIGTERM exit.

Interpreted-language risk: passing under the dev-time TestClient does not mean uvicorn can start at deploy time.
This case uses the same production path: ``uvicorn benchmark.service.main:create_app --factory``.
"""
# ruff: noqa: PLR2004

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[2]
BOOT_TIMEOUT_S = 15.0
SHUTDOWN_TIMEOUT_S = 5.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(base_url: str, proc: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=2.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError("uvicorn exited before /health became ready")
            try:
                resp = client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
    raise AssertionError(f"/health did not become ready within {timeout_s}s")


def _terminate_cleanly(proc: subprocess.Popen[bytes], log_path: Path) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    proc.kill()
    proc.wait(timeout=5)
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    raise AssertionError(
        f"service did not exit within {SHUTDOWN_TIMEOUT_S}s of SIGTERM\n--- log tail ---\n{tail}"
    )


@pytest.mark.smoke
def test_deploy_smoke(tmp_path: Path) -> None:
    """Start a real uvicorn → /health → unknown model returns 400 → the example route works → SIGTERM exits cleanly."""

    host = "127.0.0.1"
    port = _free_port()
    base_url = f"http://{host}:{port}"
    log_path = tmp_path / "uvicorn.log"

    env = os.environ.copy()
    env.setdefault("AGENT_EVAL_MODEL_API_KEY", "test-placeholder-not-used")
    env["AGENT_EVAL_DATABASE_URL"] = f"sqlite:///{tmp_path}/smoke.db"
    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(proxy_var, None)

    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "benchmark.service.main:create_app",
                "--factory",
                "--host",
                host,
                "--port",
                str(port),
                "--log-level",
                "info",
            ],
            cwd=SERVICE_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    try:
        _wait_until_ready(base_url, proc, BOOT_TIMEOUT_S)

        with httpx.Client(timeout=10.0) as client:
            health = client.get(f"{base_url}/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok", "detail": "success"}

            unknown = client.post(
                f"{base_url}/v1/chat/completions",
                json={"model": "ghost", "messages": [{"role": "user", "content": "x"}]},
            )
            assert unknown.status_code == 400

            example_payload = {
                "model": "example",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            '{"input":{"request_body":{"query":"x"}},'
                            '"output":{"component":{"text":"y"}},'
                            '"metadata":{"trace_id":"smoke-1"}}'
                        ),
                    }
                ],
            }
            example = client.post(f"{base_url}/v1/chat/completions", json=example_payload)
            assert 100 <= example.status_code < 600
            assert proc.poll() is None, "uvicorn crashed during example model request"
    finally:
        _terminate_cleanly(proc, log_path)
