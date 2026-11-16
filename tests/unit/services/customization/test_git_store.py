"""Git content repository: redaction / credential guard / non-fast-forward rebase retry (real git)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from benchmark.domain.customization_config import CustomizationConfig
from benchmark.customization.errors import CustomizationError
from benchmark.customization.git_store import GitContentStore


def _config(**overrides: object) -> CustomizationConfig:
    values: dict[str, object] = {
        "enabled": True,
        "git_username": "bot",
        "token": "secret-token",
    }
    values.update(overrides)
    return CustomizationConfig(**values)  # type: ignore[arg-type]


def test_sanitize_redacts_token_raw_and_quoted() -> None:
    store = GitContentStore(_config(token="tok en/x"))
    text = "remote https://bot:tok en/x@host/repo and tok%20en%2Fx"
    cleaned = store._sanitize(text)
    assert "tok en/x" not in cleaned
    assert "tok%20en%2Fx" not in cleaned
    assert "***" in cleaned


async def test_commit_requires_credentials() -> None:
    store = GitContentStore(CustomizationConfig(enabled=True))
    with pytest.raises(CustomizationError) as exc:
        await store.commit_bundle(git_path="p", files={"a.txt": b"x"}, message="m")
    assert exc.value.status_code == 503
    assert exc.value.code == "customization_git_unavailable"


async def test_commit_bundle_runs_sync_commit_via_explicit_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """commit goes through the explicit git thread pool (DEFAULT-EXECUTOR-001), and the sync implementation is scheduled only once."""
    store = GitContentStore(_config())
    calls: list[str] = []

    def _fake_commit(spec: object) -> str:
        calls.append(spec.git_path)  # type: ignore[attr-defined]
        return "deadbeef"

    monkeypatch.setattr(store, "_commit_sync", _fake_commit)
    sha = await store.commit_bundle(git_path="p", files={"a.txt": b"x"}, message="m")
    assert sha == "deadbeef"
    assert calls == ["p"]


async def test_read_files_falls_back_to_git_via_explicit_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the Files API fails → the git fallback read also goes through the explicit thread pool and returns the content unchanged."""
    store = GitContentStore(_config())

    async def _fail_files_api(commit: str, paths: list[str]) -> dict[str, bytes]:
        raise RuntimeError("files api down")

    monkeypatch.setattr(store, "_read_via_files_api", _fail_files_api)
    monkeypatch.setattr(store, "_read_via_git_sync", lambda commit, paths: {p: b"v" for p in paths})
    files = await store.read_files(git_commit="a" * 40, paths=["x/y.txt"])
    assert files == {"x/y.txt": b"v"}


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_push_with_retry_rebases_on_non_fast_forward(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", "-b", "master", str(remote)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(remote), str(seed)], check=True)
    _git(seed, "config", "user.email", "a@test")
    _git(seed, "config", "user.name", "a")
    (seed / "a.txt").write_text("a")
    _git(seed, "add", ".")
    _git(seed, "commit", "-q", "-m", "seed")
    _git(seed, "push", "-q", "origin", "master")

    stale = tmp_path / "stale"
    subprocess.run(["git", "clone", "-q", str(remote), str(stale)], check=True)
    _git(stale, "config", "user.email", "b@test")
    _git(stale, "config", "user.name", "b")

    # The remote advances one version first, so a stale direct push is necessarily non-fast-forward
    (seed / "b.txt").write_text("b")
    _git(seed, "add", ".")
    _git(seed, "commit", "-q", "-m", "advance")
    _git(seed, "push", "-q", "origin", "master")

    (stale / "c.txt").write_text("c")
    _git(stale, "add", ".")
    _git(stale, "commit", "-q", "-m", "stale work")

    store = GitContentStore(_config())
    store._push_with_retry(str(stale), dict(os.environ))

    subprocess.run(["git", "-C", str(seed), "fetch", "-q", "origin"], check=True)
    names = subprocess.run(
        ["git", "-C", str(seed), "ls-tree", "--name-only", "origin/master"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "c.txt" in names
