"""shared.env — .env loading + aliases."""

from __future__ import annotations

import os
from pathlib import Path

from llmobserverjudge.shared.env import _apply_aliases, load_env


def test_apply_aliases_deepseek(monkeypatch) -> None:
    monkeypatch.delenv("LLMJUDGE_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("LLMJUDGE_JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("LLMJUDGE_JUDGE_MODEL_NAME", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    _apply_aliases()
    assert os.environ["LLMJUDGE_JUDGE_API_KEY"] == "sk-test"
    assert os.environ["LLMJUDGE_JUDGE_BASE_URL"] == "https://api.deepseek.com"
    assert os.environ["LLMJUDGE_JUDGE_MODEL_NAME"] == "deepseek-v4-flash"


def test_load_env_from_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLMJUDGE_JUDGE_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LLMJUDGE_JUDGE_API_KEY=from-file\n", encoding="utf-8")
    # Reset module latch for isolated load
    import llmobserverjudge.shared.env as env_mod

    monkeypatch.setattr(env_mod, "_LOADED", False)
    path = load_env(dotenv_path=env_file, override=True)
    assert path == env_file
    assert os.environ["LLMJUDGE_JUDGE_API_KEY"] == "from-file"
