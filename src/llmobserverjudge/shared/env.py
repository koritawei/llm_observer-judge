"""Load repository ``.env`` into ``os.environ`` (secrets stay out of git)."""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def find_dotenv(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (or cwd) looking for a ``.env`` file."""
    seen: set[Path] = set()
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.append(Path.cwd().resolve())
    # src/llmobserverjudge/shared/env.py → repo root
    candidates.append(Path(__file__).resolve().parents[3])

    for base in candidates:
        cur = base
        while cur not in seen:
            seen.add(cur)
            path = cur / ".env"
            if path.is_file():
                return path
            if cur.parent == cur:
                break
            cur = cur.parent
    return None


def load_env(*, dotenv_path: Path | str | None = None, override: bool = False) -> Path | None:
    """Load ``.env`` once. Existing process env wins unless ``override=True``."""
    global _LOADED
    if _LOADED and dotenv_path is None:
        _apply_aliases()
        return None

    if dotenv_path is not None:
        path = Path(dotenv_path)
        if not path.is_file():
            _apply_aliases()
            return None
    else:
        path = find_dotenv()
        if path is None:
            _apply_aliases()
            return None

    try:
        from dotenv import load_dotenv
    except ImportError:
        # Fail open: allow explicit OS env without python-dotenv installed
        _apply_aliases()
        return None

    load_dotenv(path, override=override)
    _LOADED = True
    _apply_aliases()
    return path


def _apply_aliases() -> None:
    """Map convenience aliases → canonical LLMJUDGE_* vars (setdefault)."""
    aliases = (
        ("DEEPSEEK_API_KEY", "LLMJUDGE_JUDGE_API_KEY"),
        ("DEEPSEEK_BASE_URL", "LLMJUDGE_JUDGE_BASE_URL"),
        ("DEEPSEEK_MODEL", "LLMJUDGE_JUDGE_MODEL_NAME"),
        ("OPENAI_API_KEY", "LLMJUDGE_JUDGE_API_KEY"),
        ("OPENAI_BASE_URL", "LLMJUDGE_JUDGE_BASE_URL"),
    )
    for src, dst in aliases:
        val = (os.environ.get(src) or "").strip()
        if val and not (os.environ.get(dst) or "").strip():
            os.environ[dst] = val
