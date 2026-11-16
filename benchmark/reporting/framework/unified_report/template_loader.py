"""Load the unified daily-report HTML template.

A business may name a file under ``templates/`` in the schedule config. When it is
unspecified, illegal, or missing, the default ``benchmark_report.html`` is used.
"""

import asyncio
from functools import lru_cache
from pathlib import Path

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger("agent_eval.unified_report.template_loader")

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_DEFAULT_TEMPLATE_NAME = "benchmark_report.html"
_TEMPLATE_CACHE_SIZE = 8


class UnifiedTemplateError(Exception):
    """The static unified template is unavailable (missing / unreadable) — report generation should fall back to the webdev skill."""


def resolve_report_template_name(template_name: str | None) -> str:
    """Reduce a business-configured template name to a filename under ``templates/``.

    Blank names, path separators, non-``.html`` names, missing files, or a failed stat
    all fall back to the default template.
    """
    text = template_name.strip() if isinstance(template_name, str) else ""
    if not text:
        return _DEFAULT_TEMPLATE_NAME
    if (
        text != Path(text).name
        or not text.endswith(".html")
        or text.startswith(".")
        or ".." in text
    ):
        logger.warning(
            "invalid report_html %r, fallback to %s",
            template_name,
            _DEFAULT_TEMPLATE_NAME,
        )
        return _DEFAULT_TEMPLATE_NAME
    if not _template_file_exists(text):
        return _DEFAULT_TEMPLATE_NAME
    return text


def _template_file_exists(name: str) -> bool:
    """Whether the template file is readable.

    ``Path.is_file`` only swallows absence such as ENOENT / ENOTDIR. Other ``OSError``
    values such as NFS ESTALE, EIO, and EACCES are also treated as unavailable and fall
    back to the default template, so they do not bubble up and bypass the contract
    "fall back when the specified file is unavailable".
    """
    try:
        exists = (_TEMPLATES_DIR / name).is_file()
    except OSError:
        logger.warning(
            "report template stat failed %s, fallback to %s",
            name,
            _DEFAULT_TEMPLATE_NAME,
        )
        return False
    if exists:
        return True
    logger.warning(
        "report template missing %s, fallback to %s",
        name,
        _DEFAULT_TEMPLATE_NAME,
    )
    return False


@lru_cache(maxsize=_TEMPLATE_CACHE_SIZE)
def _read_template_sync(template_name: str) -> str:
    path = _TEMPLATES_DIR / template_name
    if not path.is_file():
        raise UnifiedTemplateError(f"unified report template missing: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UnifiedTemplateError(f"unified report template unreadable: {path}") from exc
    if not text.strip():
        raise UnifiedTemplateError(f"unified report template empty: {path}")
    if 'id="report-data"' not in text or 'id="cases-data"' not in text:
        raise UnifiedTemplateError(f"unified report template missing required slots: {path}")
    return text


def load_unified_template_sync() -> str:
    """Load the unified template synchronously for the synchronous Cluster Report render path."""

    return _read_template_sync(_DEFAULT_TEMPLATE_NAME)


async def load_unified_template(template_name: str | None = None) -> str:
    """Load the static template asynchronously (in parallel with payload assembly).

    An empty ``template_name`` uses the default template. When the specified file is
    unavailable, fall back to the default. :class:`UnifiedTemplateError` is raised only
    when the default template itself is unavailable.

    Raises:
        UnifiedTemplateError: the default template is missing, unreadable, empty, or
        missing a required slot.
    """
    # Resolution includes one stat; run it in a thread so it does not block the event loop.
    resolved = await asyncio.to_thread(resolve_report_template_name, template_name)
    try:
        return await _load_named(resolved)
    except UnifiedTemplateError:
        if resolved == _DEFAULT_TEMPLATE_NAME:
            raise
        logger.warning(
            "report template %s unusable, fallback to %s",
            resolved,
            _DEFAULT_TEMPLATE_NAME,
        )
        return await _load_named(_DEFAULT_TEMPLATE_NAME)


async def _load_named(template_name: str) -> str:
    try:
        return await asyncio.to_thread(_read_template_sync, template_name)
    except UnifiedTemplateError:
        raise
    except Exception as exc:  # noqa: BLE001 — wrap uniformly as a recognizable load failure
        raise UnifiedTemplateError(f"unified report template load failed: {exc}") from exc


def clear_template_cache() -> None:
    """For tests: clear the template cache."""
    _read_template_sync.cache_clear()


__all__ = [
    "UnifiedTemplateError",
    "clear_template_cache",
    "load_unified_template",
    "resolve_report_template_name",
    "load_unified_template_sync",
]
