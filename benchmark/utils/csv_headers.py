"""CSV header normalization — absorb BOM / leftover quotes so the first-column key does not drift.

Common shape from data pipelines / Excel export: ``utf-8-sig`` with every field quoted
(``\\xef\\xbb\\xbf\"query\",\"trace_id\",...``). With ``encoding=utf-8-sig``,
DictReader usually already yields a clean ``query``; if an upstream used ``utf-8`` by mistake or double-
encoded, the first column becomes ``\\ufeff\"query\"`` / ``\"query\"``, business interceptors look up
``query`` and get an empty string, and the judge reports ``query is required``.

This module only cleans key names and does not interpret business column semantics; it lives in ``benchmark.utils`` so
evaluator / benchmark can import it at module level without initializing the ``benchmark.services`` package.
"""

from typing import Any

_MIN_QUOTED_HEADER_LEN = 2
_QUOTE_CHARS = frozenset({'"', "'"})


def normalize_csv_header(name: object) -> str:
    """Strip BOM, surrounding whitespace, and a paired wrapping quote to get a stable column name."""
    text = str(name) if name is not None else ""
    text = text.lstrip("\ufeff").strip()
    if len(text) >= _MIN_QUOTED_HEADER_LEN and text[0] == text[-1] and text[0] in _QUOTE_CHARS:
        text = text[1:-1].lstrip("\ufeff").strip()
    return text


def normalize_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    """Rewrite every key via :func:`normalize_csv_header`; later duplicates overwrite."""
    return {normalize_csv_header(key): value for key, value in row.items()}


__all__ = ["normalize_csv_header", "normalize_csv_row"]
