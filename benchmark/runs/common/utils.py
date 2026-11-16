"""Shared helpers — dict/object compatibility and run_id date extraction.

Duplicated helpers extracted from aone_eval_service / local_eval_service,
reused by every benchmark service implementation.
"""

from typing import Any

#: Character length of the date segment in a run_id (``YYYYMMDD``).
_DATE_PART_LEN = 8


def to_dict(obj: object) -> dict[str, Any]:
    """Convert an object to a dict (compatible with dicts and instance attributes).

    ``isinstance(obj, dict)`` → return it unchanged;
    an instance whose ``__dict__`` is a dict → use that;
    anything else (a class ``__dict__`` is a mappingproxy / a ``__slots__`` instance /
    a scalar) → empty dict.

    One ``getattr`` plus ``isinstance`` is enough: do not probe with ``hasattr`` and then
    read again, and do not let a mappingproxy or other non-dict leak into the return value.
    """
    if isinstance(obj, dict):
        return obj
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return d
    return {}


def get_attr(obj: object, key: str, default: object = None) -> object:
    """Read a value from either a dict or an object attribute.

    dict → ``obj.get(key, default)``;
    any other object → ``getattr(obj, key, default)``.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_ds(run_id: str) -> str:
    """Extract the ODPS partition day ``YYYYMMDD`` from a run_id.

    A same-day sequence suffix is supported::

        ieval_external-20260812        → 20260812
        ieval_external-20260812-02     → 20260812   # do not rsplit and take the last segment
        ext-bench-20260812-01          → 20260812

    The old ``rsplit("-", 1)`` parsed ``...-20260812-02`` as ``02`` and fell back to
    "today", so ieval enrich queried the wrong partition and every response was empty.

    :raises ValueError: ``run_id`` has no 8-digit date segment. Silently falling back to
        "today" would query the wrong ODPS partition and return the wrong data (when
        production config is wrong, responses are all empty and hard to notice). Fail
        explicitly so the caller can see it.
    """
    for part in reversed(run_id.split("-")):
        if len(part) == _DATE_PART_LEN and part.isdigit():
            return part
    msg = f"run_id 不含 YYYYMMDD 日期段，无法确定 ODPS 分区: {run_id!r}"
    raise ValueError(msg)


__all__ = ["to_dict", "get_attr", "extract_ds"]
