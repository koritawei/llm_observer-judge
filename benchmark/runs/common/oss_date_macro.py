"""Date macros in an OSS path prefix: ``${yyyymmdd}`` / ``${yyyymmdd+n}`` / ``${yyyymmdd-n}``.

A path prefix on an OSS-import dataset may carry a date macro. When a batch task is
registered (immediate / scheduled), the accept side expands it to ``yyyymmdd`` using this
run's data day: the macro is the date partition. After expansion the data file lives at
``{expanded prefix}/benchmark_final_results.csv``; no extra evaluation-date partition is
appended. An illegal macro is rejected both when the dataset is stored (400) and when the
trigger is accepted (422). It must not be passed through to OSS and become ``file not found``.

A literal path whose last segment is already a valid ``yyyymmdd`` is also a date-partition
directory (including an offset day from a pre-expanded ``${yyyymmdd-1}``). The caller should
pin the object key and must not append the evaluation day again.
"""

import re
from datetime import date, datetime, timedelta

#: Valid macros: ``${yyyymmdd}`` / ``${yyyymmdd+3}`` / ``${yyyymmdd-1}``
DATE_MACRO = re.compile(r"\$\{yyyymmdd(?:([+-])(\d+))?\}")
#: Any ``${...}`` shape (used to recognize illegal macros)
_BRACED_MACRO = re.compile(r"\$\{[^}]*\}")
_OPEN = "${"
_DATE_FORMAT = "%Y%m%d"
_YYYYMMDD_LEN = len("YYYYMMDD")
_MACRO_HELP = "${yyyymmdd} / ${yyyymmdd+n} / ${yyyymmdd-n}"
#: Fixed daily wide-table file name (aligned with DailyDataLoader.FILE_NAME)
DEFAULT_BENCHMARK_CSV = "benchmark_final_results.csv"


def has_date_macro(text: str | None) -> bool:
    """Whether the text contains at least one valid date macro."""
    return bool(DATE_MACRO.search(text or ""))


def find_invalid_date_macro(text: str | None) -> str | None:
    """Return the first illegal macro text (including an unclosed ``${``), or ``None`` when every macro is valid."""
    raw = text or ""
    if _OPEN not in raw:
        return None
    for match in _BRACED_MACRO.finditer(raw):
        if not DATE_MACRO.fullmatch(match.group(0)):
            return match.group(0)
    if raw.count(_OPEN) > len(_BRACED_MACRO.findall(raw)):
        return raw[raw.rfind(_OPEN) :]
    return None


def validate_date_macros(text: str | None) -> None:
    """Validate macro syntax inside a prefix.

    Raises:
        ValueError: an illegal macro is present (dataset-doc validation maps to 400, trigger accept maps to 422).
    """
    invalid = find_invalid_date_macro(text)
    if invalid is not None:
        msg = f"oss_prefix 含非法日期宏 {invalid!r}；仅支持 {_MACRO_HELP}"
        raise ValueError(msg)


def expand_date_macros(text: str | None, *, base_date: str | date) -> str:
    """Replace date macros in the prefix with ``yyyymmdd`` offset from ``base_date`` (the anchor day).

    ``${yyyymmdd}`` = the anchor day, ``${yyyymmdd+n}`` = n days later,
    ``${yyyymmdd-n}`` = n days earlier. With no macro, return the text unchanged (stripped).

    Raises:
        ValueError: a macro is illegal, or ``base_date`` is not ``yyyymmdd``.
    """
    raw = (text or "").strip()
    if not raw or _OPEN not in raw:
        return raw
    validate_date_macros(raw)
    base = _coerce_base_date(base_date)

    def _replace(match: re.Match[str]) -> str:
        sign, offset = match.group(1), match.group(2)
        days = int(offset or 0)
        if sign == "-":
            days = -days
        return (base + timedelta(days=days)).strftime(_DATE_FORMAT)

    return DATE_MACRO.sub(_replace, raw)


def strip_oss_scheme(oss_prefix: str) -> str:
    """Extract the object key path from ``oss://bucket/path``; with no scheme, return the text unchanged."""
    prefix = (oss_prefix or "").strip()
    if prefix.startswith("oss://"):
        without_scheme = prefix[6:]
        slash_idx = without_scheme.find("/")
        if slash_idx != -1:
            return without_scheme[slash_idx + 1 :]
        return ""
    return prefix


def trailing_yyyymmdd(path: str | None) -> str | None:
    """Return the date when the last path segment is a valid ``yyyymmdd``, otherwise ``None``.

    It does not have to equal this run's data_date: an offset day from a pre-expanded
    ``${yyyymmdd-1}`` is also a date-partition directory, and appending the evaluation day would double the path.
    """
    text = strip_oss_scheme(path or "").rstrip("/")
    if not text:
        return None
    tail = text.rsplit("/", 1)[-1]
    return _parse_yyyymmdd(tail)


def partition_date_from_path(path: str | None) -> str | None:
    """Extract the partition day ``yyyymmdd`` from a full object path.

    The last segment may be the date, or a filename whose parent is the partition directory, for example::

        ieval-service/benchmark/benchmark_result/20261111
        ieval-service/benchmark/benchmark_result/20261111/benchmark_final_results.csv
        oss://bucket/ieval-service/benchmark/benchmark_result/20261110/xxx.csv
    """
    text = strip_oss_scheme(path or "").rstrip("/")
    if not text:
        return None
    direct = trailing_yyyymmdd(text)
    if direct is not None:
        return direct
    if "/" not in text:
        return None
    parent = text.rsplit("/", 1)[0]
    return trailing_yyyymmdd(parent)


def _parse_yyyymmdd(raw: str) -> str | None:
    if len(raw) != _YYYYMMDD_LEN or not raw.isdigit():
        return None
    try:
        datetime.strptime(raw, _DATE_FORMAT)
    except ValueError:
        return None
    return raw


def object_key_for_dated_prefix(
    prefix: str | None,
    *,
    file_name: str = DEFAULT_BENCHMARK_CSV,
) -> str | None:
    """When the last segment is already ``yyyymmdd``, return ``{prefix}/{file_name}``; otherwise ``None``."""
    key_path = strip_oss_scheme(prefix or "").rstrip("/")
    if trailing_yyyymmdd(key_path) is None:
        return None
    name = (file_name or DEFAULT_BENCHMARK_CSV).strip().replace("\\", "/").split("/")[-1]
    if not name or name in {".", ".."}:
        name = DEFAULT_BENCHMARK_CSV
    return f"{key_path}/{name}"


def prepare_oss_prefix_for_load(
    prefix: str | None,
    *,
    base_date: str,
    file_name: str = DEFAULT_BENCHMARK_CSV,
) -> tuple[str, str | None]:
    """Expand date macros and pin the object key when the path is already a date partition.

    Returns:
        ``(effective_prefix, object_key | None)``. When ``object_key`` is set, the loader
        must use it verbatim and must not append ``base_date``.

    Raises:
        ValueError: a macro is illegal, or ``base_date`` is illegal.
    """
    raw = (prefix or "").strip()
    if has_date_macro(raw) or find_invalid_date_macro(raw) is not None:
        validate_date_macros(raw)
        effective = expand_date_macros(raw, base_date=base_date)
        return effective, object_key_for_dated_prefix(effective, file_name=file_name)
    key = object_key_for_dated_prefix(raw, file_name=file_name)
    return raw, key


def _coerce_base_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip(), _DATE_FORMAT).date()
    except ValueError as exc:
        msg = f"日期宏基准日非法（需 yyyymmdd）：{value!r}"
        raise ValueError(msg) from exc


__all__ = [
    "DATE_MACRO",
    "DEFAULT_BENCHMARK_CSV",
    "expand_date_macros",
    "find_invalid_date_macro",
    "has_date_macro",
    "object_key_for_dated_prefix",
    "partition_date_from_path",
    "prepare_oss_prefix_for_load",
    "strip_oss_scheme",
    "trailing_yyyymmdd",
    "validate_date_macros",
]
