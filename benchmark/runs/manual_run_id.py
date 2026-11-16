"""run_id naming: two namespaces, scheduled and manual.

- Daily schedule (RemoteConfig external-daily / DB ``scheduled_task`` / WebUI schedule):
  ``{benchmark_name}-{yyyymmdd}`` (:func:`scheduled_run_id`); with an optional
  ``run_id_suffix`` it is ``{benchmark}-{date}-{suffix}``
- Manual trigger ``POST /v1/benchmarks/manual-runs``:
  - immediate task: ``{benchmark_name}-{yyyymmdd}-m{n}``
  - a periodic task triggered again: ``{benchmark_name}-{yyyymmdd}-s{n}``
  - with an ``Idempotency-Key``: that prefix plus ``-{8-char digest}``

``scheduled_task_run_id`` (``-t-{task_id}``) is kept only for compatibility; the WebUI schedule no longer uses it.
"""

import hashlib
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, TypeVar

from benchmark.records.models import BenchmarkRun
from benchmark.runs.common.oss_date_macro import partition_date_from_path

_SH_TZ = timezone(timedelta(hours=8))
ManualRunNamespace = Literal["m", "s"]
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_KEY_DIGEST_LEN = 8
_DEFAULT_ATTEMPTS = 8
# DB-RDB-003: a single query must return fewer than 10000 rows.
_MANUAL_RUN_LIST_LIMIT = 9999

T = TypeVar("T")


class ManualRunIdTakenError(Exception):
    """The target run_id is taken; the caller should try the next sequence number."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(run_id)


class ManualRunIdExhaustedError(Exception):
    """Still no free run_id after several sequence collisions in a row."""

    def __init__(self, benchmark_name: str, date: str) -> None:
        self.benchmark_name = benchmark_name
        self.date = date
        super().__init__(f"{benchmark_name}-{date}")


def resolve_external_data_date(
    date: str | None,
    *,
    now: datetime | None = None,
    storage_path: str | None = None,
) -> str:
    """Data day for an external trigger: a literal partition day → explicit date → today in UTC+8.

    Only read a path segment that is **already** a literal ``yyyymmdd`` partition
    (``.../YYYYMMDD`` or ``.../YYYYMMDD/file``). A macro path containing ``${yyyymmdd±n}``
    is **not** expanded here — the macro is expanded once on the accept path
    ``_resolve_oss_target`` and once before a manual claim, so an already-shifted day is
    not expanded a second time. With no literal partition, fall back to the explicit ``date`` / today.
    """
    from_path = partition_date_from_path(storage_path)
    if from_path is not None:
        return from_path
    if date:
        return date
    clock = now if now is not None else datetime.now(_SH_TZ)
    return clock.astimezone(_SH_TZ).strftime("%Y%m%d")


def validate_idempotency_key(key: str | None) -> str | None:
    """Validate the idempotency key. ``None`` means this click does not require dedup."""
    if key is None:
        return None
    if _IDEMPOTENCY_KEY.fullmatch(key) is None:
        raise ValueError("Idempotency-Key must match ^[A-Za-z0-9._:-]{1,128}$")
    return key


def scheduled_run_id(benchmark_name: str, date: str) -> str:
    """Default daily-schedule run_id (no suffix), shared by RemoteConfig / scheduled_task / the WebUI schedule."""
    return f"{benchmark_name}-{date}"


def scheduled_task_run_id(benchmark_name: str, date: str, task_id: str) -> str:
    """Legacy WebUI ``-t-{task_id}`` shape (production now uses :func:`scheduled_run_id`).

    Form ``{benchmark}-{yyyymmdd}-t-{task_id}``. Do not call this for new triggers.
    """
    tid = task_id.strip()
    if not tid:
        raise ValueError("task_id required for scheduled_task_run_id")
    return f"{benchmark_name}-{date}-t-{tid}"


def format_manual_sequence_run_id(
    benchmark_name: str,
    date: str,
    seq: int,
    *,
    namespace: ManualRunNamespace = "m",
) -> str:
    """run_id for the seq-th manual trigger. seq starts at 1."""
    if seq < 1:
        raise ValueError("seq must be >= 1")
    return f"{benchmark_name}-{date}-{namespace}{seq}"


def manual_run_id_for_idempotency(
    benchmark_name: str,
    date: str,
    key: str,
    *,
    namespace: ManualRunNamespace = "m",
) -> str:
    """The same idempotency key always maps to a manual run_id in the same namespace."""
    # Keep the old contract for the m-digest input so an existing Idempotency-Key retry is not assigned a new run.
    raw_key = f"{benchmark_name}\n{date}\n{key}"
    if namespace == "s":
        raw_key = f"{benchmark_name}\n{date}\ns\n{key}"
    digest = hashlib.sha256(raw_key.encode()).hexdigest()[:_KEY_DIGEST_LEN]
    return f"{benchmark_name}-{date}-{namespace}-{digest}"


def manual_suffix(run_id: str, benchmark_name: str, date: str) -> str:
    """Take the suffix off an allocated manual run_id (``m1`` / ``m-deadbeef``)."""
    prefix = f"{benchmark_name}-{date}-"
    if not run_id.startswith(prefix) or run_id == scheduled_run_id(benchmark_name, date):
        raise ValueError(f"not a manual run_id: {run_id}")
    return run_id[len(prefix) :]


def next_manual_sequence(
    existing_run_ids: Iterable[str],
    benchmark_name: str,
    date: str,
    *,
    namespace: ManualRunNamespace = "m",
) -> int:
    """Next sequence number among those already taken in the namespace; digest suffixes do not consume a number."""
    prefix = f"{benchmark_name}-{date}-"
    max_seq = 0
    for run_id in existing_run_ids:
        if not run_id.startswith(prefix):
            continue
        matched = re.fullmatch(rf"{namespace}([1-9][0-9]*)", run_id[len(prefix) :])
        if matched is None:
            continue
        max_seq = max(max_seq, int(matched.group(1)))
    return max_seq + 1


async def list_manual_run_ids(
    benchmark_name: str,
    date: str,
    *,
    namespace: ManualRunNamespace = "m",
) -> list[str]:
    """Manual run_ids already taken in this namespace today."""
    prefix = f"{benchmark_name}-{date}-{namespace}"
    rows = await (
        BenchmarkRun.filter(run_id__startswith=prefix)
        .limit(_MANUAL_RUN_LIST_LIMIT)
        .values_list("run_id", flat=True)
    )
    return [str(row) for row in rows]


@dataclass(frozen=True)
class ManualRunClaimScope:
    """Identity scope of one manual run_id claim (the namespace triple).

    When ``idempotency_key`` is set, the run_id is derived deterministically (a retry of the
    same key lands on the same id). When it is absent, claim the next sequence number
    (the two paths inside claim_manual_run_id).
    """

    benchmark_name: str
    date: str
    idempotency_key: str | None = None
    namespace: ManualRunNamespace = "m"


def plan_manual_run_id(
    scope: ManualRunClaimScope,
    existing_run_ids: Iterable[str],
) -> str:
    """Plan a manual run_id that differs from the scheduled run_id."""
    if scope.idempotency_key:
        return manual_run_id_for_idempotency(
            scope.benchmark_name,
            scope.date,
            scope.idempotency_key,
            namespace=scope.namespace,
        )
    seq = next_manual_sequence(
        existing_run_ids,
        scope.benchmark_name,
        scope.date,
        namespace=scope.namespace,
    )
    return format_manual_sequence_run_id(
        scope.benchmark_name,
        scope.date,
        seq,
        namespace=scope.namespace,
    )


async def claim_manual_run_id(
    scope: ManualRunClaimScope,
    *,
    existing_run_ids: Callable[[], Awaitable[Iterable[str]]],
    claim: Callable[[str], Awaitable[T]],
    max_attempts: int = _DEFAULT_ATTEMPTS,
) -> T:
    """Claim a manual run_id.

    With an idempotency key, try once; a conflict is returned by ``claim`` as "already exists".
    Without one, ``claim`` switches to the next sequence number on ``ManualRunIdTakenError``.
    """
    if scope.idempotency_key:
        run_id = plan_manual_run_id(scope, ())
        return await claim(run_id)

    for _ in range(max_attempts):
        run_id = plan_manual_run_id(scope, await existing_run_ids())
        try:
            return await claim(run_id)
        except ManualRunIdTakenError:
            continue
    raise ManualRunIdExhaustedError(scope.benchmark_name, scope.date)
