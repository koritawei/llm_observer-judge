"""Unit tests for eval_runtime.run_lifecycle — the shared run-level lifecycle pieces.

The Tortoise model uses an in-memory Fake stand-in: ``filter()`` returns a preset queryset,
and ``update`` / async iteration / ``update_from_dict().save()`` are all readable for assertions.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from benchmark.lifecycle.run_lifecycle import (
    STALE_RECOVERY_REASON,
    FailRunSpec,
    RunLifecycleBinding,
    fail_eval_run,
    fail_inflight_runs,
    fail_run,
    heartbeat_run,
    recover_stale_runs,
)


class _FakeRun:
    """An in-memory run row — reproduces Tortoise's update_from_dict + save behavior."""

    def __init__(
        self,
        run_id: str,
        *,
        owner: str = "worker-1",
        status: str = "running",
        heartbeat_at: datetime | None = None,
    ) -> None:
        self.run_id = run_id
        self.owner = owner
        self.status = status
        self.heartbeat_at = heartbeat_at
        self.error_message: str | None = None
        self.finished_at: datetime | None = None
        self.saved = False

    def update_from_dict(self, data: dict[str, Any]) -> "_FakeRun":
        for key, value in data.items():
            setattr(self, key, value)
        return self

    async def save(self) -> None:
        self.saved = True


class _FakeQuerySet:
    """The ``filter()`` return value — supports ``await .update()`` and ``async for``."""

    def __init__(self, rows: list[_FakeRun], update_result: int = 0) -> None:
        self._rows = rows
        self._update_result = update_result
        self.update_kwargs: dict[str, Any] | None = None

    async def update(self, **kwargs: Any) -> int:
        self.update_kwargs = kwargs
        for row in self._rows:
            for key, value in kwargs.items():
                setattr(row, key, value)
        return self._update_result

    def __aiter__(self) -> Any:
        async def _gen() -> Any:
            for row in self._rows:
                yield row

        return _gen()


class _FakeModel:
    """Return the preset querysets in call order, and record the filter conditions."""

    def __init__(self, *querysets: _FakeQuerySet) -> None:
        self._queue = list(querysets)
        self.filter_calls: list[dict[str, Any]] = []

    def filter(self, **kwargs: Any) -> _FakeQuerySet:
        self.filter_calls.append(kwargs)
        return self._queue.pop(0) if self._queue else _FakeQuerySet([])


def _binding(model: _FakeModel, logger: MagicMock | None = None) -> RunLifecycleBinding:
    return RunLifecycleBinding(
        model=model,  # type: ignore[arg-type]
        failed_status="failed",
        non_terminal_statuses=["initializing", "running"],
        logger=logger or MagicMock(),
    )


# ════════════════════════════════════════════════════════════════════════════
# heartbeat_run
# ════════════════════════════════════════════════════════════════════════════


async def test_heartbeat_run_updates_timestamp() -> None:
    qs = _FakeQuerySet([], update_result=1)
    model = _FakeModel(qs)
    log = MagicMock()

    await heartbeat_run(
        model,  # type: ignore[arg-type]
        "run-1",
        logger=log,
        no_row_log_message="run.heartbeat_no_row run_id=%s",
    )

    assert model.filter_calls == [{"run_id": "run-1"}]
    assert "heartbeat_at" in qs.update_kwargs  # type: ignore[operator]
    log.warning.assert_not_called()


async def test_heartbeat_run_warns_when_no_row() -> None:
    model = _FakeModel(_FakeQuerySet([], update_result=0))
    log = MagicMock()

    await heartbeat_run(
        model,  # type: ignore[arg-type]
        "ghost-run",
        logger=log,
        no_row_log_message="run.heartbeat_no_row run_id=%s",
    )

    log.warning.assert_called_once_with("run.heartbeat_no_row run_id=%s", "ghost-run")


# ════════════════════════════════════════════════════════════════════════════
# fail_run
# ════════════════════════════════════════════════════════════════════════════


async def test_fail_run_marks_failed_and_logs() -> None:
    qs = _FakeQuerySet([], update_result=1)
    model = _FakeModel(qs)
    log = MagicMock()

    await fail_run(
        _binding(model, log),
        "run-1",
        FailRunSpec(
            reason="eval timeout",
            no_row_log_message="run.fail_no_row run_id=%s reason=%s",
            failed_log_message="run.failed run_id=%s reason=%s",
        ),
    )

    assert qs.update_kwargs is not None
    assert qs.update_kwargs["status"] == "failed"
    assert qs.update_kwargs["error_message"] == "eval timeout"
    assert "finished_at" in qs.update_kwargs
    log.warning.assert_called_once_with("run.failed run_id=%s reason=%s", "run-1", "eval timeout")


async def test_fail_run_logs_error_when_no_row() -> None:
    model = _FakeModel(_FakeQuerySet([], update_result=0))
    log = MagicMock()

    await fail_run(
        _binding(model, log),
        "ghost",
        FailRunSpec(
            reason="boom",
            no_row_log_message="run.fail_no_row run_id=%s reason=%s",
            failed_log_message="run.failed run_id=%s reason=%s",
        ),
    )

    log.error.assert_called_once_with("run.fail_no_row run_id=%s reason=%s", "ghost", "boom")
    log.warning.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
# fail_inflight_runs
# ════════════════════════════════════════════════════════════════════════════


async def test_fail_inflight_runs_fails_all_non_terminal_rows() -> None:
    rows = [_FakeRun("run-1"), _FakeRun("run-2")]
    qs = _FakeQuerySet(rows)
    per_run_qs = [_FakeQuerySet([r], update_result=1) for r in rows]
    model = _FakeModel(qs, *per_run_qs)
    log = MagicMock()

    failed = await fail_inflight_runs(
        _binding(model, log),
        owner="worker-1",
        reason="graceful shutdown",
        log_message="run.drain_failed run_id=%s owner=%s reason=%s",
    )

    assert failed == ["run-1", "run-2"]
    assert model.filter_calls[0] == {
        "owner": "worker-1",
        "status__in": ["initializing", "running"],
    }
    for row, per_run in zip(rows, per_run_qs, strict=True):
        assert per_run.update_kwargs["status"] == "failed"  # type: ignore[index]
        assert per_run.update_kwargs["error_message"] == "graceful shutdown"  # type: ignore[index]
    assert log.warning.call_count == 2


async def test_fail_inflight_runs_empty_owner_returns_empty() -> None:
    model = _FakeModel(_FakeQuerySet([]))
    failed = await fail_inflight_runs(
        _binding(model),
        owner="nobody",
        reason="shutdown",
        log_message="run.drain_failed run_id=%s owner=%s reason=%s",
    )
    assert failed == []


# ════════════════════════════════════════════════════════════════════════════
# recover_stale_runs
# ════════════════════════════════════════════════════════════════════════════


async def test_recover_stale_runs_marks_both_stale_kinds() -> None:
    """Both kinds of stale row, heartbeat timeout and never heartbeated, must be recovered, and the query conditions are mutually exclusive."""
    heartbeat_stale = _FakeRun("run-old", heartbeat_at=datetime.now(UTC))
    never_heartbeat = _FakeRun("run-never", heartbeat_at=None)
    model = _FakeModel(
        _FakeQuerySet([heartbeat_stale]),
        _FakeQuerySet([never_heartbeat]),
    )

    recovered = await recover_stale_runs(
        _binding(model),
        log_message="run.recovered run_id=%s owner=%s heartbeat=%s prev=%s",
    )

    assert sorted(recovered) == ["run-never", "run-old"]
    # Two queries: heartbeat_at expired / heartbeat_at IS NULL + created_at expired
    assert "heartbeat_at__lt" in model.filter_calls[0]
    assert model.filter_calls[1]["heartbeat_at__isnull"] is True
    assert "created_at__lt" in model.filter_calls[1]
    for run in (heartbeat_stale, never_heartbeat):
        assert run.status == "failed"
        assert run.error_message == STALE_RECOVERY_REASON
        assert run.finished_at is not None
        assert run.saved is True


async def test_recover_stale_runs_logs_pre_recovery_status() -> None:
    """Regression pin: prev_status in the log must be the original status from before recovery.

    update_from_dict rewrites run.status in place; if the log reads run.status afterward,
    it always prints failed_status, and debugging cannot tell running from initializing.
    """
    running = _FakeRun("run-running", status="running", heartbeat_at=datetime.now(UTC))
    initializing = _FakeRun("run-init", status="initializing", heartbeat_at=datetime.now(UTC))
    model = _FakeModel(_FakeQuerySet([running, initializing]), _FakeQuerySet([]))
    log = MagicMock()

    await recover_stale_runs(
        _binding(model, log),
        log_message="run.recovered run_id=%s owner=%s heartbeat=%s prev=%s",
    )

    prev_statuses = {call.args[1]: call.args[4] for call in log.warning.call_args_list}
    assert prev_statuses == {"run-running": "running", "run-init": "initializing"}


async def test_recover_stale_runs_empty_when_nothing_stale() -> None:
    model = _FakeModel(_FakeQuerySet([]), _FakeQuerySet([]))
    assert await recover_stale_runs(_binding(model), log_message="run.recovered %s %s %s %s") == []


# ════════════════════════════════════════════════════════════════════════════
# fail_eval_run — best-effort cleanup, and it must never raise
# ════════════════════════════════════════════════════════════════════════════


class _FakeRunService:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.calls: list[tuple[str, str]] = []

    async def fail_run(self, *, run_id: str, reason: str) -> object:
        if self._error is not None:
            raise self._error
        self.calls.append((run_id, reason))
        return None


async def test_fail_eval_run_delegates_to_service() -> None:
    service = _FakeRunService()
    await fail_eval_run(service, "run-1", "eval boom", log_prefix="benchmark")
    assert service.calls == [("run-1", "eval boom")]


async def test_fail_eval_run_swallows_service_exception() -> None:
    """A failure on the cleanup path must not blow through the runner's exception handling again."""
    service = _FakeRunService(error=ConnectionError("pg down"))
    await fail_eval_run(service, "run-1", "eval boom", log_prefix="benchmark")
