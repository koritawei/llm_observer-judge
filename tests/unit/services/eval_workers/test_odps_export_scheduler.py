"""Unit tests for the OdpsExportScheduler multi-host lock and trigger paths."""

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from benchmark.domain.odps_export_config import OdpsExportConfig
from benchmark.reporting.framework.odps_export import OdpsExportResult
from benchmark.workers.odps_export_scheduler import (
    OdpsExportScheduler,
    _ExportTask,
)


def _config(**overrides: Any) -> OdpsExportConfig:
    base: dict[str, Any] = {
        "enabled": True,
        "schedule_time": "06:00",
        "schedule_timezone": "Asia/Shanghai",
        "limit": 200,
        "oss_prefix": "agent-eval/search-agent-overview",
        "sql_template": "SELECT 1 WHERE ds = '{ds}' LIMIT {limit}",
    }
    base.update(overrides)
    return OdpsExportConfig.model_validate(base)


def _redis_not_done() -> AsyncMock:
    """Redis mock: the done completion mark does not exist (``exists`` → 0, so the execution-lock flow is let through)."""
    redis = AsyncMock()
    redis.exists = AsyncMock(return_value=0)
    return redis


@pytest.mark.asyncio
async def test_trigger_skips_when_disabled() -> None:
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config(enabled=False))
    with patch.object(scheduler, "_get_redis", new_callable=AsyncMock) as get_redis:
        await scheduler._trigger_once(reason="test")
        get_redis.assert_not_awaited()


@pytest.mark.asyncio
async def test_trigger_skips_when_no_jobs() -> None:
    scheduler = OdpsExportScheduler(
        config_fetcher=lambda **_: _config(sql_template="  ", jobs=[]),
    )
    with patch.object(scheduler, "_get_redis", new_callable=AsyncMock) as get_redis:
        await scheduler._trigger_once(reason="test")
        get_redis.assert_not_awaited()


@pytest.mark.asyncio
async def test_jobs_only_format_triggers_on_startup_compensation() -> None:
    cfg = OdpsExportConfig.model_validate(
        {
            "jobs": [
                {
                    "enabled": True,
                    "schedule_time": "10:30",
                    "schedule_timezone": "Asia/Shanghai",
                    "limit": 200,
                    "oss_prefix": "search-eval/benchmark_result/",
                    "sql_template": "SELECT 1 WHERE ds='{ds}' LIMIT {limit}",
                    "project_id": "cmpwln0qs04fo6j8eryfdgznb",
                },
            ],
        },
    )
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: cfg)
    redis = _redis_not_done()
    result = OdpsExportResult(
        odps_ds="20260908",
        oss_ds="20260909",
        rows=1,
        object_key="k",
        oss_url="u",
    )
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)),
        patch(
            "benchmark.workers.odps_export_scheduler.default_odps_ds",
            return_value="20260908",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.default_oss_ds",
            return_value="20260909",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
            return_value=result,
        ) as export,
    ):
        await scheduler._trigger_once(reason="startup_compensation")
        export.assert_awaited_once()
        assert export.await_args.args[0].job_id == "default"
        assert export.await_args.args[0].oss_prefix == "search-eval/benchmark_result/"
        assert export.await_args.args[0].project_id == "cmpwln0qs04fo6j8eryfdgznb"


@pytest.mark.asyncio
async def test_trigger_skips_when_lock_not_acquired() -> None:
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=False)),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
        ) as export,
    ):
        await scheduler._trigger_once(reason="test")
        export.assert_not_awaited()


@pytest.mark.asyncio
async def test_trigger_exports_when_lock_acquired() -> None:
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    result = OdpsExportResult(
        odps_ds="20260908",
        oss_ds="20260909",
        rows=2,
        columns=["query"],
        csv_bytes=b"x",
        object_key="agent-eval/search-agent-overview/20260909/benchmark_final_results.csv",
        oss_url="https://oss/k",
    )
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)) as acquire,
        patch(
            "benchmark.workers.odps_export_scheduler.default_odps_ds",
            return_value="20260908",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.default_oss_ds",
            return_value="20260909",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
            return_value=result,
        ) as export,
        patch.object(scheduler, "_release_lock", new_callable=AsyncMock) as release,
    ):
        await scheduler._trigger_once(reason="test")
        export.assert_awaited_once()
        req = export.await_args.args[0]
        assert req.odps_ds == "20260908"
        assert req.oss_ds == "20260909"
        assert req.job_id == "default"
        acquire.assert_awaited_once()
        lock_key = acquire.await_args.args[1]
        assert "odps_export:default:20260909" in lock_key
        release.assert_not_awaited()


@pytest.mark.asyncio
async def test_multi_jobs_export_independently() -> None:
    cfg = _config(
        sql_template="",
        jobs=[
            {
                "id": "biz-a",
                "sql_template": "SELECT a WHERE ds='{ds}' LIMIT {limit}",
                "oss_prefix": "a",
            },
            {
                "id": "biz-b",
                "sql_template": "SELECT b WHERE ds='{ds}' LIMIT {limit}",
                "oss_prefix": "b",
            },
        ],
    )
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: cfg)
    redis = _redis_not_done()
    result = OdpsExportResult(
        odps_ds="20260908",
        oss_ds="20260909",
        rows=1,
        object_key="k",
        oss_url="u",
    )
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)) as acquire,
        patch(
            "benchmark.workers.odps_export_scheduler.default_odps_ds",
            return_value="20260908",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.default_oss_ds",
            return_value="20260909",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
            return_value=result,
        ) as export,
    ):
        await scheduler._trigger_once(reason="test")
        assert export.await_count == 2
        job_ids = [c.args[0].job_id for c in export.await_args_list]
        assert job_ids == ["biz-a", "biz-b"]
        lock_keys = [c.args[1] for c in acquire.await_args_list]
        assert any("odps_export:biz-a:20260909" in k for k in lock_keys)
        assert any("odps_export:biz-b:20260909" in k for k in lock_keys)


@pytest.mark.asyncio
async def test_one_job_failure_continues_others() -> None:
    cfg = _config(
        sql_template="",
        jobs=[
            {"id": "fail", "sql_template": "SELECT f WHERE ds='{ds}' LIMIT {limit}"},
            {"id": "ok", "sql_template": "SELECT o WHERE ds='{ds}' LIMIT {limit}"},
        ],
    )
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: cfg)
    redis = _redis_not_done()

    async def _export(req: Any) -> OdpsExportResult:
        if req.job_id == "fail":
            raise RuntimeError("boom")
        return OdpsExportResult(
            odps_ds=req.odps_ds,
            oss_ds=req.oss_ds,
            rows=1,
            object_key="k",
            oss_url="u",
        )

    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)),
        patch(
            "benchmark.workers.odps_export_scheduler.default_odps_ds",
            return_value="20260908",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.default_oss_ds",
            return_value="20260909",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            side_effect=_export,
        ),
        patch.object(scheduler, "_release_lock", new_callable=AsyncMock) as release,
    ):
        await scheduler._trigger_once(reason="test")
        # fail job releases lock; ok job keeps lock
        assert release.await_count == 1


@pytest.mark.asyncio
async def test_trigger_releases_lock_on_export_failure() -> None:
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)),
        patch(
            "benchmark.workers.odps_export_scheduler.default_odps_ds",
            return_value="20260908",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.default_oss_ds",
            return_value="20260909",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
            side_effect=RuntimeError("odps down"),
        ),
        patch.object(scheduler, "_release_lock", new_callable=AsyncMock) as release,
    ):
        await scheduler._trigger_once(reason="test")
        release.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_noop_when_remote_config_disabled() -> None:
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config(enabled=False))
    await scheduler.start()
    assert scheduler._hub.started is False


def test_lock_key_includes_job_id() -> None:
    with patch(
        "benchmark.libs.queue_runtime.get_prefix",
        return_value="eval",
    ):
        assert (
            OdpsExportScheduler._lock_key("20260909", "biz-a") == "eval:odps_export:biz-a:20260909"
        )


def test_calc_sleep_until_next_after_schedule() -> None:
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    config = _config(schedule_time="06:00")
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Shanghai")
    fixed = datetime(2026, 9, 9, 7, 0, 0, tzinfo=tz)

    class _FakeDateTime:
        @staticmethod
        def now(_tz: object = None) -> datetime:
            return fixed

    with patch(
        "benchmark.workers.odps_export_scheduler.datetime",
        _FakeDateTime,
    ):
        seconds = scheduler._calc_sleep_until_next_trigger(config)
    assert 22 * 3600 < seconds <= 24 * 3600


@pytest.mark.asyncio
async def test_scheduled_runs_job_whose_slot_elapsed_same_day() -> None:
    """Several hours past the fixed time (outside the old 2-minute window) it still runs the same day, and is not pushed to the next day.

    Regression: an earlier job's export duration dragged a later job out of the window, the old decision skipped it outright,
    and it slept until the next day's fixed time, so that day's export was lost entirely.
    """
    cfg = OdpsExportConfig.model_validate(
        {
            "jobs": [
                {
                    "id": "morning",
                    "schedule_time": "06:00",
                    "schedule_timezone": "Asia/Shanghai",
                    "sql_template": "SELECT 1 WHERE ds='{ds}' LIMIT {limit}",
                },
            ],
        },
    )
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: cfg)
    from zoneinfo import ZoneInfo

    fixed = datetime(2026, 9, 9, 10, 30, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    class _FakeDateTime:
        @staticmethod
        def now(_tz: object = None) -> datetime:
            return fixed

    redis = _redis_not_done()
    result = OdpsExportResult(
        odps_ds="20260908",
        oss_ds="20260909",
        rows=1,
        object_key="k",
        oss_url="u",
    )
    with (
        patch(
            "benchmark.workers.odps_export_scheduler.datetime",
            _FakeDateTime,
        ),
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
            return_value=result,
        ) as export,
    ):
        await scheduler._trigger_once(reason="scheduled")
        export.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduled_skips_job_before_its_slot() -> None:
    """A job whose fixed time has not arrived (and is not inside the early-wake slack) is not run early."""
    cfg = OdpsExportConfig.model_validate(
        {
            "jobs": [
                {
                    "id": "morning",
                    "schedule_time": "06:00",
                    "schedule_timezone": "Asia/Shanghai",
                    "sql_template": "SELECT 1 WHERE ds='{ds}' LIMIT {limit}",
                },
            ],
        },
    )
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: cfg)
    from zoneinfo import ZoneInfo

    fixed = datetime(2026, 9, 9, 5, 30, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    class _FakeDateTime:
        @staticmethod
        def now(_tz: object = None) -> datetime:
            return fixed

    with (
        patch(
            "benchmark.workers.odps_export_scheduler.datetime",
            _FakeDateTime,
        ),
        patch.object(scheduler, "_get_redis", new_callable=AsyncMock) as get_redis,
    ):
        await scheduler._trigger_once(reason="scheduled")
        get_redis.assert_not_awaited()


def test_job_due_now_scheduled_accepts_sleep_early_wake() -> None:
    """asyncio.sleep waking 1ms early still judges the 05:00 job due."""
    cfg = OdpsExportConfig.model_validate(
        {
            "jobs": [
                {
                    "id": "morning",
                    "schedule_time": "05:00",
                    "schedule_timezone": "Asia/Shanghai",
                    "sql_template": "SELECT 1 WHERE ds='{ds}' LIMIT {limit}",
                },
            ],
        },
    )
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: cfg)
    from zoneinfo import ZoneInfo

    frozen = datetime(2026, 9, 21, 4, 59, 59, 999000, tzinfo=ZoneInfo("Asia/Shanghai"))

    class _FakeDateTime:
        @staticmethod
        def now(_tz: object = None) -> datetime:
            return frozen

    with patch(
        "benchmark.workers.odps_export_scheduler.datetime",
        _FakeDateTime,
    ):
        assert scheduler._job_due_now(cfg.enabled_jobs()[0], reason="scheduled") is True


def test_seconds_until_job_skips_elapsed_slot_after_early_wake() -> None:
    """At 05:00:00.000 today's 05:00 slot has arrived, and sleep jumps to the next day rather than 0s."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    job = _config(
        sql_template="SELECT 1 WHERE ds='{ds}' LIMIT {limit}",
        schedule_time="05:00",
        schedule_timezone="Asia/Shanghai",
    ).enabled_jobs()[0]
    from zoneinfo import ZoneInfo

    frozen = datetime(2026, 9, 21, 5, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    class _FakeDateTime:
        @staticmethod
        def now(_tz: object = None) -> datetime:
            return frozen

    with patch(
        "benchmark.workers.odps_export_scheduler.datetime",
        _FakeDateTime,
    ):
        seconds = scheduler._seconds_until_job_trigger(job)
    assert 23 * 3600 < seconds <= 24 * 3600


# ── completion mark (already executed means do not execute again; idempotent across restarts)─────────────────────────


def test_done_key_includes_env_prefix_and_job_id() -> None:
    """The done key carries the same environment prefix as the execution lock (daily/pre/prod are isolated, and production and pre-release do not affect each other)."""
    with patch("benchmark.libs.queue_runtime.get_prefix", return_value="pre"):
        assert (
            OdpsExportScheduler._done_key("20260909", "biz-a")
            == "pre:odps_export:done:biz-a:20260909"
        )
        assert (
            OdpsExportScheduler._lock_key("20260909", "biz-a") == "pre:odps_export:biz-a:20260909"
        )


@pytest.mark.asyncio
async def test_trigger_skips_when_done_mark_exists() -> None:
    """Already completed today (the done mark exists) → do not execute again: do not grab the lock, and do not export.

    Covers: after the lock TTL (6h) expires, a service restart/release triggers startup compensation, and the export is not repeated.
    """
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    redis.exists = AsyncMock(return_value=1)  # The done mark already exists
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", new_callable=AsyncMock) as acquire,
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
        ) as export,
    ):
        await scheduler._trigger_once(reason="startup_compensation")
        acquire.assert_not_awaited()
        export.assert_not_awaited()


@pytest.mark.asyncio
async def test_export_success_writes_done_mark() -> None:
    """Export succeeds → write the done mark (48h TTL, and a later trigger/startup compensation skips it directly)."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    result = OdpsExportResult(
        odps_ds="20260908",
        oss_ds="20260909",
        rows=1,
        object_key="k",
        oss_url="u",
    )
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)),
        patch(
            "benchmark.workers.odps_export_scheduler.default_odps_ds",
            return_value="20260908",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.default_oss_ds",
            return_value="20260909",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
            return_value=result,
        ),
        patch("benchmark.libs.queue_runtime.get_prefix", return_value="prod"),
    ):
        await scheduler._trigger_once(reason="test")
        done_calls = [c for c in redis.set.await_args_list if ":done:" in str(c.args[0])]
        assert len(done_calls) == 1
        assert done_calls[0].args[0] == "prod:odps_export:done:default:20260909"
        assert done_calls[0].kwargs["ex"] == 48 * 3600


@pytest.mark.asyncio
async def test_export_failure_does_not_write_done_mark() -> None:
    """Export fails → do not write the done mark (a later retry is allowed) + release the execution lock."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
            side_effect=RuntimeError("odps down"),
        ),
        patch.object(scheduler, "_release_lock", new_callable=AsyncMock) as release,
    ):
        await scheduler._trigger_once(reason="test")
        release.assert_awaited_once()
        done_calls = [c for c in redis.set.await_args_list if ":done:" in str(c.args[0])]
        assert not done_calls


@pytest.mark.asyncio
async def test_done_check_error_falls_back_to_lock_flow() -> None:
    """A Redis exception while querying done → treat it as not completed, and the execution lock still falls back to preventing a duplicate."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    redis.exists = AsyncMock(side_effect=RuntimeError("redis flaky"))
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=False)),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
        ) as export,
    ):
        # _is_done raises → False → enter the lock flow; the lock was not grabbed → do not export (the no-duplicate semantics hold)
        await scheduler._trigger_once(reason="test")
        export.assert_not_awaited()


# ── retry while the lock is held + in-flight task lifecycle (20260923 production-incident regression)─────────────


@pytest.mark.asyncio
async def test_trigger_returns_true_when_lock_held() -> None:
    """The lock was not grabbed → _trigger_once returns True (_retry_locked_exports uses that to re-probe on a short cycle)."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=False)),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
        ) as export,
    ):
        assert await scheduler._trigger_once(reason="test") == {"default"}
        export.assert_not_awaited()


@pytest.mark.asyncio
async def test_lock_held_retry_takes_over_after_holder_release() -> None:
    """The first attempt finds the lock held → re-probe at 30min → the second attempt grabs the lock and the export succeeds (it no longer sleeps 24h and loses the day)."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    result = OdpsExportResult(
        odps_ds="20260908",
        oss_ds="20260909",
        rows=1,
        object_key="k",
        oss_url="u",
    )
    lock_results = iter([False, True])

    async def _acquire(_redis: object, _key: object) -> bool:
        return next(lock_results)

    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", side_effect=_acquire),
        patch(
            "benchmark.workers.odps_export_scheduler.default_odps_ds",
            return_value="20260908",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.default_oss_ds",
            return_value="20260909",
        ),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
            return_value=result,
        ) as export,
        patch(
            "benchmark.workers.odps_export_scheduler.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep_mock,
    ):
        # reason is not scheduled: bypass the due window (do not depend on the real clock), and focus on the retry semantics
        await scheduler._retry_locked_exports(reason_first="test")

    export.assert_awaited_once()
    sleep_mock.assert_awaited_once()  # Take over after exactly one 30min re-probe wait


@pytest.mark.asyncio
async def test_lock_held_retry_exhausted_gives_up() -> None:
    """The re-probe window is exhausted and the lock is still held → give up and wait for the next day's fixed time (do not retry forever)."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    trigger_count = 0

    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=False)),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            new_callable=AsyncMock,
        ) as export,
        patch(
            "benchmark.workers.odps_export_scheduler.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep_mock,
        patch(
            "benchmark.workers.odps_export_scheduler.time.monotonic",
            side_effect=lambda: 0.0,
        ),
    ):
        # monotonic stays 0 → the window never runs out; limit the sleep count to simulate a finite re-probe
        async def _limit_sleep(_s: float) -> None:
            nonlocal trigger_count
            trigger_count += 1
            if trigger_count >= 3:
                scheduler._stopped = True

        sleep_mock.side_effect = _limit_sleep
        # reason is not scheduled: bypass the due window, and focus on the give-up-when-the-window-is-exhausted semantics
        await scheduler._retry_locked_exports(reason_first="test")

    export.assert_not_awaited()
    assert sleep_mock.await_count == 3


@pytest.mark.asyncio
async def test_start_stop_tracks_compensation_task() -> None:
    """The startup-compensation task is tracked; stop cancels it too (an orphan task would be killed when the script process exits)."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    with patch.object(
        scheduler,
        "_retry_locked_exports",
        new_callable=AsyncMock,
    ):
        await scheduler.start()
        assert scheduler._hub.started is True
        assert scheduler._compensation_task is not None
        assert not scheduler._compensation_task.done()

        await scheduler.stop()

    assert scheduler._hub.started is False
    assert scheduler._compensation_task is None


@pytest.mark.asyncio
async def test_cancelled_export_releases_lock_for_takeover() -> None:
    """Cancelled mid-export (service shutdown) → release the execution lock so another node can take over, rather than holding it until the 6h TTL."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    job = _config().enabled_jobs()[0]

    async def _hang_forever(_req: object) -> OdpsExportResult:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    with (
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            side_effect=_hang_forever,
        ),
        patch.object(scheduler, "_release_lock", new_callable=AsyncMock) as release,
    ):
        task = asyncio.create_task(
            scheduler._export_one_job(
                redis,
                _ExportTask(
                    job=job,
                    odps_ds="20260908",
                    oss_ds="20260909",
                    reason="test",
                ),
            ),
        )
        for _ in range(20):
            await asyncio.sleep(0)
            if task.done():
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_export_failure_writes_fail_backoff() -> None:
    """A failed export writes a 30min fail key, and later ticks skip it until the TTL expires."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())
    redis = _redis_not_done()
    job = _config().enabled_jobs()[0]
    with (
        patch.object(scheduler, "_acquire_lock", AsyncMock(return_value=True)),
        patch.object(scheduler, "_release_lock", new_callable=AsyncMock),
        patch.object(scheduler, "_mark_fail_backoff", new_callable=AsyncMock) as mark_fail,
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            AsyncMock(side_effect=RuntimeError("odps boom")),
        ),
    ):
        held = await scheduler._export_one_job(
            redis,
            _ExportTask(
                job=job,
                odps_ds="20260908",
                oss_ds="20260909",
                reason="scheduled",
            ),
        )
    assert held is False
    mark_fail.assert_awaited_once()

    # When the fail key exists, skip directly, and do not grab the lock
    redis.exists = AsyncMock(side_effect=[0, 1])  # done=0, fail=1
    with patch.object(scheduler, "_acquire_lock", new_callable=AsyncMock) as acquire:
        held2 = await scheduler._export_one_job(
            redis,
            _ExportTask(
                job=job,
                odps_ds="20260908",
                oss_ds="20260909",
                reason="scheduled",
            ),
        )
    assert held2 is False
    acquire.assert_not_awaited()


@pytest.mark.asyncio
async def test_lock_retry_reprobes_only_locked_jobs() -> None:
    """Re-probe covers only jobs the first round skipped because of the lock — it does not open the due window (the review fix is locked).

    job_b's first-round export failed (no done mark); if the re-probe reused the non-scheduled full-due
    semantics, job_b would be retried and exported early; after filtering, the re-probe handles only job_a.
    """
    cfg = OdpsExportConfig.model_validate(
        {
            "jobs": [
                {
                    "id": "job_a",
                    "enabled": True,
                    "schedule_time": "10:30",
                    "schedule_timezone": "Asia/Shanghai",
                    "sql_template": "SELECT 1",
                },
                {
                    "id": "job_b",
                    "enabled": True,
                    "schedule_time": "23:00",
                    "schedule_timezone": "Asia/Shanghai",
                    "sql_template": "SELECT 1",
                },
            ],
        },
    )
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: cfg)
    redis = _redis_not_done()
    result = OdpsExportResult(
        odps_ds="20260908",
        oss_ds="20260909",
        rows=1,
        object_key="k",
        oss_url="u",
    )
    acquire_calls: dict[str, int] = {"job_a": 0, "job_b": 0}

    async def _acquire(_redis: object, key: str) -> bool:
        for jid in ("job_a", "job_b"):
            if jid in key:
                acquire_calls[jid] += 1
                # job_a: the first round failed to hold the lock, and the re-probe succeeds; job_b: always succeeds
                return acquire_calls[jid] > 1 if jid == "job_a" else True
        return True

    async def _export(req: Any) -> OdpsExportResult:
        if req.job_id == "job_b":
            raise RuntimeError("boom")  # job_b fails on the first round (no done mark)
        return result

    with (
        patch.object(scheduler, "_get_redis", AsyncMock(return_value=redis)),
        patch.object(scheduler, "_acquire_lock", side_effect=_acquire),
        patch(
            "benchmark.workers.odps_export_scheduler.run_odps_export",
            side_effect=_export,
        ) as export,
        patch(
            "benchmark.workers.odps_export_scheduler.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        await scheduler._retry_locked_exports(reason_first="test")

    exported_job_ids = [call.args[0].job_id for call in export.call_args_list]
    assert exported_job_ids == ["job_b", "job_a"]


@pytest.mark.asyncio
async def test_release_lock_deletes_only_when_holder_matches() -> None:
    """compare-and-delete: DEL only when the value matches; a mismatch (someone else has taken the lock) is not deleted."""
    from benchmark.workers.odps_export_scheduler import (
        _RELEASE_LOCK_LUA,
        _WORKER_NAME,
    )

    class _FakeRedis:
        def __init__(self, stored: str) -> None:
            self.store = {"k": stored}

        async def eval(self, script: str, numkeys: int, key: str, value: str) -> int:
            assert script == _RELEASE_LOCK_LUA
            assert numkeys == 1
            if self.store.get(key) == value:
                del self.store[key]
                return 1
            return 0

    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())

    holder_redis = _FakeRedis(_WORKER_NAME)
    await scheduler._release_lock(holder_redis, "k")  # type: ignore[arg-type]
    assert "k" not in holder_redis.store

    taken_over_redis = _FakeRedis("other-host")
    await scheduler._release_lock(taken_over_redis, "k")  # type: ignore[arg-type]
    assert taken_over_redis.store["k"] == "other-host"


@pytest.mark.asyncio
async def test_compensation_task_swallows_config_error() -> None:
    """A config-center failure inside the compensation task is only logged and does not leak out; the stop() shutdown flow is not interrupted."""
    calls = 0

    def _fetcher(**_: Any) -> OdpsExportConfig:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _config()
        raise RuntimeError("config center down")

    scheduler = OdpsExportScheduler(config_fetcher=_fetcher)
    with patch(
        "benchmark.workers.odps_export_scheduler.asyncio.sleep",
        new_callable=AsyncMock,
    ):
        await scheduler.start()
        for _ in range(20):
            await asyncio.sleep(0)
        await scheduler.stop()  # Must not raise RuntimeError

    assert scheduler._hub.started is False
    assert scheduler._compensation_task is None


@pytest.mark.asyncio
async def test_stop_tolerates_task_exception() -> None:
    """Defense: when a task ends with an exception other than CancelledError, stop() does not interrupt cleanup."""
    scheduler = OdpsExportScheduler(config_fetcher=lambda **_: _config())

    async def _boom() -> None:
        await asyncio.sleep(0)
        raise ValueError("stored failure")

    scheduler._compensation_task = asyncio.create_task(_boom())
    for _ in range(20):
        await asyncio.sleep(0)
    await scheduler.stop()  # Must not raise ValueError

    assert scheduler._compensation_task is None
