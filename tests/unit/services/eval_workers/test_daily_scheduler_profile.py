"""Unit tests for DailyExternalBenchmarkScheduler profile passthrough (§9 migration + acceptance #7, the 4th entry).

Locks: the scheduler-item config extension fields (``rubric_class`` / ``parse_interceptor_ref`` /
``judge_class`` / ``attribution_class`` / ``limit``) produce a payload constructed by ``_trigger_one_benchmark``
that is isomorphic with the API entry (the same Resolver + the unified submit pipeline converge); omitting them (the new fields are not passed)
is rejected. Also covers benchmark-level schedule_time / no_data_cutoff.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from benchmark.domain.external_daily_config import (
    ExternalDailyBenchmarkConfig,
    ExternalDailyConfig,
)
from benchmark.runs.schema import ExternalBenchmarkRequest
from benchmark.workers import daily_scheduler as ds_mod
from benchmark.workers.daily_scheduler import DailyExternalBenchmarkScheduler

_RUBRIC_REF = "business.runs.ieval.amap_chat.rubric:AmapChatRubricV2"


class _FakeComponent:
    """Minimal benchmark-component stand-in (the scheduler only calls get_metadata)."""

    def __init__(self, name: str) -> None:
        self._name = name

    @staticmethod
    def get_metadata() -> MagicMock:
        meta = MagicMock()
        meta.name = "demo_bench"
        meta.service_name = "demo_svc"
        return meta


async def _trigger_payload(
    monkeypatch: pytest.MonkeyPatch,
    bm_config: ExternalDailyBenchmarkConfig,
) -> ExternalBenchmarkRequest:
    """Drive _trigger_one_benchmark and capture the constructed payload."""
    benchmark_service = MagicMock()
    benchmark_service.create_placeholder_run = AsyncMock(return_value=True)
    runner = MagicMock()
    runner._benchmark_service = benchmark_service
    runner._storage_client = None

    scheduler = DailyExternalBenchmarkScheduler(
        runner=runner,  # type: ignore[arg-type]
        config=ExternalDailyConfig(),
    )
    monkeypatch.setattr(scheduler, "_compute_date", lambda *_a: "20260905")
    monkeypatch.setattr(
        scheduler,
        "_skip_existing_run",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        scheduler,
        "_check_storage_file_exists",
        AsyncMock(return_value=True),
    )
    # The benchmark switch defaults to True (do not process): scheduler cases uniformly mock it open, and focus on payload assertions
    monkeypatch.setattr(
        ds_mod,
        "is_benchmark_disabled",
        lambda: False,
    )
    monkeypatch.setattr(
        ds_mod,
        "get_default_download_prefix",
        lambda: "oss://bucket/default-prefix",
    )
    monkeypatch.setattr(
        ds_mod,
        "BENCHMARK_SERVICES_CLASSES",
        (_FakeComponent,),
    )

    captured: dict[str, Any] = {}

    async def _capture(run_id: str, payload: ExternalBenchmarkRequest, _: Any) -> None:
        captured["run_id"] = run_id
        captured["payload"] = payload

    monkeypatch.setattr(scheduler, "_run_initialize", _capture)
    await scheduler._trigger_one_benchmark(bm_config, reason="test")
    # _run_initialize is scheduled via asyncio.create_task, and yielding the event loop lets it run
    await asyncio.sleep(0)
    return captured["payload"]


@pytest.mark.asyncio
async def test_scheduler_passes_profile_fields_to_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler-item extension fields → the payload is passed through isomorphically (it converges on the same Resolver as the API entry)."""
    payload = await _trigger_payload(
        monkeypatch,
        ExternalDailyBenchmarkConfig(
            benchmark_name="demo_bench",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
            rubric_class="amap_chat_rubric_v2",
            dataset_name="demo_ds",
            parse_interceptor_ref=(
                "business.runs.ieval.amap_chat.parse:AmapChatParseInterceptor"
            ),
        ),
    )
    assert payload.benchmark_name == "demo_bench"
    assert payload.rubric_class == "amap_chat_rubric_v2"
    assert payload.parse_interceptor_ref == (
        "business.runs.ieval.amap_chat.parse:AmapChatParseInterceptor"
    )
    assert payload.judge_class == "ieval_external_judge"
    assert payload.attribution_class == "no_attribution"
    # Normalized fields (the scheduler's responsibility is unchanged)
    assert payload.effective_date == "20260905"
    assert payload.effective_oss_prefix == "oss://bucket/default-prefix"


@pytest.mark.asyncio
async def test_scheduler_run_id_date_from_oss_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partition date in the last segment of the OSS prefix overrides the scheduler's default day, and is written into run_id / effective_date."""
    payload = await _trigger_payload(
        monkeypatch,
        ExternalDailyBenchmarkConfig(
            benchmark_name="ieval_external",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
            rubric_class="amap_chat_rubric_v2",
            oss_prefix=(
                "oss://agent-eval-oss/ieval-service/benchmark/benchmark_result/20261110"
            ),
        ),
    )
    assert payload.effective_date == "20261110"
    assert payload.date == "20261110"


@pytest.mark.asyncio
async def test_scheduler_run_id_date_from_macro_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The partition date after ``${yyyymmdd-1}`` expands becomes the run_id date."""
    payload = await _trigger_payload(
        monkeypatch,
        ExternalDailyBenchmarkConfig(
            benchmark_name="ieval_external",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
            rubric_class="amap_chat_rubric_v2",
            oss_prefix=(
                "oss://agent-eval-oss/ieval-service/benchmark/benchmark_result/"
                "${yyyymmdd-1}"
            ),
        ),
    )
    # _compute_date mock = 20260905 → the macro expands to 20260904
    assert payload.effective_date == "20260904"


def test_daily_config_missing_execution_classes_is_not_runnable() -> None:
    """When the execution class name is missing, the model keeps the empty value, and the scheduler fail-fast skips that item."""
    cfg = ExternalDailyBenchmarkConfig(benchmark_name="demo_bench")
    assert cfg.judge_class == ""
    assert cfg.attribution_class == ""
    assert cfg.rubric_class == ""


def test_daily_config_accepts_profile_fields_from_remote_config_json() -> None:
    """RemoteConfig JSON (the shape after migration) → parsed as ExternalDailyBenchmarkConfig."""
    cfg = ExternalDailyBenchmarkConfig.model_validate(
        {
            "benchmark_name": "ieval_external",
            "dataset_name": "amap_chat_eval",
            "oss_prefix": "oss://agent-eval-oss/ieval-service/benchmark/benchmark_result",
            "judge_class": "ieval_external_judge",
            "attribution_class": "ieval_external_attribution",
            "rubric_class": "amap_chat_rubric_v2",
        },
    )
    assert cfg.rubric_class == "amap_chat_rubric_v2"
    assert cfg.judge_class == "ieval_external_judge"
    assert cfg.attribution_class == "ieval_external_attribution"
    assert cfg.parse_interceptor_ref is None


def test_daily_config_rejects_legacy_execution_objects() -> None:
    """An old execution/provider config is rejected immediately."""
    with pytest.raises(ValueError, match="legacy execution config"):
        ExternalDailyBenchmarkConfig.model_validate(
            {
                "benchmark_name": "ieval_external",
                "judge_class": "ieval_external_judge",
                "attribution_class": "ieval_external_attribution",
                "judge": {"execution": "async", "provider": "external"},
            },
        )


def test_daily_config_rejects_legacy_rubric_ref() -> None:
    """An old rubric_ref class-path config is rejected immediately; only the rubric_class alias is accepted."""
    with pytest.raises(ValueError, match="rubric_ref"):
        ExternalDailyBenchmarkConfig.model_validate(
            {
                "benchmark_name": "ieval_external",
                "judge_class": "ieval_external_judge",
                "attribution_class": "ieval_external_attribution",
                "rubric_ref": _RUBRIC_REF,
            },
        )


@pytest.mark.asyncio
async def test_scheduler_skips_benchmark_without_rubric_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No rubric_class → the scheduler fail-fast skips it, and does not create a run."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),  # type: ignore[arg-type]
        config=ExternalDailyConfig(),
    )
    skip = AsyncMock(return_value=False)
    monkeypatch.setattr(scheduler, "_skip_existing_run", skip)
    await scheduler._trigger_one_benchmark(
        ExternalDailyBenchmarkConfig(
            benchmark_name="demo_bench",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
        ),
        reason="test",
    )
    skip.assert_not_awaited()


@pytest.mark.asyncio
async def test_scheduler_passes_limit_to_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduler-item limit → passed through as payload.limit (a small smoke volume); omitted None = the full set."""
    payload = await _trigger_payload(
        monkeypatch,
        ExternalDailyBenchmarkConfig(
            benchmark_name="demo_bench",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
            rubric_class="component_rubric",
            limit=5,
        ),
    )
    assert payload.limit == 5

    payload_default = await _trigger_payload(
        monkeypatch,
        ExternalDailyBenchmarkConfig(
            benchmark_name="demo_bench",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
            rubric_class="component_rubric",
        ),
    )
    assert payload_default.limit is None


def _bm(time_text: str = "", tz: str = "") -> ExternalDailyBenchmarkConfig:
    return ExternalDailyBenchmarkConfig(
        benchmark_name="demo_bench",
        schedule_time=time_text,
        schedule_timezone=tz,
    )


def test_bm_due_now_startup_reason_always_due() -> None:
    """startup_compensation compensates the full set: the fixed-time window is ignored."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    assert scheduler._bm_due_now(_bm(), reason="startup_compensation") is True


def test_bm_due_now_scheduled_window_with_override() -> None:
    """A scheduled trigger: the benchmark-level schedule_time decides the due window; when it is not set, inherit the top level."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    due_time = now.strftime("%H:%M")
    later = (now + timedelta(minutes=30)).strftime("%H:%M")
    assert scheduler._bm_due_now(_bm(due_time), reason="scheduled") is True
    assert scheduler._bm_due_now(_bm(later), reason="scheduled") is False


def _freeze_now(frozen: datetime) -> type:
    class _FakeDateTime:
        @staticmethod
        def now(_tz: object = None) -> datetime:
            return frozen

    return _FakeDateTime


def test_bm_due_now_scheduled_accepts_sleep_early_wake() -> None:
    """asyncio.sleep waking 1ms early (04:59:59.999) still judges the 05:00 task due."""
    tz = ZoneInfo("Asia/Shanghai")
    frozen = datetime(2026, 9, 21, 4, 59, 59, 999000, tzinfo=tz)
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True, schedule_timezone="Asia/Shanghai"),
    )
    with patch.object(ds_mod, "datetime", _freeze_now(frozen)):
        assert scheduler._bm_due_now(_bm("05:00", "Asia/Shanghai"), reason="scheduled") is True


def test_bm_due_now_scheduled_late_same_day_still_due() -> None:
    """30 minutes past the fixed time (outside the old 2-minute window) is still due the same day, and is not pushed to the next frequency day.

    Regression: a no-data recheck of an earlier benchmark dragged a later benchmark out of the window, and
    the old decision jumped straight to its next daily/weekly/monthly slot, so the whole current period was lost.
    """
    tz = ZoneInfo("Asia/Shanghai")
    frozen = datetime(2026, 9, 21, 5, 30, 0, tzinfo=tz)
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True, schedule_timezone="Asia/Shanghai"),
    )
    with patch.object(ds_mod, "datetime", _freeze_now(frozen)):
        assert scheduler._bm_due_now(_bm("05:00", "Asia/Shanghai"), reason="scheduled") is True


@pytest.mark.asyncio
async def test_trigger_once_scheduled_processes_elapsed_benchmark() -> None:
    """On a scheduled trigger, a benchmark whose fixed time has passed is actually processed (not only judged due)."""
    tz = ZoneInfo("Asia/Shanghai")
    frozen = datetime(2026, 9, 21, 5, 30, 0, tzinfo=tz)
    bm = ExternalDailyBenchmarkConfig(
        benchmark_name="late_bench",
        schedule_time="05:00",
        schedule_timezone="Asia/Shanghai",
    )
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(
            enabled=True,
            schedule_timezone="Asia/Shanghai",
            benchmarks=[bm],
        ),
    )
    scheduler._trigger_one_benchmark = AsyncMock()  # type: ignore[method-assign]
    with patch.object(ds_mod, "datetime", _freeze_now(frozen)):
        await scheduler._trigger_once(reason="scheduled")
    scheduler._trigger_one_benchmark.assert_awaited_once()


def test_early_wake_then_exact_second_advances_to_later_benchmark() -> None:
    """Production race: after an early wake, computing sleep 1ms later must not drop today's 05:00 slot, and must not sleep back to 05:00 either.

    05:00 has entered the early-wake slack → this slot counts as reached; the next hop is 11:30 the same day (6.5h).
    """
    tz = ZoneInfo("Asia/Shanghai")
    after_tick = datetime(2026, 9, 21, 5, 0, 0, 0, tzinfo=tz)
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(
            enabled=True,
            schedule_timezone="Asia/Shanghai",
            benchmarks=[
                ExternalDailyBenchmarkConfig(
                    benchmark_name="morning",
                    schedule_time="05:00",
                    schedule_timezone="Asia/Shanghai",
                ),
                ExternalDailyBenchmarkConfig(
                    benchmark_name="search_eval",
                    schedule_time="11:30",
                    schedule_timezone="Asia/Shanghai",
                ),
            ],
        ),
    )
    with patch.object(ds_mod, "datetime", _freeze_now(after_tick)):
        seconds = scheduler._calc_sleep_until_next_trigger()
    assert 6.4 * 3600 < seconds < 6.6 * 3600


def test_calc_sleep_takes_min_of_benchmark_schedules() -> None:
    """Several benchmarks with different fixed times: the next sleep takes the nearest trigger point (an unconfigured one inherits the top level)."""
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    near = (now + timedelta(minutes=5)).strftime("%H:%M")
    far = (now + timedelta(minutes=60)).strftime("%H:%M")
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(
            enabled=True,
            schedule_time=far,
            benchmarks=[
                ExternalDailyBenchmarkConfig(benchmark_name="a"),
                ExternalDailyBenchmarkConfig(benchmark_name="b", schedule_time=near),
            ],
        ),
    )
    seconds = scheduler._calc_sleep_until_next_trigger()
    assert 0 < seconds < 6 * 60


@pytest.mark.asyncio
async def test_bm_no_data_cutoff_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark-level no_data_cutoff is passed into the recheck; the top level no longer has that field."""
    runner = MagicMock()
    runner._benchmark_service = MagicMock()
    runner._storage_client = None
    scheduler = DailyExternalBenchmarkScheduler(
        runner=runner,
        config=ExternalDailyConfig(enabled=True),
    )
    monkeypatch.setattr(scheduler, "_compute_date", lambda *_a: "20260905")
    monkeypatch.setattr(scheduler, "_skip_existing_run", AsyncMock(return_value=False))
    # The benchmark switch defaults to True (do not process): mock it open so the later recheck logic runs
    monkeypatch.setattr(ds_mod, "is_benchmark_disabled", lambda: False)
    monkeypatch.setattr(
        scheduler,
        "_check_storage_file_exists",
        AsyncMock(return_value=False),
    )
    captured: dict[str, Any] = {}

    async def _retry(*args: Any) -> bool:
        # After the signature was narrowed, args = (payload, _RetrySchedule); cutoff is on the schedule
        captured["cutoff"] = args[1].cutoff
        return False

    monkeypatch.setattr(ds_mod, "BENCHMARK_SERVICES_CLASSES", (_FakeComponent,))
    svc = MagicMock()
    svc.create_placeholder_run = AsyncMock(return_value=True)
    svc.fail_run = AsyncMock()
    runner._benchmark_service = svc
    monkeypatch.setattr(scheduler, "_retry_until_cutoff", _retry)
    await scheduler._trigger_one_benchmark(
        ExternalDailyBenchmarkConfig(
            benchmark_name="demo_bench",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
            rubric_class="component_rubric",
            no_data_cutoff="14:00",
        ),
        reason="scheduled",
    )
    assert captured["cutoff"] == "14:00"
    svc.fail_run.assert_awaited_once_with(run_id="demo_bench-20260905", reason="启动失败")


@pytest.mark.asyncio
async def test_no_data_without_cutoff_does_not_mark_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """no_data_cutoff is not configured: after one 30min probe fails, the run is not marked."""
    svc = MagicMock()
    svc.create_placeholder_run = AsyncMock(return_value=True)
    svc.fail_run = AsyncMock()
    runner = MagicMock()
    runner._benchmark_service = svc
    runner._storage_client = None
    scheduler = DailyExternalBenchmarkScheduler(
        runner=runner,
        config=ExternalDailyConfig(enabled=True),
    )
    monkeypatch.setattr(scheduler, "_compute_date", lambda *_a: "20260905")
    monkeypatch.setattr(scheduler, "_skip_existing_run", AsyncMock(return_value=False))
    monkeypatch.setattr(ds_mod, "is_benchmark_disabled", lambda: False)
    monkeypatch.setattr(
        scheduler,
        "_check_storage_file_exists",
        AsyncMock(return_value=False),
    )
    captured: dict[str, Any] = {}

    async def _retry(*args: Any) -> bool:
        captured["cutoff"] = args[1].cutoff
        return False

    monkeypatch.setattr(scheduler, "_retry_until_cutoff", _retry)
    monkeypatch.setattr(ds_mod, "BENCHMARK_SERVICES_CLASSES", (_FakeComponent,))
    await scheduler._trigger_one_benchmark(
        ExternalDailyBenchmarkConfig(
            benchmark_name="demo_bench",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
            rubric_class="component_rubric",
        ),
        reason="scheduled",
    )
    assert captured["cutoff"] is None
    svc.fail_run.assert_not_awaited()
    svc.create_placeholder_run.assert_not_awaited()


def test_seconds_until_bm_trigger_does_not_skip_day_within_early_slack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Within ≤5s of the fixed time (inside the early-wake slack) the day's slot must not be skipped to the next matching day.

    schedule_slot_elapsed uses ``candidate <= now``: before the slot has really arrived it sleeps at most 1s more
    and then triggers idempotently; the version with slack would skip a same-day slot whose now is in [target-5s, target)
    (_trigger_once has already finished, so _bm_due_now never gets another chance to judge due), and that day's schedule is lost.
    """
    tz = ZoneInfo("Asia/Shanghai")

    class _FrozenDatetime(datetime):
        """Frozen at 2s before the 05:00 fixed time (inside the early-wake slack)."""

        @classmethod
        def now(cls, tz: Any = None) -> "_FrozenDatetime":
            return cls(2026, 9, 21, 4, 59, 58, tzinfo=tz)

    monkeypatch.setattr(ds_mod, "datetime", _FrozenDatetime)
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    bm = ExternalDailyBenchmarkConfig(
        benchmark_name="demo_bench",
        schedule_time="05:00",
        schedule_timezone="Asia/Shanghai",
    )
    # The version with slack: 05:00-5s <= 04:59:58 → judged "the slot has passed" → sleep ≈ 24h;
    # the candidate <= now version: sleep = 2s.
    sleep_s = scheduler._seconds_until_bm_trigger(bm)
    assert sleep_s < 60.0


@pytest.mark.asyncio
async def test_stopped_during_retry_does_not_mark_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recheck interrupted by shutdown (_retry_until_cutoff returns False because of _stopped) does not mark a failure.

    _retry_until_cutoff returning False has two meanings, "cutoff expired" and "the service stopped";
    shutdown must not create a placeholder run and wrongly write "startup failed" — a fake-failed run would also be
    CAS-reset by the next _skip_existing_run and waste max_pull_attempts. Note that stop() cancels only
    _run_loop; the startup-compensation task is not cancelled, and after its recheck sleep ends it reaches here.
    """
    svc = MagicMock()
    svc.create_placeholder_run = AsyncMock(return_value=True)
    svc.fail_run = AsyncMock()
    runner = MagicMock()
    runner._benchmark_service = svc
    runner._storage_client = None
    scheduler = DailyExternalBenchmarkScheduler(
        runner=runner,
        config=ExternalDailyConfig(enabled=True),
    )
    monkeypatch.setattr(scheduler, "_compute_date", lambda *_a: "20260905")
    monkeypatch.setattr(scheduler, "_skip_existing_run", AsyncMock(return_value=False))
    monkeypatch.setattr(ds_mod, "is_benchmark_disabled", lambda: False)
    monkeypatch.setattr(
        scheduler,
        "_check_storage_file_exists",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(scheduler, "_retry_until_cutoff", AsyncMock(return_value=False))
    monkeypatch.setattr(ds_mod, "BENCHMARK_SERVICES_CLASSES", (_FakeComponent,))
    # Simulate shutdown: when the recheck returns False, _stopped is already set.
    scheduler._stopped = True
    await scheduler._trigger_one_benchmark(
        ExternalDailyBenchmarkConfig(benchmark_name="demo_bench", no_data_cutoff="14:00"),
        reason="startup_compensation",
    )
    svc.fail_run.assert_not_awaited()
    svc.create_placeholder_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_mark_start_failed_skips_when_run_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the placeholder run was already created by another path, do not fail_run (aligned with the _trigger_one_benchmark idempotency contract).

    The recheck window can last several hours, and during it the same run_id may be created by another replica / the external-run API
    and run to RUNNING/COMPLETED; an unconditional fail_run would cascade and destroy its non-terminal items.
    """
    svc = MagicMock()
    svc.create_placeholder_run = AsyncMock(return_value=False)
    svc.fail_run = AsyncMock()
    runner = MagicMock()
    runner._benchmark_service = svc
    runner._storage_client = None
    scheduler = DailyExternalBenchmarkScheduler(
        runner=runner,
        config=ExternalDailyConfig(enabled=True),
    )
    monkeypatch.setattr(ds_mod, "BENCHMARK_SERVICES_CLASSES", (_FakeComponent,))
    payload = ExternalBenchmarkRequest(
        benchmark_name="demo_bench",
        judge_class="ieval_external_judge",
        attribution_class="no_attribution",
        rubric_class="amap_chat_rubric_v2",
        date="20260905",
    )
    await scheduler._mark_start_failed(
        "demo_bench-20260905",
        ExternalDailyBenchmarkConfig(benchmark_name="demo_bench", no_data_cutoff="14:00"),
        payload,
    )
    svc.fail_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_until_cutoff_polls_every_30_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a cutoff: probe every 30min, and stop as soon as the file appears."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    checks = {"n": 0}

    async def _check(_payload: object) -> bool:
        checks["n"] += 1
        return checks["n"] >= 2

    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    tz = ZoneInfo("Asia/Shanghai")
    monkeypatch.setattr(scheduler, "_check_storage_file_exists", _check)
    monkeypatch.setattr(ds_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        ds_mod,
        "datetime",
        _freeze_now(datetime(2026, 9, 21, 5, 0, 0, tzinfo=tz)),
    )
    ok = await scheduler._retry_until_cutoff(
        MagicMock(),
        ds_mod._RetrySchedule(cutoff="12:00", date="20260921", tz=tz),
    )
    assert ok is True
    assert sleeps == [30 * 60, 30 * 60]
    assert checks["n"] == 2


@pytest.mark.asyncio
async def test_retry_until_cutoff_oneshot_without_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no cutoff configured: sleep another 30min and probe once."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    checks = {"n": 0}

    async def _check(_payload: object) -> bool:
        checks["n"] += 1
        return False

    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    tz = ZoneInfo("Asia/Shanghai")
    monkeypatch.setattr(scheduler, "_check_storage_file_exists", _check)
    monkeypatch.setattr(ds_mod.asyncio, "sleep", _sleep)
    ok = await scheduler._retry_until_cutoff(
        MagicMock(),
        ds_mod._RetrySchedule(cutoff=None, date="20260921", tz=tz),
    )
    assert ok is False
    assert sleeps == [30 * 60]
    assert checks["n"] == 1


@pytest.mark.asyncio
async def test_retry_until_cutoff_caps_last_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When less than 30min remains until cutoff, shorten the sleep to the time left."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    tz = ZoneInfo("Asia/Shanghai")
    monkeypatch.setattr(scheduler, "_check_storage_file_exists", AsyncMock(return_value=True))
    monkeypatch.setattr(ds_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        ds_mod,
        "datetime",
        _freeze_now(datetime(2026, 9, 21, 11, 50, 0, tzinfo=tz)),
    )
    ok = await scheduler._retry_until_cutoff(
        MagicMock(),
        ds_mod._RetrySchedule(cutoff="12:00", date="20260921", tz=tz),
    )
    assert ok is True
    assert sleeps == [10 * 60]


@pytest.mark.asyncio
async def test_retry_until_cutoff_already_past_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cutoff has already passed when the recheck starts → do not sleep, and fail directly."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    tz = ZoneInfo("Asia/Shanghai")
    check = AsyncMock(return_value=True)
    monkeypatch.setattr(scheduler, "_check_storage_file_exists", check)
    monkeypatch.setattr(ds_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        ds_mod,
        "datetime",
        _freeze_now(datetime(2026, 9, 21, 12, 0, 0, tzinfo=tz)),
    )
    ok = await scheduler._retry_until_cutoff(
        MagicMock(),
        ds_mod._RetrySchedule(cutoff="12:00", date="20260921", tz=tz),
    )
    assert ok is False
    assert sleeps == []
    check.assert_not_awaited()


def _trigger_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    check: Any,
    skip: Any = None,
    cutoff: str | None = None,
) -> tuple[DailyExternalBenchmarkScheduler, dict[str, int], ExternalDailyBenchmarkConfig]:
    """Build a scheduler that can drive ``_trigger_one_benchmark`` (the switch / registration / placeholder are let through)."""
    svc = MagicMock()
    svc.create_placeholder_run = AsyncMock(return_value=True)
    svc.fail_run = AsyncMock()
    runner = MagicMock()
    runner._benchmark_service = svc
    runner._storage_client = None
    scheduler = DailyExternalBenchmarkScheduler(
        runner=runner,
        config=ExternalDailyConfig(enabled=True),
    )
    monkeypatch.setattr(scheduler, "_compute_date", lambda *_a: "20260905")
    monkeypatch.setattr(
        scheduler,
        "_skip_existing_run",
        skip if skip is not None else AsyncMock(return_value=False),
    )
    monkeypatch.setattr(scheduler, "_check_storage_file_exists", check)
    monkeypatch.setattr(ds_mod, "is_benchmark_disabled", lambda: False)
    monkeypatch.setattr(ds_mod, "BENCHMARK_SERVICES_CLASSES", (_FakeComponent,))
    started: dict[str, int] = {"n": 0}

    async def _init(*_a: object, **_k: object) -> None:
        started["n"] += 1

    monkeypatch.setattr(scheduler, "_run_initialize", _init)
    bm = ExternalDailyBenchmarkConfig(
        benchmark_name="demo_bench",
        judge_class="ieval_external_judge",
        attribution_class="no_attribution",
        rubric_class="component_rubric",
        no_data_cutoff=cutoff,
    )
    return scheduler, started, bm


@pytest.mark.asyncio
async def test_first_check_success_does_not_enter_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first precheck succeeds: start the run, and do not enter the 30min recheck."""
    check = AsyncMock(return_value=True)
    retry = AsyncMock(return_value=True)
    scheduler, started, bm = _trigger_scheduler(monkeypatch, check=check)
    monkeypatch.setattr(scheduler, "_retry_until_cutoff", retry)
    await scheduler._trigger_one_benchmark(bm, reason="scheduled")
    await asyncio.sleep(0)
    assert started["n"] == 1
    retry.assert_not_awaited()
    assert check.await_count == 1


@pytest.mark.asyncio
async def test_retry_success_stops_further_checks_and_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the recheck finds the file, start immediately, and do not keep probing every 30min."""
    checks = {"n": 0}

    async def _check(_payload: object) -> bool:
        checks["n"] += 1
        return checks["n"] >= 2

    sleeps: list[float] = []
    orig_sleep = asyncio.sleep

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await orig_sleep(0)

    tz = ZoneInfo("Asia/Shanghai")
    scheduler, started, bm = _trigger_scheduler(monkeypatch, check=_check, cutoff="14:00")
    monkeypatch.setattr(ds_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        ds_mod,
        "datetime",
        _freeze_now(datetime(2026, 9, 21, 5, 0, 0, tzinfo=tz)),
    )
    await scheduler._trigger_one_benchmark(bm, reason="scheduled")
    await orig_sleep(0)
    assert started["n"] == 1
    assert checks["n"] == 2
    assert sleeps == [30 * 60]


@pytest.mark.asyncio
async def test_started_run_second_trigger_skips_without_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Triggered again after a successful start: skip idempotently, and do not do another OSS precheck / recheck."""
    check = AsyncMock(return_value=True)
    skip = AsyncMock(side_effect=[False, True])
    scheduler, started, bm = _trigger_scheduler(monkeypatch, check=check, skip=skip)
    await scheduler._trigger_one_benchmark(bm, reason="scheduled")
    await asyncio.sleep(0)
    assert check.await_count == 1
    assert started["n"] == 1

    await scheduler._trigger_one_benchmark(bm, reason="scheduled")
    await asyncio.sleep(0)
    assert check.await_count == 1
    assert started["n"] == 1
    assert skip.await_count == 2


# ── schedule frequency (daily / weekly / monthly)────────────────────────


def _bm_freq(
    frequency: str = "daily",
    *,
    weekdays: list[int] | None = None,
    month_days: list[int] | None = None,
    time_text: str = "",
) -> ExternalDailyBenchmarkConfig:
    return ExternalDailyBenchmarkConfig(
        benchmark_name="demo_bench",
        schedule_frequency=frequency,
        schedule_weekdays=weekdays,
        schedule_month_days=month_days,
        schedule_time=time_text,
    )


def test_schedule_frequency_default_and_normalization() -> None:
    """frequency defaults to daily; case and whitespace are normalized; an illegal value falls back to daily (the whole parse does not blow up)."""
    assert ExternalDailyBenchmarkConfig().schedule_frequency == "daily"
    cfg = ExternalDailyBenchmarkConfig(schedule_frequency=" Weekly ")
    assert cfg.schedule_frequency == "weekly"
    assert ExternalDailyBenchmarkConfig(schedule_frequency="hourly").schedule_frequency == "daily"


def test_schedule_weekdays_month_days_filter_invalid_items() -> None:
    """Out-of-range weekdays/month_days items are filtered; all illegal → None (take the default)."""
    cfg = ExternalDailyBenchmarkConfig(schedule_weekdays=[1, 9, -1])
    assert cfg.schedule_weekdays == [1]
    assert ExternalDailyBenchmarkConfig(schedule_weekdays=[9]).schedule_weekdays is None
    cfg = ExternalDailyBenchmarkConfig(schedule_month_days=[1, 32])
    assert cfg.schedule_month_days == [1]
    assert ExternalDailyBenchmarkConfig(schedule_month_days=[0]).schedule_month_days is None


def test_bm_due_now_weekly_matches_configured_weekday() -> None:
    """weekly: due inside the fixed-time window only on a configured weekday; startup compensation is constrained by the frequency too."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    today_wd = now.weekday()
    other_wd = (today_wd + 1) % 7
    due_time = now.strftime("%H:%M")
    assert (
        scheduler._bm_due_now(
            _bm_freq("weekly", weekdays=[today_wd], time_text=due_time),
            reason="scheduled",
        )
        is True
    )
    assert (
        scheduler._bm_due_now(
            _bm_freq("weekly", weekdays=[other_wd], time_text=due_time),
            reason="scheduled",
        )
        is False
    )
    # startup compensation: backfill on a trigger day (ignore the fixed time), and do not backfill on a non-trigger day
    assert (
        scheduler._bm_due_now(
            _bm_freq("weekly", weekdays=[today_wd]),
            reason="startup_compensation",
        )
        is True
    )
    assert (
        scheduler._bm_due_now(
            _bm_freq("weekly", weekdays=[other_wd]),
            reason="startup_compensation",
        )
        is False
    )


def test_bm_due_now_monthly_matches_configured_month_day() -> None:
    """monthly: due inside the fixed-time window only on a configured date."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    due_time = now.strftime("%H:%M")
    other_day = 15 if now.day != 15 else 16
    assert (
        scheduler._bm_due_now(
            _bm_freq("monthly", month_days=[now.day], time_text=due_time),
            reason="scheduled",
        )
        is True
    )
    assert (
        scheduler._bm_due_now(
            _bm_freq("monthly", month_days=[other_day], time_text=due_time),
            reason="scheduled",
        )
        is False
    )


def test_seconds_until_trigger_weekly_jumps_to_next_matching_day() -> None:
    """weekly: the next trigger jumps to the next matching weekday (rather than the next day)."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    target_wd = (now.weekday() + 2) % 7  # The weekday that corresponds to the day after tomorrow (avoiding today's boundary)
    seconds = scheduler._seconds_until_bm_trigger(
        _bm_freq("weekly", weekdays=[target_wd], time_text="23:59"),
    )
    target_dt = now + timedelta(seconds=seconds)
    assert target_dt.weekday() == target_wd
    assert target_dt.date() > now.date()


def test_seconds_until_trigger_monthly_jumps_to_configured_day() -> None:
    """monthly: the next trigger jumps to the next matching date (it may cross a month)."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    target_day = 15 if now.day != 15 else 16
    seconds = scheduler._seconds_until_bm_trigger(
        _bm_freq("monthly", month_days=[target_day], time_text="23:59"),
    )
    target_dt = now + timedelta(seconds=seconds)
    assert target_dt.day == target_day
    assert target_dt > now


def test_seconds_until_trigger_daily_keeps_legacy_semantics() -> None:
    """daily (the default): the semantics are unchanged — if the fixed time has passed, trigger the next day, otherwise trigger today."""
    scheduler = DailyExternalBenchmarkScheduler(
        runner=MagicMock(),
        config=ExternalDailyConfig(enabled=True),
    )
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    seconds = scheduler._seconds_until_bm_trigger(_bm(time_text="23:59"))
    target_dt = now + timedelta(seconds=seconds)
    assert target_dt > now
    assert target_dt.date() in (now.date(), now.date() + timedelta(days=1))


# ── RemoteConfig null-placeholder compatibility (existing-defect regression)──────────────────────────


def test_remote_config_null_fields_parse_without_validation_error() -> None:
    """The null-placeholder style in the RemoteConfig example JSON (schedule_time/no_data_cutoff and similar) parses compatibly.

    Regression: an existing ``str`` field receiving ``None`` raised ValidationError → the whole config failed to parse
    → the scheduler degraded to enabled=False and ran nothing.
    """
    cfg = ExternalDailyConfig.model_validate(
        {
            "enabled": True,
            "schedule_time": None,  # A top-level null → normalize to the default
            "schedule_timezone": None,
            "no_data_cutoff": None,
            "benchmarks": [
                {
                    "benchmark_name": "demo_bench",
                    "schedule_time": None,
                    "schedule_timezone": None,
                    "no_data_cutoff": None,
                    "run_id_suffix": None,
                    "rubric": None,
                    "limit": None,
                },
            ],
        },
    )
    assert cfg.enabled is True
    # A top-level null: schedule_time/timezone normalize to the default; the top-level no_data_cutoff has been removed (extra ignore)
    assert cfg.schedule_time == "02:00"
    assert cfg.schedule_timezone == "Asia/Shanghai"
    assert "no_data_cutoff" not in ExternalDailyConfig.model_fields
    bm = cfg.benchmarks[0]
    assert bm.schedule_time is None
    assert bm.schedule_timezone is None
    assert bm.no_data_cutoff is None


def test_remote_config_blank_top_level_fields_fall_back_to_defaults() -> None:
    """A blank top-level schedule_time → normalize to the default; a leftover top-level no_data_cutoff is ignored."""
    cfg = ExternalDailyConfig.model_validate(
        {"enabled": True, "schedule_time": "  ", "no_data_cutoff": "06:00"},
    )
    assert cfg.schedule_time == "02:00"
    assert "no_data_cutoff" not in ExternalDailyConfig.model_fields
    bm = ExternalDailyBenchmarkConfig(benchmark_name="demo_bench", schedule_time="")
    assert bm.schedule_time == ""
    assert (bm.schedule_time or cfg.schedule_time) == "02:00"
    assert (
        ExternalDailyBenchmarkConfig(
            benchmark_name="demo_bench", no_data_cutoff="  "
        ).no_data_cutoff
        is None
    )


@pytest.mark.asyncio
async def test_trigger_skipped_when_benchmark_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark switch is on (RemoteConfig benchmark_disabled=true, the default) → return directly, and do not trigger a run."""
    monkeypatch.setattr(
        "benchmark.workers.daily_scheduler.is_benchmark_disabled",
        lambda: True,
    )
    runner = MagicMock()
    scheduler = DailyExternalBenchmarkScheduler(
        runner=runner,  # type: ignore[arg-type]
        config=ExternalDailyConfig(),
    )
    spy = AsyncMock()
    monkeypatch.setattr(scheduler, "_run_initialize", spy)
    await scheduler._trigger_one_benchmark(
        ExternalDailyBenchmarkConfig(benchmark_name="demo_bench"),
        reason="test",
    )
    await asyncio.sleep(0)
    spy.assert_not_called()
