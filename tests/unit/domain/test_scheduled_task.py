"""Scheduled-task documents: cron → schedule-item mapping and DB snapshot aggregation (RemoteConfig external-daily is masked)."""

from benchmark.domain.external_daily_config import ExternalDailyConfig
from benchmark.domain.scheduled_task import (
    ScheduledTaskDocument,
    external_daily_config_from_documents,
)


def _doc(**overrides: object) -> ScheduledTaskDocument:
    body: dict[str, object] = {
        "task_id": "task-1",
        "benchmark_name": "search_eval",
        "enabled": True,
        "judge_class": "search_eval_judge",
        "attribution_class": "no_attribution",
        "rubric_class": "search_eval_rubric",
    }
    body.update(overrides)
    return ScheduledTaskDocument.model_validate(body)


def test_daily_cron_maps_time_and_frequency() -> None:
    item = _doc(cron="5 2 * * *").to_benchmark_config()
    assert item is not None
    assert item.schedule_time == "02:05"
    assert item.schedule_frequency == "daily"


def test_weekly_cron_maps_weekday() -> None:
    # cron 1 = Monday; the schedule model 0 = Monday
    item = _doc(cron="0 3 * * 1").to_benchmark_config()
    assert item is not None
    assert item.schedule_frequency == "weekly"
    assert item.schedule_weekdays == [0]
    assert item.schedule_time == "03:00"


def test_monthly_cron_maps_days() -> None:
    item = _doc(cron="0 4 15 * *").to_benchmark_config()
    assert item is not None
    assert item.schedule_frequency == "monthly"
    assert item.schedule_month_days == [15]


def test_unsupported_cron_is_skipped() -> None:
    """A cron was written but cannot be recognized: it must not quietly run as the default daily."""
    assert _doc(cron="*/5 * * * *").to_benchmark_config() is None
    assert _doc(cron="0 2 * *").to_benchmark_config() is None


def test_profile_overrides_flat_fields() -> None:
    item = _doc(
        judge_class="flat_judge",
        profile={"judge_class": "profile_judge", "oss_prefix": "amap/x"},
    ).to_benchmark_config()
    assert item is not None
    assert item.judge_class == "profile_judge"
    assert item.oss_prefix == "amap/x"


def test_invalid_item_is_skipped() -> None:
    # rubric is allowed only for attribution_only (an illegal combination carried in the profile must be rejected)
    assert _doc(profile={"rubric": "r1", "mode": "judge_attribution"}).to_benchmark_config() is None


def test_config_from_documents_filters_disabled_and_marks_enabled() -> None:
    empty = external_daily_config_from_documents([])
    assert isinstance(empty, ExternalDailyConfig)
    assert empty.enabled is False

    config = external_daily_config_from_documents(
        [
            _doc(task_id="a", cron="0 2 * * *"),
            _doc(task_id="b", benchmark_name="other", enabled=False, cron="0 2 * * *"),
        ],
    )
    assert config.enabled is True
    assert [item.benchmark_name for item in config.benchmarks] == ["search_eval"]

    # Everything disabled → do not enable scheduling
    disabled = external_daily_config_from_documents(
        [_doc(enabled=False, cron="0 2 * * *")],
    )
    assert disabled.enabled is False
    assert disabled.benchmarks == []
