"""The DailyExternalBenchmarkScheduler constructor signature and DailySchedulerOptions assembly."""

from unittest.mock import MagicMock

from benchmark.domain.external_daily_config import ExternalDailyConfig
from benchmark.workers.daily_scheduler import (
    DailyExternalBenchmarkScheduler,
    DailySchedulerOptions,
)


def test_accepts_benchmark_registry_via_options() -> None:
    runner = MagicMock()
    registry = object()
    scheduler = DailyExternalBenchmarkScheduler(
        runner=runner,  # type: ignore[arg-type]
        config=ExternalDailyConfig(enabled=False),
        options=DailySchedulerOptions(benchmark_registry=registry),
    )
    assert scheduler._benchmark_registry is registry


def test_rejects_legacy_benchmark_registry_keyword() -> None:
    runner = MagicMock()
    try:
        DailyExternalBenchmarkScheduler(
            runner=runner,  # type: ignore[arg-type]
            config=ExternalDailyConfig(enabled=False),
            benchmark_registry=object(),  # type: ignore[call-arg]
        )
    except TypeError as exc:
        assert "benchmark_registry" in str(exc)
    else:
        raise AssertionError("expected TypeError for legacy keyword")
