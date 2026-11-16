"""Scheduled-task document (``runtime_config_records`` kind=``scheduled_task``).

At runtime RemoteConfig ``agent-eval.benchmark.external-daily`` is **fully ignored**:
batch schedule items are read only from here (DB row → :class:`ExternalDailyConfig` →
``DailyExternalBenchmarkScheduler``). The document shape matches design §2.2: ``profile``
holds the profile JSON, and the other flat fields are isomorphic to
``external-daily.benchmarks[]``. ``cron`` is a 5-field expression in UTC+8 and supports
the three common daily shapes ``M H * * *`` / ``M H * * W`` / ``M H D * *``.
"""

import re
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, field_validator

from benchmark.domain.external_daily_config import (
    ExternalDailyBenchmarkConfig,
    ExternalDailyConfig,
)

logger = get_logger(__name__)

_CRON_FIELDS = 5
_CRON_RE = re.compile(r"^\s*(\d{1,2})\s+(\d{1,2})\s+(\S+)\s+\*\s+(\S+)\s*$")
_MAX_MONTH_DAY = 31
_CRON_WEEKDAYS = {
    "0": 6,
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
}


class ScheduledTaskDocument(BaseModel):
    """One page scheduled task (kind=scheduled_task).

    Attributes:
        task_id: Task id (record_key suffix).
        task_name: Display name.
        benchmark_name: Execution-side benchmark name (run_id = ``{name}-{yyyymmdd}``).
        enabled: Whether it is enabled; create defaults to false, the page enables it.
        cron: 5-field cron in UTC+8 (``M H * * *`` / ``M H * * W`` / ``M H D * *``).
        profile: Profile JSON (isomorphic to external-daily ``benchmarks[]``, preferred
            over the flat fields).
        schedule_time / schedule_timezone / schedule_frequency / schedule_weekdays /
        schedule_month_days / no_data_cutoff / oss_prefix / dataset_name /
        judge_class / attribution_class / rubric_class / mode / run_id_suffix /
        limit / require_case_id: Flat fields (used when cron does not cover them).
    """

    model_config = {"extra": "ignore"}

    task_id: str = ""
    task_name: str = ""
    benchmark_name: str = ""
    enabled: bool = False
    cron: str = ""
    profile: dict[str, Any] = Field(default_factory=dict)
    dataset_name: str = ""
    oss_prefix: str = ""
    judge_class: str = ""
    attribution_class: str = ""
    rubric_class: str = ""
    mode: str = ""
    run_id_suffix: str | None = None
    schedule_time: str | None = None
    schedule_timezone: str | None = None
    schedule_frequency: str = ""
    schedule_weekdays: list[int] | None = None
    schedule_month_days: list[int] | None = None
    no_data_cutoff: str | None = None
    limit: int | None = None
    require_case_id: bool | None = None

    @field_validator("profile", mode="before")
    @classmethod
    def _profile_must_be_dict(cls, value: object) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def to_benchmark_config(self) -> ExternalDailyBenchmarkConfig | None:
        """Map to a schedule item. ``None`` when cron is unrecognized or a required field is missing."""
        body: dict[str, Any] = self._flat_body()
        body.update(self.profile)
        body.setdefault("benchmark_name", self.benchmark_name.strip())
        schedule = self._schedule_from_cron()
        if (self.cron or "").strip() and schedule is None:
            # cron was set but cannot be recognized: skip the task, do not silently run as daily.
            return None
        if schedule is not None:
            body.setdefault("schedule_time", schedule[0])
            body.setdefault("schedule_frequency", schedule[1])
            body.setdefault("schedule_weekdays", schedule[2])
            body.setdefault("schedule_month_days", schedule[3])
        try:
            return ExternalDailyBenchmarkConfig.model_validate(body)
        except Exception:
            logger.warning(
                "scheduled_task.invalid task_id=%r benchmark=%r — skipped",
                self.task_id,
                self.benchmark_name,
                exc_info=True,
            )
            return None

    def _flat_body(self) -> dict[str, Any]:
        """Non-empty flat fields → a profile-shaped dict (profile wins, so these are defaults)."""
        candidates = {
            "benchmark_name": self.benchmark_name,
            "dataset_name": self.dataset_name,
            "oss_prefix": self.oss_prefix,
            "judge_class": self.judge_class,
            "attribution_class": self.attribution_class,
            "rubric_class": self.rubric_class,
            "mode": self.mode,
            "run_id_suffix": self.run_id_suffix,
            "schedule_time": self.schedule_time,
            "schedule_timezone": self.schedule_timezone,
            "schedule_frequency": self.schedule_frequency,
            "schedule_weekdays": self.schedule_weekdays,
            "schedule_month_days": self.schedule_month_days,
            "no_data_cutoff": self.no_data_cutoff,
            "limit": self.limit,
            "require_case_id": self.require_case_id,
        }
        return {
            key: value
            for key, value in candidates.items()
            if value is not None and value not in ("", [])
        }

    def _schedule_from_cron(self) -> tuple[str, str, list[int] | None, list[int] | None] | None:
        """UTC+8 cron → (schedule_time, frequency, weekdays, month_days).

        Only the three common daily shapes are supported. Unrecognized input returns
        ``None`` (the caller skips the task).
        """
        text = (self.cron or "").strip()
        if not text:
            return None
        matched = _CRON_RE.match(text) if len(text.split()) == _CRON_FIELDS else None
        result: tuple[str, str, list[int] | None, list[int] | None] | None = None
        if matched is None:
            self._log_unsupported_cron(text)
        else:
            minute, hour, dom, dow = matched.groups()
            schedule_time = f"{int(hour):02d}:{int(minute):02d}"
            if dow != "*":
                weekday = _CRON_WEEKDAYS.get(dow)
                if weekday is None:
                    self._log_unsupported_cron(text)
                else:
                    result = (schedule_time, "weekly", [weekday], None)
            elif dom != "*":
                if dom.isdigit() and 1 <= int(dom) <= _MAX_MONTH_DAY:
                    result = (schedule_time, "monthly", None, [int(dom)])
                else:
                    self._log_unsupported_cron(text)
            else:
                result = (schedule_time, "daily", None, None)
        return result

    def _log_unsupported_cron(self, text: str) -> None:
        logger.warning(
            "scheduled_task.unsupported_cron task_id=%r cron=%r "
            "（仅支持 'M H * * *' / 'M H * * W' / 'M H D * *'，5 段）",
            self.task_id,
            text,
        )


def external_daily_config_from_documents(
    documents: list[ScheduledTaskDocument],
) -> ExternalDailyConfig:
    """DB scheduled-task documents → ``ExternalDailyConfig`` (disabled / invalid items skipped).

    When no task is enabled, return an empty config with ``enabled=False``. The scheduler
    then logs a single disabled line.
    """
    benchmarks: list[ExternalDailyBenchmarkConfig] = []
    for document in documents:
        if not document.enabled:
            continue
        item = document.to_benchmark_config()
        if item is not None:
            benchmarks.append(item)
    return ExternalDailyConfig(enabled=bool(benchmarks), benchmarks=benchmarks)


__all__ = ["ScheduledTaskDocument", "external_daily_config_from_documents"]
