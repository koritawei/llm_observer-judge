"""Scheduled config for externally triggered benchmarks — runtime reads only the DB snapshot.

Domain layer. Contains:
  - :class:`ExternalDailyBenchmarkConfig` — one benchmark schedule item (Pydantic)
  - :class:`ExternalDailyConfig` — top-level schedule config (Pydantic)
  - :func:`fetch_external_daily_config` — read the in-process snapshot (does not read RemoteConfig)

RemoteConfig ``agent-eval.benchmark.external-daily`` is **fully shielded**: config now
comes from ``runtime_config_records`` ``kind=scheduled_task``
(:mod:`benchmark.domain.scheduled_task`), and a background job refreshes the snapshot every 30s.

JSON shape (the historical RemoteConfig document / scheduled-task profile are isomorphic;
see ``docs/remote_config-external-daily-config-example.json``)::

    {
        "enabled": true,
        "schedule_time": "02:00",
        "schedule_timezone": "Asia/Shanghai",
        "no_data_retry_interval_min": 30,
        "max_pull_attempts": 3,
        "benchmarks": [
            {
                "benchmark_name": "ieval_external",
                "dataset_name": "amap_chat_eval",
                "oss_prefix": "oss://agent-eval-oss/ieval-service/benchmark/benchmark_result",
                "judge_class": "ieval_external_judge",
                "attribution_class": "ieval_external_attribution",
                "rubric_class": "amap_chat_rubric_v2",
                "generate_report": true,
                "mode": "judge_attribution",
                "schedule_time": null,
                "schedule_frequency": "daily",
                "schedule_weekdays": null,
                "schedule_month_days": null,
                "no_data_cutoff": null,
                "limit": null,
                "run_id_suffix": null,
                "parse_interceptor_ref": null,
            }
        ]
    }

``judge_class`` / ``attribution_class`` / ``rubric_class`` are stable aliases registered
on a server allowlist. execution/provider/target are declared by the execution class,
and criteria/aggregation/interceptors are declared by the Rubric class. RemoteConfig no
longer composes them. The old ``rubric_ref`` is rejected outright.

``schedule_time`` / ``schedule_timezone`` may be overridden per benchmark (empty = inherit
the top level). ``no_data_cutoff`` is benchmark-level only: when unset, a missing OSS
object is probed once more after 30 minutes and then stops; when set, it is probed every
30 minutes until the cutoff, and a timeout is marked failed (reason "startup failed").
Use it when benchmarks trigger at different times, for example search_eval depends on an
upstream ODPS export and must trigger after that export's schedule.

``schedule_frequency``: ``daily`` (every day, the default) / ``weekly`` (with
``schedule_weekdays``, ``0`` = Monday … ``6`` = Sunday, default ``[0]``) / ``monthly``
(with ``schedule_month_days``, 1-31, default ``[1]``; a day that does not exist in the
month is skipped). Frequency is a benchmark-level setting. Trigger-day semantics are
unchanged: each trigger still uses that day's data date (``yyyymmdd``).

Usage::

    from benchmark.domain.external_daily_config import (
        ExternalDailyConfig,
        fetch_external_daily_config,
    )

    cfg = fetch_external_daily_config()
    if not cfg.enabled:
        # scheduler disabled
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from benchmark.domain.runtime_snapshot import current_external_daily_config

logger = get_logger(__name__)

#: Shielded RemoteConfig schedule data_id (historical docs / import scripts only; not read at runtime).
EXTERNAL_DAILY_DATA_ID = "agent-eval.benchmark.external-daily"
#: Shielded RemoteConfig schedule group.
EXTERNAL_DAILY_GROUP = "agent-eval.service.group"

_DEFAULT_SCHEDULE_TIME = "02:00"
_DEFAULT_SCHEDULE_TIMEZONE = "Asia/Shanghai"
_DEFAULT_NO_DATA_RETRY_INTERVAL_MIN = 30
_DEFAULT_MAX_PULL_ATTEMPTS = 3
#: Default attribution provider: local attribution (example agent).
_DEFAULT_GENERATE_REPORT = True
_DEFAULT_MODE = "judge_attribution"
#: Default schedule frequency: every day.
_DEFAULT_SCHEDULE_FREQUENCY = "daily"
#: Legal schedule-frequency values.
_SCHEDULE_FREQUENCIES = ("daily", "weekly", "monthly")

#: Default map for top-level schedule fields (normalize RemoteConfig ``null`` / blank so parse does not fail).
_TOP_LEVEL_STR_DEFAULTS: dict[str, str] = {
    "schedule_time": _DEFAULT_SCHEDULE_TIME,
    "schedule_timezone": _DEFAULT_SCHEDULE_TIMEZONE,
}


def _normalize_int_list(
    value: object,
    *,
    lo: int,
    hi: int,
    field: str,
) -> list[int] | None:
    """Normalize an int list: drop out-of-range / non-numeric items and warn. Empty becomes ``None`` (default)."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        logger.warning("invalid %s %r (not a list), fallback to default", field, value)
        return None
    items: list[int] = []
    for item in value:
        try:
            num = int(item)
        except (TypeError, ValueError):
            logger.warning("invalid %s item %r (not int), skipped", field, item)
            continue
        if lo <= num <= hi:
            items.append(num)
        else:
            logger.warning(
                "invalid %s item %r (out of %d-%d), skipped",
                field,
                item,
                lo,
                hi,
            )
    return items or None


class ExternalDailyBenchmarkConfig(BaseModel):
    """Schedule item for one benchmark.

    Attributes:
        benchmark_name: Registered benchmark name (used to look up BENCHMARK_SERVICES_CLASSES).
        dataset_name: Dataset name.
        oss_prefix: OSS data-file prefix (``oss://{bucket}/{key_path}``). Empty uses
            :func:`benchmark.domain.report_storage_config.get_default_download_prefix`.
        judge_class: Stable alias of the Judge execution class.
        attribution_class: Stable alias of the Attribution execution class.
        rubric_class: Stable alias of the Rubric class (``component_rubric`` = the component metadata rubric).
        generate_report: Whether to generate an HTML report, default ``True``.
        mode: Eval mode, default ``"judge_attribution"``.
        run_id_suffix: run_id suffix (several runs of one benchmark on one day), optional.
        rubric: Rubric config id, optional.
        schedule_time: Fixed trigger time ``HH:MM``; ``null`` / empty = inherit the top level.
        schedule_timezone: Trigger timezone (IANA); ``null`` / empty = inherit the top level.
        schedule_frequency: ``daily`` / ``weekly`` / ``monthly``, default ``daily``.
            An illegal value falls back to ``daily`` and warns.
        schedule_weekdays: Used when ``weekly``: weekdays to trigger
            (``0`` = Monday … ``6`` = Sunday). Empty = ``[0]`` (every Monday).
        schedule_month_days: Used when ``monthly``: days of month to trigger (1-31).
            Empty = ``[1]`` (the 1st). A day that does not exist (for example Feb 31) is skipped.
        no_data_cutoff: Retry-until time ``HH:MM`` when there is no data. ``null`` / empty =
            unset (probe once more after 30 minutes, then stop).
        limit: Sample count (passed through as ``ExternalBenchmarkRequest.limit``).
            ``None`` = the full set.
    """

    model_config = {"extra": "ignore"}

    benchmark_name: str = ""
    judge_class: str = ""
    attribution_class: str = ""
    rubric_class: str = ""
    dataset_name: str = ""
    oss_prefix: str = ""
    generate_report: bool = _DEFAULT_GENERATE_REPORT
    mode: str = _DEFAULT_MODE
    run_id_suffix: str | None = None
    rubric: str | None = None
    schedule_time: str | None = None
    schedule_timezone: str | None = None
    schedule_frequency: str = _DEFAULT_SCHEDULE_FREQUENCY
    schedule_weekdays: list[int] | None = None
    schedule_month_days: list[int] | None = None
    no_data_cutoff: str | None = None
    limit: int | None = None

    @field_validator("schedule_frequency", mode="before")
    @classmethod
    def _normalize_schedule_frequency(cls, value: object) -> str:
        """Normalize frequency: strip and lowercase. An illegal value falls back to ``daily`` and warns.

        Does not raise: one bad frequency must not fail the whole external-daily parse
        (a RemoteConfig parse failure means ``enabled=False`` and nothing runs).
        """
        if value is None:
            return _DEFAULT_SCHEDULE_FREQUENCY
        text = str(value).strip().lower()
        if text not in _SCHEDULE_FREQUENCIES:
            logger.warning(
                "invalid schedule_frequency %r, fallback to %s",
                value,
                _DEFAULT_SCHEDULE_FREQUENCY,
            )
            return _DEFAULT_SCHEDULE_FREQUENCY
        return text

    @field_validator("schedule_weekdays", mode="before")
    @classmethod
    def _normalize_schedule_weekdays(cls, value: object) -> list[int] | None:
        """Drop illegal weekdays (legal 0-6). An empty list becomes ``None`` (take the default)."""
        return _normalize_int_list(value, lo=0, hi=6, field="schedule_weekdays")

    @field_validator("schedule_month_days", mode="before")
    @classmethod
    def _normalize_schedule_month_days(cls, value: object) -> list[int] | None:
        """Drop illegal month days (legal 1-31). An empty list becomes ``None`` (take the default)."""
        return _normalize_int_list(value, lo=1, hi=31, field="schedule_month_days")

    @field_validator("no_data_cutoff", mode="before")
    @classmethod
    def _normalize_optional_cutoff(cls, value: object) -> str | None:
        """``null`` / blank → unset (probe once more after 30 minutes, then stop)."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    # ── Optional configured-runtime extensions ──
    parse_interceptor_ref: str | None = None
    """Data-parse interceptor ref ``module:Class`` (default = the value mounted on the Rubric class)."""

    # ── case_id normalization (D5) ──
    require_case_id: bool = False
    """Whether a row missing every case_id source column (case_id/id/task_id) fails at parse (default false)."""

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_execution_config(cls, value: object) -> object:
        """The old execution/provider object and rubric_ref are no longer accepted, so two config truths cannot coexist."""
        if isinstance(value, dict):
            legacy = {"judge", "attribution", "attribution_provider", "rubric_ref"} & value.keys()
            if legacy:
                fields = ", ".join(sorted(legacy))
                raise ValueError(
                    f"legacy execution config is not supported: {fields}; "
                    "use judge_class/attribution_class/rubric_class",
                )
        return value

    @model_validator(mode="after")
    def _rubric_only_attribution_only(self) -> "ExternalDailyBenchmarkConfig":
        """A ``rubric`` config is allowed only for ``attribution_only`` (D9 tightening, same rule as the API)."""
        if self.rubric and self.mode != "attribution_only":
            raise ValueError(
                "rubric is only allowed in attribution_only mode; "
                "use rubric_class for judge_attribution mode",
            )
        return self


class ExternalDailyConfig(BaseModel):
    """Top-level schedule config.

    Attributes:
        enabled: Whether the schedule is enabled, default ``False``.
        schedule_time: Fixed trigger time ``HH:MM``, default ``"02:00"``.
        schedule_timezone: Schedule timezone, default ``"Asia/Shanghai"``.
        no_data_retry_interval_min: Compatibility field (the scheduler probes on a fixed 30-minute interval).
        max_pull_attempts: Max pull retries, default 3.
        benchmarks: Scheduled benchmark list.

    ``schedule_time`` / ``schedule_timezone`` tolerate RemoteConfig ``null`` / blank
    (normalized to the default). The top-level ``no_data_cutoff`` has been removed
    (``extra=ignore`` keeps old JSON working). Cutoff is written only on
    ``benchmarks[].no_data_cutoff``.
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False
    schedule_time: str = _DEFAULT_SCHEDULE_TIME
    schedule_timezone: str = _DEFAULT_SCHEDULE_TIMEZONE
    no_data_retry_interval_min: int = _DEFAULT_NO_DATA_RETRY_INTERVAL_MIN
    max_pull_attempts: int = _DEFAULT_MAX_PULL_ATTEMPTS
    benchmarks: list[ExternalDailyBenchmarkConfig] = Field(default_factory=list)

    @field_validator("schedule_time", "schedule_timezone", mode="before")
    @classmethod
    def _normalize_top_level_str(cls, value: object, info: ValidationInfo) -> str:
        """Top-level schedule fields: ``null`` / blank normalize to the default (RemoteConfig often uses null as a placeholder)."""
        if value is None:
            return _TOP_LEVEL_STR_DEFAULTS[info.field_name]
        text = str(value).strip()
        return text or _TOP_LEVEL_STR_DEFAULTS[info.field_name]


def fetch_external_daily_config(
    *,
    use_cache: bool = True,
) -> ExternalDailyConfig:
    """Return the DB scheduled-task snapshot (``kind=scheduled_task``).

    RemoteConfig ``agent-eval.benchmark.external-daily`` is **fully shielded**: this
    function no longer reads RemoteConfig. The snapshot is maintained by the background
    refresh (``refresh_external_daily_snapshot``). When no task is enabled, return an
    empty config with ``enabled=False`` (the scheduler does not run).

    Args:
        use_cache: Kept for old callers. It does not affect the read (the snapshot is refreshed in the background).
    """
    del use_cache
    snapshot = current_external_daily_config()
    return snapshot if isinstance(snapshot, ExternalDailyConfig) else ExternalDailyConfig()


__all__ = [
    "EXTERNAL_DAILY_DATA_ID",
    "EXTERNAL_DAILY_GROUP",
    "ExternalDailyBenchmarkConfig",
    "ExternalDailyConfig",
    "fetch_external_daily_config",
]
