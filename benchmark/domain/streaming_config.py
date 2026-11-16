"""Declarative benchmark streaming config — RemoteConfig ``agent-eval.benchmark.streaming`` (§5.2).

Domain layer. Contains:
  - :class:`StreamingTaskDailyReportConfig` — task daily-report link config (Pydantic)
  - :class:`StreamingTaskConfig` — one streaming task model (Pydantic)
  - :class:`StreamingConfig` — top-level streaming config (Pydantic)
  - :func:`fetch_streaming_config` — domain service that loads config from RemoteConfig

The config is the **declarative source of truth** (§6.1): adding or removing topics,
and adding, deleting, pausing, resuming, or stopping tasks, all take effect through a
config publish. An in-process reconciler converges on a schedule.

RemoteConfig JSON shape (see ``docs/remote_config-streaming-config-example.json``)::

    {
        "enabled": true,
        "allow_empty_tasks": false,
        "reconcile_interval_s": 30,
        "tasks": [
            {
                "name": "amap_agent_daily",
                "topics": ["langfuse.records"],
                "group_id": null,
                "state": "running",
                "judge_class": "evaluator_service_judge",
                "judge_execution": "sync",
                "triage_class": "amap_agent_triage",
                "attribution_class": "amap_agent_attribution",
                "attribution_execution": "sync",
                "rubric_class": "amap_agent_kafka_rubric",
                "sampling": null,
                "daily_report": {"enabled": true}
            }
        ]
    }

**A parse failure keeps the last successful config** (§11): on a JSON / schema failure,
fall back to the in-process last-known-good and log an error. **Do not fall back to an
empty set** — a mistaken RemoteConfig wipe or bad config must not stop all consumption.
A real full stop must explicitly publish ``{"enabled": false, "allow_empty_tasks": true}``
(a second confirmation before converging to empty).

Usage::

    from benchmark.domain.streaming_config import (
        StreamingConfig,
        fetch_streaming_config,
    )

    cfg = fetch_streaming_config()
    if cfg is None:
        # No declarative config (unpublished / parse failed and no last-good) → env cold-start fallback.
"""

import threading

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, field_validator

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.settings.setting import settings

logger = get_logger(__name__)

#: RemoteConfig data_id for the declarative benchmark streaming config.
STREAMING_DATA_ID = "agent-eval.benchmark.streaming"
#: RemoteConfig group for the declarative benchmark streaming config.
STREAMING_GROUP = "agent-eval.service.group"

_DEFAULT_RECONCILE_INTERVAL_S = 30


def _normalize_optional_str(value: object) -> str | None:
    """``null`` / blank → ``None`` (unset)."
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class StreamingTaskDailyReportConfig(BaseModel):
    """Task daily-report link config (v4 ``t1_enabled``; link check only, does not drive the schedule, §7.2).

    Schedule time is decided by the trace-daily-report item's own ``schedule_time``.
    This section does not repeat ``schedule_time`` (avoids a second source of truth).
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False


class StreamingTaskConfig(BaseModel):
    """One streaming task config (a ``tasks[]`` entry).

    Attributes:
        name: Task name (registry primary key; by convention the same as the
            trace-daily-report item ``name``).
        topics: Subscribed topic set (adding or removing an element adds or removes
            that task's subscription, §6.3).
        group_id: Consumer group. ``null`` is allowed for **exactly one** migration
            task (= ``JUDGE_KAFKA_GROUP_ID``, so the cutover continues the offset).
            Every other task must set a distinct group (any membership change inside a
            group triggers a group-level rebalance, §6.3).
        state: ``running`` / ``paused`` / ``stopped`` (§6.2). Deleting the entry deletes
            the task. Illegal values are rejected by the reconciler's per-entry check.
        judge_class: Streaming judge alias (``STREAMING_JUDGE_CLASS_REFS`` allowlist).
            ``null`` = the current zero-code route.
        judge_execution: Task-level override ``sync`` / ``async``. ``null`` = the class
            declaration (default sync). May be combined with attribution_execution.
        triage_class: Streaming triage alias (``STREAMING_TRIAGE_CLASS_REFS``).
            ``null`` = look up classifiers_registry by ``evaluator_name``.
        attribution_class: ``null`` = the current synchronous triage attribution chain.
            ``no_attribution`` = off. Explicit aliases are on the allowlist
            (for example ``amap_agent_attribution`` / ``amap_agent_async_attribution``).
        attribution_execution: Task-level override ``sync`` / ``async`` / ``none``.
            ``null`` = the class declaration. Any combination with judge_execution is allowed.
        rubric_class: ``RUBRIC_CLASS_REFS`` allowlist alias (optional; a link-check key,
            not read at runtime — report scope follows the trace-daily-report item, §5.1).
        sampling: Sampling-rate override ``(0, 1]``. ``null`` = follow the RemoteConfig
            eval config (semantics at request time). An override diverges from the online
            config and is only for emergency mitigation. ``0`` is equivalent to pause;
            use ``state=paused`` instead.
        daily_report: Daily-report link (v4 ``t1_enabled``; link check only, §7.2).
    """

    model_config = {"extra": "ignore"}

    name: str = ""
    topics: list[str] = Field(default_factory=list)
    group_id: str | None = None
    state: str = "running"
    judge_class: str | None = None
    judge_execution: str | None = None
    triage_class: str | None = None
    attribution_class: str | None = None
    attribution_execution: str | None = None
    rubric_class: str | None = None
    sampling: float | None = None
    daily_report: StreamingTaskDailyReportConfig = Field(
        default_factory=StreamingTaskDailyReportConfig,
    )

    @field_validator("state", mode="before")
    @classmethod
    def _normalize_state(cls, value: object) -> str:
        """Normalize state: strip and lowercase; ``null`` → ``running``.

        Does not raise and does not reject illegal values in the domain layer (one bad
        item must not fail the whole parse). The reconciler's per-entry check skips an
        illegal state and logs an error.
        """
        if value is None:
            return "running"
        return str(value).strip().lower()

    @field_validator(
        "group_id",
        "judge_class",
        "judge_execution",
        "triage_class",
        "attribution_class",
        "attribution_execution",
        "rubric_class",
        mode="before",
    )
    @classmethod
    def _normalize_optional_fields(cls, value: object) -> str | None:
        """``null`` / blank → ``None``."""
        return _normalize_optional_str(value)

    @field_validator("judge_execution", "attribution_execution", mode="after")
    @classmethod
    def _normalize_execution_case(cls, value: str | None) -> str | None:
        """Lowercase an execution field (legal values are checked by the reconciler)."""
        return value.strip().lower() if value else None

    @field_validator("sampling", mode="before")
    @classmethod
    def _normalize_sampling(cls, value: object) -> float | None:
        """Normalize the sampling rate. An illegal value becomes ``None`` (no override) and warns.

        The range ``(0, 1]`` is rejected by the reconciler's per-entry check.
        """
        if value is None:
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning("invalid streaming task sampling %r, fallback to None", value)
            return None

    @field_validator("topics", mode="before")
    @classmethod
    def _normalize_topics(cls, value: object) -> list[str]:
        """Normalize topics: a non-list becomes an empty list (the reconciler rejects the entry)."""
        if not isinstance(value, (list, tuple)):
            return []
        return [str(item) for item in value]

    @field_validator("daily_report", mode="before")
    @classmethod
    def _normalize_daily_report(cls, value: object) -> StreamingTaskDailyReportConfig:
        """Normalize daily_report: ``null`` / non-dict → the default (not enabled)."""
        if isinstance(value, StreamingTaskDailyReportConfig):
            return value
        if isinstance(value, dict):
            return StreamingTaskDailyReportConfig.model_validate(value)
        return StreamingTaskDailyReportConfig()


class StreamingConfig(BaseModel):
    """Top-level streaming config.

    Attributes:
        enabled: Master switch (``false`` → the reconciler may converge to empty only
            with a second confirmation ``allow_empty_tasks=true``, §11).
        allow_empty_tasks: Empty-set protection (§11). When ``enabled=false`` or
            ``tasks=[]``, every instance is stopped only if this field is ``true``;
            otherwise the current state is kept and an error is logged.
        reconcile_interval_s: Reconcile period in seconds (≥ 10, default 30).
        tasks: Declarative task list.
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False
    allow_empty_tasks: bool = False
    reconcile_interval_s: int = _DEFAULT_RECONCILE_INTERVAL_S
    tasks: list[StreamingTaskConfig] = Field(default_factory=list)

    @field_validator("reconcile_interval_s", mode="before")
    @classmethod
    def _normalize_reconcile_interval(cls, value: object) -> int:
        """Normalize the reconcile period. An illegal value falls back to 30s and warns (floor clamped by the reconciler)."""
        if value is None:
            return _DEFAULT_RECONCILE_INTERVAL_S
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "invalid reconcile_interval_s %r, fallback to %d",
                value,
                _DEFAULT_RECONCILE_INTERVAL_S,
            )
            return _DEFAULT_RECONCILE_INTERVAL_S


# ── last-known-good (§11: keep the last success on parse failure, do not fall back to empty) ──
# One-element dict container: avoid a global statement (PLW0603), same pattern as _cache.
_last_good_cache: dict[str, StreamingConfig] = {}
_last_good_lock = threading.Lock()


def _get_last_good() -> StreamingConfig | None:
    with _last_good_lock:
        return _last_good_cache.get("cfg")


def _set_last_good(cfg: StreamingConfig) -> None:
    with _last_good_lock:
        _last_good_cache["cfg"] = cfg


def fetch_streaming_config(
    *,
    data_id: str | None = None,
    group: str | None = None,
    use_cache: bool = True,
) -> StreamingConfig | None:
    """Load and parse the declarative streaming config from RemoteConfig.

    Return semantics:

    - Parse succeeds → update last-known-good and return the config.
    - RemoteConfig unpublished / empty body → return last-good when one exists (so a
      mistaken delete does not stop everything), otherwise ``None`` (the reconciler
      cold-starts from env, §8).
    - Parse fails (JSON / schema) → **keep the last successful config** and log an error
      (§11, do not fall back to empty). Return ``None`` when there is no last-good.

    The raw body is TTL-cached by ``fetch_cached_settings`` (300s), so the declarative
    SLA is at most ~330s (§6.4). The parsed result is not cached again (validation is
    cheap, and last-good requires a real reparse each round).

    Args:
        data_id: RemoteConfig data_id, default read from settings.
        group: RemoteConfig group, default read from settings.
        use_cache: Whether to use the RemoteConfig raw cache, default ``True``.
    """
    data_id = data_id or settings.benchmark_streaming_remote_config_data_id
    group = group or settings.benchmark_streaming_remote_config_group

    raw = fetch_cached_settings(data_id, group, use_cache=use_cache)
    if not raw:
        last_good = _get_last_good()
        if last_good is not None:
            logger.warning(
                "RemoteConfig streaming config empty (%s/%s) — keeping last known good",
                data_id,
                group,
            )
        return last_good
    try:
        cfg = StreamingConfig.model_validate(raw)
    except Exception as exc:
        logger.error(
            "RemoteConfig streaming config parse failed (%s/%s): %s — keeping last known good",
            data_id,
            group,
            exc,
        )
        return _get_last_good()
    _set_last_good(cfg)
    return cfg


__all__ = [
    "STREAMING_DATA_ID",
    "STREAMING_GROUP",
    "StreamingConfig",
    "StreamingTaskConfig",
    "StreamingTaskDailyReportConfig",
    "_last_good_cache",
    "fetch_streaming_config",
]
