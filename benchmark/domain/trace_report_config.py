"""Trace pipeline daily report (T+1) schedule config — fetched from RemoteConfig, not stored in the DB.

Domain layer. It contains:
  - :class:`TraceReportItemConfig` — one report schedule item (Pydantic)
  - :class:`TraceReportConfig` — top-level schedule config (Pydantic)
  - :func:`fetch_trace_report_config` — domain service that fetches the schedule from RemoteConfig

Unlike ``external_daily_config``, this path does **not** pull OSS data files and does **not**
run judge/attribution. It aggregates existing ``trace_pipeline_records`` eval results by
condition and reuses the shared report flow. The data date is fixed to **T-1** of the trigger
day (interpreted in ``schedule_timezone``) and is not configurable.

RemoteConfig JSON shape (see
``docs/remote_config-trace-daily-report-config-example.json``)::

    {
        "enabled": true,
        "schedule_time": "09:00",
        "schedule_timezone": "Asia/Shanghai",
        "reports": [
            {
                "name": "globalvoice_daily",
                "benchmark_name": "example",
                "rubric_class": "amap_agent_kafka_rubric",
                "source": "kafka",
                "agent_name": "amap_chat",
                "project_id": "cmpwln0qs04fo6j8eryfdgznb",
                "sample_size": null,
                "generate_report": true,
                "report_html": null,
                "rule_link_url": null,
                "schedule_time": null,
                "schedule_timezone": null
            }
        ]
    }

``name`` is the report namespace: run_id = ``{name}-{yyyymmdd(T-1)}``. Daily trends and
day-over-day comparison group by that name (fully isolated from existing benchmark trends).
``benchmark_name`` is only used to resolve rubric / aggregation (reuse registered component
metadata) and does not trigger any benchmark run.

``source`` values: ``kafka`` / ``metaq`` / ``http`` / ``sidecar`` / ``benchmark``
(exact match on ``trace_meta_json.source``) or ``dataset`` (matches the compound prefix
``dataset-*``; see :func:`benchmark.domain.enums.is_dataset_pipeline_source`).

Usage::

    from benchmark.domain.trace_report_config import (
        TraceReportConfig,
        fetch_trace_report_config,
    )

    cfg = fetch_trace_report_config()
    if not cfg.enabled:
        # scheduler disabled
"""

from dataclasses import dataclass

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.settings.setting import settings
from benchmark.domain._cache import get_or_compute


# Must bind before the RemoteConfig import: schema refers back to this function in the config load chain.
def normalize_report_html_name(value: object) -> str | None:
    """A bare html filename under ``templates/``. A path, blank, or illegal value returns ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if (
        "/" in text
        or "\\" in text
        or text.startswith(".")
        or ".." in text
        or not text.endswith(".html")
    ):
        return None
    return text


logger = get_logger(__name__)

#: RemoteConfig data_id for the trace daily-report schedule.
TRACE_DAILY_REPORT_DATA_ID = "agent-eval.benchmark.trace-daily-report"
#: RemoteConfig group for the trace daily-report schedule.
TRACE_DAILY_REPORT_GROUP = "agent-eval.service.group"

#: Default clock time for auto report generation (when neither the item nor the top level sets one).
DEFAULT_TRACE_REPORT_SCHEDULE_TIME = "09:00"
#: Origin tag for a default-registered report item. A user override (same-name RemoteConfig item) wins by this tag.
TRACE_REPORT_ORIGIN_AUTO_DEFAULT = "auto_default"
_DEFAULT_SCHEDULE_TIME = DEFAULT_TRACE_REPORT_SCHEDULE_TIME
_DEFAULT_SCHEDULE_TIMEZONE = "Asia/Shanghai"
_DEFAULT_SOURCE = "kafka"
_DEFAULT_GENERATE_REPORT = True

#: Legal data-source values (``dataset`` matches a compound prefix; the rest are exact).
_REPORT_SOURCES = ("kafka", "metaq", "http", "sidecar", "benchmark", "dataset")

#: Default map for top-level schedule fields (RemoteConfig ``null`` / blank is normalized so parse does not fail).
_TOP_LEVEL_STR_DEFAULTS: dict[str, str] = {
    "schedule_time": _DEFAULT_SCHEDULE_TIME,
    "schedule_timezone": _DEFAULT_SCHEDULE_TIMEZONE,
}


class TraceReportItemConfig(BaseModel):
    """One trace report schedule item.

    Attributes:
        name: Report namespace (run_id prefix and daily-trend group name). Required.
        benchmark_name: Registered benchmark name used as the rubric / aggregation source
            (looked up in ``BENCHMARK_SERVICES_CLASSES``; does not run a benchmark).
        rubric_class: Stable Rubric class alias (``RUBRIC_CLASS_REFS`` allowlist).
            When non-empty it **wins over** ``benchmark_name`` — rubric / aggregation come
            straight from the Rubric class, and ``benchmark_name`` may be empty.
        source: Data-source channel, default ``kafka``. ``dataset`` matches a compound prefix.
        agent_name: Filter on ``trace_pipeline_records.agent_name``. ``null`` = no filter.
        project_id: Filter on ``trace_pipeline_records.project_id``. ``null`` = no filter.
        sample_size: Sample cap (latest N rows in the window). ``None`` = all rows.
        generate_report: Whether to generate an HTML report, default ``True``.
        rule_link_url: External link for the report scoring-rules slot. ``null`` = keep the slot as plain text.
        report_html: Report HTML filename under ``templates/``. ``null`` / blank / illegal
            uses the default ``benchmark_report.html``.
        schedule_time: Clock trigger ``HH:MM``. ``null`` / empty inherits the top level.
        schedule_timezone: Trigger timezone (IANA). ``null`` / empty inherits the top level.
        origin: Origin tag. ``auto_default`` = system default registration (a same-name user
            override wins). Anything else, or empty, is a user or page registration (it wins on a name clash).
        agent_registration_id: Business-custom registration id. Written when the example dual chain
            reports per registration. Sampling, backfill, and call counts use the same scope.
        stream_instance_id: Live-stream instance id. When set, sampling uses only that stream's
            stored key (``stem[@reg]@stream``) so multiple data sources do not mix into one report.
        judge_content_version: Filter on the judge prompt ``content_version``
            (string form aligned with the stored lineage). When set, only that version's traces are sampled.
    """

    model_config = {"extra": "ignore"}

    name: str = ""
    benchmark_name: str = ""
    rubric_class: str | None = None
    source: str = _DEFAULT_SOURCE
    agent_name: str | None = None
    project_id: str | None = None
    sample_size: int | None = None
    generate_report: bool = _DEFAULT_GENERATE_REPORT
    origin: str | None = None
    rule_link_url: str | None = None
    report_html: str | None = None
    schedule_time: str | None = None
    schedule_timezone: str | None = None
    agent_registration_id: str | None = None
    stream_instance_id: str | None = None
    judge_content_version: str | None = None

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: object) -> str:
        """Normalize source: strip whitespace and lowercase. An illegal value falls back to ``kafka`` and warns.

        Does not raise: a bad source on one item must not fail the whole trace-daily-report parse
        (a RemoteConfig parse failure means ``enabled=False`` and nothing runs).
        """
        if value is None:
            return _DEFAULT_SOURCE
        text = str(value).strip().lower()
        if text not in _REPORT_SOURCES:
            logger.warning(
                "invalid trace report source %r, fallback to %s",
                value,
                _DEFAULT_SOURCE,
            )
            return _DEFAULT_SOURCE
        return text

    @field_validator("benchmark_name", mode="before")
    @classmethod
    def _normalize_benchmark_name(cls, value: object) -> str:
        """Normalize ``null`` / blank to ``""`` (empty is allowed in rubric_class mode, and matches
        the common RemoteConfig null placeholder)."""
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("report_html", mode="before")
    @classmethod
    def _normalize_report_html(cls, value: object) -> str | None:
        """Accept only a bare filename under ``templates/``. Anything else is unset (the default template)."""
        name = normalize_report_html_name(value)
        if value not in (None, "") and name is None and str(value).strip():
            logger.warning("invalid report_html %r, fallback to default template", value)
        return name

    @field_validator(
        "agent_name",
        "project_id",
        "rule_link_url",
        "rubric_class",
        "origin",
        "agent_registration_id",
        "stream_instance_id",
        "judge_content_version",
        mode="before",
    )
    @classmethod
    def _normalize_optional_str(cls, value: object) -> str | None:
        """``null`` / blank → ``None`` (no filter / not configured)."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("sample_size", mode="before")
    @classmethod
    def _normalize_sample_size(cls, value: object) -> int | None:
        """Normalize the sample cap: a non-positive integer or illegal value becomes ``None`` (all rows)."""
        if value is None:
            return None
        try:
            num = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning("invalid sample_size %r (not int), fallback to full scan", value)
            return None
        if num <= 0:
            logger.warning("invalid sample_size %r (<= 0), fallback to full scan", value)
            return None
        return num


class TraceReportConfig(BaseModel):
    """Top-level schedule config for the trace daily report.

    Attributes:
        enabled: Whether the schedule is on, default ``False``.
        schedule_time: Clock trigger ``HH:MM``, default ``"09:00"``.
        schedule_timezone: Schedule timezone, default ``"Asia/Shanghai"``.
        reports: Scheduled report list.

    ``schedule_time`` / ``schedule_timezone`` tolerate RemoteConfig ``null`` / blank
    (normalized to the default). The data date is not configurable — it is always T-1 of the trigger day.
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False
    schedule_time: str = _DEFAULT_SCHEDULE_TIME
    schedule_timezone: str = _DEFAULT_SCHEDULE_TIMEZONE
    reports: list[TraceReportItemConfig] = Field(default_factory=list)

    @field_validator("schedule_time", "schedule_timezone", mode="before")
    @classmethod
    def _normalize_top_level_str(cls, value: object, info: ValidationInfo) -> str:
        """Top-level schedule field: ``null`` / blank is normalized to the default (the common RemoteConfig null placeholder)."""
        if value is None:
            return _TOP_LEVEL_STR_DEFAULTS[info.field_name]
        text = str(value).strip()
        return text or _TOP_LEVEL_STR_DEFAULTS[info.field_name]


def fetch_trace_report_config(
    *,
    data_id: str | None = None,
    group: str | None = None,
    use_cache: bool = True,
) -> TraceReportConfig:
    """Fetch and parse the trace daily-report schedule from RemoteConfig. Unavailable RemoteConfig returns defaults.

    The parsed result is cached by default (TTL 300s; see :mod:`benchmark.domain._cache`).
    Pass ``use_cache=False`` to force a refresh.

    Args:
        data_id: RemoteConfig data_id, default from settings.
        group: RemoteConfig group, default from settings.
        use_cache: Whether to use the cache, default ``True``.
    """
    data_id = data_id or settings.trace_daily_report_remote_config_data_id
    group = group or settings.trace_daily_report_remote_config_group

    def _load_and_parse() -> TraceReportConfig:
        raw = fetch_cached_settings(data_id, group, use_cache=use_cache)
        if not raw:
            return TraceReportConfig()
        try:
            return TraceReportConfig.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "RemoteConfig trace-daily-report config parse failed (%s/%s): %s",
                data_id,
                group,
                exc,
            )
            return TraceReportConfig()

    return get_or_compute(
        (data_id, group),
        _load_and_parse,
        use_cache=use_cache,
    )


def benchmark_name_for_evaluator(evaluator_name: str) -> str | None:
    """Evaluator stem → registered benchmark name (reverse lookup of the unified agent name).

    Used when a default report is registered for an eval item that has no benchmark context,
    so the rubric source can be located. Returns ``None`` when there is no benchmark
    (evaluator-only and no component). The caller skips registration — that agent has no
    rubric source, so a registered item still would not resolve a report definition.
    """
    from benchmark.runs.common.agent_catalog import (  # noqa: PLC0415
        canonical_agent_name,
    )
    from benchmark.runs.registry import (  # noqa: PLC0415
        BENCHMARK_SERVICES_CLASSES,
    )

    stem = evaluator_name.strip()
    for cls in BENCHMARK_SERVICES_CLASSES:
        try:
            meta = cls.get_metadata()
        except RuntimeError:
            continue
        if canonical_agent_name(meta.name) == stem:
            return meta.name
    return None


#: Report-name separator between dataset and registration id (``{dataset}__reg__{registrationId}``).
TRACE_REPORT_REGISTRATION_NAME_SEP = "__reg__"

#: Live-stream instance timestamp / id separator (``…__ts__{streamInstanceId}``). Same-day registrations stay distinct.
TRACE_REPORT_STREAM_NAME_SEP = "__ts__"

#: T+1 report-item namespace key in run.config_json (``{dataset}``, or with ``__reg__`` / ``__ts__``).
#: Matches the ``run_id`` prefix so trends group by day. The ``benchmark_name`` column stores the internal agent name.
TRACE_REPORT_NAMESPACE_CONFIG_KEY = "trace_report_namespace"

#: Legacy daily namespace / column name for amap live T+1 before registration isolation (``amap_agent_daily-YYYYMMDD``).
AMAP_AGENT_LEGACY_DAILY_NAMESPACE = "amap_agent_daily"

#: Only this logical component folds the legacy ``amap_agent_daily`` into trends and the task-page daily list.
_AMAP_AGENT_BENCHMARK_LOGICAL = "amap_agent_benchmark"


def amap_agent_legacy_daily_namespaces(logical_benchmark_name: str) -> tuple[str, ...]:
    """``amap_agent_benchmark`` (including ``@reg``) → the legacy daily namespace. Other components return empty.

    The new run_id / column name does not match legacy ``amap_agent_daily``. Trends and the task
    page must fall back to that namespace to join history. It applies **only** to amap_agent_benchmark.
    """
    from benchmark.domain.eval_config import benchmark_run_id_stem  # noqa: PLC0415

    stem = benchmark_run_id_stem(logical_benchmark_name)
    if stem != _AMAP_AGENT_BENCHMARK_LOGICAL:
        return ()
    return (AMAP_AGENT_LEGACY_DAILY_NAMESPACE,)


def versioned_trace_report_name(
    dataset_name: str,
    registration_id: str = "",
    stream_instance_id: str = "",
) -> str:
    """T+1 report namespace isolated by registration / live-stream instance.

    ``{dataset}[__reg__{registrationId}][__ts__{streamInstanceId}]``.
    """
    name = dataset_name.strip()
    if not name:
        return name
    reg = registration_id.strip()
    stream = stream_instance_id.strip()
    if reg:
        name = f"{name}{TRACE_REPORT_REGISTRATION_NAME_SEP}{reg}"
    if stream:
        name = f"{name}{TRACE_REPORT_STREAM_NAME_SEP}{stream}"
    return name


def registered_dataset_name_from_report_name(report_name: str) -> str:
    """Recover the page-registered dataset name from a report item name (drop ``__reg__`` / ``__ts__`` suffixes)."""
    text = report_name.strip()
    for sep in (TRACE_REPORT_STREAM_NAME_SEP, TRACE_REPORT_REGISTRATION_NAME_SEP):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text


@dataclass(frozen=True)
class TraceReportSeed:
    """Minimum description for registering a default T+1 report item (the identity tuple of one registration).

    ``source`` matches the stored ``trace_meta_json.source`` values (see the module header).
    Every other display / schedule field uses the :class:`TraceReportItemConfig` default.
    When ``agent_registration_id`` is set, the namespace and sample version are isolated per registration.
    """

    name: str
    benchmark_name: str
    source: str
    agent_name: str | None = None
    project_id: str | None = None
    agent_registration_id: str | None = None
    stream_instance_id: str | None = None
    judge_content_version: str | None = None


async def ensure_default_trace_report(seed: TraceReportSeed) -> str:
    """Idempotently register one default T+1 report item and return the namespace actually used.

    The only registration path for a live stream's default report. It writes only required fields
    (name / benchmark_name / source / project_id). The rest use :class:`TraceReportItemConfig`
    defaults (09:00, Asia/Shanghai, full sample, generate HTML, default template) and the
    ``origin=auto_default`` tag.

    With ``agent_registration_id`` / ``stream_instance_id``, the namespace becomes
    ``{dataset}[__reg__…][__ts__…]`` and ``judge_content_version`` is written — sampling
    takes only that lineage version. Same-day registrations, and live streams that differ
    only by sample rate, do not mix into one report.

    **Fallback only, never overwrite.** When a same-name item already exists (default or user
    override), its payload is kept as-is — a later eval-item save must not wipe a user's clock,
    template, or sample customizations. A soft-deleted row is only re-enabled; its payload is
    also left alone. When the same name is already taken by another benchmark (several
    benchmarks share one dataset name), the namespace gets a benchmark prefix so a later write
    does not silently drop an earlier report.
    A failed registration is logged and not raised — report registration must not fail the eval-item write.
    """
    name = seed.name
    benchmark_name = seed.benchmark_name
    source = seed.source
    agent_name = seed.agent_name
    project_id = seed.project_id
    registration_id = (seed.agent_registration_id or "").strip() or None
    stream_instance_id = (seed.stream_instance_id or "").strip() or None
    judge_cv = (seed.judge_content_version or "").strip() or None
    # Lazy import: same ordering constraint as list_db_trace_report_items.
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        KIND_TRACE_REPORT,
        SCOPE_GLOBAL,
        RuntimeConfigRecord,
        project_scope_key,
        trace_report_record_key,
    )
    from benchmark.domain.runtime_config_store import (  # noqa: PLC0415
        restore_runtime_config,
        upsert_runtime_config,
    )

    item_name = versioned_trace_report_name(
        name.strip(),
        registration_id or "",
        stream_instance_id or "",
    )
    if not item_name or not benchmark_name.strip():
        logger.warning(
            "default trace report skipped: name=%r benchmark_name=%r",
            name,
            benchmark_name,
        )
        return ""
    try:
        existing = await RuntimeConfigRecord.filter(
            kind=KIND_TRACE_REPORT,
            record_key=trace_report_record_key(item_name),
        ).first()
        if existing is not None and isinstance(existing.payload, dict):
            occupied_by = str(existing.payload.get("benchmark_name") or "")
            if occupied_by and occupied_by != benchmark_name:
                item_name = f"{benchmark_name}-{item_name}"
                existing = await RuntimeConfigRecord.filter(
                    kind=KIND_TRACE_REPORT,
                    record_key=trace_report_record_key(item_name),
                ).first()
        if existing is not None:
            # Same-name item already exists (default or custom): keep the payload and only clear the soft-delete.
            if existing.deleted_at is not None:
                await restore_runtime_config(
                    KIND_TRACE_REPORT,
                    trace_report_record_key(item_name),
                )
            return item_name
        payload: dict[str, object] = {
            "name": item_name,
            "benchmark_name": benchmark_name,
            "source": source,
            "schedule_time": DEFAULT_TRACE_REPORT_SCHEDULE_TIME,
            "project_id": project_id,
            "sample_size": None,
            "origin": TRACE_REPORT_ORIGIN_AUTO_DEFAULT,
        }
        if agent_name:
            payload["agent_name"] = agent_name
        if registration_id:
            payload["agent_registration_id"] = registration_id
        if stream_instance_id:
            payload["stream_instance_id"] = stream_instance_id
        if judge_cv:
            payload["judge_content_version"] = judge_cv
        await upsert_runtime_config(
            kind=KIND_TRACE_REPORT,
            record_key=trace_report_record_key(item_name),
            payload=payload,
            scope_key=project_scope_key(project_id) if project_id else SCOPE_GLOBAL,
        )
    except Exception:
        logger.warning(
            "default trace report register failed: name=%s benchmark=%s",
            item_name,
            benchmark_name,
            exc_info=True,
        )
        return ""
    return item_name


async def list_db_trace_report_items() -> list[TraceReportItemConfig]:
    """Read T+1 report items written automatically when a live stream is registered (active ``kind=trace_report`` rows).

    Merged with RemoteConfig ``reports[]`` for TraceDailyReportScheduler. A live-stream task
    registered in the webUI (amap_agent_benchmark and others) gets a T+1 report without a
    hand-written same-name RemoteConfig entry. A row that fails to parse is logged at warning
    and skipped (one dirty row must not take down the whole schedule).
    """
# Lazy import: keep the schema load chain from assembling data_model before RemoteConfig is bound
# (the same "bind before the RemoteConfig import" constraint as the file header).
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        KIND_TRACE_REPORT,
        RuntimeConfigRecord,
    )

    rows = await RuntimeConfigRecord.filter(
        kind=KIND_TRACE_REPORT,
        deleted_at__isnull=True,
    )
    items: list[TraceReportItemConfig] = []
    for row in rows:
        try:
            items.append(TraceReportItemConfig.model_validate(row.payload))
        except Exception:
            logger.warning(
                "db trace report item parse failed: record_key=%s",
                row.record_key,
            )
    return items


__all__ = [
    "AMAP_AGENT_LEGACY_DAILY_NAMESPACE",
    "DEFAULT_TRACE_REPORT_SCHEDULE_TIME",
    "TRACE_DAILY_REPORT_DATA_ID",
    "TRACE_REPORT_ORIGIN_AUTO_DEFAULT",
    "TRACE_DAILY_REPORT_GROUP",
    "TRACE_REPORT_NAMESPACE_CONFIG_KEY",
    "TRACE_REPORT_REGISTRATION_NAME_SEP",
    "TRACE_REPORT_STREAM_NAME_SEP",
    "TraceReportConfig",
    "TraceReportItemConfig",
    "TraceReportSeed",
    "amap_agent_legacy_daily_namespaces",
    "benchmark_name_for_evaluator",
    "ensure_default_trace_report",
    "fetch_trace_report_config",
    "list_db_trace_report_items",
    "normalize_report_html_name",
    "registered_dataset_name_from_report_name",
    "versioned_trace_report_name",
]
