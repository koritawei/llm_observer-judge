"""Aggregate data collection — two sources: PG (judge + attribution) and Langfuse (performance).

Design points:

1. **Recall by batch_tag and narrow by evaluator_name**: the ``TraceRecord`` composite
   business key includes ``evaluator_name``, so one trace can be one row per evaluator.
   Without the narrow, ``judged_count`` is inflated and ``expectedTotal`` fills early forever.
2. **Converge with a time window**: ``judge_json->'extra'->>'batch_tag'`` is unindexed, and a
   full-table scan is unacceptable. batch_tag carries ``YYYYmmdd-HHMMSS``; decode the start
   time and intersect it with a ``created_at`` range (that column is indexed) first. batch_tag
   timestamps are always built as **UTC+8** aware datetimes (the load generator is in the
   Beijing timezone), so they do not drift from the ORM storage timezone.
3. **JSONB filter dialect**: PG uses ``judge_json->'extra'->>'batch_tag'``. SQLite (unit
   tests) has no JSONB operators, so it degrades to a time window plus Python-side filtering.
   The dialect decision stays here and is not leaked upward.
4. **A performance-collection failure does not block**: when Langfuse is unreachable, each
   case records ``perf.error``, and the effectiveness/attribution blocks are still produced (AC-13).
5. **Progress counts use a narrow-column path**: the watcher queries ``count_progress`` once
   per poll and only needs judge_json.extra (tag/dedupe key) + the three attribution states +
   the old/new comparison columns. ``.only()`` therefore drops large buckets such as
   ``trace_meta_json`` (full evaluation IO) and ``notification_job_json`` (payload that
   includes the full report), so a heavy query is not used as a heartbeat (T6).
6. **Langfuse fetches trickle**: perf collection is a burst against Langfuse (two requests
   per trace), and Langfuse itself has limited CPU/DB headroom. On top of the concurrency
   cap, per-trace starts are throttled by a minimum interval so the burst becomes a trickle (T7).
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.reporting.cluster.constants import (
    BATCH_TAG_RE,
    DEFAULT_PERF_CONCURRENCY,
    DEFAULT_PERF_MIN_INTERVAL_S,
    DEFAULT_QUERY_WINDOW_S,
    OBSERVATION_NAME_TO_BUCKET,
    QUERY_WINDOW_BACKOFF_S,
)
from benchmark.reporting.cluster.schema import (
    AttributionView,
    BatchCaseRecord,
    BucketPerf,
    CaseKey,
    CasePerf,
)
from benchmark.records.models import TraceRecord
from benchmark.pipeline.normalize import normalize_eval_extra
from benchmark.trace.http_backend import ObservationListQuery

logger = get_logger(__name__)

#: Timezone basis of the batch_tag timestamp (the load generator is in the Beijing timezone).
_UTC_PLUS_8 = timezone(timedelta(hours=8))

#: Per-trace fetch cap for Langfuse observations (v1 page size 100; the client pages internally).
_OBS_LIMIT = 200
#: Per-trace performance-collection timeout (seconds).
_PERF_TIMEOUT_S = 30.0
_TRACE_METRIC_FIELDS = "core,io,metadata,metrics"
_GENERATION_METRIC_FIELDS = "core,basic,io,usage,metrics"
_INPUT_FIELDS = "core,io"
_FIRST_TEXT_METADATA_KEY = "first_text_response_time_ms"
_TRACE_ROUTE_TAGS = {
    "lite_sdm": "lite_sdm",
    "lite_dag": "lite_dag",
    "lite_delivered": "lite_dag",
    "standard": "standard",
    # Compatible with existing VUI trace tags: sdm/lite and their early aliases.
    "sdm": "lite_sdm",
    "lite": "lite_dag",
    "dag": "lite_dag",
    "fallback": "standard",
}

#: Narrow column set for count_progress: batch_tag filter and dedupe key (judge_json.extra), old/new comparison
#: (created_at / updated_at / trace_id), and the three attribution states (attribution_detail_json /
#: the root_cause scalar). Explicitly excludes trace_meta_json (full evaluation IO), notification_job_json
#: (payload includes the full report), and other large buckets. Progress polling is a hot path; loading every column would use
#: a heavy query as a heartbeat (T6). Accessing a column that was not selected makes tortoise .only() raise AttributeError
#: and fail fast, rather than silently reading None.
_PROGRESS_COLUMNS = (
    "id",
    "trace_id",
    "created_at",
    "updated_at",
    "judge_json",
    "attribution_detail_json",
    "root_cause",
)


def parse_batch_window(
    batch_tag: str,
    *,
    window_s: int = DEFAULT_QUERY_WINDOW_S,
) -> tuple[datetime, datetime] | None:
    """Decode batch_tag into a ``created_at`` query range (UTC+8 aware).

    ``None`` means the batch_tag shape is invalid. The caller must return an incomplete
    recall result and must not degrade to a full-table scan with no time window.
    """

    match = BATCH_TAG_RE.match(batch_tag or "")
    if match is None:
        return None
    try:
        start = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S").replace(
            tzinfo=_UTC_PLUS_8
        )
    except ValueError:
        return None
    return (
        start - timedelta(seconds=QUERY_WINDOW_BACKOFF_S),
        start + timedelta(seconds=window_s),
    )


@dataclass(frozen=True)
class BatchCollectionOptions:
    window_s: int = DEFAULT_QUERY_WINDOW_S
    include_late_records: bool = False


_DEFAULT_COLLECTION_OPTIONS = BatchCollectionOptions()


@dataclass(frozen=True)
class _BatchQuery:
    batch_tag: str
    evaluator_name: str
    options: BatchCollectionOptions
    narrow: bool = False


@dataclass(frozen=True)
class _RecordScan:
    batch_tag: str | None
    evaluator_name: str
    window: tuple[datetime, datetime] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    narrow: bool = False


class PgCollector:
    """Recall judge + attribution data from ``trace_pipeline_records`` by batch."""

    async def fetch_batch_records(
        self,
        *,
        batch_tag: str,
        evaluator_name: str,
        options: BatchCollectionOptions = _DEFAULT_COLLECTION_OPTIONS,
    ) -> list[BatchCaseRecord]:
        """Recall batch cases inside the bounded window given by the caller, and keep the latest record per case."""

        rows = await self._query_rows(
            _BatchQuery(batch_tag=batch_tag, evaluator_name=evaluator_name, options=options)
        )
        deduped = _dedupe_by_case_key(rows, batch_tag=batch_tag)
        return [self._to_record(row) for row in deduped]

    async def fetch_time_window_records(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        evaluator_name: str,
    ) -> list[BatchCaseRecord]:
        """Recall online evaluation records by time range; the upstream does not have to write batch_tag."""

        rows = await self._scan_rows(
            _RecordScan(
                batch_tag=None,
                evaluator_name=evaluator_name,
                created_after=start_at,
                created_before=end_at,
            )
        )
        # An online sample is uniquely located by the database composite execution key; different traces of the same query are all kept.
        # Offline case/retry dedupe is only for the batch API and must not be used for the online time window.
        return [self._to_record(row) for row in rows]

    async def fetch_source_batch_records(
        self,
        *,
        source_batch_tags: list[str],
        evaluator_name: str,
        options: BatchCollectionOptions = _DEFAULT_COLLECTION_OPTIONS,
    ) -> list[BatchCaseRecord]:
        """Recall records from several source batches; dedupe within a batch and keep multiple executions across batches."""

        records: list[BatchCaseRecord] = []
        for batch_tag in source_batch_tags:
            records.extend(
                await self.fetch_batch_records(
                    batch_tag=batch_tag,
                    evaluator_name=evaluator_name,
                    options=options,
                )
            )
        return records

    async def count_progress(
        self,
        *,
        batch_tag: str,
        evaluator_name: str,
        options: BatchCollectionOptions = _DEFAULT_COLLECTION_OPTIONS,
    ) -> dict[str, int]:
        """Three-state counts: judged / should_attribute / attributed / pending.

        Attribution completion is whether ``root_cause`` exists (the same write scale as
        ``TraceRecordService``). ``attribution_status`` only distinguishes "in progress"
        from "not triggered".

        Same scale as ``fetch_batch_records`` (same window / same dedupe), but on the
        narrow-column path: the watcher's hot poll takes only the columns the counts need
        and does not load large buckets such as trace_meta_json / notification_job_json
        (T6). Counts do not need a full ``BatchCaseRecord``, so ``_to_record`` is skipped.
        """

        rows = await self._query_rows(
            _BatchQuery(
                batch_tag=batch_tag, evaluator_name=evaluator_name, options=options, narrow=True
            )
        )
        deduped = _dedupe_by_case_key(rows, batch_tag=batch_tag)
        attributed = sum(1 for row in deduped if _attribution_status_of(row) == "completed")
        pending = sum(1 for row in deduped if _attribution_status_of(row) == "pending")
        return {
            "judged_count": len(deduped),
            "should_attribute_count": attributed + pending,
            "attributed_count": attributed,
            "pending": pending,
        }

    # ── internals ────────────────────────────────────────────────────────

    async def _query_rows(self, query_spec: _BatchQuery) -> list[TraceRecord]:
        """Fetch records by batch_tag + evaluator_name inside a bounded time window.

        ``batch_tag`` has no index of its own, so exact JSON comparison happens in the
        application. To avoid a full-table scan by evaluator when the tag is invalid or
        the window misses: an unparseable tag returns empty immediately, and a parseable
        tag always scans only the window the ``created_at`` index can converge. Report
        build, retry, and stability pass a 3-day window to cover late rows; an ordinary
        normal first build still uses 6 hours.

        ``narrow=True`` (count_progress only) takes the ``_PROGRESS_COLUMNS`` narrow set
        and does not load large buckets; the caller must not access columns that were not selected.
        """
        window = parse_batch_window(query_spec.batch_tag, window_s=query_spec.options.window_s)
        if window is None:
            logger.warning(
                "cluster_report.batch_tag_unparsable batch_tag=%r returning incomplete recall",
                query_spec.batch_tag,
            )
            return []
        rows = await self._scan_rows(
            _RecordScan(
                query_spec.batch_tag,
                query_spec.evaluator_name,
                window=window,
                narrow=query_spec.narrow,
            )
        )
        if not rows:
            logger.warning(
                "cluster_report.window_miss batch_tag=%r window=%s returning incomplete recall",
                query_spec.batch_tag,
                window,
            )
        return rows

    @staticmethod
    async def _scan_rows(scan: _RecordScan) -> list[TraceRecord]:
        query = TraceRecord.filter(evaluator_name=scan.evaluator_name)
        if scan.window is not None:
            query = query.filter(created_at__gte=scan.window[0], created_at__lte=scan.window[1])
        if scan.created_after is not None:
            query = query.filter(created_at__gte=scan.created_after)
        if scan.created_before is not None:
            query = query.filter(created_at__lt=scan.created_before)
        if scan.narrow:
            query = query.only(*_PROGRESS_COLUMNS)
        rows = await query
        if scan.batch_tag is None:
            return list(rows)
        return [row for row in rows if _extra_of(row).get("batch_tag") == scan.batch_tag]

    @staticmethod
    def _to_record(row: TraceRecord) -> BatchCaseRecord:
        judge = dict(row.judge_json or {})
        raw_extra = (row.judge_json or {}).get("extra")
        extra = _extra_of(row)
        envelope = _envelope_sources(row)
        trace_meta = dict(row.trace_meta_json or {})
        case_meta = dict(extra.get("case_meta") or {})
        if isinstance(raw_extra, dict):
            raw_query = _query_from_extra(raw_extra)
            if raw_query and not extra.get("query") and not extra.get("user_query"):
                extra["user_query"] = raw_query
        failure_summary = dict(extra.get("failure_summary") or {})
        # Effectiveness conclusions of unified-envelope rows take the shared core as authority (shared aggregation uses the envelope ``dimension_results``
        # as the source of truth; see design §9.2). Historical rows have an empty shared core and fall back entirely to the old read view, with no behavior change.
        primary_score = envelope["primary_score"]
        legacy_score = extra.get("score", judge.get("score"))
        dimension_results = envelope["dimension_results"] or list(
            extra.get("dimension_results") or []
        )

        return BatchCaseRecord(
            trace_id=row.trace_id,
            project_id=str(row.project_id or ""),
            record_id=str(row.id),
            evaluator_name=row.evaluator_name,
            source_batch_tag=str(extra.get("batch_tag") or ""),
            trace_meta=trace_meta,
            case_meta=case_meta,
            case_key=_build_case_key(extra, case_meta),
            score=_to_float(primary_score if primary_score is not None else legacy_score),
            reason=envelope["summary_reason"]
            or str(extra.get("reason") or judge.get("reason") or ""),
            triage_level=str(row.triage_level or judge.get("triage_level") or ""),
            dimension_results=dimension_results,
            checker_results=list(extra.get("checker_results") or []),
            hard_rule_hits=list(extra.get("hard_rule_hits") or []),
            risk_flags=list(extra.get("risk_flags") or []),
            failure_modes=list(failure_summary.get("failure_modes") or []),
            judge_confidence=_to_float(extra.get("confidence")),
            business_id=envelope["business_id"],
            scenario=envelope["scenario"],
            profile_id=envelope["profile_id"],
            profile_version=envelope["profile_version"],
            metric_observations=envelope["metric_observations"],
            diagnostic_findings=envelope["diagnostic_findings"],
            attribution=_build_attribution(row),
            notification_workitem_id=_notification_workitem_id(row),
        )


def _extra_of(row: TraceRecord) -> dict[str, Any]:
    extra = (row.judge_json or {}).get("extra")
    if not isinstance(extra, dict):
        return {}
    # Migration compatibility: the unified envelope is restored to the old read view by business type;
    # historical v3 passes through as-is, so batch recall and dedupe-key reads do not change.
    return normalize_eval_extra(extra)


#: Direct-read field set of the unified-envelope shared core (not via the normalized adapter).
_EMPTY_ENVELOPE_SOURCES: dict[str, Any] = {
    "primary_score": None,
    "summary_reason": "",
    "dimension_results": [],
    "business_id": "",
    "scenario": "",
    "profile_id": "",
    "profile_version": "",
    "metric_observations": [],
    "diagnostic_findings": [],
}


def _envelope_sources(row: TraceRecord) -> dict[str, Any]:
    """Read the shared core of the unified envelope ``agent_eval_extra.v1`` directly (authority for report slices and effectiveness conclusions).

    Division of labor with ``_extra_of``: ``_extra_of`` restores the old business read
    view through the normalized adapter (business fields such as ``batch_tag`` recall,
    the dedupe key, and ``case_meta``). This function takes only shared-core fields and
    does not depend on business-extension semantics. Historical non-envelope rows return
    all-empty defaults, and report output matches the baseline.
    """

    extra = (row.judge_json or {}).get("extra")
    if not isinstance(extra, dict) or extra.get("schema_version") != "agent_eval_extra.v1":
        return dict(_EMPTY_ENVELOPE_SOURCES)
    return {
        "primary_score": extra.get("primary_score"),
        "summary_reason": str(extra.get("summary_reason") or ""),
        "dimension_results": [
            item for item in (extra.get("dimension_results") or []) if isinstance(item, dict)
        ],
        "business_id": str(extra.get("business_id") or ""),
        "scenario": str(extra.get("scenario") or ""),
        "profile_id": str(extra.get("profile_id") or ""),
        "profile_version": str(extra.get("profile_version") or ""),
        "metric_observations": [
            item for item in (extra.get("metric_observations") or []) if isinstance(item, dict)
        ],
        "diagnostic_findings": [
            item for item in (extra.get("diagnostic_findings") or []) if isinstance(item, dict)
        ],
    }


def _query_from_extra(extra: dict[str, Any]) -> str:
    """Fall back to extracting the VUI input from the raw unified envelope, so the compatibility view does not drop Query."""

    extension = extra.get("business_extension")
    payload = extension.get("payload") if isinstance(extension, dict) else None
    for source in (extra, extension, payload):
        if not isinstance(source, dict):
            continue
        for key in ("query", "user_query"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _build_case_key(extra: dict[str, Any], case_meta: dict[str, Any]) -> CaseKey:
    """Assemble the alignment key from extra + case_meta.

    Vehicle ``tag`` / ``query`` / ``turn_index`` sit at the top of extra; the VUI Query
    sits in ``user_query``. ``suite`` / ``case_id`` can come only from case_meta (the
    trace itself does not carry ownership). When old data lacks these keys, each
    component is empty and no error is raised.
    """

    return CaseKey(
        suite=str(case_meta.get("suite") or ""),
        case_id=str(case_meta.get("case_id") or ""),
        tag=str(extra.get("tag") or ""),
        query=str(extra.get("query") or extra.get("user_query") or ""),
        turn_index=str(extra.get("turn_index") or ""),
    )


def _has_case_identity(key: CaseKey) -> bool:
    """Whether this key is enough to uniquely identify a case (otherwise it must not join dedupe).

    ``suite`` is a batch-level component (shared by the whole batch) and cannot identify
    a case by itself. ``CaseKey.normalized()`` collapses to one empty key when every
    business component is empty (for example old data missing case_meta / extra). If
    those rows joined dedupe they would all merge into one and data would be lost, so
    at least one of case_id / tag / query must be non-empty.
    """

    return bool(key.case_id or key.tag or key.query)


def _row_recency(row: TraceRecord) -> tuple[Any, ...]:
    """Old/new comparison key for a row — ``created_at`` is primary.

    Several rows of the same case are several requests (each retry is a new trace_id →
    one row), so only ``created_at`` reflects request order. ``updated_at`` is pushed
    later by attribution/dispatch write-back, so an older request's row may be updated
    later and must not be the primary order. When concurrent inserts share created_at,
    ``updated_at`` / ``trace_id`` break the tie so the order stays deterministic.
    """

    return (row.created_at, row.updated_at, row.trace_id or "")


def _dedupe_by_case_key(rows: list[TraceRecord], *, batch_tag: str) -> list[TraceRecord]:
    """When the same case has several rows in one batch, keep only the latest row.

    Why this is required: the ``TraceRecord`` composite key includes ``trace_id``, and
    client retries (``max_retries`` in ``run-eval.py``; both timeout and business failure
    retry) plus server redelivery give one case several traces while ``batch_tag`` /
    ``case_meta`` stay identical → several rows in one batch. Without dedupe,
    ``total`` / ``pass_rate`` / dimension failure rate / ``eval_coverage`` are all
    inflated (timeout retries are the worst: the server finishes both runs, both are
    judged normally, and both rows carry a score).

    The dedupe key is ``CaseKey.normalized()`` (this field exists for that: prefer
    case_id when present, because queries repeat often). Rows with no valid identity
    are all kept (see ``_has_case_identity``). Return order is stable by the first time
    each key was seen (report/xlsx detail row order does not jitter).
    """

    latest: dict[tuple[str, ...], TraceRecord] = {}
    unidentified: list[TraceRecord] = []
    for row in rows:
        extra = _extra_of(row)
        key = _build_case_key(extra, dict(extra.get("case_meta") or {}))
        if not _has_case_identity(key):
            unidentified.append(row)
            continue
        norm = key.normalized()
        previous = latest.get(norm)
        if previous is None or _row_recency(row) >= _row_recency(previous):
            latest[norm] = row

    kept = [*latest.values(), *unidentified]
    dropped = len(rows) - len(kept)
    if dropped > 0:
        logger.info(
            "cluster_report.dedupe_retries batch=%s rows=%d kept=%d dropped=%d",
            batch_tag,
            len(rows),
            len(kept),
            dropped,
        )
    return kept


def _attribution_status_of(row: TraceRecord) -> str:
    """Three-state attribution decision: completed (root_cause exists) / pending (in progress) / none.

    root_cause is read from the attribution_detail_json bucket first, then the scalar
    column. The pending mark lives only in the bucket (there is no scalar column), so
    the narrow-column path must include that bucket (see ``_PROGRESS_COLUMNS``).
    ``_build_attribution`` and ``count_progress`` share this so the report scale and the
    progress-count scale stay the same.
    """

    bucket = dict(row.attribution_detail_json or {})
    if bucket.get("root_cause") or row.root_cause:
        return "completed"
    if str(bucket.get("attribution_status") or "") == "pending":
        return "pending"
    return "none"


def _build_attribution(row: TraceRecord) -> AttributionView:
    """Three-state attribution plus the full field view (for the report path; counts should use ``_attribution_status_of``)."""

    bucket = dict(row.attribution_detail_json or {})
    status = _attribution_status_of(row)
    root_cause = str(bucket.get("root_cause") or row.root_cause or "")

    return AttributionView(
        status=status,
        root_cause=root_cause,
        root_cause_code=str(bucket.get("root_cause_code") or ""),
        all_root_cause_codes=[
            str(code).strip()
            for code in (bucket.get("all_root_cause_codes") or [])
            if str(code or "").strip()
        ],
        # The standalone category key wins; old data falls back to owner_domain (historically that stored the category value).
        category=str(bucket.get("category") or bucket.get("owner_domain") or ""),
        owner_group=str(bucket.get("owner_group") or row.owner_group or ""),
        conclusion_type=str(bucket.get("conclusion_type") or ""),
        confidence=_to_float(bucket.get("confidence")),
        corrected_triage_level=str(bucket.get("corrected_triage_level") or ""),
        needs_human_review=bool(bucket.get("needs_human_review") or False),
        evidence_details=list(bucket.get("evidence_details") or []),
        action_hint=str(bucket.get("action_hint") or ""),
        candidate_root_cause=bucket.get("candidate_root_cause"),
    )


def _notification_workitem_id(row: TraceRecord) -> str:
    """Extract a dedupeable Aone ticket id from a persisted dispatch reference."""

    dispatch = getattr(row, "notification_dispatch_json", None)
    if not isinstance(dispatch, dict):
        return ""
    return str(
        dispatch.get("notification_ref") or dispatch.get("workitem_id") or dispatch.get("workitemId") or ""
    ).strip()


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TraceReadClient(Protocol):
    """Langfuse read interface required for performance collection (satisfied by trace_resolver.LangfuseClient).

    A Protocol rather than a concrete class: tests can inject a light fake, and the
    collection layer is not pinned to one client implementation. Both methods are
    optional (when missing, collection degrades instead of raising).

    Note: this does **not** include ``resolve_primary_id``. Performance collection is a
    DB consumption path, and the ``trace_id`` it receives is already the Langfuse
    primary id (rewritten by before_eval at the eval entry). The trace_resolver contract
    forbids resolving it again (see ``LangfusePerfCollector._collect``).
    """

    async def get_trace(
        self, primary_id: str, *, fields: str = "core", expand_metadata: str | None = None
    ) -> object: ...

    async def list_observations_for_trace(
        self, primary_id: str, query: ObservationListQuery
    ) -> list[dict[str, Any]]: ...


class TracePerfCollector:
    """Backfill performance metrics from Langfuse (trace-level latency + observation-level tokens).

    Scale aligned with ``generate-eval-report.py`` (pinned by the T13 comparison test):
      - end-to-end latency = ``trace.latency × 1000``, truncated to int
      - first-text latency = ``trace.metadata.first_text_response_time_ms``
      - three buckets = ``type=GENERATION`` observations mapped by ``obs.name``
    """

    def __init__(
        self,
        *,
        client: TraceReadClient | object,
        concurrency: int = DEFAULT_PERF_CONCURRENCY,
        min_interval_s: float = DEFAULT_PERF_MIN_INTERVAL_S,
    ) -> None:
        self._client = client
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._min_interval_s = max(0.0, min_interval_s)
        self._pace_lock = asyncio.Lock()
        self._next_start = 0.0  # Next allowed start time on the event-loop timeline

    async def fetch_perf(self, trace_ids: list[str]) -> dict[str, CasePerf]:
        """Fetch per-trace performance concurrently; one failure affects only that trace (recorded as error)."""

        results = await asyncio.gather(
            *(self._fetch_one(tid) for tid in trace_ids),
            return_exceptions=True,
        )
        out: dict[str, CasePerf] = {}
        for trace_id, result in zip(trace_ids, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "cluster_report.perf_failed trace_id=%s err=%s",
                    trace_id,
                    type(result).__name__,
                )
                out[trace_id] = CasePerf(error=type(result).__name__)
            else:
                out[trace_id] = result
        return out

    async def fetch_route_modes(self, trace_ids: list[str]) -> dict[str, str]:
        """Read the persisted VUI route result from Langfuse trace tags."""

        results = await asyncio.gather(
            *(self._fetch_route_mode(trace_id) for trace_id in trace_ids),
            return_exceptions=True,
        )
        return {
            trace_id: route_mode
            for trace_id, route_mode in zip(trace_ids, results, strict=True)
            if isinstance(route_mode, str) and route_mode
        }

    async def fetch_inputs(self, trace_ids: list[str]) -> dict[str, object]:
        """Backfill a missing original Trace input, only so the report presentation layer can recover Query."""

        results = await asyncio.gather(
            *(self._fetch_input(trace_id) for trace_id in trace_ids),
            return_exceptions=True,
        )
        return {
            trace_id: value
            for trace_id, value in zip(trace_ids, results, strict=True)
            if value is not None and not isinstance(value, BaseException)
        }

    async def _fetch_input(self, trace_id: str) -> object | None:
        if not trace_id:
            return None
        await self._pace()
        async with self._semaphore:
            try:
                trace = await asyncio.wait_for(
                    self._get_trace(trace_id, fields=_INPUT_FIELDS), timeout=_PERF_TIMEOUT_S
                )
            except TimeoutError:
                return None
        return _attr_or_key(trace, "input")

    async def _fetch_route_mode(self, trace_id: str) -> str:
        if not trace_id:
            return ""
        await self._pace()
        async with self._semaphore:
            try:
                trace = await asyncio.wait_for(self._get_trace(trace_id), timeout=_PERF_TIMEOUT_S)
            except TimeoutError:
                return ""
        return _route_mode_from_trace(trace)

    async def _fetch_one(self, trace_id: str) -> CasePerf:
        await self._pace()
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    self._collect(trace_id),
                    timeout=_PERF_TIMEOUT_S,
                )
            except TimeoutError:
                return CasePerf(error="timeout")

    async def _pace(self) -> None:
        """Start throttling: adjacent per-trace fetch starts are at least ``min_interval_s`` apart.

        Division of labor with the semaphore: the semaphore caps how many are in flight
        at once (peak concurrency); this caps the start rate (a trickle). Together,
        Langfuse sees neither a high-concurrency peak nor a sustained high rate.
        """
        if self._min_interval_s <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._pace_lock:
            start_at = max(loop.time(), self._next_start)
            self._next_start = start_at + self._min_interval_s
        delay = start_at - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)

    async def _collect(self, trace_id: str) -> CasePerf:
        # trace_id is already the Langfuse primary id when stored (the eval entry before_eval
        # rewrote it from the business metadata id via resolve_primary_id_for). This is a DB consumption
        # path, and the trace_resolver contract says it must not be resolved again: resolving again treats the primary id as
        # a business id and filters metadata, which always misses, so every row is falsely reported as trace_not_found
        # (the root cause of perf collected=0 for the whole block). Query with trace_id as the primary id directly.
        if not trace_id:
            return CasePerf(error="trace_not_found")
        primary_id = trace_id

        trace = await self._get_trace(primary_id)
        observations = await self._list_observations(primary_id)
        if trace is None and not observations:
            # This primary id matches no trace / observation in Langfuse → a real not found.
            return CasePerf(error="trace_not_found")

        perf = CasePerf()
        if trace is not None:
            latency = _to_float(_attr_or_key(trace, "latency"))
            if latency is not None:
                # Scale aligned with the script side: latency (seconds) × 1000, truncated to int.
                perf.e2e_ms = float(round(latency * 1000))
            metadata = _attr_or_key(trace, "metadata") or {}
            if isinstance(metadata, dict):
                perf.first_text_ms = _to_float(metadata.get("first_text_response_time_ms"))
            perf.route_mode = _route_mode_from_trace(trace)

        perf.buckets = aggregate_buckets(observations)
        perf.models = aggregate_models(observations)
        return perf

    async def _get_trace(self, primary_id: str, *, fields: str = _TRACE_METRIC_FIELDS) -> object:
        getter = getattr(self._client, "get_trace", None)
        if getter is None:
            return None
        return await getter(
            primary_id,
            fields=fields,
            expand_metadata=_FIRST_TEXT_METADATA_KEY if "metadata" in fields.split(",") else None,
        )

    async def _list_observations(self, primary_id: str) -> list[dict[str, Any]]:
        lister = getattr(self._client, "list_observations_for_trace", None)
        if lister is None:
            return []
        return (
            await lister(
                primary_id,
                ObservationListQuery(
                    limit=_OBS_LIMIT,
                    type="GENERATION",
                    fields=_GENERATION_METRIC_FIELDS,
                ),
            )
            or []
        )


def aggregate_buckets(observations: list[dict[str, Any]]) -> dict[str, BucketPerf]:
    """Aggregate GENERATION observations into three buckets by ``name``.

    Scale aligned with ``generate-eval-report.py``: skip observations with no tokens;
    when ``totalTokens`` is missing, fall back to ``prompt + completion``; ``latency``
    (seconds) × 1000. A name that is not in the map is ignored (do not guess a bucket).
    """

    buckets: dict[str, BucketPerf] = {}
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        prompt_tokens = int(obs.get("promptTokens") or 0)
        completion_tokens = int(obs.get("completionTokens") or 0)
        total_tokens = int(obs.get("totalTokens") or 0)
        if not (prompt_tokens or completion_tokens or total_tokens):
            continue
        bucket_name = OBSERVATION_NAME_TO_BUCKET.get(str(obs.get("name") or ""), "other")
        bucket = buckets.setdefault(bucket_name, BucketPerf())
        bucket.calls += 1
        bucket.prompt_tokens += prompt_tokens
        bucket.completion_tokens += completion_tokens
        bucket.total_tokens += total_tokens or (prompt_tokens + completion_tokens)
        bucket.cost += float(obs.get("calculatedTotalCost") or 0.0)
        model_name = _observation_model_name(obs)
        if model_name:
            bucket.models[model_name] = bucket.models.get(model_name, 0) + 1
        latency = _to_float(obs.get("latency"))
        if latency is not None:
            bucket.latency_ms += latency * 1000
    return buckets


def aggregate_models(observations: list[dict[str, Any]]) -> dict[str, BucketPerf]:
    """Aggregate one case's performance by the actual Langfuse Generation model."""

    models: dict[str, BucketPerf] = {}
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        prompt_tokens = int(observation.get("promptTokens") or 0)
        completion_tokens = int(observation.get("completionTokens") or 0)
        total_tokens = int(observation.get("totalTokens") or 0)
        if not (prompt_tokens or completion_tokens or total_tokens):
            continue
        model_name = _observation_model_name(observation) or "未采集"
        model = models.setdefault(model_name, BucketPerf())
        model.calls += 1
        model.prompt_tokens += prompt_tokens
        model.completion_tokens += completion_tokens
        model.total_tokens += total_tokens or (prompt_tokens + completion_tokens)
        model.cost += float(observation.get("calculatedTotalCost") or 0.0)
        latency = _to_float(observation.get("latency"))
        if latency is not None:
            model.latency_ms += latency * 1000
    return models


def _observation_model_name(observation: dict[str, Any]) -> str:
    """Read the real model name of a Langfuse Generation; when missing, do not guess the service default."""

    for value in (
        observation.get("model"),
        _mapping_value(observation.get("metadata"), "model"),
        _mapping_value(observation.get("modelParameters"), "model"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _mapping_value(value: object, key: str) -> object:
    return value.get(key) if isinstance(value, dict) else None


def case_perf_from_observations(observations: list[dict[str, Any]]) -> CasePerf | None:
    """Project unified metrics into the old performance-report view, so new envelope rows do not need to query Langfuse again."""
    perf = CasePerf()
    seen = False
    for item in observations:
        if not isinstance(item, dict) or item.get("status") == "unknown":
            continue
        key = str(item.get("metric_key") or "")
        value = _to_float(item.get("value"))
        if value is None:
            continue
        if key == "total_latency_ms":
            perf.e2e_ms = value
        elif key == "first_packet_latency_ms":
            perf.first_text_ms = value
        else:
            scope = str(item.get("sample_scope") or "")
            if scope not in OBSERVATION_NAME_TO_BUCKET.values():
                continue
            bucket = perf.buckets.setdefault(scope, BucketPerf())
            if key == "llm_call_count":
                bucket.calls = int(value)
            elif key == "llm_token_count":
                bucket.total_tokens = int(value)
            elif key == "llm_latency_ms":
                bucket.latency_ms = value
            else:
                continue
        seen = True
    return perf if seen else None


def _route_mode_from_trace(trace: object) -> str:
    """Normalize the race winner already persisted for VUI, using Langfuse tags."""

    tags = _attr_or_key(trace, "tags") or []
    if not isinstance(tags, list | tuple | set):
        return ""
    normalized_tags = {str(tag).strip().lower() for tag in tags}
    for tag, route_mode in _TRACE_ROUTE_TAGS.items():
        if tag in normalized_tags:
            return route_mode
    return ""


def _attr_or_key(obj: object, name: str) -> object:
    """Accept both a pydantic model and a dict as the trace representation."""

    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


__all__ = [
    "TracePerfCollector",
    "PgCollector",
    "aggregate_buckets",
    "aggregate_models",
    "case_perf_from_observations",
    "parse_batch_window",
]
