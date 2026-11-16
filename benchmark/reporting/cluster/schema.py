"""Cluster-report data contracts — BatchCaseRecord / L1Stats / ClusterReport.

Layers:
- :class:`BatchCaseRecord` — one-case row view (judge + attribution + performance,
  three sources in one), assembled by the collector from PG (judge_json.extra /
  attribution_detail_json) + Langfuse (perf). ``case_meta`` is an open dict passed
  through in full, so a new case source needs no contract change.
- :class:`L1Stats` — deterministic statistics from pure code arithmetic (effectiveness /
  attribution / performance / crosstab).
- :class:`ClusterReport` — root object of the machine-readable artifact, with
  ``schema_version`` and ``batch_meta`` in the header, so P1 cross-batch comparison
  can join on ``case_key``.
"""

from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, field_validator, model_validator

from benchmark.reporting.cluster.constants import (
    OVERTURN_CONCLUSION_TYPES,
    REPORT_SCHEMA_VERSION,
)
from benchmark.attribution.components.candidate_root_cause import CandidateRootCause

logger = get_logger("agent_eval.cluster_report.schema")

_SCORE_BY_TRIAGE_LEVEL: dict[str, float] = {
    "good": 2.0,
    "suspicious_good": 2.0,
    "watch": 1.0,
    "mild_bad": 1.0,
    "severe_bad": 0.0,
}


class CaseKey(BaseModel):
    """Alignment key for cross-batch comparison, with a priority fallback.

    When ``case_id`` is present, use ``(suite, case_id, turn_index)``. Sampled cases
    repeat queries often, so only case_id is reliably unique. Without case_id, fall
    back to ``(suite, tag, query, turn_index)``, compatible with the existing
    (tag, query) alignment in ``eval_scoring.py`` / ``eval-compare.py``. ``turn_index``
    is the multi-turn component (empty for a single turn): turns of the same scenario
    share a tag, and short queries may repeat, so omitting it collides keys and
    mis-joins across batches.
    """

    suite: str = ""
    case_id: str = ""
    tag: str = ""
    query: str = ""
    turn_index: str = ""

    def normalized(self) -> tuple[str, ...]:
        """Normalize to a comparable tuple (join key)."""
        if self.case_id:
            return ("id", self.suite, self.case_id, self.turn_index)
        return ("q", self.suite, self.tag, self.query, self.turn_index)


class BucketPerf(BaseModel):
    """Per-case performance subtotal of one call stage, plus the actual model call count."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    models: dict[str, int] = Field(default_factory=dict)


class CasePerf(BaseModel):
    """Per-case performance view (from the Langfuse trace + observations)."""

    #: Historical call-stage detail, kept only for compatibility with existing stats and diagnosis.
    buckets: dict[str, BucketPerf] = Field(default_factory=dict)
    #: Performance detail aggregated by the actual Langfuse Generation model; a missing model name goes into the uncollected-model sentinel.
    models: dict[str, BucketPerf] = Field(default_factory=dict)
    #: End-to-end latency (trace.latency × 1000, truncated to int; scale aligned with generate-eval-report.py).
    e2e_ms: float | None = None
    #: First-text reply latency (trace metadata.first_text_response_time_ms).
    first_text_ms: float | None = None
    #: Collection-failure reason; non-empty means this case's performance is missing (excluded from stats, marked on the report).
    error: str = ""
    #: VUI route normalized from Langfuse trace tags (lite_sdm / lite_dag / standard).
    route_mode: str = ""


class AttributionView(BaseModel):
    """Per-case attribution view (from attribution_detail_json).

    The three states are :attr:`status`: ``none`` not triggered / ``pending`` in
    progress / ``completed`` done. The decision is whether ``root_cause`` exists (not
    only the ``attribution_status`` key in the bucket), matching the write scale of
    TraceRecordService.
    """

    status: str = "none"
    root_cause: str = ""
    root_cause_code: str = ""
    #: All root-cause codes (including the primary); old data without this field falls back to ``[root_cause_code]``.
    all_root_cause_codes: list[str] = Field(default_factory=list)
    category: str = ""
    owner_group: str = ""
    conclusion_type: str = ""
    confidence: float | None = None
    corrected_triage_level: str = ""
    needs_human_review: bool = False
    evidence_details: list[dict[str, Any]] = Field(default_factory=list)
    action_hint: str = ""
    #: Structured projection of `other:` candidates; dirty historical JSON degrades to empty and must not abort the whole batch report.
    candidate_root_cause: CandidateRootCause | None = None

    @field_validator("candidate_root_cause", mode="before")
    @classmethod
    def normalize_candidate_root_cause(cls, value: object) -> CandidateRootCause | None:
        if isinstance(value, CandidateRootCause):
            return value
        if not isinstance(value, dict):
            return None
        try:
            return CandidateRootCause.model_validate(value)
        except ValueError as exc:
            logger.warning("candidate_root_cause_degraded error=%s", exc)
            return None

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def effective_root_cause_codes(self) -> list[str]:
        """All root causes used for stats: the new field wins; old data falls back to a single-primary list."""

        if self.all_root_cause_codes:
            return self.all_root_cause_codes
        return [self.root_cause_code] if self.root_cause_code else []

    @property
    def is_pending(self) -> bool:
        return self.status == "pending"

    @property
    def is_overturn(self) -> bool:
        """Whether the evaluation conclusion was overturned (conclusion_type hits one of the three counter-evidence types)."""
        return self.conclusion_type in OVERTURN_CONCLUSION_TYPES


class BatchCaseRecord(BaseModel):
    """Merged one-row view of a case: judge + attribution + performance."""

    trace_id: str = ""
    #: Langfuse project of the original business Trace; used only by the Cluster private presentation layer to build a jump link.
    project_id: str = ""
    record_id: str = ""
    evaluator_name: str = ""
    #: Source batch this row actually belongs to; used when stability dedupes within a batch and keeps rows across batches.
    source_batch_tag: str = ""
    #: Local read view of the full evaluation IO; used only after stability L1 extracts the whitelist, and not serialized into artifacts.
    trace_meta: dict[str, Any] = Field(default_factory=dict, exclude=True)
    #: Case metadata injected by the load generator, an open dict passed through in full (source/suite/case_id/...).
    case_meta: dict[str, Any] = Field(default_factory=dict)
    case_key: CaseKey = Field(default_factory=CaseKey)

    # ── judge / effective score side ──
    #: Compatibility field: the raw PG Judge score. New consumers should read ``judge_score`` first.
    score: float | None = None
    #: Raw Judge score, not overwritten by the attribution result.
    judge_score: float | None = None
    #: Attribution-corrected score; empty when there is no valid ``corrected_triage_level``.
    attribution_score: float | None = None
    #: Score finally used by aggregation: the attribution-corrected score wins, otherwise fall back to the raw Judge score.
    effective_score: float | None = None
    reason: str = ""
    triage_level: str = ""
    dimension_results: list[dict[str, Any]] = Field(default_factory=list)
    checker_results: list[dict[str, Any]] = Field(default_factory=list)
    hard_rule_hits: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    failure_modes: list[dict[str, Any]] = Field(default_factory=list)
    judge_confidence: float | None = None

    # ── unified-envelope shared core (read directly from ``agent_eval_extra.v1``; historical non-envelope rows default to empty) ──
    #: Envelope identity slice keys of this row (business / scene / Profile version); the report slices by these.
    business_id: str = ""
    scenario: str = ""
    profile_id: str = ""
    profile_version: str = ""
    #: Independent Metric/Diagnostic observations: they do not enter effectiveness scoring, and are used only for report slices and alerts.
    metric_observations: list[dict[str, Any]] = Field(default_factory=list)
    diagnostic_findings: list[dict[str, Any]] = Field(default_factory=list)

    # ── attribution side ──
    attribution: AttributionView = Field(default_factory=AttributionView)
    #: Dispatched Aone defect ticket id; used only to dedupe stats across cases, and the dispatch payload is not passed through.
    notification_workitem_id: str = ""

    # ── performance side ──
    perf: CasePerf | None = None

    @model_validator(mode="after")
    def populate_score_views(self) -> "BatchCaseRecord":
        """Fill in the three scores; compatible with existing data and tests that only provide the old ``score`` field."""

        if self.judge_score is None:
            self.judge_score = self.score
        elif self.score is None:
            self.score = self.judge_score
        corrected_level = self.attribution.corrected_triage_level
        if self.attribution_score is None and corrected_level:
            self.attribution_score = _SCORE_BY_TRIAGE_LEVEL.get(corrected_level)
        if self.effective_score is None:
            self.effective_score = (
                self.attribution_score if self.attribution_score is not None else self.judge_score
            )
        return self

    @property
    def suite(self) -> str:
        """Basis for xlsx sheet split and stats grouping; when missing, the renderer lands on ``UNGROUPED_SHEET``."""
        return str(self.case_meta.get("suite") or "")

    @property
    def tag(self) -> str:
        return self.case_key.tag


class Coverage(BaseModel):
    """Coverage declaration — marked explicitly in the report header, so counts are not dropped silently."""

    #: Expected case count (successful dispatches), passed in by the trigger.
    expected_total: int = 0
    #: Number of judge records actually recalled.
    judged: int = 0
    #: Case count the triage let into the attribution flow (pending + completed).
    should_attribute: int = 0
    #: Count of completed attributions.
    attributed: int = 0
    #: Attributions still pending (counted in the backlog when close happens on a stall).
    pending: int = 0
    #: Count of successful performance collections (0 when includePerf=false).
    perf_collected: int = 0
    #: Why the performance block degraded (Langfuse unreachable / not enabled) — describes performance only, and does not carry watcher notes.
    perf_note: str = ""
    #: Watcher close note (judge timeout / attribution stall). Separate from perf_note: the two mean different things,
    #: and mixing them renders wrong copy such as "performance block: closed on judge timeout".
    watcher_note: str = ""

    @property
    def eval_rate(self) -> float:
        return self.judged / self.expected_total if self.expected_total else 0.0

    @property
    def attribution_rate(self) -> float:
        return self.attributed / self.should_attribute if self.should_attribute else 0.0


class L1Stats(BaseModel):
    """L1 deterministic statistics (pure code arithmetic, no LLM)."""

    effectiveness: dict[str, Any] = Field(default_factory=dict)
    attribution: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)
    crosstab: dict[str, Any] = Field(default_factory=dict)
    #: Metric/Diagnostic/business/scene/Profile-version slices (read directly from the unified envelope; empty for historical batches).
    observations: dict[str, Any] = Field(default_factory=dict)
    #: Waterfall and root-cause × dimension matrix shared with the daily report, from deterministic L1 aggregation.
    enhancements: dict[str, Any] = Field(default_factory=dict)


class ClusterReport(BaseModel):
    """Root object of the cluster-report machine-readable artifact (report.json)."""

    schema_version: str = REPORT_SCHEMA_VERSION
    batch_tag: str = ""
    #: Evaluator aggregated by this report, used as the business prefix of artifacts and dispatches.
    evaluator_name: str = ""
    #: Batch metadata passed through by the trigger (model/agent_mode/endpoint/testset/planned count...).
    batch_meta: dict[str, Any] = Field(default_factory=dict)
    coverage: Coverage = Field(default_factory=Coverage)
    stats: L1Stats = Field(default_factory=L1Stats)
    cases: list[BatchCaseRecord] = Field(default_factory=list)
    #: L2 LLM semantic-clustering output (problem clusters / feedback suggestions / Aone suggestions); an empty dict when missing.
    llm_summary: dict[str, Any] = Field(default_factory=dict)
    #: Report generation time (UTC+8 ISO).
    generated_at: str = ""
    #: Time range actually used by the online sample query; both ends are empty for an ordinary batch with no explicit window.
    data_window: dict[str, str | None] = Field(default_factory=lambda: {"start": None, "end": None})


__all__ = [
    "AttributionView",
    "BatchCaseRecord",
    "BucketPerf",
    "CaseKey",
    "CasePerf",
    "ClusterReport",
    "Coverage",
    "L1Stats",
]
