"""cluster_report_agent — L2 semantic clustering (registered alongside attribution_vehicle_agent).

Responsibility boundary (aligned with "rules first; the LLM does not do statistics"):
- **Consume only L1 results**: the input is already-computed deterministic stats, not raw case details
- **Do only semantic work**: problem-cluster summaries, cross-root-cause links, feedback suggestions, Aone suggestion sections
- **Do not recount**: every number must cite an L1 value, soft-checked by :func:`guard_llm_numbers`
  (normalize + tolerance compare; a miss only warns and is not hard-deleted, so a legal phrase is not killed)
- **Do not self-rate confidence**: confidence cites the attribution side's calibrated ``final_validation_gate`` value

When L2 is not configured or the finite retries are exhausted, raise ClusterReportL2Error and let the caller mark the report failed.
L1 numbers stay deterministic; L2 does not recompute statistics.
"""

import asyncio
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from benchmark.libs.agenthost.agents import AgentCard, AgentProviderContext, BaseAgent
from benchmark.libs.agenthost.agents.component_metadata import agent_component
from benchmark.libs.agenthost.agents.config.schemas import AgentConfigSchema
from benchmark.libs.agenthost.agents.request import (
    AgentRequest,
    AgentResponse,
    ContentPart,
    ResponseStatus,
)
from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from benchmark.reporting.cluster.constants import (
    CLUSTER_PROMPT_CONF_DIR,
    DEFAULT_CLUSTER_PROMPT_NAME,
    PROMPT_FILE_SUFFIX,
)
from benchmark.reporting.cluster.stats.stability import (
    stability_governance,
    stability_group_key,
    stability_group_sort_key,
)
from benchmark.utils.llm_json import strip_json_fence

logger = get_logger(__name__)

#: Number-token match: integer / decimal / percent / thousands separator.
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")

#: Soft-check tolerance: rounding error allowed when the LLM restates an L1 number.
#: Relative error alone is too loose on large numbers (around 100, ±0.5 would all match and would let through
#: fabrications such as "99.9%"), so an absolute-error cap is added: both must hold to count as a match.
_NUMBER_REL_TOLERANCE = 0.005
_NUMBER_ABS_TOLERANCE = 0.05

#: Small-number whitelist: common non-stat numbers such as Top-N indexes and section numbers, excluded from the check to avoid false alarms.
_IGNORED_NUMBERS = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0})
#: Cap on automatic related-trace backfill, so one problem cluster does not flood the report.
_RELATED_CASE_LIMIT = 20
_STABILITY_PROMPT_NAME = "cluster_summary_stability"
_VUI_LOW_SCORE_JOB_TYPE = "vui_low_score"
_STABILITY_CLASSIFICATIONS = frozenset(
    {
        "context_diff",
        "expected_diversity",
        "tool_data_variance",
        "nondeterminism",
        "quality_regression",
        "evaluation_inconsistency",
    }
)

#: Total L2 attempts: the first call plus two retries.
_L2_MAX_ATTEMPTS = 3
_L2_RETRY_DELAYS_S = (1.0, 2.0)


class ClusterReportL2Error(RuntimeError):
    """Raised when L2 cannot produce usable JSON within the finite retries."""


class ClusterLLM(Protocol):
    """Minimal LLM interface required by L2 clustering (satisfied by :class:`_LazyChatModel`)."""

    async def complete(self, system_prompt: str, user_prompt: str) -> object: ...


class PromptManager(Protocol):
    """Prompt repository interface (satisfied by agentgraph runtime.prompt_manager).

    The return shape varies by implementation (a string, or an object with a content
    or template attribute), so it is declared as ``object`` and ``_load_prompt`` extracts it.
    """

    def get(self, name: str) -> object: ...


class ClusterReportConfigSchema(AgentConfigSchema):
    """ext_schema for ``agents.cluster_report_agent``."""

    type: str | None = Field(default=None)
    default_model: str | None = Field(default=None)
    prompt_name: str = DEFAULT_CLUSTER_PROMPT_NAME
    #: Cap on Top-N problem clusters (controls prompt size and output length).
    max_clusters: int = 8


@dataclass(frozen=True)
class ClusterReportAgentOptions:
    """Name and output config of the clustering agent; runtime dependencies are injected separately."""

    name: str
    prompt_name: str
    max_clusters: int


@agent_component(
    name="cluster_report_agent",
    ext_schema=ClusterReportConfigSchema,
)
class ClusterReportAgent(BaseAgent):
    """Batch clustering-report agent: turn L1 stats into actionable problem clusters and suggestions."""

    def __init__(
        self,
        options: ClusterReportAgentOptions,
        *,
        llm: ClusterLLM | None,
        prompt_manager: PromptManager | None,
    ) -> None:
        super().__init__(name=options.name)
        self._llm = llm
        self._prompt_manager = prompt_manager
        self._prompt_name = options.prompt_name
        self._max_clusters = options.max_clusters

    @classmethod
    def from_config(
        cls,
        cfg: ClusterReportConfigSchema,
        ctx: AgentProviderContext,
    ) -> "ClusterReportAgent":
        runtime = ctx.runtime
        llm: Any | None = None
        if cfg.default_model:
            llm = _LazyChatModel(lambda: runtime.get_llm(cfg.default_model))
        return cls(
            options=ClusterReportAgentOptions(
                name=ctx.name, prompt_name=cfg.prompt_name, max_clusters=cfg.max_clusters
            ),
            llm=llm,
            prompt_manager=getattr(runtime, "prompt_manager", None),
        )

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.name,
            description=(
                "批次评测聚类报告 agent — 只吃 L1 确定性统计，做问题簇归纳与回流建议，"
                "不重新计数、不改 root_cause_code 分组"
            ),
        )

    async def run(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        return await self.call(request, **kwargs)

    async def call(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        """Input is L1 stats JSON; output is semantic-clustering result JSON."""

        stats = _extract_stats(request)
        summary = await self.summarize(stats)
        return AgentResponse(
            traceid=getattr(request, "traceid", "") or "",
            status=ResponseStatus.SUCCESS,
            content_parts=[
                ContentPart(type="text", data=json.dumps(summary, ensure_ascii=False)),
            ],
        )

    async def summarize(self, l1_stats: dict[str, Any]) -> dict[str, Any]:
        """Summarize L1 stats into problem clusters, feedback suggestions, and Aone suggestions.

        L2 is a prerequisite of a successful normal/stability report: a missing model, a
        failed call, or illegal JSON raises after finite retries, and job orchestration
        marks the job failed.
        """

        job_type = str(l1_stats.get("report_job_type") or "normal")
        if self._llm is None:
            raise ClusterReportL2Error("cluster report L2 model is not configured")

        system_prompt = self._load_prompt(job_type)
        user_prompt = _build_user_prompt(l1_stats, max_clusters=self._max_clusters)
        summary = await self._complete_summary(system_prompt, user_prompt)
        if job_type == "stability":
            summary = _normalize_stability_summary(summary, l1_stats)
        else:
            _isolate_candidate_cases_from_formal_output(summary, l1_stats)
            # The LLM may cite only case_id (see _trim_stats); trace_id is filled back by a program-side lookup.
            # A miss (fabricated by the LLM, or a mistyped case_id) is dropped, so a forged trace_id cannot enter the report.
            _resolve_related_case_trace_ids(summary, l1_stats)
            _resolve_candidate_cluster_trace_ids(summary, l1_stats)
        # T31 soft check: numbers must come from L1; a miss is marked as a warning rather than deleting the phrase.
        warnings = guard_llm_numbers(summary, l1_stats)
        if warnings:
            summary["number_warnings"] = warnings
            logger.warning(
                "cluster_report.number_guard_hits count=%d sample=%s",
                len(warnings),
                warnings[:3],
            )
        if job_type != "stability":
            _sync_affected_counts(summary)
        return summary

    async def _complete_summary(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call L2 and retry a finite number of times with jitter on exceptions or illegal JSON."""

        assert self._llm is not None
        failure = "unknown"
        for attempt in range(_L2_MAX_ATTEMPTS):
            try:
                summary = _parse_summary(await self._llm.complete(system_prompt, user_prompt))
                if not summary.get("parse_error"):
                    return summary
                failure = str(summary["parse_error"])
            except Exception as exc:  # noqa: BLE001 - the Provider error type is decided by the implementation
                failure = type(exc).__name__
                logger.warning(
                    "cluster_report.l2_call_failed attempt=%d error_type=%s detail=%r",
                    attempt + 1,
                    failure,
                    str(exc)[:500],
                )
            if attempt == _L2_MAX_ATTEMPTS - 1:
                break
            delay = _L2_RETRY_DELAYS_S[attempt] + random.uniform(0.0, 1.0)
            logger.warning(
                "cluster_report.l2_retry attempt=%d err=%s delay_s=%.3f",
                attempt + 1,
                failure,
                delay,
            )
            await asyncio.sleep(delay)
        raise ClusterReportL2Error(
            f"cluster report L2 failed after {_L2_MAX_ATTEMPTS} attempts: {failure}"
        )

    def _load_prompt(self, job_type: str = "normal") -> str:
        """Load the system prompt: prompt_manager first, then a .prompt file under conf.

        Same pattern as the attribution side (see prompt loading in attribution/agent.py).
        When both are missing, use the built-in default prompt so semantic clustering is
        not fully unavailable just because config is missing.
        """

        base = ""
        prompt_name = _STABILITY_PROMPT_NAME if job_type == "stability" else self._prompt_name
        manager = self._prompt_manager
        if manager is not None:
            try:
                loaded = manager.get(prompt_name)
                text = (
                    getattr(loaded, "content", None) or getattr(loaded, "template", None) or loaded
                )
                base = str(text or "")
            except Exception:  # noqa: BLE001 - a missing prompt must not interrupt the report
                base = ""
        if not base:
            path = (
                Path(__file__)
                .resolve()
                .parents[2]
                .joinpath(
                    *CLUSTER_PROMPT_CONF_DIR,
                    f"{prompt_name}{PROMPT_FILE_SUFFIX}",
                )
            )
            if path.is_file():
                base = path.read_text(encoding="utf-8")
        if job_type == _VUI_LOW_SCORE_JOB_TYPE:
            return (base.strip() or _FALLBACK_SYSTEM_PROMPT) + _VUI_LOW_SCORE_SYSTEM_APPENDIX
        if base.strip():
            return base.strip()
        if job_type == "stability":
            return _FALLBACK_STABILITY_SYSTEM_PROMPT
        return _FALLBACK_SYSTEM_PROMPT


class _LazyChatModel:
    """Light adapter that builds the chat model lazily (aligned with ChatModelAttributionLLM on the attribution side).

    Output is accumulated from a stream. A non-streaming long generation has no bytes
    on the connection while waiting for response headers, and the gateway idle timeout
    (~180s) cuts it (httpx.ReadError → ProviderError). Streaming chunks keep the
    connection alive and avoid that. Retries are handled uniformly by ``ClusterReportAgent``.
    """

    def __init__(self, factory: Callable[[], object]) -> None:
        self._factory = factory
        self._model: object | None = None

    async def complete(self, system_prompt: str, user_prompt: str) -> object:
        from benchmark.attribution.badcase.model_client import (  # noqa: PLC0415
            maybe_await,
        )

        if self._model is None:
            self._model = self._factory()
        model = await maybe_await(self._model)
        return await self._invoke_once(model, system_prompt, user_prompt)

    async def _invoke_once(self, model: object, system_prompt: str, user_prompt: str) -> object:
        from benchmark.attribution.badcase.model_client import (  # noqa: PLC0415
            ModelMessage,
            call_model,
        )

        astream = getattr(model, "astream", None)
        if not callable(astream):
            return await call_model(model, system_prompt, user_prompt)
        messages = [
            ModelMessage(role="system", content=system_prompt).model_dump(mode="json"),
            ModelMessage(role="user", content=user_prompt).model_dump(mode="json"),
        ]
        parts: list[str] = []
        async for chunk in astream(messages):
            content = getattr(chunk, "content", None)
            if isinstance(content, str):
                parts.append(content)
        return "".join(parts)


def guard_llm_numbers(
    summary: dict[str, Any],
    l1_stats: dict[str, Any],
) -> list[str]:
    """Soft check: numbers in the LLM output should be findable in the L1 stats.

    Design tradeoff (second review): **warn only, do not hard-delete**. Number formats
    vary (percent / decimal / thousands separator), and hard deletion would kill legal
    phrases (such as "Top 3 problem clusters" or "D0-D6, 7 dimensions in total"). So:
    - Normalize: drop thousands commas; a percent sign keeps both the raw value and value/100
    - Tolerance: 0.5% relative error (allows rounding when the LLM restates a number)
    - Whitelist: common indexes/bases such as 0-10 and 100 are not checked

    Returns the list of number literals that missed (the report marks them as ``number_warnings``).
    """

    l1_values = _collect_numbers(l1_stats)
    if not l1_values:
        return []

    warnings: list[str] = []
    for literal in _iter_number_literals(summary):
        for candidate in _normalize_number(literal):
            if candidate in _IGNORED_NUMBERS:
                break
            if _matches_any(candidate, l1_values):
                break
        else:
            warnings.append(literal)
    return warnings


def _matches_any(value: float, l1_values: set[float]) -> bool:
    """Whether value hits the L1 value set (exact equality, or within both tolerances)."""

    for known in l1_values:
        if value == known:
            return True
        diff = abs(value - known)
        if diff > _NUMBER_ABS_TOLERANCE:
            continue
        scale = max(abs(known), abs(value), 1e-9)
        if diff / scale <= _NUMBER_REL_TOLERANCE:
            return True
    return False


def _normalize_number(literal: str) -> list[float]:
    """Normalize a number literal into a list of candidate values.

    ``"12.5%"`` → ``[12.5, 0.125]``: a ratio in L1 may be stored as 0.125 or as 12.5.
    Either counts as a hit, so a unit-scale difference is not a false alarm.
    """

    cleaned = literal.replace(",", "").strip()
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return []
    return [value, value / 100] if is_percent else [value]


def _iter_number_literals(payload: object) -> list[str]:
    """Recursively collect number literals in the summary.

    Scan both numbers inside strings and JSON numeric fields (such as
    ``affected_count: 777``). The latter is also a place the LLM may fabricate, and
    skipping it would make the check a no-op.
    """

    literals: list[str] = []
    if isinstance(payload, bool):
        return literals
    if isinstance(payload, (int, float)):
        literals.append(_format_number(payload))
    elif isinstance(payload, str):
        literals.extend(_NUMBER_RE.findall(payload))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            # number_warnings / raw are not checked: the former avoids a self-referential loop,
            # and the latter is the raw text kept when parsing fails (it is not itself a conclusion).
            if key in ("number_warnings", "raw", "parse_error", "skipped"):
                continue
            literals.extend(_iter_number_literals(value))
    elif isinstance(payload, list):
        for item in payload:
            literals.extend(_iter_number_literals(item))
    return literals


def _format_number(value: float) -> str:
    """Number → literal (integers have no decimal point, so warning text stays readable)."""

    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return str(value)


def _collect_numbers(payload: object) -> set[float]:
    """Recursively collect every number in the L1 stats (including numbers stored as strings).

    Ratios and percents recognize each other (``0.85`` is also registered as ``85.0``),
    so an LLM writing "85%" is not a false miss. **This expansion applies only to ratios
    in 0~1**: multiplying every number by 100 would inflate count metrics for no reason
    (``overturn_count=4`` would also match 400) and make the guard a no-op.
    """

    values: set[float] = set()
    if isinstance(payload, bool):
        return values
    if isinstance(payload, (int, float)):
        value = float(payload)
        values.add(value)
        if abs(value) <= 1.0:
            values.add(round(value * 100, 6))
    elif isinstance(payload, str):
        for literal in _NUMBER_RE.findall(payload):
            values.update(_normalize_number(literal))
    elif isinstance(payload, dict):
        for value_item in payload.values():
            values |= _collect_numbers(value_item)
    elif isinstance(payload, list):
        for item in payload:
            values |= _collect_numbers(item)
    return values


def _extract_stats(request: AgentRequest) -> dict[str, Any]:
    """Parse L1 stats JSON from the contents of an AgentRequest.

    Accepts both ``content_parts`` (the strongly typed path) and ``metadata`` (passed straight by the orchestration layer).
    """

    for part in getattr(request, "content_parts", None) or []:
        raw = getattr(part, "data", None)
        if not raw:
            continue
        parsed = _try_json(raw)
        if isinstance(parsed, dict):
            return parsed
    metadata = getattr(request, "metadata", None) or {}
    stats = metadata.get("l1_stats") if isinstance(metadata, dict) else None
    return stats if isinstance(stats, dict) else {}


def _try_json(raw: object) -> object:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(strip_json_fence(str(raw)))
    except (json.JSONDecodeError, TypeError):
        return None


class _SummaryItem(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    related_cases: list[str] = Field(default_factory=list)
    root_cause_codes: list[str] = Field(default_factory=list)
    representative_traces: list[str] = Field(default_factory=list)
    title: str = ""
    summary: str = ""
    evidence: str = ""


class _SummarySchema(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    headline: str = ""
    clusters: list[_SummaryItem] = Field(default_factory=list)
    candidate_clusters: list[_SummaryItem] = Field(default_factory=list)
    findings: list[_SummaryItem] = Field(default_factory=list)
    governance_queue: list[_SummaryItem] = Field(default_factory=list)
    feedback_suggestions: list[_SummaryItem] = Field(default_factory=list)
    notification_suggestions: list[_SummaryItem] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


def _parse_summary(raw: object) -> dict[str, Any]:
    """Parse LLM output into a summary dict; on parse failure keep the raw text for a human to read."""

    text = raw if isinstance(raw, str) else getattr(raw, "content", None) or str(raw)
    try:
        parsed = json.loads(strip_json_fence(str(text)))
    except (json.JSONDecodeError, TypeError):
        return {
            "parse_error": "invalid_json",
            "raw": str(text)[:2000],
            "clusters": [],
            "candidate_clusters": [],
        }
    if not isinstance(parsed, dict):
        return {
            "parse_error": "not_an_object",
            "raw": str(text)[:2000],
            "clusters": [],
            "candidate_clusters": [],
        }
    try:
        _SummarySchema.model_validate(parsed)
    except ValidationError:
        return {
            "parse_error": "invalid_structure",
            "raw": str(text)[:2000],
            "clusters": [],
            "candidate_clusters": [],
        }
    parsed.setdefault("clusters", [])
    parsed.setdefault("candidate_clusters", [])
    return parsed


def _normalize_stability_summary(
    summary: dict[str, Any],
    l1_stats: dict[str, Any],
) -> dict[str, Any]:
    """Normalize stability L2 explanations against the L1 groups, and fill in the full governance queue."""

    groups = [
        group
        for group in ((l1_stats.get("observations") or {}).get("stability") or [])
        if isinstance(group, dict)
    ]
    groups_by_key = {stability_group_key(group): group for group in groups}
    allowed_traces = _stability_trace_ids(l1_stats)
    findings: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in summary.get("findings") or []:
        if not isinstance(item, dict):
            continue
        key = stability_group_key(
            {
                "scene_tag": str(item.get("scene_tag") or ""),
                "query": str(item.get("query") or ""),
            }
        )
        group = groups_by_key.get(key)
        if group is None or key in seen_keys:
            continue
        classification = str(group.get("agent_execution_observation") or "")
        if classification not in _STABILITY_CLASSIFICATIONS:
            continue
        seen_keys.add(key)
        findings.append(_stability_finding_from_group(item, group, allowed_traces))

    findings.sort(key=_stability_finding_sort_key)
    governance_queue = [
        _stability_governance_item(group, findings, allowed_traces)
        for group in sorted(groups, key=stability_group_sort_key)
        if str(group.get("agent_execution_observation") or "") != "stable"
    ]
    governance_queue.sort(key=_stability_finding_sort_key)
    normalized = {
        "headline": str(summary.get("headline") or ""),
        "findings": findings[:5],
        "governance_queue": governance_queue,
        "caveats": [str(item) for item in summary.get("caveats") or [] if item],
    }
    for key in ("skipped", "parse_error", "raw"):
        if summary.get(key):
            normalized[key] = summary[key]
    return normalized


def _stability_finding_from_group(
    item: dict[str, Any],
    group: dict[str, Any],
    allowed_traces: set[str],
) -> dict[str, Any]:
    """Keep only the LLM's prose explanation; category, priority, and governance fields are overwritten by the L1 group."""

    traces = [
        trace_id
        for trace_id in item.get("representative_traces") or []
        if isinstance(trace_id, str)
        and trace_id in allowed_traces
        and trace_id in _group_trace_ids(group)
    ]
    if not traces:
        traces = _group_trace_ids(group)
    observation = str(group.get("agent_execution_observation") or "insufficient_evidence")
    governance = stability_governance(observation, str(group.get("evidence_strength") or "weak"))
    return {
        "classification": observation,
        "scene_tag": str(group.get("scene_tag") or ""),
        "query": str(group.get("query") or ""),
        "evidence_strength": str(group.get("evidence_strength") or "weak"),
        "priority": governance["priority"],
        "reason": governance["reason"],
        "governance": governance,
        "summary": str(item.get("summary") or governance["reason"]),
        "evidence": str(item.get("evidence") or _stability_group_evidence(group)),
        "representative_traces": traces,
        "governance_action": _governance_action(governance),
    }


def _stability_governance_item(
    group: dict[str, Any],
    findings: list[dict[str, Any]],
    allowed_traces: set[str],
) -> dict[str, Any]:
    """Keep a governance action for every non-stable L1 group, even if the LLM did not pick it as a Top finding."""

    key = stability_group_key(group)
    finding = next(
        (item for item in findings if stability_group_key(item) == key),
        None,
    )
    if finding is not None:
        return finding
    observation = str(group.get("agent_execution_observation") or "insufficient_evidence")
    governance = stability_governance(observation, str(group.get("evidence_strength") or "weak"))
    return {
        "classification": observation,
        "scene_tag": str(group.get("scene_tag") or ""),
        "query": str(group.get("query") or ""),
        "evidence_strength": str(group.get("evidence_strength") or "weak"),
        "priority": governance["priority"],
        "reason": governance["reason"],
        "governance": governance,
        "summary": governance["reason"],
        "evidence": _stability_group_evidence(group),
        "representative_traces": [
            trace_id for trace_id in _group_trace_ids(group) if trace_id in allowed_traces
        ],
        "governance_action": _governance_action(governance),
    }


def _stability_finding_sort_key(finding: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        {"P0": 0, "P1": 1, "P2": 2}.get(str(finding.get("priority") or "P2"), 2),
        {"strong": 0, "medium": 1, "weak": 2}.get(
            str(finding.get("evidence_strength") or "weak"), 2
        ),
        str(finding.get("scene_tag") or ""),
        str(finding.get("query") or ""),
    )


def _group_trace_ids(group: dict[str, Any]) -> list[str]:
    return [
        str(run.get("trace_id") or "")
        for run in group.get("representative_runs") or []
        if isinstance(run, dict) and run.get("trace_id")
    ]


def _stability_group_evidence(group: dict[str, Any]) -> str:
    distribution = group.get("judge_score_distribution") or {}
    score_counts = "/".join(str(distribution.get(score, 0)) for score in ("0", "1", "2"))
    return (
        f"运行次数={group.get('run_count', 0)}；judge 0/1/2={score_counts}；"
        f"span={group.get('span', 0)}。"
    )


def _governance_action(governance: dict[str, Any]) -> str:
    if governance.get("is_agent_defect"):
        return "fix"
    if governance.get("priority") in {"P0", "P1"}:
        return "investigate"
    return "observe"


def _stability_trace_ids(l1_stats: dict[str, Any]) -> set[str]:
    groups = (l1_stats.get("observations") or {}).get("stability") or []
    return {
        str(run.get("trace_id") or "")
        for group in groups
        if isinstance(group, dict)
        for run in group.get("representative_runs") or []
        if isinstance(run, dict) and run.get("trace_id")
    }


def _build_user_prompt(l1_stats: dict[str, Any], *, max_clusters: int) -> str:
    """Build the user-side input: trimmed L1 stats JSON plus the output requirements.

    Why trim: the detail ``cases`` list is large and most fields (root_cause/score/...)
    add nothing to the semantic summary (the LLM only needs the distribution and Top-N).
    Keep only ``case_id`` (for ``clusters[].related_cases`` /
    ``candidate_clusters[].related_cases`` / ``aone_suggestions[].related_cases``).
    trace_id is not exposed, so the LLM cannot fabricate one; the real trace_id is
    filled back by a program-side lookup on case_id.
    """

    trimmed = _trim_stats(l1_stats)
    job_type = str(trimmed.get("report_job_type") or "normal")
    instruction = (
        "这是 stability 报告：根据 request/env ContextDiff、执行产出和评分证据判断 Agent "
        "执行稳定性；不生成 aone_suggestions。只解释输入中已有的 group，最多输出 5 个 "
        "优先治理 finding；每项说明差异原因、责任域和治理建议，"
        "不将评测、输入或证据问题误报为 Agent 缺陷。"
        if job_type == "stability"
        else (
            "这是 VUI 原始评测低分报告：只根据 evaluation_low_score_analysis 中的原始评测分、"
            "维度扣分与评测原因归纳问题簇；不得引用或推断归因、复核、推翻、unknown、"
            "root_cause、owner 或 Aone 信息。related_cases 只能引用其中 cases 的 case_id。"
            if job_type == _VUI_LOW_SCORE_JOB_TYPE
            else "这是 normal 报告：聚焦单批次问题归因与可执行改进建议。"
        )
    )
    output_limit = 5 if job_type == "stability" else max_clusters
    return (
        "以下是评测的确定性统计（L1，由代码算出，不可修改）：\n"
        f"```json\n{json.dumps(trimmed, ensure_ascii=False, indent=2)}\n```\n\n"
        f"{instruction}\n请输出至多 {output_limit} 个问题簇。所有数字必须直接引用上面的 "
        "L1 原值，不要自行换算百分比或重新统计。"
    )


def _trim_stats(l1_stats: dict[str, Any]) -> dict[str, Any]:
    """Trim per-case details: keep only ``case_id`` for the LLM to cite, and hide trace_id and the other large fields.

    ``cases`` is not removed entirely. L2 ``clusters[].related_cases`` and
    ``aone_suggestions[].related_cases`` need to cite a concrete case, but the LLM must
    not see or fabricate trace_id (trace_id is filled back by
    :func:`_resolve_related_case_trace_ids` via a program-side lookup on case_id).
    """

    # VUI low-score clustering must keep the original score evidence; other details keep only the short id.
    trimmed = _trim_cases_recursively(l1_stats)
    if l1_stats.get("report_job_type") == _VUI_LOW_SCORE_JOB_TYPE:
        analysis = l1_stats.get("evaluation_low_score_analysis") or {}
        if isinstance(analysis, dict) and isinstance(analysis.get("cases"), list):
            trimmed["evaluation_low_score_analysis"]["cases"] = [
                {
                    key: case[key]
                    for key in ("case_id", "judge_score", "reason", "dimension_deductions")
                    if key in case
                }
                for case in analysis["cases"]
                if isinstance(case, dict) and case.get("case_id")
            ]
    return trimmed


def _trim_cases_recursively(payload: object) -> object:
    if isinstance(payload, dict):
        if isinstance(payload.get("cases"), list):
            return {
                k: (_case_ids_only(v) if k == "cases" else _trim_cases_recursively(v))
                for k, v in payload.items()
            }
        return {k: _trim_cases_recursively(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_trim_cases_recursively(item) for item in payload]
    return payload


def _case_ids_only(cases: object) -> list[dict[str, str]]:
    """Compress a detail case list to ``case_id`` only (the only key the LLM may cite)."""

    if not isinstance(cases, list):
        return []
    return [
        {"case_id": str(case["case_id"])}
        for case in cases
        if isinstance(case, dict) and case.get("case_id")
    ]


def _collect_case_id_to_trace_id(l1_stats: dict[str, Any]) -> dict[str, str]:
    """Build a case_id → trace_id map from the untrimmed L1 stats (for the program-side lookup)."""

    mapping: dict[str, str] = {}
    _collect_case_id_to_trace_id_recursively(l1_stats, mapping, set())
    return mapping


def _collect_case_id_to_trace_id_recursively(
    payload: object, mapping: dict[str, str], ambiguous: set[str]
) -> None:
    if isinstance(payload, dict):
        cases = payload.get("cases")
        if isinstance(cases, list):
            for case in cases:
                if not isinstance(case, dict):
                    continue
                _index_case_reference(case, mapping, ambiguous)
        for value in payload.values():
            _collect_case_id_to_trace_id_recursively(value, mapping, ambiguous)
    elif isinstance(payload, list):
        for item in payload:
            _collect_case_id_to_trace_id_recursively(item, mapping, ambiguous)


def _index_case_reference(
    case: dict[str, Any], mapping: dict[str, str], ambiguous: set[str]
) -> None:
    """Reject a duplicate short id across suite/turn, and do not pick the last trace."""
    case_id = str(case.get("case_id") or "")
    trace_id = str(case.get("trace_id") or "")
    if not case_id or not trace_id or case_id in ambiguous:
        return
    if case_id in mapping and mapping[case_id] != trace_id:
        mapping.pop(case_id)
        ambiguous.add(case_id)
        return
    mapping[case_id] = trace_id


def _collect_root_cause_to_trace_ids(l1_stats: dict[str, Any]) -> dict[str, list[str]]:
    """Build a root_cause_code → trace_id list from attribution.root_cause_cases."""

    result: dict[str, list[str]] = {}
    root_cause_cases = (l1_stats.get("attribution") or {}).get("root_cause_cases")
    if not isinstance(root_cause_cases, dict):
        return result
    for code, payload in root_cause_cases.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
            continue
        traces: list[str] = []
        seen: set[str] = set()
        for case in payload["cases"]:
            if not isinstance(case, dict):
                continue
            trace_id = str(case.get("trace_id") or "")
            if trace_id and trace_id not in seen:
                seen.add(trace_id)
                traces.append(trace_id)
        if traces:
            result[str(code)] = traces
    return result


def _trace_ids_for_root_causes(
    root_cause_codes: object,
    root_cause_to_trace_ids: dict[str, list[str]],
    *,
    limit: int = _RELATED_CASE_LIMIT,
) -> list[str]:
    if not isinstance(root_cause_codes, list):
        return []
    traces: list[str] = []
    seen: set[str] = set()
    for code in root_cause_codes:
        for trace_id in root_cause_to_trace_ids.get(str(code), []):
            if trace_id not in seen:
                seen.add(trace_id)
                traces.append(trace_id)
            if len(traces) >= limit:
                return traces
    return traces


def _effective_fallback_limit(item: dict[str, Any]) -> int:
    """Compute the trace truncation cap of the fallback path.

    Prefer the LLM's ``affected_count`` (the LLM's judgment of cluster size is more
    accurate than expanding everything). If ``affected_count`` is missing or invalid,
    fall back to the global cap ``_RELATED_CASE_LIMIT``.
    """
    count = item.get("affected_count")
    if isinstance(count, int) and count > 0:
        return min(count, _RELATED_CASE_LIMIT)
    return _RELATED_CASE_LIMIT


def _resolve_related_case_trace_ids(summary: dict[str, Any], l1_stats: dict[str, Any]) -> None:
    """Fill related traces back onto the LLM output, without depending on the LLM citing cases itself.

    Prefer ``related_cases`` from the LLM output (it may cite only case_ids that
    ``_trim_stats`` exposed; a miss — fabricated or mistyped — is dropped). If that is
    empty, look up trace_ids from the L1 deterministic details by ``root_cause_codes``
    (not depending on LLM output), capped at ``_RELATED_CASE_LIMIT``. Neither path
    leaks trace_id to the LLM.

    Path B (fallback) also truncates with the LLM's ``affected_count``: one
    root_cause_code may have dozens of cases in L1, but the LLM cluster only treats
    some of them as members. Expanding all of them would make related traces far
    larger than the cluster.
    """

    candidate_traces = {
        trace_id
        for source in _collect_candidate_cluster_sources(l1_stats).values()
        for trace_id in source["trace_ids"]
    }
    case_id_to_trace_id = {
        ref: trace_id
        for ref, trace_id in _collect_case_id_to_trace_id(l1_stats).items()
        if trace_id not in candidate_traces
    }
    root_cause_to_trace_ids = {
        code: [trace_id for trace_id in traces if trace_id not in candidate_traces]
        for code, traces in _collect_root_cause_to_trace_ids(l1_stats).items()
    }
    for items in (summary.get("clusters") or [], summary.get("aone_suggestions") or []):
        for item in items:
            if not isinstance(item, dict):
                continue
            refs = item.get("related_cases") or []
            resolved = [
                case_id_to_trace_id[ref]
                for ref in refs
                if isinstance(ref, str) and ref in case_id_to_trace_id
            ]
            if not resolved and not refs:
                # fallback: look up the full set by root_cause_codes, but truncate with the LLM's
                # affected_count, so cases of the same root cause that belong to other clusters are not over-included.
                limit = _effective_fallback_limit(item)
                resolved = _trace_ids_for_root_causes(
                    item.get("root_cause_codes"),
                    root_cause_to_trace_ids,
                    limit=limit,
                )
            item["related_cases"] = resolved[:_RELATED_CASE_LIMIT]


def _collect_candidate_cluster_sources(
    l1_stats: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index deterministic cases/traces by ``(agent_scope, candidate_key)``.

    Candidate root causes may be clustered only inside their own scope. The index comes
    from L1 ``other_candidate_stats``. The LLM must not fabricate a candidate key or
    merge candidates from different scopes.
    """
    candidates = ((l1_stats.get("attribution") or {}).get("other_candidate_stats") or {}).get(
        "candidates"
    )
    if not isinstance(candidates, list):
        return {}
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        scope = str(candidate.get("agent_scope") or "")
        key = str(candidate.get("candidate_key") or "")
        cases = candidate.get("cases")
        if not scope or not key or not isinstance(cases, list):
            continue
        case_id_to_trace_id: dict[str, str] = {}
        ambiguous: set[str] = set()
        trace_ids: list[str] = []
        seen: set[str] = set()
        for case in cases:
            if not isinstance(case, dict):
                continue
            trace_id = str(case.get("trace_id") or "")
            _index_case_reference(case, case_id_to_trace_id, ambiguous)
            if trace_id and trace_id not in seen:
                seen.add(trace_id)
                trace_ids.append(trace_id)
        sources[(scope, key)] = {
            "case_id_to_trace_id": case_id_to_trace_id,
            "trace_ids": trace_ids,
        }
    return sources


def _isolate_candidate_cases_from_formal_output(
    summary: dict[str, Any],
    l1_stats: dict[str, Any],
) -> None:
    """Remove cases that point at candidate root causes from formal clusters and Aone suggestions.

    Prompt constraints are not enough to stop the LLM from wrapping an ``other:``
    candidate as an ``unclassified`` suggestion. Before the case_id → trace_id backfill,
    isolate them deterministically against the L1 candidate-case set.
    """
    candidate_case_ids = {
        case_id
        for source in _collect_candidate_cluster_sources(l1_stats).values()
        for case_id in source["case_id_to_trace_id"]
    }
    if not candidate_case_ids:
        return

    root_cause_cases = (l1_stats.get("attribution") or {}).get("root_cause_cases") or {}
    formal_root_cause_codes = {
        str(code)
        for code, payload in root_cause_cases.items()
        if isinstance(payload, dict)
        and any(
            str(case.get("case_id") or "") not in candidate_case_ids
            for case in (payload.get("cases") or [])
            if isinstance(case, dict)
        )
    }
    formal_clusters: list[dict[str, Any]] = []
    for item in summary.get("clusters") or []:
        if not isinstance(item, dict):
            continue
        codes = [str(code) for code in item.get("root_cause_codes") or [] if code]
        if codes and not set(codes).issubset(formal_root_cause_codes):
            logger.warning("cluster_report.formal_cluster_dropped_candidate_root_causes")
            continue
        # Keep the model's original citations; the backfill stage rejects candidates and ambiguous ids uniformly.
        # Clearing them early would treat an explicitly invalid citation as a missing one and turn the root-cause fallback back on.
        formal_clusters.append(item)
    summary["clusters"] = formal_clusters

    suggestions: list[dict[str, Any]] = []
    for item in summary.get("aone_suggestions") or []:
        if not isinstance(item, dict):
            continue
        refs = item.get("related_cases") or []
        remaining = [
            case_id
            for case_id in refs
            if isinstance(case_id, str) and case_id not in candidate_case_ids
        ]
        if refs and not remaining:
            logger.warning("cluster_report.aone_suggestion_dropped_candidate_cases")
            continue
        item["related_cases"] = remaining
        suggestions.append(item)
    summary["aone_suggestions"] = suggestions


def _resolve_candidate_cluster_trace_ids(summary: dict[str, Any], l1_stats: dict[str, Any]) -> None:
    """Validate a candidate cluster's scope/key and fill back its deterministic traces.

    ``candidate_clusters`` is a separate governance channel: each item may cite only a
    single ``agent_scope + candidate_key`` that exists in L1. Invalid or cross-scope
    items are dropped, and they never enter the formal ``clusters`` / Aone child-ticket path.
    """
    sources = _collect_candidate_cluster_sources(l1_stats)
    valid: list[dict[str, Any]] = []
    for item in summary.get("candidate_clusters") or []:
        if not isinstance(item, dict):
            continue
        scope = str(item.get("agent_scope") or "")
        key = str(item.get("candidate_key") or "")
        source = sources.get((scope, key))
        if source is None:
            logger.warning(
                "cluster_report.candidate_cluster_dropped scope=%s key=%s",
                scope or "-",
                key or "-",
            )
            continue
        refs = item.get("related_cases") or []
        case_id_to_trace_id = source["case_id_to_trace_id"]
        resolved = [
            case_id_to_trace_id[ref]
            for ref in refs
            if isinstance(ref, str) and ref in case_id_to_trace_id
        ]
        if not resolved and not refs:
            limit = _effective_fallback_limit(item)
            resolved = source["trace_ids"][:limit]
        item["related_cases"] = resolved[:_RELATED_CASE_LIMIT]
        valid.append(item)
    summary["candidate_clusters"] = valid


def _sync_affected_counts(summary: dict[str, Any]) -> None:
    """Sync ``affected_count`` to the actual length of ``related_cases``.

    Must be called after ``guard_llm_numbers``: the number check first sees the LLM's
    raw value (to catch a fabricated ``affected_count``), then deterministic data
    overwrites it, so the title count and the related-trace list match at render time.
    """
    for items in (
        summary.get("clusters") or [],
        summary.get("candidate_clusters") or [],
        summary.get("aone_suggestions") or [],
    ):
        for item in items:
            if isinstance(item, dict):
                item["affected_count"] = len(item.get("related_cases") or [])


_FALLBACK_STABILITY_SYSTEM_PROMPT = (
    "你是车机 Agent 执行稳定性分析专家。只依据 L1 的 request/env ContextDiff、"
    "输出证据和评分分布判断执行稳定性。分类只能是 context_diff、expected_diversity、"
    "tool_data_variance、nondeterminism、quality_regression、evaluation_inconsistency。"
    "禁止生成 aone_suggestions 或子工单，禁止输出 DTO、工具原始入参/返回和 env 原始值。"
    '输出 {"headline":"","findings":[],"governance_queue":[],"caveats":[]}。'
)


_VUI_LOW_SCORE_SYSTEM_APPENDIX = (
    "\n\nVUI 原始评测低分聚类附加约束：\n"
    "- 只使用 `evaluation_low_score_analysis` 字段，禁止使用或提及归因、复核、推翻、"
    "unknown、root cause、责任方或 Aone。\n"
    "- 每个问题簇的 related_cases 只能引用该字段 `cases` 内的 case_id；"
    "没有关联 case 时输出空数组。\n"
    "- 共性与依据只能描述原始评测分、评测维度扣分和评测原因。\n"
)


_FALLBACK_SYSTEM_PROMPT = """你是车机评测聚类分析专家。基于给定的 L1 确定性统计，
输出可行动的问题簇与建议。

严格约束：
1. 所有数字必须直接引用输入中的 L1 原值，禁止重新计数、禁止自行换算百分比
2. 禁止改动 root_cause_code 的分组（分组由规则确定）
3. 禁止自评置信度（置信度已由归因侧校准）
4. 只做语义归纳：同类根因的共性描述、跨根因的隐含关联、修复建议
5. `other_candidate_stats` 只能写入独立 candidate_clusters；每项必须照抄一个已有
   agent_scope 与 candidate_key，只能引用该 key 自身 cases，不得跨 scope/key 合并，
   禁止写入 clusters/aone_suggestions
6. related_cases 只能引用输入 cases 列表里出现过的 case_id 原值，禁止编造、禁止输出
   trace_id 或其他形式的标识；没有可确定关联的 case 就填空数组 []。clusters、
   candidate_clusters 与 aone_suggestions 均应输出 related_cases

输出 JSON：
{
  "clusters": [
    {"title": "问题簇标题", "root_cause_codes": ["..."], "affected_count": 0,
     "owner_group": "...", "summary": "共性描述", "evidence": "引用的 L1 指标",
     "related_cases": ["case_id"]}
  ],
  "candidate_clusters": [
    {"title": "候选新问题簇", "agent_scope": "...", "candidate_key": "...",
     "affected_count": 0, "summary": "共性描述", "evidence": "引用的 L1 指标",
     "related_cases": ["case_id"]}
  ],
  "feedback_suggestions": [
    {"target": "prompt|checker|dataset|business", "action": "具体修复动作", "rationale": "依据"}
  ],
  "aone_suggestions": [
    {"title": "工作项标题", "priority": "P0|P1|P2", "owner_group": "...",
     "acceptance": "验收指标", "related_cases": ["case_id"]}
  ]
}
"""


__all__ = ["ClusterReportAgent", "ClusterReportConfigSchema", "guard_llm_numbers"]
