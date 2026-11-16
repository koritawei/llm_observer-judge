"""Stability L1: organize multi-run evaluation evidence into Agent execution-stability observations.

Input context consists only of request and env. DTOs, tool calls, and tool returns are
execution outputs; they may appear only as redacted output evidence and must not participate
in the ContextDiff decision.
"""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from benchmark.reporting.cluster.schema import BatchCaseRecord

_MIN_RUNS = 2
_MAX_REPRESENTATIVE_RUNS = 2
_PREVIEW_LIMIT = 240
_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2}
_EVIDENCE_RANK = {"strong": 0, "medium": 1, "weak": 2}
_GOVERNANCE_BY_OBSERVATION: dict[str, dict[str, Any]] = {
    "stable": {
        "priority": "P2",
        "reason": "相同场景与请求的多次运行结果一致。",
        "domain": "无需治理",
        "action": "持续观察",
        "is_agent_defect": False,
    },
    "context_diff": {
        "priority": "P2",
        "reason": "多次运行的 request 或 env 存在差异，结果不可直接视为 Agent 波动。",
        "domain": "评测输入与数据",
        "action": "核查并对齐评测集、调用输入与运行环境。",
        "is_agent_defect": False,
    },
    "expected_diversity": {
        "priority": "P2",
        "reason": "输入与依赖相近，但输出存在可接受的表达或候选差异。",
        "domain": "验收准则",
        "action": "明确允许差异边界，并补充验收规则。",
        "is_agent_defect": False,
    },
    "tool_data_variance": {
        "priority": "P1",
        "reason": "工具数据、召回或排序结果变化，导致执行结果差异。",
        "domain": "工具与数据",
        "action": "核查工具依赖、数据时效性和召回/排序一致性。",
        "is_agent_defect": False,
    },
    "nondeterminism": {
        "priority": "P0",
        "reason": "相同输入与依赖下，执行输出或评测结果出现不一致。",
        "domain": "Agent 执行",
        "action": "排查模型参数、编排分支、缓存和调用链路。",
        "is_agent_defect": True,
    },
    "quality_regression": {
        "priority": "P0",
        "reason": "可比运行中出现关键质量或约束满足度退化。",
        "domain": "Agent 执行",
        "action": "定位退化版本并修复关键质量约束。",
        "is_agent_defect": True,
    },
    "evaluation_inconsistency": {
        "priority": "P0",
        "reason": "相同输出在可比评测标准下得到不同判定。",
        "domain": "评测规则",
        "action": "核查 rubric、checker 和评测版本配置。",
        "is_agent_defect": False,
    },
    "evaluation_comparability_risk": {
        "priority": "P2",
        "reason": "评测版本未知或不一致，评分差异不可用于判断 Agent 稳定性。",
        "domain": "评测规则",
        "action": "对齐 profile、rubric 与评测配置版本。",
        "is_agent_defect": False,
    },
    "insufficient_evidence": {
        "priority": "P2",
        "reason": "缺少足够的输入上下文或输出证据，无法判断稳定性。",
        "domain": "可观测性",
        "action": "补齐 trace 的 request、env 和输出证据采集。",
        "is_agent_defect": False,
    },
}


def build_stability_groups(records: list[BatchCaseRecord]) -> list[dict[str, Any]]:
    """Aggregate cross-batch runs by scene tag + normalized query, and produce deterministic evidence."""

    grouped: dict[tuple[str, str], list[BatchCaseRecord]] = {}
    for record in records:
        key = (record.tag.strip(), _normalize_query(record.case_key.query))
        grouped.setdefault(key, []).append(record)

    groups = [
        _build_group(scene_tag, query, runs)
        for (scene_tag, query), runs in grouped.items()
        if len(runs) >= _MIN_RUNS
    ]
    return sorted(groups, key=stability_group_sort_key)


def stability_group_key(group: dict[str, Any]) -> tuple[str, str]:
    """Return the stable key used to bind an L2 finding exactly to an L1 group."""

    return (str(group.get("scene_tag") or ""), _normalize_query(group.get("query")))


def stability_group_sort_key(group: dict[str, Any]) -> tuple[int, int, int, str, str]:
    """Governance risk first, then evidence strength, run count, and stable lexicographic order."""

    return (
        _PRIORITY_RANK.get(str(group.get("priority") or "P2"), _PRIORITY_RANK["P2"]),
        _EVIDENCE_RANK.get(str(group.get("evidence_strength") or "weak"), _EVIDENCE_RANK["weak"]),
        -int(group.get("run_count") or 0),
        str(group.get("scene_tag") or ""),
        str(group.get("query") or ""),
    )


def build_stability_overview(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the stability overview from L1 groups; do not depend on the LLM to recount."""

    total_queries = len(groups)
    observations = Counter(
        str(group.get("agent_execution_observation") or "insufficient_evidence") for group in groups
    )
    domains = Counter(
        str((group.get("governance") or {}).get("domain") or "可观测性")
        for group in groups
        if group.get("agent_execution_observation") != "stable"
    )
    stable_queries = observations.get("stable", 0)
    governance_count = total_queries - stable_queries
    return {
        "total_queries": total_queries,
        "stable_queries": stable_queries,
        "stable_rate": stable_queries / total_queries if total_queries else 0.0,
        "high_risk_queries": sum(1 for group in groups if group.get("priority") == "P0"),
        "comparable_risk_queries": observations.get("evaluation_comparability_risk", 0),
        "insufficient_evidence_queries": observations.get("insufficient_evidence", 0),
        "governance_count": governance_count,
        "classification_distribution": dict(sorted(observations.items())),
        "governance_domain_distribution": dict(sorted(domains.items())),
    }


def stability_governance(
    observation: str,
    evidence_strength: str,
) -> dict[str, Any]:
    """Set governance priority, reason, and owner domain from L1 observations and evidence strength."""

    governance = dict(
        _GOVERNANCE_BY_OBSERVATION.get(
            observation,
            _GOVERNANCE_BY_OBSERVATION["insufficient_evidence"],
        )
    )
    priority = str(governance["priority"])
    if priority == "P0" and evidence_strength == "medium":
        priority = "P1"
    elif priority in {"P0", "P1"} and evidence_strength == "weak":
        priority = "P2"
    governance["priority"] = priority
    return governance


def _build_group(scene_tag: str, query: str, records: list[BatchCaseRecord]) -> dict[str, Any]:
    runs = [_run_evidence(record) for record in records]
    scores = [run["judge_score"] for run in runs if run["judge_score"] in (0.0, 1.0, 2.0)]
    distribution = Counter(str(int(score)) for score in scores)
    span = int(max(scores) - min(scores)) if scores else 0
    contexts = {run["context"]["fingerprint"] for run in runs if run["context"]["available"]}
    outputs = {
        run["output_evidence"]["fingerprint"] for run in runs if run["output_evidence"]["available"]
    }
    versions = {
        run["evaluation_version"]["signature"] for run in runs if run["evaluation_version"]["known"]
    }
    comparable = len(versions) == 1 and all(run["evaluation_version"]["known"] for run in runs)
    context_same = len(contexts) == 1 and bool(contexts)
    output_same = len(outputs) == 1 and bool(outputs)
    observation, strength = _execution_observation(
        _ExecutionEvidence(
            score_span=span,
            context_same=context_same,
            output_same=output_same,
            comparable=comparable,
            has_context=bool(contexts),
            has_output=bool(outputs),
        )
    )
    governance = stability_governance(observation, strength)
    return {
        "scene_tag": scene_tag,
        "query": query,
        "run_count": len(runs),
        "judge_score_distribution": {key: distribution.get(key, 0) for key in ("0", "1", "2")},
        "span": span,
        "evaluation_comparability": {
            "status": "comparable" if comparable else "risk",
            "reason": _comparability_reason(versions, runs),
            "known_versions": sorted(versions),
        },
        "agent_execution_observation": observation,
        "evidence_strength": strength,
        "priority": governance["priority"],
        "governance": governance,
        "representative_runs": _representative_runs(runs),
    }


def _run_evidence(record: BatchCaseRecord) -> dict[str, Any]:
    context = _context_evidence(record)
    output = _output_evidence(record)
    version = _evaluation_version(record)
    return {
        "source_batch_tag": record.source_batch_tag,
        "trace_id": record.trace_id,
        "judge_score": record.judge_score,
        "context": context,
        "output_evidence": output,
        "evaluation_version": version,
        "dimension_failures": _dimension_failures(record.dimension_results),
        "root_cause": record.attribution.root_cause_code or record.attribution.root_cause,
    }


def _context_evidence(record: BatchCaseRecord) -> dict[str, Any]:
    meta = record.trace_meta if isinstance(record.trace_meta, dict) else {}
    input_value = meta.get("input")
    request = input_value.get("request_body", input_value) if isinstance(input_value, dict) else {}
    debug = request.get("debug") if isinstance(request.get("debug"), dict) else {}
    eval_config = debug.get("eval_config") if isinstance(debug.get("eval_config"), dict) else {}
    multi_turn = (
        eval_config.get("multi_turn") if isinstance(eval_config.get("multi_turn"), dict) else {}
    )
    output = meta.get("output") if isinstance(meta.get("output"), dict) else {}
    env = output.get("env")
    request_summary = {
        "query": _bounded_string(request.get("query")),
        "scene_id": _bounded_string(request.get("scene_id")),
        "tag": record.tag,
        "case_meta_fingerprint": _fingerprint(eval_config.get("case_meta")),
        "turn_index": _bounded_string(multi_turn.get("turn_index")),
    }
    env_summary = _env_summary(env)
    available = any(request_summary.values()) or bool(env_summary["paths"])
    return {
        "available": available,
        "request": request_summary,
        "env": env_summary,
        "fingerprint": _fingerprint({"request": request_summary, "env": env_summary}),
    }


def _env_summary(env: object) -> dict[str, Any]:
    """env has no unified schema: do not emit values; expose only the path and an irreversible fingerprint."""

    if not isinstance(env, dict):
        return {"paths": [], "fingerprint": ""}
    paths = sorted(_scalar_paths(env))
    return {"paths": paths, "fingerprint": _fingerprint(env)}


def _scalar_paths(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        return [
            path
            for key, item in value.items()
            for path in _scalar_paths(item, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list):
        return [
            path
            for index, item in enumerate(value)
            for path in _scalar_paths(item, f"{prefix}[{index}]")
        ]
    return [prefix] if prefix and isinstance(value, (str, int, float, bool)) else []


def _output_evidence(record: BatchCaseRecord) -> dict[str, Any]:
    meta = record.trace_meta if isinstance(record.trace_meta, dict) else {}
    output = meta.get("output") if isinstance(meta.get("output"), dict) else {}
    component = output.get("component") if isinstance(output.get("component"), dict) else {}
    text = str(component.get("text") or "").strip()
    content_desc = (
        output.get("content_desc") if isinstance(output.get("content_desc"), dict) else {}
    )
    payload = {
        "code": output.get("code"),
        "component_type": _bounded_string(component.get("type")),
        "response_fingerprint": _fingerprint(_normalize_query(text)),
        "response_preview": _bounded_string(text),
        "content_category": _bounded_string(content_desc.get("category")),
        "content_subcategory": _bounded_string(content_desc.get("subcategory")),
        "used_tools": sorted(str(tool) for tool in content_desc.get("used_tool", []) if tool),
    }
    available = bool(
        payload["response_fingerprint"] or payload["component_type"] or payload["code"] is not None
    )
    return {"available": available, "fingerprint": _fingerprint(payload), **payload}


def _evaluation_version(record: BatchCaseRecord) -> dict[str, Any]:
    fields = {"profile_id": record.profile_id, "profile_version": record.profile_version}
    known = all(str(value).strip() for value in fields.values())
    return {"known": known, "signature": _fingerprint(fields) if known else "", "fields": fields}


@dataclass(frozen=True)
class _ExecutionEvidence:
    score_span: int
    context_same: bool
    output_same: bool
    comparable: bool
    has_context: bool
    has_output: bool


def _execution_observation(evidence: _ExecutionEvidence) -> tuple[str, str]:
    if not evidence.comparable:
        return "evaluation_comparability_risk", "weak"
    if not evidence.has_context or not evidence.has_output:
        return "insufficient_evidence", "weak"
    if not evidence.context_same:
        return "context_diff", "strong"
    if evidence.score_span:
        return (
            ("evaluation_inconsistency", "strong")
            if evidence.output_same
            else ("nondeterminism", "strong")
        )
    if not evidence.output_same:
        return "expected_diversity", "medium"
    return "stable", "strong"


def _comparability_reason(versions: set[str], runs: list[dict[str, Any]]) -> str:
    if len(versions) > 1:
        return "evaluation_version_diff"
    if not versions or any(not run["evaluation_version"]["known"] for run in runs):
        return "evaluation_version_unknown"
    return "same_known_version"


def _representative_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        runs,
        key=lambda run: (
            run["judge_score"] is None,
            run["judge_score"],
            run["trace_id"],
        ),
    )
    if len(ordered) <= _MAX_REPRESENTATIVE_RUNS:
        return ordered
    return [ordered[0], ordered[-1]]


def _dimension_failures(results: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("dimension") or item.get("name") or "")
        for item in results
        if isinstance(item, dict) and item.get("passed") is False
    ]


def _bounded_string(value: object) -> str:
    text = str(value or "").strip()
    return text[:_PREVIEW_LIMIT]


def _normalize_query(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _fingerprint(value: object) -> str:
    if value in (None, "", {}, []):
        return ""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


__all__ = [
    "build_stability_groups",
    "build_stability_overview",
    "stability_governance",
    "stability_group_key",
    "stability_group_sort_key",
]
