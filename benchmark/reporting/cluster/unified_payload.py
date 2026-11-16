"""Private display data for the VUI Cluster Report."""

from typing import Any

from benchmark.reporting.cluster.schema import BatchCaseRecord, ClusterReport
VUI_TAXONOMY: tuple = ()

_ROOT_CAUSE_LABELS: dict[str, str] = {}
_FULL_SCORE = 2


def build_vui_metrics(report: ClusterReport) -> dict[str, Any]:
    """Build run observations and low-score cluster data consumed only by the VUI private Fragment."""

    return {
        "routes": _mapping(report.stats.performance.get("route_modes")),
        "models": _mapping(report.stats.performance.get("models")),
        "low_score_total": sum(
            record.judge_score is not None and record.judge_score < _FULL_SCORE
            for record in report.cases
        ),
        "clusters": _low_score_clusters(report),
    }


def _low_score_clusters(report: ClusterReport) -> list[dict[str, Any]]:
    """Keep only problem clusters that L2 linked to original low-score evaluation cases."""

    case_by_trace = {
        record.trace_id: record
        for record in report.cases
        if record.trace_id
        and record.judge_score is not None
        and record.judge_score < _FULL_SCORE
    }
    clusters: list[dict[str, Any]] = []
    for cluster in _dict_list(report.llm_summary.get("clusters")):
        records = _cluster_records(case_by_trace, cluster.get("related_cases"))
        if not records:
            continue
        clusters.append(
            {
                "name": str(cluster.get("title") or "未命名问题簇"),
                "count": len(records),
                "summary": str(cluster.get("summary") or "未生成共性分析"),
                "evidence": str(cluster.get("evidence") or ""),
                "cases": [_low_score_case(record) for record in records[:3]],
            }
        )
    return clusters


def _cluster_records(
    case_by_trace: dict[str, BatchCaseRecord], related_cases: object
) -> list[BatchCaseRecord]:
    trace_ids = related_cases if isinstance(related_cases, list) else []
    records: list[BatchCaseRecord] = []
    seen: set[str] = set()
    for trace_id in trace_ids:
        if not isinstance(trace_id, str) or trace_id in seen or trace_id not in case_by_trace:
            continue
        seen.add(trace_id)
        records.append(case_by_trace[trace_id])
    return records


def _low_score_case(record: BatchCaseRecord) -> dict[str, Any]:
    return {
        "id": record.case_key.case_id or record.trace_id or "实时请求",
        "query": _query(record) or "未采集",
        "score": record.judge_score,
    }


def _query(record: BatchCaseRecord) -> str:
    for source in (record.case_key.__dict__, record.case_meta, record.trace_meta):
        value = _query_from(source)
        if value:
            return value
    return ""


def _query_from(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    for key in ("query", "user_query", "query_text"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    for key in ("input", "input_snapshot", "request", "request_body", "payload"):
        found = _query_from(value.get(key))
        if found:
            return found
    return ""


def _root_cause_label(value: str) -> str:
    for code, label in _ROOT_CAUSE_LABELS.items():
        if code in value:
            return label
    return value


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_list(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


__all__ = ["build_vui_metrics"]
