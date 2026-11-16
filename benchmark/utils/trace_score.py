"""Langfuse score writeback helpers — judge + attribution two-stage overwrite.

Stage 1 (judge):  :func:`write_judge_score` — write the judge score after after_eval completes.
Stage 2 (attribution): :func:`write_attribution_score` — overwrite with the attribution result after triage completes.

A Langfuse score is uniquely indexed by ``trace_id + name``. Writing the same name again overwrites
(last write wins), which naturally supports "attribution overwrites judge".

Error isolation: every writeback is fire-and-forget. Failure only logs a warning and does not affect the main flow.
"""

from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from langfuse import Langfuse

from benchmark.utils.trace_report import get_report_client

logger = get_logger("agent_eval.langfuse_score")

#: Attribution-corrected grade → Langfuse score value (same 0/1/2 scale as evaluation).
_LEVEL_TO_SCORE: dict[str, float] = {
    "good": 2.0,
    "suspicious_good": 2.0,
    "watch": 1.0,
    "mild_bad": 1.0,
    "severe_bad": 0.0,
}


def _rubric_details(result: Any) -> dict[str, Any] | None:  # noqa: ANN401 - EvaluatorResult
    """从统一信封的业务扩展中读取已有子 rubric 结果。

    公共层只透出 ``evaluation_detail.dimensions``，不解释业务维度。旧信封、
    非 VUI 业务或不含结构化维度时返回 ``None``，保持原 score metadata 不变。
    """

    extra = getattr(result, "extra", None)
    if not isinstance(extra, dict):
        return None
    extension = extra.get("business_extension")
    payload = extension.get("payload") if isinstance(extension, dict) else None
    detail = payload.get("evaluation_detail") if isinstance(payload, dict) else None
    dimensions = detail.get("dimensions") if isinstance(detail, dict) else None
    return dimensions if isinstance(dimensions, dict) and dimensions else None


def _resolve_score(
    attribution: Any,  # noqa: ANN401 - AttributionResult
    original_score: float | None,
    metadata: dict[str, Any],
) -> float:
    """默认保留 judge 原分；归因推翻评测时改写为纠正后的分数。

    只改写已过归因门禁（``corrected_triage_level`` 非空即意味置信度达阀且证据可回查）
    的纠正；原分同时写入 metadata.original_judge_score 留痕，不丢信息。
    """

    fallback = original_score if original_score is not None else 0.0
    level = getattr(attribution, "corrected_triage_level", None)
    corrected = _LEVEL_TO_SCORE.get(str(level)) if level else None
    if corrected is None:
        return fallback
    metadata["original_judge_score"] = fallback
    metadata["score_overridden_by_attribution"] = True
    return corrected


@dataclass(frozen=True)
class JudgeScoreWrite:
    """``write_judge_score`` 入参（FUNC-PARAMS-001 参数收敛）。

    ``client`` 为 Langfuse SDK 客户端实例（per-job），确保 score 写入正确的
    project 且能被 ``client.flush()`` 刷新；``dataset_name`` 可选，
    dataset eval 路径写入 score metadata，Kafka/HTTP 不传；``score_name``
    可选，显式指定 Langfuse score 名称（如 ``ScoreWritebackPolicy`` 配置），
    未传时保持以 ``evaluator_name`` 命名的旧行为。
    """

    trace_id: str
    evaluator_name: str
    result: Any
    client: Langfuse
    dataset_name: str | None = None
    score_name: str | None = None


async def write_judge_score(
    write: JudgeScoreWrite,
) -> None:
    """Stage 1: 将 judge 评判结果写入 Langfuse score.

    Parameters
    ----------
    write:
        :class:`JudgeScoreWrite`（trace_id / evaluator_name / result /
        client / dataset_name / score_name，字段语义见该类 docstring）。
    """
    trace_id = write.trace_id
    evaluator_name = write.evaluator_name
    result = write.result
    client = write.client
    dataset_name = write.dataset_name
    score_name = write.score_name
    try:
        resolved_score_name = score_name or evaluator_name
        metadata: dict[str, Any] = {"stage": "judge"}
        if result.sub_scores:
            metadata["subScores"] = result.sub_scores
        rubric_details = _rubric_details(result)
        if rubric_details:
            metadata["rubricDetails"] = rubric_details
        if dataset_name:
            metadata["dataset_name"] = dataset_name

        logger.info(
            "langfuse_score.judge_writing trace_id=%s evaluator=%s score_name=%s score=%s",
            trace_id,
            evaluator_name,
            resolved_score_name,
            result.score,
        )
        client.create_score(
            trace_id=trace_id,
            name=resolved_score_name,
            value=result.score,
            data_type="NUMERIC",
            comment=result.reason,
            metadata=metadata,
        )
        logger.info(
            "langfuse_score.judge_written trace_id=%s evaluator=%s score_name=%s score=%s",
            trace_id,
            evaluator_name,
            resolved_score_name,
            result.score,
        )
    except Exception:
        logger.warning(
            "langfuse_score.judge_write_failed trace_id=%s evaluator=%s score=%s",
            trace_id,
            evaluator_name,
            getattr(result, "score", None),
            exc_info=True,
        )


@dataclass(frozen=True)
class AttributionScoreWrite:
    """``write_attribution_score`` 入参（FUNC-PARAMS-001 参数收敛）。

    ``original_score`` 为原始 judge 分数；有效纠正按 TriageLevel 映射为
    2/1/0 并在 metadata 留原分，否则保留 judge 分数。``client`` 为 ``None`` 时回退到上报项目
    :func:`get_report_client`（向后兼容；dataset eval 应显式传 client）。
    """

    trace_id: str
    evaluator_name: str
    attribution: Any
    original_score: float | None = None
    client: Langfuse | None = None


async def write_attribution_score(
    write: AttributionScoreWrite,
) -> None:
    """Stage 2: 归因结果覆盖写入 Langfuse score（覆盖同 name 的 judge score）。

    Parameters
    ----------
    write:
        :class:`AttributionScoreWrite`（trace_id / evaluator_name /
        attribution / original_score / client，字段语义见该类 docstring）。
    """
    trace_id = write.trace_id
    evaluator_name = write.evaluator_name
    attribution = write.attribution
    original_score = write.original_score
    client = write.client
    try:
        if client is None:
            client = get_report_client()
        metadata: dict[str, Any] = {
            "stage": "attribution",
            "root_cause": attribution.root_cause,
        }
        if attribution.category:
            metadata["owner_domain"] = attribution.category
        if attribution.confidence is not None:
            metadata["confidence"] = attribution.confidence
        if attribution.corrected_triage_level is not None:
            metadata["corrected_triage_level"] = attribution.corrected_triage_level

        _value = _resolve_score(attribution, original_score, metadata)
        logger.info(
            "langfuse_score.attribution_writing trace_id=%s evaluator=%s value=%s root_cause=%s",
            trace_id,
            evaluator_name,
            _value,
            attribution.root_cause,
        )
        client.create_score(
            trace_id=trace_id,
            name=evaluator_name,
            value=_value,
            data_type="NUMERIC",
            comment=attribution.root_cause,
            metadata=metadata,
        )
        logger.info(
            "langfuse_score.attribution_written trace_id=%s evaluator=%s value=%s root_cause=%s",
            trace_id,
            evaluator_name,
            _value,
            attribution.root_cause,
        )
    except Exception:
        logger.warning(
            "langfuse_score.attribution_write_failed trace_id=%s evaluator=%s original_score=%s",
            trace_id,
            evaluator_name,
            original_score,
            exc_info=True,
        )
