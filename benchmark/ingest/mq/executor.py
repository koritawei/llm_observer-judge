"""消息处理器 —— 解码后的 :class:`BaseTaskMessage` → EvaluatorService 评测。

**通用编排层**，不含任何来源/项目定制逻辑（定制在 consumers/<source> 的
转换器/补充器里），也不含 MQ 细节（offset / commit / 重试），因此可脱离 broker 单测。
错误策略（配合 :mod:`benchmark.ingest.mq.runtime.service` 的"重试直到成功"语义）：

- **评测器名**：由消息携带（``evaluator_name`` / 别名，由来源转换器填充，如
  luoshu 转换器固定 ``competitor_pairwise`` → judge_competitor_pairwise）；
  缺失 → warn 跳过（坏路由，handler 不猜默认值）。
- **可评测性**：补数后询问 ``message.unevaluatable_reason()``（具体判据由来源声 明，见
  :meth:`~benchmark.ingest.mq.protocol.BaseTaskMessage.unevaluatable_reason`）；
  非 ``None`` → warn 跳过。
  数据不完整是永久性缺陷，**进评测前就拦下**：不烧重试额度（每轮重试都会重跑 enrich），
  也不依赖评测器把错误码标对。
- **永久性评测错误**（4xx，如 ``model_not_found`` / ``evaluator_prompt_not_found``）：
  坏消息/坏路由 → warn 后跳过（正常返回）。这类错误重试永远不会成功，无限重试将
  阻塞其后所有消息，违背"每条都要消费"。
- **可重试评测错误**（5xx，如 ``llm_failed`` / ``llm_unparseable`` / ``evaluation_failed``）
  与其它未预期异常：**向上抛出** → 由消费循环退避重试（上限 ``retry_max_attempts``，
  耗尽则跳过并提交）。

即：``handle`` 正常返回 = "该条已消费，可推进"；抛异常 = "本条需重试"。
"""

from typing import Any

from benchmark.libs.agenthost.init import request_context
from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.setting import settings
from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY, PipelineSource
from benchmark.ingest.mq.enrichment import EnrichmentUnavailableError, enrich
from benchmark.ingest.mq.protocol import BaseTaskMessage
from benchmark.services.evaluation import EvalRunOptions, EvaluatorService
from benchmark.utils.exceptions import EvaluatorError
from benchmark.utils.trace_report import (
    ReportSpan,
    eval_report_root,
    finish_report_child_span,
    start_report_child_span,
)
from benchmark.utils.trace_naming import build_eval_trace_name

logger = get_logger("agent_eval.mq.executor")

#: HTTP 状态 < 该阈值视为永久性客户端错误（4xx）→ 跳过而非重试。
_PERMANENT_ERROR_STATUS_CEILING = 500

#: 唯一的“真的评出分了”终态；其余终态均为失败类（见 :func:`_outcome_payload`）。
_STATUS_EVALUATED = "evaluated"


def _update_report_root(root: ReportSpan | None, payload: dict[str, Any]) -> None:
    """把终态写回 report root；root 为 None（上报不可用）或写失败均 no-op。

    上报属于旁路：写 span 失败绝不能影响“该消息是否已消费”的判定。
    """

    if root is None:
        return
    try:
        root.update(**payload)
    except Exception:
        logger.debug("mq.report_root_update_failed", exc_info=True)


def _message_report_input(message: BaseTaskMessage) -> dict[str, Any]:
    """消息级 root 的 input：标识字段 + MQ **接收到的完整原始信息**。

    在 enrich 之前取（handle 开 root 时 message 尚未补数），故 ``message`` 键为“收到什么”
    的原貌；即便后续跳过/失败，完整 MQ 信息也已落在这条 trace 上，不再依赖 evaluate
    跑通才可见（旧行为：完整消息仅随 evaluate 的候选上报，未评就丢）。

    ``by_alias`` 回到 MQ 原始键（camelCase）；序列化失败不能阻断上报，退化为仅标识字段。
    """

    payload: dict[str, Any] = {
        "source": message.source,
        "trace_id": getattr(message, "trace_id", None),
        "project_id": getattr(message, "project_id", None),
        "generate_time_b": getattr(message, "generate_time_b", None),
    }
    try:
        payload["message"] = message.model_dump(mode="json", by_alias=True)
    except Exception:
        logger.debug("mq.message_dump_failed", exc_info=True)
    return payload


def _outcome_payload(outcome: dict[str, Any]) -> dict[str, Any]:
    """消费终态 → root 更新载荷；非 ``evaluated`` 一律标 ERROR。

    “已消费但没评出分”（被门禁拦下、坏路由、永久性评测错误）在 MQ 语义上是正常返回，
    若 root 也跟着进正常态，Langfuse 上就只剩一条无异常、无评分子节点的空 trace，无法
    按 level 筛出这批未评测的消息。
    """

    payload: dict[str, Any] = {"output": outcome}
    if outcome.get("status") != _STATUS_EVALUATED:
        detail = outcome.get("detail") or outcome.get("reason")
        payload["level"] = "ERROR"
        payload["status_message"] = f"{outcome.get('status')}: {detail}"
    return payload


class EvalMessageExecutor:
    """把一条评测消息交给 :class:`EvaluatorService` 执行。"""

    def __init__(self, evaluator_service: EvaluatorService) -> None:
        self._evaluator_service = evaluator_service

    async def handle(self, message: BaseTaskMessage) -> None:
        """处理单条消息：打开 request context + Langfuse report root，再交 :meth:`_evaluate`。

        正常返回 = 已消费（推进）；抛异常 = 需重试。见模块 docstring 的错误策略。
        """

        # 校验门:消息体必须是合法 JSON 结构。非 JSON(codec 已打标)→ 报错 + 跳过后续流程,
        # 正常返回(= 已消费)→ worker 提交推进到下一条(非 JSON 是永久坏消息,重试无意义)。
        if message.decode_error is not None:
            logger.error(
                "mq.handle.reject reason=not_json_structure err=%s",
                message.decode_error,
            )
            return

        # MQ 链路无 HTTP 中间件：此处补齐 request context（与
        # langfuse_trace_context_middleware 同一形状）并开一条**消息级** report root，
        # 让评测前的补数也落在同一条 Langfuse trace 上。评测器/归因节点仍由
        # EvaluatorService 的既有上报逻辑产出，此处不复制它的字段。
        trace_name = build_eval_trace_name(message.source)
        with (
            request_context(
                workflow_name=trace_name,
                service="agent-eval",
                environment=settings.env,
            ),
            eval_report_root(
                name=trace_name,
                input_value=_message_report_input(message),
                metadata={PIPELINE_SOURCE_METADATA_KEY: PipelineSource.METAQ.value},
            ) as root,
        ):
            # root.output 只记「这条消息怎么消费的」：评分与评测状态由 evaluate 的既有
            # 上报写在它自己的节点上，不在此重复。否则跳过/重试的消息在 Langfuse 上
            # 只剩一条空 trace，看不出为何没评。
            try:
                outcome = await self._evaluate(message)
            except Exception as exc:
                _update_report_root(
                    root,
                    {
                        "level": "ERROR",
                        "status_message": f"{type(exc).__name__}: {exc}",
                        "output": {"status": "retry", "detail": type(exc).__name__},
                    },
                )
                raise
            # 非 evaluated 终态（跳过 / 永久性评测错误）也得在 root 上标 ERROR，否则
            # “evaluate 未执行/失败”的消息在 Langfuse 上与正常评测无法区分。
            _update_report_root(root, _outcome_payload(outcome))

    async def _evaluate(self, message: BaseTaskMessage) -> dict[str, Any]:
        """补数 → 门禁 → evaluate；全程在 :meth:`handle` 打开的 report root 之内。

        返回**消费终态**写入 root.output（``status`` 与日志的 ``mq.handle.*`` 理由同源）；
        正常返回 = 已消费，抛异常 = 需重试。
        """

        # 转换后、评测前：按来源(message.source)异步补充外部数据(如 Langfuse 按
        # project_id+trace_id 取数)。无来源/无补充器/补充失败均原样返回,继不阻断评测。
        # 作为 report root 下的子节点上报，让“取数”这一步在 trace 上可见。
        enrich_span = start_report_child_span(
            name="mq.enrich",
            input_value={
                "source": message.source,
                "trace_id": getattr(message, "trace_id", None),
                "project_id": getattr(message, "project_id", None),
            },
        )
        try:
            message = await enrich(message)
        finally:
            finish_report_child_span(
                enrich_span,
                output={"enrich_status": getattr(message, "enrich_status", None)},
            )

        # 瞬时失败门（必须在可评测性门**之前**）：补数因外部服务抖动（超时/5xx/网络）
        # 失败时应重试，而不是当永久缺陷跳过。旧实现把两者都归结为“补数字段为 None”，
        # 于是 Langfuse 抖一下这批消息就静默丢了。具体哪种成因可重试由来源声明
        # （handler 不知道任何来源的字段语义）。
        retry_reason = message.enrich_retry_reason()
        if retry_reason is not None:
            logger.warning(
                "mq.handle.retry reason=enrich_unavailable detail=%s source=%s trace_id=%s",
                retry_reason,
                message.source,
                getattr(message, "trace_id", None),
            )
            raise EnrichmentUnavailableError(retry_reason)

        # 可评测性门（补数后才能判）：数据不完整 = 永久性缺陷，重试永远不会成功 → warn 跳过。
        # 具体判据由来源自己声明（handler 不知道任何来源的字段语义）；在此拦下而不是放进
        # 评测，避开了“靠 EvaluatorError.code 分级、标错就反复重试”（每轮重试还会重跑 enrich）。
        unevaluatable = message.unevaluatable_reason()
        if unevaluatable is not None:
            logger.warning(
                "mq.handle.skip reason=unevaluatable detail=%s source=%s trace_id=%s",
                unevaluatable,
                message.source,
                getattr(message, "trace_id", None),
            )
            return {"status": "skipped", "reason": "unevaluatable", "detail": unevaluatable}

        # 评测器名由消息携带(来源转换器填充,已经 schema 归一化)。缺失 = 坏路由 → warn 跳过,
        # handler 不猜默认值(“本 topic 就是 luoshu”的假设属于来源定制,不属于通用 handler)。
        evaluator_name = (message.evaluator_name or "").strip()
        if not evaluator_name:
            logger.warning(
                "mq.handle.skip reason=no_evaluator_name trace_id=%s",
                getattr(message, "trace_id", None),
            )
            return {"status": "skipped", "reason": "no_evaluator_name"}
        request = message.to_evaluator_request()
        try:
            # logger.info(
            #     "mq.handle.dry_run evaluator=%s trace_id=%s request=%s",
            #     evaluator_name,
            #     request.trace_id,
            #     request.model_dump(mode="json"),
            # )
            # 是否走 before_eval（业务 trace_id→主键解析）由消息自身声明透传：通用 handler 不含来源
            # 判断，仅转发 message.requires_trace_resolution（如 luoshu 声明
            # False → 跳过 before_eval）。
            await self._evaluator_service.evaluate(
                evaluator_name,
                request,
                options=EvalRunOptions(
                    resolve_trace=message.requires_trace_resolution,
                ),
            )
            return {"status": _STATUS_EVALUATED}
        except EvaluatorError as exc:
            if exc.code < _PERMANENT_ERROR_STATUS_CEILING:
                logger.warning(
                    "mq.handle.skip reason=permanent_eval_error evaluator=%s "
                    "trace_id=%s type=%s code=%s msg=%s",
                    evaluator_name,
                    request.trace_id,
                    exc.error_type,
                    exc.code,
                    exc.message,
                )
                return {
                    "status": "skipped",
                    "reason": "permanent_eval_error",
                    "detail": f"{exc.error_type}: {exc.message}",
                }
            logger.warning(
                "mq.handle.retry reason=eval_error evaluator=%s trace_id=%s type=%s code=%s msg=%s",
                evaluator_name,
                request.trace_id,
                exc.error_type,
                exc.code,
                exc.message,
            )
            raise


__all__ = ["EvalMessageExecutor"]
