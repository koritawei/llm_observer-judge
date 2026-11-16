"""AfterRunHook 装饰器 — 归因完成后一次性下发到 Label Studio。"""

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.labeling.dispatcher import LabelStudioDispatcher
from benchmark.attribution.service.service import AfterRunHook, AfterRunInput, TriageOutcome

logger = get_logger("agent_eval.label_studio.hook")


class LsCaseReviewAfterRunHook:
    """AfterRunHook 装饰器：归因完成后一次性下发到 Label Studio。

    合并 judge 打分审核 + 归因结果审核为一个 LS task。
    失败只记日志，不影响 inner hook (AoneAfterRunHook) 的返回值。

    Dataset / Kafka / HTTP / Benchmark 均会下发 LS；来源区分靠
    ``trace_meta_json.source``（见 TraceRecordService.list_by_pipeline_source）。
    """

    def __init__(self, *, dispatcher: LabelStudioDispatcher, inner: AfterRunHook) -> None:
        self._dispatcher = dispatcher
        self._inner = inner

    async def __call__(
        self,
        input: AfterRunInput,
    ) -> TriageOutcome:
        outcome = await self._inner(input)
        attribution = input.attribution
        if attribution is not None:
            try:
                await self._dispatcher.dispatch_case_review(
                    evaluator_name=input.evaluator_name,
                    evaluator_request=input.evaluator_request,
                    attribution=attribution,
                )
            except Exception:
                logger.warning(
                    "ls_case_review_dispatch_failed evaluator=%s",
                    input.evaluator_name,
                    exc_info=True,
                )
        return outcome
