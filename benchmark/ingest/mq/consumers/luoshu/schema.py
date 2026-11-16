"""洛书 luoshu 的消息 schema —— 直接建模 luoshu topic 下发的原始 JSON。

路由按 **topic → source → converter**：凡从 luoshu topic 收到的消息均由本来源解析，
不再按 ``eventType`` 之类的内容字段区分消息。该来源自带具体 schema :class:`LuoshuTaskMessage`，
在 :class:`BaseTaskMessage` 契约之上：

- 把原始 JSON 字段建模为 pydantic 字段（``project_id`` / ``trace_id`` / ``query`` /
  ``conversationHistoryMessages`` 等；camelCase 键用 alias 映射，``populate_by_name`` 让
  字段名/别名都可入参）。
- ``source`` 固定为 ``luoshu``（供补充数据层按来源派发）；``evaluator_name`` 固定为
  ``competitor_pairwise``（→ ``EvaluatorService._select`` 拼 ``judge_`` 前缀 → ``judge_competitor_pairwise``，
  即横向对比评测器）。

**角色边界**：本模块只做**数据结构 + property**——建模字段、暴露框架契约的 ``source`` /
``requires_trace_resolution`` / ``metadata`` property 与 ``unevaluatable_reason()`` 门禁。
所有**结构转换**（投影为评测入参、构造横向对比候选、必要字段校验）内聚到
:mod:`~benchmark.ingest.mq.consumers.luoshu.converter`；契约方法
:meth:`~LuoshuTaskMessage.to_evaluator_request` 仅**委托**给
:func:`~benchmark.ingest.mq.consumers.luoshu.converter.build_evaluator_request`，保持 handler
对消息实例的统一调用契约不变。

``project_id`` 用于选 Langfuse instance，``trace_id`` 直接作为 trace 主键直查；补充数据由
:class:`~benchmark.ingest.mq.consumers.luoshu.enrichment.LuoshuLangfuseEnricher` 回填到
:attr:`~LuoshuTaskMessage.langfuse`，并作为 ``gaode`` 候选值暴露给评测。
"""

from typing import Any

from pydantic import ConfigDict, Field

from benchmark.evaluators.base import EvaluatorRequest
from benchmark.ingest.mq.protocol import BaseTaskMessage

#: 本来源标识（BaseTaskMessage.source / metadata["source"]，供补充数据层按来源派发）。
SOURCE = "luoshu"

#: ``enrich_status`` 取值（luoshu 来源语义）。四种失败成因必须分开：旧实现全部
#: 表现为 ``langfuse is None``，导致日志把“凭证未配”也报成“trace 不存在”，且
#: 瞬时失败被当永久缺陷丢弃。定义在 schema 层：enrichment 已依赖本模块，
#: 反方向依赖会成环。
STATUS_OK = "ok"
STATUS_NO_IDS = "no_ids"
STATUS_UNKNOWN_PROJECT = "unknown_project"
STATUS_QUERY_ERROR = "query_error"
STATUS_TRACE_MISS = "trace_miss"

#: 洛书评测器逻辑名：_select 拼 ``judge_`` 前缀 → agent_manager 的 ``judge_competitor_pairwise``（横向对比）。
EVALUATOR_NAME = "competitor_pairwise"


class LuoshuTaskMessage(BaseTaskMessage):
    """洛书 luoshu topic 采样结果的具体消息 schema。"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    #: 洛书固定走横向对比评测器 judge_competitor_pairwise（EvaluatorService._select 会加 judge_ 前缀）；覆写 base 默认值。
    evaluator_name: str = EVALUATOR_NAME
    #: 事件类型（如 ``GUI_CHAT_SAMPLE_RESULT``）；路由**不依赖**它（按 topic→source 派发），
    #: 仅留档并随完整 MQ 信息透传给评测（baidu 候选）。
    event_type: str | None = Field(default=None, alias="eventType")
    #: 选 Langfuse instance（映射 custom.langfuse.instances 的 key）。
    project_id: str | None = None
    #: 直接作为 Langfuse trace 主键直查。
    trace_id: str | None = None
    app: str | None = None
    query: str | None = None
    coordinate: str | None = None
    #: 用户位置的可读地址（如 "山西省太原市万柏林区滨河西路辅路"）；MQ 原始键 ``current_location``（字段名与 MQ 键同名，无需 alias）。
    current_location: str | None = None
    #: 采样生成时间（毫秒时间戳，如 "1789092215409"）；MQ 原始键 ``generate_time_b``。
    #: 兼容 str / int 两种下发形态——类型收窄成 str 会让数字型下发在 model_validate 处
    #: 抛错，整条消息退化成 rejected 被丢弃。
    generate_time_b: str | int | None = None
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list, alias="conversationHistoryMessages"
    )
    screenshots: list[Any] | None = None
    #: Langfuse 补充数据（由 LuoshuLangfuseEnricher 回填；经 metadata["langfuse"] 暴露）。
    langfuse: dict[str, Any] | None = None
    #: DEBUG：模拟失败路径的开关（``empty_candidate`` / ``llm_failure``）。发一条带该
    #: 字段的 MQ 消息即可在主干链路上端到端验证“失败也入库”（经
    #: :meth:`to_evaluator_request` 透传到 ``metadata["_simulate"]``，由
    #: competitor_pairwise 的 evaluate 入口识别）。真实消息不带即不触发。
    simulate: str | None = Field(default=None, alias="_simulate")

    @property
    def source(self) -> str:
        return SOURCE

    @property
    def requires_trace_resolution(self) -> bool:
        """洛书 trace_id 本身即 Langfuse 主键、候选内容随请求自带，无需 before_eval 的主键解析。"""

        return False

    @property
    def metadata(self) -> dict[str, Any]:
        """投影给评测的附加上下文（去掉 None 键）；含 Langfuse 补充数据（若已回填）。"""

        data: dict[str, Any] = {
            "source": SOURCE,
            "app": self.app,
            "coordinate": self.coordinate,
            "generate_time_b": self.generate_time_b,
            "screenshots": self.screenshots,
        }
        if self.langfuse is not None:
            data["langfuse"] = self.langfuse
        return {k: v for k, v in data.items() if v is not None}

    def to_evaluator_request(self) -> EvaluatorRequest:
        """契约方法：委托给 converter 的投影逻辑（结构转换不落在数据模型上）。

        延迟 import 打破 schema ↔ converter 的循环依赖（converter 在模块顶层依赖本 schema）。
        """

        from benchmark.ingest.mq.consumers.luoshu.converter import build_evaluator_request

        return build_evaluator_request(self)

    def unevaluatable_reason(self) -> str | None:
        """仅拦截“物理上无法入库”的结构性缺陷；其他失败一律放行到评测器留痕。

        入库门在 :meth:`~benchmark.services.evaluation.EvaluatorService.after_eval` 硬要求
        ``trace_id + project_id + agent_name`` 才能 upsert（见 evaluation.py 的
        ``after_eval.skipped`` 分支）。缺 ``trace_id`` / ``project_id`` 的消息即便进评
        测器也拿不到入库位，故此处直接拦下；``agent_name`` 由 converter 固定为
        ``SOURCE``，不参与此判据。

        **不再拦 langfuse 相关成因**：老行为把 ``trace_miss`` / ``unknown_project``
        当永久缺陷跳过，导致这些消息永久静默丢失。改造后由
        :class:`~business.evaluators.competitor_pairwise.agent.CompetitorPairwiseEvaluatorAgent`
        以 :data:`~business.evaluators.competitor_pairwise.scoring.UNEVALUATED_SCORE`
        （``-1``）落库留痕，保证“发多少条消息就入多少条库”。

        ``query_error``（瞬时失败）由 :meth:`enrich_retry_reason` 转 MQ 重试，仍不在
        本方法拦下。
        """

        if not self.trace_id or not self.project_id:
            return (
                f"missing ids (trace_id={self.trace_id!r} "
                f"project_id={self.project_id!r})"
            )
        return None

    def enrich_retry_reason(self) -> str | None:
        """只有“Langfuse 查询异常”算瞬时失败，其余成因重试永不会成功。

        逐个判定的理由：

        - ``query_error``（超时 / 5xx / 网络抖动）→ **重试**，这是旧实现丢消息的那类；
        - ``no_ids``（消息本身缺 id）→ 永久，重试不会变好；
        - ``unknown_project``（实例映射里没有该 project）→ 实例映射是**进程级缓存**，
          同一进程内重试永不会成功，重试只会造成 head-of-line blocking；
        - ``trace_miss``（404）→ 按现有语义认定为永久事实。
        """

        if self.langfuse is None and self.enrich_status == STATUS_QUERY_ERROR:
            return (
                f"langfuse query failed transiently "
                f"(project_id={self.project_id} trace_id={self.trace_id})"
            )
        return None


__all__ = ["SOURCE", "LuoshuTaskMessage"]
