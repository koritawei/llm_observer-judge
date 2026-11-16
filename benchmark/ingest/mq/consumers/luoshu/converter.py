"""洛书 luoshu 协议转换器 —— 结构转换 / 投影逻辑的归属层 + 认领来源的桥。

**角色分工**（与 :mod:`~benchmark.ingest.mq.consumers.luoshu.schema` 的边界）：

- ``schema.py`` 只定义**数据结构**（把原始 JSON 建模为 pydantic 字段）与框架契约的
  **property**（``source`` / ``requires_trace_resolution`` / ``metadata``）——纯粹的数据视图。
- 本模块承载所有**结构转换**：把 :class:`LuoshuTaskMessage` 投影为领域入参
  :class:`EvaluatorRequest`、构造横向对比候选映射、必要字段校验等业务逻辑；并作为协议
  注册框架（:mod:`benchmark.ingest.mq.protocol.dispatch`）与具体 schema 的桥
  （:class:`LuoshuConverter` 认领来源 + ``convert``）。

路由按 **topic → source → converter**：凡从 luoshu topic 收到的消息均由本转换器
（``source = luoshu``）解析，不再按 ``eventType`` 之类的内容字段判别。

投影语义（:func:`build_evaluator_request`）产出**横向对比**评测入参：``query``（缺失时取
末条 user 消息）、``location``（取自 MQ ``coordinate``）、``address``（取自 MQ ``current_location``）
→ ``input.request_body``；``output.candidates`` 为 ``{候选名: 原始内容}`` 的扁平映射——
key1 = 消息 ``app``（如 ``baidu``，来源 MQ），内容 = **完整 MQ 信息**（原样透传，除
``langfuse`` 补充）；key2 = 固定 ``gaode``（来源 Langfuse），内容 = 原始 Langfuse 数据。
两键**恒定产出**（横向对比始终双候选）；Langfuse 未命中时 ``gaode`` 为空串（``""``）。
"""

from typing import Any

from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY, PipelineSource
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.ingest.mq.consumers.luoshu.schema import SOURCE, LuoshuTaskMessage

__all__ = [
    "SOURCE",
    "LuoshuConverter",
    "build_evaluator_request",
    "missing_required_fields",
    "mq_candidate_key",
    "record_trace_id",
]

#: 横向对比中 Langfuse 侧的固定候选键（key2）；key1 为消息内的 ``app`` 值（如 baidu）。
GAODE_KEY = "gaode"
#: ``app`` 缺省时 MQ 侧候选键（key1）的回退值。
_MQ_KEY_FALLBACK = "mq"
#: 入库 trace 键的候选后缀分隔符（``<trace_id>-<候选名>``）。
_RECORD_KEY_SEP = "-"


def mq_candidate_key(message: LuoshuTaskMessage) -> str:
    """MQ 侧候选名（``gaode`` / ``baidu`` / ``qwen`` …），取自消息 ``app``。

    候选映射的 key1（:func:`_candidate_contents`）与入库键的后缀
    （:func:`record_trace_id`）必须取同一个值，否则会出现“库里标的是 qwen、
    实际评的是其他候选”这类无法从数据反推的错位，故抽成单一来源。
    """

    return message.app or _MQ_KEY_FALLBACK


def record_trace_id(message: LuoshuTaskMessage) -> str:
    """入库用的 trace 键：``<trace_id>-<候选名>``（如 ``abc123-qwen``）。

    **为何需要后缀**：``TraceRecordService.upsert`` 按复合业务键
    ``(trace_id, project_id, agent_name, evaluator_name)`` 定位记录，而 luoshu 链路上
    后三项均为常量（``agent_name`` 恒为 ``luoshu``，``evaluator_name`` 恒为
    ``competitor_pairwise``）——唯一键实际退化成 ``trace_id`` 一个变量。同一条
    trace 被多个竞品（baidu / qwen / …）各采样一次时，后到的消息会命中
    ``existing is not None`` 分支而**整行覆盖**前一条（重新插入语义：未传入的列
    置 None，并清空归因派生列）。加候选后缀后各竞品各占一行。

    **为何安全**（后缀后的键不再是 Langfuse 主键，拿它查 Langfuse 必 404）：

    1. **时序**：:func:`~benchmark.ingest.mq.enrichment.enrich` 在
       :meth:`~benchmark.ingest.mq.protocol.BaseTaskMessage.to_evaluator_request` 之前执行
       （见 handler），补数用的是未加后缀的 ``message.trace_id``；
    2. **before_eval**：``requires_trace_resolution`` 为 ``False`` → 不做主键解析；
    3. **评测器内 trace 回捞**：competitor_pairwise 仅当候选内容带
       ``spans`` / ``observations`` 列表时才会拿 ``request.trace_id`` 去拉观测详情；
       luoshu 两个候选都不带该结构（gaode = trace 顶层 input/output/metadata，
       MQ 侧 = 原始消息 + ``messages`` 别名），回捞分支走不到。

    原始 ``trace_id`` 同时写进 ``metadata["source_trace_id"]`` 保留血缘，供回查
    Langfuse。若将来有新候选开始携带 trace 结构，则需让回捞改用该字段。
    """

    return f"{message.trace_id}{_RECORD_KEY_SEP}{mq_candidate_key(message)}"


def _last_content_by_role(messages: Any, role: str) -> str:
    """从 ``conversationHistoryMessages`` 取最后一条指定 role 的 ``content``（无则空串）。

    结构不确定时容错：非 list / 非 dict 元素跳过，绝不抛错（坏结构退化为空串，
    交由下游按空回答评判，而不是在映射层炸掉整条消息）。
    """

    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == role:
            content = item.get("content")
            if isinstance(content, str):
                return content
    return ""


def missing_required_fields(message: LuoshuTaskMessage) -> list[str]:
    """返回缺失的必要字段名列表（空列表 = 全部具备）。

    路由已按 topic 确定为 luoshu（从 luoshu topic 取信息即可，不靠任何事件类型字段判别）；
    但仍需保证评测所需的核心字段到位：

    - ``trace_id``：Langfuse trace 主键（补数用）+ 入库键前缀（见 :func:`record_trace_id`）；
    - ``project_id``：选 Langfuse instance；
    """

    missing: list[str] = []
    if not message.trace_id:
        missing.append("trace_id")
    if not message.project_id:
        missing.append("project_id")
    return missing


def _candidate_contents(message: LuoshuTaskMessage) -> dict[str, Any]:
    """候选内容映射 ``{候选名: 原始内容}``（完整透传 MQ 消息 + resolver 可解析别名）。

    两个候选**恒定产出**（横向对比始终双候选，键位稳定便于下游对齐）：

    - key1 = 消息 ``app``（如 ``baidu``，来源 MQ），内容 = **完整 MQ 信息**——即本条消息
      除 ``langfuse`` 补充外的全部原始字段（``model_dump(by_alias=True)``，一个字段都不删减）；
      另以 ``conversationHistoryMessages`` 完整内容再落一份 ``messages`` 别名，供下游
      competitor_pairwise 的答案提取器直接识别（只读 ``role``/``content``，多余字段无害）。
    - key2 = 固定 ``gaode``（来源 Langfuse），内容 = 原始 Langfuse 数据；未命中时为空串（``""``）。
    """

    mq_content = message.model_dump(by_alias=True, exclude={"langfuse"})
    # 以 messages 别名直接透传 conversationHistoryMessages 完整内容，命中 competitor_pairwise
    # 的 messages 分支以物化候选答案（resolver 只读 role/content，turn_index 等无害）。
    mq_content["messages"] = message.conversation_history
    return {
        mq_candidate_key(message): mq_content,
        GAODE_KEY: message.langfuse or "",
    }


def build_evaluator_request(message: LuoshuTaskMessage) -> EvaluatorRequest:
    """把 :class:`LuoshuTaskMessage` 投影为横向对比领域入参 :class:`EvaluatorRequest`。"""

    query = message.query or _last_content_by_role(message.conversation_history, "user")
    candidates = _candidate_contents(message)

    # request_body：query 必带；location 取自 MQ 的 coordinate、address 取自 MQ 的 current_location
    # （缺失则不落该键，避免 null 噪声）。评测器从 request_body.location 读经纬度上下文。
    request_body: dict[str, Any] = {"query": query}
    if message.coordinate is not None:
        request_body["location"] = message.coordinate
    if message.current_location is not None:
        request_body["address"] = message.current_location

    metadata: dict[str, Any] = {
        # MetaQ (luoshu RocketMQ) path → metaq for trace_meta_json.source filtering.
        PIPELINE_SOURCE_METADATA_KEY: PipelineSource.METAQ.value,
        # 入库键带了候选后缀（见 record_trace_id），原始 Langfuse 主键在此保留血缘：
        # 不存的话只能靠字符串切分反推，而 trace_id 本身可能含分隔符。
        "source_trace_id": message.trace_id,
        # 本行记录对应哪个竞品（gaode / baidu / qwen …）。
        "candidate_key": mq_candidate_key(message),
    }
    # DEBUG：把消息里的模拟开关透传给评测器（competitor_pairwise 的 evaluate 入口识别）；
    # 真实消息不带 _simulate 即不落该键、不触发。
    if message.simulate is not None:
        metadata["_simulate"] = message.simulate

    return EvaluatorRequest(
        input={"request_body": request_body},
        output={
            "eval_mode": "horizontal_compare",
            # {候选名: 原始内容} 扁平映射（不加工），如 {"baidu": "...", "gaode": "..."}。
            "candidates": candidates,
        },
        metadata=metadata,
        # 入库键 = <trace_id>-<候选名>，避开同一 trace 多竞品互相覆盖（详见
        # record_trace_id 的安全前提：补数已在此之前完成，不再用它查 Langfuse）。
        trace_id=record_trace_id(message),
        # 采样生成时间走既有的「源 trace 请求时间」通道（EvaluatorRequest.trace_timestamp），
        # 不再另立字段：评测器侧取时间上下文只认这一处。
        trace_timestamp=message.generate_time_b,
        # key1 = 消息 app（来源 MQ，如 baidu）；app 缺省回退。
        agent_name=SOURCE,
        project_id=message.project_id,
    )


class LuoshuConverter:
    """认领来源 :data:`SOURCE`，解析为 :class:`LuoshuTaskMessage` 并校验必要字段。"""

    #: 认领的来源标识（对应订阅的 luoshu topic 与 conf 里的 source 键）。
    source = SOURCE

    def convert(self, raw: dict[str, Any]) -> LuoshuTaskMessage:
        # 必要字段缺失 → 抛错，由 convert_for_source 兜底转为 rejected（携原因，handler 跳过）。
        msg = LuoshuTaskMessage.model_validate(raw)
        missing = missing_required_fields(msg)
        if missing:
            raise ValueError(
                f"luoshu message missing required fields: {', '.join(missing)}"
            )
        return msg
