"""洛书 luoshu 的 Langfuse 补充数据 —— 按 ``project_id`` + ``trace_id`` 拉取补充数据。

转换（:mod:`~benchmark.ingest.mq.consumers.luoshu.converter`）后消息已带 ``project_id`` +
``trace_id``；本补充器用洛书专用
:class:`~benchmark.ingest.mq.consumers.luoshu.trace_fetch.LuoshuTraceResolver`
（平台 :class:`benchmark.trace.TraceResolver` 的子类扩展）按 ``(project_id, trace_id)``
拉取 Langfuse **trace 顶层 input/output/metadata**（对应
Langfuse UI Preview 里的 Input/Output/Metadata），作为补充数据。``project_id`` 映射到
``AGENT_EVAL_LANGFUSE_INSTANCES`` 里 ``name`` 等于
``conf/app.<env>.yaml`` 的 ``custom.langfuse.instance_name`` 的那一条。
``trace_id`` **直接作为 Langfuse trace 主键**查询（``resolve_primary=False``）。
消息 ``project_id`` 必须等于该条 JSON 的 ``projectId``。

合并规则：把拉到的 trace 顶层 ``input`` / ``output`` / ``metadata`` 三项核心内容挂到
``langfuse`` 字段（经 :attr:`LuoshuTaskMessage.metadata` 暴露为 ``metadata["langfuse"]``，
并作为横向对比的 ``gaode`` 候选值），**不改 query/其余候选**。

降级语义：缺 ``project_id`` / ``trace_id``、未知 project、trace 未命中、查询异常，
均**原样返回**消息（补充是增强，绝不阻断评测；上层 :func:`benchmark.ingest.mq.enrichment.enrich`
也会再兜一层 try）。
"""

from pathlib import Path

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.trace_endpoints import (
    read_trace_instance_name,
    select_trace_instances_by_name,
)
from benchmark.settings.setting import settings
from benchmark.ingest.mq.consumers.luoshu.schema import (
    SOURCE,
    STATUS_NO_IDS,
    STATUS_OK,
    STATUS_QUERY_ERROR,
    STATUS_TRACE_MISS,
    STATUS_UNKNOWN_PROJECT,
    LuoshuTaskMessage,
)
from benchmark.ingest.mq.consumers.luoshu.trace_fetch import LuoshuTraceResolver
from benchmark.ingest.mq.protocol import BaseTaskMessage
from benchmark.trace import TraceEndpoint, TraceDocument

logger = get_logger("agent_eval.mq.consumers.luoshu.enrichment")


class LuoshuTraceEnricher:
    """按 ``project_id`` + ``trace_id`` 从 Langfuse 拉 trace 顶层
    input/output/metadata，挂到 message.langfuse。"""

    source = SOURCE

    def __init__(self, config_path: Path | None = None) -> None:
        #: 显式指定时用之（供测试/特殊部署）；默认按 ``settings.env`` 解析。
        self._config_path = config_path
        #: 进程级缓存：instances 配置在 env yaml 部署期即固定，只加载一次，避免每条消息
        #: 同步读盘 + 解析 YAML 阻塞事件循环。
        self._instances: dict[str, TraceEndpoint] | None = None
        #: 进程级复用的 resolver（内部持长生命周期 httpx client）：避免每条消息新建/销毁
        #: TCP+TLS 连接，复用 keep-alive 连接池；关停时经 :meth:`aclose` 释放。
        self._resolver: LuoshuTraceResolver | None = None

    def _resolve_config_path(self) -> Path:
        if self._config_path is not None:
            return self._config_path
        # 延迟 import：benchmark.paths 属叶子模块，避免模块加载期依赖重模块。
        from benchmark.paths import app_config_path  # noqa: PLC0415

        return app_config_path(settings.env)

    def _get_instances(self) -> dict[str, TraceEndpoint]:
        # 同步、无 await：单事件循环下首个到达的协程原子完成初始化，无需加锁。
        if self._instances is None:
            self._instances = load_luoshu_trace_instances(self._resolve_config_path())
        return self._instances

    def _get_resolver(self) -> LuoshuTraceResolver:
        # 同步、无 await：懒建一次并复用（并发安全同 _get_instances）。
        if self._resolver is None:
            self._resolver = LuoshuTraceResolver(self._get_instances())
        return self._resolver

    async def aclose(self) -> None:
        """关停时释放共享 httpx client（无则 no-op）；幂等，关后再 enrich 会重新懒建。"""

        if self._resolver is not None:
            await self._resolver.aclose()
            self._resolver = None

    async def enrich(self, message: BaseTaskMessage) -> BaseTaskMessage:
        # 本补充器只处理 luoshu 具体 schema（非则原样返回，理论上不会发生：
        # enrich 按 source=luoshu 派发到本器）。
        if not isinstance(message, LuoshuTaskMessage):
            return message
        project_id = message.project_id
        trace_id = message.trace_id
        if not project_id or not trace_id:
            logger.info(
                "luoshu.enrich.skip reason=no_ids project_id=%r trace_id=%r",
                project_id,
                trace_id,
            )
            return message.model_copy(update={"enrich_status": STATUS_NO_IDS})

        instances = self._get_instances()
        if project_id not in instances:
            # 空列表 = 未配 instance_name，或 JSON 里没有这个 name。
            # 非空列表 = 消息 project_id 对不上该实例的 projectId。
            logger.warning(
                "luoshu.enrich.skip reason=unknown_project project_id=%s "
                "trace_id=%s known_projects=%s",
                project_id,
                trace_id,
                sorted(instances),
            )
            return message.model_copy(update={"enrich_status": STATUS_UNKNOWN_PROJECT})

        # 复用进程级 resolver（内部持长生命周期 httpx client）：不再每条消息新建/关闭连接，
        # 复用 keep-alive 连接池；关停时由 :meth:`aclose` 统一释放。
        resolver = self._get_resolver()
        try:
            # trace_id 直接作为 Langfuse trace 主键查询（resolve_primary=False）,
            # 拉 trace 顶层 input/output/metadata。
            trace = await resolver.fetch_trace(trace_id, project_id, resolve_primary=False)
        except Exception as exc:
            # 标记为瞬时失败（超时 / 5xx / 网络抖动），由 handler 转成重试；
            # 旧行为会把它当永久缺陷直接丢弃消息。
            logger.warning(
                "luoshu.enrich.error project_id=%s trace_id=%s err=%r",
                project_id,
                trace_id,
                exc,
            )
            return message.model_copy(update={"enrich_status": STATUS_QUERY_ERROR})

        if trace is None:
            logger.info("luoshu.enrich.miss project_id=%s trace_id=%s", project_id, trace_id)
            return message.model_copy(update={"enrich_status": STATUS_TRACE_MISS})

        logger.info(
            "luoshu.enrich.hit project_id=%s trace_id=%s has_input=%s "
            "has_output=%s has_metadata=%s",
            project_id,
            trace_id,
            trace.input is not None,
            trace.output is not None,
            trace.metadata is not None,
        )
        return self._merge(message, trace)

    def _merge(self, message: LuoshuTaskMessage, trace: TraceDocument) -> LuoshuTaskMessage:
        """把 Langfuse trace 顶层 input/output/metadata 回填到 ``langfuse`` 字段。

        仅提取评测/横向对比关心的三项核心内容（对应 Langfuse UI 的 Input/Output/Metadata），
        原样挂到 ``langfuse``（经 :attr:`LuoshuTaskMessage.metadata` 暴露为
        ``metadata["langfuse"]``，并作为 ``gaode`` 候选值），不改 query/其余候选。
        """

        return message.model_copy(
            update={
                "enrich_status": STATUS_OK,
                "langfuse": {
                    "input": trace.input,
                    "output": trace.output,
                    "metadata": trace.metadata,
                },
            }
        )


def load_luoshu_trace_instances(config_path: Path) -> dict[str, TraceEndpoint]:
    """按 yaml 里的实例名，从 ``AGENT_EVAL_LANGFUSE_INSTANCES`` 取出凭证。

    返回值的 key 是该实例的 project id。名为空或 JSON 里没有同名项时返回 ``{}``。
    """
    name = read_trace_instance_name(config_path)
    if not name:
        return {}
    return select_trace_instances_by_name(settings.trace_instances, name)


__all__ = ["LuoshuTraceEnricher", "load_luoshu_trace_instances"]
