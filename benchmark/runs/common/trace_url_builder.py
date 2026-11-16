"""trace_url 生产工厂 — 业务只提供 trace 身份提取，其余链路全业务共用。

业务按 benchmark 组件名注册「trace 身份提取器」：judge 回调 payload
→ ``(project_id, trace_id)``。其余部分（Langfuse primary_id 解析、
agentfuse 面板链接拼接、异常回退）所有业务共用同一实现。

默认提取器 = ieval 形态：直取 ``payload.project_id`` / ``payload.trace_id``
（Aone judge 回调透传字段）。未注册业务自动落到默认提取器；新业务
数据形态不同（trace 身份埋在别的字段）时注册自己的提取器即可。

sync judge 路径（无 Aone 回调 payload）走 item 形态入口
:func:`build_trace_url_for_item`：身份从业务 item（parse 拦截器产物的
``metadata.project_id`` + ``source_trace_id``）提取，默认提取器已覆盖
「身份落在 item 字段」的形态；业务身份位置特殊时经
:func:`register_item_trace_identity_extractor` 注册覆写。存量 sync 业务
metadata 无 ``project_id`` → 提取为 ``None`` → 回退透传值（行为不变）。

两类提取器均可经 RemoteConfig ``eval.plugins.trace_identity_extractors`` /
``item_trace_identity_extractors`` 按组件名改为 ``remote`` / ``class_ref``，
优先级高于注册表；``enabled=false`` 强制默认提取器。

时间窗说明：primary_id 解析窗口由 :class:`TraceResolver` 构造级
``resolve_lookback_days``（默认 14 天）承接，不在本模块做 per-call
窗口——judge 回调通常在数据日后 1-2 天内到达，14 天窗口足够覆盖。
"""

from collections.abc import Callable, Mapping
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.trace_endpoints import load_trace_instances
from benchmark.settings.setting import settings as app_settings
from benchmark.runs.schema import JudgeCallback
from benchmark.plugins.host import PluginResolution, call_plugin_callable, resolve_plugin
from benchmark.plugins.remote_adapters import (
    RemoteItemTraceIdentityExtractor,
    RemoteTraceIdentityExtractor,
)
from benchmark.trace.resolver import TraceResolver

logger = get_logger("agent_eval.benchmark.common.trace_url_builder")

#: Langfuse 面板固定入口 —— trace_url 构建不走 per-instance base_url
#: （实例配置的 base_url 是 API host，非面板域名），统一 agentfuse 面板。
TRACE_TRACE_URL_BASE = "https://agentfuse-v4.amap-inc.com"


def build_direct_trace_url(project_id: str, trace_id: str) -> str:
    """直拼 Langfuse 面板链接（``trace_id`` 即主键，跳过 primary_id 解析）。

    形态与 :func:`build_trace_url_for_item` 直拼模式一致：
    ``{base}/project/{project_id}/traces/{trace_id}?traceId={trace_id}``。
    """
    return f"{TRACE_TRACE_URL_BASE}/project/{project_id}/traces/{trace_id}?traceId={trace_id}"


#: trace 身份提取器签名：judge 回调 payload → ``(project_id, trace_id)``。
TraceIdentityExtractor = Callable[[JudgeCallback], "tuple[str | None, str | None]"]


def default_trace_identity(payload: JudgeCallback) -> tuple[str | None, str | None]:
    """默认提取器（ieval 形态）：直取 payload.project_id / payload.trace_id。"""
    return payload.project_id, payload.trace_id


#: benchmark 组件名 → 身份提取器（未注册回退默认提取器）。
#: 以组件名（如 ``ieval_external``）为键而非 service_name —— 与
#: attribution_trigger_policy 注册表同构。
_EXTRACTOR_REGISTRY: dict[str, TraceIdentityExtractor] = {}


def register_trace_identity_extractor(
    benchmark_name: str,
    extractor: TraceIdentityExtractor,
) -> None:
    """业务注册自己的 trace 身份提取器（同名覆盖，幂等）。"""
    _EXTRACTOR_REGISTRY[benchmark_name] = extractor
    logger.info(
        "trace_url.extractor_registered benchmark=%s extractor=%s",
        benchmark_name,
        getattr(extractor, "__qualname__", type(extractor).__name__),
    )


def resolve_trace_identity_extractor(benchmark_name: str | None) -> TraceIdentityExtractor:
    """按 benchmark 组件名解析提取器：RemoteConfig binding → 注册表 → 默认（ieval 形态）。"""
    return resolve_plugin(
        "trace_identity_extractors",
        benchmark_name,
        resolution=PluginResolution(
            registry=_EXTRACTOR_REGISTRY,
            default=default_trace_identity,
            remote_factory=lambda binding: RemoteTraceIdentityExtractor(
                binding,
                fallback=default_trace_identity,
            ),
        ),
    )


def clear_trace_identity_extractors() -> None:
    """清空业务注册表（测试隔离用）。"""
    _EXTRACTOR_REGISTRY.clear()


async def build_trace_url(
    payload: JudgeCallback,
    *,
    benchmark_name: str | None,
    fallback: str | None,
) -> str | None:
    """共用主流程：身份提取 → Langfuse primary id 解析 → 拼面板链接。

    project_id / trace_id 任一缺失、project_id 无配置实例、窗口内未命中、
    请求抛错均只记日志并回退 ``fallback``（judge 侧透传 trace_url），
    绝不阻塞回调主流程。
    """
    extractor = resolve_trace_identity_extractor(benchmark_name)
    project_id, trace_id = await call_plugin_callable(extractor, payload)
    return await _resolve_and_build(
        project_id,
        trace_id,
        fallback=fallback,
    )


# ════════════════════════════════════════════════════════════════════════════
# item 形态（sync judge 路径：身份从业务 item 字段提取）
# ════════════════════════════════════════════════════════════════════════════

#: item 形态身份提取器签名：业务 item 视图（``item_id`` / ``source_trace_id`` /
#: ``metadata`` 键值映射）→ ``(project_id, trace_id)``。
ItemTraceIdentityExtractor = Callable[
    [Mapping[str, Any]],
    "tuple[str | None, str | None]",
]


def default_item_trace_identity(
    item: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """默认 item 提取器：``metadata.project_id`` + ``source_trace_id``。

    parse 拦截器产物（如 search_eval：csv ``project`` 列 →
    ``metadata.project_id``、csv ``trace_id`` → ``source_trace_id``）
    直接命中；存量 sync 业务 metadata 无 ``project_id`` → ``None``
    （调用侧回退透传值，行为与现网一致）。
    """
    metadata = item.get("metadata")
    project_id = metadata.get("project_id") if isinstance(metadata, Mapping) else None
    trace_id = item.get("source_trace_id")
    return (
        str(project_id) if project_id else None,
        str(trace_id) if trace_id else None,
    )


#: benchmark 组件名 → item 形态身份提取器（未注册回退默认 item 提取器）。
_ITEM_EXTRACTOR_REGISTRY: dict[str, ItemTraceIdentityExtractor] = {}


def register_item_trace_identity_extractor(
    benchmark_name: str,
    extractor: ItemTraceIdentityExtractor,
) -> None:
    """业务注册自己的 item 形态 trace 身份提取器（同名覆盖，幂等）。"""
    _ITEM_EXTRACTOR_REGISTRY[benchmark_name] = extractor
    logger.info(
        "trace_url.item_extractor_registered benchmark=%s extractor=%s",
        benchmark_name,
        getattr(extractor, "__qualname__", type(extractor).__name__),
    )


def resolve_item_trace_identity_extractor(
    benchmark_name: str | None,
) -> ItemTraceIdentityExtractor:
    """按 benchmark 组件名解析 item 提取器：RemoteConfig binding → 注册表 → 默认。"""
    return resolve_plugin(
        "item_trace_identity_extractors",
        benchmark_name,
        resolution=PluginResolution(
            registry=_ITEM_EXTRACTOR_REGISTRY,
            default=default_item_trace_identity,
            remote_factory=lambda binding: RemoteItemTraceIdentityExtractor(
                binding,
                fallback=default_item_trace_identity,
            ),
        ),
    )


def clear_item_trace_identity_extractors() -> None:
    """清空 item 形态业务注册表（测试隔离用）。"""
    _ITEM_EXTRACTOR_REGISTRY.clear()
    _DIRECT_URL_BENCHMARKS.clear()


#: 直拼模式的 benchmark 组件名集合：该集合内业务的 trace_id 即 Langfuse
#: 主键 id，无需 TraceResolver 解析，直接拼面板链接（例：search_eval
#: 的 csv trace_id 列就是 Langfuse 主键）。
_DIRECT_URL_BENCHMARKS: set[str] = set()


def register_direct_trace_url(benchmark_name: str) -> None:
    """业务注册直拼模式（trace_id 即 Langfuse 主键，跳过 primary_id 解析）。"""
    _DIRECT_URL_BENCHMARKS.add(benchmark_name)
    logger.info("trace_url.direct_registered benchmark=%s", benchmark_name)


async def build_trace_url_for_item(
    item: Mapping[str, Any],
    *,
    benchmark_name: str | None,
    fallback: str | None = None,
) -> str | None:
    """item 形态共用主流程（sync judge 路径）：提取 → 解析 → 拼链接。

    与 :func:`build_trace_url` 共用 ``_resolve_and_build`` 主流程，差异仅在
    身份来源（业务 item 字段 vs Aone 回调 payload）。任一环节失败回退
    ``fallback``，绝不阻塞 worker 主流程。
    """
    extractor = resolve_item_trace_identity_extractor(benchmark_name)
    project_id, trace_id = await call_plugin_callable(extractor, item)
    if not project_id or not trace_id:
        return fallback
    # 直拼模式：trace_id 即 Langfuse 主键 id，跳过 primary_id 解析
    if benchmark_name in _DIRECT_URL_BENCHMARKS:
        url = f"{TRACE_TRACE_URL_BASE}/project/{project_id}/traces/{trace_id}?traceId={trace_id}"
        logger.info(
            "callback.trace_url_built project_id=%s trace_id=%s url=%s",
            project_id,
            trace_id,
            url,
        )
        return url
    return await _resolve_and_build(
        project_id,
        trace_id,
        fallback=fallback,
    )


async def _resolve_and_build(
    project_id: str | None,
    trace_id: str | None,
    *,
    fallback: str | None,
) -> str | None:
    """身份解析 + 链接拼接主流程（payload 形态 / item 形态共用）。

    project_id / trace_id 任一缺失、project_id 无配置实例、窗口内未命中、
    请求抛错均只记日志并回退 ``fallback``，绝不阻塞调用方主流程。
    """
    if not project_id or not trace_id:
        return fallback
    try:
        instances = load_trace_instances(app_settings.trace_instances)
        if project_id not in instances:
            logger.info(
                "callback.trace_url_unknown_project project_id=%s",
                project_id,
            )
            return fallback
        resolver = TraceResolver(instances)
        try:
            primary_id = await resolver.resolve_primary_id_for(project_id, trace_id)
        finally:
            await resolver.aclose()
        if not primary_id:
            logger.info(
                "callback.trace_url_no_primary project_id=%s trace_id=%s",
                project_id,
                trace_id,
            )
            return fallback
        url = (
            f"{TRACE_TRACE_URL_BASE}/project/{project_id}"
            f"/traces/{primary_id}?traceId={primary_id}"
        )
        logger.info(
            "callback.trace_url_built project_id=%s trace_id=%s url=%s",
            project_id,
            trace_id,
            url,
        )
        return url
    except Exception:
        logger.warning(
            "callback.trace_url_build_failed project_id=%s trace_id=%s",
            project_id,
            trace_id,
            exc_info=True,
        )
        return fallback


__all__ = [
    "TRACE_TRACE_URL_BASE",
    "ItemTraceIdentityExtractor",
    "TraceIdentityExtractor",
    "build_direct_trace_url",
    "build_trace_url",
    "build_trace_url_for_item",
    "clear_item_trace_identity_extractors",
    "clear_trace_identity_extractors",
    "default_item_trace_identity",
    "default_trace_identity",
    "register_direct_trace_url",
    "register_item_trace_identity_extractor",
    "register_trace_identity_extractor",
    "resolve_item_trace_identity_extractor",
    "resolve_trace_identity_extractor",
]
