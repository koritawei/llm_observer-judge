"""SubmitPipeline — 统一提交管线：所有入口共享的提交序列（§7.2）。

**拦截点的唯一落点**：4 个入口（push ``/{name}/run`` / ``/run-inline`` +
pull ``/run-external``(含 v2) / 日调度）在「Resolver → 提交管线」处汇流；
items 的失败落库与 judge 入队只存在于本管线内部 —— **不存在绕过
拦截点的通道**（§4.1 强保证论证）。

提交序列：

1. 【拦截点】``DataParseInterceptor.parse``：raw 全量 KV → 业务 item
   （无拦截器 → 全量 KV 直传，§7.4 框架默认）；
2. parse 失败 / 输出非法的 item → 调用方注入的 ``mark_item_failed``
   落库 ``eval_failed`` stub + 推进完成计数，**item 级失败不阻断 run**（§7.6）；
3. 业务 items 经 ``enqueue_items`` 回调入统一队列（§8.3）。
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.runs.common.parse_interceptor import (
    DataParseContext,
    validate_business_item,
)
from benchmark.runs.common.profile_resolver import ResolvedProfile
from benchmark.plugins.remote_parse import RemoteParseInterceptor

logger = get_logger("agent_eval.benchmark.common.submit_pipeline")

#: 单条 parse 异常信息截断长度（日志 / 落库列宽保护）。
_ERR_MSG_LIMIT = 500


@dataclass(frozen=True)
class SubmitSummary:
    """提交管线结果汇总（日志 / 调用方计数对齐 / 测试断言用）。"""

    total_raw: int
    """输入 raw items 总数。"""
    parsed: int
    """拦截成功（或无拦截器透传）的 item 数。"""
    parse_failed: int
    """拦截失败（异常 / 输出非法）按 item 级失败的条数。"""
    enqueued: int
    """实际入队条数。"""


def _raw_item_id(raw: dict[str, Any], run_id: str, seq: int) -> str:
    """raw item 的 item_id（日志 / 失败落库用；缺失时按序合成）。"""
    return str(raw.get("item_id") or raw.get("id") or f"{run_id}:{seq}")


@dataclass(frozen=True)
class SubmitPipelineSpec:
    """``submit_prepared_items`` 提交规格（FUNC-PARAMS-001 参数收敛）。

    ``enqueue_items``：parsed items 入队回调，返回入队数。
    ``mark_item_failed``：item 级失败回调 —— 写 ``eval_failed`` stub 落库 +
        推进 run 完成计数（计数模型由调用方按入口语义实现）。
    ``snapshot``：config_json 快照（拦截器经 ``profile_snapshot`` 读自定义键）。
    """

    run_id: str
    resolved: ResolvedProfile
    raw_items: list[object]
    enqueue_items: Callable[[list[dict[str, Any]]], Awaitable[int]]
    mark_item_failed: Callable[[dict[str, Any], str], Awaitable[None]]
    snapshot: dict[str, Any] | None = None


#: 与 search_eval parse 对齐的 Langfuse project 列别名。
_PROJECT_COLUMNS: tuple[str, ...] = ("project", "project_id")
#: Langfuse / 业务 trace 列别名（``traceid`` 兼容旧样例）。
_TRACE_ID_COLUMNS: tuple[str, ...] = ("trace_id", "traceid")


def _first_nonempty_text(row: dict[str, Any], columns: tuple[str, ...]) -> str | None:
    """按别名序取第一个非空列值；全空返回 None。"""
    for column in columns:
        value = row.get(column)
        if value is None:
            continue
        text = value.strip() if isinstance(value, str) else str(value).strip()
        if text:
            return text
    return None


def _lift_common_csv_columns(raw: dict[str, Any]) -> dict[str, Any]:
    """无业务拦截器时的通用 CSV 列提升（**仅批量/OSS 提交管线**）。

    **不影响** example 实时流：Kafka 路径走
    ``streaming.request.to_evaluator_request``，不经本函数。

    loader 产出的 raw item 只把全量 KV 放在 ``metadata.csv_row``（§7.1
    传输层透传），顶层无 input/output。自定义 benchmark（动态组件）
    无 ``parse_interceptor_ref`` 时 passthrough 直透会导致预写 run_items
    的 input/output 为 None。此处把 csv_row 中存在的通用列
    （input / output / expected_output，webUI 上传弹窗的字段契约）
    提升到顶层；并与 search_eval parse 对齐，提升 Langfuse 身份：

    - ``project`` / ``project_id`` → ``metadata.project_id``
    - ``trace_id`` / ``traceid`` → 顶层 ``source_trace_id``

    列不存在或顶层/metadata 已有值时不覆盖——不掩盖业务拦截器职责
    （有拦截器的链路不走本分支，行为不变）。
    """
    metadata = raw.get("metadata")
    row = metadata.get("csv_row") if isinstance(metadata, dict) else None
    if not isinstance(row, dict):
        return raw
    lifted = dict(raw)
    for column in ("input", "output", "expected_output"):
        value = row.get(column)
        if value is not None and column not in lifted:
            lifted[column] = value

    meta_out = dict(metadata) if isinstance(metadata, dict) else {}
    project_id = _first_nonempty_text(row, _PROJECT_COLUMNS)
    existing_project = meta_out.get("project_id")
    if project_id and not (isinstance(existing_project, str) and existing_project.strip()):
        meta_out["project_id"] = project_id

    existing_trace = lifted.get("source_trace_id")
    if not (isinstance(existing_trace, str) and existing_trace.strip()):
        trace_id = _first_nonempty_text(row, _TRACE_ID_COLUMNS)
        if trace_id:
            lifted["source_trace_id"] = trace_id

    lifted["metadata"] = meta_out
    return lifted


async def submit_prepared_items(
    spec: SubmitPipelineSpec,
) -> SubmitSummary:
    """统一提交序列：【拦截点】→ item 级失败落库 → 统一入队。

    Args:
        spec: 提交规格（字段语义见 :class:`SubmitPipelineSpec`）。

    Returns:
        :class:`SubmitSummary`（parse 失败条数 + 入队数）。
    """
    run_id = spec.run_id
    resolved = spec.resolved
    raw_items = spec.raw_items
    enqueue_items = spec.enqueue_items
    mark_item_failed = spec.mark_item_failed
    snapshot = spec.snapshot
    total_raw = len(raw_items)
    interceptor = resolved.parse_interceptor
    profile_snapshot = dict(snapshot or {})

    # §7.4：无拦截器 = 全量 KV 直传（框架默认行为）；通用 CSV 列提升
    # 见 _lift_common_csv_columns（自定义 benchmark 无拦截器时保证
    # input/output 进预写，避免 run_items 空值）。
    if interceptor is None:
        passthrough: list[dict[str, Any]] = []
        parse_failed = 0
        for seq, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                item_id = f"{run_id}:{seq}"
                await mark_item_failed(
                    {"id": item_id, "item_id": item_id},
                    f"raw item must be dict, got {type(raw).__name__}",
                )
                parse_failed += 1
                continue
            passthrough.append(_lift_common_csv_columns(raw))
        enqueued = await enqueue_items(passthrough) if passthrough else 0
        logger.info(
            "submit_pipeline.passthrough run_id=%s total=%d lifted=%d failed=%d enqueued=%d",
            run_id,
            total_raw,
            len(passthrough),
            parse_failed,
            enqueued,
        )
        return SubmitSummary(
            total_raw=total_raw,
            parsed=len(passthrough),
            parse_failed=parse_failed,
            enqueued=enqueued,
        )

    prepared: list[dict[str, Any]] = []
    parse_failed = 0
    for seq, raw in enumerate(raw_items):
        item_id = _raw_item_id(raw, run_id, seq) if isinstance(raw, dict) else f"{run_id}:{seq}"
        if not isinstance(raw, dict):
            await mark_item_failed(
                {"id": item_id, "item_id": item_id},
                f"raw item must be dict, got {type(raw).__name__}",
            )
            parse_failed += 1
            continue
        # 逐 item 调用拦截器：单条异常 / 输出非法只失败该 item（§7.6），
        # 不阻断整个 run。拦截器协议为批量签名，此处以单元素列表调用。
        # remote 拦截器实现 aparse；本地实现同步 parse。
        try:
            ctx = DataParseContext(
                benchmark_name=resolved.benchmark_name,
                rubric_ref=resolved.rubric_ref or "",
                raw_items=[raw],
                profile_snapshot=profile_snapshot,
            )
            if isinstance(interceptor, RemoteParseInterceptor):
                parsed_out = await interceptor.aparse(ctx)
            else:
                parsed_out = interceptor.parse(ctx)
        except Exception as exc:
            logger.warning(
                "parse.interceptor_failed run_id=%s item_id=%s err=%s",
                run_id,
                item_id,
                exc,
                exc_info=True,
            )
            await mark_item_failed(raw, f"{type(exc).__name__}: {exc}"[:_ERR_MSG_LIMIT])
            parse_failed += 1
            continue
        item = parsed_out[0] if parsed_out else None
        err = (
            validate_business_item(item) if item is not None else "interceptor returned empty list"
        )
        if err is not None:
            logger.warning(
                "parse.interceptor_invalid_output run_id=%s item_id=%s err=%s",
                run_id,
                item_id,
                err,
            )
            await mark_item_failed(raw, err[:_ERR_MSG_LIMIT])
            parse_failed += 1
            continue
        prepared.append(item)

    enqueued = await enqueue_items(prepared) if prepared else 0
    logger.info(
        "submit_pipeline.done run_id=%s total=%d parsed=%d parse_failed=%d enqueued=%d",
        run_id,
        total_raw,
        len(prepared),
        parse_failed,
        enqueued,
    )
    return SubmitSummary(
        total_raw=total_raw,
        parsed=len(prepared),
        parse_failed=parse_failed,
        enqueued=enqueued,
    )


__all__ = [
    "SubmitPipelineSpec",
    "SubmitSummary",
    "submit_prepared_items",
]
