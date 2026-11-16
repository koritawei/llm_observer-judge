"""DataParseInterceptor — 统一提交管线内的数据 parse 拦截点协议（§7.3）。

全系统**唯一的业务语义出口**：loader / push 请求体 / Langfuse fetch 只产
全量 KV raw items（传输层透传）；拦截器在「judge 提交与落库之前」把
raw KV 组装为业务 item（现网 ``_parse_row`` 同款结构），解析产物同时
服务落库、judge、归因、报告四个下游，消费侧不再 parse。

约束（§7.6）：拦截器**只做数据变换** —— 不得触网提交、不得改 run 状态、
不得写库。

另含 :func:`normalize_case_identity`（D5）：三处 ingest 路径（OSS loader /
push ``_to_raw_item`` / dataset ingest）共用的 case_id 归一 helper。
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger("agent_eval.benchmark.common.parse_interceptor")

#: 业务 item 输出契约必填字段（现网 ``_parse_row`` 输出契约，§7.6）。
REQUIRED_ITEM_FIELDS: tuple[str, ...] = ("id", "item_id", "input", "output", "metadata")

#: case_id 来源列优先级（D5：``case_id`` > ``id`` > ``task_id`` > seq 回退）。
CASE_ID_COLUMNS: tuple[str, ...] = ("case_id", "id", "task_id")


@dataclass(frozen=True)
class DataParseContext:
    """拦截器输入上下文（``DataParseInterceptor.parse`` 参数对象）。

    Attributes:
        benchmark_name: 组件注册名（F9）。
        rubric_ref: 本次 run 的 Rubric 类引用（缺省 component 元数据时为 ``""``）。
        raw_items: 框架全量 KV raw items（已归一 ``id`` / ``item_id``，
            OSS 路径含 ``metadata.csv_row``）。
        profile_snapshot: config_json 快照（业务可读自定义键）。
    """

    benchmark_name: str
    raw_items: list[dict[str, Any]]
    rubric_ref: str = ""
    profile_snapshot: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DataParseInterceptor(Protocol):
    """数据 parse 拦截器协议 — raw 全量 KV → 业务 item。"""

    def parse(self, ctx: DataParseContext) -> list[dict[str, Any]]:
        """输入 raw items（全量 KV），输出业务 items（``_parse_row`` 同款结构）。

        输出每条必须满足 :func:`validate_business_item` 契约：
        含 ``id`` / ``item_id`` / ``input`` / ``output`` / ``metadata``
        （保留 ``metadata.csv_row``）。
        """
        ...


def validate_business_item(item: object) -> str | None:
    """校验拦截器输出是否满足业务 item 契约（§7.6）。

    Returns:
        ``None`` 表示合法；否则返回人读错误信息（调用方按 item 级失败处理，
        不阻断整个 run）。
    """
    if not isinstance(item, dict):
        return f"item must be dict, got {type(item).__name__}"
    for field_name in REQUIRED_ITEM_FIELDS:
        if field_name not in item:
            return f"item missing required field: {field_name}"
    if not item.get("id") or not item.get("item_id"):
        return "item id / item_id must be non-empty"
    if not isinstance(item.get("metadata"), dict):
        return "item metadata must be dict"
    return None


@dataclass(frozen=True)
class CaseIdentity:
    """归一后的行身份（D5）。

    ``item_id`` 恒为技术定位键 ``{run_id}:{seq}``（Aone judge / 归因回调与
    ``external_wait`` job 的定位键）——不采用 v4 的 ``{run_id}:{case_id}``：
    业务 ``case_id`` 可能重复或含特殊字符，改 item_id 会让重复 case_id 互相
    覆盖回调；按业务 case 定位的诉求由 ``case_id`` 独立落库 + run 内唯一性
    校验满足。
    """

    id: str
    item_id: str
    case_id: str


@dataclass(frozen=True)
class CaseIdentityScope:
    """行身份归一的定位范围（:func:`normalize_case_identity` 参数对象）。

    Attributes:
        run_id: 所属 run。
        seq: 行序号（须在 run 内唯一；多批追加场景调用方传全局递增序号）。
        require_case_id: ``True`` 时缺少全部 case_id 来源列即抛
            :class:`ValueError`（v4 §1.5.5）；``False`` 回退 ``{run_id}:{seq}``。
        seen_case_ids: run 内已见 case_id 集合（传入时启用唯一性校验）：
            重复 → warning + 该行 ``case_id`` 回退 ``{run_id}:{seq}``。
    """

    run_id: str
    seq: int
    require_case_id: bool = False
    seen_case_ids: set[str] | None = None


def normalize_case_identity(
    row: Mapping[str, Any],
    scope: CaseIdentityScope | None = None,
    **legacy: object,
) -> CaseIdentity:
    """归一行身份（D5）：列优先级 ``case_id > id > task_id > seq``。

    Args:
        row: 行 KV（CSV 行 / push item / dataset item 的字段视图）。
        scope: run/seq 定位与唯一性校验范围，见 :class:`CaseIdentityScope`。

    Returns:
        :class:`CaseIdentity`：``id`` = 归一 ``case_id``；``item_id`` 恒
        ``{run_id}:{seq}``。

    Raises:
        ValueError: ``require_case_id=True`` 且缺少全部来源列。
    """
    # 兼容旧调用方的 run_id/seq 关键字；新调用方统一传 CaseIdentityScope。
    if scope is None:
        allowed = {"run_id", "seq", "require_case_id", "seen_case_ids"}
        unknown = sorted(set(legacy) - allowed)
        if unknown:
            raise TypeError(f"normalize_case_identity unexpected arguments: {unknown}")
        run_id = legacy.get("run_id")
        seq = legacy.get("seq")
        if run_id is None or seq is None:
            raise TypeError("normalize_case_identity requires scope or run_id/seq")
        seen = legacy.get("seen_case_ids")
        scope = CaseIdentityScope(
            run_id=str(run_id),
            seq=int(seq),
            require_case_id=bool(legacy.get("require_case_id", False)),
            seen_case_ids=seen if isinstance(seen, set) else None,
        )
    elif legacy:
        raise TypeError("normalize_case_identity cannot combine scope with legacy arguments")
    run_id = scope.run_id
    seq = scope.seq
    require_case_id = scope.require_case_id
    seen_case_ids = scope.seen_case_ids
    fallback = f"{run_id}:{seq}"
    case_id = ""
    for column in CASE_ID_COLUMNS:
        value = row.get(column)
        if value is None:
            continue
        text = value.strip() if isinstance(value, str) else str(value).strip()
        if text:
            case_id = text
            break
    if not case_id:
        if require_case_id:
            raise ValueError(
                f"row missing all case_id source columns "
                f"{CASE_ID_COLUMNS} at run_id={run_id} seq={seq}",
            )
        case_id = fallback
    elif seen_case_ids is not None:
        if case_id in seen_case_ids:
            logger.warning(
                "duplicate case_id %r in run %s at seq=%d; fallback to seq form",
                case_id,
                run_id,
                seq,
            )
            case_id = fallback
        else:
            seen_case_ids.add(case_id)
    return CaseIdentity(id=case_id, item_id=fallback, case_id=case_id)


__all__ = [
    "CASE_ID_COLUMNS",
    "REQUIRED_ITEM_FIELDS",
    "CaseIdentity",
    "CaseIdentityScope",
    "DataParseContext",
    "DataParseInterceptor",
    "normalize_case_identity",
    "validate_business_item",
]
