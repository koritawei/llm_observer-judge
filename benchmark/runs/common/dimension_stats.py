"""维度统计注册表 — 按 benchmark 组件名注册「统计维度」（§8 聚合扩展点）。

背景：rubric criteria 同时驱动「worker 评测循环」（逐 criterion 调 evaluator）
与「报告维度行渲染」——holistic judge（一次调用产出多维 sub_scores，如
search_ai_card）若把维度写进 rubric，会被 worker 当成多 criterion 重复评测。
本注册表把**统计维度**从评测语义中拆出：

- 业务在组件模块（import 副作用）注册自己的统计维度（``DimensionStatSpec``）；
- 聚合层按 run 的 benchmark 名解析维度，从 fused 行
  ``judge_json.criterion_scores[*].sub_scores`` 展开 per-维度统计，
  并入 ``criterion_metrics`` 供报告渲染；
- 未注册业务不展开、维度集为空 —— 存量行为零变化。

分制约定：``full_score`` 为维度满分（原始分口径，报告满分率 = mean /
full_score）；judge sub_scores 落库为原始分（v3 协议起 0/1/2 直传，
不归一），展开统计直读原值。

统计维度 keys 同时经 ``DimensionStatSpec`` 并入报告 spec 的 ``criteria``
（``weight=0`` 仅展示，不参与加权），报告「维度得分明细」按
``criterion_metrics[key].mean`` 渲染（满分 2 分口径走报告默认分支）。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.runs.common.rubric_base import default_rubric_ref
from benchmark.plugins.class_loader import PluginClassLoadError, load_plugin_class

logger = get_logger("agent_eval.benchmark.common.dimension_stats")


@dataclass(frozen=True)
class DimensionStatSpec:
    """单个统计维度定义。

    Attributes:
        key: 维度键（= judge sub_scores 的键，如 ``intent_understanding``）。
        name: 展示名（报告维度行标题，如 ``意图理解``）。
        full_score: 维度满分（原始分口径，报告满分率 = mean / full_score）。
        description: 维度释义（报告 criteria 描述列；缺省自动回填 ``name``）。
    """

    key: str
    name: str
    full_score: float = 2.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("DimensionStatSpec.key must not be empty")
        # description 缺省 = name（冻结实例经 object.__setattr__ 回填）
        if not self.description:
            object.__setattr__(self, "description", self.name)


#: benchmark 组件名 → 统计维度列表（未注册 = 无展开）。
_DIMENSION_STATS_REGISTRY: dict[str, tuple[DimensionStatSpec, ...]] = {}


def register_stat_dimensions(
    benchmark_name: str,
    dimensions: list[DimensionStatSpec] | tuple[DimensionStatSpec, ...],
) -> None:
    """业务注册自己的统计维度（同名覆盖，幂等）。"""
    _DIMENSION_STATS_REGISTRY[benchmark_name] = tuple(dimensions)
    logger.info(
        "dimension_stats.registered benchmark=%s dims=%s",
        benchmark_name,
        [d.key for d in dimensions],
    )


def resolve_stat_dimensions(benchmark_name: str | None) -> tuple[DimensionStatSpec, ...]:
    """按 benchmark 组件名解析统计维度；未注册 / 名空 → 空（不展开）。"""
    if benchmark_name and benchmark_name in _DIMENSION_STATS_REGISTRY:
        return _DIMENSION_STATS_REGISTRY[benchmark_name]
    return ()


def resolve_run_stat_dimensions(
    config: dict[str, Any] | None,
    benchmark_name: str | None,
) -> tuple[DimensionStatSpec, ...]:
    """报告/聚合用统计维度解析（与线上 finalize、regenerate 共用）。

    优先级（任一非空即返回）：
    1. ``config_json.stat_dimensions`` 快照（run 级真相）；
    2. 进程内 ``register_stat_dimensions`` 注册表；
    3. ``config_json.rubric_ref`` / 默认 rubric 映射加载 Rubric.stat_dimensions
       （对齐 ``rule_link_url`` 存量 run 回退，避免快照缺失时报告只剩主维）。

    未命中 → 空元组（ieval 等无子维业务行为不变）。
    """
    cfg = config if isinstance(config, dict) else {}
    dims = deserialize_stat_dimensions(cfg.get("stat_dimensions"))
    if dims:
        return dims

    dims = resolve_stat_dimensions(benchmark_name)
    if dims:
        return dims

    rubric_ref = cfg.get("rubric_ref")
    if not isinstance(rubric_ref, str) or not rubric_ref:
        rubric_ref = default_rubric_ref(benchmark_name) if benchmark_name else None
    if not rubric_ref:
        return ()
    try:
        rubric_cls = load_plugin_class(rubric_ref, what="rubric_ref")
    except PluginClassLoadError:
        logger.warning(
            "dimension_stats.rubric_ref_fallback_failed benchmark=%s rubric_ref=%s",
            benchmark_name,
            rubric_ref,
            exc_info=True,
        )
        return ()
    loaded = tuple(getattr(rubric_cls, "stat_dimensions", ()) or ())
    if loaded:
        logger.info(
            "dimension_stats.rubric_ref_fallback benchmark=%s rubric_ref=%s dims=%d",
            benchmark_name,
            rubric_ref,
            len(loaded),
        )
    return loaded


def clear_stat_dimensions() -> None:
    """清空注册表（测试隔离用）。"""
    _DIMENSION_STATS_REGISTRY.clear()


def collect_dimension_scores(
    fused_rows: list[Any],
    dimensions: tuple[DimensionStatSpec, ...],
) -> dict[str, list[tuple[float | None, bool]]]:
    """从 fused 行 judge_json 收集维度分（满分口径见 ``DimensionStatSpec.full_score``）。

    读取来源优先级（每行单源，不重复计）：
    1. ``judge_json.raw_json``（judge 原始返回原值形态）：``raw_json[cid]
       .sub_scores``；
    2. 回退 ``judge_json.criterion_scores[*].sub_scores``（框架裁剪形态 /
       直存形态同名字段）。

    sub_scores 落库口径 = judge 原始分（v3 协议起 0/1/2 直传，不归一）。
    行无 sub_scores / 维度键缺失 → 该行该维度记 ``(None, False)``
    （仅计入样本数，不参与均值——与 ``_criterion_stats`` 的 null 行
    语义一致）。

    Returns:
        ``{dimension_key: [(raw_score | None, eval_failed)]}``，维度集为空
        或无 fused 行时返回空 dict。
    """
    if not dimensions:
        return {}
    by_key: dict[str, list[tuple[float | None, bool]]] = {d.key: [] for d in dimensions}
    collected = False
    for rec in fused_rows:
        judge_json = getattr(rec, "judge_json", None)
        judge_json = judge_json if isinstance(judge_json, dict) else {}
        row_failed = getattr(rec, "status", None) == "failed"
        # 该行各维度的取值（多个 criterion 的 sub_scores 并集，后者不覆盖前者）
        row_scores: dict[str, float] = {}
        # raw_json 原值优先；缺失时回退 criterion_scores（每行单源）
        raw_json = judge_json.get("raw_json")
        sources: list[Any] = []
        if isinstance(raw_json, dict) and raw_json:
            sources.extend(v for v in raw_json.values() if isinstance(v, dict))
        if not sources:
            criterion_scores = judge_json.get("criterion_scores")
            criterion_scores = criterion_scores if isinstance(criterion_scores, dict) else {}
            sources.extend(v for v in criterion_scores.values() if isinstance(v, dict))
        for cdata in sources:
            sub_scores = cdata.get("sub_scores")
            if not isinstance(sub_scores, dict):
                continue
            for dim in dimensions:
                raw = sub_scores.get(dim.key)
                if raw is None or dim.key in row_scores:
                    continue
                try:
                    row_scores[dim.key] = float(raw)
                    collected = True
                except (TypeError, ValueError):
                    continue
        for dim in dimensions:
            by_key[dim.key].append((row_scores.get(dim.key), row_failed))
    return by_key if collected else {}


def serialize_stat_dimensions(
    dimensions: tuple[DimensionStatSpec, ...],
) -> list[dict[str, Any]]:
    """统计维度 → 可序列化 dict 列表（config_json 快照写入用）。"""
    return [
        {
            "key": d.key,
            "name": d.name,
            "full_score": d.full_score,
            "description": d.description,
        }
        for d in dimensions
    ]


def deserialize_stat_dimensions(
    data: Any,  # noqa: ANN401 — config_json 快照为异构 JSON
) -> tuple[DimensionStatSpec, ...]:
    """config_json 快照 → 统计维度（脏值/缺失 → 空）。"""
    if not isinstance(data, list):
        return ()
    out: list[DimensionStatSpec] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        out.append(
            DimensionStatSpec(
                key=str(item["key"]),
                name=str(item.get("name") or item["key"]),
                full_score=float(item.get("full_score") or 2.0),
                description=str(item.get("description") or ""),
            )
        )
    return tuple(out)


def build_report_criteria(
    rubric: Sequence[Any],
    stat_dimensions: Sequence[DimensionStatSpec],
) -> list[dict[str, Any]]:
    """组装报告 spec.criteria 展示行。

    - 注册统计维度的业务（如 search_eval，holistic judge + sub_scores 六维）：
      评测维度区只展示子维度行，不展示 holistic 主维行（主维分数已在头部
      综合分/KPI 区体现，避免重复展示）；
    - 未注册业务（ieval 等）：criteria = rubric 原样，行为不变。
    """
    if stat_dimensions:
        return [
            {
                "id": d.key,
                "name": d.name,
                "description": d.description,
                "weight": 0.0,
                "evaluator_ref": "sub_scores",
                # 满分制供报告雷达图按维归一化渲染（见 webdev prompt
                # 雷达约定：v = mean / full_score，满分到最外圈）
                "full_score": d.full_score,
            }
            for d in stat_dimensions
        ]
    return [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "weight": c.weight,
            "evaluator_ref": c.evaluator_ref,
        }
        for c in rubric
    ]


def uses_external_report_assembly(
    config: dict[str, Any],
    benchmark_name: str | None,
) -> bool:
    """判断 sync-judge finalize 是否应走 external 报告组装。

    仅当存在 ``stat_dimensions``（当前仅 search_eval 注册）时返回 True。
    ieval 无统计维度注册 → False，继续原 ``BenchmarkRunner`` 报告路径，
    行为与改前一致。不以 ``oss_prefix`` / ``parse_interceptor_ref`` 判定，
    避免把 ieval OSS 提交误导入 external 报告组装。
    """
    return bool(resolve_run_stat_dimensions(config, benchmark_name))


__all__ = [
    "DimensionStatSpec",
    "build_report_criteria",
    "clear_stat_dimensions",
    "collect_dimension_scores",
    "deserialize_stat_dimensions",
    "register_stat_dimensions",
    "resolve_run_stat_dimensions",
    "resolve_stat_dimensions",
    "serialize_stat_dimensions",
    "uses_external_report_assembly",
]
