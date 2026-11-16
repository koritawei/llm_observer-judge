"""Benchmark agent 能力目录：多数 agent 只属批量或实时流**一条链**；
``example_benchmark`` 可同时声明两条链（统一 EvaluatorService 别名）。

实时流（streaming：Kafka 采样 / sidecar 实时评测）与批量（batch：离线
数据集触发评测）是两种触发形态。能力绑定组装入口：实时流链路由键是
evaluator_ref，由 :func:`streaming_capabilities` 按 ``STREAMING_*`` 注册表
统一生成；批量链由静态组件以 ``benchmark_capabilities`` 声明，无声明或
页面注册的动态 agent 走 :func:`default_batch_capabilities` 缺省链。
:func:`declared_capabilities` 保留声明中的全部合法 pipeline 并去重。

- **批量 agent**（static_csv / static_json / OSS 导入 / 语料回放）：
  judge / 归因 / 报表按缺省链推导（rubric 类 ``judge_execution_policy``
  → 专用本地 judge，否则进程默认 EvaluatorService 链）；
- **实时流 agent**（realtime 线上 trace 采样）：judge 按 evaluator_ref
  路由 ``judge_<ref>``，triage / 归因按 evaluator_ref 查专属插件
  （``amap_agent`` / ``vehicle_agent``），其余走 evaluator_service 默认链，
  报表为 T+1 每日 benchmark-report。
- **双链 agent**（``example_benchmark``）：batch 与 streaming 并存，
  统一 ``evaluator_service_judge`` / ``evaluator_service_attribution``，
  落地 ``judge_example`` / ``attribution_example``；注册级 customization
  与两条链共用。

现有静态 agent 的链归属（业务约定）：

======================  =========  =====================================
benchmark name          链形态      能力绑定
======================  =========  =====================================
``ieval_external``      batch      Aone 异步 judge + 异步归因 + IEval 报表
``search_eval``         batch      本地 sync judge（judge 内嵌归因明细）
``example_benchmark``   batch+     双链统一 ``evaluator_service_judge`` /
                        streaming  ``evaluator_service_attribution``
                                  （落地 judge_example / attribution_example）
``amap_agent_benchmark`` streaming  evaluator_service judge +
                                  amap_agent_triage + amap_agent_attribution
======================  =========  =====================================

批量触发别名（WebUI / 定时调度）：取 batch 链 judge/attribution 的
``alias``；若 agent 仅声明 streaming 且 judge 为本地 sync，回退
``judge_class=evaluator_service_judge``，避免静默落到进程默认 Aone async URL。

名称约定（对外统一 agent name，见 ``agent_catalog.canonical_agent_name``）：

- agent name：对外唯一名称，benchmark name 与实时流 ``evaluator_name``
  都归一到它（``search_eval`` → ``search_ai_card``）；
- benchmark name：执行侧注册名，仅用于注册与触发；
- service_name：被测业务域（如 ``amap_chat``），落库 ``TraceRecord.agent_name``；
- evaluator_ref：与 agent name 同茎，路由 ``judge_<ref>`` /
  ``classifiers_registry[ref]`` / ``attribution_<ref>`` 组件。

展示只读：不参与 run 快照与执行路由（执行真相仍是
``ProfileResolver`` 解析后的 ``config_json`` 快照）。
"""

from collections.abc import Iterable
from dataclasses import dataclass, replace

from benchmark.runs.common.execution_classes import (
    STREAMING_ATTRIBUTION_CLASS_REFS,
    STREAMING_JUDGE_CLASS_REFS,
    STREAMING_TRIAGE_CLASS_REFS,
)
from benchmark.runs.common.rubric_base import BaseRubric
from benchmark.runs.schema import (
    AgentCustomizationPolicyView,
    AttributionToggleView,
    CustomizationKindView,
    StageCustomizationView,
)
from benchmark.reporting.framework.base import (
    BenchmarkCustomizationPolicy,
    PromptSlot,
    StageCustomizationPolicy,
)

#: 实时流专属 triage / 归因插件的 evaluator 名（streaming 任务级配置按名路由）。
_STREAMING_AGENT_BINDINGS: dict[str, tuple[str, str]] = {}

#: 实时流 judge：走 EvaluatorService 零代码路由（sync 本地，按 evaluator 名分发）。
_STREAMING_JUDGE_ALIAS = "evaluator_service_judge"
#: 实时流默认归因：按 evaluator_name 路由 attribution agent。
_STREAMING_DEFAULT_ATTRIBUTION_ALIAS = "evaluator_service_attribution"

#: 批量报表解析类的稳定展示引用（运行时按 benchmark 名缺省解析）。
_DEFAULT_REPORT_PARSER_REF = (
    "benchmark.reporting.framework.unified_report.report_parser:DefaultReportParser"
)
_DEFAULT_REPORT_DESCRIPTION = "使用默认报表解析类生成评测结果与样例明细"


def _default_report_capability(pipeline: str) -> "CapabilityBindingView":
    return CapabilityBindingView(
        pipeline=pipeline,
        kind="report",
        execution="sync",
        provider="webdev",
        description=_DEFAULT_REPORT_DESCRIPTION,
        class_ref=_DEFAULT_REPORT_PARSER_REF,
        note="未显式声明 report，自动使用默认报表解析类",
    )


@dataclass(frozen=True)
class CapabilityBindingView:
    """一项能力在该 agent 所属链上的绑定（只读展示）。"""

    pipeline: str
    """链形态：``batch``（离线批量）/ ``streaming``（实时流）；多数 agent 只属一条，
    ``example_benchmark`` 可同时声明两条。"""

    kind: str
    """能力类型：``judge`` / ``triage`` / ``attribution`` / ``report``。"""

    execution: str
    """执行方式：``sync`` / ``async`` / ``none``。"""

    provider: str
    """执行提供方：``aone`` / ``local`` / ``webdev`` / ``none``。"""

    description: str = ""
    """该能力做什么的一句话描述（随名称返回，面向页面展示）。"""

    alias: str | None = None
    """配置侧稳定别名（如 ``search_eval_judge``；触发请求可显式指定）。"""

    class_ref: str | None = None
    """实现类 ``module:Class``（服务端白名单加载，展示用）。"""

    note: str | None = None
    """补充备注（如 async 可选形态 / 报告节奏）。"""


def streaming_capabilities(evaluator_refs: Iterable[str]) -> list[CapabilityBindingView]:
    """实时流 agent 的能力链：judge / triage / 归因 / T+1 报告。

    路由真相键是 evaluator_name（与 criteria 的 ``evaluator_ref`` 同源）：
    专属 evaluator 逐个展示专属 triage / 归因插件，未登记的走
    evaluator_service 默认链。
    """
    refs = sorted({ref.strip() for ref in evaluator_refs if ref and ref.strip()})
    bindings: list[CapabilityBindingView] = [
        CapabilityBindingView(
            pipeline="streaming",
            kind="judge",
            execution="sync",
            provider="local",
            description="对采样的线上 trace 按所选 rubric 维度实时打分",
            alias=_STREAMING_JUDGE_ALIAS,
            class_ref=STREAMING_JUDGE_CLASS_REFS.get(_STREAMING_JUDGE_ALIAS),
            note=(
                f"按 evaluator 名路由 judge_<ref>（{', '.join(refs)}）"
                if refs
                else "按实时评测配置的 evaluator 名路由 judge_<ref>"
            ),
        ),
    ]
    dedicated_refs = [ref for ref in refs if ref in _STREAMING_AGENT_BINDINGS]
    default_refs = [ref for ref in refs if ref not in _STREAMING_AGENT_BINDINGS]
    for ref in dedicated_refs:
        triage_alias, attribution_alias = _STREAMING_AGENT_BINDINGS[ref]
        bindings.append(
            CapabilityBindingView(
                pipeline="streaming",
                kind="triage",
                execution="sync",
                provider="local",
                description=f"按 {ref} 分流规则对评过 trace 分级（好 / 轻坏 / 重坏）",
                alias=triage_alias,
                class_ref=STREAMING_TRIAGE_CLASS_REFS.get(triage_alias),
                note=f"{ref} 专属分流分类器",
            ),
        )
        bindings.append(
            CapabilityBindingView(
                pipeline="streaming",
                kind="attribution",
                execution="sync",
                provider="local",
                description="对不满分 trace 定位责任环节并取证归因",
                alias=attribution_alias,
                class_ref=STREAMING_ATTRIBUTION_CLASS_REFS.get(attribution_alias),
                note="另有 async 提交形态可配（任务级 attribution_execution 覆盖）",
            ),
        )
    if default_refs or not refs:
        # 未登记专属插件的 evaluator：走 evaluator_service 默认链。
        triage_note = "按 evaluator 名查 classifiers_registry 默认分类器"
        if default_refs:
            triage_note = f"{triage_note}（{', '.join(default_refs)}）"
        bindings.append(
            CapabilityBindingView(
                pipeline="streaming",
                kind="triage",
                execution="sync",
                provider="local",
                description="按 evaluator 名查注册表分流分类器，对评过 trace 分级",
                note=triage_note,
            ),
        )
        bindings.append(
            CapabilityBindingView(
                pipeline="streaming",
                kind="attribution",
                execution="sync",
                provider="local",
                description="按 evaluator 名路由归因 agent，对不满分 trace 取证归因",
                alias=_STREAMING_DEFAULT_ATTRIBUTION_ALIAS,
                class_ref=STREAMING_ATTRIBUTION_CLASS_REFS.get(
                    _STREAMING_DEFAULT_ATTRIBUTION_ALIAS,
                ),
                note="按 evaluator 名路由 attribution agent",
            ),
        )
    bindings.append(
        CapabilityBindingView(
            pipeline="streaming",
            kind="report",
            execution="sync",
            provider="webdev",
            description="每日 09:00 自动生成前一日的 HTML 评测报表（T+1，DB 直读）",
            note="T+1 每日 benchmark-report（09:00 更新前一日，DB 直读）",
        ),
    )
    return bindings


def default_batch_capabilities(rubric_cls: type[BaseRubric] | None) -> list[CapabilityBindingView]:
    """动态批量 agent（未显式声明）的缺省链。

    rubric 类声明 ``judge_execution_policy='local'`` → 专用本地 judge；
    其余走进程默认 EvaluatorService 链（RemoteConfig external-daily 可覆盖
    async 形态）。归因为同步 triage 归因链，报表按 benchmark 名缺省解析。
    """
    local_policy = getattr(rubric_cls, "judge_execution_policy", None) == "local"
    judge_note = (
        "rubric 声明专用本地 evaluator（judge_execution_policy=local）"
        if local_policy
        else "进程默认 EvaluatorService 链（RemoteConfig external-daily 可覆盖 async）"
    )
    return [
        CapabilityBindingView(
            pipeline="batch",
            kind="judge",
            execution="sync",
            provider="local",
            description="对数据集样本按所选 rubric 维度逐条打分",
            alias="local_judge" if not local_policy else None,
            class_ref=None,
            note=judge_note,
        ),
        CapabilityBindingView(
            pipeline="batch",
            kind="attribution",
            execution="sync",
            provider="local",
            description="对不满分样本定位责任环节并取证归因",
            note="同步 triage 归因链（按 rubric 维度扣分统计）",
        ),
        CapabilityBindingView(
            pipeline="batch",
            kind="report",
            execution="sync",
            provider="webdev",
            description="评测完成后生成 HTML 报表（维度得分 / 归因分布 / 样例明细）",
            class_ref=_DEFAULT_REPORT_PARSER_REF,
            note="按 benchmark 名缺省解析报表插件与模板",
        ),
    ]


_PIPELINES = frozenset({"batch", "streaming"})


def declared_capabilities(
    declared: Iterable[CapabilityBindingView],
) -> list[CapabilityBindingView]:
    """静态组件声明的能力链：保留全部合法 pipeline，按键去重。

    多数 agent 只声明 batch 或 streaming 一条；``example_benchmark`` 可
    同时声明两条（共用本地 example 配置）。同
    ``(pipeline, kind, alias, class_ref)`` 的重复项只留第一项；非法
    pipeline 丢弃。
    """
    seen: set[tuple[str, str, str | None, str | None]] = set()
    bindings: list[CapabilityBindingView] = []
    for item in declared:
        if item.pipeline not in _PIPELINES:
            continue
        normalized = item
        if item.kind == "report" and not item.class_ref:
            normalized = replace(
                item,
                class_ref=_DEFAULT_REPORT_PARSER_REF,
                description=item.description or _DEFAULT_REPORT_DESCRIPTION,
            )
        key = (item.pipeline, item.kind, item.alias, normalized.class_ref)
        if key in seen:
            continue
        seen.add(key)
        bindings.append(normalized)
    for pipeline in ("batch", "streaming"):
        has_pipeline = any(item.pipeline == pipeline for item in bindings)
        has_report = any(item.pipeline == pipeline and item.kind == "report" for item in bindings)
        if has_pipeline and not has_report:
            bindings.append(_default_report_capability(pipeline))
    return bindings


def capabilities_or_default(
    declared: Iterable[CapabilityBindingView] | None,
    rubric_cls: type[BaseRubric] | None = None,
) -> list[CapabilityBindingView]:
    """有声明用声明（去重，可含双链），无声明用批量缺省链。"""
    if declared:
        return declared_capabilities(declared)
    return default_batch_capabilities(rubric_cls)


def batch_trigger_execution_aliases(
    declared: Iterable[CapabilityBindingView] | None,
) -> dict[str, str]:
    """批量触发应携带的 ``judge_class`` / ``attribution_class`` 别名。

    优先取 batch 链声明的 alias；若 agent 仅有 streaming 链且 judge 为
    本地 sync，回退 ``evaluator_service_judge``——否则 ProfileResolver 在
    进程配置了 Aone 默认 pipeline 时会把空 judge 推成 async URL（example /
    amap_agent_benchmark 等本地 agent 曾因此打到 pre-ieval）。
    """
    bindings = declared_capabilities(declared or ())
    aliases: dict[str, str] = {}
    for capability in bindings:
        if capability.pipeline != "batch" or not capability.alias:
            continue
        if capability.kind == "judge":
            aliases.setdefault("judge_class", capability.alias)
        elif capability.kind == "attribution":
            aliases.setdefault("attribution_class", capability.alias)
    if aliases:
        return aliases
    if not bindings or bindings[0].pipeline != "streaming":
        return {}
    streaming_judge = next(
        (
            item
            for item in bindings
            if item.kind == "judge" and item.execution == "sync" and item.provider == "local"
        ),
        None,
    )
    if streaming_judge is None:
        return {}
    return {"judge_class": "evaluator_service_judge"}


# ─── 业务 prompt/skill 定制策略（policy → 视图转换与按名查找） ───


def _framework_customization_policies() -> dict[str, BenchmarkCustomizationPolicy]:
    """框架侧集中声明的存量 agent 定制策略（benchmark name → policy）。

    存量 benchmark agent 不逐个改声明：policy 由框架在此统一登记，
    与能力链注册表（``STREAMING_*`` / ``JUDGE_CLASS_REFS``）同属一处。
    新 agent 也可以在 ``BenchmarkComponentMetadata.customization`` 自行
    声明（框架登记优先于组件声明）。
    """
    # judge / attribution 定制模板上传时扫 ${var}。
    # 文档化常用宏（judge 语义别名 / build_prompt 运行时 / amap 粘贴兼容）；
    # 末尾 ``*`` = 放行任意标识符：example 按 input 实际字段扩展宏
    # （如 ${scene_id}）；袋中无值的宏渲染为空串。
    _example_prompt_variables = (
        "answer",
        "axes",
        "checkpoints",
        "context_info",
        "env_city",
        "env_location",
        "env_time",
        "expected_output",
        "history",
        "input",
        "output",
        "query",
        "response_text",
        "score_max",
        "score_min",
        "trace_evidence",
        "used_tool",
        "*",
    )
    example_judge = StageCustomizationPolicy(
        allow_prompt=True,
        allow_skill=True,
        prompt_slots=(PromptSlot(key="prompt", variables=_example_prompt_variables),),
    )
    example_attribution = StageCustomizationPolicy(
        allow_prompt=True,
        allow_skill=True,
        prompt_slots=(PromptSlot(key="prompt", variables=_example_prompt_variables),),
    )
    return {
        # prompt/skill 定制仅 example_benchmark 开放（judge / attribution
        # 均单槽位 + skill，可关闭归因）；其余存量 agent 四格全关。
        "example_benchmark": BenchmarkCustomizationPolicy(
            judge=example_judge,
            attribution=example_attribution,
            allow_attribution_toggle=True,
        ),
    }


def customization_policy_for_name(benchmark_name: str) -> BenchmarkCustomizationPolicy:
    """按 benchmark 名取定制策略（不实例化组件）。

    顺序：框架集中登记 → 组件 metadata 自行声明 → 默认全关。未找到
    静态组件（纯页面自定义名）返回默认全关——动态 agent 只有在同名
    静态组件存在时才继承其 policy。
    """
    # 注册表会反向导入本模块（services 类），只能惰性导入
    from benchmark.runs.registry import BENCHMARK_SERVICES_CLASSES  # noqa: PLC0415

    target = benchmark_name.strip()
    framework = _framework_customization_policies()
    if target in framework:
        return framework[target]
    for cls in BENCHMARK_SERVICES_CLASSES:
        try:
            meta = cls.get_metadata()
        except RuntimeError:
            continue
        if meta.name == target:
            return meta.customization
    return BenchmarkCustomizationPolicy()


def customization_policy_view(
    policy: BenchmarkCustomizationPolicy,
    capabilities: Iterable[CapabilityBindingView] = (),
) -> AgentCustomizationPolicyView:
    """定制策略 dataclass → API 视图；归因支持与否由能力链推导。"""

    def _stage(stage: str, stage_policy: object) -> StageCustomizationView:
        return StageCustomizationView(
            stage=stage,  # type: ignore[arg-type]
            prompt=CustomizationKindView(
                allowed=bool(getattr(stage_policy, "allow_prompt", False)),
                max_bytes=int(getattr(stage_policy, "max_prompt_bytes", 0) or 0),
                slots=[slot.key for slot in getattr(stage_policy, "prompt_slots", ())],
                note=str(getattr(stage_policy, "note", "") or ""),
            ),
            skill=CustomizationKindView(
                allowed=bool(getattr(stage_policy, "allow_skill", False)),
                max_bytes=int(getattr(stage_policy, "max_skill_bytes", 0) or 0),
                note=str(getattr(stage_policy, "note", "") or ""),
            ),
        )

    supported = any(
        item.kind == "attribution" and item.execution != "none" for item in capabilities
    )
    return AgentCustomizationPolicyView(
        judge=_stage("judge", policy.judge),
        attribution=_stage("attribution", policy.attribution),
        attribution_toggle=AttributionToggleView(
            allowed=bool(policy.allow_attribution_toggle),
            supported=supported,
            note=policy.note,
        ),
    )


__all__ = [
    "CapabilityBindingView",
    "batch_trigger_execution_aliases",
    "capabilities_or_default",
    "customization_policy_for_name",
    "customization_policy_view",
    "declared_capabilities",
    "default_batch_capabilities",
    "streaming_capabilities",
]
