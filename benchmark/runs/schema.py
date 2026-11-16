"""Benchmark 业务 schema — 请求/响应 payload 与外部触发模型。

本模块是 ``benchmark.runs`` 的业务数据契约。rubric、聚合、
数据集引用与统计模型等**框架通用符号**由
:mod:`benchmark.reporting.framework.schema` 定义，此处 re-export
以保持 ``from benchmark.runs.schema import ...`` 单点可用。
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from benchmark.records.enums import BenchmarkStatus
from benchmark.domain.trace_report_config import normalize_report_html_name
from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    AggregationConfig,
    AggregationMethod,
    AttributionStats,
    Criterion,
    CriterionStats,
    DatasetRef,
)
from benchmark.utils.callback_retry import is_retryable_callback_error

# ════════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════════


class TriageMode(StrEnum):
    """When triage/attribution is triggered during a benchmark run.

    - ``FUSED``: all criteria finish → fuse scores → single triage per item
    - ``PER_EVAL``: each eval completes → sync triage immediately
    - ``NONE``: no triage at all (score-only benchmark)
    """

    FUSED = "fused"
    PER_EVAL = "per_eval"
    NONE = "none"


#: Criterion ID reserved for the fused-triage pseudo criterion.  Fused
#: attribution rows are keyed under this literal in
#: ``AggregatedMetrics.attribution_metrics`` — never broadcast onto real
#: criterion IDs.
FUSED_CRITERION_ID = "fused"


# ════════════════════════════════════════════════════════════════════════════
# Dataclasses — internal service payloads (non-pydantic)
# ════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ExternalJudgeItemResult:
    """external judge 回调结果（``cas_advance_external_judge_item`` 参数对象）。

    - ``failed`` / ``score`` / ``judge_json``：判定结果
    - ``source_trace_id`` / ``trace_id`` / ``trace_url``：溯源三件套
    - ``item_input`` / ``item_output``：case 原始数据（终败补写，防报告明细全空）
    - ``criterion_id`` / ``evaluator_ref``：fused 维度元数据
    """

    run_id: str
    item_id: str
    failed: bool
    score: float
    judge_json: dict[str, Any]
    source_trace_id: str | None = None
    trace_id: str | None = None
    trace_url: str | None = None
    item_input: dict[str, Any] | None = None
    item_output: dict[str, Any] | None = None
    error_message: str | None = None
    criterion_id: str = FUSED_CRITERION_ID
    evaluator_ref: str = "fused"


# ════════════════════════════════════════════════════════════════════════════
# Pydantic models — request / response payloads
# ════════════════════════════════════════════════════════════════════════════


class BenchmarkRequest(BaseModel):
    """Inbound POST /v1/benchmarks/{name}/run body.

    Overrides the benchmark's default dataset sampling and execution config.
    Fields left as ``None`` fall back to the ``@benchmark_component`` defaults.
    """

    sample_size: int | None = Field(
        default=None,
        ge=1,
        description="Max items to evaluate; null = use benchmark default",
    )
    sample_seed: int | None = Field(
        default=None,
        description="Random seed for sampling; null = use DatasetRef.sample_seed",
    )
    max_concurrency: int = Field(
        default=10,
        ge=1,
        description="Max concurrent evaluations per criterion",
    )
    parallel_evaluators: bool = Field(
        default=True,
        description="Run all criteria in parallel (True) or serially (False)",
    )
    triage_mode: TriageMode = Field(
        default=TriageMode.FUSED,
        description="When to trigger triage: fused (post-eval), per_eval (per-item), none",
    )
    dataset_name: str | None = Field(
        default=None,
        description="Override Langfuse dataset name; null = use DatasetRef.dataset_name",
    )
    project_id: str | None = Field(
        default=None,
        description="Override Langfuse project ID; null = use DatasetRef.project_id",
    )


class BenchmarkRunResult(BaseModel):
    """Internal result of a completed (or failed) benchmark run.

    Returned by ``BenchmarkRunner.run()`` and persisted to PG.
    """

    run_id: str = Field(description="Benchmark run UUID")
    final_score: float | None = Field(
        default=None,
        description="Aggregated final score 0.0~1.0 (None if failed)",
    )
    aggregation_method: str | None = Field(
        default=None,
        description="Aggregation method used (None if failed)",
    )
    metrics: AggregatedMetrics | None = Field(
        default=None,
        description="Full aggregated metrics (None if failed)",
    )
    report_url: str | None = Field(
        default=None,
        description="HTML report URL (None if report generation failed)",
    )
    status: BenchmarkStatus = Field(
        description="Final run status (completed / failed)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if status=failed",
    )


class BenchmarkRunResponse(BaseModel):
    """HTTP 200 response for sync mode — full run result."""

    run_id: str = Field(description="Benchmark run UUID")
    benchmark_name: str = Field(description="Benchmark services name")
    status: BenchmarkStatus = Field(description="Run status")
    final_score: float | None = Field(default=None, description="Final score 0.0~1.0")
    metrics: AggregatedMetrics | None = Field(default=None, description="Aggregated metrics")
    report_url: str | None = Field(default=None, description="HTML report URL")
    error: str | None = Field(default=None, description="Error message if failed")


class BenchmarkRunCreatedResponse(BaseModel):
    """HTTP 202 response for async mode — run created, poll for results."""

    run_id: str = Field(description="Benchmark run UUID")
    benchmark_name: str = Field(description="Benchmark services name")
    status: BenchmarkStatus = Field(
        default=BenchmarkStatus.RUNNING,
        description="Initial run status (always 'running' on 202)",
    )
    poll_url: str = Field(description="URL to poll for run status")


class AgentCapabilityView(BaseModel):
    """单个 agent 的一项能力绑定（批量 / 实时流两条链，只读展示）。

    由 agent-eval 在 agent 实现内统一管理，随统一 agent 视图对外透出；
    不参与 run 快照与执行路由。
    """

    pipeline: Literal["batch", "streaming"] = Field(
        description="流程形态：batch（离线批量）/ streaming（实时流）",
    )
    kind: Literal["judge", "triage", "attribution", "report"] = Field(
        description="能力类型",
    )
    execution: Literal["sync", "async", "none"] = Field(description="执行方式")
    provider: str = Field(description="执行提供方（aone / local / webdev / none）")
    description: str = Field(
        default="",
        description="该能力做什么的一句话描述（随名称返回，面向页面展示）",
    )
    alias: str | None = Field(
        default=None,
        description="配置侧稳定别名（触发请求可显式指定）",
    )
    class_ref: str | None = Field(
        default=None,
        description="实现类 module:Class（服务端白名单加载，展示用）",
    )
    note: str | None = Field(default=None, description="展示备注")


class CustomizationKindView(BaseModel):
    """一种内容形态（prompt / skill）在某阶段是否允许业务定制。"""

    allowed: bool = Field(description="是否允许业务上传该形态")
    max_bytes: int = Field(default=0, description="单形态大小上限（字节）")
    slots: list[str] = Field(
        default_factory=list,
        description="可覆盖的 prompt 槽位（仅 prompt 形态有意义）",
    )
    note: str = Field(default="", description="展示备注")


class StageCustomizationView(BaseModel):
    """一个阶段（judge / attribution）的业务定制能力视图。"""

    stage: Literal["judge", "attribution"] = Field(description="阶段")
    prompt: CustomizationKindView = Field(description="prompt 模板覆盖能力")
    skill: CustomizationKindView = Field(description="skill 前置指令能力")


class AttributionToggleView(BaseModel):
    """attribution 业务开关的许可与支持情况。"""

    allowed: bool = Field(description="agent 是否允许业务关闭归因")
    supported: bool = Field(description="agent 是否真有归因能力")
    note: str = Field(default="", description="展示备注")


class AgentCustomizationPolicyView(BaseModel):
    """agent 级业务定制策略视图（随统一 agent 视图只读展示）。"""

    judge: StageCustomizationView = Field(description="judge 阶段定制能力")
    attribution: StageCustomizationView = Field(description="attribution 阶段定制能力")
    attribution_toggle: AttributionToggleView = Field(description="归因开关许可")


def deny_customization_view() -> AgentCustomizationPolicyView:
    """兜底策略视图：四格全关、无归因能力（未组装 policy 的 agent 行）。"""
    deny = CustomizationKindView(allowed=False)
    return AgentCustomizationPolicyView(
        judge=StageCustomizationView(stage="judge", prompt=deny, skill=deny),
        attribution=StageCustomizationView(stage="attribution", prompt=deny, skill=deny),
        attribution_toggle=AttributionToggleView(allowed=False, supported=False),
    )


class BenchmarkDefinition(BaseModel):
    """Benchmark definition — metadata exposed via GET /v1/benchmarks.

    统一 agent 视图：静态 ``@benchmark_component`` 组件与页面注册的动态
    benchmark-agent 合并对外展现；``description`` / ``owner`` /
    ``registered_at`` 仅动态 agent 携带（静态组件为 ``None``）；
    ``capabilities`` 为批量 / 实时流两条链的 judge·归因·报表绑定
    （只读展示，能力由 agent-eval 在 agent 实现内管理）。

    名称约定（对外统一 agent name，执行标识保持兼容）：

    - ``agent_name``：对外唯一 agent 名。benchmark name 与实时流
      ``evaluator_name`` 都归一到它（``search_eval`` → ``search_ai_card``、
      ``amap_agent_benchmark`` → ``amap_agent``），归一规则在
      ``agent_catalog.canonical_agent_name``；
    - ``name``：执行侧注册名（benchmark name，如 ``search_eval``），
      仅用于 agent_manager 注册与触发，与 ``agent_name`` 一一对应；
    - ``service_name``：被测业务域（如 ``amap_chat``），落库
      ``TraceRecord.agent_name``；
    - ``evaluators``：能力实现路由键（criteria 的 ``evaluator_ref`` 集合，
      与 ``agent_name`` 同茎），按它路由 ``judge_<ref>`` /
      ``classifiers_registry[ref]`` / ``attribution_<ref>`` 组件。
    """

    name: str = Field(description="Benchmark services component name")
    agent_name: str = Field(
        default="",
        description="统一 agent 名（benchmark name / evaluator_name 归一，对外视图主键）",
    )
    service_name: str = Field(description="Evaluated business service (not agentgraph agent id)")
    dataset_name: str = Field(description="Langfuse dataset name")
    project_id: str = Field(description="Langfuse project ID")
    criteria: list[dict[str, Any]] = Field(
        description="Rubric criteria (id, name, description, weight, evaluator_ref)",
    )
    aggregation_method: str = Field(description="Aggregation method")
    attribution_services: list[str] = Field(
        default_factory=list,
        description="Attribution services logic names",
    )
    description: str | None = Field(
        default=None,
        description="Page-registered agent description (static components: None)",
    )
    owner: str | None = Field(
        default=None,
        description="Page-registered agent owner (static components: None)",
    )
    registered_at: str | None = Field(
        default=None,
        description="Page-registered agent creation label (static components: None)",
    )
    capabilities: list[AgentCapabilityView] = Field(
        default_factory=list,
        description="批量 / 实时流两条链的 judge·归因·报表绑定（只读展示）",
    )
    evaluators: list[str] = Field(
        default_factory=list,
        description="criteria 的 evaluator_ref 去重集合（judge/归因/分流组件路由键）",
    )
    rubric_class: str = Field(
        default="",
        description="Rubric 实现类 module:Class（component_rubric 表示由组件元数据承载）",
    )
    rubric_description: str = Field(
        default="",
        description="rubric 评分标准说明（维度构成 / 档位口径 / 门控与阈值，随名称返回）",
    )
    customization: AgentCustomizationPolicyView = Field(
        default_factory=deny_customization_view,
        description="业务 prompt/skill 定制策略（四格 + 归因开关，只读展示）",
    )


class PipelineBenchmarkRequest(BaseModel):
    """POST /v1/benchmarks/{name}/run-pipeline body.

    Triggers a **retrospective benchmark** over Kafka-sourced
    ``trace_pipeline_records`` — scans records by date range, samples,
    and aggregates existing ``judge_json.score`` values without
    re-invoking LLM evaluators.
    """

    date_start: datetime = Field(description="Scan start time (UTC)")
    date_end: datetime = Field(description="Scan end time (UTC)")
    sample_size: int | None = Field(
        default=None,
        ge=1,
        description="Max records to sample; null = full scan (no limit)",
    )
    sample_seed: int = Field(default=42, description="Random seed for sampling")
    generate_report: bool = Field(default=True, description="Generate HTML report via webdev-agent")
    project_id: str | None = Field(
        default=None,
        description="Filter by Langfuse project_id; null = all projects",
    )


# ════════════════════════════════════════════════════════════════════════════
# External trigger models — POST /run-external + callback API (§3/§4)
# ════════════════════════════════════════════════════════════════════════════


class JudgeProfileConfig(BaseModel):
    """judge 配置（profile 内嵌，§5.1）。

    ``target`` 按 ``execution`` 多态：``async`` = 完整 trigger URL
    （``https://…/api/pipelines/{id}/trigger``，host 白名单校验）；
    ``sync`` = 本地评测类名 ``module:Class``（白名单加载）。
    ``execution`` 缺省按 target 形态推导（http(s):// → async，否则 → sync）。
    """

    execution: Literal["sync", "async"] | None = Field(
        default=None,
        description="执行方式：sync=本地评测类 / async=URL 提交+回调；缺省按 target 形态推导",
    )
    target: str | None = Field(
        default=None,
        description="async=完整 trigger URL；sync=本地评测类名 module:Class；缺省=进程默认",
    )
    provider: Literal["external", "local"] | None = Field(
        default=None,
        description="显式 provider 优先；缺省由 target 形态推导（URL→external，类名→local）",
    )
    dims: str | None = Field(
        default=None,
        description="评分维度（逗号分隔）；缺省由 Rubric criteria() 的 id 拼接",
    )
    prompt_version: str | None = Field(default=None, description="judge prompt 版本")
    class_ref: str | None = Field(
        default=None,
        description="执行类引用 module:Class（由 judge_class 别名解析；async 按它提交/解析回调）",
    )


class AttributionProfileConfig(BaseModel):
    """归因配置（profile 内嵌，§5.1 / §6.4）。

    ``target`` 多态语义与 judge 对齐；``async`` 缺省回退 judge.target
    的 URL（对齐现网 F3 回退语义）；``none`` 需显式声明。
    """

    execution: Literal["sync", "async", "none"] | None = Field(
        default=None,
        description="执行方式：sync=本地归因类 / async=URL 提交+回调 / none=不归因",
    )
    target: str | None = Field(
        default=None,
        description="async=完整 trigger URL；sync=本地归因类名 module:Class",
    )
    provider: Literal["example", "external", "local", "none"] | None = Field(
        default=None,
        description="归因 provider；缺省由 target 形态推导",
    )

    do_cross_verify: bool | None = Field(default=None, description="是否交叉验证")
    class_ref: str | None = Field(
        default=None,
        description="执行类引用 module:Class（attribution_class 别名解析；按它提交/解析回调）",
    )


class BenchmarkRuntimeProfile(BaseModel):
    """Benchmark 运行时 profile（配置化运行时契约，§5.1）。

    push 入口（``/{name}/run`` / ``/run-inline``）以可选字段形式嵌入请求体；
    pull 入口（``/run-external/v2``）直接作为请求体；pull 日调度经 RemoteConfig
    ``external_daily.benchmarks[]`` 同构配置项构造。
    """

    benchmark_name: str = Field(
        description="Benchmark services 注册名（对应 @benchmark_component name）",
    )
    rubric_ref: str | None = Field(
        default=None,
        description="Rubric 类引用 module:Class；push 缺省回退 meta.rubric（兼容）",
    )
    component_rubric: bool = Field(
        default=False,
        description="显式使用 component meta.rubric（跳过 benchmark 默认 Rubric 映射）",
    )
    parse_interceptor_ref: str | None = Field(
        default=None,
        description="数据 parse 拦截器引用 module:Class；缺省取 Rubric 类挂载值",
    )
    judge: JudgeProfileConfig = Field(default_factory=JudgeProfileConfig)
    attribution: AttributionProfileConfig = Field(
        default_factory=AttributionProfileConfig,
    )
    agent_registration_id: str | None = Field(
        default=None,
        description=(
            "页面注册行 id（版本级选择）；缺省按 benchmark 名规则解析"
            "（唯一/默认/存量最近，歧义 400）"
        ),
    )
    aggregation: AggregationConfig | None = Field(
        default=None,
        description="聚合配置覆盖（缺省取 Rubric 类自带配置 → component meta）",
    )
    # ── pull 专用（数据获取）──
    oss_prefix: str | None = Field(default=None, description="OSS 路径前缀覆盖")
    data_date: str | None = Field(
        default=None,
        pattern=r"^\d{8}$",
        description="数据日期（yyyymmdd），缺省 = 当日",
    )
    limit: int | None = Field(default=None, ge=1, description="取样条数（前 N 行）")
    run_id_suffix: str | None = Field(default=None, description="run_id 后缀（同日重跑）")
    # ── 现网语义保留 ──
    mode: Literal["judge_attribution", "attribution_only"] = Field(
        default="judge_attribution",
        description="评测模式（attribution_only 仅 attribution async 时生效）",
    )
    generate_report: bool = Field(default=True, description="是否自动生成报告")
    dataset_name: str | None = Field(default=None, description="数据集标识")
    project_id: str | None = Field(default=None, description="Langfuse project ID 覆盖")


def _reject_duplicate_profiles(
    items: list[JudgeProfileConfig] | list[AttributionProfileConfig],
    label: str,
) -> None:
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.class_ref or "", item.target or "")
        if key in seen:
            msg = f"{label} 评估器不能重复"
            raise ValueError(msg)
        seen.add(key)


class ExternalBenchmarkRequest(BaseModel):
    """POST /v1/benchmarks/run-external 请求体（手动触发 / 补跑）。

    调度器（§5.1）与手动端点（§3）在调用 runner 前将触发参数归一化为
    ``effective_date`` / ``effective_oss_prefix``（请求覆盖或调度配置默认）。

    契约迁移窗口（§9）：执行类别名（``judge_class`` / ``attribution_class`` /
    ``rubric_class``）为推荐契约；v1 兼容字段（``judge`` / ``attribution`` /
    ``rubric_ref`` / ``attribution_provider``）标记 deprecated，缺省别名时
    由 ``profile_from_external_request`` 回退解析，旧调用方在迁移期内不断裂。
    """

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str = Field(
        description="Benchmark services 注册名（对应 @benchmark_component name）",
    )
    judge_class: str | None = Field(
        default=None,
        min_length=1,
        description="Judge 执行类稳定别名（推荐契约），由服务端白名单注册表加载",
    )
    attribution_class: str | None = Field(
        default=None,
        min_length=1,
        description="Attribution 执行类稳定别名（推荐契约）；关闭归因使用 no_attribution",
    )
    rubric_class: str | None = Field(
        default=None,
        min_length=1,
        description="Rubric 类稳定别名（推荐契约）；无 Rubric 类的新业务使用 component_rubric",
    )
    date: str | None = Field(
        default=None,
        pattern=r"^\d{8}$",
        description="数据日期（yyyymmdd），缺省 = 当日",
    )
    run_id_suffix: str | None = Field(
        default=None,
        description="run_id 后缀，用于同日重跑",
    )
    manual_run_namespace: Literal["m", "s"] | None = Field(
        default=None,
        description=(
            "manual-runs 显式命名空间：立即=m，定时再次触发=s；"
            "仅 manual-runs 使用，缺省按 eval_task.trigger 推导"
        ),
    )
    oss_prefix: str | None = Field(
        default=None,
        description="OSS 路径前缀覆盖（缺省用调度配置中的固定 prefix）",
    )
    oss_object_key: str | None = Field(
        default=None,
        description=(
            "服务端拼装后的完整 OSS object key（执行侧解析结果）："
            "已上传数据集由受理入口按数据集文档 oss_prefix + "
            "oss_partition_date 拼出后写入，loader 逐字使用；"
            "调用方一般不直接传"
        ),
    )
    source_id: str | None = Field(
        default=None,
        description=(
            "再次触发的克隆源 run 行主键 id：非空时 benchmark/dataset/OSS/"
            "执行类配置以该 run 的 config_json 快照为准，仅分配新 run_id"
        ),
    )
    dataset_name: str | None = Field(
        default=None,
        description="数据集标识（语义记录，写入 config_json）",
    )
    dataset_id: str | None = Field(
        default=None,
        description=(
            "数据集文档 id：非空时服务端按 id 精确反查数据集文档"
            "（OSS 路径拼装 / 实时分流），同名数据集不串数据；"
            "缺省按 dataset_name 查（存量调用方兼容）"
        ),
    )
    eval_task_id: str | None = Field(
        default=None,
        description=(
            "webUI 评测任务 id（归属标）：非空时写入 run 快照，页面"
            "报告按任务隔离展示只认此标；缺省 = 不归属任何任务"
        ),
    )
    agent_registration_id: str | None = Field(
        default=None,
        description=(
            "页面注册行 id（版本级选择）：非空时按该注册的 prompt/skill"
            "定制与 attribution 开关执行；缺省按 benchmark 名规则解析"
        ),
    )
    project_id: str | None = Field(
        default=None,
        description="Langfuse project ID 覆盖",
    )
    generate_report: bool = Field(default=True, description="是否自动生成报告")
    aggregation: AggregationConfig | None = Field(
        default=None,
        description="聚合配置覆盖（缺省用 @benchmark_component 的 aggregation）",
    )
    mode: Literal["judge_attribution", "attribution_only"] = Field(
        default="judge_attribution",
        description=(
            "评测模式：judge_attribution（judge→归因，当前默认流程）；"
            "attribution_only（跳过 judge + task 拆分，全量 items 作为一个"
            " Aone 归因任务提交，仅异步 Aone Attribution 类时生效）"
        ),
    )
    rubric: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "外部 rubric 配置覆盖（attribution_only 模式用于注入归因维度）；"
            "缺省 = 从 @benchmark_component meta.rubric 序列化"
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description=(
            "取样条数：只解析/处理/入库 CSV 前 N 行（文件仍需整体下载，"
            "StorageClient 无流式接口）；缺省 = 全量。生效时 total_samples、"
            "judge_expected_count 等下游计数均按 N 对齐（否则 run 会等不齐"
            "回调卡死）"
        ),
    )

    # ── v1 兼容字段（deprecated，迁移窗口；别名契约优先）──
    attribution_provider: Literal["example", "external", "local", "none"] = Field(
        default="example",
        description="[deprecated] 归因来源（v1 兼容）；未传 attribution/attribution_class 时生效",
    )

    rubric_ref: str | None = Field(
        default=None,
        description="[deprecated] Rubric 类引用 module:Class；请改用 rubric_class",
    )
    judge: JudgeProfileConfig | None = Field(
        default=None,
        description="[deprecated] judge 结构化配置（execution + target）；请改用 judge_class",
    )
    attribution: AttributionProfileConfig | None = Field(
        default=None,
        description="[deprecated] 归因结构化配置；请改用 attribution_class",
    )

    # ── 配置化运行时可选扩展 ──
    parse_interceptor_ref: str | None = Field(
        default=None,
        description="数据 parse 拦截器引用 module:Class（缺省 = Rubric 类挂载值）",
    )
    report_parser_class: str | None = Field(
        default=None,
        max_length=512,
        description="报表解析类 module:Class。写入 run 的 config_json，供统一出报加载",
    )
    report_html: str | None = Field(
        default=None,
        max_length=128,
        description="templates 目录下的报告 HTML 文件名。写入 config_json，缺省用默认模板",
    )
    judges: list[JudgeProfileConfig] | None = Field(
        default=None,
        max_length=8,
        description="并行 Judge 列表。judge 仍是第一条，现有执行读 judge",
    )
    attributions: list[AttributionProfileConfig] | None = Field(
        default=None,
        max_length=8,
        description="并行 Attribution 列表。attribution 仍是第一条",
    )
    attribution_report_parser_class: str | None = Field(
        default=None,
        max_length=512,
        description="归因报表解析类 module:Class，写入 config_json",
    )
    attribution_report_html: str | None = Field(
        default=None,
        max_length=128,
        description="归因报表 HTML 文件名。与 report_html 分开写入 config_json",
    )
    # ── 归一化字段（由调度器/手动端点在调用 runner 前设置）──
    effective_date: str | None = Field(
        default=None,
        description="归一化后的数据日期（yyyymmdd），由调用方设置",
    )
    effective_oss_prefix: str | None = Field(
        default=None,
        description="归一化后的 OSS prefix，由调用方设置",
    )
    # ── case_id 归一（D5）──
    require_case_id: bool = Field(
        default=False,
        description=(
            "缺少全部 case_id 来源列（case_id/id/task_id）的行是否解析即失败；"
            "默认 false（回退 {run_id}:{seq}）"
        ),
    )

    @field_validator("report_html", "attribution_report_html", mode="before")
    @classmethod
    def _report_html_filename(cls, value: object) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        name = normalize_report_html_name(value)
        if name is None:
            msg = "report_html 只能是 templates 目录下的 html 文件名"
            raise ValueError(msg)
        return name

    @model_validator(mode="after")
    def _rubric_override_only_attribution_only(self) -> "ExternalBenchmarkRequest":
        """``rubric`` JSON 覆盖仅 ``attribution_only`` 允许（D9 收紧）。

        其余模式由 ``rubric_class`` 别名承载——快照 ``rubric`` 与
        ``rubric_ref`` 可不一致是双口径风险；非 attribution_only 传 rubric
        → 422（pydantic ValidationError）。
        """
        if self.rubric is not None and self.mode != "attribution_only":
            raise ValueError(
                "rubric override is only allowed in attribution_only mode; "
                "use rubric_class for judge_attribution mode",
            )
        if self.judges:
            _reject_duplicate_profiles(self.judges, "Judge")
            if self.judge is None:
                self.judge = self.judges[0]
        if self.attributions:
            _reject_duplicate_profiles(self.attributions, "Attribution")
            if self.attribution is None:
                self.attribution = self.attributions[0]
        return self


class CriterionScore(BaseModel):
    """单个 criterion 的评分结果（judge 回调，§4.4）。

    Aone pipeline 可能返回 >1.0 的原始分数（如 2.0），
    schema 不做上限约束；下游聚合时再做归一化。
    sub_scores 可能为嵌套 dict（如 ``{'score': 1.0, 'reason': ...}``），
    用 ``Any`` 兼容。
    """

    score: float = Field(ge=0.0, description="评分（≥0，上限不约束）")
    reason: str = Field(default="", description="评分理由")
    sub_scores: dict[str, Any] | None = Field(
        default=None,
        description="子维度分数（可为嵌套 dict）",
    )


class AttributionResult(BaseModel):
    """归因结果（attribution 回调，§7.3）。

    设计文档 ``async-attribution-design.md`` §7.3 定义的归因字段集。
    Aone pipeline 可能返回额外字段（``issue`` / ``failure_modes`` /
    ``directed_dimension`` 等），通过 ``extra="allow"`` 全部保留到
    ``model_dump()``，由下游 ``parse_aone_attribution_detail`` 过滤。
    """

    model_config = ConfigDict(extra="allow")

    # ── 评测误判检测（Phase 1，先于归因）──
    is_eval_misjudge: bool = Field(default=False, description="是否评测误判")
    eval_misjudge_reason: str = Field(default="", description="误判理由")

    # ── 三层归因（Phase 2，命中即停：工具 > 规划 > 生成）──
    primary_issue_layer: str = Field(
        default="",
        description="归因主层：tool / planning / generation / none",
    )
    primary_issue_category: str = Field(
        default="",
        description="归因子类（9 选 1，见 §4.2.2）",
    )
    primary_issue_detail: str = Field(default="", description="归因详情描述")

    # ── 维度关联（Phase 3）──
    affected_dims: list[str] = Field(
        default_factory=list,
        description="直接影响的 D 维度列表",
    )

    # ── 归因结论 ──
    severity: str = Field(
        default="",
        description="严重度：critical / major / minor",
    )
    root_cause_chain: str = Field(
        default="",
        description="根因链路文本",
    )
    improvement_priority: str = Field(
        default="",
        description="改进优先级：P0 / P1 / P2",
    )
    improvement_suggestion: str = Field(default="", description="改进建议")

    # ── 兼容旧字段（Aone 可能返回，非设计文档定义）──
    triage_level: str = Field(
        default="",
        description="[兼容] 旧分诊级别，优先使用 primary_issue_layer",
    )
    root_cause: str = Field(
        default="",
        description="[兼容] 旧根因文本，优先使用 root_cause_chain",
    )
    owner_group: str = Field(default="", description="[兼容] 责任团队标识")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="[兼容] 置信度")
    evidence: str = Field(default="", description="[兼容] 证据 / 上下文")

    @property
    def effective_triage_level(self) -> str:
        """取 primary_issue_layer，空则回退 triage_level。"""
        return self.primary_issue_layer or self.triage_level

    @property
    def effective_root_cause(self) -> str:
        """取 root_cause_chain，空则回退 root_cause。"""
        return self.root_cause_chain or self.root_cause


# ═════════════════════════════════════════════════════════════════════════
# Callback error 文本分类（§8.3）— 实现下沉至 app/utils/callback_retry.py
# （eval_queue 基建同样需要该分类，不得反向依赖业务包），此处 re-export。
# ═════════════════════════════════════════════════════════════════════════


class JudgeCallback(BaseModel):
    """Judge 回调请求体（type="judge"，§4.2）。

    judge 侧会带上非契约字段（如 ``odps`` token 计数），报告「性能 & 效率」
    栏依赖它们；``extra="allow"`` 让这些字段随 ``model_dump()`` 原样进入
    ``judge_json.raw_json``（与 :class:`AttributionResult` 同策略）。
    """

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(description="Benchmark run ID")
    type: Literal["judge"] = Field(default="judge")
    job_id: str | None = Field(default=None, description="Aone 侧 job_id")
    item_id: str | None = Field(default=None, description="Run item ID")
    criterion_scores: dict[str, CriterionScore] = Field(
        default_factory=dict,
        description="per-criterion 评分结果",
    )
    task_id: str | None = Field(default=None, description="Aone 侧 batch 任务 ID")
    project_id: str | None = Field(
        default=None,
        description="数据源 Langfuse project_id（judge 侧返回；可能缺失或为空）",
    )
    trace_id: str | None = Field(
        default=None,
        description="数据源原始 trace_id（judge 侧返回；可能缺失或为空）",
    )
    trace_url: str | None = Field(
        default=None,
        description="数据源 trace 页面链接（judge 侧返回；可能缺失或为空）",
    )
    error: str | None = Field(default=None, description="错误信息")

    @property
    def is_success(self) -> bool:
        """无 error 视为成功回调。"""
        return not (self.error and str(self.error).strip())

    @property
    def is_retryable(self) -> bool:
        """失败回调是否可 running→wait 重试（设计 §8.3）。"""
        return is_retryable_callback_error(self.error)


class AttributionCallback(BaseModel):
    """Attribution 回调请求体（type="attribution"，§7.3）。

    决策链路三列可在顶层 ``evidence``，也可在 ``metadata.evidence``。
    字段为 ``full_llm_calls_json`` / ``full_tool_calls_json`` /
    ``full_last_generation_input``。
    """

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(description="Benchmark run ID")
    type: Literal["attribution"] = Field(default="attribution")
    job_id: str | None = Field(
        default=None,
        description="对账 Job ID（params.job_id 原样回传）",
    )
    item_id: str | None = Field(default=None, description="Run item ID")
    ieval_job_id: str | None = Field(
        default=None,
        description="平台 Job ID（便于调用方轮询 GET /api/jobs/{ieval_job_id}）",
    )
    attribution: AttributionResult | None = Field(
        default=None,
        description="归因结果（§7.3）",
    )
    evidence: dict[str, Any] | None = Field(
        default=None,
        description=(
            "决策链路证据。也可放在 metadata.evidence。"
            "full_llm_calls_json / full_tool_calls_json / full_last_generation_input"
        ),
    )
    upstream: dict[str, Any] | None = Field(
        default=None,
        description="上游关联（upstream_job_id / upstream_celery_task_id / upstream_pipeline）",
    )
    task_id: str | None = Field(default=None, description="Aone 侧 batch 任务 ID")
    error: str | None = Field(default=None, description="错误信息")

    @property
    def is_success(self) -> bool:
        """无 error 视为成功回调（对齐 JudgeCallback）。"""
        return not (self.error and str(self.error).strip())

    @property
    def is_retryable(self) -> bool:
        """失败回调是否可 running→wait 重试（设计 §8.3，对齐 JudgeCallback）。"""
        return is_retryable_callback_error(self.error)


class AttributionBatchCallback(BaseModel):
    """整体归因回调请求体（attribution-only 模式，type="attribution_batch"）。

    Aone attribution 服务处理完全量 items 后发送一次批量回调，
    ``results`` 列表携带每个 item 的归因结果。
    """

    run_id: str = Field(description="Benchmark run ID")
    type: Literal["attribution_batch"] = Field(default="attribution_batch")
    task_id: str | None = Field(default=None, description="Aone 侧 batch 任务 ID")
    results: list[dict[str, Any]] = Field(
        default_factory=list,
        description=("per-item 归因结果列表 [{item_id, attribution: AttributionResult, error}]"),
    )
    error: str | None = Field(default=None, description="整体错误信息")


#: 统一回调类型（§4.5），根据 ``type`` 字段路由分发。
BenchmarkCallback = JudgeCallback | AttributionCallback | AttributionBatchCallback


class ExternalRunAcceptedResponse(BaseModel):
    """POST /run-external 202 响应（§3.3）。"""

    #: run 行主键（页面以此关联/删除单次评测；用户不可感知）
    id: str | None = Field(default=None, description="run 行主键 id")
    run_id: str = Field(description="服务端生成的 run_id")
    benchmark_name: str = Field(description="Benchmark services 名称")
    dataset_name: str | None = Field(default=None, description="数据集标识")
    status: str = Field(default="initializing", description="初始状态")
    total_samples: int = Field(default=0, description="样本数（初始为 0）")
    message: str = Field(default="", description="提示信息")


class DailyParseResult(BaseModel):
    """每日数据文件解析结果（§5.4）。"""

    items: list[dict[str, Any]] = Field(description="解析后的 item 字典列表")
    total_count: int = Field(description="数据行数（不含 header 行）")
    title_columns: list[str] = Field(description="header 行列名")
    object_key: str = Field(description="实际拉取的 OSS object key")


__all__ = [
    # Constants
    "FUSED_CRITERION_ID",
    # utils re-export（回调重试分类，保持既有导入路径兼容）
    "is_retryable_callback_error",
    # Enums
    "AggregationMethod",
    "TriageMode",
    "BenchmarkStatus",
    # Frozen dataclasses (framework re-export)
    "Criterion",
    "AggregationConfig",
    "DatasetRef",
    # Pydantic models
    "BenchmarkRequest",
    "CriterionStats",
    "AttributionStats",
    "AggregatedMetrics",
    "BenchmarkRunResult",
    "BenchmarkRunResponse",
    "BenchmarkRunCreatedResponse",
    "BenchmarkDefinition",
    "PipelineBenchmarkRequest",
    # External trigger models
    "JudgeProfileConfig",
    "AttributionProfileConfig",
    "BenchmarkRuntimeProfile",
    "ExternalBenchmarkRequest",
    "CriterionScore",
    "AttributionResult",
    "JudgeCallback",
    "AttributionCallback",
    "AttributionBatchCallback",
    "BenchmarkCallback",
    "ExternalRunAcceptedResponse",
    "DailyParseResult",
]
