"""Badcase 归因的数据契约（Langfuse tool calling 硬切后）。

本地 evidence store / context_ops / grep-expand 已删除；
本模块只保留归因决策、运行时选项、Aone 派单与输入信封。
"""

from enum import StrEnum
from typing import Literal

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from benchmark.records.enums import TriageLevel
from benchmark.attribution.badcase.contracts import (
    CONCLUSION_TYPES,
    CandidateRootCause,
)

logger = get_logger(__name__)

#: EvidenceDetail.source 闭集（与 category/root_cause_code 同策略：闭集外归一，不抛错）。
_EVIDENCE_SOURCES: frozenset[str] = frozenset({"extra", "trace_tool", ""})

DEFAULT_MAX_REACT_ROUNDS = 4
DEFAULT_MAX_TOOL_CALLS_PER_ROUND = 3
#: Hard ceiling on a single decision's tool_calls list (parse-time).
HARD_MAX_TOOL_CALLS_PER_REQUEST = 10
#: 平台兼容默认值（未显式配置时的保守回退）。各归因 agent 的实际阈值在
#: conf/app.<env>.yaml 里显式声明（如 vehicle=0.7、example=0.65），业务决策不下沉到此。
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_LLM_CALL_TIMEOUT_S = 300.0

#: 子 judge 自定义失败类型前缀：custom 值永不进 root_cause_code（owner 路由 key）。
_CUSTOM_CODE_PREFIX = "other:"

#: FailureAssessment.disposition 的合法值；闭集外归一为 unresolved 不抛错。
_FAILURE_DISPOSITIONS: frozenset[str] = frozenset(
    {"root_cause", "downstream_symptom", "refuted", "unresolved"}
)

#: 归因根因分类闭集 — 与 badcase_attribution.prompt 的 category 枚举保持一致。
#:
#: 粗粒度分类（兼容历史归因结果）+ 三层归因体系细粒度子类
#: （tool / planning / generation 层，见 docs/async-attribution-design.md §4.2）。
ATTRIBUTION_CATEGORIES: tuple[str, ...] = (
    # ── 粗粒度分类（历史兼容）──
    "prompt",
    "tool",
    "model",
    "data",
    "orchestration",
    "infra",
    "unknown",
    # ── 中文别名（历史兼容）──
    "模型输出",
    "工具能力",
    "上下文管理",
    "工程可靠性",
    "编排控制",
    "用户语义歧义",
    "语音识别",
    # 1. 模型输出
    "模型输出|意图理解错误",
    "模型输出|工具选择错误",
    "模型输出|参数填充错误",
    "模型输出|多步推理失败",
    "模型输出|模型反思能力",
    "模型输出|幻觉",
    "模型输出|回复生成质量",
    "模型输出|输出格式违约",
    "模型输出|澄清决策失误",
    "模型输出|安全合规",
    # 2. 工具能力
    "工具能力|数据准确性",
    "工具能力|功能覆盖缺口",
    "工具能力|工具执行异常",
    "工具能力|工具缺失",
    "工具能力|工具并行度不足",
    "工具能力|工具执行超时",
    # 3. 上下文管理
    "上下文管理|prompt膨胀",
    "上下文管理|上下文溢出",
    "上下文管理|工具schema冗余",
    "上下文管理|上下文压缩损失",
    "上下文管理|个人画像/历史记忆注入",
    # 4. 编排控制
    "编排控制|循环与终止控制",
    "编排控制|节点间路由异常",
    "编排控制|异常兜底处理",
    # 5. 工程可靠性
    "工程可靠性|数据截断/丢失",
    "工程可靠性|解析失败",
    "工程可靠性|超时机制",
    "工程可靠性|流式拼接",
    "工程可靠性|渲染缺陷",
    "工程可靠性|可观测性",
    "工程可靠性|并发/限流",
    # ── 三层归因细粒度子类 ──
    # tool 工具层
    "tool_unavailable",
    "tool_exec_error",
    "data_insufficient",
    # planning 规划层
    "context_lost",
    # generation 生成层
    "factuality_error",
    "completeness_error",
    "expression_error",
)

#: 归因可回写的 triage_level 闭集（与 TriageLevel 对齐）。
_TRIAGE_LEVEL_VALUES: frozenset[str] = frozenset(m.value for m in TriageLevel)


class JudgeFailureSignal(BaseModel):
    """上游 judge / triage 给出的低分失败信号。"""

    model_config = ConfigDict(extra="allow")

    score: float | None = None
    level: str = ""
    label: str = ""
    reason: str = ""
    evaluator: str = ""


class EvidenceDetail(BaseModel):
    """字段级证据对比：不回 Langfuse 即可理解归因依据。"""

    field: str = ""
    expected: str = ""
    actual: str = ""
    source: Literal["extra", "trace_tool", ""] = ""
    observation_id: str = ""

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: object) -> str:
        """闭集外值归一为空串，不抛错（与 AttributionResult.normalize_category 同策略）。

        诱因：prompt 里的 "Judge Failure Signal" 章节标题会被模型联想成来源类别，
        自造 judge_signal 等野值；裸 Literal 校验会因此炸掉整条 evidence_details
        （连带丢弃已算出的归因结论），而这一个字段脏了不该拖累全局。
        """

        normalized = str(value or "").strip()
        if normalized in _EVIDENCE_SOURCES:
            return normalized
        logger.warning("evidence_detail.source_normalized invalid_value=%s", normalized)
        return ""


class ReactFindings(BaseModel):
    """ReAct 补证产出 —— 只答"查到了什么"，不答"结论是什么"。

    supports_low_score：证据支持/反驳评测低分，None=仍不确定。
    action_hint：ReAct 对后续行动的提示（如指向评测规则修订 / case 歧义）。
    解析失败时不伪造：空 findings 由统一决策走 insufficient_evidence。
    """

    observations_summary: str = ""
    evidence_details: list[EvidenceDetail] = Field(default_factory=list)
    discovered_refs: list[str] = Field(default_factory=list)
    supports_low_score: bool | None = None
    uncertainty_note: str = ""
    action_hint: str = Field(
        default="",
        description=(
            "ReAct 对后续行动的提示（如 rubric 修订 / case 歧义），"
            "用于 eval_rule_issue / dataset_issue 判定"
        ),
    )


class AttributionResult(BaseModel):
    """最终归因结果。"""

    trace_id: str = ""
    case_id: str = ""
    #: 数据源 trace 页面链接（benchmark 回调路径由 fused 行补充，
    #: 在线归因路径无此值保持空串）。
    trace_url: str = ""
    #: 用户 query（与 trace_id 同级下发到 Aone 工单正文 / AI Context）。
    query: str = ""
    category: str = "unknown"
    root_cause: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    action_hint: str = ""
    needs_human_review: bool = False
    raw_response: str | None = None
    should_dispatch: bool = True
    #: 归因推翻 judge/triage 时给出的纠正分级；平台据此将标量写为
    #: attribution_overridden、把本值与 original_triage_level 写入 bucket，并拒派 Aone。
    corrected_triage_level: TriageLevel | None = None
    #: 七类结论（None 兼容旧数据）；category 闭集归一会吞未知值，故独立字段。
    conclusion_type: str | None = None
    #: 稳定根因短标识（复用评测侧 failure_mode 枚举），供 owner 路由与聚类。
    root_cause_code: str | None = None
    #: 全部根因码（含主根因，去重保序）；仅用于诊断统计/聚类，
    #: owner 路由与 Aone 派单仍只认 root_cause_code。
    all_root_cause_codes: list[str] = Field(default_factory=list)
    #: 字段级证据对比，使归因结论自解释。
    evidence_details: list[EvidenceDetail] = Field(default_factory=list)
    #: `other:` 的结构化投影；不属于稳定根因码或派单路由键。
    candidate_root_cause: CandidateRootCause | None = None

    @model_validator(mode="after")
    def enforce_candidate_safety(self) -> "AttributionResult":
        """隔离未治理候选，避免污染正式路由、派单与评分纠偏。

        正式根因码与候选同时出现时，正式根因优先，候选仅记录安全告警后丢弃；
        仅有候选时，结果必须转为人工治理路径，禁止派单和评分纠偏。已审核并可
        派单的结论必须提升为正式 ``root_cause_code``，不能绕过该门禁。
        """

        if self.candidate_root_cause is None:
            return self
        if self.root_cause_code or self.all_root_cause_codes:
            logger.warning("attribution.candidate_root_cause_discarded_formal_code_present")
            self.candidate_root_cause = None
            return self
        self.root_cause_code = None
        self.all_root_cause_codes = []
        self.should_dispatch = False
        self.needs_human_review = True
        self.corrected_triage_level = None
        return self

    @field_validator("conclusion_type", mode="before")
    @classmethod
    def normalize_conclusion_type(cls, value: object) -> str | None:
        """闭集外值丢弃为 None，不抛错（与 corrected_triage_level 同策略）。"""

        if value is None:
            return None
        normalized = str(value).strip()
        return normalized if normalized in CONCLUSION_TYPES else None

    @field_validator("root_cause_code", mode="before")
    @classmethod
    def normalize_root_cause_code(cls, value: object) -> str | None:
        """custom（other:）值永不进路由 key；空串归一为 None。"""

        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized or normalized.startswith(_CUSTOM_CODE_PREFIX):
            return None
        return normalized

    @field_validator("all_root_cause_codes", mode="before")
    @classmethod
    def normalize_all_root_cause_codes(cls, value: object) -> list[str]:
        """去重保序，过滤空值与 custom（other:）码；与 normalize_root_cause_code 同策略。"""

        if not isinstance(value, (list, tuple)):
            return []
        codes: list[str] = []
        seen: set[str] = set()
        for item in value:
            normalized = str(item or "").strip()
            if not normalized or normalized.startswith(_CUSTOM_CODE_PREFIX):
                continue
            if normalized not in seen:
                seen.add(normalized)
                codes.append(normalized)
        return codes

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: object) -> str:
        """归一到 :data:`ATTRIBUTION_CATEGORIES` 闭集；非法值回落 unknown。

        回落而非报错：category 脏值不应废弃整条归因结果（root_cause /
        evidence_refs 仍有价值），但必须拦在 Aone 工单之前。
        """

        normalized = str(value or "").strip().lower()
        if normalized in ATTRIBUTION_CATEGORIES:
            return normalized
        return "unknown"

    @field_validator("corrected_triage_level", mode="before")
    @classmethod
    def normalize_corrected_triage_level(cls, value: object) -> TriageLevel | None:
        """空串/非法值丢弃为 None；合法值归一为 TriageLevel 枚举。"""

        if value is None:
            return None
        if isinstance(value, TriageLevel):
            return value
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        if normalized in _TRIAGE_LEVEL_VALUES:
            return TriageLevel(normalized)
        return None

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        """限制模型输出的置信度范围。"""

        return min(max(float(value), 0.0), 1.0)


class TraceToolCall(BaseModel):
    """归因模型请求的一次 Langfuse tool 调用。"""

    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class FailureAssessment(BaseModel):
    """单个评测现象的因果评估，仅供归因门禁使用，不进入最终结果。"""

    failure_mode: str = ""
    disposition: Literal["root_cause", "downstream_symptom", "refuted", "unresolved"] = "unresolved"
    root_cause_code: str | None = None
    upstream_failure_modes: list[str] = Field(default_factory=list)
    evaluation_fact_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("disposition", mode="before")
    @classmethod
    def normalize_disposition(cls, value: object) -> str:
        """闭集外值归一为 unresolved，不抛错（与 EvidenceDetail.source 同策略）。

        诱因：模型偶发把 conclusion_type 值（如 non_attributable_outcome）写进
        disposition，literal 校验炸掉整个 decision，连带丢失全部已算出的诊断。
        unresolved 会被门禁按「存在未决现象」正常拒绝，不放大错误。
        """

        normalized = str(value or "").strip()
        if normalized in _FAILURE_DISPOSITIONS:
            return normalized
        logger.warning("failure_assessment.disposition_normalized invalid_value=%s", normalized)
        return "unresolved"


class BadcaseAttributionDecision(BaseModel):
    """LLM 单轮 ReAct 决策（Langfuse tool calling）。"""

    decision_type: Literal["tool_calls", "final", "human_review"]
    rationale_summary: str = ""
    tool_calls: list[TraceToolCall] = Field(
        default_factory=list,
        max_length=HARD_MAX_TOOL_CALLS_PER_REQUEST,
    )
    final: AttributionResult | None = None
    #: 对每个 material failure mode 的内部因果评估，不持久化到 AttributionResult。
    failure_assessments: list[FailureAssessment] = Field(default_factory=list)
    raw_response: str | None = None

    @field_validator("rationale_summary")
    @classmethod
    def normalize_rationale_summary(cls, value: str) -> str:
        """rationale summary 是可审计摘要。"""

        return value.strip()

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "BadcaseAttributionDecision":
        """强制区分 tool 决策和 final 决策。"""

        if self.decision_type == "tool_calls":
            if not self.tool_calls:
                raise ValueError("tool_calls decision must include tool_calls.")
            if self.final is not None:
                raise ValueError("tool_calls decision must not include final.")
            return self
        if self.tool_calls:
            raise ValueError(f"{self.decision_type} decision must not include tool_calls.")
        if self.final is None:
            raise ValueError(f"{self.decision_type} decision must include final.")
        return self


class EvidenceRoundDecision(BaseModel):
    """补证模式的单轮 ReAct 决策：要么继续调工具，要么交回取证发现。

    与 :class:`BadcaseAttributionDecision`（final 模式，供旧路径回退）并存；
    补证模式下 LLM 不产出最终结论，只产出 :class:`ReactFindings`。
    """

    decision_type: Literal["tool_calls", "findings"]
    rationale_summary: str = ""
    tool_calls: list[TraceToolCall] = Field(
        default_factory=list,
        max_length=HARD_MAX_TOOL_CALLS_PER_REQUEST,
    )
    findings: ReactFindings | None = None
    raw_response: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "EvidenceRoundDecision":
        """tool_calls 与 findings 互斥；findings 决策必须携带 findings。"""

        if self.decision_type == "tool_calls":
            if not self.tool_calls:
                raise ValueError("tool_calls decision must include tool_calls.")
            if self.findings is not None:
                raise ValueError("tool_calls decision must not include findings.")
            return self
        if self.tool_calls:
            raise ValueError("findings decision must not include tool_calls.")
        if self.findings is None:
            raise ValueError("findings decision must include findings.")
        return self


class BadcaseAttributionRuntimeOptions(BaseModel):
    """归因 agent 运行时配置。"""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    max_react_rounds: int = DEFAULT_MAX_REACT_ROUNDS
    llm_call_timeout_s: float = DEFAULT_LLM_CALL_TIMEOUT_S


class TraceContextPack(BaseModel):
    """LLM 归因调用的最小身份上下文（不再承载本地 evidence）。"""

    trace_id: str = ""
    case_id: str = ""


NotificationCategory = Literal["req", "bug", "task", "risk", "nodeflownode", "plugintask"]


class NotificationRuntimeOptions(BaseModel):
    """Aone 分发配置。"""

    model_config = ConfigDict(extra="forbid")

    #: 总开关。默认关闭；yaml / per-run 显式 true 才开启。
    #: 注册文档 ``notification_dispatch_enabled=false`` 也可关闭单 agent。
    enabled: bool = False
    dry_run: bool = False
    project: str = ""
    category: NotificationCategory = "task"
    workitem_type: str = ""
    title_prefix: str = "[AgentEval Badcase]"
    #: 指派给（单责任人；Aone 工单「指派给」只允许一人）。
    assignee: str = ""
    #: 抄送人列表（花名 / 邮箱 / 工号）→ a1 ``--participant`` 逗号
    #: 拼接。Aone 无独立 CC 字段，「参与人」即抄送角色（收到工单通知）。
    participants: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    a1_bin: str = "a1"
    timeout_seconds: float = 10.0


class NotificationDispatchStatus(StrEnum):
    """Aone 分发终态。"""

    SKIPPED = "skipped"
    BLOCKED = "blocked"
    CREATED = "created"
    FAILED = "failed"


class NotificationWorkitemPayload(BaseModel):
    """创建 Aone 工单所需的稳定 payload。"""

    project: str
    category: NotificationCategory
    type: str = ""
    title: str
    body: str
    assignee: str = ""
    participants: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    priority: str = ""
    project_url: str = ""
    #: 本地文件附件（绝对路径）；经 ``a1 --attachment`` 上传，空=不带附件。
    attachments: list[str] = Field(default_factory=list)
    #: 父工单 ID；非空时经 ``a1 --relation parent:<id>`` 挂成子工单，空=平级工单。
    parent_workitem_id: str = ""


class NotificationWorkitemLink(BaseModel):
    """可由调用方直接打开的 Aone 工作项链接。"""

    cluster_index: int = 0
    title: str = ""
    workitem_id: str
    url: str


class NotificationDispatchResult(BaseModel):
    """Aone 分发结果。"""

    status: NotificationDispatchStatus
    workitem_id: str | None = None
    workitem_url: str = ""
    message: str = ""
    cluster_workitems: list[NotificationWorkitemLink] = Field(default_factory=list)
    payload: NotificationWorkitemPayload | None = None


class NotificationAIContextPayload(BaseModel):
    """Aone 正文下半段给后续自动化读取的 JSON。"""

    schema_version: str
    trace_id: str = ""
    query: str = ""
    case_id: str = ""
    trace_url: str = ""
    category: str = "unknown"
    confidence: float = 0.0
    root_cause: str = ""
    reasoning: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    action_hint: str = ""
    needs_human_review: bool = False


__all__ = [
    "ATTRIBUTION_CATEGORIES",
    "CONCLUSION_TYPES",
    "CandidateRootCause",
    "NotificationAIContextPayload",
    "NotificationCategory",
    "NotificationDispatchResult",
    "NotificationDispatchStatus",
    "NotificationRuntimeOptions",
    "NotificationWorkitemPayload",
    "AttributionResult",
    "BadcaseAttributionDecision",
    "BadcaseAttributionRuntimeOptions",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_LLM_CALL_TIMEOUT_S",
    "DEFAULT_MAX_REACT_ROUNDS",
    "DEFAULT_MAX_TOOL_CALLS_PER_ROUND",
    "EvidenceDetail",
    "EvidenceRoundDecision",
    "FailureAssessment",
    "HARD_MAX_TOOL_CALLS_PER_REQUEST",
    "JudgeFailureSignal",
    "TraceToolCall",
    "ReactFindings",
    "TraceContextPack",
]
