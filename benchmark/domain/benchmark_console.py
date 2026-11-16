"""agent-park 评测控制台文档。

字段与页面内存对象一致，落在 ``runtime_config_records``。
这里只保存页面文档和页面上的状态切换，不进入 benchmark 执行链路。
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, NamedTuple, Self

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from tortoise.expressions import Q

from benchmark.records.models.benchmark_run import BenchmarkRun
from benchmark.records.models.runtime_config_record import (
    KIND_AGENT_SCOPE,
    KIND_BENCHMARK_AGENT,
    KIND_BENCHMARK_DATASET,
    KIND_EVAL_TASK,
    KIND_EVALUATOR,
    KIND_TRACE_REPORT,
    SCOPE_GLOBAL,
    RuntimeConfigRecord,
    agent_scope_key,
    agent_scope_record_key,
    benchmark_agent_record_key,
    benchmark_dataset_record_key,
    eval_task_record_key,
    parse_evaluator_key_tail,
    split_evaluator_record_key,
    trace_report_record_key,
)
from benchmark.records.models.trace_record import TraceRecord
from benchmark.domain.agent_scope_ownership import (
    RegistrationVisibility,
    get_ownership_policy,
    is_registration_visible,
)
from benchmark.domain.eval_config import parse_eval_identity_key
from benchmark.domain.eval_count_keys import eval_count_name_aliases, normalize_eval_count_key
from benchmark.domain.runtime_config_store import (
    refresh_eval_config_snapshot,
    restore_runtime_config,
    retire_runtime_config,
    upsert_runtime_config,
)
from benchmark.domain.schemas import RemoteConfigModelConfig
from benchmark.domain.trace_report_config import normalize_report_html_name
from benchmark.runs.common.oss_date_macro import validate_date_macros
from benchmark.runs.common.rubric_classes import (
    RUBRIC_CLASS_REFS,
    resolve_rubric_class,
)
from benchmark.customization.manifest import slugify
from benchmark.reporting.framework.schema import Criterion
from benchmark.reporting.framework.unified_report.report_parser import (
    ReportParserError,
    resolve_report_parser,
    validate_rubric_class_ref,
)
from benchmark.plugins.class_loader import load_plugin_class

logger = get_logger(__name__)

AgentScopeId = Literal["amap-agent", "vehicle-agent", "vui-agent", "other", "skill"]
#: 内置评测对象（页面预置，不可删除）。自定义对象经 console 接口落库后同权参与隔离。
BUILTIN_AGENT_SCOPES: tuple[str, ...] = (
    "amap-agent",
    "vehicle-agent",
    "vui-agent",
    "other",
    "skill",
)
DatasetSource = Literal["static_csv", "static_json", "oss", "realtime"]
DatasetStatus = Literal["parsing", "ready", "sampling", "collecting"]
TaskStatus = Literal["active", "stopped", "complete", "failed"]
TriggerMode = Literal["immediate", "weekly", "daily", "cron"]
RubricFormula = Literal["weighted-average", "pass-rate", "weighted-sum", "custom"]
AgentCapability = Literal["judge", "attribution"]

_CRON_FIELD_COUNT = 5
_SAMPLING_TERM_MAX = 256
#: Aone 指派/抄送：数字工号或花名（中英文/数字/._-）。
_NOTIFICATION_PERSON_RE = re.compile(
    r"^(?:[0-9]{1,32}|[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_.\-]{0,63})$"
)
_PAGE_FORMULA: dict[str, RubricFormula] = {
    "weighted_avg": "weighted-average",
    "weighted-average": "weighted-average",
    "pass_rate": "pass-rate",
    "pass-rate": "pass-rate",
    "weighted_sum": "weighted-sum",
    "weighted-sum": "weighted-sum",
}


class ConsoleError(Exception):
    """页面文档无法按当前状态写入。"""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class _PageModel(BaseModel):
    """页面 JSON 使用 camelCase。未知字段拒绝，避免密钥混进文档。"""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SampleRow(_PageModel):
    """数据集样例行。"""

    id: str = Field(min_length=1, max_length=128)
    input: str
    output: str = ""
    expected_output: str | None = Field(default=None, alias="expectedOutput")
    metadata: dict[str, str] = Field(default_factory=dict)
    trace_id: str = Field(default="", alias="traceId")


class DatasetVersion(_PageModel):
    """数据集版本。"""

    version_id: str = Field(alias="versionId", min_length=1, max_length=32)
    origin: Literal["initial", "replay"]
    items_count: int = Field(alias="itemsCount", ge=0)
    created_at: str = Field(alias="createdAt", min_length=1, max_length=64)
    owner: str = Field(min_length=1, max_length=128)
    agent_request_url: str | None = Field(default=None, alias="agentRequestUrl")
    sample_rows: list[SampleRow] | None = Field(default=None, alias="sampleRows")


class SamplingVariableMapping(_PageModel):
    """线上 trace 字段到模板变量的一条映射。"""

    template_variable: str = Field(alias="templateVariable", min_length=1, max_length=128)
    trace_object: Literal["trace", "observation"] = Field(alias="traceObject")
    selected_column_id: str = Field(alias="selectedColumnId", min_length=1, max_length=128)
    json_selector: str | None = Field(default=None, alias="jsonSelector", max_length=256)

    @field_validator("json_selector", mode="before")
    @classmethod
    def _blank_selector(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class TraceSamplingConditions(_PageModel):
    """线上 trace 实时采样条件。空列表表示该条件不限，不进入 Kafka 执行。"""

    #: 数据摄取方式：与 ``EvalConfigEntry.ingest`` 同枚举。缺省 kafka（Kafka 采样）；
    #: sidecar 时由 sidecar ingest 提取，Kafka 不采样。
    ingest: Literal["kafka", "sidecar"] = Field(default="kafka")
    environment: list[str] = Field(default_factory=list, max_length=32)
    name: list[str] = Field(default_factory=list, max_length=64)
    tags: list[str] = Field(default_factory=list, max_length=64)
    variable_mapping: list[SamplingVariableMapping] = Field(
        default_factory=list,
        alias="variableMapping",
        max_length=32,
    )

    @field_validator("environment", "name", "tags")
    @classmethod
    def _clean_terms(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if any(len(item) > _SAMPLING_TERM_MAX for item in cleaned):
            msg = "采样条件单项不能超过 256 个字符"
            raise ValueError(msg)
        return cleaned


class CollectionInfo(_PageModel):
    """语料收集进度。端点只记录，不在这里发起调用。"""

    spec_agent_endpoint: str = Field(alias="specAgentEndpoint", min_length=1, max_length=512)
    collected_traces: int = Field(alias="collectedTraces", ge=0)
    done: bool = False
    started_at: str | None = Field(default=None, alias="startedAt")
    completed_at: str | None = Field(default=None, alias="completedAt")


class AgentScopeDocument(_PageModel):
    """页面上的一个自定义评测对象（业务传入 agent name 即建）。

    落库 ``kind=agent_scope``（同表新增 kind，不新建表）；``id`` 即
    业务 agent name，作为数据集 / 任务隔离键（与内置 scope 同权）。
    """

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    description: str = ""
    owner: str = Field(min_length=1, max_length=128)
    created_at: str = Field(alias="createdAt", min_length=1, max_length=64)


class DatasetDocument(_PageModel):
    """页面上的一条数据集。"""

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    agent_scope: AgentScopeId | str = Field(alias="agentScope")
    name: str = Field(min_length=1, max_length=256)
    source: DatasetSource
    items_count: int = Field(alias="itemsCount", ge=0)
    status: DatasetStatus
    created_at: str = Field(alias="createdAt", min_length=1, max_length=64)
    #: 实时采样数据集归属的 Langfuse project（实时评测链路由键；
    #: 缺省由 agent registry 按 service_name 反查）。
    project_id: str | None = Field(default=None, alias="projectId", max_length=128)
    #: 流级业务归属 agent 名（多 agent 同 project 场景；须命中 registry
    #: 候选集，空则不指定归属）。
    agent_name: str | None = Field(default=None, alias="agentName", max_length=128)
    #: 实时评测登记的 evaluator 名（停止采样/任务时定位并停用
    #: kind=evaluator 条目的路由键；老数据集缺失时回退按 agentName 扫描）。
    evaluator_name: str | None = Field(default=None, alias="evaluatorName", max_length=128)
    #: 实时流实例 id（默认等于数据集 id；创建时带 ms，保证同日多次登记可区分）。
    #: 激活时写入 evaluator record_key / 采样键 / T+1 报表命名空间。
    stream_instance_id: str | None = Field(default=None, alias="streamInstanceId", max_length=160)
    #: 实时评测 prompt 覆盖（${variable} 语法；缺省用评测器 YAML 默认）。
    prompt: str | None = Field(default=None, max_length=16384)
    #: 实时评测 judge 模型覆盖（apiKeyEnv 只存环境变量名，密钥不入库）。
    evaluator_model: RemoteConfigModelConfig | None = Field(
        default=None,
        alias="evaluatorModel",
    )
    tool_name: str | None = Field(default=None, alias="toolName")
    skill_name: str | None = Field(default=None, alias="skillName")
    collection: CollectionInfo | None = None
    oss_bucket: str | None = Field(default=None, alias="ossBucket")
    #: OSS 导入型数据集的路径前缀，允许携带日期宏
    #: ``${yyyymmdd}`` / ``${yyyymmdd+n}`` / ``${yyyymmdd-n}``（触发时按本次
    #: run 数据日展开）；非法宏在此直接拒绝（400），不透传到 OSS。
    oss_prefix: str | None = Field(default=None, alias="ossPrefix")

    @field_validator("oss_prefix")
    @classmethod
    def _check_oss_date_macro(cls, value: str | None) -> str | None:
        """前缀里的日期宏只允许 yyyymmdd 形态，非法宏拒绝落库。"""
        if value:
            validate_date_macros(value)
        return value

    #: 上传型数据集的 OSS 分区日期（yyyymmdd，上传落盘当日）：触发时由
    # 服务端拉取本字段与 oss_prefix 拼出完整 object key
    # （{prefix}/{ossPartitionDate}/benchmark_final_results.csv），不存
    # 冻结整串。管道日产型数据集无此字段，按评测日期取当日分区。
    oss_partition_date: str | None = Field(
        default=None, alias="ossPartitionDate", pattern=r"^\d{8}$"
    )
    #: 当日上传顺序号（数字字符串，服务端计数器 1 起递增，可看上传
    # 先后）：同 prefix 同日多次上传落不同 object key 不互相覆盖。
    # 拼装形态
    # {prefix}/{ossPartitionDate}/{ossUploadId}/benchmark_final_results.csv；
    # 存量文档无此字段，按旧布局 {prefix}/{date}/{file} 拼装。
    oss_upload_id: str | None = Field(
        default=None, alias="ossUploadId", pattern=r"^[A-Za-z0-9._-]+$", max_length=128
    )

    @model_validator(mode="before")
    @classmethod
    def _convert_legacy_oss_object_key(cls, data: object) -> object:
        """存量文档一次性转换：旧文档只有冻结整串 ossObjectKey，读入时

        拆出倒数第二段的分区日期转存 ossPartitionDate 并丢弃整串，
        此后文档即纯 prefix+日期形态（下次保存落库也不再带整串）。
        这是存量数据迁移，不是触发兜底：触发路径只做拼装。
        无法解析日期段的历史文档整串直接丢弃（不可拼装，不保留）。
        """
        if not isinstance(data, dict):
            return data
        legacy = data.pop("ossObjectKey", None) or data.pop("oss_object_key", None)
        if legacy and not (data.get("ossPartitionDate") or data.get("oss_partition_date")):
            segments = str(legacy).rstrip("/").split("/")
            date_segment = segments[-2] if len(segments) > 1 else ""
            if re.fullmatch(r"\d{8}", date_segment):
                data["ossPartitionDate"] = date_segment
        return data

    sampling_ratio: float | None = Field(default=None, alias="samplingRatio", ge=0, le=1)
    trace_service: str | None = Field(default=None, alias="traceService")
    trace_sampling: TraceSamplingConditions | None = Field(default=None, alias="traceSampling")
    replay_enabled: bool | None = Field(default=None, alias="replayEnabled")
    agent_request_url: str | None = Field(default=None, alias="agentRequestUrl")
    session_ref: str | None = Field(default=None, alias="sessionRef")
    userprofile_ref: str | None = Field(default=None, alias="userprofileRef")
    memory_ref: str | None = Field(default=None, alias="memoryRef")
    versions: list[DatasetVersion] | None = None
    sample_rows: list[SampleRow] = Field(default_factory=list, alias="sampleRows", max_length=2000)


class CapabilityBinding(_PageModel):
    """Judge 或 Attribution 的一种执行组合。异步本地会被拒绝。"""

    execution: Literal["sync", "async"]
    placement: Literal["local", "remote"]
    class_ref: str | None = Field(default=None, alias="classRef", max_length=512)
    target_url: str | None = Field(default=None, alias="targetUrl", max_length=512)

    @model_validator(mode="after")
    def _match_agent_eval_target(self) -> Self:
        if self.execution == "async" and self.placement == "local":
            msg = "异步执行需要远程 trigger URL，不能与本地类名组合"
            raise ValueError(msg)
        ref = (self.class_ref or "").strip()
        if ":" not in ref:
            msg = "执行需要类名 module:Class"
            raise ValueError(msg)
        self.class_ref = ref
        if self.placement == "local":
            self.target_url = None
            return self
        url = (self.target_url or "").strip()
        if not url.startswith(("http://", "https://")):
            msg = "远程执行需要 http(s) trigger URL"
            raise ValueError(msg)
        self.target_url = url
        return self


def _reject_duplicate_bindings(bindings: list[CapabilityBinding], label: str) -> None:
    seen: set[tuple[str, str]] = set()
    for item in bindings:
        key = (item.class_ref or "", item.target_url or "")
        if key in seen:
            msg = f"{label} 评估器不能重复"
            raise ValueError(msg)
        seen.add(key)


def _check_report_binding(
    enabled: bool,
    parser: str | None,
    html: str | None,
    title: str,
) -> None:
    if enabled:
        if not parser:
            msg = f"生成{title}需要填写报表解析类 module:Class"
            raise ValueError(msg)
    elif parser or html:
        msg = f"未勾选生成{title}时不能填写报表解析类或报表 HTML"
        raise ValueError(msg)
    if parser is not None:
        try:
            resolve_report_parser(parser)
        except ReportParserError as exc:
            raise ValueError(str(exc)) from exc


class CustomizationFile(_PageModel):
    """bundle 内一个文件的校验信息（Git 内容的不可变指纹）。"""

    path: str
    sha256: str
    bytes: int = Field(ge=0)


class PromptContentRef(_PageModel):
    """prompt 定制引用：覆盖的 slot 集合 + 文件校验信息。"""

    slots: list[str] = Field(default_factory=list)
    files: list[CustomizationFile] = Field(default_factory=list)


class SkillContentRef(_PageModel):
    """skill 定制引用：入口文件 + 全部文件校验信息。"""

    entry: str = "SKILL.md"
    files: list[CustomizationFile] = Field(default_factory=list)


class StageCustomizationRef(_PageModel):
    """一个 stage 的定制内容引用（指向 Git 仓库中一个不可变版本目录）。"""

    stage: Literal["judge", "attribution"]
    content_version: int = Field(alias="contentVersion", ge=1)
    git_branch: str = Field(alias="gitBranch")
    git_path: str = Field(alias="gitPath")
    git_commit: str = Field(alias="gitCommit")
    bundle_sha256: str = Field(alias="bundleSha256")
    prompt: PromptContentRef | None = None
    skill: SkillContentRef | None = None
    created_at: str = Field(alias="createdAt")
    created_by: str = Field(default="", alias="createdBy")
    status: Literal["active", "archived"] = "active"


class AgentCustomization(_PageModel):
    """注册的业务定制：judge / attribution 两 stage 各自可空。"""

    judge: StageCustomizationRef | None = None
    attribution: StageCustomizationRef | None = None


class AgentDocument(_PageModel):
    """页面上的一个 benchmark-agent。"""

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    endpoint: str = Field(default="", max_length=512)
    capabilities: list[AgentCapability] = Field(default_factory=lambda: ["judge"])
    judge: CapabilityBinding | None = None
    judges: list[CapabilityBinding] = Field(default_factory=list, max_length=8)
    attribution: CapabilityBinding | None = None
    attributions: list[CapabilityBinding] = Field(default_factory=list, max_length=8)
    rubric_ids: list[str] = Field(default_factory=list, alias="rubricIds", max_length=64)
    rubric_class: str | None = Field(default=None, alias="rubricClass", max_length=512)
    rubric_doc_url: str | None = Field(default=None, alias="rubricDocUrl", max_length=1024)
    generate_report: bool = Field(default=False, alias="generateReport")
    report_parser: str | None = Field(default=None, alias="reportParser", max_length=512)
    report_html: str | None = Field(default=None, alias="reportHtml", max_length=128)
    generate_attribution_report: bool = Field(default=False, alias="generateAttributionReport")
    attribution_report_parser: str | None = Field(
        default=None,
        alias="attributionReportParser",
        max_length=512,
    )
    attribution_report_html: str | None = Field(
        default=None,
        alias="attributionReportHtml",
        max_length=128,
    )
    owner: str = Field(min_length=1, max_length=128)
    #: 业务 owner 花名列表；展示主责取 ``owners[0]``（与 ``owner`` 同步）。
    owners: list[str] = Field(default_factory=list)
    registered_at: str = Field(alias="registeredAt", min_length=1, max_length=64)
    status: Literal["online", "offline"] = "online"
    evaluation_count: int = Field(default=0, alias="evaluationCount", ge=0)
    #: 分链分量（只读派生，不落库）：批量 = run items；实时流 = trace 行
    batch_count: int = Field(default=0, alias="batchCount", ge=0)
    streaming_count: int = Field(default=0, alias="streamingCount", ge=0)
    # ── 业务定制（全部可选，存量文档零改动）──
    #: 业务标识（slug）；启用 prompt/skill 定制时必填
    business: str | None = Field(default=None, max_length=128)
    #: 业务变体名（slug）；与 business 一起定位一组注册版本
    variant_key: str | None = Field(default=None, alias="variantKey", max_length=128)
    #: 注册版本（同一 (name, business, variantKey) 单调递增，服务端分配）
    version: int = Field(default=1, ge=1)
    #: 同 name 至多一个默认注册（触发未指定版本时用）
    is_default: bool = Field(default=False, alias="isDefault")
    #: 业务是否启用 attribution；false = 该注册的所有 run 强制 no_attribution。
    #: 顶层开关（不在 customization 内）：没有内容定制也可以单独关闭。
    attribution_enabled: bool = Field(default=True, alias="attributionEnabled")
    #: 该注册是否派通知。false 时归因仍落库，派单跳过（agent_notification_disabled）。
    notification_dispatch_enabled: bool = Field(
        default=True,
        alias="notificationDispatchEnabled",
    )
    #: 非空时覆盖远程配置里的项目 ID。
    notification_project_id: str | None = Field(
        default=None,
        alias="notificationProjectId",
    )
    #: 指派给：花名或数字工号。空表示使用远程配置里的 assignee。
    notification_assignee: str | None = Field(
        default=None,
        alias="notificationAssignee",
    )
    #: 抄送：花名或数字工号列表。空表示使用远程配置里的 participants。
    notification_participants: list[str] = Field(
        default_factory=list,
        alias="notificationParticipants",
    )
    #: prompt/skill 内容定制引用（judge / attribution 各自可空）
    customization: AgentCustomization | None = None
    #: 定制注册绑定的评测对象（L2）；无定制的全局底版可空
    agent_scope: AgentScopeId | str | None = Field(default=None, alias="agentScope", max_length=128)
    #: 内部 agent 名（同代码多实例隔离，设计 §12.2）：服务端首次保存时
    #: 按实例（name+business+variant+scope）分配，格式 agx_{stem}_{序号}；
    #: 永不变更、永不复用，客户端传值一律忽略；对外 API 不回传（见 _public）。
    internal_agent_name: str | None = Field(default=None, alias="internalAgentName", max_length=64)

    @model_validator(mode="after")
    def _check_customization(self) -> Self:
        if self.customization:
            for key, ref in (
                ("judge", self.customization.judge),
                ("attribution", self.customization.attribution),
            ):
                if ref is None:
                    continue
                if ref.stage != key:
                    msg = f"{key} 定制必须属于 {key} 阶段"
                    raise ValueError(msg)
                if ref.prompt is None and ref.skill is None:
                    msg = f"{key} 阶段至少需要 prompt 或 skill 之一"
                    raise ValueError(msg)
            if not (self.business and self.variant_key):
                msg = "启用 prompt/skill 定制必须填写 business 与 variantKey"
                raise ValueError(msg)
            if not (self.agent_scope and str(self.agent_scope).strip()):
                msg = "启用 prompt/skill 定制必须填写 agentScope"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _normalize_owners(self) -> Self:
        """归一 ``owners`` / ``owner``。

        - 显式 ``owners``：每项必须是花名或数字工号（新字段，严格校验）。
        - 仅有存量 ``owner`` 时宽松回填到 ``owners``，避免邮箱/团队名等旧
          自由文本在读路径 ``model_validate`` 时 500。
        """
        cleaned: list[str] = []
        for item in self.owners:
            text = str(item).strip()
            if not text or text in cleaned:
                continue
            if _NOTIFICATION_PERSON_RE.fullmatch(text) is None:
                msg = "owners 只能包含花名或数字工号"
                raise ValueError(msg)
            cleaned.append(text)
        owner_text = self.owner.strip()
        if not cleaned and owner_text:
            # 存量自由文本 owner：不按花名规则拒绝，仅回填为 owners[0]。
            cleaned = [owner_text]
        if not cleaned:
            msg = "owners 至少需要一个花名"
            raise ValueError(msg)
        return self.model_copy(update={"owners": cleaned, "owner": cleaned[0]})

    @field_validator("notification_project_id", "notification_assignee", mode="before")
    @classmethod
    def _blank_notification_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("notification_assignee")
    @classmethod
    def _person_assignee(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _NOTIFICATION_PERSON_RE.fullmatch(value) is None:
            msg = "notificationAssignee 必须是花名或数字工号"
            raise ValueError(msg)
        return value

    @field_validator("notification_participants", mode="before")
    @classmethod
    def _blank_participants(cls, value: object) -> object:
        if value is None:
            return []
        return value

    @field_validator("notification_participants")
    @classmethod
    def _person_participants(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            if isinstance(item, bool):
                msg = "notificationParticipants 只能包含花名或数字工号"
                raise ValueError(msg)
            text = str(item).strip()
            if not text:
                continue
            if _NOTIFICATION_PERSON_RE.fullmatch(text) is None:
                msg = "notificationParticipants 只能包含花名或数字工号"
                raise ValueError(msg)
            if text not in cleaned:
                cleaned.append(text)
        return cleaned

    @field_validator("rubric_doc_url", mode="before")
    @classmethod
    def _blank_doc_url(cls, value: object) -> object:
        if value is None or not isinstance(value, str):
            return value
        cleaned = value.strip()
        return cleaned or None

    @field_validator("report_html", "attribution_report_html", mode="before")
    @classmethod
    def _report_html_filename(cls, value: object) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        name = normalize_report_html_name(value)
        if name is None:
            msg = "报表 HTML 只能是 templates 目录下的文件名"
            raise ValueError(msg)
        return name

    @model_validator(mode="after")
    def _known_report_binding(self) -> Self:
        self._expand_evaluator_lists()
        if self.rubric_class is not None:
            validate_rubric_class_ref(self.rubric_class)
        _check_report_binding(
            self.generate_report,
            self.report_parser,
            self.report_html,
            "报表",
        )
        _check_report_binding(
            self.generate_attribution_report,
            self.attribution_report_parser,
            self.attribution_report_html,
            "归因报表",
        )
        if self.rubric_doc_url and not self.rubric_doc_url.startswith(("http://", "https://")):
            msg = "rubric 文档 URL 需要 http 或 https"
            raise ValueError(msg)
        return self

    def _expand_evaluator_lists(self) -> None:
        """旧文档只有单条时收进列表；列表存在时单条字段保持为第一条。"""
        if not self.judges and self.judge is not None:
            self.judges = [self.judge]
        elif self.judges:
            self.judge = self.judges[0]
        if not self.attributions and self.attribution is not None:
            self.attributions = [self.attribution]
        elif self.attributions:
            self.attribution = self.attributions[0]
        _reject_duplicate_bindings(self.judges, "Judge")
        _reject_duplicate_bindings(self.attributions, "Attribution")


#: judge / attribution / 报表能力已由 agent-eval 在 agent 实现内统一管理，
#: 页面历史遗留的执行链字段写前一律剥离（读路径保留 schema 兼容存量文档）。
_AGENT_EXECUTION_FIELD_DEFAULTS: dict[str, object] = {
    "judge": None,
    "judges": [],
    "attribution": None,
    "attributions": [],
    "report_parser": None,
    "report_html": None,
    "generate_report": False,
    "attribution_report_parser": None,
    "attribution_report_html": None,
    "rubric_doc_url": None,
}


def strip_agent_execution_fields(document: AgentDocument) -> AgentDocument:
    """写入前剥离 agent 文档的能力执行链字段（服务端管理，页面不再采集）。"""
    return document.model_copy(update=dict(_AGENT_EXECUTION_FIELD_DEFAULTS))


class DailyReport(_PageModel):
    """实时采样任务的一日报告指针。"""

    date: str = Field(min_length=1, max_length=32)
    score: float
    items_count: int = Field(alias="itemsCount", ge=0)
    report_url: str | None = Field(default=None, alias="reportUrl", max_length=1024)
    oss_report_url: str | None = Field(default=None, alias="ossReportUrl", max_length=2048)
    #: 报告所属 run 的行主键（唯一键，用户不可感知；逐期删除按此定位）
    id: str | None = Field(default=None, max_length=64)
    #: 业务 run_id（查看全部 cases 优先用；缺省时前端可回退 id）
    run_id: str | None = Field(default=None, alias="runId", max_length=128)


class EvalTaskDocument(_PageModel):
    """页面上的一条评测任务。"""

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    agent_scope: AgentScopeId | str = Field(alias="agentScope")
    name: str = Field(min_length=1, max_length=256)
    agent_id: str = Field(alias="agentId", min_length=1, max_length=128)
    dataset_id: str = Field(alias="datasetId", min_length=1, max_length=128)
    formula: RubricFormula
    trigger: TriggerMode
    eval_mode: Literal["realtime-span", "dataset-replay"] | None = Field(
        default=None,
        alias="evalMode",
    )
    cron_expression: str | None = Field(default=None, alias="cronExpression")
    status: TaskStatus
    processed_count: int = Field(alias="processedCount", ge=0)
    total_items: int = Field(alias="totalItems", ge=0)
    started_at: str = Field(alias="startedAt", min_length=1, max_length=64)
    completed_at: str | None = Field(default=None, alias="completedAt")
    score: float | None = None
    report_url: str | None = Field(default=None, alias="reportUrl")
    failure_reason: str | None = Field(default=None, alias="failureReason")
    daily_reports: list[DailyReport] | None = Field(default=None, alias="dailyReports")
    manual_run_id: str | None = Field(default=None, alias="manualRunId", max_length=128)
    #: 当前 run 的行主键（页面以 id 关联/删除单次评测；run_id 仅执行键）
    manual_run_pk: str | None = Field(default=None, alias="manualRunPk", max_length=64)

    @field_validator("cron_expression")
    @classmethod
    def _blank_cron(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @model_validator(mode="after")
    def _require_cron(self) -> Self:
        if self.trigger == "cron" and not self.cron_expression:
            msg = "自定义周期需要 cron 表达式"
            raise ValueError(msg)
        if self.cron_expression and len(self.cron_expression.split()) != _CRON_FIELD_COUNT:
            msg = "cron 需要 5 段"
            raise ValueError(msg)
        return self


def _now_label() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _payload(row: RuntimeConfigRecord) -> dict[str, Any]:
    if isinstance(row.payload, dict):
        return row.payload
    return {}


def _try_parse_agent_document(
    payload: dict[str, Any],
    *,
    record_key: str,
) -> AgentDocument | None:
    """列表/目录读路径容错：坏存量文档跳过，避免单条脏数据拖垮整页。

    写路径（``save_agent`` / API body）仍走严格 ``AgentDocument.model_validate``，
    不放宽「启用 prompt/skill 定制必须填写 agentScope」等约束。
    """
    try:
        return AgentDocument.model_validate(payload)
    except ValidationError as exc:
        logger.warning(
            "skip invalid benchmark_agent record_key=%s errors=%s",
            record_key,
            exc.errors(),
        )
        return None


def _public(document: _PageModel, row: RuntimeConfigRecord) -> dict[str, Any]:
    body = document.model_dump(by_alias=True, exclude_none=True)
    # 内部 agent 名业务不感知：对外响应一律屏蔽（设计 §12.2），仅在
    # 服务端快照/记录等数据面使用。
    body.pop("internalAgentName", None)
    body["schema_version"] = row.schema_version
    body["revision"] = row.revision
    body["scope_key"] = row.scope_key
    return body


def stop_sampling(dataset: DatasetDocument) -> DatasetDocument:
    """停止实时采样：状态改为就绪，已写入的样例保留。"""
    if dataset.status != "sampling":
        raise ConsoleError(409, "只有采样中的数据集可以停止")
    return dataset.model_copy(update={"status": "ready"})


def start_sampling(dataset: DatasetDocument) -> DatasetDocument:
    """开始实时采样：状态改为采样中（realtime 数据集专属）。

    已处于 ``sampling`` 时幂等返回原文档；非 realtime 来源或未就绪
    （parsing / collecting）的数据集不能开始采样。
    """
    if dataset.status == "sampling":
        return dataset
    if dataset.source != "realtime":
        raise ConsoleError(409, "只有实时采样数据集可以开始采样")
    if dataset.status != "ready":
        raise ConsoleError(409, "只有就绪的数据集可以开始采样")
    return dataset.model_copy(update={"status": "sampling"})


def start_collection(dataset: DatasetDocument) -> DatasetDocument:
    """开始语料收集：只改文档状态，不调用 spec-agent 或 agent 地址。"""
    if dataset.source == "realtime":
        raise ConsoleError(409, "实时采样数据集不进行语料收集")
    target = dataset.tool_name or dataset.skill_name
    agent_replay = dataset.replay_enabled is True and bool(dataset.agent_request_url)
    if not target and not agent_replay:
        raise ConsoleError(400, "语料收集需要工具名、Skill 名或 agent 请求地址")
    endpoint = (
        f"http://spec-agent-{target}.vs.amap.com/invoke"
        if target
        else str(dataset.agent_request_url)
    )
    collection = CollectionInfo(
        specAgentEndpoint=endpoint,
        collectedTraces=0,
        done=False,
        startedAt=_now_label(),
    )
    return dataset.model_copy(update={"status": "collecting", "collection": collection})


def stop_collection(dataset: DatasetDocument) -> DatasetDocument:
    """停止语料收集：状态改为就绪，已记录的收集信息保留。"""
    if dataset.status != "collecting":
        raise ConsoleError(409, "只有收集中的数据集可以停止")
    return dataset.model_copy(update={"status": "ready"})


def stop_task(task: EvalTaskDocument) -> EvalTaskDocument:
    """停止任务调度（文档置 ``stopped``）。

    已处于 ``stopped`` 时幂等返回原文档（重复点击停止不算冲突；
    前端列表状态可能滞后于首次停止，重复请求直接成功）。

    立即触发：仅 ``active`` 可停；``complete`` / ``failed`` 为终态不可停。
    定时任务（daily/weekly/cron）：调度启停只认 active/stopped；历史误把
    单次 run 终态写回文档时，仍允许从 ``complete`` / ``failed`` 停止以关停调度。
    """
    if task.status == "stopped":
        return task
    reason = task.failure_reason or "手动停止"
    if task.status == "active":
        return task.model_copy(update={"status": "stopped", "failure_reason": reason})
    if task.trigger != "immediate" and task.status in ("complete", "failed"):
        return task.model_copy(update={"status": "stopped", "failure_reason": reason})
    raise ConsoleError(409, "只有运行中的任务可以停止")


def rerun_task(task: EvalTaskDocument) -> EvalTaskDocument:
    """按原配置重新运行已停止的任务。"""
    if task.status != "stopped":
        raise ConsoleError(409, "只有已停止的任务可以重新运行")
    return task.model_copy(
        update={"status": "active", "failure_reason": None, "started_at": _now_label()},
    )


def retry_task(task: EvalTaskDocument) -> EvalTaskDocument:
    """执行失败后从头重试。"""
    if task.status != "failed":
        raise ConsoleError(409, "只有执行失败的任务可以重试")
    return task.model_copy(
        update={
            "status": "active",
            "processed_count": 0,
            "failure_reason": None,
            "started_at": _now_label(),
        },
    )


async def _ensure_scope(agent_scope: str | None) -> str | None:
    """校验评测对象：内置五项直接放行，其余须为已落库的自定义 scope。

    自定义 scope 与内置同权参与数据集 / 任务的隔离过滤；未注册值 400
    （提示先在评测对象栏创建）。
    """
    if agent_scope is None or agent_scope == "":
        return None
    if agent_scope in BUILTIN_AGENT_SCOPES:
        return agent_scope
    row = await RuntimeConfigRecord.filter(
        kind=KIND_AGENT_SCOPE,
        record_key=agent_scope_record_key(agent_scope),
        deleted_at__isnull=True,
    ).first()
    if row is None:
        raise ConsoleError(400, "未知的评测对象（请先在评测对象栏创建）")
    return agent_scope


async def list_scopes() -> list[dict[str, Any]]:
    """列出全部自定义评测对象（内置项由页面预置，不落库不返回）。"""
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_AGENT_SCOPE,
        deleted_at__isnull=True,
    ).order_by("-updated_at")
    return [_public(AgentScopeDocument.model_validate(_payload(row)), row) for row in rows]


async def save_scope(document: AgentScopeDocument) -> dict[str, Any]:
    """保存自定义评测对象。id 与内置项冲突时拒绝（避免双源同名）。"""
    if document.id in BUILTIN_AGENT_SCOPES:
        raise ConsoleError(409, f"评测对象 {document.id} 为内置项，不能重复创建")
    row = await upsert_runtime_config(
        kind=KIND_AGENT_SCOPE,
        record_key=agent_scope_record_key(document.id),
        payload=document.model_dump(by_alias=True, exclude_none=True),
        scope_key=SCOPE_GLOBAL,
    )
    return _public(document, row)


async def delete_scope(scope_id: str) -> dict[str, Any]:
    """软删自定义评测对象。内置项不可删；删除后其下数据集 / 任务仍在
    （按 agentScope 过滤时因 scope 不再合法而 400，需要先迁移数据）。"""
    if scope_id in BUILTIN_AGENT_SCOPES:
        raise ConsoleError(409, "内置评测对象不可删除")
    deleted = await retire_runtime_config(
        KIND_AGENT_SCOPE,
        agent_scope_record_key(scope_id),
    )
    if not deleted:
        raise ConsoleError(404, "评测对象不存在")
    return {"deleted": True}


async def _require_row(kind: str, record_key: str) -> RuntimeConfigRecord:
    row = await RuntimeConfigRecord.filter(
        kind=kind,
        record_key=record_key,
        deleted_at__isnull=True,
    ).first()
    if row is None:
        raise ConsoleError(404, "记录不存在")
    return row


async def save_dataset(document: DatasetDocument) -> dict[str, Any]:
    """保存数据集全文。realtime 缺省盖入 ``streamInstanceId=id``（含创建 ms）。"""
    to_save = document
    if document.source == "realtime" and not (document.stream_instance_id or "").strip():
        to_save = document.model_copy(update={"stream_instance_id": document.id.strip()})
    row = await upsert_runtime_config(
        kind=KIND_BENCHMARK_DATASET,
        record_key=benchmark_dataset_record_key(to_save.id),
        payload=to_save.model_dump(by_alias=True, exclude_none=True),
        scope_key=agent_scope_key(to_save.agent_scope),
    )
    return _public(to_save, row)


async def list_datasets(agent_scope: str | None = None) -> list[dict[str, Any]]:
    """列出数据集。传入评测对象时只返回该对象（自定义对象须已注册）。"""
    scope = await _ensure_scope(agent_scope)
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_DATASET,
        deleted_at__isnull=True,
    ).order_by("-updated_at")
    items: list[dict[str, Any]] = []
    for row in rows:
        document = DatasetDocument.model_validate(_payload(row))
        if scope is not None and document.agent_scope != scope:
            continue
        items.append(_public(document, row))
    await _fill_realtime_processed_counts(items)
    return items


async def _fill_realtime_processed_counts(items: list[dict[str, Any]]) -> None:
    """实时流数据集条数 = 该 ``streamInstanceId`` 下 Trace 处理次数。

    落库 ``evaluator_name`` 为 ``stem[@reg]@streamInstanceId``；按后缀
    ``@{stream}`` 过滤。无 stream 的存量回退按 project + evaluator/agent。

    Trace 为 0 时的 agent 调用次数回退由 services 层
    ``eval_count_enrichment`` 注入 ``run_counts`` 后调用
    :func:`fill_realtime_bound_agent_counts`（避免 domain 顶栏依赖
    ``BenchmarkService`` 循环 import）。
    """
    targets = [
        item
        for item in items
        if item.get("source") == "realtime" and str(item.get("projectId") or "").strip()
    ]
    if not targets:
        return
    legacy_targets = [
        item for item in targets if not str(item.get("streamInstanceId") or "").strip()
    ]
    legacy_project_ids = {str(item["projectId"]).strip() for item in legacy_targets}
    entry_names = await _evaluator_names_by_project(legacy_project_ids)
    names_by_project = {
        project_id: set(entry_names.get(project_id, set())) for project_id in legacy_project_ids
    }
    agents_by_project: dict[str, set[str]] = {
        project_id: set() for project_id in legacy_project_ids
    }
    for item in legacy_targets:
        project_id = str(item["projectId"]).strip()
        own = str(item.get("evaluatorName") or "").strip()
        if own:
            names_by_project[project_id].add(own)
        agent = str(item.get("agentName") or "").strip()
        if agent:
            agents_by_project[project_id].add(agent)

    legacy_counts: dict[str, int] = {}
    for project_id in legacy_project_ids:
        if names_by_project[project_id]:
            legacy_counts[project_id] = await TraceRecord.filter(
                project_id=project_id,
                evaluator_name__in=sorted(names_by_project[project_id]),
            ).count()
        elif agents_by_project[project_id]:
            legacy_counts[project_id] = await TraceRecord.filter(
                project_id=project_id,
                agent_name__in=sorted(agents_by_project[project_id]),
            ).count()
        else:
            legacy_counts[project_id] = await TraceRecord.filter(project_id=project_id).count()

    for item in targets:
        project_id = str(item["projectId"]).strip()
        stream = str(item.get("streamInstanceId") or "").strip()
        if not stream:
            item["itemsCount"] = legacy_counts.get(project_id, 0)
            continue
        item["itemsCount"] = (
            await TraceRecord.filter(project_id=project_id)
            .filter(Q(evaluator_name__endswith=f"@{stream}") | Q(evaluator_name=stream))
            .count()
        )


async def _agent_ids_by_dataset(dataset_ids: set[str]) -> dict[str, set[str]]:
    """未删评测任务：datasetId → 绑定的 agentId（registrationId）集合。"""
    mapping: dict[str, set[str]] = {dataset_id: set() for dataset_id in dataset_ids}
    if not dataset_ids:
        return mapping
    task_rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVAL_TASK,
        deleted_at__isnull=True,
    )
    for row in task_rows:
        document = EvalTaskDocument.model_validate(_payload(row))
        dataset_id = document.dataset_id.strip()
        if dataset_id not in mapping:
            continue
        agent_id = document.agent_id.strip()
        if agent_id:
            mapping[dataset_id].add(agent_id)
    return mapping


async def _registration_ids_by_stream(
    streams: set[str],
) -> dict[str, set[str]]:
    """从 evaluator 条目解析 ``stem@registrationId@streamInstanceId``。

    落库 ``evaluator_name`` / record_key 尾段 / payload 任一处 stream 命中，
    即把对应 registrationId 记入该流绑定集合（同一流可绑多个 agent）。
    """
    mapping: dict[str, set[str]] = {stream: set() for stream in streams if stream}
    if not mapping:
        return mapping

    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVALUATOR,
        deleted_at__isnull=True,
    )
    for row in rows:
        key_parts = split_evaluator_record_key(row.record_key)
        evaluator_name = key_parts[1]
        row_tail = key_parts[3]
        payload = _payload(row)
        reg_from_tail, stream_from_tail = parse_evaluator_key_tail(row_tail)
        payload_stream = str(
            payload.get("stream_instance_id") or payload.get("streamInstanceId") or "",
        ).strip()
        payload_reg = _payload_registration_id(payload)
        own_name = str(
            payload.get("evaluatorName") or payload.get("evaluator_name") or evaluator_name or "",
        ).strip()
        name_identity = parse_eval_identity_key(own_name)
        name_reg = name_identity[1]
        name_stream = name_identity[2]

        candidates: list[tuple[str, str]] = []
        if stream_from_tail and reg_from_tail:
            candidates.append((stream_from_tail, reg_from_tail))
        if payload_stream and payload_reg:
            candidates.append((payload_stream, payload_reg))
        if name_stream and name_reg:
            candidates.append((name_stream, name_reg))

        for stream, registration_id in candidates:
            bound = mapping.get(stream)
            if bound is not None and registration_id:
                bound.add(registration_id)
    return mapping


def _index_evaluation_counts(
    counts: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """run-counts 条目 → (registrationId→count, 归一化名→count)。"""
    by_registration: dict[str, int] = {}
    by_name: dict[str, int] = {}
    for entry in counts:
        evaluation_count = int(entry.get("evaluationCount") or 0)
        registration_id = entry.get("registrationId")
        if registration_id:
            parsed = parse_eval_identity_key(str(registration_id))[1]
            bare = parsed or str(registration_id).strip()
            if bare:
                by_registration[bare] = evaluation_count
            continue
        by_name[normalize_eval_count_key(str(entry.get("benchmarkName") or ""))] = evaluation_count
    return by_registration, by_name


async def _evaluation_count_for_agent(
    agent_id: str,
    *,
    by_registration: dict[str, int],
    by_name: dict[str, int],
) -> int:
    """单个 agent 的调用次数：优先 registrationId，再按名（example 除外）。"""
    parsed_id = parse_eval_identity_key(agent_id)[1]
    bare_id = parsed_id or agent_id
    if bare_id in by_registration:
        return by_registration[bare_id]
    agent = await find_agent_document_by_id(agent_id)
    if agent is None:
        return 0
    stem_keys = {normalize_eval_count_key(alias) for alias in eval_count_name_aliases(agent.name)}
    if "example" in stem_keys:
        return 0
    for key in stem_keys:
        if key in by_name:
            return by_name[key]
    return 0


async def _bound_agent_ids_for_realtime_items(
    items: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """datasetId → 绑定 registrationId（任务 + evaluator ``stem@reg@stream``）。"""
    dataset_ids = {
        str(item.get("id") or "").strip() for item in items if str(item.get("id") or "").strip()
    }
    mapping = await _agent_ids_by_dataset(dataset_ids)

    stream_to_dataset: dict[str, set[str]] = {}
    for item in items:
        dataset_id = str(item.get("id") or "").strip()
        if not dataset_id:
            continue
        stream = str(item.get("streamInstanceId") or "").strip() or dataset_id
        stream_to_dataset.setdefault(stream, set()).add(dataset_id)

    regs_by_stream = await _registration_ids_by_stream(set(stream_to_dataset))
    for stream, registration_ids in regs_by_stream.items():
        for dataset_id in stream_to_dataset.get(stream, set()):
            mapping.setdefault(dataset_id, set()).update(registration_ids)
    return mapping


async def fill_realtime_bound_agent_counts(
    items: list[dict[str, Any]],
    run_counts: list[dict[str, Any]],
) -> None:
    """Trace 为 0 时：按该流绑定的多个 benchmark-agent evaluationCount 求和。

    ``run_counts`` 由调用方（services 层）注入，避免本模块 import
    ``BenchmarkService``。
    """
    agent_ids_by_dataset = await _bound_agent_ids_for_realtime_items(items)
    all_agent_ids = {agent_id for ids in agent_ids_by_dataset.values() for agent_id in ids}
    if not all_agent_ids:
        return

    by_registration, by_name = _index_evaluation_counts(run_counts)
    agent_count_cache = {
        agent_id: await _evaluation_count_for_agent(
            agent_id,
            by_registration=by_registration,
            by_name=by_name,
        )
        for agent_id in all_agent_ids
    }

    for item in items:
        if int(item.get("itemsCount") or 0) != 0:
            continue
        dataset_id = str(item.get("id") or "").strip()
        bound = agent_ids_by_dataset.get(dataset_id) or set()
        if not bound:
            continue
        item["itemsCount"] = sum(agent_count_cache.get(agent_id, 0) for agent_id in bound)


async def _evaluator_names_by_project(project_ids: set[str]) -> dict[str, set[str]]:
    """project → 该项目下所有实时评测条目的 evaluator 名（未删除即算）。"""
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVALUATOR,
        deleted_at__isnull=True,
    )
    names: dict[str, set[str]] = {}
    for row in rows:
        project_id, evaluator_name, _, _ = split_evaluator_record_key(row.record_key)
        if project_id in project_ids and evaluator_name.strip():
            names.setdefault(project_id, set()).add(evaluator_name.strip())
    return names


async def get_dataset(dataset_id: str) -> dict[str, Any]:
    """读取一条数据集。"""
    row = await _require_row(KIND_BENCHMARK_DATASET, benchmark_dataset_record_key(dataset_id))
    return _public(DatasetDocument.model_validate(_payload(row)), row)


async def delete_dataset(dataset_id: str) -> dict[str, Any]:
    """软删数据集；realtime 数据集同步 retire 执行面 evaluator / T+1 报表项。

    页面删数据集若不收敛 ``kind=evaluator``，会留下孤儿执行面（streaming
    仍 Configs found）。先按文档收敛条目，再软删数据集行。
    """
    dataset = await find_dataset_document(dataset_id)
    if dataset is None:
        raise ConsoleError(404, "记录不存在")
    if dataset.source == "realtime":
        await _retire_dataset_evaluator_entry(dataset)
    deleted = await retire_runtime_config(
        KIND_BENCHMARK_DATASET,
        benchmark_dataset_record_key(dataset_id),
    )
    if not deleted:
        raise ConsoleError(404, "记录不存在")
    return {"deleted": True}


async def _mutate_dataset(
    dataset_id: str,
    mutate: Callable[[DatasetDocument], DatasetDocument],
) -> dict[str, Any]:
    row = await _require_row(KIND_BENCHMARK_DATASET, benchmark_dataset_record_key(dataset_id))
    document = mutate(DatasetDocument.model_validate(_payload(row)))
    return await save_dataset(document)


async def apply_stop_sampling(dataset_id: str) -> dict[str, Any]:
    """停止采样并保存；联动停用条目 + 停止使用该采样的所有任务。

    实时流评测要求「采样中 + 任务运行中」同时成立：采样停止后执行面
    （``kind=evaluator`` 条目）立即停用，同时把该数据集下所有运行中的
    任务置 ``stopped``——否则任务列表仍显示运行中、与执行面不一致。
    """
    result = await _mutate_dataset(dataset_id, stop_sampling)
    dataset = await find_dataset_document(dataset_id)
    if dataset is not None:
        await _stop_active_tasks_of_dataset(dataset.id)
        await _retire_dataset_evaluator_entry(dataset)
    return result


async def apply_start_sampling(dataset_id: str) -> dict[str, Any]:
    """开始实时采样并保存；仅当存在运行中任务时恢复条目。

    评测进行需「采样中 + 任务运行中」两个开关同时为真：无 active 任务
    时开始采样不恢复 evaluator 条目，等任务重新运行（rerun）时再恢复。
    """
    result = await _mutate_dataset(dataset_id, start_sampling)
    dataset = await find_dataset_document(dataset_id)
    if dataset is not None and await _active_task_ids_of_dataset(dataset.id):
        await _revive_dataset_evaluator_entry(dataset)
    return result


async def apply_update_sampling_ratio(
    dataset_id: str,
    sampling_ratio: float,
) -> dict[str, Any]:
    """修改实时数据集采样率，并同步已登记的 evaluator 条目（不改 stream 身份）。

    仅 ``source=realtime``；``sampling_ratio`` ∈ ``(0, 1]``。改率不换
    ``stream_instance_id`` / record_key，统计与报表命名空间保持连续。
    """
    if not (0.0 < sampling_ratio <= 1.0):
        raise ConsoleError(400, "samplingRatio 须在 (0, 1]")
    dataset = await find_dataset_document(dataset_id)
    if dataset is None:
        raise ConsoleError(404, "数据集不存在")
    if dataset.source != "realtime":
        raise ConsoleError(400, "仅实时数据集可修改采样率")
    updated = dataset.model_copy(update={"sampling_ratio": sampling_ratio})
    result = await save_dataset(updated)
    synced = 0
    for key in await _dataset_evaluator_record_keys(updated):
        row = await RuntimeConfigRecord.filter(record_key=key).first()
        if row is None or not isinstance(row.payload, dict):
            continue
        payload = dict(row.payload)
        payload["sampling"] = sampling_ratio
        await upsert_runtime_config(
            kind=KIND_EVALUATOR,
            record_key=key,
            payload=payload,
            scope_key=row.scope_key or SCOPE_GLOBAL,
        )
        synced += 1
    if synced:
        await refresh_eval_config_snapshot()
    logger.info(
        "realtime sampling updated: dataset=%s ratio=%s synced_entries=%d",
        dataset_id,
        sampling_ratio,
        synced,
    )
    return {**result, "syncedEvaluatorEntries": synced}


async def apply_start_collection(dataset_id: str) -> dict[str, Any]:
    """开始收集并保存。"""
    return await _mutate_dataset(dataset_id, start_collection)


async def apply_stop_collection(dataset_id: str) -> dict[str, Any]:
    """停止收集并保存。"""
    return await _mutate_dataset(dataset_id, stop_collection)


async def list_agent_documents() -> list[AgentDocument]:
    """列出未删除的 benchmark-agent 文档（统一 agent 视图 / 动态注册共用）。"""
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        deleted_at__isnull=True,
    ).order_by("-updated_at")
    items: list[AgentDocument] = []
    for row in rows:
        document = _try_parse_agent_document(_payload(row), record_key=row.record_key)
        if document is not None:
            items.append(document)
    return items


@dataclass(frozen=True, slots=True)
class _RegistrationIdParts:
    """带内容定制的注册 id 拼装入参（FUNC-PARAMS-001 收敛）。"""

    name: str
    business: str
    variant_key: str
    version: int
    agent_scope: str


def _registration_id(parts: _RegistrationIdParts) -> str:
    """带内容定制的注册 id：含评测对象维。

    ``{slug(name)}--{slug(scope)}--{slug(business)}--{slug(variant)}--v{n}``
    """
    return (
        f"{slugify(parts.name)}--{slugify(parts.agent_scope)}--"
        f"{slugify(parts.business)}--{slugify(parts.variant_key)}--v{parts.version}"
    )[:128]


def _instance_group_key(document: AgentDocument) -> tuple[str, str, str, str]:
    """实例分组键：(name, business, variant, agentScope) 一组多版本，同组共享内部名。"""
    return (
        document.name.strip(),
        (document.business or "").strip(),
        (document.variant_key or "").strip(),
        (document.agent_scope or "").strip(),
    )


def _internal_agent_name(name: str, sequence: int) -> str:
    """内部 agent 名：``agx_{stem}_{10 位序号}``（设计 §12.2）。

    stem 取统一 agent 茎并规范为 ``[a-z0-9_]``（连字符一律下划线，
    避免归一化碰撞）；序号全局单调、永不复用。业务不可见、不可指定。
    """
    from benchmark.runs.common.agent_catalog import (  # noqa: PLC0415
        canonical_agent_name,
    )

    stem = (
        re.sub(r"[^a-z0-9_]+", "_", canonical_agent_name(name, name).lower()).strip("_")[:40]
        or "agent"
    )
    return f"agx_{stem}_{sequence:010d}"


async def _assign_internal_agent_name(
    document: AgentDocument,
    existing: AgentDocument | None,
) -> AgentDocument:
    """为注册分配/继承内部 agent 名（服务端权威，客户端传值忽略）。

    继承顺序：同 id 存量 → 同实例组任一版本 → 新领全局序号。同组多
    版本共享一个内部名；新组才领新号，序号永不复用（软删进墓碑）。
    """
    if existing is not None and existing.internal_agent_name:
        internal = existing.internal_agent_name
    else:
        group = _instance_group_key(document)
        internal = next(
            (
                item.internal_agent_name
                for item in await list_agent_documents()
                if _instance_group_key(item) == group and item.internal_agent_name
            ),
            None,
        )
        if internal is None:
            from benchmark.domain.runtime_config_store import (  # noqa: PLC0415
                next_instance_sequence,
            )

            internal = _internal_agent_name(document.name, await next_instance_sequence())
    if document.internal_agent_name == internal:
        return document
    return document.model_copy(update={"internal_agent_name": internal})


async def _check_instance_quotas(
    document: AgentDocument,
    existing: AgentDocument | None,
) -> None:
    """多实例配额（设计 §12.3 INV-4/INV-5），仅对新组/新版本生效。

    配额从定制 RemoteConfig 配置读（business 白名单 + 三项上限）；上限的
    意义是防无限注册，不是审批——存量更新（同 id 保存）永不拦截。
    """
    from benchmark.domain.customization_config import fetch_customization_config  # noqa: PLC0415

    config = fetch_customization_config()
    business = (document.business or "").strip()
    if (
        document.customization is not None
        and config.business_allowlist
        and business not in config.business_allowlist
    ):
        raise ConsoleError(
            400,
            f"business_not_allowed: business {business or '-'} 不在白名单内",
        )
    if existing is not None:
        return
    group = _instance_group_key(document)
    groups: dict[tuple[str, str, str, str], int] = {}
    for item in await list_agent_documents():
        key = _instance_group_key(item)
        groups[key] = groups.get(key, 0) + 1
    if group in groups:
        if groups[group] >= config.max_versions_per_instance:
            raise ConsoleError(
                403,
                "instance_quota_exceeded: 实例 "
                f"{group[0]}/{group[1] or '-'}/{group[2] or '-'}/{group[3] or '-'} "
                f"活跃版本已达上限 {config.max_versions_per_instance}，请先下线旧版本",
            )
        return
    per_component = {key for key in groups if key[0] == group[0]}
    if len(per_component) >= config.max_instances_per_component:
        raise ConsoleError(
            403,
            "instance_quota_exceeded: agent "
            f"{group[0]} 活跃实例已达上限 {config.max_instances_per_component}",
        )
    if business:
        per_business = {key for key in groups if key[1] == business}
        if len(per_business) >= config.max_instances_per_business:
            raise ConsoleError(
                403,
                "instance_quota_exceeded: business "
                f"{business} 活跃实例已达上限 {config.max_instances_per_business}",
            )


def _check_attribution_toggle(document: AgentDocument) -> None:
    """关闭 attribution 默认放行（归因为可选流程）；仅 policy 显式置 False
    的 agent 才拒绝（403 attribution_toggle_not_allowed）。"""
    if document.attribution_enabled:
        return
    from benchmark.runs.common.agent_capabilities import (  # noqa: PLC0415
        customization_policy_for_name,
    )

    policy = customization_policy_for_name(document.name)
    if not policy.allow_attribution_toggle:
        raise ConsoleError(
            403,
            f"attribution_toggle_not_allowed: agent {document.name} 不允许业务关闭归因",
        )


def check_customization_policy(
    document: AgentDocument,
    *,
    status_code: int,
    code: str,
) -> None:
    """注册中每个 (stage, kind) 必须在当前 policy 中 allowed（保存 403 / 触发 409）。"""
    if document.customization is None:
        return
    from benchmark.runs.common.agent_capabilities import (  # noqa: PLC0415
        customization_policy_for_name,
    )

    policy = customization_policy_for_name(document.name)
    for stage_name, ref in (
        ("judge", document.customization.judge),
        ("attribution", document.customization.attribution),
    ):
        if ref is None:
            continue
        stage_policy = policy.judge if stage_name == "judge" else policy.attribution
        if ref.prompt is not None and not stage_policy.allow_prompt:
            raise ConsoleError(
                status_code, f"{code}: {document.name} 的 {stage_name} prompt 已不允许定制"
            )
        if ref.skill is not None and not stage_policy.allow_skill:
            raise ConsoleError(
                status_code, f"{code}: {document.name} 的 {stage_name} skill 已不允许定制"
            )


async def _verify_customization_refs(document: AgentDocument) -> None:
    """读 Git manifest 复核 ref（commit 存在、sha256 一致）；失败显式报错不落库。"""
    if document.customization is None:
        return
    from benchmark.customization.bundle_store import get_bundle_store  # noqa: PLC0415
    from benchmark.customization.errors import CustomizationError  # noqa: PLC0415

    store = get_bundle_store()
    for ref in (document.customization.judge, document.customization.attribution):
        if ref is None:
            continue
        try:
            await store.ensure(ref.model_dump())
        except CustomizationError as exc:
            raise ConsoleError(exc.status_code, f"{exc.code}: {exc.detail}") from exc


async def _prepare_customized_registration(
    document: AgentDocument,
    existing: AgentDocument | None,
) -> AgentDocument:
    """带内容定制的注册保存前处理：开关校验 → policy → Git 复核 → 版本/id 分配。

    不可变：已存在的 id 再次保存时 customization 必须与存量一致（省略 =
    不变由调用方恢复）；内容变化必须走新注册版本。
    """
    from benchmark.domain.customization_config import fetch_customization_config  # noqa: PLC0415

    if not fetch_customization_config().enabled:
        raise ConsoleError(503, "customization_disabled: 业务定制未启用，不能保存带定制的注册")
    _check_attribution_toggle(document)
    check_customization_policy(document, status_code=403, code="customization_not_allowed")
    await _verify_customization_refs(document)

    business = document.business or ""
    variant_key = document.variant_key or ""
    agent_scope = (document.agent_scope or "").strip()
    if existing is not None:
        if existing.customization != document.customization:
            raise ConsoleError(
                409,
                "customization_immutable: 已存在的注册不能修改 prompt/skill 内容，请新建注册版本",
            )
        return document

    siblings = [
        item
        for item in await list_agent_documents()
        if item.name == document.name
        and (item.business or "") == business
        and (item.variant_key or "") == variant_key
        and (item.agent_scope or "").strip() == agent_scope
    ]
    for sibling in siblings:
        if sibling.customization == document.customization:
            # 同内容重复保存：幂等回到既有版本（id/version 以存量为准）
            return document.model_copy(update={"id": sibling.id, "version": sibling.version})
    version = max((item.version for item in siblings), default=0) + 1
    return document.model_copy(
        update={
            "version": version,
            "id": _registration_id(
                _RegistrationIdParts(
                    name=document.name,
                    business=business,
                    variant_key=variant_key,
                    version=version,
                    agent_scope=agent_scope,
                )
            ),
        }
    )


async def _clear_other_defaults(document: AgentDocument) -> None:
    """同 name + 同评测对象至多一个 isDefault（底版 agentScope 空视为同一维）。"""
    doc_scope = (document.agent_scope or "").strip()
    for other in await list_agent_documents():
        if other.name != document.name or other.id == document.id or not other.is_default:
            continue
        if (other.agent_scope or "").strip() != doc_scope:
            continue
        cleared = other.model_copy(update={"is_default": False})
        # 与 save_agent persist_scope 同口径：有 agent_scope 即写 agent:{scope}
        other_scope_key = agent_scope_key(other.agent_scope) if other.agent_scope else SCOPE_GLOBAL
        await upsert_runtime_config(
            kind=KIND_BENCHMARK_AGENT,
            record_key=benchmark_agent_record_key(other.id),
            payload=cleared.model_dump(by_alias=True, exclude_none=True),
            scope_key=other_scope_key,
        )


async def _reuse_plain_registration(
    document: AgentDocument,
    existing: AgentDocument | None,
) -> AgentDocument:
    """同名且无 customization 的注册：幂等复用已有文档 id。

    WebUI 每次「注册」会生成新的 ``ba-${Date.now()}``；若同名已有无定制
    文档，改写到既有 id，避免实时流 ``evaluator_name`` / T+1 报表命名空间
    随重复注册不断漂移。带 customization 的版本链仍走
    :func:`_prepare_customized_registration`（``--vN`` 语义不变）。

    多条无定制同名并存时：优先 ``isDefault``，其次按 id 字典序取稳。
    复用时保留原 ``registeredAt`` / ``version``，日报历史锚点不因重提
    交描述而重置。
    """
    if document.customization is not None:
        return document
    if existing is not None and existing.customization is None:
        return document
    target = document.name.strip()
    if not target:
        return document
    doc_scope = (document.agent_scope or "").strip()
    siblings = [
        item
        for item in await list_agent_documents()
        if item.name.strip() == target
        and item.customization is None
        and (item.agent_scope or "").strip() == doc_scope
    ]
    if not siblings:
        return document
    siblings.sort(key=lambda item: (0 if item.is_default else 1, item.id))
    chosen = siblings[0]
    if chosen.id == document.id:
        return document
    logger.info(
        "plain_registration.reuse requested_id=%s reused_id=%s name=%s",
        document.id,
        chosen.id,
        target,
    )
    return document.model_copy(
        update={
            "id": chosen.id,
            "version": chosen.version,
            "registered_at": chosen.registered_at,
        },
    )


async def _load_agent_by_id(agent_id: str) -> AgentDocument | None:
    """按注册文档 id 读活跃 agent；不存在返回 ``None``。"""
    row = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        record_key=benchmark_agent_record_key(agent_id),
        deleted_at__isnull=True,
    ).first()
    if row is None:
        return None
    return AgentDocument.model_validate(_payload(row))


async def _bind_agent_scope(document: AgentDocument, *, required_message: str) -> AgentDocument:
    """校验并写入评测对象归属（L1 白名单）。"""
    scope = await _ensure_scope(document.agent_scope)
    if scope is None:
        raise ConsoleError(400, required_message)
    if not get_ownership_policy().allows(scope, document.name):
        raise ConsoleError(
            403,
            f"agent_not_allowed_in_scope: {document.name} 不允许归属评测对象 {scope}",
        )
    return document.model_copy(update={"agent_scope": scope})


async def _prepare_plain_or_toggle_save(
    document: AgentDocument,
    existing: AgentDocument | None,
) -> tuple[AgentDocument, AgentDocument | None]:
    """无定制正文时的保存：保留已有定制维度，或绑定 scope 后幂等复用底版。"""
    _check_attribution_toggle(document)
    if existing is not None and existing.customization is not None:
        if existing.attribution_enabled != document.attribution_enabled:
            logger.info(
                "customization.attribution_toggled id=%s old=%s new=%s",
                document.id,
                existing.attribution_enabled,
                document.attribution_enabled,
            )
        document = document.model_copy(
            update={
                "customization": existing.customization,
                "business": existing.business,
                "variant_key": existing.variant_key,
                "version": existing.version,
                "agent_scope": existing.agent_scope,
            }
        )
        return document, existing
    document = await _bind_agent_scope(document, required_message="agentScope is required")
    document = await _reuse_plain_registration(document, existing)
    if existing is None or existing.id != document.id:
        existing = await _load_agent_by_id(document.id)
    return document, existing


async def save_agent(document: AgentDocument) -> dict[str, Any]:
    """保存评测代理（能力绑定由服务端管理，写前剥离历史遗留执行链字段）。"""
    document = strip_agent_execution_fields(document)
    existing = await _load_agent_by_id(document.id)
    if document.customization is not None:
        document = await _bind_agent_scope(
            document,
            required_message="启用 prompt/skill 定制必须填写 agentScope",
        )
        document = await _prepare_customized_registration(document, existing)
        # 定制路径可能改写 id（同内容幂等 / 新版本 --vN）；按最终 id 重载存量。
        if existing is None or existing.id != document.id:
            existing = await _load_agent_by_id(document.id)
    else:
        document, existing = await _prepare_plain_or_toggle_save(document, existing)
    # 多实例隔离（设计 §12）：配额校验后分配/继承内部 agent 名
    await _check_instance_quotas(document, existing)
    document = await _assign_internal_agent_name(document, existing)
    persist_scope = agent_scope_key(document.agent_scope) if document.agent_scope else SCOPE_GLOBAL
    row = await upsert_runtime_config(
        kind=KIND_BENCHMARK_AGENT,
        record_key=benchmark_agent_record_key(document.id),
        payload=document.model_dump(by_alias=True, exclude_none=True),
        scope_key=persist_scope,
    )
    if document.is_default:
        await _clear_other_defaults(document)
    if document.status == "offline":
        # 注册下线 = 禁止任何实时流数据再进入该评测器：与 delete_agent 同口径，
        # 级联停用绑定该 registration 的 evaluator 条目并立即刷新快照，
        # 使 Kafka / sidecar 消费侧在下一个对账周期内停止消费（Kafka 逐条拦截）。
        # 重新 online 不自动复活（恢复评测走 rerun / 重新激活的显式路径）。
        await _cascade_retire_evaluator_for_registration(document.id)
    if document.customization is not None:
        logger.info(
            "customization.registration_saved id=%s version=%d attribution_enabled=%s",
            document.id,
            document.version,
            document.attribution_enabled,
        )
    return _public(document, row)


async def resolve_agent_registration(
    name: str,
    registration_id: str | None = None,
    *,
    agent_scope: str | None = None,
) -> AgentDocument | None:
    """触发时解析本次 run 使用的注册（版本级选择，设计 §4.4）。

    显式 id 精确加载（不存在 404、离线 409）；若带 ``agent_scope`` 且命中
    定制注册，归属不一致 → 409。未指定 id 时：无 scope → **排除定制注册**
    （仅全局底版参与）；有 scope → L1+L2 可见集。其后：0 条 → None；
    1 条 → 用它；多条 → 唯一 isDefault；存量全无定制 → 最新更新并
    warning；否则 400 ambiguous。
    """
    target = name.strip()
    scope = (agent_scope or "").strip() or None
    if registration_id:
        row = await RuntimeConfigRecord.filter(
            kind=KIND_BENCHMARK_AGENT,
            record_key=benchmark_agent_record_key(registration_id),
            deleted_at__isnull=True,
        ).first()
        if row is None:
            raise ConsoleError(404, f"agent registration not found: {registration_id}")
        document = AgentDocument.model_validate(_payload(row))
        if document.status != "online":
            raise ConsoleError(409, f"agent registration offline: {registration_id}")
        if document.name.strip() != target:
            raise ConsoleError(
                400,
                f"agent registration {registration_id} belongs to {document.name}, not {target}",
            )
        if (
            scope is not None
            and document.customization is not None
            and (document.agent_scope or "").strip() != scope
        ):
            raise ConsoleError(
                409,
                f"customization_scope_mismatch: registration {registration_id} "
                f"bound to {document.agent_scope or '-'}, not {scope}",
            )
        return document
    documents = [
        item
        for item in await list_agent_documents()
        if item.name.strip() == target and item.status == "online"
    ]
    if scope is None:
        documents = [item for item in documents if item.customization is None]
    else:
        policy = get_ownership_policy()
        documents = [
            item
            for item in documents
            if is_registration_visible(
                RegistrationVisibility(
                    agent_scope=scope,
                    agent_name=item.name,
                    customization=item.customization,
                    registration_scope=item.agent_scope,
                    policy=policy,
                )
            )
        ]
    if not documents:
        return None
    if len(documents) == 1:
        return documents[0]
    defaults = [item for item in documents if item.is_default]
    if defaults:
        if len(defaults) > 1:
            logger.warning(
                "agent registration %s has %d defaults, using latest updated",
                target,
                len(defaults),
            )
        return defaults[0]
    if all(item.customization is None for item in documents):
        logger.warning(
            "agent registration %s has %d legacy duplicates, using latest updated",
            target,
            len(documents),
        )
        return documents[0]
    candidates = ", ".join(
        f"{item.id}(business={item.business or '-'}, variant={item.variant_key or '-'}, "
        f"scope={item.agent_scope or '-'}, v{item.version})"
        for item in documents
    )
    raise ConsoleError(
        400,
        f"ambiguous_agent_registration: {target} 有多个注册且无默认版本，"
        f"请显式指定 agent_registration_id（候选：{candidates}）",
    )


def registration_snapshot(document: AgentDocument) -> dict[str, Any]:
    """注册 → run 快照（config_json.agent_registration，snake_case 内部形态）。"""
    customization: dict[str, Any] | None = None
    if document.customization is not None:
        customization = {
            "judge": (
                document.customization.judge.model_dump() if document.customization.judge else None
            ),
            "attribution": (
                document.customization.attribution.model_dump()
                if document.customization.attribution
                else None
            ),
        }
    return {
        "id": document.id,
        "name": document.name,
        "business": document.business,
        "variant_key": document.variant_key,
        "version": document.version,
        "is_default": document.is_default,
        "attribution_enabled": document.attribution_enabled,
        "notification_dispatch_enabled": document.notification_dispatch_enabled,
        "notification_project_id": document.notification_project_id,
        "notification_assignee": document.notification_assignee,
        "notification_participants": list(document.notification_participants),
        "owner": document.owner,
        "owners": list(document.owners),
        "customization": customization,
        # 多实例数据面归属键（设计 §12.2）：记录/报告按此分组，业务不感知
        "internal_agent_name": document.internal_agent_name,
    }


async def list_agents(agent_scope: str | None = None) -> list[dict[str, Any]]:
    """列出评测代理。

     console 列表必须带评测对象：按 L1+L2 过滤。缺 ``agent_scope`` 时 400
    （全量总览见 :func:`list_all_agent_registrations`）。
    """
    if agent_scope is None or not str(agent_scope).strip():
        raise ConsoleError(400, "agentScope is required")
    scope = await _ensure_scope(agent_scope)
    if scope is None:
        raise ConsoleError(400, "agentScope is required")
    policy = get_ownership_policy()
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        deleted_at__isnull=True,
    ).order_by("-updated_at")
    items: list[dict[str, Any]] = []
    for row in rows:
        document = _try_parse_agent_document(_payload(row), record_key=row.record_key)
        if document is None:
            continue
        if not is_registration_visible(
            RegistrationVisibility(
                agent_scope=scope,
                agent_name=document.name,
                customization=document.customization,
                registration_scope=document.agent_scope,
                policy=policy,
            )
        ):
            continue
        items.append(_public(document, row))
    return items


async def list_all_agent_registrations() -> list[dict[str, Any]]:
    """全量注册总览：所有评测对象下全部带版本注册（含所属 agentScope）。"""
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        deleted_at__isnull=True,
    ).order_by("-updated_at")
    items: list[dict[str, Any]] = []
    for row in rows:
        document = _try_parse_agent_document(_payload(row), record_key=row.record_key)
        if document is None:
            continue
        items.append(_public(document, row))
    return items


async def get_agent(agent_id: str) -> dict[str, Any]:
    """读取一个评测代理。"""
    row = await _require_row(KIND_BENCHMARK_AGENT, benchmark_agent_record_key(agent_id))
    return _public(AgentDocument.model_validate(_payload(row)), row)


#: judge_json 中 content_version 的候选 JSON 路径（生产 lineage 写 extra.prompt_customization）。
_JUDGE_CONTENT_VERSION_PATHS: tuple[tuple[str, ...], ...] = (
    ("extra", "prompt_customization", "content_version"),
    ("prompt_customization", "content_version"),
    ("prompt", "content_version"),
    ("content_version",),
)
#: attribution_detail_json 中 content_version 的候选路径。
_ATTR_CONTENT_VERSION_PATHS: tuple[tuple[str, ...], ...] = (
    ("prompt_customization", "content_version"),
    ("extra", "prompt_customization", "content_version"),
    ("prompt", "content_version"),
    ("attribution", "prompt_customization", "content_version"),
    ("content_version",),
)


def evaluator_name_candidates(agent_name: str) -> frozenset[str]:
    """注册名 → 可能对应的 ``trace_pipeline_records.evaluator_name``。

    流式落库用 RemoteConfig ``evaluatorName``（如 ``example``），注册页展示名
    可能是执行侧名（``example_benchmark``）或带 ``_agent`` / ``-agent``
    后缀；两侧都纳入候选以免漏计。
    """
    from benchmark.runs.common.agent_catalog import (  # noqa: PLC0415
        canonical_agent_name,
    )

    name = agent_name.strip()
    if not name:
        return frozenset()
    names = {name, canonical_agent_name(name)}
    for suffix in ("_agent", "-agent"):
        if name.endswith(suffix) and len(name) > len(suffix):
            names.add(name[: -len(suffix)])
    return frozenset(n for n in names if n)


def _nested_str(bucket: Any, path: tuple[str, ...]) -> str | None:  # noqa: ANN401
    """沿 path 取嵌套 dict 叶子并归一成非空字符串。"""
    cur: Any = bucket
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if cur is None:
        return None
    text = str(cur).strip()
    return text or None


def content_version_from_bucket(
    bucket: Any,  # noqa: ANN401
    paths: tuple[tuple[str, ...], ...],
) -> str | None:
    """从 JSON bucket 按候选路径取 content_version（先命中先返回）。"""
    if not isinstance(bucket, dict):
        return None
    for path in paths:
        found = _nested_str(bucket, path)
        if found is not None:
            return found
    return None


def registration_content_versions(item: dict[str, Any]) -> tuple[str | None, str | None]:
    """从注册文档公开形态取 (judge_cv, attribution_cv)。"""
    customization = item.get("customization")
    if not isinstance(customization, dict):
        return None, None
    judge = customization.get("judge")
    attribution = customization.get("attribution")
    judge_cv = None
    attr_cv = None
    if isinstance(judge, dict):
        raw = judge.get("contentVersion", judge.get("content_version"))
        if raw is not None and str(raw).strip():
            judge_cv = str(raw).strip()
    if isinstance(attribution, dict):
        raw = attribution.get("contentVersion", attribution.get("content_version"))
        if raw is not None and str(raw).strip():
            attr_cv = str(raw).strip()
    return judge_cv, attr_cv


class AgentRegistrationMatch(NamedTuple):
    """``row_matches_agent_registration`` 的匹配探针（收敛 FUNC-PARAMS）。"""

    evaluator_name: str
    candidates: frozenset[str]
    judge_cv: str | None = None
    attr_cv: str | None = None
    want_judge_cv: str | None = None
    want_attr_cv: str | None = None


def row_matches_agent_registration(match: AgentRegistrationMatch) -> bool:
    """单条 trace 是否归属某注册版本。

    优先按 judge prompt 的 content_version 对齐（好 case 也会落 judge）；
    仅有 attribution 定制时再按 attribution 版本；无定制的注册只统计
    未打版本戳的行，避免把其他版本的量算进来。
    """
    if match.evaluator_name not in match.candidates:
        return False
    if match.want_judge_cv is not None:
        return match.judge_cv == match.want_judge_cv
    if match.want_attr_cv is not None:
        return match.attr_cv == match.want_attr_cv
    return match.judge_cv is None and match.attr_cv is None


async def fill_agent_evaluation_counts(  # noqa: PLR0912 — batch+streaming 分链计数分支
    items: list[dict[str, Any]],
    run_counts: list[dict[str, Any]],
) -> None:
    """注册页调用次数：复用 ``run-counts`` 分链口径（batch + streaming）。

    与数据集 ``_fill_realtime_processed_counts`` 同思路：查询时派生只读值，
    不回写 AgentDocument。``run_counts`` 由 services 层注入，避免本模块
    import ``BenchmarkService``。页面快照仍会再经 webui ``mergeRunCounts``
    覆盖；此处保证 agents API 单读时分量也不丢。
    """
    if not items:
        return

    by_registration: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for entry in run_counts:
        registration_id = entry.get("registrationId")
        if registration_id:
            # 裸注册文档 id；若误传 stem@reg[@stream] 则取 reg 段（与 WebUI 对齐）
            parsed = parse_eval_identity_key(str(registration_id))[1]
            bare = parsed or str(registration_id).strip()
            if bare:
                by_registration[bare] = entry
            continue
        by_name[normalize_eval_count_key(str(entry.get("benchmarkName") or ""))] = entry
    for item in items:
        agent_id = str(item.get("id") or "").strip()
        matched: dict[str, Any] | None = None
        if agent_id:
            parsed_id = parse_eval_identity_key(agent_id)[1]
            matched = by_registration.get(parsed_id or agent_id)
            # example 双链只认 registrationId；共享茎仅非 example 卡片
            if matched is None:
                stem_keys = {
                    normalize_eval_count_key(alias)
                    for alias in eval_count_name_aliases(str(item.get("name") or ""))
                }
                if "example" not in stem_keys:
                    for key in stem_keys:
                        matched = by_name.get(key)
                        if matched is not None:
                            break
        else:
            # 无注册 id：example 双链禁止按名共享（否则同名多版本串同一个数）
            stem_keys = {
                normalize_eval_count_key(alias)
                for alias in eval_count_name_aliases(str(item.get("name") or ""))
            }
            if "example" not in stem_keys:
                for key in stem_keys:
                    matched = by_name.get(key)
                    if matched is not None:
                        break
        if matched is None:
            item["evaluationCount"] = 0
            item["batchCount"] = 0
            item["streamingCount"] = 0
            continue
        item["evaluationCount"] = int(matched.get("evaluationCount") or 0)
        item["batchCount"] = int(matched.get("batchCount") or 0)
        item["streamingCount"] = int(matched.get("streamingCount") or 0)


async def find_agent_document_by_name(name: str) -> AgentDocument | None:
    """按业务自定义名查找评测代理文档（动态 benchmark 组件的触发依据）。

    供触发类入口（manual-runs 等）在静态注册未命中时反查页面注册文档。
    名称匹配按 trim 后全等（注册与触发两侧同名归一，不做模糊/大小写容错）。
    未命中返回 ``None``，由调用方决定 404。
    """
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        deleted_at__isnull=True,
    )
    target = name.strip()
    for row in rows:
        payload = _payload(row)
        document_name = str(payload.get("name", "")).strip()
        if document_name == target:
            return AgentDocument.model_validate(payload)
    return None


async def find_agent_document_by_id(agent_id: str) -> AgentDocument | None:
    """按注册文档 id 查找评测代理（定时任务按任务文档的 agentId 回溯）。

    任务文档只存 agentId；触发时需要其中的 ``name``（执行侧 benchmark 名）
    与 rubric 声明。已删/未命中返回 ``None``。
    """
    row = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        record_key=benchmark_agent_record_key(agent_id.strip()),
        deleted_at__isnull=True,
    ).first()
    if row is None:
        return None
    return AgentDocument.model_validate(_payload(row))


async def find_dataset_document(dataset_id: str) -> DatasetDocument | None:
    """按 id 查找数据集文档（未命中/已删返回 None）。"""
    row = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_DATASET,
        record_key=benchmark_dataset_record_key(dataset_id),
        deleted_at__isnull=True,
    ).first()
    if row is None:
        return None
    return DatasetDocument.model_validate(_payload(row))


async def find_dataset_document_any(dataset_id: str) -> DatasetDocument | None:
    """按 id 查找数据集（含已软删行）。

    删任务联动收敛执行面时：若页面先删了数据集再删任务，有效行已不存在，
    但仍须按原文档的 projectId/agentName 去 retire ``kind=evaluator``，
    否则会留下孤儿执行面（WebUI 已空、streaming 仍 Configs found）。
    """
    row = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_DATASET,
        record_key=benchmark_dataset_record_key(dataset_id.strip()),
    ).first()
    if row is None:
        return None
    return DatasetDocument.model_validate(_payload(row))


async def _dataset_evaluator_record_keys(dataset: DatasetDocument) -> list[str]:
    """定位数据集相关的 evaluator 条目 record_key（含多 prompt 注册槽）。

    优先按 ``stream_instance_id``（payload 或 record_key ``__si__`` 后缀）
    精确匹配，避免同 agent 下其它实时流被误停。无 stream 的存量数据回退
    project+agentName 扫描。含软删行以便 revive。
    """
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        parse_evaluator_key_tail,
    )

    project_id = (dataset.project_id or "").strip()
    if not project_id:
        logger.warning(
            "realtime cascade skipped: dataset=%s has no projectId",
            dataset.id,
        )
        return []
    stream_id = (dataset.stream_instance_id or "").strip() or dataset.id.strip()
    agent_name = (dataset.agent_name or "").strip()
    keys: list[str] = []
    seen: set[str] = set()
    rows = await RuntimeConfigRecord.filter(kind=KIND_EVALUATOR)
    for row in rows:
        row_project, _row_evaluator, row_agent, row_tail = split_evaluator_record_key(
            row.record_key
        )
        if row_project != project_id:
            continue
        _reg, row_stream = parse_evaluator_key_tail(row_tail)
        payload = _payload(row)
        payload_stream = str(
            payload.get("stream_instance_id") or payload.get("streamInstanceId") or "",
        ).strip()
        matched_stream = (row_stream and row_stream == stream_id) or (
            payload_stream and payload_stream == stream_id
        )
        if stream_id and matched_stream:
            pass
        elif not row_stream and not payload_stream:
            # 存量无 stream 键：回退同 agent 扫描（兼容升级前条目）
            if row_agent.strip() != agent_name:
                continue
        else:
            continue
        if row.record_key not in seen:
            seen.add(row.record_key)
            keys.append(row.record_key)
    return keys


def _payload_registration_id(payload: dict[str, Any]) -> str:
    """从 evaluator payload 取注册 id（兼容 snake / camel）。"""
    return str(
        payload.get("agent_registration_id") or payload.get("agentRegistrationId") or ""
    ).strip()


async def _retire_evaluator_entries_by_registration(registration_id: str) -> int:
    """软删绑定该 registrationId 的全部 evaluator 条目（含各 stream 槽）。"""
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        parse_evaluator_key_tail,
    )

    reg = registration_id.strip()
    if not reg:
        return 0
    retired = 0
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVALUATOR,
        deleted_at__isnull=True,
    )
    for row in rows:
        _project, _evaluator, _agent, row_tail = split_evaluator_record_key(row.record_key)
        row_reg, _row_stream = parse_evaluator_key_tail(row_tail)
        if row_reg != reg and _payload_registration_id(_payload(row)) != reg:
            continue
        if await retire_runtime_config(KIND_EVALUATOR, row.record_key):
            retired += 1
    return retired


async def _datasets_referenced_by_agent_tasks(
    agent_id: str,
    *,
    include_deleted_tasks: bool = False,
) -> list[DatasetDocument]:
    """引用该 agentId 的 realtime 数据集（去重）。

    ``include_deleted_tasks=True``：页面先删任务再删 agent 时，仍能凭
    软删任务上的 datasetId 定位并收敛默认槽（无 registration 的条目）。
    """
    target = agent_id.strip()
    datasets: list[DatasetDocument] = []
    seen: set[str] = set()
    query = RuntimeConfigRecord.filter(kind=KIND_EVAL_TASK)
    if not include_deleted_tasks:
        query = query.filter(deleted_at__isnull=True)
    rows = await query
    for row in rows:
        document = EvalTaskDocument.model_validate(_payload(row))
        if document.agent_id.strip() != target:
            continue
        dataset = await find_dataset_document_any(document.dataset_id)
        if dataset is None or dataset.source != "realtime":
            continue
        if dataset.id in seen:
            continue
        seen.add(dataset.id)
        datasets.append(dataset)
    return datasets


async def _dataset_has_other_agent_tasks(
    dataset_id: str,
    *,
    exclude_agent_id: str,
) -> bool:
    """同数据集是否还有其它 agent 的未删任务（删 agent 时保护共享默认槽）。"""
    exclude = exclude_agent_id.strip()
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVAL_TASK,
        deleted_at__isnull=True,
    )
    for row in rows:
        document = EvalTaskDocument.model_validate(_payload(row))
        if document.dataset_id != dataset_id:
            continue
        if document.agent_id.strip() == exclude:
            continue
        return True
    return False


async def _retire_dataset_evaluator_keys_for_registration(
    dataset: DatasetDocument,
    *,
    registration_id: str,
    include_default_slots: bool,
) -> int:
    """按注册收敛数据集下的 evaluator 键；可选一并清无 registration 默认槽。"""
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        parse_evaluator_key_tail,
    )

    reg = registration_id.strip()
    retired = 0
    for key in await _dataset_evaluator_record_keys(dataset):
        _project, _evaluator, _agent, row_tail = split_evaluator_record_key(key)
        row_reg, _row_stream = parse_evaluator_key_tail(row_tail)
        if row_reg:
            if row_reg != reg:
                continue
        elif not include_default_slots:
            continue
        if await retire_runtime_config(KIND_EVALUATOR, key):
            retired += 1
    return retired


def _dataset_trace_report_names(dataset: DatasetDocument) -> list[str]:
    """该实时数据集可能占用的 T+1 报表命名空间（含新旧形态）。"""
    from benchmark.domain.trace_report_config import versioned_trace_report_name  # noqa: PLC0415

    names: list[str] = []
    base = dataset.name.strip()
    if base:
        names.append(base)
    stream = (dataset.stream_instance_id or "").strip() or dataset.id.strip()
    versioned = versioned_trace_report_name(base, "", stream)
    if versioned and versioned not in names:
        names.append(versioned)
    return names


async def _retire_dataset_evaluator_entry(dataset: DatasetDocument) -> int:
    """停用数据集登记的实时评测条目（evaluator + T+1 报表项）并刷新快照。"""
    retired = 0
    for key in await _dataset_evaluator_record_keys(dataset):
        if await retire_runtime_config(KIND_EVALUATOR, key):
            retired += 1
    for report_name in _dataset_trace_report_names(dataset):
        if await retire_runtime_config(
            KIND_TRACE_REPORT,
            trace_report_record_key(report_name),
        ):
            retired += 1
    if retired:
        await refresh_eval_config_snapshot()
        logger.info(
            "realtime eval entries retired on stop: dataset=%s project=%s count=%d",
            dataset.id,
            dataset.project_id,
            retired,
        )
    return retired


async def _revive_dataset_evaluator_entry(dataset: DatasetDocument) -> int:
    """恢复数据集登记的实时评测条目（evaluator + T+1 报表项）并刷新快照。"""
    revived = 0
    for key in await _dataset_evaluator_record_keys(dataset):
        if await restore_runtime_config(KIND_EVALUATOR, key):
            revived += 1
    for report_name in _dataset_trace_report_names(dataset):
        if await restore_runtime_config(
            KIND_TRACE_REPORT,
            trace_report_record_key(report_name),
        ):
            revived += 1
    if revived:
        await refresh_eval_config_snapshot()
        logger.info(
            "realtime eval entries revived: dataset=%s project=%s count=%d",
            dataset.id,
            dataset.project_id,
            revived,
        )
    return revived


async def find_dataset_document_by_id(dataset_id: str) -> DatasetDocument | None:
    """按数据集 id 精确查找文档（触发入口优先路径）。

    同名数据集可有多条文档（同文件多次上传），按名查只能取首命中，
    同名改内容重传时会拼到另一条文档的 OSS 路径；触发请求带
    dataset_id 时必须按 id 定位，按名查仅供未带 id 的存量调用方。
    """
    row = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_DATASET,
        record_key=benchmark_dataset_record_key(dataset_id.strip()),
        deleted_at__isnull=True,
    ).first()
    if row is None:
        return None
    return DatasetDocument.model_validate(_payload(row))


async def find_dataset_document_by_name(dataset_name: str) -> DatasetDocument | None:
    """按名称查找数据集文档（实时采样流程分流依据）。

    供触发类入口（manual-runs 等）判定「数据集为实时采样时走实时评测
    流程」：返回整个文档，``source == "realtime"`` 表示线上 trace 实时
    采样数据集（触发时应激活实时链而非批量回放）；其余来源（上传 /
    OSS 导入）由调用方按批量回放处理，行为不变。名称匹配按 trim 后
    全等；未命中返回 ``None``（调用方按批量处理）。
    """
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_DATASET,
        deleted_at__isnull=True,
    )
    target = dataset_name.strip()
    for row in rows:
        if str(_payload(row).get("name", "")).strip() == target:
            return DatasetDocument.model_validate(_payload(row))
    return None


async def _cascade_retire_evaluator_for_registration(agent_id: str) -> int:
    """按 registrationId 级联停用实时执行面条目并刷新快照（删 agent / 下线共用）。"""
    datasets = await _datasets_referenced_by_agent_tasks(
        agent_id,
        include_deleted_tasks=True,
    )
    retired = await _retire_evaluator_entries_by_registration(agent_id)
    for dataset in datasets:
        shared = await _dataset_has_other_agent_tasks(
            dataset.id,
            exclude_agent_id=agent_id,
        )
        retired += await _retire_dataset_evaluator_keys_for_registration(
            dataset,
            registration_id=agent_id,
            include_default_slots=not shared,
        )
    if retired:
        await refresh_eval_config_snapshot()
        logger.info(
            "agent offline/delete cascaded evaluator retire: agent=%s count=%d",
            agent_id,
            retired,
        )
    return retired


async def delete_agent(agent_id: str) -> dict[str, Any]:
    """软删评测代理，并级联 retire 绑定该 registration 的实时执行面条目。

    WebUI 删 agent 只清 ``benchmark_agent`` 时，``kind=evaluator`` 仍会
    被 Kafka consumer 命中（Configs found）。此处按 registrationId（键与
    payload）收敛；若仍有任务挂在该 agent 上，再按任务数据集清理本注册
    槽（无其它 agent 共享同数据集时一并清默认槽）。
    """
    agent_id = agent_id.strip()
    agent = await find_agent_document_by_id(agent_id)
    if agent is None:
        raise ConsoleError(404, "记录不存在")
    deleted = await retire_runtime_config(
        KIND_BENCHMARK_AGENT,
        benchmark_agent_record_key(agent_id),
    )
    if not deleted:
        raise ConsoleError(404, "记录不存在")
    retired = await _cascade_retire_evaluator_for_registration(agent_id)
    if retired:
        logger.info(
            "agent delete cascaded evaluator retire: agent=%s name=%s count=%d",
            agent_id,
            agent.name,
            retired,
        )
    return {"deleted": True}


async def save_task(document: EvalTaskDocument) -> dict[str, Any]:
    """保存评测任务全文。不触发 benchmark 运行。"""
    scope = await _ensure_scope(document.agent_scope)
    if scope is None:
        raise ConsoleError(400, "agentScope is required")
    agent = await find_agent_document_by_id(document.agent_id)
    # 已落库注册：L1+L2（含绑定 agentScope 的普通注册）。文档未落库时放行
    # （页面联写顺序兼容）。
    if agent is not None and not is_registration_visible(
        RegistrationVisibility(
            agent_scope=scope,
            agent_name=agent.name,
            customization=agent.customization,
            registration_scope=agent.agent_scope,
            policy=get_ownership_policy(),
        )
    ):
        raise ConsoleError(
            403,
            f"agent_not_visible_in_scope: {agent.name}({document.agent_id}) "
            f"在评测对象 {scope} 下不可见",
        )
    row = await upsert_runtime_config(
        kind=KIND_EVAL_TASK,
        record_key=eval_task_record_key(document.id),
        payload=document.model_dump(by_alias=True, exclude_none=True),
        scope_key=agent_scope_key(document.agent_scope),
    )
    return _public(document, row)


#: run_id 尾部数据日期长度（yyyymmdd）
_DAILY_DATE_SUFFIX_LEN = 8


def _daily_report_date(run: BenchmarkRun) -> str | None:
    """日报数据日期：优先 ``run_id`` 中的 yyyymmdd 段，回退完成日。

    兼容 ``{stem}-{yyyymmdd}`` 与停止窗口 ``{stem}-{yyyymmdd}-stopped-{ts}``。
    """
    parts = str(run.run_id or "").split("-")
    for part in reversed(parts):
        if len(part) == _DAILY_DATE_SUFFIX_LEN and part.isdigit():
            return f"{part[:4]}-{part[4:6]}-{part[6:]}"
    finished = run.finished_at or run.created_at
    return finished.strftime("%Y-%m-%d") if finished is not None else None


def _calendar_date_label(label: str | None) -> date | None:
    """文档时间标签（``registeredAt`` / ``startedAt``）→ 日历日。

    仅解析前缀 ``YYYY-MM-DD``；非法/空返回 ``None``（不做时间下界过滤）。
    """
    text = (label or "").strip()
    _ymd_prefix_len = 10
    if len(text) >= _ymd_prefix_len and text[4] == "-" and text[7] == "-":
        try:
            return date.fromisoformat(text[:_ymd_prefix_len])
        except ValueError:
            return None
    return None


def _task_started_date(started_at: str) -> date | None:
    """兼容旧名：同 :func:`_calendar_date_label`（日报下界已改用注册时间）。"""
    return _calendar_date_label(started_at)


def _benchmark_name_aliases(agent_or_benchmark_name: str) -> set[str]:
    """执行侧 benchmark 名与对外 agent 名互认（含 ``_benchmark`` 后缀与 -/_）。

    实时 T+1 run 的 ``benchmark_name`` 字段实际是**数据集名**
    （如 ``amap-agent-sampling-2026-10-20``），不是 agent.name；agent
    身份只出现在 ``config_json.benchmark_component``。别名仅用于消歧
    命名空间与 component 软匹配，不能拿来当报表主键。
    """
    from benchmark.runs.common.agent_catalog import (  # noqa: PLC0415
        canonical_agent_name,
    )

    cleaned = agent_or_benchmark_name.strip()
    if not cleaned:
        return set()
    stem = canonical_agent_name(cleaned)
    aliases = {
        cleaned,
        stem,
        f"{stem}_benchmark",
        cleaned.replace("_", "-"),
        cleaned.replace("-", "_"),
        stem.replace("_", "-"),
        stem.replace("-", "_"),
    }
    return {item for item in aliases if item}


def _trace_report_namespaces(
    dataset_name: str,
    agent_name: str,
    *,
    registration_id: str | None = None,
) -> list[str]:
    """旧版 T+1 日报落在 ``run.benchmark_name`` 上的报表命名空间（兼容读）。

    新语义：``dataset_name``=页面数据集名，``benchmark_name``=内部 agent；
    命名空间改存 ``config.trace_report_namespace``。本函数仅供双读
    旧 run（``dataset_name`` 曾为 ``trace_pipeline:*``）。
    """
    from benchmark.domain.trace_report_config import (  # noqa: PLC0415
        versioned_trace_report_name,
    )

    names = [dataset_name]
    if registration_id:
        names.insert(0, versioned_trace_report_name(dataset_name, registration_id))
    for alias in _benchmark_name_aliases(agent_name):
        names.append(f"{alias}-{dataset_name}")
        if registration_id:
            names.append(versioned_trace_report_name(f"{alias}-{dataset_name}", registration_id))
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _is_legacy_trace_daily_run(run: BenchmarkRun) -> bool:
    """旧 T+1 落库：``dataset_name`` 为 ``trace_pipeline:*`` 或空。"""
    dn = (run.dataset_name or "").strip()
    return (not dn) or dn.startswith("trace_pipeline:")


class DailyReportMatchContext(NamedTuple):
    """日报归属任务的匹配维度（FUNC-PARAMS-001 收敛）。"""

    agent_name: str
    dataset_name: str
    registration_id: str | None = None
    require_version_isolation: bool = False
    stream_instance_id: str | None = None


def _daily_report_matches_benchmark(  # noqa: PLR0911, PLR0912 — 新/旧日报双读匹配矩阵
    run: BenchmarkRun,
    config: dict[str, Any],
    *,
    ctx: DailyReportMatchContext,
) -> bool:
    """日报是否归属本任务。

    **新语义**（``dataset_name`` = 页面注册数据集名）：
    - 列 ``dataset_name`` 对齐任务数据集文档名；
    - 列 ``benchmark_name`` / ``benchmark_component`` = 内部 agent（软匹配）；
    - ``config.agent_registration_id`` 与任务注册 id 精确匹配（若有）；
    - ``require_version_isolation`` 时必须带本注册 id。

    **旧语义**（``dataset_name`` 为 ``trace_pipeline:*``）：仍按
    ``run.benchmark_name`` ∈ 数据集命名空间 / ``{bench}-{dataset}`` /
    ``{dataset}__reg__{reg}`` 匹配。
    """
    from benchmark.domain.trace_report_config import (  # noqa: PLC0415
        AMAP_AGENT_LEGACY_DAILY_NAMESPACE,
        TRACE_REPORT_NAMESPACE_CONFIG_KEY,
        TRACE_REPORT_REGISTRATION_NAME_SEP,
        amap_agent_legacy_daily_namespaces,
        registered_dataset_name_from_report_name,
        versioned_trace_report_name,
    )

    agent_name = ctx.agent_name
    dataset_name = ctx.dataset_name
    registration_id = ctx.registration_id
    require_version_isolation = ctx.require_version_isolation
    stream_instance_id = (ctx.stream_instance_id or "").strip()
    run_stream = str(config.get("stream_instance_id") or "").strip()
    if stream_instance_id and run_stream and run_stream != stream_instance_id:
        return False
    run_reg = str(config.get("agent_registration_id") or "").strip() or None
    # run 已盖注册戳：与任务 agent.id 不一致则剔除；任务侧尚无 id 时不因
    # 「有戳」误杀（旧逻辑在 registration_id=None 时一律 False，导致
    # example 回填带 agent_registration_id 的 run 永远不进任务页）。
    if run_reg is not None and registration_id is not None and run_reg != registration_id:
        return False
    # 版本隔离只约束「已打注册戳」的 run。无戳的旧 T+1（如
    # benchmark_name=amap_agent_daily、dataset=trace_pipeline:*）视为
    # 注册前共享历史，仍按命名空间回填，避免任务页只剩带 @reg 的当天。
    if require_version_isolation and run_reg is not None:
        if not registration_id or run_reg != registration_id:
            return False

    # amap_agent_benchmark 专属：旧列名 amap_agent_daily 合并进任务页日报列表
    # （新 run_id / 数据集命名空间与旧格式无关，需显式互认；其他组件不启用）。
    if (
        amap_agent_legacy_daily_namespaces(agent_name)
        and (run.benchmark_name or "").strip() == AMAP_AGENT_LEGACY_DAILY_NAMESPACE
    ):
        return True

    aliases = _benchmark_name_aliases(agent_name)

    if not _is_legacy_trace_daily_run(run):
        if stream_instance_id and run_stream != stream_instance_id:
            return False
        page_ds = registered_dataset_name_from_report_name(
            (run.dataset_name or "").strip(),
        )
        if page_ds != dataset_name:
            return False
        from benchmark.domain.eval_config import parse_registration_eval_key  # noqa: PLC0415

        col = (run.benchmark_name or "").strip()
        _stem, col_reg = parse_registration_eval_key(col)
        # 列已落 stem@reg / 裸 registrationId：与任务注册 id 对齐即可
        if registration_id and registration_id in {col, col_reg}:
            return True
        # 裸报表命名空间与页面数据集同名（历史 amap_agent_daily）：无 @reg 也归属
        if col == dataset_name or (_stem == dataset_name and not col_reg):
            return True
        # 内部 agent 软匹配：列茎 / config.benchmark_component
        component = config.get("benchmark_component")
        candidates = {col, _stem} if _stem else {col}
        if isinstance(component, str) and component.strip():
            candidates.add(component.strip())
        candidates.discard("")
        if candidates and aliases:
            if not any(aliases & _benchmark_name_aliases(c) for c in candidates):
                return False
        return True

    # —— 旧 run 双读 ——
    legacy_ns = (run.benchmark_name or "").strip()
    report_ns = str(config.get(TRACE_REPORT_NAMESPACE_CONFIG_KEY) or "").strip()
    if run_reg is None and registration_id and TRACE_REPORT_REGISTRATION_NAME_SEP in legacy_ns:
        expected = versioned_trace_report_name(dataset_name, registration_id)
        if legacy_ns != expected:
            return False

    if registration_id:
        versioned = versioned_trace_report_name(dataset_name, registration_id)
        if versioned in {legacy_ns, report_ns}:
            return True
    if dataset_name in {legacy_ns, report_ns}:
        # 无注册戳的旧报：共享历史，版本隔离注册也可见
        if require_version_isolation and run_reg is not None:
            return False
        component = config.get("benchmark_component")
        if not isinstance(component, str) or not component.strip():
            return True
        if not aliases:
            return True
        return bool(aliases & _benchmark_name_aliases(component.strip()))
    if not aliases:
        return False
    return any(legacy_ns == f"{alias}-{dataset_name}" for alias in aliases)


async def _daily_reports_for(document: EvalTaskDocument) -> list[DailyReport] | None:
    """实时任务的每日报告回填：T+1 日报落在 benchmark_runs，任务文档只存占位。

    任务创建时前端固定写 ``dailyReports: []``，文档本身永远不会被回写。
    读路径按「页面数据集名 + 内部 agent + agent 注册时间」过滤已完成的
    T+1 日报（``config_json.trace_daily_report``）。

    - **新语义**：``run.dataset_name`` = 数据集注册页名称（如
      ``example-v56`` / ``amap-agent-sampling-${date}``）；
      ``run.benchmark_name`` = registrationId（有则；否则逻辑组件名）；
      逻辑名在 ``config.benchmark_component``。
    - **旧语义双读**：``dataset_name`` 曾为 ``trace_pipeline:*``，报表主键
      在 ``run.benchmark_name``（数据集名 / ``{dataset}__reg__{reg}``）。
    - **时间**：日报数据日 ≥ **agent ``registeredAt``** 日历日（不用任务
      ``startedAt``）。任务中途停止再启动后仍能看到注册以来的历史日报。

    非实时任务（``daily_reports is None`` 且数据集非 realtime）返回
    ``None`` 表示不回填；查不到/DB 异常回退文档存量值，不阻断列表。
    """
    from tortoise.expressions import Q  # noqa: PLC0415

    dataset = await find_dataset_document(document.dataset_id)
    if document.daily_reports is None and (dataset is None or dataset.source != "realtime"):
        return None
    if dataset is None:
        return document.daily_reports
    agent = await find_agent_document_by_id(document.agent_id)
    agent_name = agent.name.strip() if agent is not None else ""
    registration_id = agent.id.strip() if agent is not None and agent.id else None
    # 带 judge 定制的注册（example 双链）强制按 registration 隔离回填
    require_version = bool(
        agent is not None
        and agent.customization is not None
        and agent.customization.judge is not None
    )
    # 查询旧命名空间时带上 registration（若有），避免漏查版本隔离旧 run
    legacy_namespaces = _trace_report_namespaces(
        dataset.name,
        agent_name,
        registration_id=registration_id,
    )
    # 历史可见性按注册锚点，任务启停不截断已出日报。
    anchor_date = _calendar_date_label(agent.registered_at) if agent is not None else None
    try:
        # 新：按页面数据集名；旧：按 benchmark_name 命名空间
        runs = (
            await BenchmarkRun.filter(deleted_at__isnull=True)
            .filter(Q(dataset_name=dataset.name) | Q(benchmark_name__in=legacy_namespaces))
            .order_by("-created_at")
        )
    except Exception:
        logger.warning("daily reports lookup failed: dataset=%s", dataset.name, exc_info=True)
        return document.daily_reports
    reports: list[DailyReport] = []
    skipped_no_report_site = 0
    for run in runs:
        config = run.config_json if isinstance(run.config_json, dict) else {}
        if config.get("trace_daily_report") is not True:
            continue
        # completed：正常 T+1；stopped：人工停止窗口阶段性报告（§6.8）
        run_status = str(getattr(run.status, "value", run.status))
        if run_status not in ("completed", "stopped"):
            continue
        if not _daily_report_matches_benchmark(
            run,
            config,
            ctx=DailyReportMatchContext(
                agent_name=agent_name,
                dataset_name=dataset.name,
                registration_id=registration_id,
                require_version_isolation=require_version,
                stream_instance_id=dataset.stream_instance_id,
            ),
        ):
            continue
        meta = run.report_meta_json if isinstance(run.report_meta_json, dict) else {}
        # Pages 与 OSS 分字段返回；仅有 OSS 时也保留日报供页面回退展示。
        report_url = meta.get("report_site_url")
        oss_report_url = meta.get("report_url")
        has_pages = isinstance(report_url, str) and bool(report_url)
        has_oss = isinstance(oss_report_url, str) and bool(oss_report_url)
        date_label = _daily_report_date(run)
        if not has_pages and not has_oss:
            skipped_no_report_site += 1
            continue
        if run.final_score is None:
            continue
        if date_label is None:
            continue
        if anchor_date is not None and date_label < anchor_date.isoformat():
            continue
        reports.append(
            DailyReport(
                date=date_label,
                score=float(run.final_score),
                itemsCount=int(run.total_samples or 0),
                reportUrl=report_url if has_pages else None,
                ossReportUrl=oss_report_url if has_oss else None,
                id=str(run.id),
                runId=(str(run.run_id).strip() or None) if run.run_id else None,
            ),
        )
    if skipped_no_report_site and not reports:
        logger.warning(
            "daily reports: %d completed T+1 run(s) matched dataset=%s but lack "
            "report_site_url and report_url — task page will stay empty until a report is available",
            skipped_no_report_site,
            dataset.name,
        )
    reports.sort(key=lambda item: item.date, reverse=True)
    # 任务详情按 10 条/页展示全部日报，不再截断历史。
    return reports


async def _reconcile_task_document_for_view(document: EvalTaskDocument) -> EvalTaskDocument:
    """读路径纠偏并落库：刷新存活 run 的 pk；定时任务误写终态纠回 active。

    CLI 同 ``run_id`` 重跑成功后，文档常停在 ``failed`` 且钉着旧 pk；
    console 列表/详情读取时一并修好，已重跑完成的任务无需再等调度 tick。
    """
    updates: dict[str, Any] = {}
    if document.manual_run_id:
        try:
            live_run = await BenchmarkRun.filter(
                run_id=document.manual_run_id,
                deleted_at__isnull=True,
            ).first()
        except Exception:
            live_run = None
        if live_run is not None and document.manual_run_pk != str(live_run.id):
            updates["manual_run_pk"] = str(live_run.id)
    if (
        document.trigger != "immediate"
        and document.status != "stopped"
        and document.status in ("complete", "failed")
    ):
        updates["status"] = "active"
        updates["failure_reason"] = None
    if not updates:
        return document
    healed = document.model_copy(update=updates)
    try:
        await save_task(healed)
        return healed
    except Exception:
        logger.warning(
            "eval task view reconcile failed: id=%s updates=%s",
            document.id,
            updates,
            exc_info=True,
        )
        return document


async def _task_view(document: EvalTaskDocument, row: RuntimeConfigRecord) -> dict[str, Any]:
    """任务对外视图：文档字段 + 实时任务的每日报告回填。"""
    document = await _reconcile_task_document_for_view(document)
    body = _public(document, row)
    reports = await _daily_reports_for(document)
    if reports is not None:
        body["dailyReports"] = [item.model_dump(by_alias=True) for item in reports]
    return body


async def list_tasks(agent_scope: str | None = None) -> list[dict[str, Any]]:
    """列出评测任务。传入评测对象时只返回该对象（自定义对象须已注册）。"""
    scope = await _ensure_scope(agent_scope)
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVAL_TASK,
        deleted_at__isnull=True,
    ).order_by("-updated_at")
    items: list[dict[str, Any]] = []
    for row in rows:
        document = EvalTaskDocument.model_validate(_payload(row))
        if scope is not None and document.agent_scope != scope:
            continue
        items.append(await _task_view(document, row))
    return items


async def find_task_document(task_id: str) -> EvalTaskDocument | None:
    """按 id 查找任务文档（定时触发时重新读最新状态/配置）。"""
    row = await RuntimeConfigRecord.filter(
        kind=KIND_EVAL_TASK,
        record_key=eval_task_record_key(task_id.strip()),
        deleted_at__isnull=True,
    ).first()
    if row is None:
        return None
    return EvalTaskDocument.model_validate(_payload(row))


async def list_schedulable_tasks() -> list[EvalTaskDocument]:
    """列出需要注册定时调度的任务（天级 / 周级 / 自定义 cron，且调度中）。

    立即触发（immediate）不在调度范围内；已停止任务由用户显式关停，
    不因定时到点而复活。仅 ``status=active`` 注册调度。

    历史 converge 曾把单次 run 的 complete/failed 写回定时任务文档：此处
    纠回 ``active`` 并落库，避免调度静默丢失或停止按钮不可用。
    读失败向上抛（调度侧记日志并保留上一轮 job）。
    """
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVAL_TASK,
        deleted_at__isnull=True,
    )
    tasks: list[EvalTaskDocument] = []
    for row in rows:
        try:
            document = EvalTaskDocument.model_validate(_payload(row))
        except Exception:
            logger.warning(
                "eval task document invalid, skip scheduling: key=%s",
                row.record_key,
                exc_info=True,
            )
            continue
        if document.trigger == "immediate" or document.status == "stopped":
            continue
        if document.status in ("complete", "failed"):
            healed = document.model_copy(
                update={"status": "active", "failure_reason": None},
            )
            try:
                await save_task(healed)
                document = healed
            except Exception:
                logger.warning(
                    "eval task schedule heal failed: key=%s status=%s",
                    row.record_key,
                    document.status,
                    exc_info=True,
                )
                continue
        if document.status != "active":
            continue
        tasks.append(document)
    return tasks


async def record_task_run(task_id: str, *, run_id: str, run_pk: str) -> None:
    """定时触发成功后回写任务文档的最新 run（状态/报告回填按它关联）。"""
    row = await _require_row(KIND_EVAL_TASK, eval_task_record_key(task_id))
    document = EvalTaskDocument.model_validate(_payload(row))
    updated = document.model_copy(
        update={"manual_run_id": run_id, "manual_run_pk": run_pk},
    )
    await save_task(updated)


async def get_task(task_id: str) -> dict[str, Any]:
    """读取一条评测任务。"""
    row = await _require_row(KIND_EVAL_TASK, eval_task_record_key(task_id))
    return await _task_view(EvalTaskDocument.model_validate(_payload(row)), row)


async def delete_task(task_id: str) -> dict[str, Any]:
    """软删任务文档本身（deleted_at 标记，列表/详情从此不返回）。

    只删任务：该任务名下的 benchmark runs / run_items / 报告等评测
    结果一律保留、不做处理（不终止在跑的 run、不标记 failed）。
    如需先收敛执行面，应先走 stop 动作再删除。
    """
    await _require_row(KIND_EVAL_TASK, eval_task_record_key(task_id))
    deleted = await retire_runtime_config(KIND_EVAL_TASK, eval_task_record_key(task_id))
    if not deleted:
        raise ConsoleError(404, "记录不存在")
    return {"deleted": True}


async def apply_remove_task(task_id: str) -> dict[str, Any]:
    """删除任务文档；实时流任务删除后按剩余运行中任务联动收敛条目。

    删除与停止同属「任务不再启用」：该任务是最后一个运行中的实时流任务
    时，删除后停用 evaluator 条目，避免无任务但评测仍在进行。数据集
    采样状态不随任务删除变更（同 ``apply_stop_task``）。
    """
    document = await get_task(task_id)
    result = await delete_task(task_id)
    await _retire_realtime_entries_when_idle(
        document.get("datasetId"),
        task_id=task_id,
    )
    return result


async def _mutate_task(
    task_id: str,
    mutate: Callable[[EvalTaskDocument], EvalTaskDocument],
) -> dict[str, Any]:
    row = await _require_row(KIND_EVAL_TASK, eval_task_record_key(task_id))
    document = mutate(EvalTaskDocument.model_validate(_payload(row)))
    return await save_task(document)


async def apply_stop_task(task_id: str) -> dict[str, Any]:
    """停止任务并保存；实时流任务按「是否仍有其他运行中任务」联动收敛条目。

    执行面（``kind=evaluator`` 条目）是数据集级的：仍有其他任务在使用
    同一采样时只停当前任务，不动条目（不误伤）；最后一个运行中任务
    停止时停用条目（无任务即不评测）。数据集采样状态不随任务停止
    变更——停止采样是独立的页面动作（见 ``apply_stop_sampling``）。
    """
    result = await _mutate_task(task_id, stop_task)
    await _retire_realtime_entries_when_idle(result.get("datasetId"), task_id=task_id)
    return result


async def apply_rerun_task(task_id: str) -> dict[str, Any]:
    """重新运行并保存；实时流任务联动恢复采样与 evaluator 条目。"""
    result = await _mutate_task(task_id, rerun_task)
    await _cascade_start_realtime_dataset(result.get("datasetId"))
    return result


async def _retire_realtime_entries_when_idle(
    dataset_id: object,
    *,
    task_id: str | None = None,
) -> None:
    """无运行中任务时停用 realtime 数据集的执行面条目（非 realtime 跳过）。

    任务停止/删除只收敛执行面：共享采样的其他任务仍运行时不动条目；
    最后一个运行中任务消失时 retire 条目（无任务即不评测）。数据集
    采样状态不在此变更——它是页面侧显式意图（「采样中」开关），仅
    停止采样（``apply_stop_sampling``）可改，任务生命周期不越权修改；
    任务重新运行（rerun）时条目与采样状态由联动恢复一并拉回。

    数据集若已被页面先软删，仍按软删行上的 projectId/agentName 收敛
    （见 :func:`find_dataset_document_any`），避免孤儿 evaluator。
    """
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        logger.warning(
            "realtime cascade skipped: missing datasetId task=%s",
            task_id or "-",
        )
        return
    dataset = await _realtime_dataset_of(dataset_id, include_deleted=True)
    if dataset is None:
        logger.warning(
            "realtime cascade skipped: dataset not found or not realtime dataset=%s task=%s",
            dataset_id,
            task_id or "-",
        )
        return
    others = await _active_task_ids_of_dataset(dataset.id, exclude_task_id=task_id)
    if others:
        logger.info(
            "realtime cascade deferred: other active tasks remain dataset=%s task=%s others=%s",
            dataset.id,
            task_id or "-",
            others,
        )
        return
    await _retire_dataset_evaluator_entry(dataset)


async def _active_task_ids_of_dataset(
    dataset_id: str,
    *,
    exclude_task_id: str | None = None,
) -> list[str]:
    """该数据集下仍在运行（active）的评测任务 id 列表。"""
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_EVAL_TASK,
        deleted_at__isnull=True,
    )
    ids: list[str] = []
    for row in rows:
        document = EvalTaskDocument.model_validate(_payload(row))
        if document.dataset_id != dataset_id or document.status != "active":
            continue
        if exclude_task_id is not None and document.id == exclude_task_id:
            continue
        ids.append(document.id)
    return ids


async def _stop_active_tasks_of_dataset(dataset_id: str) -> int:
    """停止该数据集下所有运行中的任务（采样停止的联动）。"""
    stopped = 0
    for task_id in await _active_task_ids_of_dataset(dataset_id):
        await _mutate_task(task_id, stop_task)
        stopped += 1
    if stopped:
        logger.info(
            "realtime tasks stopped with sampling: dataset=%s count=%d",
            dataset_id,
            stopped,
        )
    return stopped


async def _cascade_start_realtime_dataset(dataset_id: object) -> None:
    """恢复 realtime 数据集的采样与执行面条目（重新运行任务的联动）。"""
    dataset = await _realtime_dataset_of(dataset_id)
    if dataset is None:
        return
    if dataset.status == "ready":
        await _mutate_dataset(dataset.id, start_sampling)
    await _revive_dataset_evaluator_entry(dataset)


async def _realtime_dataset_of(
    dataset_id: object,
    *,
    include_deleted: bool = False,
) -> DatasetDocument | None:
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return None
    lookup = find_dataset_document_any if include_deleted else find_dataset_document
    dataset = await lookup(dataset_id.strip())
    if dataset is None or dataset.source != "realtime":
        return None
    return dataset


async def apply_retry_task(task_id: str) -> dict[str, Any]:
    """重试并保存。"""
    return await _mutate_task(task_id, retry_task)


def page_rubrics() -> list[dict[str, Any]]:
    """把代码白名单 rubric 读成页面 Rubric 形状。不改注册表。"""
    items: list[dict[str, Any]] = []
    for alias in sorted(RUBRIC_CLASS_REFS):
        spec = resolve_rubric_class(alias)
        criteria: list[dict[str, Any]] = []
        formula: RubricFormula = "custom"
        if spec.class_ref is not None:
            cls = load_plugin_class(spec.class_ref, what="rubric_class")
            formula = _PAGE_FORMULA.get(str(cls.aggregation_method), "custom")
            criteria = [_criterion(item) for item in cls.criteria()]
        items.append(
            {
                "id": alias,
                "rubricId": alias,
                "name": alias,
                "description": spec.class_ref or "",
                "classRef": spec.class_ref or "",
                "criteria": criteria,
                "formula": formula,
                "createdAt": "",
                "owner": "agent-eval",
                "usageCount": 0,
            },
        )
    return items


def _criterion(item: Criterion) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": item.id,
        "name": item.name,
        "description": item.description or "",
        "weight": item.weight,
    }
    evaluator_ref = getattr(item, "evaluator_ref", None)
    if evaluator_ref:
        body["evaluatorRef"] = evaluator_ref
    return body
