"""RuntimeConfigRecord — 页面配置的统一落库表。

评测条目、ODPS 任务，以及后续定时 / Dataset 任务都写这张表，用 ``kind`` 区分。
不再按配置种类各建一表。
"""

from dataclasses import dataclass

from benchmark.libs.models import BaseModelWithSoftDelete
from benchmark.libs.models.fields import fields

KIND_EVALUATOR = "evaluator"
KIND_ODPS_JOB = "odps_job"
KIND_SCHEDULED_TASK = "scheduled_task"
KIND_DATASET_TASK = "dataset_task"
KIND_KAFKA_TASK = "kafka_task"
KIND_BENCHMARK_AGENT = "benchmark_agent"
KIND_BENCHMARK_DATASET = "benchmark_dataset"
KIND_EVAL_TASK = "eval_task"
KIND_AGENT_SCOPE = "agent_scope"
KIND_TRACE_REPORT = "trace_report"
KIND_AGENT_VIEW = "agent_view"
KIND_OSS_UPLOAD_SEQUENCE = "oss_upload_sequence"
KIND_CUSTOMIZATION_SEQUENCE = "customization_sequence"
KIND_INSTANCE_SEQUENCE = "instance_sequence"

# 当前写入的 payload 契约。形状变化时递增，已有行保持自己的版本。
SCHEMA_VERSION = 1
SCOPE_GLOBAL = "global"


#: 第四段复合后缀：``{registrationId}__si__{streamInstanceId}``。
#: 实时流用 ``stream_instance_id`` 隔离同过滤条件/同采样意图的多数据集，
#: 使同 topic/同 group 下多登记互不覆盖；``registration_id`` 仍可并行。
STREAM_INSTANCE_KEY_SEP = "__si__"


def compose_evaluator_key_tail(
    registration_id: str = "",
    stream_instance_id: str = "",
) -> str:
    """拼 record_key 第四段：有 stream 时恒含 ``__si__``，便于还原。"""
    reg = registration_id.strip()
    stream = stream_instance_id.strip()
    if stream:
        return f"{reg}{STREAM_INSTANCE_KEY_SEP}{stream}"
    return reg


def parse_evaluator_key_tail(tail: str) -> tuple[str, str]:
    """拆第四段 → ``(registration_id, stream_instance_id)``。"""
    text = (tail or "").strip()
    if not text:
        return "", ""
    sep = STREAM_INSTANCE_KEY_SEP
    if sep in text:
        reg, _, stream = text.partition(sep)
        return reg.strip(), stream.strip()
    return text, ""


def evaluator_record_key(
    project_id: str,
    evaluator_name: str,
    agent_name: str,
    key_tail: str = "",
) -> str:
    """评测条目的唯一键。

    三段 ``evaluator:{project}:{evaluator}:{agent}``：无定制注册（amap 等）
    的默认槽位。四段再挂 ``:{key_tail}``：``key_tail`` 为 registration_id，
    或 ``compose_evaluator_key_tail(reg, streamInstanceId)``（实时多数据集隔离）。
    """
    base = f"{KIND_EVALUATOR}:{project_id}:{evaluator_name}:{agent_name}"
    tail = key_tail.strip()
    return f"{base}:{tail}" if tail else base


def odps_record_key(job_id: str) -> str:
    """ODPS 任务的唯一键。"""
    return f"{KIND_ODPS_JOB}:{job_id}"


def benchmark_agent_record_key(agent_id: str) -> str:
    """评测代理的唯一键。代理跨评测对象共享。"""
    return f"{KIND_BENCHMARK_AGENT}:{agent_id}"


def benchmark_dataset_record_key(dataset_id: str) -> str:
    """数据集的唯一键。"""
    return f"{KIND_BENCHMARK_DATASET}:{dataset_id}"


def eval_task_record_key(task_id: str) -> str:
    """评测任务的唯一键。"""
    return f"{KIND_EVAL_TASK}:{task_id}"


def agent_scope_record_key(scope_id: str) -> str:
    """自定义评测对象的唯一键（id 即业务传入的 agent name）。"""
    return f"{KIND_AGENT_SCOPE}:{scope_id}"


def scheduled_task_record_key(task_id: str) -> str:
    """页面定时任务的唯一键（kind=scheduled_task）。"""
    return f"{KIND_SCHEDULED_TASK}:{task_id}"


def trace_report_record_key(name: str) -> str:
    """实时流自动登记的 T+1 报表项唯一键（name = 报表命名空间，
    即数据集名；run_id = {name}-{yyyymmdd} 以此归组）。"""
    return f"{KIND_TRACE_REPORT}:{name}"


def agent_view_record_key(agent_name: str) -> str:
    """统一 agent 视图行的唯一键（视图主键 = 统一 agent 名）。"""
    return f"{KIND_AGENT_VIEW}:{agent_name}"


def oss_upload_sequence_record_key(partition_date: str) -> str:
    """OSS 上传当日顺序号计数器键（一天一行，payload 存 last_id）。"""
    return f"{KIND_OSS_UPLOAD_SEQUENCE}:{partition_date}"


def instance_sequence_record_key() -> str:
    """多实例内部 agent 名序号计数器键（全局一行，payload 存 last_id）。

    序号只增不减、永不复用，保证软删实例的内部名进墓碑后不被再分配
    （设计 §12.2：内部名永不复用）。
    """
    return f"{KIND_INSTANCE_SEQUENCE}:global"


@dataclass(frozen=True)
class CustomizationKey:
    """业务定制内容定位键：agent / business / variant / stage [+ agent_scope]。

    上传统计计数器行（DB 定位）与 Git 版本目录（落盘定位）共用同一把键，
    避免两处各拼一份后漂移。``agent_scope`` 缺省空串时 Git 路径不加
    scope 段（兼容存量）；非空时插入 agent 与 business 之间。
    """

    agent_name: str
    business: str
    variant_key: str
    stage: str
    agent_scope: str = ""

    def record_key(self) -> str:
        """本键在 runtime_config_records 中的计数器行键。"""
        return customization_sequence_record_key(self)


def customization_sequence_record_key(key: CustomizationKey) -> str:
    """业务定制内容版本计数器键。

    有 ``agent_scope`` 时：``(agent, scope, business, variant, stage)``；
    无 scope（存量）时保持四段键兼容。
    """
    scope = (key.agent_scope or "").strip()
    if scope:
        return (
            f"{KIND_CUSTOMIZATION_SEQUENCE}:{key.agent_name}:{scope}:"
            f"{key.business}:{key.variant_key}:{key.stage}"
        )
    return (
        f"{KIND_CUSTOMIZATION_SEQUENCE}:{key.agent_name}:"
        f"{key.business}:{key.variant_key}:{key.stage}"
    )


def agent_scope_key(agent_scope: str) -> str:
    """评测对象归属。数据集和任务按页面 scope 隔离。"""
    return f"agent:{agent_scope.strip()}"


def benchmark_scope_key(benchmark_name: str) -> str:
    """按 benchmark 归属（定时任务等）。"""
    return f"benchmark:{benchmark_name.strip()}"


def project_scope_key(project_id: str) -> str:
    """按 Langfuse project 列表。空 project 归到全局。"""
    cleaned = project_id.strip()
    if not cleaned:
        return SCOPE_GLOBAL
    return f"project:{cleaned}"


def split_evaluator_record_key(record_key: str) -> tuple[str, str, str, str]:
    """从记录键还原 ``(project, evaluator, agent, key_tail)``。

    ``key_tail`` 为第四段原串（可能含 ``__si__`` stream 后缀）。
    需要逻辑 registration / stream 时用 :func:`parse_evaluator_key_tail`。
    老三段键的 ``key_tail`` 为空串；四段及以上末段为 tail（不含 ``:``）。
    """
    parts = record_key.split(":")
    if not parts or parts[0] != KIND_EVALUATOR:
        msg = f"不是评测配置键: {record_key}"
        raise ValueError(msg)
    # 键段数即键形态契约：4 段 = 老三段键（tail 为空），≥5 段末段为 tail。
    key_legacy_parts = 4
    if len(parts) == key_legacy_parts:
        return parts[1], parts[2], parts[3], ""
    if len(parts) > key_legacy_parts:
        return parts[1], parts[2], parts[3], ":".join(parts[4:])
    msg = f"评测配置键段数不足: {record_key}"
    raise ValueError(msg)


class RuntimeConfigRecord(BaseModelWithSoftDelete):
    """一条页面配置（表: runtime_config_records）。

    新 kind 只增加取值和 payload 形状，不新增列、不新建表。
    归属用 ``scope_key`` 前缀区分：``project:``、``benchmark:``、``agent:``、``global``。
    种类特有字段放 ``payload``。要按某个新字段高频过滤时，再把它升成列。

    删除语义与 benchmark_runs 同轨（``BaseModelWithSoftDelete``）：统一
    用软删列（``is_deleted`` / ``deleted_at``，读侧过滤 ``deleted_at``
    为空）表达存在性，同键再次保存即复活。不再保留 ``enabled`` 列 ——
    它曾是从文档状态派生的只写镜像（无任一查询按它过滤），已删列；
    存量表经 table_ops 先 add ``is_deleted`` 并按 ``deleted_at`` 回填、
    发版后再 drop ``enabled``。
    """

    kind = fields.CharField(
        max_length=64,
        description=(
            "开放取值。现有：evaluator、odps_job、scheduled_task、benchmark_agent、"
            "benchmark_dataset、eval_task、agent_scope；预留 dataset_task、kafka_task"
        ),
    )
    record_key = fields.CharField(
        max_length=512,
        unique=True,
        description="kind 内身份，全局唯一。软删后再次保存复活同一行",
    )
    schema_version = fields.IntField(
        default=SCHEMA_VERSION,
        description="payload 契约版本。写入时盖当前版本，旧行保留原版本",
    )
    revision = fields.IntField(
        default=1,
        description="每次保存加一，供对账判断文档是否变化",
    )
    scope_key = fields.CharField(
        max_length=256,
        default=SCOPE_GLOBAL,
        description="归属查询键。project:{id} / benchmark:{name} / agent:{name} / global",
    )
    payload = fields.JSONField(description="该 kind 的配置文档")

    class Meta:
        table = "runtime_config_records"
        indexes = (("kind", "scope_key"),)


__all__ = [
    "CustomizationKey",
    "KIND_AGENT_SCOPE",
    "KIND_AGENT_VIEW",
    "KIND_CUSTOMIZATION_SEQUENCE",
    "KIND_INSTANCE_SEQUENCE",
    "KIND_OSS_UPLOAD_SEQUENCE",
    "KIND_BENCHMARK_AGENT",
    "KIND_BENCHMARK_DATASET",
    "KIND_DATASET_TASK",
    "KIND_EVAL_TASK",
    "KIND_EVALUATOR",
    "KIND_KAFKA_TASK",
    "KIND_ODPS_JOB",
    "KIND_SCHEDULED_TASK",
    "KIND_TRACE_REPORT",
    "SCHEMA_VERSION",
    "SCOPE_GLOBAL",
    "STREAM_INSTANCE_KEY_SEP",
    "RuntimeConfigRecord",
    "compose_evaluator_key_tail",
    "parse_evaluator_key_tail",
    "agent_scope_key",
    "agent_scope_record_key",
    "agent_view_record_key",
    "benchmark_scope_key",
    "customization_sequence_record_key",
    "instance_sequence_record_key",
    "oss_upload_sequence_record_key",
    "benchmark_agent_record_key",
    "benchmark_dataset_record_key",
    "eval_task_record_key",
    "evaluator_record_key",
    "odps_record_key",
    "project_scope_key",
    "scheduled_task_record_key",
    "trace_report_record_key",
    "split_evaluator_record_key",
]
