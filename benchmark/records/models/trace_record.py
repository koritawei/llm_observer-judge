"""trace 管线记录模型。

用法:
    from benchmark.records.models import TraceRecord
"""


from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.models import BaseModel
from benchmark.libs.models.fields import fields
from benchmark.libs.models.mixin import ObserverMixin

from benchmark.records.enums import TraceStatus, TriageLevel
from benchmark.records.models.validators import (
    validate_notification_dispatch_json,
    validate_notification_job_json,
    validate_attribution_detail_json,
    validate_judge_json,
    validate_label_studio_attribution_annotation_json,
    validate_label_studio_badcase_judge_annotation_json,
)

logger = get_logger(__name__)

#: 业务查找键 —— save() 自动截断时排除，避免破坏 get_by_trace_id 等查询。
_CHAR_CLIP_EXCLUDE: frozenset[str] = frozenset({
    "trace_id",
    "project_id",
    "agent_name",
    "evaluator_name",
})


def clip_char_value(
    model: type[BaseModel],
    field_name: str,
    value: str | None,
    *,
    record_id: object | None = None,
) -> str | None:
    """按模型字段 ``max_length`` 截断字符串；长度来自 ``_meta.fields_map``。

    用于绕过 ``save()`` 的路径（如 ``filter().update()``）。``None`` / 无
    ``max_length`` / 未超长原样返回。
    """
    if value is None:
        return None
    field = model._meta.fields_map.get(field_name)
    max_length = getattr(field, "max_length", None)
    if not isinstance(max_length, int) or len(value) <= max_length:
        return value
    logger.warning(
        "trace_record.char_field_truncated field=%s max_length=%s "
        "original_len=%s record_id=%s",
        field_name,
        max_length,
        len(value),
        record_id,
    )
    return value[:max_length]


class TraceRecord(BaseModel, ObserverMixin):
    """一条 trace 在反馈管线中的完整记录（表: trace_pipeline_records）。

    支持多 Evaluator 并行评估：同一 trace 可由不同 evaluator 独立评估，
    通过复合唯一键 ``(trace_id, project_id, agent_name, evaluator_name)``
    保证数据隔离。

    字段分为两层:

    **标量列** — 独立于 bucket 之外，因为它们是查询热路径字段，
    需要参与 WHERE / GROUP BY / 唯一索引查询。
    JSON bucket 内部的 key 无法被数据库索引，因此将高频查询维度
    提取为标量列以保证查询性能:
      - 标识: trace_id, project_id, evaluator_name
      - Agent: agent_name（字符串，元数据从 RemoteConfig 配置拉取）
      - 场景: scene（数据来源场景标识，默认非语音）
      - 管线: status (TraceStatus)
      - 分级: owner_group
      - 报表统计: triage_level, root_cause, notification_job_status

    **Bucket（JSON 对象）** — 只写不查的关联字段按领域分组，
    大字段 JSON 文本也合并到对应 bucket 中以减少表列数:
      - judge_json: 评判结果
      - attribution_detail_json: 归因详情
      - notification_dispatch_json: 派发信息
      - notification_job_json: Aone 工单详情
      - label_studio_attribution_annotation_json: 归因结果标注
      - label_studio_badcase_judge_annotation_json: badcase judge结果标注
      - trace_meta_json: 来源通道 source(kafka|http|dataset|benchmark) + dataset
        明细 / IO / 历史 digest

    通过 ``ObserverMixin`` 挂载 ``TraceRecordObserver``，在
    ``save()`` / ``delete()`` 后自动触发生命周期日志。
    Agent 元数据由 :mod:`benchmark.domain.agent_config` 从 RemoteConfig 配置拉取。
    """

    # ── 标识 ──────────────────────────────────────────────────────────
    # 标量列: 复合唯一键的组成部分，需参与 WHERE 查询和唯一索引
    trace_id = fields.CharField(
        max_length=128,
        description="trace 唯一标识，用于跨系统追踪",
    )
    project_id = fields.CharField(
        max_length=128,
        null=True,
        description="项目标识，用于数据隔离",
    )
    evaluator_name = fields.CharField(
        max_length=512,
        default="default",
        description="评估器名称，支持同一 trace 的多评估器并行评估",
    )

    # ── Agent ────────────────────────────────────────────────────────
    # 标量列: agent 名称字符串，元数据从 RemoteConfig 配置拉取（见 domain.agent_config）
    agent_name = fields.CharField(
        max_length=128,
        null=True,
        description="Agent 名称（元数据从 RemoteConfig 配置拉取）",
    )

    # ── 数据来源场景 ──────────────────────────────────────────────────
    # 标量列: 数据来源场景标识，用于区分语音/非语音场景，支持 WHERE 过滤
    scene = fields.CharField(
        max_length=64,
        default="non_voice",
        description="数据来源场景标识（voice=语音场景，non_voice=非语音场景，默认非语音）",
    )

    # ── 管线状态 ──────────────────────────────────────────────────────
    # 标量列: 管线核心状态字段，高频 WHERE 查询和分页排序
    status = fields.CharField(
        max_length=64,
        description="trace 在管线中的当前状态（new/scored/triaged/dispatched/closed）",
    )

    # ── 报表统计标量（查询热路径）──────────────────────────────
    # 标量列: 查询热路径字段，支持 DB 端 WHERE/GROUP BY。
    # 写入责任: triage_level / root_cause 由 service 层直写（update_triage_level /
    # update_attribution）；notification_job_status 仍由 observer 从 bucket 同步。
    # ── 分级（标量，查询热路径）────────────────────────────────
    # 标量列: 分级核心维度，用于查询过滤
    owner_group = fields.CharField(
        max_length=128,
        null=True,
        description="负责团队标识（由 TraceRecordService.update_attribution 直写，来源 domain.resolve_owner_route）",  # noqa: E501
    )

    # ── 报表统计标量（查询热路径）──────────────────────────────
    triage_level = fields.CharEnumField(
        TriageLevel,
        max_length=32,
        null=True,
        description="分级结果（由 TraceRecordService.update_triage_level 直写，取值受 TriageLevel 枚举约束；NULL ≡ 跑过 triage 但 classifier 未给出合法 level，如 score 缺失）",  # noqa: E501
    )
    root_cause = fields.CharField(
        max_length=256,
        null=True,
        description="根因分类（由 TraceRecordService.update_attribution 直写，不再从 attribution_detail_json 同步）",  # noqa: E501
    )
    notification_job_status = fields.CharField(
        max_length=32,
        null=True,
        description="Aone 工单状态（由 TraceRecordObserver.on_save 从 notification_job_json bucket 自动同步）",  # noqa: E501
    )

    def _clip_overlong_char_fields(self) -> None:
        """按 ``_meta`` 中 CharField.max_length 截断超长标量（DDL/模型同源）。

        排除业务查找键；JSON bucket 不受影响（可保留全文）。改字段长度时只改
        模型定义即可，无需同步硬编码常量。
        """
        for name, field in self._meta.fields_map.items():
            if name in _CHAR_CLIP_EXCLUDE:
                continue
            max_length = getattr(field, "max_length", None)
            if not isinstance(max_length, int):
                continue
            value = getattr(self, name, None)
            if not isinstance(value, str) or len(value) <= max_length:
                continue
            logger.warning(
                "trace_record.char_field_truncated field=%s max_length=%s "
                "original_len=%s record_id=%s",
                name,
                max_length,
                len(value),
                getattr(self, "id", None),
            )
            setattr(self, name, value[:max_length])

    async def save(self, *args: object, **kwargs: object) -> object:
        """覆写 save(): 先按字段 meta 截断超长 CharField，再写库并触发 observer。

        ``TraceRecordObserver.on_save`` 负责 bucket→标量同步和管线日志。
        """
        self._clip_overlong_char_fields()
        result = await super().save(*args, **kwargs)
        observer = type(self).get_observer()
        if observer is not None:
            await observer.on_save(self)
        return result

    # ── Bucket: 评判结果 ─────────────────────────────────────────────
    # JSONField: 评分结果 + 分级原始 JSON 文本（原 triage_json 大字段已合并）
    #
    # Schema:
    #   score: str               — 评分结果（如 "0.85"）
    #   triage_level: str        — 分级结果（如 "good"/"watch"/"mild_bad"/"severe_bad"）
    #   reason: str              — 评分理由
    #   triage_raw: str | None   — 分级原始 JSON 文本（原 triage_json 大字段）
    #   updated_at: str          — bucket 最后更新时间（UTC+8 ISO datetime）
    judge_json = fields.JSONField(
        null=True,
        description="评判结果 bucket — score/triage_level/reason/triage_raw（triage_level 由 update_triage_level 维护，标量列另由 service 直写）",  # noqa: E501
        validators=[validate_judge_json],
    )

    # ── Bucket: 归因详情 ─────────────────────────────────────────────
    # JSONField: 归因详情 + 归因/证据/推理原始 JSON 文本（原 3 个大字段已合并）
    #
    # Schema:
    #   root_cause: str          — 根因分类
    #   owner_domain: str        — 负责领域（AttributionResult.category）
    #   owner_group: str         — 责任组
    #   confidence: float        — 置信度（0.0-1.0）
    #   attribution_raw: str | None — 归因原始 JSON（raw_response 或完整快照）
    #   evidence_raw: str | None    — 证据引用 JSON 文本（evidence_refs）
    #   reasoning_raw: str | None  — 判断依据原文（AttributionResult.reasoning）
    #   action_hint: str | None    — 建议动作
    #   needs_human_review: bool | None — 是否需要人工复核
    #   corrected_triage_level / original_triage_level — 归因推翻分级
    #   updated_at: str             — bucket 最后更新时间（UTC+8 ISO datetime）
    attribution_detail_json = fields.JSONField(
        null=True,
        description="归因详情 bucket — 与 AttributionResult 对齐（root_cause/owner_*/confidence/attribution_raw/evidence_raw/reasoning_raw/action_hint/needs_human_review）；由 update_attribution 维护，reasoning_raw 也可由 update_reasoning 单独补写",  # noqa: E501
        validators=[validate_attribution_detail_json],
    )

    # ── Bucket: 派发信息 ────────────────────────────────────────────
    # JSONField: Aone 派发的工单引用、URL、指派人等
    #
    # Schema:
    #   notification_ref: str           — 派发引用（工单号）
    #   notification_url: str           — 派发 URL
    #   notification_assigned_to: str   — 指派人
    #   notification_title: str         — 工单标题
    #   notification_dispatched_at: str — 派发时间（ISO datetime）
    #   updated_at: str         — bucket 最后更新时间（UTC+8 ISO datetime）
    notification_dispatch_json = fields.JSONField(
        null=True,
        description="派发信息 bucket — notification_ref/notification_url/notification_assigned_to/notification_title/notification_dispatched_at",  # noqa: E501
        validators=[validate_notification_dispatch_json],
    )

    # ── Bucket: Aone 工单详情 ────────────────────────────────────────
    # JSONField: 工单状态、ID、幂等键、重试信息、payload 和结果
    #
    # Schema:
    #   notification_job_status: str             — 工单任务状态（pending/running/created/failed）
    #   notification_job_id: str                 — 工单 ID
    #   notification_job_idempotency_key: str    — 幂等键
    #   notification_job_attempt_count: int      — 重试次数（非负）
    #   notification_job_next_attempt_at: str | None — 下次重试时间（ISO datetime）
    #   notification_job_payload: dict            — 工单创建请求体
    #   notification_job_result: dict | None     — 工单创建结果
    #   notification_job_last_error: str | None  — 最近一次错误信息
    #   updated_at: str                   — bucket 最后更新时间（UTC+8 ISO datetime）
    notification_job_json = fields.JSONField(
        null=True,
        description="Aone 工单详情 bucket — notification_job_status/notification_job_id/idempotency_key/attempt_count/payload/result/last_error",  # noqa: E501
        validators=[validate_notification_job_json],
    )

    # ── Bucket: 归因结果标注 ─────────────────────────────────────────
    # JSONField: Label Studio 归因标注结果和人工确认信息
    #
    # Schema:
    #   attribution_result: dict    — 原始 LS 标注结果（{"raw": [...]}）
    #   human_primary_cause: str    — 人工确认的主要根因
    #   human_owner_group: str      — 人工确认的负责团队
    #   human_attribution_correct: str — 归因是否正确（yes/no/partial）
    #   annotator: str              — 标注人
    #   annotated_at: str           — 标注时间（ISO datetime）
    #   pushed_to_ls_at: str        — 推送到 LS 的时间（ISO datetime）
    #   updated_at: str             — bucket 最后更新时间（UTC+8 ISO datetime）
    label_studio_attribution_annotation_json = fields.JSONField(
        null=True,
        description="归因结果标注 bucket — attribution_result/human_primary_cause/human_owner_group/annotator/annotated_at/pushed_to_ls_at",  # noqa: E501
        validators=[validate_label_studio_attribution_annotation_json],
    )

    # ── Bucket: badcase judge结果标注 ──────────────────────────────
    # JSONField: Label Studio badcase judge标注结果和人工修正信息
    #
    # Schema:
    #   triage_result: dict                 — 原始 LS 标注结果（{"raw": [...]}）
    #   human_judge_score_correct: str      — 评分是否正确（yes/no）
    #   human_corrected_score: str          — 人工修正后的评分
    #   human_corrected_reason: str         — 修正理由
    #   human_triage_level_correct: str     — 分级是否正确（yes/no）
    #   human_corrected_triage_level: str   — 人工修正后的分级
    #   triage_annotator: str              — 标注人
    #   triage_annotated_at: str           — 标注时间（ISO datetime）
    #   pushed_to_ls_triage_at: str        — 推送到 LS 的时间（ISO datetime）
    #   updated_at: str                    — bucket 最后更新时间（UTC+8 ISO datetime）
    label_studio_badcase_judge_annotation_json = fields.JSONField(
        null=True,
        description="badcase judge结果标注 bucket — triage_result/human_judge_score_correct/human_corrected_score/triage_annotator/pushed_to_ls_triage_at",  # noqa: E501
        validators=[validate_label_studio_badcase_judge_annotation_json],
    )

    # ── Bucket: 管线元数据 / 来源追踪 ─────────────────────────────────
    # JSONField: 评测来源通道 + dataset 明细 +（历史）digest
    #
    # Schema（现行）:
    #   source: str              — 评测入口通道：
    #                              kafka | metaq | http | benchmark |
    #                              dataset-{dataset_name}-{run_name}
    #                              （由 pipeline_source → after_eval；不写裸值 dataset）
    #   dataset: dict | None     — dataset 入口附加明细（与复合 source 并存）
    #     name: str              — Langfuse dataset 名
    #     item_id: str           — dataset item id
    #     run_name: str | None   — experiment / run 名
    #   kafka: dict | None       — Judge Kafka 消息位置
    #     topic / partition / offset / key / timestamp_ms / headers
    #   input: Any | None        — 评测完整 input（EvaluatorRequest.input）
    #   output: Any | None       — 评测完整 output（EvaluatorRequest.output）
    #   updated_at: str          — bucket 最后更新时间（UTC+8 ISO datetime，service 注入）
    #
    # 写入语义（TraceRecordService.upsert）：
    #   传入 trace_meta_json 时与已有 bucket 浅合并（同名键覆盖，保留未提及键）；
    #   未传入时保留原值，不整桶清空。
    #
    # 示例:
    #   {"source": "dataset-nav/hard-r1", "dataset": {"name": "nav/hard", "item_id": "…", "run_name": "r1"},  # noqa: E501
    #    "input": "…", "output": "…"}
    #   {"source": "kafka", "kafka": {"topic": "langfuse.records", "partition": 0, "offset": 42},
    #    "input": {…}, "output": {…}}
    #   {"source": "http", "input": "…", "output": "…"}
    #   {"source": "benchmark", "input": "…", "output": "…"}
    #
    # 历史字段（平台零预拉后通常不再写入；与顶层 source 勿混淆）:
    #   digest: dict             — Langfuse trace 摘要
    #   source: dict             — 若出现在 digest 旁的旧形状：Langfuse 实例
    #                              instance/base_url/metadata_key（嵌套，非顶层通道）
    #   fetched_at: str          — 拉取时间（ISO 8601 UTC）
    #   raw_trace / signal_payload — 已废弃
    trace_meta_json = fields.JSONField(
        null=True,
        description=(
            "管线元数据 bucket — "
            "source(kafka|http|benchmark|dataset-{name}-{run}); "
            "dataset{name,item_id,run_name}; "
            "kafka{topic,partition,offset,...}; "
            "input/output(完整评测 IO); "
            "历史 digest/fetched_at"
        ),
    )

    class Meta:
        table = "trace_pipeline_records"


__all__ = ["TraceRecord", "clip_char_value"]
