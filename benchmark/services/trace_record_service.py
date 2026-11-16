"""trace 管线记录的业务服务（无状态）。

基于 ``libs/data_models`` 的 Tortoise ORM。Schema 由 ``Tortoise.generate_schemas``
在 session fixture 中创建，每个测试的 SQLite 文件由 ``conftest.py`` 重建。

Agent 元数据由 :mod:`benchmark.domain.agent_config` 从 RemoteConfig 配置拉取。

用法:
    from benchmark.services import TraceRecordService
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.constant import DEFAULT_PAGE_SIZE
from benchmark.records.enums import NotificationTaskJobStatus, TriageLevel
from benchmark.records.models import TraceRecord

logger = get_logger(__name__)

# 可通过 upsert() 的 **kwargs 传入的可选列名清单。
# 包含标量列和 bucket 字段名。
# 注意: triage_level / notification_job_status 为自动同步标量列，
# 由 TraceRecordObserver.on_save() 从 bucket 提取并通过 filter().update() 直写 DB，
# 不在此处直接传入。
_OPTIONAL_COLUMNS: tuple[str, ...] = (
    # 标量列 — 查询热路径
    "owner_group",
    "scene",
    "root_cause",
    # Bucket — JSONField（校验器由 Model 字段定义自动调用）
    "judge_json",
    "attribution_detail_json",
    "notification_dispatch_json",
    "notification_job_json",
    "label_studio_attribution_annotation_json",
    "label_studio_badcase_judge_annotation_json",
    "trace_meta_json",
)

# 重新插入时需清空的派生标量列（不在 _OPTIONAL_COLUMNS 中，
# 由 update_* 方法或 observer 维护，但在重新插入时也需清空）。
_REINSERT_CLEAR_SCALARS: tuple[str, ...] = (
    "triage_level",
    "notification_job_status",
)

# 所有 JSON bucket 字段名（用于 _stamp_bucket 自动注入 updated_at）
_BUCKET_FIELDS: frozenset[str] = frozenset(
    {
        "judge_json",
        "attribution_detail_json",
        "notification_dispatch_json",
        "notification_job_json",
        "label_studio_attribution_annotation_json",
        "label_studio_badcase_judge_annotation_json",
        "trace_meta_json",
    }
)

# UTC+8 时区
_UTC_PLUS_8 = timezone(timedelta(hours=8))


def _stamp_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    """向 bucket 注入 updated_at 时间戳（UTC+8 ISO 格式）。

    业务层不感知此字段——在 service 层每次写入 bucket 时自动调用。
    仅对非 None 的 bucket 字典注入，None 值跳过（不标记空 bucket）。
    """
    bucket["updated_at"] = datetime.now(_UTC_PLUS_8).isoformat()
    return bucket


def _merge_trace_meta(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """浅合并 ``trace_meta_json``：保留已有键，incoming 同名键覆盖。

    用于避免整桶替换抹掉历史 ``digest`` / 已有 ``source``/``dataset``。
    ``dataset`` 若传入则为整块替换（不深合并子字段）。
    """
    merged = dict(existing or {})
    merged.update(incoming)
    return _stamp_bucket(merged)


@dataclass(frozen=True)
class TraceRecordKey:
    """TraceRecord 复合业务键（FUNC-PARAMS-001 参数收敛）。

    ``(trace_id, project_id, agent_name, evaluator_name)`` 四元组唯一定位
    一条管线记录，多 Evaluator 并行评估互不覆盖。
    """

    trace_id: str
    project_id: str
    agent_name: str
    evaluator_name: str


class _CandidateRootCauseUnset(Enum):
    UNSET = auto()


@dataclass(frozen=True)
class AttributionUpdate:
    """``update_attribution`` 的归因写入内容（FUNC-PARAMS-001 参数收敛）。

    ``None`` 字段不覆盖已有 bucket 值；``root_cause`` 必填（归因结论核心）。
    ``candidate_root_cause`` 为三态：省略保留旧值，None清除，字典替换。
    """

    root_cause: str
    owner_domain: str | None = None
    owner_group: str | None = None
    confidence: float | None = None
    attribution_raw: str | None = None
    evidence_raw: str | None = None
    reasoning_raw: str | None = None
    action_hint: str | None = None
    needs_human_review: bool | None = None
    corrected_triage_level: TriageLevel | None = None
    original_triage_level: str | None = None
    triage_level: TriageLevel | None = None
    category: str | None = None
    root_cause_code: str | None = None
    all_root_cause_codes: list[str] | None = None
    conclusion_type: str | None = None
    evidence_details: list[dict[str, Any]] | None = None
    candidate_root_cause: dict[str, Any] | None | _CandidateRootCauseUnset = (
        _CandidateRootCauseUnset.UNSET
    )


@dataclass(frozen=True)
class DispatchUpdate:
    """``update_dispatch`` 的派发信息写入内容（FUNC-PARAMS-001 参数收敛）。

    ``None`` 字段不写入 bucket；``dispatch_skip_reason`` 与成功派发字段互斥。
    """

    dispatch_ref: str | None = None
    dispatch_url: str | None = None
    dispatch_assigned_to: str | None = None
    dispatch_title: str | None = None
    dispatched_at: datetime | None = None
    dispatch_skip_reason: str | None = None


@dataclass(frozen=True)
class NotificationTaskEnqueue:
    """``enqueue_aone_task`` 的入队任务描述（FUNC-PARAMS-001 参数收敛）。"""

    idempotency_key: str
    payload: dict[str, Any] = field(default_factory=dict)
    force_pending: bool = False


@dataclass(frozen=True)
class PipelineSourceQuery:
    """``list_by_pipeline_source`` 的过滤与分页参数（FUNC-PARAMS-001 参数收敛）。"""

    dataset_name: str | None = None
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE


class TraceRecordService:
    """Trace 管线记录的业务服务（无状态）。

    管理 ``trace_pipeline_records`` 表的增删改查操作。

    支持多 Evaluator 并行评估：通过复合业务键
    ``(trace_id, project_id, agent_name, evaluator_name)`` 计算确定性 PK 定位记录。

    提供:
      - Trace 增删改查: upsert（插入或更新）、按 PK 查询、按状态分页列表
      - 局部更新: update_reasoning（归因推理）、update_dispatch（派发信息）

    Agent 元数据由 :mod:`benchmark.domain.agent_config` 从 RemoteConfig 配置拉取。
    """

    # ------------------------------------------------------------------
    # TraceRecord 增删改查
    # ------------------------------------------------------------------

    async def upsert(
        self,
        key: TraceRecordKey,
        status: str,
        **kwargs: Any,
    ) -> TraceRecord:
        """插入或重新插入一条 trace 记录（三态语义）。

        通过复合业务键 ``(trace_id, project_id, agent_name, evaluator_name)``
        定位记录。

        三态数据操作语义:

        1. **首次插入**: 记录不存在，创建新记录，写入本次传入的字段。
        2. **局部更新**: 通过 ``update_*`` 系列方法更新指定字段，
           不影响其他已写入的数据域。
        3. **重新插入**: 记录已存在且通过 ``upsert`` 再次调用时，
           只保留本次传入字段的值，未传入的所有数据域置为空
           （``scene`` 重置为默认值 ``"non_voice"``）。
           ``root_cause`` 可通过 ``upsert`` kwargs 传入，传入时覆盖，
           未传入时清空为 None。
           **例外** ``trace_meta_json``：传入时与已有 bucket **浅合并**
           （保留已有 ``source``/``dataset``/``digest`` 等键，同名键覆盖）；
           未传入时保留原值，不整桶清空。

        示例::

            upsert(a, b, c, d)       → DB: a, b, c, d
            update_attribution(e)    → DB: a, b, c, d, e
            upsert(a, b, c, d1)      → DB: a, b, c, d1  (e 被清空)

        所有 bucket（JSONField）字段校验由 Model 字段定义中的 ``validators``
        自动执行，无需在 service 层显式调用。
        """

        existing = await TraceRecord.filter(
            trace_id=key.trace_id,
            project_id=key.project_id,
            agent_name=key.agent_name,
            evaluator_name=key.evaluator_name,
        ).first()
        if existing is not None:
            # 重新插入语义：更新传入字段，清空未传入的数据域。
            # 例外：trace_meta_json 写入为 merge；未传入时保留已有（不整桶清空）。
            existing.status = status
            for col in _OPTIONAL_COLUMNS:
                if col in kwargs:
                    value = self._prepare_optional_column(
                        col,
                        kwargs[col],
                        existing=existing,
                    )
                    setattr(existing, col, value)
                elif col == "scene":
                    # scene 有 NOT NULL 约束，重置为默认值
                    existing.scene = "non_voice"
                elif col == "trace_meta_json":
                    pass  # 保留已有 source/dataset/digest
                else:
                    setattr(existing, col, None)
            # 清空由 update_* / observer 维护的派生标量列
            for col in _REINSERT_CLEAR_SCALARS:
                setattr(existing, col, None)
            await existing.save()
            return existing
        fields: dict[str, Any] = {
            "trace_id": key.trace_id,
            "project_id": key.project_id,
            "agent_name": key.agent_name,
            "evaluator_name": key.evaluator_name,
            "status": status,
        }
        for col in _OPTIONAL_COLUMNS:
            if col in kwargs:
                fields[col] = self._prepare_optional_column(
                    col,
                    kwargs[col],
                    existing=None,
                )
        try:
            return await TraceRecord.create(**fields)
        except Exception as exc:
            # Race condition during Kafka rebalance: another node created
            # the record first (unique_together constraint violation).
            # Fall back to update instead of propagating the error, which
            # would trigger unnecessary backoff retry.
            existing = await TraceRecord.filter(
                trace_id=key.trace_id,
                project_id=key.project_id,
                agent_name=key.agent_name,
                evaluator_name=key.evaluator_name,
            ).first()
            if existing is None:
                raise  # Not a race condition — re-raise original error
            logger.warning(
                "upsert race condition resolved, retrying as update: "
                "trace_id=%s evaluator_name=%s error=%s",
                key.trace_id,
                key.evaluator_name,
                exc,
            )
            existing.status = status
            for col in _OPTIONAL_COLUMNS:
                if col in kwargs:
                    value = self._prepare_optional_column(
                        col,
                        kwargs[col],
                        existing=existing,
                    )
                    setattr(existing, col, value)
                elif col == "scene":
                    existing.scene = "non_voice"
                elif col == "trace_meta_json":
                    pass
                else:
                    setattr(existing, col, None)
            for col in _REINSERT_CLEAR_SCALARS:
                setattr(existing, col, None)
            await existing.save()
            return existing

    async def enqueue_notification_task(
        self,
        *,
        record_id: str,
        task: NotificationTaskEnqueue,
    ) -> TraceRecord | None:
        """入队一条 Aone 工单创建任务。

        在指定 TraceRecord 的 ``notification_job_json`` bucket 中设置 ``notification_job_status=PENDING``
        并写入工单详情。如需重复创建 Aone 工单，每次入队应
        传入新的 ``idempotency_key``；同一 key + ``force_pending=True`` 时
        强制重置为 PENDING（重试场景）。
        """
        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None

        notification_job = dict(record.notification_job_json or {})
        existing_key = notification_job.get("notification_job_idempotency_key")

        if existing_key == task.idempotency_key:
            current_status = notification_job.get("notification_job_status")
            if task.force_pending and current_status != NotificationTaskJobStatus.CREATED:
                notification_job["notification_job_status"] = NotificationTaskJobStatus.PENDING
                notification_job["notification_job_next_attempt_at"] = None
                notification_job["notification_job_payload"] = task.payload
                notification_job["notification_job_result"] = None
                notification_job["notification_job_last_error"] = None
                record.notification_job_json = _stamp_bucket(notification_job)
                await record.save()
            return record

        job_id = f"aone_job_{uuid.uuid4().hex}"
        notification_job["notification_job_status"] = NotificationTaskJobStatus.PENDING
        notification_job["notification_job_id"] = job_id
        notification_job["notification_job_idempotency_key"] = task.idempotency_key
        notification_job["notification_job_attempt_count"] = 0
        notification_job["notification_job_next_attempt_at"] = None
        notification_job["notification_job_payload"] = task.payload
        notification_job["notification_job_result"] = None
        notification_job["notification_job_last_error"] = None
        record.notification_job_json = _stamp_bucket(notification_job)
        await record.save()
        return record

    async def get_by_trace_id(
        self,
        key: TraceRecordKey,
    ) -> TraceRecord | None:
        return await TraceRecord.filter(
            trace_id=key.trace_id,
            project_id=key.project_id,
            agent_name=key.agent_name,
            evaluator_name=key.evaluator_name,
        ).first()

    async def list_by_status(
        self,
        status: str,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> tuple[list[TraceRecord], int]:
        qs = TraceRecord.filter(status=status).order_by("-updated_at")
        total = await qs.count()
        offset = (page - 1) * page_size
        return (
            list(await qs.offset(offset).limit(page_size)),
            total,
        )

    async def list_by_pipeline_source(
        self,
        source: str,
        *,
        query: PipelineSourceQuery | None = None,
    ) -> tuple[list[TraceRecord], int]:
        """按 ``trace_meta_json.source`` 分页列出管线记录。

        ``source`` 取值：``kafka`` / ``metaq`` / ``http`` / ``dataset`` /
        ``benchmark`` / ``sidecar``。
        查询 ``dataset`` 时只匹配复合值 ``dataset-{name}-{run}``（``LIKE 'dataset-%'``），
        **不**匹配裸值 ``dataset`` 或历史 ``dataset+…``。
        可选再按 ``trace_meta_json.dataset.name`` 过滤。

        实现为内存过滤（SQLite / PG 通用）；生产库可用表达式索引::

            WHERE (
              (:source <> 'dataset' AND trace_meta_json->>'source' = :source)
              OR (:source = 'dataset'
                  AND trace_meta_json->>'source' LIKE 'dataset-%')
            )
              AND (:dataset_name IS NULL
                   OR trace_meta_json->'dataset'->>'name' = :dataset_name)
        """
        from benchmark.domain.enums import (  # noqa: PLC0415
            PipelineSource,
            is_dataset_pipeline_source,
        )

        q = query or PipelineSourceQuery()
        wanted = source.strip().lower()
        matched: list[TraceRecord] = []
        async for rec in TraceRecord.all().order_by("-updated_at"):
            meta = rec.trace_meta_json or {}
            src = str(meta.get("source") or "")
            if wanted == PipelineSource.DATASET.value:
                if not is_dataset_pipeline_source(src):
                    continue
            elif src.lower() != wanted:
                continue
            if q.dataset_name is not None:
                ds = meta.get("dataset") if isinstance(meta.get("dataset"), dict) else {}
                if (ds or {}).get("name") != q.dataset_name:
                    continue
            matched.append(rec)
        total = len(matched)
        offset = (q.page - 1) * q.page_size
        return matched[offset : offset + q.page_size], total

    async def list_dataset_records(
        self,
        *,
        dataset_name: str | None = None,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> tuple[list[TraceRecord], int]:
        """``list_by_pipeline_source('dataset', ...)`` 的便捷封装。"""
        return await self.list_by_pipeline_source(
            "dataset",
            query=PipelineSourceQuery(
                dataset_name=dataset_name,
                page=page,
                page_size=page_size,
            ),
        )

    async def update_triage_level(
        self,
        *,
        record_id: str,
        triage_level: TriageLevel,
    ) -> TraceRecord | None:
        """分流阶段写入：直写 ``triage_level`` 标量列 + merge 进 ``judge_json`` bucket。

        标量列是查询热路径（WHERE/索引），由本方法直接写入；
        ``judge_json.triage_level`` 同步一份保留完整原始记录。
        原 bucket→标量 observer 同步已移除，标量列不再从 bucket 派生。

        强类型校验：``triage_level`` 必须是 ``TriageLevel`` 枚举（service 层
        主动 ``TriageLevel(...)`` coerce；传非法字符串会抛 ``ValueError``）。
        """
        # 主动 enum 校验：Tortoise CharEnumField 在 save 时不校验，
        # 只在从 DB 读出时 coerce → 必须在 service 层兜底。
        triage_level = TriageLevel(triage_level)

        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None
        record.triage_level = triage_level
        judge = dict(record.judge_json or {})
        judge["triage_level"] = triage_level.value
        record.judge_json = _stamp_bucket(judge)
        await record.save()
        return record

    async def update_attribution(
        self,
        *,
        record_id: str,
        update: AttributionUpdate,
    ) -> TraceRecord | None:
        """归因阶段写入：直写 ``root_cause`` / ``owner_group`` 标量列 + merge bucket。

        标量列是查询热路径，由本方法直接写入；bucket 保留完整归因原始记录。
        ``owner_group`` 来自 :func:`domain.owner_mapping.resolve_owner_route` 解析结果，
        与 ``root_cause`` 同阶段写入。

        ``original_triage_level`` 仅在归因推翻时传入，保留 classifier 原始分级，
        避免被 ``ATTRIBUTION_OVERRIDDEN`` 覆盖后不可追溯。

        ``triage_level`` 为归因推翻后的最终分级（``ATTRIBUTION_OVERRIDDEN``），与
        bucket **同一次 UPDATE** 落库。归因结论与分级是一次决策，拆成两条写会在
        中途被取消/并发覆盖时留下「bucket 已推翻、标量仍是原分级」的不一致行。

        ``category`` / ``root_cause_code`` / ``all_root_cause_codes`` /
        ``conclusion_type`` / ``needs_human_review`` / ``evidence_details`` /
        ``candidate_root_cause`` / ``action_hint`` 为聚合报告消费的结构化字段；
        candidate 仅保存 `other:` 的结构化投影，不参与 owner 路由。
        随归因结果一并 merge 进 bucket（validator 无字段白名单，新键安全）。
        ``owner_domain`` 历史上存的是 category 值，保留兼容旧读取方；新消费方
        一律读独立的 ``category`` 键。

        完成时同步将 ``attribution_status`` 置为 ``completed``，覆盖受理阶段的
        ``pending`` 标记（见 :meth:`mark_attribution_pending`）。

        ``reasoning_raw`` / ``evidence_raw`` / ``action_hint`` / ``needs_human_review``
        与 ``AttributionResult`` 对齐；未传的 key 不覆盖已有值。
        ``update_reasoning`` 仍可用于单独补写推理。

        原 bucket→标量 observer 同步已移除，``root_cause`` / ``owner_group`` 标量
        不再从 bucket 派生。
        """

        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None
        # 标量可赋全文；TraceRecord.save() 按字段 meta.max_length 截断。
        # bucket 先写入全文，截断只影响标量列，避免归因整行写库失败。
        record.root_cause = update.root_cause
        if update.owner_group is not None:
            record.owner_group = update.owner_group
        bucket = dict(record.attribution_detail_json or {})
        bucket["root_cause"] = update.root_cause
        if update.owner_domain is not None:
            bucket["owner_domain"] = update.owner_domain
        if update.owner_group is not None:
            bucket["owner_group"] = update.owner_group
        if update.confidence is not None:
            bucket["confidence"] = update.confidence
        if update.attribution_raw is not None:
            bucket["attribution_raw"] = update.attribution_raw
        if update.evidence_raw is not None:
            bucket["evidence_raw"] = update.evidence_raw
        if update.reasoning_raw is not None:
            bucket["reasoning_raw"] = update.reasoning_raw
        if update.action_hint is not None:
            bucket["action_hint"] = update.action_hint
        if update.needs_human_review is not None:
            bucket["needs_human_review"] = update.needs_human_review
        if update.corrected_triage_level is not None:
            bucket["corrected_triage_level"] = TriageLevel(update.corrected_triage_level).value
        if update.original_triage_level is not None:
            bucket["original_triage_level"] = update.original_triage_level
        if update.category is not None:
            bucket["category"] = update.category
        if update.root_cause_code is not None:
            bucket["root_cause_code"] = update.root_cause_code
        if update.all_root_cause_codes is not None:
            bucket["all_root_cause_codes"] = update.all_root_cause_codes
        if update.conclusion_type is not None:
            bucket["conclusion_type"] = update.conclusion_type
        if update.evidence_details is not None:
            bucket["evidence_details"] = update.evidence_details
        # 回放省略字段时保留已有候选；正式归因显式None时清除。
        if update.candidate_root_cause is not _CandidateRootCauseUnset.UNSET:
            bucket["candidate_root_cause"] = update.candidate_root_cause
        # 受理阶段的 pending 标记在完整结果落库时关闭；三态判定以
        # root_cause 存在为完成基准，此处仅保持标记语义自洽。
        bucket["attribution_status"] = "completed"
        record.attribution_detail_json = _stamp_bucket(bucket)
        if update.triage_level is not None:
            # 与 update_triage_level 同源：标量 + judge_json 同步一份。
            triage_level = TriageLevel(update.triage_level)
            record.triage_level = triage_level
            judge = dict(record.judge_json or {})
            judge["triage_level"] = triage_level.value
            record.judge_json = _stamp_bucket(judge)
        await record.save()
        return record

    async def mark_attribution_pending(
        self,
        *,
        record_id: str,
    ) -> TraceRecord | None:
        """归因受理标记：triage 放行后、归因 agent 跑完前落 ``attribution_status=pending``。

        merge 式局部更新（读旧 bucket → merge → save），**不走 upsert**——upsert
        是重插语义，会清空未传入的其他 bucket（如 judge_json）。

        键名用 ``attribution_status`` 而非 ``status``，避免与 ``TraceRecord.status``
        标量列（TraceStatus 枚举）混淆；本标记不影响标量列。

        三态消费口径（聚合 progress）：
          未触发 = 无 attribution_status 且无 root_cause；
          进行中 = attribution_status == "pending" 且无 root_cause；
          已完成 = root_cause 存在；失败 = attribution_status == "failed"。
        """

        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None
        bucket = dict(record.attribution_detail_json or {})
        bucket["attribution_status"] = "pending"
        record.attribution_detail_json = _stamp_bucket(bucket)
        await record.save()
        return record

    async def mark_attribution_failed(self, *, record_id: str) -> TraceRecord | None:
        """结束未完成的pending；已成功落库的completed保持不变。"""
        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None
        bucket = dict(record.attribution_detail_json or {})
        if bucket.get("attribution_status") == "pending":
            bucket["attribution_status"] = "failed"
            record.attribution_detail_json = _stamp_bucket(bucket)
            await record.save()
        return record

    async def update_reasoning(
        self,
        *,
        record_id: str,
        reasoning_json: str,
    ) -> TraceRecord | None:
        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None
        bucket = dict(record.attribution_detail_json or {})
        bucket["reasoning_raw"] = reasoning_json
        record.attribution_detail_json = _stamp_bucket(bucket)
        await record.save()
        return record

    async def update_dispatch(
        self,
        *,
        record_id: str,
        update: DispatchUpdate,
    ) -> TraceRecord | None:
        """更新 ``notification_dispatch_json`` bucket 中的派发信息。

        ``dispatch_skip_reason`` 仅在派发被 agent 主动跳过等场景写入，
        与 ``dispatch_ref`` / ``dispatch_title`` 互斥（成功派发时留空）。
        """

        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None
        dispatch = dict(record.notification_dispatch_json or {})
        if update.dispatch_ref is not None:
            dispatch["notification_ref"] = update.dispatch_ref
        if update.dispatch_url is not None:
            dispatch["notification_url"] = update.dispatch_url
        if update.dispatch_assigned_to is not None:
            dispatch["notification_assigned_to"] = update.dispatch_assigned_to
        if update.dispatch_title is not None:
            dispatch["notification_title"] = update.dispatch_title
        if update.dispatched_at is not None:
            dispatch["notification_dispatched_at"] = update.dispatched_at.isoformat()
        if update.dispatch_skip_reason is not None:
            dispatch["notification_skip_reason"] = update.dispatch_skip_reason
        record.notification_dispatch_json = _stamp_bucket(dispatch)
        await record.save()
        return record

    async def update_ls_push_time(
        self,
        *,
        record_id: str,
    ) -> TraceRecord | None:
        """LS 推送成功后记录 pushed_at 到两个 bucket。"""
        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None
        now_iso = datetime.now(UTC).isoformat()
        attr_bucket = dict(record.label_studio_attribution_annotation_json or {})
        attr_bucket["pushed_to_ls_at"] = now_iso
        record.label_studio_attribution_annotation_json = attr_bucket
        judge_bucket = dict(record.label_studio_badcase_judge_annotation_json or {})
        judge_bucket["pushed_to_ls_triage_at"] = now_iso
        record.label_studio_badcase_judge_annotation_json = judge_bucket
        await record.save()
        return record

    async def update_ls_annotation(
        self,
        *,
        record_id: str,
        judge_annotation_data: dict[str, Any] | None = None,
        attribution_annotation_data: dict[str, Any] | None = None,
    ) -> TraceRecord | None:
        """LS webhook 回调后将标注结果同时写入两个 bucket。

        last-write-wins：比较新数据 ``annotated_at`` 与已有 bucket
        中的 ``triage_annotated_at`` / ``annotated_at``，旧事件跳过。
        """
        record = await TraceRecord.filter(id=record_id).first()
        if record is None:
            return None
        if judge_annotation_data:
            if not self._is_stale_annotation(
                record.label_studio_badcase_judge_annotation_json,
                judge_annotation_data,
                timestamp_key="triage_annotated_at",
            ):
                bucket = dict(record.label_studio_badcase_judge_annotation_json or {})
                bucket.update(judge_annotation_data)
                record.label_studio_badcase_judge_annotation_json = _stamp_bucket(bucket)
        if attribution_annotation_data:
            if not self._is_stale_annotation(
                record.label_studio_attribution_annotation_json,
                attribution_annotation_data,
                timestamp_key="annotated_at",
            ):
                bucket = dict(record.label_studio_attribution_annotation_json or {})
                bucket.update(attribution_annotation_data)
                record.label_studio_attribution_annotation_json = _stamp_bucket(bucket)
        await record.save()
        return record

    @staticmethod
    def _is_stale_annotation(
        existing_bucket: dict[str, Any] | None,
        new_data: dict[str, Any],
        *,
        timestamp_key: str,
    ) -> bool:
        """判断新 annotation 是否比已有 bucket 更旧（乱序事件）。"""
        if not existing_bucket:
            return False
        existing_ts = existing_bucket.get(timestamp_key)
        new_ts = new_data.get(timestamp_key)
        if not existing_ts or not new_ts:
            return False
        try:
            existing_dt = datetime.fromisoformat(str(existing_ts).replace("Z", "+00:00"))
            new_dt = datetime.fromisoformat(str(new_ts).replace("Z", "+00:00"))
            return new_dt < existing_dt
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _prepare_optional_column(
        col: str,
        value: Any,
        *,
        existing: TraceRecord | None,
    ) -> Any:
        """upsert 可选列预处理：bucket stamp / trace_meta 浅合并。

        CharField 超长截断由 :meth:`TraceRecord.save` 按 ``_meta.max_length`` 处理。
        """
        if col == "trace_meta_json" and value is not None:
            prior = None
            if existing is not None and isinstance(existing.trace_meta_json, dict):
                prior = existing.trace_meta_json
            return _merge_trace_meta(prior, dict(value))
        if col in _BUCKET_FIELDS and value is not None:
            return _stamp_bucket(dict(value))
        return value


__all__ = ["TraceRecordService"]
