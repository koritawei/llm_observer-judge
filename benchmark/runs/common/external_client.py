"""Aone pipeline HTTP 客户端 — Rubric 打分 + 归因 Pipeline 封装。

基于 ``agent_eval_调用_ieval.md`` / ``async-attribution-design.md``：
- Endpoint: ``POST /api/pipelines/{pipeline_id}/trigger``
- Headers: ``X-API-Key``, ``Content-Type: application/json``
- Rubric body: ``{"params": {"records": [...], "ds": "...", ...}}``
- Attribution body: ``{"params": {"adiu": "...", "ds_start": "...",
  "judge_result": {...}, "issue": {...}, ...}}``
- Response: ``{"job_id": "...", "celery_task_id": "..."}``
- Job status: ``GET /api/jobs/{job_id}``

只处理 Aone pipeline HTTP 交互（trigger / job status），
不包含业务层 record 构建（``build_record`` 留在各业务子包）。

从 :mod:`business.runs.ieval.aone_client` 提取，
供后续服务实现直接复用。
"""

import json
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field

from benchmark.utils.http_transport import AsyncClient, HTTPError

logger = get_logger("agent_eval.benchmark.common.aone_client")

#: Aone pipeline trigger timeout (async submit, not waiting for result).
NOTIFICATION_SUBMIT_TIMEOUT = 30.0

#: job 查询 404 视为 not_found 终态（不发 raise_for_status）。
_HTTP_NOT_FOUND = 404

#: HTTP 响应用于日志时截断，避免 params.records / tool_calls 刷屏。
_LOG_BODY_LIMIT = 2000

#: 请求体日志脱敏字段 — 用户输入 / 模型输出 / 评测明细，可能含隐私数据。
_SENSITIVE_LOG_FIELDS = frozenset(
    {"query", "response", "records", "judge_result", "issue", "full_tool_calls_json"}
)

#: 脱敏递归深度上限 — Pydantic 请求体来自类型化的 AoneRubricParams /
#: AoneAttributionParams（结构浅、层级可控），正常请求远达不到此深度；
#: 超限视为异常输入（自引用环 / 意外深嵌套），整体占位不再下钻，
#: 防御极端场景下的深递归开销与栈溢出。
_REDACT_MAX_DEPTH = 32


def _clip_body_for_log(text: str, limit: int = _LOG_BODY_LIMIT) -> str:
    """截断 HTTP body 用于日志，保留长度便于对照完整响应。"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated, original_chars={len(text)}]"


def _placeholder(value: object) -> str:
    """敏感值的脱敏占位符，保留长度信息便于排查（如确认 query 非空）。"""
    if isinstance(value, str):
        return f"<redacted chars={len(value)}>"
    if isinstance(value, list):
        return f"<redacted items={len(value)}>"
    if isinstance(value, dict):
        return f"<redacted keys={len(value)}>"
    return "<redacted>"


def _redact_for_log(value: object, field: str = "", depth: int = 0) -> object:
    """递归脱敏请求体用于日志：敏感字段替换为带长度的占位符，其余结构保留。

    ``depth`` 超过 ``_REDACT_MAX_DEPTH`` 时整体占位（防御自引用环 /
    异常深嵌套，避免栈溢出）；正常 Pydantic 请求体远达不到该深度。
    """
    if field in _SENSITIVE_LOG_FIELDS:
        return _placeholder(value)
    if depth >= _REDACT_MAX_DEPTH:
        return _placeholder(value)
    if isinstance(value, dict):
        return {k: _redact_for_log(v, k, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_for_log(v, depth=depth + 1) for v in value]
    return value


# ════════════════════════════════════════════════════════════════════════════
# Request / Response models (matching agent_eval_调用_ieval.md §2)
# ════════════════════════════════════════════════════════════════════════════


class ExternalRubricParams(BaseModel):
    """Rubric 打分 pipeline 请求参数（文档 §2 params）。

    ``run_id`` / ``task_id`` / ``job_id`` / ``item_id`` 为透传字段，
    回调时原样回传，用于外部对账。
    """

    records: list[dict[str, Any]] = Field(description="case 列表，每条须含 task_id")
    ds: str = Field(description="ODPS 分区日 YYYYMMDD")
    dims: str = Field(default="D0,D1,D2,D3,D4,D5,D6", description="评分维度")
    prompt_version: str = Field(default="v2", description="prompt 版本")
    concurrency: int = Field(default=2, description="打分并发")
    limit: int | None = Field(default=None, description="最多评测条数")
    # 透传字段（回调时原样回传，用于外部对账）
    run_id: str | None = Field(default=None, description="回调时原样回传")
    task_id: str | None = Field(default=None, description="回调时原样回传")
    job_id: str | None = Field(default=None, description="回调时原样回传")
    item_id: str | None = Field(default=None, description="回调时原样回传")


class ExternalRubricTriggerRequest(BaseModel):
    """Rubric 打分 pipeline 触发请求体。"""

    params: ExternalRubricParams


class ExternalAttributionParams(BaseModel):
    """归因 Pipeline 触发 params（对齐 ``async-attribution-design.md`` §4.2）。

    ``query`` / ``response`` 为 pipeline 必填字段；
    ``judge_result`` 与 ``issue`` 同级：完整 judge + 精简 CaseIssue。
    """

    # pipeline 必填业务字段（§4.2）
    query: str = Field(default="", description="用户 query（pipeline 必填）")
    response: str = Field(default="", description="模型回复文本（pipeline 必填）")
    # pipeline 可选业务字段
    full_tool_calls_json: str | None = Field(
        default=None,
        description="工具调用记录 JSON",
    )
    # 归因上下文
    adiu: str = Field(description="设备 ID")
    ds_start: str = Field(description="开始日 YYYYMMDD")
    ds_end: str | None = Field(default=None, description="结束日，缺省=ds_start")
    trace_id: str | None = Field(default=None, description="锁定某 trace")
    do_cross_verify: bool = Field(default=True, description="是否交叉验证")
    judge_result: dict[str, Any] = Field(description="完整 judge 回调体")
    issue: dict[str, Any] = Field(description="精简 CaseIssue（不满足维度）")
    # 对账字段（回调原样回传）
    run_id: str | None = Field(default=None, description="外部 run")
    task_id: str | None = Field(default=None, description="外部批次 ID")
    job_id: str | None = Field(default=None, description="外部对账 Job ID")
    item_id: str | None = Field(default=None, description="条目级 ID")
    upstream_job_id: str | None = Field(default=None, description="上游平台 job_id")
    upstream_celery_task_id: str | None = Field(
        default=None,
        description="上游 Celery task id",
    )
    upstream_pipeline: str | None = Field(
        default=None,
        description="上游 pipeline 标记",
    )


class ExternalAttributionTriggerRequest(BaseModel):
    """归因 Pipeline 触发请求体（与 TriggerIn 同形，仅 params 业务字段不同）。"""

    params: ExternalAttributionParams


class ExternalRubricTriggerResponse(BaseModel):
    """Rubric / Attribution pipeline 触发响应。

    触发成功返回 ``job_id`` 和 ``celery_task_id``，
    可用 ``job_id`` 跟踪进度；打分结果通过回调返回。

    Attribution pipeline 额外回传：
    - ``item_id``：原样回传的 ``params.task_id``（= ``f{run_id}:{item_id}``），
      用于校验归因 job_id 与 case 的对应关系
    - ``cached``：幂等命中（已完成），同 ``job_id`` 复用已有结果
    - ``deduplicated``：去重命中（pending/running），同 ``job_id`` 复用已有任务
    """

    job_id: str = Field(description="Aone 侧 job ID")
    celery_task_id: str | None = Field(default=None, description="Celery 任务 ID")
    item_id: str | None = Field(
        default=None,
        description="原样回传的 params.task_id（归因校验用）",
    )
    cached: bool = Field(default=False, description="幂等命中（completed）")
    deduplicated: bool = Field(
        default=False,
        description="去重命中（pending/running）",
    )


class ExternalJobStatusResponse(BaseModel):
    """Aone pipeline job 状态查询响应（GET /api/jobs/{job_id} 返回的 jobs 表整行）。

    healer callback timeout 时用 ``status`` 分流；``completed`` / ``failed``
    时把 ``result`` / ``error_message`` 还原成 Judge/Attribution 回调体，
    再走 ``handle_callback``（与 HTTP 回调同一路径）。
    """

    id: int | None = Field(default=None, description="自增主键")
    job_id: str = Field(description="Aone 侧 job ID")
    pipeline_id: str | None = Field(default=None, description="Pipeline ID")
    run_id: str | None = Field(default=None, description="关联 EvalRun，可空串")
    rubric_id: str | None = Field(default=None, description="可空串")
    type: str | None = Field(default=None, description="如 data_production / judge")
    status: str = Field(description="pending/running/completed/failed/cancelled")
    progress: dict[str, Any] | None = Field(default=None, description="{current, total, stage}")
    celery_task_id: str | None = Field(default=None, description="Celery 任务 ID")
    params: dict[str, Any] | None = Field(default=None, description="触发参数")
    result: dict[str, Any] | None = Field(default=None, description="终态结果摘要")
    error_message: str | None = Field(default=None, description="失败信息")
    trigger_type: str | None = Field(default=None, description="如 manual")
    parent_job_id: str | None = Field(default=None, description="父 job ID")
    created_by: str | None = Field(default=None, description="创建者")
    created_at: str | None = Field(default=None, description="时间戳")
    updated_at: str | None = Field(default=None, description="时间戳")
    # HTTP/解析层面的错误（非 job 自身状态）
    error: str | None = Field(default=None, description="查询错误信息（HTTP/解析层面）")


#: GET /jobs 仍在途（回调未终态）。healer 必须等待，不得当丢回调回插。
NOTIFICATION_IN_FLIGHT_JOB_STATUSES = frozenset({"pending", "running"})


#: base_url 兼容裁剪的路径后缀（配置可能带 /api/pipelines 结尾）。
_PIPELINES_PATH_SUFFIX = "/api/pipelines"


def normalize_pipeline_base_url(base_url: str) -> str:
    """归一化 Aone base URL：剥尾部斜杠 + 可选 ``/api/pipelines`` 后缀。

    现网配置允许 base_url 携带 ``/api/pipelines`` 后缀（如
    ``https://pre-ieval.amap-inc.com/api/pipelines``）；client 与 factory
    统一按本规则归一后，再拼 ``/api/pipelines/{id}/trigger``，避免双重拼接。
    """
    url = base_url.rstrip("/")
    if url.endswith(_PIPELINES_PATH_SUFFIX):
        url = url[: -len(_PIPELINES_PATH_SUFFIX)]
    return url


# ════════════════════════════════════════════════════════════════════════════
# HTTP Clients
# ════════════════════════════════════════════════════════════════════════════


#: trigger 请求体（judge 批量 records / attribution 单条），共用 POST 流程。
_TriggerRequest = ExternalRubricTriggerRequest | ExternalAttributionTriggerRequest


@dataclass(frozen=True)
class ExternalPipelineConfig:
    """``_AonePipelineClient`` 构造配置（FUNC-PARAMS-001 参数收敛）。"""

    base_url: str
    api_key: str
    pipeline_id: str
    timeout: float = NOTIFICATION_SUBMIT_TIMEOUT


class _ExternalPipelineClient:
    """Aone pipeline HTTP 客户端公共基座 — URL 归一 / 认证头 / trigger POST / job 查询。

    judge / attribution 两类客户端的 HTTP 交互完全同构
    （``POST /api/pipelines/{id}/trigger`` + ``GET /api/jobs/{job_id}``），
    仅 pipeline 语义与日志事件前缀不同：
    - ``_JOB_LOG_PREFIX``：job 状态查询日志前缀（子类覆盖）
    - ``_post_trigger(log_event=...)``：trigger 日志前缀（调用处声明）
    对外仍暴露语义化的 ``AoneJudgeClient`` / ``AoneAttributionClient``。
    """

    #: job 状态查询日志事件前缀（judge 默认，attribution 覆盖为 aone.attr）。
    _JOB_LOG_PREFIX = "external"

    def __init__(
        self,
        config: ExternalPipelineConfig,
    ) -> None:
        self._base_url = normalize_pipeline_base_url(config.base_url)
        self._api_key = config.api_key
        self._pipeline_id = config.pipeline_id
        self._timeout = config.timeout

    async def _post_trigger(
        self,
        request: _TriggerRequest,
        *,
        log_event: str,
        record_count: int | None = None,
    ) -> ExternalRubricTriggerResponse:
        """POST trigger 请求体（脱敏日志 + 非 2xx raise + 响应解析）。

        ``log_event`` 为日志事件前缀（如 ``aone`` / ``aone.attribution``）；
        ``record_count`` 仅批量提交时携带，日志多一个 ``records=N`` 对账字段。
        """
        url = f"{self._base_url}/api/pipelines/{self._pipeline_id}/trigger"
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }
        body = request.model_dump(exclude_none=True)
        if record_count is not None:
            logger.info(
                "%s.http_request url=%s method=POST "
                "headers={X-API-Key: ***, Content-Type: application/json} "
                "records=%d body=%s",
                log_event,
                url,
                record_count,
                json.dumps(_redact_for_log(body), ensure_ascii=False),
            )
        else:
            logger.info(
                "%s.http_request url=%s method=POST "
                "headers={X-API-Key: ***, Content-Type: application/json} "
                "body=%s",
                log_event,
                url,
                json.dumps(_redact_for_log(body), ensure_ascii=False),
            )
        async with AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
            logger.info(
                "%s.http_response status=%d headers=%s body=%s",
                log_event,
                resp.status_code,
                dict(resp.headers),
                _clip_body_for_log(resp.text),
            )
            resp.raise_for_status()
            data = resp.json()
            return ExternalRubricTriggerResponse.model_validate(data)

    async def query_job_status(self, job_id: str) -> ExternalJobStatusResponse:
        """查询 Aone pipeline job 状态（GET /api/jobs/{job_id}）。

        返回 ``AoneJobStatusResponse``，含 jobs 表整行字段。
        状态主要读 ``status``；进度看 ``progress``；
        终态结果看 ``result`` / ``error_message``。

        Raises:
            httpx.HTTPError: 网络错误（非 404）
        """
        url = f"{self._base_url}/api/jobs/{job_id}"
        headers = {
            "X-API-Key": self._api_key,
            "Accept": "application/json",
        }
        logger.info(
            "%s.job_status_query url=%s method=GET job_id=%s",
            self._JOB_LOG_PREFIX,
            url,
            job_id,
        )
        try:
            async with AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, headers=headers)
                logger.info(
                    "%s.job_status_response status=%d body=%s",
                    self._JOB_LOG_PREFIX,
                    resp.status_code,
                    _clip_body_for_log(resp.text),
                )
                if resp.status_code == _HTTP_NOT_FOUND:
                    return ExternalJobStatusResponse(
                        job_id=job_id,
                        status="not_found",
                        error=f"job not found (404): {job_id}",
                    )
                resp.raise_for_status()
                data = resp.json()
                return ExternalJobStatusResponse.model_validate(data)
        except HTTPError:
            logger.warning(
                "%s.job_status_query_failed job_id=%s",
                self._JOB_LOG_PREFIX,
                job_id,
                exc_info=True,
            )
            raise


class ExternalJudgeClient(_ExternalPipelineClient):
    """Aone Rubric 打分 pipeline HTTP 客户端（judge 提交）。

    基于文档 ``agent_eval_调用_ieval.md``:
    - ``POST /api/pipelines/{pipeline_id}/trigger``
    - ``X-API-Key`` 认证
    - 异步接口，立即返回 ``job_id``
    """

    async def submit_evaluation(
        self,
        request: ExternalRubricTriggerRequest,
    ) -> ExternalRubricTriggerResponse:
        """提交 Rubric 打分到 Aone pipeline。

        Raises:
            httpx.HTTPError: 网络错误 / Aone 返回非 2xx
        """
        return await self._post_trigger(request, log_event="external")


class ExternalAttributionClient(_ExternalPipelineClient):
    """Aone Rubric 打分 pipeline HTTP 客户端（归因提交）。

    attribution-only 模式同样通过 Rubric 打分 pipeline 提交，
    区别在于透传字段和回调处理方式。
    """

    #: 归因 job 状态查询日志前缀（沿用既有 aone.attr.* 事件名）。
    _JOB_LOG_PREFIX = "aone.attr"

    async def submit_attribution(
        self,
        request: ExternalAttributionTriggerRequest,
    ) -> ExternalRubricTriggerResponse:
        """提交单条归因到 Aone attribution pipeline。

        Body 对齐 ``async-attribution-design.md`` §4.2：
        ``params.judge_result`` + ``params.issue`` 同级对象（非 Rubric records）。

        Raises:
            httpx.HTTPError: 网络错误 / Aone 返回非 2xx
        """
        return await self._post_trigger(request, log_event="aone.attribution")

    async def submit_batch_attribution(
        self,
        request: ExternalRubricTriggerRequest,
    ) -> ExternalRubricTriggerResponse:
        """提交整体归因到 Aone pipeline（attribution-only 模式）。

        全量 items 作为一个 Aone 任务提交，
        Aone 处理完成后发送回调。

        Raises:
            httpx.HTTPError: 网络错误 / Aone 返回非 2xx
        """
        return await self._post_trigger(
            request,
            log_event="aone.attribution_batch",
            record_count=len(request.params.records),
        )


__all__ = [
    "NOTIFICATION_IN_FLIGHT_JOB_STATUSES",
    "NOTIFICATION_SUBMIT_TIMEOUT",
    "ExternalJudgeClient",
    "ExternalAttributionClient",
    "ExternalAttributionParams",
    "ExternalAttributionTriggerRequest",
    "ExternalJobStatusResponse",
    "ExternalPipelineConfig",
    "ExternalRubricParams",
    "ExternalRubricTriggerRequest",
    "ExternalRubricTriggerResponse",
    "normalize_pipeline_base_url",
]
