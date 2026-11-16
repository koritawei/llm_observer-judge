"""评测控制台文档接口。

接受 agent-park 页面上的数据集、评测代理和评测任务。
不调用 benchmark 运行、数据集评测或流式任务。

例外：``POST /eval-tasks/{id}/stop`` 联动终止批量 run（见
:func:`_terminate_manual_run`）——批量任务的执行面是 benchmark run +
BullMQ job，只改任务文档不停 run 会让评测继续执行。
"""

import json
import logging
from dataclasses import dataclass
from typing import Annotated, Any, NoReturn, TypeVar

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from benchmark.domain.benchmark_console import (
    AgentDocument,
    AgentScopeDocument,
    ConsoleError,
    DatasetDocument,
    EvalTaskDocument,
    apply_remove_task,
    apply_rerun_task,
    apply_retry_task,
    apply_start_sampling,
    apply_stop_collection,
    apply_stop_sampling,
    apply_stop_task,
    apply_update_sampling_ratio,
    delete_agent,
    delete_dataset,
    delete_scope,
    get_agent,
    get_dataset,
    get_task,
    list_agents,
    list_all_agent_registrations,
    list_datasets,
    list_scopes,
    list_tasks,
    page_rubrics,
    save_agent,
    save_dataset,
    save_scope,
    save_task,
)
from benchmark.domain.customization_config import fetch_customization_config
from benchmark.libs.jobqueue.job_listing import list_queue_jobs
from benchmark.runs.eval_count_enrichment import (
    enrich_agent_evaluation_counts,
    enrich_realtime_bound_agent_counts,
)
from benchmark.runs.run_lifecycle import ListRunsQuery
from benchmark.runs.run_stop import (
    FinalizeStoppedWrite,
    align_expectations_for_stop,
    count_running_benchmark_items,
    finalize_as_stopped,
    inspect_stop_report_readiness,
    mark_stop_requested,
    publish_stop_signal,
    stop_items_ready_for_report,
)
from benchmark.runs.stop_inflight import (
    JOB_TYPES_PARKED,
    STOP_RUNNING_ALL_QUEUES,
    count_stop_inflight_jobs,
    is_stop_inflight_job,
    job_belongs_to_run,
)
from benchmark.customization.errors import CustomizationError
from benchmark.customization.git_store import GitContentStore
from benchmark.customization.manifest import CustomizationKey, stage_base_path
from benchmark.queue.inflight_slot import (
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.queue_factory import (
    ATTRIBUTION_WAIT,
    BENCHMARK_CASES,
    EXTERNAL_WAIT,
)
from benchmark.workers.finalize_trigger import (
    try_finalize_benchmark,
    try_finalize_external,
)

router = APIRouter(prefix="/v1/runtime-config/console", tags=["benchmark-console"])

logger = logging.getLogger("agent_eval.api.benchmark_console")

#: stop-run 只删除 **wait** 中未开始的 job；**running 在途保留**，等结果
#: 后再出阶段性报告（避免 drain running → item 停在 running 预写 → report 失败）。
_JOB_TYPES_ALL = ["waiting", "delayed", "active", "paused", "prioritized"]
_STOP_WAIT_QUEUE_MAP: tuple[tuple[str, list[str], str], ...] = (
    (EXTERNAL_WAIT, _JOB_TYPES_ALL, ""),
    (BENCHMARK_CASES, _JOB_TYPES_ALL, ""),
    (ATTRIBUTION_WAIT, _JOB_TYPES_ALL, ""),
)


@router.get("/agent-scopes")
async def read_scopes() -> list[dict[str, Any]]:
    """列出自定义评测对象（内置项页面预置，不在此返回）。"""
    return await list_scopes()


@router.put("/agent-scopes")
async def write_scope(body: dict[str, Any]) -> dict[str, Any]:
    """保存一个自定义评测对象。请求体与页面对象一致，不触发评测链路。"""
    document = _parse(AgentScopeDocument, body)
    try:
        return await save_scope(document)
    except ConsoleError as exc:
        _raise_console(exc)


@router.delete("/agent-scopes/{scope_id}")
async def remove_scope(scope_id: str) -> dict[str, Any]:
    """删除一个自定义评测对象（内置项不可删）。"""
    try:
        return await delete_scope(scope_id)
    except ConsoleError as exc:
        _raise_console(exc)


def _raise_console(exc: ConsoleError) -> NoReturn:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


DocumentT = TypeVar("DocumentT", bound=BaseModel)


def _parse(model: type[DocumentT], body: object) -> DocumentT:
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc


@router.get("/datasets")
async def read_datasets(
    agent_scope: str | None = Query(default=None, alias="agentScope"),
) -> list[dict[str, Any]]:
    """列出数据集。"""
    try:
        items = await list_datasets(agent_scope)
    except ConsoleError as exc:
        _raise_console(exc)
    await enrich_realtime_bound_agent_counts(items)
    return items


@router.put("/datasets")
async def write_dataset(body: dict[str, Any]) -> dict[str, Any]:
    """保存一条数据集。请求体与页面数据集对象一致。"""
    document = _parse(DatasetDocument, body)
    try:
        return await save_dataset(document)
    except ConsoleError as exc:
        _raise_console(exc)


@router.get("/datasets/{dataset_id}")
async def read_dataset(dataset_id: str) -> dict[str, Any]:
    """读取一条数据集。"""
    try:
        return await get_dataset(dataset_id)
    except ConsoleError as exc:
        _raise_console(exc)


@router.delete("/datasets/{dataset_id}")
async def remove_dataset(dataset_id: str) -> dict[str, Any]:
    """删除一条数据集；realtime 同步 retire evaluator 执行面。"""
    try:
        return await delete_dataset(dataset_id)
    except ConsoleError as exc:
        _raise_console(exc)


@router.post("/datasets/{dataset_id}/start-sampling")
async def start_dataset_sampling(dataset_id: str) -> dict[str, Any]:
    """开始实时采样（realtime 数据集专属）。"""
    try:
        return await apply_start_sampling(dataset_id)
    except ConsoleError as exc:
        _raise_console(exc)


@router.post("/datasets/{dataset_id}/stop-sampling")
async def stop_dataset_sampling(dataset_id: str) -> dict[str, Any]:
    """停止实时采样。"""
    try:
        return await apply_stop_sampling(dataset_id)
    except ConsoleError as exc:
        _raise_console(exc)


class SamplingRatioUpdate(BaseModel):
    """实时数据集采样率修改请求。"""

    model_config = {"populate_by_name": True}

    sampling_ratio: float = Field(alias="samplingRatio", gt=0, le=1)


@router.patch("/datasets/{dataset_id}/sampling")
async def update_dataset_sampling(
    dataset_id: str,
    body: SamplingRatioUpdate,
) -> dict[str, Any]:
    """修改实时数据集采样率并同步 evaluator 条目（身份不变）。"""
    try:
        return await apply_update_sampling_ratio(dataset_id, body.sampling_ratio)
    except ConsoleError as exc:
        _raise_console(exc)


@router.post("/datasets/{dataset_id}/collection/start", response_model=None)
async def start_dataset_collection(dataset_id: str) -> NoReturn:
    """真实回放执行器尚未接入，不把控制面状态伪装成执行中。"""
    raise HTTPException(
        status_code=501,
        detail=f"corpus collection executor is not implemented: {dataset_id}",
    )


@router.post("/datasets/{dataset_id}/collection/stop")
async def stop_dataset_collection(dataset_id: str) -> dict[str, Any]:
    """停止语料收集。"""
    try:
        return await apply_stop_collection(dataset_id)
    except ConsoleError as exc:
        _raise_console(exc)


@router.get("/benchmark-agents")
async def read_agents(
    agent_scope: str | None = Query(default=None, alias="agentScope"),
) -> list[dict[str, Any]]:
    """列出评测代理（必填评测对象，按归属白名单过滤）。"""
    try:
        items = await list_agents(agent_scope)
    except ConsoleError as exc:
        _raise_console(exc)
    await enrich_agent_evaluation_counts(items)
    return items


@router.get("/benchmark-agents/overview")
async def read_agents_overview() -> list[dict[str, Any]]:
    """全量注册总览：所有评测对象下全部带版本注册。"""
    items = await list_all_agent_registrations()
    await enrich_agent_evaluation_counts(items)
    return items


@router.put("/benchmark-agents")
async def write_agent(body: dict[str, Any]) -> dict[str, Any]:
    """保存一个评测代理（保存后自动同步统一 agent 视图，用户无感知）。"""
    document = _parse(AgentDocument, body)
    try:
        result = await save_agent(document)
    except ConsoleError as exc:
        _raise_console(exc)
    await _sync_agent_view()
    return result


async def _sync_agent_view() -> None:
    """页面配置变更后重同步 agent 视图物化行（失败不影响保存结果）。"""
    from benchmark.routes.benchmark import sync_agent_view_catalog  # noqa: PLC0415

    try:
        await sync_agent_view_catalog()
    except Exception:
        logger.exception("agent view catalog sync failed")


class CustomizationUploadForm(BaseModel):
    """定制上传的 multipart 文本字段收拢（字段名与现网 Form alias 一致）。"""

    agent_name: str = Field(alias="agentName")
    business: str
    variant_key: str = Field(alias="variantKey")
    stage: str
    prompt_text: str | None = Field(default=None, alias="promptText")
    slot: str | None = None
    created_by: str = Field(default="", alias="createdBy")


async def _customization_form(request: Request) -> CustomizationUploadForm:
    """从 multipart 文本字段解析上传入参。

    FastAPI 不支持「Form 模型 + File 参数」在同一端点混用：模型会被当成名为
    ``form`` 的单个表单字段（JSON 字符串），请求恒报 ``Missing required
    field: form``。故直接读已解析的表单字段（文件字段跳过）再交给模型校验，
    校验失败按普通请求校验错误抛出。
    """
    form = await request.form()
    values = {key: value for key, value in form.multi_items() if isinstance(value, str)}
    try:
        return CustomizationUploadForm.model_validate(values)
    except ValidationError as exc:
        raise RequestValidationError(
            [{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]
        ) from exc


@router.post("/benchmark-agents/customizations", status_code=201)
async def upload_customization(
    form: Annotated[CustomizationUploadForm, Depends(_customization_form)],
    prompt_files: Annotated[list[UploadFile] | None, File(alias="promptFiles")] = None,
    skill_files: Annotated[list[UploadFile] | None, File(alias="skillFiles")] = None,
) -> dict[str, Any]:
    """上传并提交一个 stage 的业务定制（prompt/skill 至少一组，原子提交 Git）。

    校验与失败语义见设计 §4.7：policy 不允许 403、内容非法 400、
    Git 不可用 503（不分配 contentVersion、不落注册）。
    """
    from benchmark.customization.manifest import (  # noqa: PLC0415
        prompt_slot_from_filename,
    )
    from benchmark.customization.service import (  # noqa: PLC0415
        StageUploadRequest,
        upload_stage_customization,
    )

    try:
        prompt_texts: dict[str, str] = {}
        if form.prompt_text is not None and form.prompt_text.strip():
            if not form.slot:
                raise CustomizationError(400, "slot_required", "promptText 需配 slot 指定目标槽位")
            prompt_texts[form.slot] = form.prompt_text
        for upload in prompt_files or []:
            raw = await upload.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CustomizationError(
                    400, "invalid_encoding", f"prompt 文件不是 UTF-8：{upload.filename}"
                ) from exc
            prompt_texts[prompt_slot_from_filename(upload.filename or "")] = text
        skill_blobs: dict[str, bytes] = {}
        for upload in skill_files or []:
            name = (upload.filename or "").replace("\\", "/")
            if name.startswith("skill/"):
                name = name[len("skill/") :]
            skill_blobs[name] = await upload.read()
        ref = await upload_stage_customization(
            StageUploadRequest(
                agent_name=form.agent_name,
                business=form.business,
                variant_key=form.variant_key,
                stage=form.stage,
                prompt_texts=prompt_texts,
                skill_files=skill_blobs,
                created_by=form.created_by,
            )
        )
    except CustomizationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.detail},
        ) from exc
    return {
        "stage": ref["stage"],
        "contentVersion": ref["content_version"],
        "gitBranch": ref["git_branch"],
        "gitPath": ref["git_path"],
        "gitCommit": ref["git_commit"],
        "bundleSha256": ref["bundle_sha256"],
        "prompt": ref["prompt"],
        "skill": ref["skill"],
        "createdAt": ref["created_at"],
        "createdBy": ref["created_by"],
    }


@dataclass(frozen=True)
class _StageRefFromGitSpec:
    """``_stage_ref_from_git`` 参数包（FUNC-PARAMS-001 参数收敛）。"""

    stage: str
    git_path: str
    git_branch: str
    content_version: int


async def _stage_ref_from_git(
    store: GitContentStore,
    spec: _StageRefFromGitSpec,
) -> dict[str, Any]:
    """从版本目录 manifest.json 组装完整 StageCustomizationRef（camelCase）。

    供版本列表复用绑定：注册落库需要 prompt/skill/bundleSha256，不能只给定位字段。
    """
    git_path = spec.git_path
    content_version = spec.content_version
    git_commit = await store.latest_commit_for_path(git_path)
    files = await store.read_files(git_commit=git_commit, paths=[f"{git_path}/manifest.json"])
    raw = files.get(f"{git_path}/manifest.json", b"")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"版本 v{content_version} manifest 无法解析：{git_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"版本 v{content_version} manifest 格式非法：{git_path}")
    bundle_sha = str(manifest.get("bundleSha256") or "")
    prompt = manifest.get("prompt")
    skill = manifest.get("skill")
    if not bundle_sha or (prompt is None and skill is None):
        raise ValueError(
            f"版本 v{content_version} manifest 缺少 bundleSha256/prompt/skill：{git_path}"
        )
    return {
        "stage": spec.stage,
        "contentVersion": int(manifest.get("contentVersion") or content_version),
        "gitBranch": spec.git_branch,
        "gitPath": git_path,
        "gitCommit": git_commit,
        "bundleSha256": bundle_sha,
        "prompt": prompt,
        "skill": skill,
        "createdAt": str(manifest.get("createdAt") or ""),
        "createdBy": str(manifest.get("createdBy") or ""),
    }


@router.get("/benchmark-agents/customizations/versions")
async def list_customization_versions(
    agent_name: str = Query(alias="agentName"),
    business: str = Query(default=""),
    variant_key: str = Query(default="", alias="variantKey"),
    stage: str = Query(),
) -> list[dict[str, Any]]:
    """列出某 agent+stage 已上传的定制版本（完整 StageCustomizationRef，contentVersion 倒序）。

    每项含 prompt/skill/bundleSha256，可选中后直接复用注册，无需再点上传。
    """
    if stage not in ("judge", "attribution"):
        raise HTTPException(status_code=400, detail="stage 须为 judge 或 attribution")
    config = fetch_customization_config()
    store = GitContentStore(config)
    key = CustomizationKey(
        agent_name=agent_name,
        business=business,
        variant_key=variant_key,
        stage=stage,
    )
    try:
        base = stage_base_path(config.base_path, key)
        versions = await store.list_stage_versions(base)
    except CustomizationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.detail},
        ) from exc
    result: list[dict[str, Any]] = []
    for version in versions:
        git_path = f"{base}/v{version}"
        try:
            result.append(
                await _stage_ref_from_git(
                    store,
                    _StageRefFromGitSpec(
                        stage=stage,
                        git_path=git_path,
                        git_branch=config.branch,
                        content_version=version,
                    ),
                )
            )
        except Exception as exc:
            # 单版本不可读时跳过，避免整表失败；前端仍可手填上传
            logger.warning(
                "customization.version_ref_skip path=%s err=%s",
                git_path,
                type(exc).__name__,
            )
    return result


@router.get("/benchmark-agents/customizations/preview")
async def preview_customization(
    git_path: str = Query(alias="gitPath"),
    git_commit: str = Query(alias="gitCommit"),
    # 禁止用 query 名 path：网关/WAF 常剥该参数；Next.js [...path] 也会与之撞名。
    file_path: str = Query(alias="filePath"),
) -> dict[str, Any]:
    """按 commit 预览 bundle 内单个文本文件（≤64KB，页面确认上传内容）。"""
    import hashlib  # noqa: PLC0415

    config = fetch_customization_config()
    store = GitContentStore(config)
    try:
        files = await store.read_files(git_commit=git_commit, paths=[f"{git_path}/{file_path}"])
    except CustomizationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.detail},
        ) from exc
    data = files.get(f"{git_path}/{file_path}", b"")
    if len(data) > 64 * 1024:
        raise HTTPException(status_code=400, detail="文件超过预览上限 64KB")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="文件不是 UTF-8 文本") from exc
    return {
        "path": file_path,
        "text": text,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


@router.get("/benchmark-agents/{agent_id}")
async def read_agent(agent_id: str) -> dict[str, Any]:
    """读取一个评测代理。"""
    try:
        item = await get_agent(agent_id)
    except ConsoleError as exc:
        _raise_console(exc)
    await enrich_agent_evaluation_counts([item])
    return item


@router.delete("/benchmark-agents/{agent_id}")
async def remove_agent(agent_id: str) -> dict[str, Any]:
    """删除一个评测代理；级联 retire 绑定的 realtime evaluator 执行面。"""
    try:
        result = await delete_agent(agent_id)
    except ConsoleError as exc:
        _raise_console(exc)
    await _sync_agent_view()
    return result


@router.get("/eval-tasks")
async def read_tasks(
    agent_scope: str | None = Query(default=None, alias="agentScope"),
) -> list[dict[str, Any]]:
    """列出评测任务。"""
    try:
        return await list_tasks(agent_scope)
    except ConsoleError as exc:
        _raise_console(exc)


@router.put("/eval-tasks")
async def write_task(body: dict[str, Any]) -> dict[str, Any]:
    """保存一条评测任务。不启动 benchmark。"""
    document = _parse(EvalTaskDocument, body)
    try:
        return await save_task(document)
    except ConsoleError as exc:
        _raise_console(exc)


@router.get("/eval-tasks/{task_id}")
async def read_task(task_id: str) -> dict[str, Any]:
    """读取一条评测任务。"""
    try:
        return await get_task(task_id)
    except ConsoleError as exc:
        _raise_console(exc)


@router.delete("/eval-tasks/{task_id}")
async def remove_task(task_id: str) -> dict[str, Any]:
    """删除一条评测任务：软删文档；实时流任务联动采样与条目（同 stop）。

    删除与停止同属「任务不再启用」：最后一个运行中实时流任务被删除时，
    采样与 evaluator 条目一并停止，避免无任务但实时评测仍在进行。批量
    任务名下的评测结果保留不动。
    """
    try:
        return await apply_remove_task(task_id)
    except ConsoleError as exc:
        _raise_console(exc)


async def _drain_wait_jobs(qm: Any, run_id: str) -> int:  # noqa: ANN401 — queue manager 鸭子类型
    """只删 wait 中未开始的 job；running 在途保留等结果。返回移除数。"""
    removed = 0
    for queue_name, job_types, slot_kind in _STOP_WAIT_QUEUE_MAP:
        queue = qm.get_queue(queue_name)
        try:
            jobs = await list_queue_jobs(queue, job_types)
        except Exception:
            logger.warning(
                "console.stop_run_list_failed queue=%s run_id=%s",
                queue_name,
                run_id,
                exc_info=True,
            )
            continue
        for job in jobs:
            if job is None:
                continue
            data = dict(job.data or {})
            if str(data.get("run_id") or "") != run_id:
                continue
            slot_key = data.get("slot_key")
            if slot_key:
                await release_slot(str(slot_key))
            kind = str(data.get("cluster_kind") or slot_kind or "")
            if kind:
                await release_cluster_slot(kind)
            try:
                await job.remove()
                removed += 1
            except Exception:
                logger.warning(
                    "console.stop_run_remove_failed job_id=%s run_id=%s",
                    job.id,
                    run_id,
                    exc_info=True,
                )
    return removed


async def _drain_parked_non_inflight_running_jobs(qm: Any, run_id: str) -> int:  # noqa: ANN401
    """删 running 队列中未开始的 parked job；保留 active 与 pending_callback。"""
    removed = 0
    for queue_name in STOP_RUNNING_ALL_QUEUES:
        queue = qm.get_queue(queue_name)
        try:
            jobs = await list_queue_jobs(queue, JOB_TYPES_PARKED)
        except Exception:
            logger.warning(
                "console.stop_run_parked_list_failed queue=%s run_id=%s",
                queue_name,
                run_id,
                exc_info=True,
            )
            continue
        for job in jobs:
            if job is None or not job_belongs_to_run(job, run_id):
                continue
            if is_stop_inflight_job(job, listed_as_active=False):
                continue
            slot_key = dict(job.data or {}).get("slot_key")
            if slot_key:
                await release_slot(str(slot_key))
            kind = str(dict(job.data or {}).get("cluster_kind") or "")
            if kind:
                await release_cluster_slot(kind)
            try:
                await job.remove()
                removed += 1
            except Exception:
                logger.warning(
                    "console.stop_run_parked_remove_failed job_id=%s run_id=%s",
                    job.id,
                    run_id,
                    exc_info=True,
                )
    return removed


async def _count_inflight_jobs(
    qm: Any,  # noqa: ANN401 — queue manager 鸭子类型
    run_id: str,
) -> tuple[int, int] | None:
    """清点真正在途 → (judge, attribution)；不确定时 ``None``（见 ``stop_inflight``）。"""
    return await count_stop_inflight_jobs(qm, run_id)


async def _force_partial_finalize(request: Request, run_id: str) -> None:
    """尽力触发阶段性报告：依次尝试 external / sync 公开 finalize CAS。

    不直接调用 ``external_runner._finalize_run``：其对非 external / 未达
    计数条件的 run 可能静默 return，导致 sync 批量阶段性报告被跳过。
    ``try_finalize_*`` 按各自 CAS 条件路由，互不短路。
    仍有 running items 时直接跳过（与 stop-run 等待终态同口径）。
    """
    if not await stop_items_ready_for_report(run_id):
        running = await count_running_benchmark_items(run_id)
        logger.info(
            "console.stop_run_force_finalize_deferred run_id=%s running_items=%d",
            run_id,
            running,
        )
        return
    benchmark_service = getattr(request.app.state, "benchmark_service", None)
    if benchmark_service is None:
        return
    try:
        await try_finalize_external(run_id, benchmark_service)
    except Exception:
        logger.warning(
            "console.stop_run_try_finalize_external_failed run_id=%s",
            run_id,
            exc_info=True,
        )
    try:
        await try_finalize_benchmark(run_id, benchmark_service)
    except Exception:
        logger.warning(
            "console.stop_run_try_finalize_benchmark_failed run_id=%s",
            run_id,
            exc_info=True,
        )


_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "stopped"})


@dataclass(frozen=True)
class StoppableRunRef:
    """stop / stop-run 解析到的可停批量 run。

    ``run_pk`` 必须与 ``run_id`` 指向同一条 run；回落解析时不得沿用过期
    文档 ``manualRunPk``，避免调用方按 pk 直达停错 run。
    """

    run_id: str
    run_pk: str | None = None


def _document_manual_run_pk(task: dict[str, Any]) -> str | None:
    raw = task.get("manualRunPk")
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    return text or None


def _benchmark_run_pk(run: object) -> str | None:
    raw = getattr(run, "id", None)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


async def _resolve_stoppable_run(request: Request, task: dict[str, Any]) -> StoppableRunRef | None:
    """解析任务当前可停的批量 run。

    优先文档 ``manualRunId``（仍非终态时）；否则按 ``eval_task_id`` 取最新
    非终态 run。避免页面按 taskId 归属展示「停止本次运行」，但文档指针
    缺失/过期导致 stop / stop-run 空转。
    """
    benchmark_service = getattr(request.app.state, "benchmark_service", None)
    preferred = task.get("manualRunId")
    preferred_id = preferred.strip() if isinstance(preferred, str) else ""
    found: StoppableRunRef | None = None

    if preferred_id and benchmark_service is None:
        found = StoppableRunRef(
            run_id=preferred_id,
            run_pk=_document_manual_run_pk(task),
        )
    elif preferred_id and benchmark_service is not None:
        status: str | None = None
        preferred_pk: str | None = None
        try:
            run = await benchmark_service.get_run(preferred_id)
            if run is not None:
                status = str(getattr(run.status, "value", run.status))
                preferred_pk = _benchmark_run_pk(run)
        except Exception:
            logger.warning(
                "console.resolve_run_failed run_id=%s",
                preferred_id,
                exc_info=True,
            )
            status = None
        if status is None or status not in _TERMINAL_RUN_STATUSES:
            # 查无 / 服务异常 / 仍在跑：沿用文档指针（不把整个 stop 打成 500）
            found = StoppableRunRef(
                run_id=preferred_id,
                run_pk=preferred_pk or _document_manual_run_pk(task),
            )

    if found is not None:
        return found
    if benchmark_service is None:
        return None

    task_id = str(task.get("id") or "").strip()
    if not task_id:
        return None
    try:
        runs = await benchmark_service.list_runs(ListRunsQuery(task_id=task_id, limit=50))
    except Exception:
        logger.warning(
            "console.resolve_stoppable_run_failed task_id=%s",
            task_id,
            exc_info=True,
        )
        return None
    for run in runs:
        status = str(getattr(run.status, "value", run.status))
        if status in _TERMINAL_RUN_STATUSES:
            continue
        run_id = str(getattr(run, "run_id", "") or "").strip()
        if run_id:
            found = StoppableRunRef(run_id=run_id, run_pk=_benchmark_run_pk(run))
            break
    return found


@dataclass(frozen=True)
class _StopWaitingWrite:
    """stop-run 等待在途 / running items / 归因齐套时的响应入参（FUNC-PARAMS-001）。"""

    run_id: str
    removed: int
    completed_cases: int
    inflight_jobs: int | None
    running_items: int | None = None
    attribution_pending: int | None = None


def _stop_waiting_payload(write: _StopWaitingWrite) -> dict[str, Any]:
    """stop-run 仍在等待在途队列 / running items / 归因时的统一响应。"""
    payload: dict[str, Any] = {
        "runId": write.run_id,
        "jobsRemoved": write.removed,
        "status": "running",
        "reportStatus": "waiting_inflight",
        "partial": True,
        "completedCases": write.completed_cases,
        "inflightJobs": write.inflight_jobs,
        "idempotentReplay": False,
    }
    if write.running_items is not None:
        payload["runningItems"] = write.running_items
    if write.attribution_pending is not None:
        payload["attributionPending"] = write.attribution_pending
    return payload


async def _terminate_manual_run(request: Request, run_id: str) -> dict[str, Any]:
    """停止本次批量 run：门闩 + 清 wait + 等 items/归因齐套后再阶段性报告。

    不再 ``fail_run``（设计 §6.8）。幂等：已终态直接返回。
    有 BullMQ 在途、仍有 ``benchmark_run_items.status=running``、或已登记
    归因未到齐时，不立刻出报告 / ``finalize_as_stopped``；归因提交流程
    不因 stop 中断。``stopped`` 后仍接受迟到回调落库（见 external_runner）。
    """
    benchmark_service = getattr(request.app.state, "benchmark_service", None)
    qm = getattr(request.app.state, "eval_queue_manager", None)
    if benchmark_service is None:
        logger.warning(
            "console.stop_run_skipped_unavailable run_id=%s (benchmark_service not ready)",
            run_id,
        )
        return {
            "runId": run_id,
            "jobsRemoved": 0,
            "status": "unavailable",
            "reportStatus": "skipped_unavailable",
        }

    marked = await mark_stop_requested(run_id=run_id)
    redis = getattr(request.app.state, "redis", None) or getattr(
        request.app.state, "redis_client", None
    )
    await publish_stop_signal(redis, run_id=run_id)

    removed = 0
    if qm is not None:
        removed = await _drain_wait_jobs(qm, run_id)
        removed += await _drain_parked_non_inflight_running_jobs(qm, run_id)
    else:
        logger.warning(
            "console.stop_run_drain_skipped run_id=%s (queue manager not ready)",
            run_id,
        )

    completed_cases = int(marked.get("completed_cases") or 0)
    if marked.get("noop") and marked.get("status") in {"stopped", "completed", "failed"}:
        logger.info(
            "console.stop_run_idempotent run_id=%s status=%s jobs_removed=%d",
            run_id,
            marked.get("status"),
            removed,
        )
        return {
            "runId": run_id,
            "jobsRemoved": removed,
            "status": marked.get("status"),
            "reportStatus": "noop",
            "partial": marked.get("status") == "stopped",
            "completedCases": completed_cases,
            "idempotentReplay": True,
        }

    # 先快照 processed，再清点在途：避免回调落在「清点→读行」之间导致
    # 同一 item 同时计入 processed 与 inflight，stop_target 永久不可达。
    run_snap = await benchmark_service.get_run(run_id)
    processed_snapshot = 0
    if run_snap is not None:
        processed_snapshot = int(run_snap.evaluated_items or 0) + int(
            run_snap.failed_items or 0,
        )

    # 清点失败（Redis 抖动等）→ fail-closed：不 align(0)、不立刻 finalize，
    # 保留 provisional stop_target，等 healer / 再次 stop 收尾。
    counted: tuple[int, int] | None = (0, 0)
    if qm is not None:
        counted = await _count_inflight_jobs(qm, run_id)
    if counted is None:
        logger.warning(
            "console.stop_run_inflight_uncertain run_id=%s jobs_removed=%d "
            "(list_queue_jobs failed; defer finalize)",
            run_id,
            removed,
        )
        return _stop_waiting_payload(
            _StopWaitingWrite(
                run_id=run_id,
                removed=removed,
                completed_cases=completed_cases or processed_snapshot,
                inflight_jobs=None,
            ),
        )

    inflight_judge, inflight_attr = counted
    aligned = await align_expectations_for_stop(
        run_id,
        inflight_judge=inflight_judge,
        inflight_attribution=inflight_attr,
        processed_snapshot=processed_snapshot,
    )
    inflight_total = int(aligned.get("inflight_total") or 0)
    completed_cases = int(aligned.get("processed") or completed_cases)
    readiness = await inspect_stop_report_readiness(run_id)
    running_items = readiness.running_items
    attribution_pending = readiness.attribution_pending

    # 队列在途 / item 仍 running / 归因未齐：禁止提前出报告
    if inflight_total > 0 or not readiness.ready:
        logger.warning(
            "console.stop_run_waiting_inflight run_id=%s jobs_removed=%d "
            "inflight_judge=%d inflight_attr=%d running_items=%d "
            "attribution_pending=%d stop_target=%s",
            run_id,
            removed,
            inflight_judge,
            inflight_attr,
            running_items,
            attribution_pending,
            aligned.get("stop_target_count"),
        )
        return _stop_waiting_payload(
            _StopWaitingWrite(
                run_id=run_id,
                removed=removed,
                completed_cases=completed_cases,
                inflight_jobs=inflight_total,
                running_items=running_items,
                attribution_pending=attribution_pending,
            ),
        )

    # 无在途且 items 已终态：仅有已完成 case 时才强制阶段性报告
    if completed_cases > 0:
        await _force_partial_finalize(request, run_id)

    await finalize_as_stopped(FinalizeStoppedWrite(run_id=run_id))
    run = await benchmark_service.get_run(run_id)
    run_status = getattr(run, "status", None) if run is not None else None
    status = str(getattr(run_status, "value", run_status)) if run_status is not None else "stopped"
    meta = getattr(run, "report_meta_json", None) if run is not None else None
    report_status = "skipped_no_cases"
    if isinstance(meta, dict):
        report_status = str(meta.get("report_status") or report_status)
    elif completed_cases > 0:
        report_status = "generating"
    logger.info(
        "console.stop_run_terminated run_id=%s jobs_removed=%d status=%s",
        run_id,
        removed,
        status,
    )
    return {
        "runId": run_id,
        "jobsRemoved": removed,
        "status": status,
        "reportStatus": report_status,
        "partial": True,
        "completedCases": completed_cases,
        "idempotentReplay": False,
    }


async def _trigger_realtime_stopped_report(request: Request, task: dict[str, Any]) -> str | None:
    """实时流停止后尽力出 ``*-stopped-*`` 窗口报告；失败不阻断 stop。

    经 ``find_report_item`` 匹配 RemoteConfig+DB 合并项；匹配不到返回 ``None``，
    禁止回退到任意 ``reports[0]`` / 仅 RemoteConfig 列表。
    """
    scheduler = getattr(request.app.state, "trace_report_scheduler", None)
    if scheduler is None or not hasattr(scheduler, "publish_stopped_window"):
        return None
    registration_id = str(task.get("agentRegistrationId") or task.get("agentId") or "").strip()
    stream_instance_id = str(task.get("streamInstanceId") or task.get("datasetId") or "").strip()
    if not registration_id:
        return None
    matched = None
    finder = getattr(scheduler, "find_report_item", None)
    if callable(finder):
        try:
            matched = await finder(
                registration_id=registration_id,
                stream_instance_id=stream_instance_id or None,
            )
        except Exception:
            logger.warning(
                "console.realtime_stopped_report_lookup_failed task=%s",
                task.get("id"),
                exc_info=True,
            )
            return None
    if matched is None:
        return None
    try:
        return await scheduler.publish_stopped_window(matched)
    except Exception:
        logger.warning(
            "console.realtime_stopped_report_failed task=%s",
            task.get("id"),
            exc_info=True,
        )
        return None


@router.post("/eval-tasks/{task_id}/stop")
async def stop_eval_task(task_id: str, request: Request) -> dict[str, Any]:
    """停止任务：文档置 stopped；有进行中执行则级联 stop-run；实时流收敛执行面。"""
    try:
        result = await apply_stop_task(task_id)
    except ConsoleError as exc:
        _raise_console(exc)
    # 文档指针可能过期：按 eval_task_id 回落找仍在跑的 run，保证「停止任务」真停执行。
    resolved = await _resolve_stoppable_run(request, result)
    if resolved is not None:
        stop_info = await _terminate_manual_run(request, resolved.run_id)
        result = {**result, "stopRun": stop_info}
    else:
        stopped_report = await _trigger_realtime_stopped_report(request, result)
        if stopped_report:
            result = {
                **result,
                "stopRun": {
                    "runId": stopped_report,
                    "status": "stopped",
                    "reportStatus": "generating",
                    "partial": True,
                },
            }
    return result


@router.post("/eval-tasks/{task_id}/stop-run")
async def stop_eval_task_run(task_id: str, request: Request) -> dict[str, Any]:
    """仅停止本次运行并出阶段性报告；不改变周期任务调度启用态。

    立即/定时批量：停当前可停 run（优先 ``manualRunId``，否则按任务归属找
    非终态 run）；文档调度态不变。
    实时流（无 batch run）：收敛执行面并尽力出 ``*-stopped-*`` 窗口报告。
    无当次可停 run 的定时任务：noop，**不**调用 ``apply_stop_task``。
    """
    try:
        task = await get_task(task_id)
    except ConsoleError as exc:
        _raise_console(exc)
    if not isinstance(task, dict):
        return {"taskId": task_id, "status": "unknown", "reportStatus": "noop"}
    resolved = await _resolve_stoppable_run(request, task)
    if resolved is not None:
        stop_info = await _terminate_manual_run(request, resolved.run_id)
        return {
            "taskId": task_id,
            "runId": resolved.run_id,
            "runPk": resolved.run_pk,
            **stop_info,
        }

    is_realtime = str(task.get("evalMode") or "") == "realtime-span"
    if is_realtime:
        try:
            result = await apply_stop_task(task_id)
        except ConsoleError as exc:
            _raise_console(exc)
        report_task = {**task, **result}
        stopped_report = await _trigger_realtime_stopped_report(request, report_task)
        stop_run: dict[str, Any] = {
            "status": "stopped",
            "partial": True,
            "reportStatus": "generating" if stopped_report else "skipped_no_match",
        }
        if stopped_report:
            stop_run["runId"] = stopped_report
        return {
            **result,
            "status": result.get("status") or "stopped",
            "reportStatus": stop_run["reportStatus"],
            "partial": True,
            "stopRun": stop_run,
            "idempotentReplay": result.get("status") == "stopped",
        }

    # 定时/立即但无可停当次 run：保持调度文档不动
    return {
        "taskId": task_id,
        "status": task.get("status") or "active",
        "reportStatus": "noop_no_active_run",
        "partial": False,
        "idempotentReplay": True,
    }


@router.post("/eval-tasks/{task_id}/rerun")
async def rerun_eval_task(task_id: str) -> dict[str, Any]:
    """重新运行已停止的任务。"""
    try:
        return await apply_rerun_task(task_id)
    except ConsoleError as exc:
        _raise_console(exc)


@router.post("/eval-tasks/{task_id}/retry")
async def retry_eval_task(task_id: str) -> dict[str, Any]:
    """重试执行失败的任务。"""
    try:
        return await apply_retry_task(task_id)
    except ConsoleError as exc:
        _raise_console(exc)


@router.get("/rubrics")
async def read_rubrics() -> list[dict[str, Any]]:
    """只读列出代码白名单 rubric，形状对齐页面。"""
    return page_rubrics()
