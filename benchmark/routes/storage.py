"""OSS 文件上传 API。

复用 RemoteConfig ``agent-eval.benchmark.oss``（经
:func:`build_data_storage_client`），供业务/运维通过 HTTP 上传任意文件。

路径前缀：``/v1/storage``
"""

import csv
import io
import mimetypes
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError

from benchmark.domain.report_storage_config import get_default_download_prefix
from benchmark.domain.runtime_config_store import next_oss_upload_id
from benchmark.runs.common.dataset_normalizer import (
    DatasetNormalizationError,
    normalize_dataset_json,
)
from benchmark.reporting.framework.report_storage import build_data_storage_client

logger = get_logger("agent_eval.api.storage")

#: benchmark 数据集规范化后的约定文件名（与 loader/前端对齐）
_BENCHMARK_DATASET_CSV_NAME = "benchmark_final_results.csv"

router = APIRouter(prefix="/v1/storage", tags=["storage"])

#: 单文件上限 50 MiB（与常见评测 CSV/HTML 量级对齐）。
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
#: 未指定 prefix / object_key 时的默认前缀。
_DEFAULT_PREFIX = "agent-eval/uploads"
#: 默认预签名有效期 — 30 天（与报告链路一致）。
_DEFAULT_SIGN_EXPIRES_SECONDS = 30 * 24 * 60 * 60
#: object key 安全字符（相对路径段）。
_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9._\-/=]+$")

#: benchmark 数据集 CSV 必备列（拉取侧 loader 以这两列作评测输入/期望输出，
#: 缺列的数据集跑起来才报错，上传时前置拦截）。
_REQUIRED_DATASET_COLUMNS = ("input", "expected_output")
#: 列校验豁免前缀：amap-agent 的 benchmark 数据集为 trace 语义
#: （如 amap_agent_ieval 的 id/query 列），不套用 input/expected_output 契约。
#: 判定取 oss_prefix 首段，归一（小写、下划线转连字符）后等于或以
#: ``amap-agent-`` 开头即豁免（覆盖 amap-agent/benchmark、amap_agent_ieval 等）。
_DATASET_COLUMN_EXEMPTION_PREFIX = "amap-agent"


class _StorageUploader(Protocol):
    """最小上传协议（避免模块级 import libs/storage）。"""

    async def upload(
        self,
        object_key: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> str | None: ...

    async def sign_url(
        self,
        object_key: str,
        *,
        expires_seconds: int = ...,
    ) -> str | None: ...


def _safe_basename(filename: str | None) -> str:
    """Strip path traversal；空名回退 ``upload.bin``。"""
    raw = (filename or "").strip() or "upload.bin"
    name = PurePosixPath(raw.replace("\\", "/")).name
    name = name.strip() or "upload.bin"
    if name in (".", ".."):
        return "upload.bin"
    return name


def _benchmark_upload_key(key_path: str, partition_date: str, upload_id: int, basename: str) -> str:
    """benchmark 语义落盘 key：``{prefix}/{yyyymmdd}/{upload_id}/{basename}``。

    upload_id 为当日上传顺序号（1 起递增，见
    :func:`benchmark.domain.runtime_config_store.next_oss_upload_id`）：同
    prefix 同日多次上传不落同一 key 互相覆盖，每次上传即不可变快照，
    旧对象保留供旧 run 克隆按快照 key 复现。
    """
    return f"{key_path.strip('/')}/{partition_date}/{upload_id}/{basename}"


def _resolve_object_key(
    *,
    filename: str,
    prefix: str | None,
    object_key: str | None,
) -> str:
    """``object_key`` 优先；否则 ``{prefix}/{yyyyMMdd}/{basename}``。"""
    if object_key and object_key.strip():
        # 先拒绝对路径前缀，再规范化；避免 "/a" → lstrip → "a" 后
        # ``startswith("/")`` 永远不可达。
        raw = object_key.strip()
        if raw.startswith("/"):
            raise HTTPException(
                status_code=400,
                detail="object_key/prefix contains invalid characters or path traversal",
            )
        key = raw.lstrip("/")
    else:
        cleaned = (prefix or "").strip().strip("/") or _DEFAULT_PREFIX
        day = datetime.now(UTC).strftime("%Y%m%d")
        key = f"{cleaned}/{day}/{filename}"
    if ".." in key.split("/") or not _SAFE_KEY_RE.match(key):
        raise HTTPException(
            status_code=400,
            detail="object_key/prefix contains invalid characters or path traversal",
        )
    return key


def _validate_dataset_columns(*, basename: str, data: bytes, key_path: str) -> None:
    """benchmark 语义上传（oss_prefix 模式）的 CSV 列校验。

    数据集 CSV 必须含 ``input`` / ``expected_output`` 两列，缺列 400 且不落
    OSS；amap-agent 家族前缀（trace 语义）豁免；非 CSV 文件不做列校验。
    """
    head = key_path.strip("/").split("/", 1)[0].lower().replace("_", "-")
    if head == _DATASET_COLUMN_EXEMPTION_PREFIX or head.startswith(
        f"{_DATASET_COLUMN_EXEMPTION_PREFIX}-"
    ):
        return
    if not basename.lower().endswith(".csv"):
        return
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="dataset file is not a valid UTF-8 CSV"
        ) from exc
    header = next(csv.reader(io.StringIO(text)), [])
    columns = {cell.strip() for cell in header}
    missing = [name for name in _REQUIRED_DATASET_COLUMNS if name not in columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "dataset CSV missing required columns: "
                + ", ".join(missing)
                + " (required: input, expected_output)"
            ),
        )


@dataclass(frozen=True)
class _BenchmarkUploadTarget:
    """benchmark 语义上传的落盘目标（原样前缀 + 分区日期 + 当日顺序号）。"""

    key: str
    oss_prefix: str
    partition_date: str
    upload_id: int
    key_path: str


async def _resolve_benchmark_upload(
    *, oss_prefix: str, basename: str, filename: str | None
) -> _BenchmarkUploadTarget:
    """解析 benchmark 语义前缀并领当日顺序号，拼出最终落盘 object key。"""
    key_path = _strip_oss_scheme(oss_prefix.strip())
    if not key_path.strip("/"):
        logger.warning(
            "storage.upload_oss_prefix_rejected raw=%r (empty path)",
            oss_prefix,
        )
        raise HTTPException(status_code=400, detail="oss_prefix path is empty")
    partition_date = datetime.now(UTC).strftime("%Y%m%d")
    # 先领当日顺序号再拼 key：并发上传由 DB 计数器串行化领号，
    # 同 prefix 同日也不会撞 key。校验拒绝/上传失败会跳号（可接受，
    # 顺序号只保证递增看顺序，不保证连续）。
    upload_id = await next_oss_upload_id(partition_date)
    key = _benchmark_upload_key(key_path, partition_date, upload_id, basename)
    if ".." in key.split("/") or not _SAFE_KEY_RE.match(key):
        logger.warning(
            "storage.upload_key_rejected raw_prefix=%r key=%r",
            oss_prefix,
            key,
        )
        raise HTTPException(
            status_code=400,
            detail="object_key/prefix contains invalid characters or path traversal",
        )
    resolved_oss_prefix = f"{_bucket_scheme_prefix()}/{key_path.strip('/')}"
    logger.info(
        "storage.upload_received mode=benchmark_prefix filename=%r raw_prefix=%r "
        "object_key=%s resolved_prefix=%s",
        filename,
        oss_prefix,
        key,
        resolved_oss_prefix,
    )
    return _BenchmarkUploadTarget(
        key=key,
        oss_prefix=resolved_oss_prefix,
        partition_date=partition_date,
        upload_id=upload_id,
        key_path=key_path,
    )


async def _read_upload_limited(file: UploadFile, *, max_bytes: int) -> bytes:
    """分块读取上传流；累计超过 ``max_bytes`` 立即中止，避免整包入内存。"""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds {max_bytes} bytes limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


class StorageUploadForm(BaseModel):
    """``POST /v1/storage/upload`` multipart 文本字段（FUNC-PARAMS-001 收敛）。"""

    prefix: str | None = Field(
        default=None,
        description="Object key 前缀；缺省 agent-eval/uploads/{yyyyMMdd}/",
    )
    object_key: str | None = Field(
        default=None,
        description="完整 object key（覆盖 prefix 拼接）",
    )
    oss_prefix: str | None = Field(
        default=None,
        description=(
            "benchmark 数据集语义前缀：**照原样落盘**（不拼接公共前缀），"
            "落盘路径 = {oss_prefix}/{yyyyMMdd}/{upload_id}/{filename}"
            "（upload_id 为当日上传顺序号，1 起递增）。响应回填带 scheme "
            "全路径 oss://{bucket}/{前缀}，前端原样写入数据集文档；评测提交"
            "透传后拉取侧原样保留、loader 剔 scheme 即命中落盘位置"
        ),
    )
    source_format: Literal["csv", "json"] | None = Field(
        default=None,
        description=(
            "benchmark 数据集源格式。仅显式 json 时规范化为标准 CSV；"
            "缺省或 csv 保持原样落盘（含 *.json 扩展名）"
        ),
    )
    content_type: str | None = Field(
        default=None,
        description="显式 Content-Type；缺省按文件名猜测",
    )
    sign: bool = Field(
        default=False,
        description="true 时额外返回预签名 GET URL",
    )
    sign_expires_seconds: int = Field(
        default=_DEFAULT_SIGN_EXPIRES_SECONDS,
        ge=60,
        le=30 * 24 * 60 * 60,
        description="预签名有效期（秒），默认 30 天",
    )


@dataclass(frozen=True)
class _UploadSourceSpec:
    """上传源格式与实际落盘文件名。"""

    format: Literal["csv", "json"]
    filename: str


def _upload_source_spec(form: StorageUploadForm, basename: str) -> _UploadSourceSpec:
    # 仅显式 source_format=json 才规范化；避免 oss_prefix 下 *.json 被静默改写。
    source_format: Literal["csv", "json"] = form.source_format or "csv"
    filename = (
        _BENCHMARK_DATASET_CSV_NAME
        if form.oss_prefix and form.oss_prefix.strip() and source_format == "json"
        else basename
    )
    return _UploadSourceSpec(format=source_format, filename=filename)


def _normalize_upload_data(
    data: bytes,
    *,
    source: _UploadSourceSpec,
    dataset_key_path: str | None,
) -> bytes:
    if dataset_key_path is not None and source.format == "json":
        try:
            return normalize_dataset_json(data)
        except DatasetNormalizationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return data


def _upload_content_type(
    form: StorageUploadForm,
    *,
    source: _UploadSourceSpec,
    dataset_key_path: str | None,
    fallback: str | None,
) -> str:
    if dataset_key_path is not None and source.format == "json":
        return "text/csv"
    if form.content_type:
        return form.content_type
    guessed, _ = mimetypes.guess_type(source.filename)
    return guessed or fallback or "application/octet-stream"


async def _upload_form(request: Request) -> StorageUploadForm:
    """从 multipart 文本字段解析上传选项。

    FastAPI 不支持「Form 模型 + File 参数」在同一端点混用：模型会被当成名为
    ``form`` 的单个表单字段（JSON 字符串）——带默认值时更隐蔽，所有文本字段
    被静默丢弃。故直接读已解析的表单字段（文件字段跳过）再交给模型校验。
    """
    form = await request.form()
    values = {key: value for key, value in form.multi_items() if isinstance(value, str)}
    try:
        return StorageUploadForm.model_validate(values)
    except ValidationError as exc:
        raise RequestValidationError(
            [{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]
        ) from exc


@router.post("/upload")
async def upload_file(
    file: Annotated[UploadFile, File(description="待上传文件")],
    form: Annotated[StorageUploadForm, Depends(_upload_form)],
) -> dict[str, Any]:
    """上传用户给定文件到 OSS（RemoteConfig ``agent-eval.benchmark.oss``）。"""
    client = build_data_storage_client()
    if client is None:
        logger.warning(
            "storage.upload_unavailable filename=%r (RemoteConfig agent-eval.benchmark.oss "
            "incomplete or libs/storage not installed)",
            file.filename,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "OSS storage unavailable: RemoteConfig agent-eval.benchmark.oss "
                "incomplete or libs/storage not installed"
            ),
        )

    basename = _safe_basename(file.filename)
    if form.source_format == "json" and not (
        isinstance(form.oss_prefix, str) and form.oss_prefix.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="source_format=json requires oss_prefix (benchmark dataset upload)",
        )
    source = _upload_source_spec(form, basename)
    # benchmark 语义前缀：**照原样落盘**——不拼公共前缀（业务要求：
    # amap_agent_ieval 就落 amap_agent_ieval/{date}/，不加
    # ieval-service/benchmark/benchmark_result/）。仅做安全校验（拒绝 ..
    # 段），含 scheme 时剔后取路径。响应回填 **带 scheme 全路径**
    # （oss://{bucket}/{裸路径}）：前端原样回填 dataset.ossPrefix，评测
    # 提交透传后拉取侧 resolve_oss_prefix 对含 scheme 值原样保留（裸路径
    # 同样原样保留，不拼公共前缀）→ loader 剔 scheme 后得到裸路径 →
    # 与上传落盘位置恒等。
    resolved_oss_prefix: str | None = None
    partition_date: str | None = None
    upload_id: int | None = None
    dataset_key_path: str | None = None
    if isinstance(form.oss_prefix, str) and form.oss_prefix.strip():
        target = await _resolve_benchmark_upload(
            oss_prefix=form.oss_prefix, basename=source.filename, filename=file.filename
        )
        key = target.key
        resolved_oss_prefix = target.oss_prefix
        partition_date = target.partition_date
        upload_id = target.upload_id
        dataset_key_path = target.key_path
    else:
        key = _resolve_object_key(filename=basename, prefix=form.prefix, object_key=form.object_key)
        logger.info(
            "storage.upload_received mode=raw_key filename=%r prefix=%r object_key=%r "
            "resolved_key=%s",
            file.filename,
            form.prefix,
            form.object_key,
            key,
        )

    data = await _read_upload_limited(file, max_bytes=_MAX_UPLOAD_BYTES)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    data = _normalize_upload_data(
        data,
        source=source,
        dataset_key_path=dataset_key_path,
    )
    if dataset_key_path is not None:
        # benchmark 数据集：落 OSS 前校验必备列（缺列直接 400，不产生对象）
        _validate_dataset_columns(
            basename=source.filename,
            data=data,
            key_path=dataset_key_path,
        )

    resolved_ct = _upload_content_type(
        form,
        source=source,
        dataset_key_path=dataset_key_path,
        fallback=file.content_type,
    )

    storage: _StorageUploader = client  # type: ignore[assignment]
    url = await storage.upload(key, data, content_type=resolved_ct)
    if url is None:
        logger.error(
            "storage.upload_failed object_key=%s bytes=%d (OSS client returned None)",
            key,
            len(data),
        )
        raise HTTPException(status_code=502, detail="OSS upload failed")

    logger.info(
        "storage.upload_done object_key=%s bytes=%d content_type=%s url=%s",
        key,
        len(data),
        resolved_ct,
        url,
    )

    result: dict[str, Any] = {
        "object_key": key,
        "url": url,
        "bytes": len(data),
        "content_type": resolved_ct,
        "filename": source.filename,
    }
    if resolved_oss_prefix is not None:
        result["oss_prefix"] = resolved_oss_prefix
    if partition_date is not None:
        # 分区日期 + 上传批次段随响应返回：前端写入数据集文档
        # ossPartitionDate / ossUploadId，触发评测时服务端据此拼完整
        # object key（{prefix}/{date}/{upload_id}/{file}）；日期以服务
        # 端时钟为准，避免前端本地时区推算差一天。
        result["partition_date"] = partition_date
        result["upload_id"] = upload_id
    if form.sign:
        signed = await storage.sign_url(key, expires_seconds=form.sign_expires_seconds)
        if signed is None:
            raise HTTPException(status_code=502, detail="OSS upload ok but sign_url failed")
        result["signed_url"] = signed
        result["sign_expires_seconds"] = form.sign_expires_seconds

    return {"data": result}


def _strip_oss_scheme(oss_prefix: str) -> str:
    """从 ``oss://bucket/path`` 提取 object key 路径（与 loader 同构）。"""
    text = oss_prefix.strip()
    if text.startswith("oss://"):
        without_scheme = text[6:]
        slash_idx = without_scheme.find("/")
        if slash_idx != -1:
            return without_scheme[slash_idx + 1 :]
        return ""
    return text


def _bucket_scheme_prefix() -> str:
    """``oss://{bucket}`` 前缀（从 RemoteConfig 默认下载前缀截取 scheme+bucket 段，

    不含公共路径）——回填给前端的 oss_prefix 以全路径形态返回，拉取侧
    resolve_oss_prefix 对含 scheme 值原样保留，loader 剔 scheme 后与
    上传落盘的裸路径恒等。
    """
    default = get_default_download_prefix()  # oss://{bucket}/{common_prefix}
    parts = default.split("/")
    return "/".join(parts[:3])  # oss:// + '' + bucket


@router.get("/oss-entries")
async def list_oss_entries(
    dataset_id: str,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> dict[str, Any]:
    """按数据集 OSS 前缀列出最近 N 条（默认 5）日期分区。

    只读：仅列目录/对象，不下载内容。日期宏在列目录前剔除，返回的
    ``path`` 是 OSS 中**真实存在**的日期分区前缀（不含 bucket），页面上
    直接把宏替换成真实日期展示。

    - 数据集不存在 / 非 OSS 来源 → 404
    - 前缀为空 → 空列表（合法：用平台默认前缀）
    - OSS 不可达 / 凭证缺失 → 503（不暴露底层异常文本）
    """
    from benchmark.domain.benchmark_console import find_dataset_document_by_id  # noqa: PLC0415
    from benchmark.runs.oss_prefix_listing import (  # noqa: PLC0415
        list_recent_oss_entries,
    )

    document = await find_dataset_document_by_id(dataset_id)
    if document is None or document.source != "oss":
        raise HTTPException(status_code=404, detail=f"oss dataset not found: {dataset_id}")
    prefix = (document.oss_prefix or "").strip()
    if not prefix:
        return {"dataset_id": dataset_id, "entries": []}
    try:
        entries = await list_recent_oss_entries(prefix, limit=limit)
    except Exception:
        logger.warning(
            "storage.oss_entries_failed dataset_id=%s prefix=%r",
            dataset_id,
            prefix,
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail="OSS listing unavailable: bucket/credential not ready",
        ) from None
    return {
        "dataset_id": dataset_id,
        "entries": [
            {"path": item.path, "date": item.date, "modified_at": item.modified_at}
            for item in entries
        ],
    }


__all__ = ["router"]
