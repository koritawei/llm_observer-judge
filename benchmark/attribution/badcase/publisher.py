"""归因结果到 Aone 工单创建 — 顶层函数 + 内部 helper。"""

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import AliasChoices, BaseModel, Field

from benchmark.domain.enums import CasePriority
from benchmark.domain.owner_mapping import resolve_route_assignment
from benchmark.attribution.badcase.notification_content import (
    NotificationWorkitemOverrides,
    build_workitem_payload,
)
from benchmark.attribution.badcase.schemas import (
    NotificationDispatchResult,
    NotificationDispatchStatus,
    NotificationRuntimeOptions,
    NotificationWorkitemPayload,
    AttributionResult,
)
from benchmark.runs.config import a1_cli_env_overrides, notification_workitem_url_env_overrides

if TYPE_CHECKING:
    from benchmark.domain.owner_mapping import OwnerMapping

logger = get_logger(__name__)

_DEFAULT_WORKITEM_URL_TEMPLATE = (
    "https://project.aone.alibaba-inc.com/v2/project/{project_id}/{category}"
    "#openWorkitemIdentifier={workitem_id}"
)
_LEGACY_WORKITEM_URL_RE = re.compile(
    r"^(?P<base>https?://[^/]+/v2/project/\d+)/(?P<category>[a-zA-Z][a-zA-Z0-9_-]*)/(?P<id>\d+)/?$"
)


class NotificationConfig(NotificationRuntimeOptions):
    """Aone 创建配置。

    ``dispatch_profile`` 来自 yaml ``project`` 逻辑名；``project`` 为 RemoteConfig 解析后的
    Aone 项目 ID，供 ``a1 --project`` 使用。

    ``enabled`` 默认 False（继承 :class:`AoneRuntimeOptions`）。daily/pre 默认关；
    prod 或显式 yaml/per-run ``enabled=true`` 才建单。注册文档
    ``notification_dispatch_enabled=false`` 可关闭单 agent。
    """

    dispatch_profile: str = ""


class NotificationCliData(BaseModel):
    """Aone CLI JSON 输出里的工单标识与链接字段。"""

    id: str | int | None = None
    identifier: str | int | None = None
    workitem_id: str | int | None = Field(
        default=None, validation_alias=AliasChoices("workitem_id", "workItemId")
    )
    url: str | None = None
    workitem_url: str | None = Field(
        default=None, validation_alias=AliasChoices("workitem_url", "workitemUrl")
    )
    html_url: str | None = None


class NotificationCliResponse(NotificationCliData):
    """Aone CLI JSON 输出。"""

    data: NotificationCliData | None = None


class NotificationCreateError(RuntimeError):
    """Aone CLI 在 payload 校验通过后执行失败。"""


@dataclass(frozen=True)
class NotificationWorkitemReference:
    """创建成功的 Aone 工作项标识及可打开 URL。"""

    workitem_id: str
    url: str


def map_priority_to_notification(priority: CasePriority | None) -> str:
    """映射 CasePriority → a1 CLI ``--priority`` 接受的字符串。

    a1 接受 ``urgent / high / medium / low``；当前覆盖 P0/P1/P2/None 四种入参。
    ``None`` 视为 ``medium``（与 P2 同档）。
    """
    return {
        CasePriority.P0: "urgent",
        CasePriority.P1: "high",
        CasePriority.P2: "medium",
        None: "medium",
    }[priority]


async def create_notification_task(
    *,
    notification_config: NotificationConfig,
    attribution_result: AttributionResult,
    resolved_route: "OwnerMapping | None" = None,
) -> NotificationDispatchResult:
    """按归因结果创建 Aone 工单。

    Args:
        resolved_route: OwnerMapping 命中结果（trace 级路由）。单人 assigned_to、
            priority、project_url 覆盖 aone_config 兜底。多人由 assign_type 决定：
            ``random`` 抽一人指派；``all`` 第一人指派、其余并入参与人。
            未命中或字段为空时走 aone_config 兜底。

    Returns:
        :class:`AoneDispatchResult`, 状态枚举: skipped / blocked / created / failed。
    """

    assignee, participants = resolve_route_assignment(
        resolved_route,
        fallback_assignee=notification_config.assignee,
        fallback_participants=notification_config.participants,
    )
    priority_value = (
        map_priority_to_notification(resolved_route.priority)
        if (resolved_route and resolved_route.priority is not None)
        else ""
    )
    project_url = (
        resolved_route.project_url if resolved_route and resolved_route.project_url else ""
    )

    payload = build_workitem_payload(
        attribution_result,
        project=notification_config.project,
        category=notification_config.category,
        options=notification_config,
        overrides=NotificationWorkitemOverrides(
            assignee=assignee,
            priority=priority_value,
            project_url=project_url,
            participants=participants,
        ),
    )
    if not notification_config.enabled:
        logger.info(
            "aone.dispatch_skipped reason=disabled profile=%s title=%s",
            notification_config.dispatch_profile or "-",
            payload.title,
        )
        return NotificationDispatchResult(
            status=NotificationDispatchStatus.SKIPPED,
            message="Aone dispatch is disabled.",
            payload=payload,
        )
    if notification_config.dry_run:
        logger.info(
            "aone.dispatch_skipped reason=dry_run profile=%s title=%s",
            notification_config.dispatch_profile or "-",
            payload.title,
        )
        return NotificationDispatchResult(
            status=NotificationDispatchStatus.SKIPPED,
            message="Aone dry_run enabled; workitem was not created.",
            payload=payload,
        )

    validation_error = _validate_payload(payload, notification_config)
    if validation_error is not None:
        logger.warning(
            "aone.dispatch_blocked profile=%s reason=%s title=%s",
            notification_config.dispatch_profile or "-",
            validation_error,
            payload.title,
        )
        return NotificationDispatchResult(
            status=NotificationDispatchStatus.BLOCKED,
            message=validation_error,
            payload=payload,
        )

    try:
        # 走 reference：保留 a1 JSON / 模板拼出的可打开 URL，供下游写 notification_url。
        reference = await create_workitem_reference(payload, notification_config)
    except Exception as exc:
        logger.exception(
            "aone.dispatch_failed profile=%s title=%s err=%s",
            notification_config.dispatch_profile or "-",
            payload.title,
            exc,
        )
        return NotificationDispatchResult(
            status=NotificationDispatchStatus.FAILED,
            message=str(exc),
            payload=payload,
        )
    logger.info(
        "aone.dispatch_created profile=%s workitem_id=%s url=%s title=%s",
        notification_config.dispatch_profile or "-",
        reference.workitem_id,
        reference.url or "-",
        payload.title,
    )
    return NotificationDispatchResult(
        status=NotificationDispatchStatus.CREATED,
        workitem_id=reference.workitem_id,
        workitem_url=reference.url,
        message="Aone workitem created.",
        payload=payload,
    )


def _resolve_a1_bin(configured: str) -> str:
    """解析 ``a1`` 可执行文件路径（本地调试 env 优先于 yaml，见 setting.a1_cli_env_overrides）。"""
    env_bin, _ = a1_cli_env_overrides()
    for candidate in (env_bin, configured):
        if candidate and (os.path.isabs(candidate) or shutil.which(candidate)):
            return candidate
    return configured or "a1"


def _a1_subprocess_cwd() -> str | None:
    """``a1 link`` 目录；未配置时跟随父进程 cwd。"""
    _, project_dir = a1_cli_env_overrides()
    if project_dir and os.path.isdir(project_dir):
        return project_dir
    return None


async def create_workitem(payload: NotificationWorkitemPayload, config: NotificationConfig) -> str:
    """调用本地 `a1` CLI 创建工单并返回工单 ID（兼容既有调用方）。"""

    return (await create_workitem_reference(payload, config)).workitem_id


async def create_workitem_reference(
    payload: NotificationWorkitemPayload,
    config: NotificationConfig,
) -> NotificationWorkitemReference:
    """创建工作项并返回 ID 与可打开 URL。"""

    cmd = [
        _resolve_a1_bin(config.a1_bin),
        "project",
        "workitem",
        "create",
        "--category",
        payload.category,
        "--title",
        payload.title,
        "--body",
        payload.body,
        "--format",
        "json",
    ]
    if payload.project:
        cmd.extend(["--project", payload.project])
    workitem_type = _sanitize_workitem_type(payload.type, payload.category)
    if workitem_type:
        cmd.extend(["--type", workitem_type])
    if payload.assignee:
        cmd.extend(["--assignee", payload.assignee])
    if payload.participants:
        cmd.extend(["--participant", ",".join(payload.participants)])
    if payload.priority:
        cmd.extend(["--priority", payload.priority])
    for tag in payload.tags:
        cmd.extend(["--tag", tag])
    for path in payload.attachments:
        cmd.extend(["--attachment", path])
    if payload.parent_workitem_id:
        cmd.extend(["--relation", f"parent:{payload.parent_workitem_id}"])

    output = await _run_a1_command(cmd, config, action="create")
    workitem_id = _parse_workitem_id(output)
    return NotificationWorkitemReference(
        workitem_id=workitem_id,
        url=_parse_workitem_url(output)
        or _build_workitem_url(
            workitem_id,
            project=payload.project or config.project,
            category=payload.category,
        ),
    )


async def update_workitem_body(
    workitem_id: str,
    body: str,
    config: NotificationConfig,
) -> None:
    """回写主工单正文，不创建任何新工作项。

    ``a1 project workitem update`` 以 workitem id 定位，不接受 ``--project``。
    """

    cmd = [
        _resolve_a1_bin(config.a1_bin),
        "project",
        "workitem",
        "update",
        str(workitem_id),
        "--body",
        body,
        "--format",
        "json",
    ]
    await _run_a1_command(cmd, config, action="update")


async def update_workitem_assignee(
    workitem_id: str,
    assignee: str,
    config: NotificationConfig,
) -> None:
    """修改已有工单指派人（花名 / 工号 / 邮箱，a1 均接受）。

    ``a1 project workitem update`` 不接受 ``--project``（传了会
    ``unknown flag: --project`` → 网关侧看到 502）。workitem id 已足够定位。
    """

    person = assignee.strip()
    if not person:
        raise NotificationCreateError("assignee is required")
    cmd = [
        _resolve_a1_bin(config.a1_bin),
        "project",
        "workitem",
        "update",
        str(workitem_id),
        "--assignee",
        person,
        "--format",
        "json",
    ]
    await _run_a1_command(cmd, config, action="update-assignee")


async def _run_a1_command(cmd: list[str], config: NotificationConfig, *, action: str) -> str:
    """执行单次 a1 命令并统一处理超时与失败。"""

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=_a1_subprocess_cwd(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=config.timeout_seconds
        )
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise NotificationCreateError(f"Aone {action} timed out.") from exc
    if process.returncode != 0:
        raise NotificationCreateError(_decode(stderr) or _decode(stdout) or f"Aone {action} failed.")
    return _decode(stdout)


#: a1 接受的 category 大类；误把这些值塞给 --type 会报 unknown workitem type。
_NOTIFICATION_CATEGORY_VALUES: frozenset[str] = frozenset(
    {"req", "bug", "task", "risk", "nodeflownode", "plugintask"}
)


def _sanitize_workitem_type(workitem_type: str, category: str) -> str:
    """过滤误配：``--type`` 不能等于 ``--category`` 大类值。

    a1 的 ``--type`` 接受项目内具体子类型名（如 ``"缺陷(BUG)"``、``"故事(STO)"``），
    与 ``--category`` 同值时 Aone 会返回 ``unknown workitem type "<x>" for <Category>``。
    """
    if not workitem_type or not workitem_type.strip():
        return ""
    normalized = workitem_type.strip().lower()
    if normalized in _NOTIFICATION_CATEGORY_VALUES:
        logger.warning(
            "workitem_type %r 与 category 大类重名，跳过 --type（走项目默认子类型）",
            workitem_type,
        )
        return ""
    return workitem_type


def _validate_payload(
    payload: NotificationWorkitemPayload,
    config: NotificationConfig,
) -> str | None:
    """创建前校验必要字段,缺失时返回阻塞原因。

    仅做配置完整性校验（project / title / body / assignee），不做业务
    过滤——派单判定唯一依据是分数门槛（``should_dispatch_attribution``），
    归因 category=unknown（judge 回传缺失/脏值）照常建单。assignee 空
    → 黑洞工单（建了无人被指派/抄送，通知链断裂），拦截逼配置补齐。
    """

    if not config.project:
        profile = config.dispatch_profile or "unknown"
        return f"Aone project id missing for dispatch profile {profile!r}."
    if not payload.title:
        return "Aone title is required."
    if not payload.body:
        return "Aone body is required."
    if not payload.assignee:
        profile = config.dispatch_profile or "unknown"
        return (
            f"Aone assignee missing for dispatch profile {profile!r} — "
            "workitem would have no one to notify."
        )
    return None


def _parse_workitem_id(text: str) -> str:
    """从 Aone CLI 的 JSON 输出解析 workitem id；无法解析即抛错。

    不做正则兜底：CLI 静默失败（returncode=0 但输出错误文本）时，文本中
    的任意数字（错误码/请求 ID/时间戳）会被误抓成工单号，落库假
    ``notification_ref`` 且误判 CREATED。非 JSON / 无 id 字段一律抛
    :class:`AoneCreateError`，走 FAILED 可见路径。
    """

    try:
        payload = NotificationCliResponse.model_validate_json(text)
    except ValueError as exc:
        raise NotificationCreateError(f"Aone CLI stdout is not JSON: {text[:200]!r}") from exc
    for value in (
        payload.id,
        payload.identifier,
        payload.workitem_id,
        payload.data.id if payload.data else None,
        payload.data.identifier if payload.data else None,
        payload.data.workitem_id if payload.data else None,
    ):
        if value:
            return str(value)
    raise NotificationCreateError(f"Aone CLI stdout has no workitem id: {text[:200]!r}")


def _parse_workitem_url(text: str) -> str:
    """从 a1 JSON 响应抽取并正规化 URL。"""

    try:
        payload = NotificationCliResponse.model_validate(json.loads(text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    for value in (
        payload.url,
        payload.workitem_url,
        payload.html_url,
        payload.data.url if payload.data else None,
        payload.data.workitem_url if payload.data else None,
        payload.data.html_url if payload.data else None,
    ):
        if value:
            return _normalize_workitem_url(value)
    return ""


def _build_workitem_url(workitem_id: str, *, project: str, category: str) -> str:
    """在 a1 未返回 URL 时使用显式配置或当前项目构造可打开链接。"""

    template_override, fallback_project = notification_workitem_url_env_overrides()
    template = (
        template_override if template_override is not None else _DEFAULT_WORKITEM_URL_TEMPLATE
    )
    project_id = project or fallback_project or ""
    if not project_id:
        return ""
    fields = {
        "project_id": project_id,
        "category": category or "workitem",
        "workitem_id": workitem_id,
    }
    try:
        return template.format(**fields)
    except (LookupError, ValueError, AttributeError, TypeError) as exc:
        logger.warning(
            "aone.workitem_url_template_invalid error=%s; using default URL",
            type(exc).__name__,
        )
        return _DEFAULT_WORKITEM_URL_TEMPLATE.format(**fields)


def _normalize_workitem_url(url: str) -> str:
    """将 a1 旧 path URL 转为新版 Aone hash 路由。"""

    stripped = url.strip()
    if "#openWorkitemIdentifier=" in stripped:
        return stripped
    match = _LEGACY_WORKITEM_URL_RE.match(stripped)
    if match:
        return (
            f"{match.group('base')}/{match.group('category')}"
            f"#openWorkitemIdentifier={match.group('id')}"
        )
    return stripped


def _decode(data: bytes | None) -> str:
    """将 CLI stdout/stderr bytes 解码成人可读字符串。"""

    return (data or b"").decode("utf-8", errors="replace").strip()


__all__ = [
    "NotificationConfig",
    "NotificationCreateError",
    "NotificationWorkitemReference",
    "create_notification_task",
    "create_workitem",
    "create_workitem_reference",
    "map_priority_to_notification",
    "update_workitem_assignee",
    "update_workitem_body",
]
