"""Reassign the case-list assignee: update the Aone ticket and write back
``notification_dispatch_json.notification_assigned_to``.
"""

import re

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.models import TraceRecord
from benchmark.domain.notification_dispatch_resolve import (
    load_agent_notification_context,
    resolve_effective_notification_targets,
)
from benchmark.attribution.badcase.publisher import NotificationConfig, update_workitem_assignee
from benchmark.runs.run_case_page import (
    NOTIFICATION_SOURCE_TRACE,
    notification_dispatch_fields,
    load_item_notification_row,
    resolve_notification_source,
    run_config_dict,
    trace_case_in_run,
)

logger = get_logger(__name__)

#: Nickname or numeric employee id (aligned with the WebUI / AgentDocument owners rules).
_ASSIGNEE_PATTERN = re.compile(
    r"^(?:[0-9]{1,32}|[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_.\-]{0,63})$"
)


class NotificationAssigneeError(Exception):
    """Reassignment failed (mapped to HTTP 4xx/5xx)."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


async def _load_trace_row(case_id: str) -> TraceRecord | None:
    return await TraceRecord.filter(trace_id=case_id).first()


async def reassign_case_notification_assignee(
    _service: object,
    run: object,
    *,
    case_id: str,
    assignee: str,
) -> dict[str, str]:
    """Reassign and write the bucket back. Returns the updated ref / url / assigned_to.

    ``assignee`` is a single nickname / employee id typed by the user, and no longer has to
    be one of the agent owners.

    Do not call ``load_report_cases`` (a full load can 502 the gateway). After a light
    ownership check against the persisted aone row, read the row and reassign directly.
    ``_service`` is kept so the signature matches the API layer.
    """
    person = assignee.strip()
    if not person:
        raise NotificationAssigneeError(400, "assignee 不能为空")
    if not _ASSIGNEE_PATTERN.match(person):
        raise NotificationAssigneeError(400, "assignee 必须是花名或数字工号")

    config = run_config_dict(run)
    run_id = str(getattr(run, "run_id", "") or "")
    notification_source = resolve_notification_source(config)

    if notification_source == NOTIFICATION_SOURCE_TRACE:
        if not await trace_case_in_run(run, case_id):
            raise NotificationAssigneeError(404, f"case not found: {case_id}")
        row = await _load_trace_row(case_id)
    else:
        row = await load_item_notification_row(run_id, case_id)
    if row is None:
        raise NotificationAssigneeError(404, f"case not found: {case_id}")

    fields = notification_dispatch_fields(getattr(row, "notification_dispatch_json", None))
    workitem_id = fields.get("notification_ref")
    notification_url = fields.get("notification_url")
    if not workitem_id or not notification_url:
        raise NotificationAssigneeError(400, "该 case 没有 aone url，无法改指派人")

    agent_name, snapshot = await load_agent_notification_context(config)
    notification_project_id, _default_assignee, _parts, _dispatch_enabled = resolve_effective_notification_targets(
        agent_name,
        snapshot,
    )
    # update locates the work item by id and does not require project; still pass the resolved project through for
    # logs/compatibility.
    notification_config = NotificationConfig(
        enabled=True,
        dry_run=False,
        project=notification_project_id or "",
    )

    try:
        await update_workitem_assignee(workitem_id, person, notification_config)
    except Exception as exc:
        logger.warning(
            "aone.reassign_failed case_id=%s workitem_id=%s err=%s",
            case_id,
            workitem_id,
            exc,
        )
        raise NotificationAssigneeError(502, f"Aone 改指派失败：{exc}") from exc

    bucket = dict(getattr(row, "notification_dispatch_json", None) or {})
    bucket["notification_assigned_to"] = person
    row.notification_dispatch_json = bucket
    await row.save()
    return {
        "notification_ref": workitem_id,
        "notification_url": notification_url,
        "notification_assigned_to": person,
    }


__all__ = [
    "NotificationAssigneeError",
    "reassign_case_notification_assignee",
]
