# -*- coding: utf-8 -*-
"""Aggregate notifications by evaluation root cause and hand each group to its configured audience.

Within one evaluation batch, items with the same ``root_cause_code`` become one notification.
The audience comes from the profile's ``root_cause_groups``: an exact root cause first, then ``*``,
then the profile's assignee and participants. The first member is the assignee; the rest are participants.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.attribution.badcase.schemas import (
    AttributionResult,
    NotificationDispatchResult,
    NotificationDispatchStatus,
)
from benchmark.notification.registry import NotificationPublisher

UNCLASSIFIED_ROOT_CAUSE = "unclassified"


@dataclass(frozen=True)
class NotificationCase:
    """One evaluation result that takes part in aggregation."""

    item_id: str
    root_cause: str
    summary: str = ""


@dataclass(frozen=True)
class AggregatedNotification:
    """A notification for one root cause: the audience and the evaluation items it covers."""

    root_cause: str
    members: tuple[str, ...]
    cases: tuple[NotificationCase, ...]


def case_from_attribution(result: AttributionResult) -> NotificationCase:
    """Prefer the formal root-cause code; without a code, use the root-cause text; if both are empty, file it as unclassified."""

    code = (result.root_cause_code or "").strip()
    text = (result.root_cause or "").strip()
    root_cause = code or text or UNCLASSIFIED_ROOT_CAUSE
    return NotificationCase(
        item_id=(result.case_id or result.trace_id or "").strip(),
        root_cause=root_cause,
        summary=(result.query or result.reasoning or "").strip(),
    )


def aggregate_by_root_cause(
    cases: Sequence[NotificationCase],
    groups: Mapping[str, Sequence[str]],
    *,
    default_members: Sequence[str] = (),
) -> list[AggregatedNotification]:
    """Group by root-cause appearance order and resolve an audience for each group."""

    order: list[str] = []
    buckets: dict[str, list[NotificationCase]] = {}
    for case in cases:
        key = case.root_cause.strip() or UNCLASSIFIED_ROOT_CAUSE
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(NotificationCase(case.item_id, key, case.summary))
    return [
        AggregatedNotification(
            root_cause=key,
            members=_members(key, groups, default_members),
            cases=tuple(buckets[key]),
        )
        for key in order
    ]


async def publish_root_cause_groups(
    results: Sequence[AttributionResult],
    *,
    groups: Mapping[str, Sequence[str]],
    notification_config: NotificationConfig,
    publisher: NotificationPublisher,
    default_members: Sequence[str] = (),
) -> list[NotificationDispatchResult]:
    """Send one notification per root-cause audience. A group with no members is recorded as skipped."""

    subjects = [case_from_attribution(result) for result in results]
    dispatched: list[NotificationDispatchResult] = []
    for group in aggregate_by_root_cause(subjects, groups, default_members=default_members):
        if not group.members:
            dispatched.append(NotificationDispatchResult(
                status=NotificationDispatchStatus.SKIPPED,
                message="root_cause_group_unassigned",
            ))
            continue
        config = notification_config.model_copy(update={
            "assignee": group.members[0],
            "participants": list(group.members[1:]),
        })
        dispatched.append(await publisher.create_task(
            notification_config=config,
            attribution_result=_group_attribution(group),
            resolved_route=None,
        ))
    return dispatched


def _case_line(case: NotificationCase) -> str:
    if case.item_id and case.summary:
        return f"{case.item_id}: {case.summary}"
    return case.item_id or case.summary


def _members(
    root_cause: str,
    groups: Mapping[str, Sequence[str]],
    default_members: Sequence[str],
) -> tuple[str, ...]:
    chosen = groups.get(root_cause)
    if chosen is None:
        chosen = groups.get("*", default_members)
    seen: list[str] = []
    for item in chosen:
        text = str(item).strip()
        if text and text not in seen:
            seen.append(text)
    return tuple(seen)


def _group_attribution(group: AggregatedNotification) -> AttributionResult:
    lines = [_case_line(case) for case in group.cases]
    code = None if group.root_cause == UNCLASSIFIED_ROOT_CAUSE else group.root_cause
    return AttributionResult(
        case_id=group.root_cause,
        category="unknown",
        root_cause=group.root_cause,
        root_cause_code=code,
        query="\n".join(line for line in lines if line),
        reasoning=f"共 {len(group.cases)} 条",
        should_dispatch=True,
    )
