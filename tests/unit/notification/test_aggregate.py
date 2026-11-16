# -*- coding: utf-8 -*-
"""Root-cause aggregate notifications."""

import pytest

from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.attribution.badcase.schemas import (
    AttributionResult,
    NotificationDispatchResult,
    NotificationDispatchStatus,
)
from benchmark.attribution.service.service import AfterRunInput, NotificationAfterRunHook
from benchmark.notification.aggregate import (
    NotificationCase,
    aggregate_by_root_cause,
    case_from_attribution,
    publish_root_cause_groups,
)


class _Request:
    agent_name = "shop"
    trace_id = "trace-1"
    project_id = "project-1"


class _Publisher:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    async def create_task(
        self,
        *,
        notification_config: object,
        attribution_result: object,
        resolved_route: object = None,
    ) -> NotificationDispatchResult:
        del resolved_route
        self.calls.append((notification_config, attribution_result))
        return NotificationDispatchResult(
            status=NotificationDispatchStatus.CREATED,
            workitem_id="W-1",
            message="created",
        )


def test_aggregate_by_root_cause_keeps_first_seen_order() -> None:
    groups = {"tool_error": ["alice", "bob"], "*": ["fallback"]}
    aggregated = aggregate_by_root_cause(
        [
            NotificationCase("a", "tool_error", "查库存"),
            NotificationCase("b", "knowledge_gap", "沸点"),
            NotificationCase("c", "tool_error", "再查"),
            NotificationCase("d", "", "没写根因"),
        ],
        groups,
        default_members=("owner",),
    )
    assert [item.root_cause for item in aggregated] == [
        "tool_error",
        "knowledge_gap",
        "unclassified",
    ]
    assert aggregated[0].members == ("alice", "bob")
    assert [case.item_id for case in aggregated[0].cases] == ["a", "c"]
    assert aggregated[1].members == ("fallback",)
    assert aggregated[2].members == ("fallback",)
    without_star = aggregate_by_root_cause(
        [NotificationCase("d", "", "没写根因")],
        {"tool_error": ["alice"]},
        default_members=("owner",),
    )
    assert without_star[0].members == ("owner",)


def test_case_from_attribution_prefers_root_cause_code() -> None:
    case = case_from_attribution(AttributionResult(
        case_id="case-1",
        root_cause="工具失败",
        root_cause_code="tool_error",
        query="查库存",
    ))
    assert case == NotificationCase("case-1", "tool_error", "查库存")


@pytest.mark.asyncio
async def test_publish_root_cause_groups_sends_one_notice_per_cause() -> None:
    publisher = _Publisher()
    results = await publish_root_cause_groups(
        [
            AttributionResult(case_id="a", root_cause_code="tool_error", query="查库存"),
            AttributionResult(case_id="b", root_cause_code="knowledge_gap", query="沸点"),
            AttributionResult(case_id="c", root_cause_code="tool_error", query="再查"),
        ],
        groups={"tool_error": ["alice", "bob"], "knowledge_gap": ["carol"]},
        notification_config=NotificationConfig(enabled=True, project="p1"),
        publisher=publisher,
    )
    assert [item.status for item in results] == [
        NotificationDispatchStatus.CREATED,
        NotificationDispatchStatus.CREATED,
    ]
    first_config, first_result = publisher.calls[0]
    second_config, second_result = publisher.calls[1]
    assert first_config.assignee == "alice"
    assert first_config.participants == ["bob"]
    assert first_result.root_cause_code == "tool_error"
    assert "a: 查库存" in first_result.query
    assert "c: 再查" in first_result.query
    assert second_config.assignee == "carol"
    assert second_config.participants == []
    assert second_result.case_id == "knowledge_gap"


@pytest.mark.asyncio
async def test_publish_skips_group_without_members() -> None:
    publisher = _Publisher()
    results = await publish_root_cause_groups(
        [AttributionResult(case_id="a", root_cause_code="tool_error")],
        groups={},
        notification_config=NotificationConfig(enabled=True),
        publisher=publisher,
    )
    assert results[0].status == NotificationDispatchStatus.SKIPPED
    assert results[0].message == "root_cause_group_unassigned"
    assert publisher.calls == []


@pytest.mark.asyncio
async def test_hook_flushes_pending_cases_by_root_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = _Publisher()
    monkeypatch.setattr(
        "benchmark.attribution.service.service.resolve_notification_root_cause_groups",
        lambda *_args, **_kwargs: {"tool_error": ("alice", "bob"), "knowledge_gap": ("carol",)},
    )
    monkeypatch.setattr(
        "benchmark.attribution.service.service.resolve_effective_notification_targets",
        lambda *_args, **_kwargs: ("proj-1", "", [], True),
    )
    monkeypatch.setattr(
        "benchmark.attribution.service.service.resolve_notification_publisher",
        lambda *_args, **_kwargs: publisher,
    )
    monkeypatch.setattr(
        "benchmark.attribution.service.service.load_agent_notification_context",
        _empty_context,
    )
    hook = NotificationAfterRunHook(notification_config=NotificationConfig(enabled=True, project="p1"))
    first = await hook(_input(AttributionResult(case_id="a", root_cause_code="tool_error", query="查库存")))
    second = await hook(_input(AttributionResult(
        case_id="b",
        root_cause_code="knowledge_gap",
        query="沸点",
    )))
    assert first.reason == "root_cause_group_pending"
    assert second.reason == "root_cause_group_pending"
    dispatched = await hook.flush_groups()
    assert len(dispatched) == 2
    assert [call[0].assignee for call in publisher.calls] == ["alice", "carol"]
    assert await hook.flush_groups() == []


@pytest.mark.asyncio
async def test_hook_still_dispatches_safety_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = _Publisher()
    monkeypatch.setattr(
        "benchmark.attribution.service.service.resolve_notification_root_cause_groups",
        lambda *_args, **_kwargs: {"tool_error": ("alice",)},
    )
    monkeypatch.setattr(
        "benchmark.attribution.service.service.load_agent_notification_context",
        _empty_context,
    )
    monkeypatch.setattr(
        "benchmark.attribution.service.service.resolve_notification_publisher",
        lambda *_args, **_kwargs: publisher,
    )
    monkeypatch.setattr(
        "benchmark.attribution.service.service.resolve_effective_notification_targets",
        lambda *_args, **_kwargs: ("proj-1", "owner", [], True),
    )
    hook = NotificationAfterRunHook(notification_config=NotificationConfig(enabled=True, project="p1"))
    outcome = await hook(_input(AttributionResult(
        case_id="safe",
        root_cause_code="safety_violation",
        query="越界",
    )))
    assert outcome.status == "created"
    assert len(publisher.calls) == 1
    assert await hook.flush_groups() == []


async def _empty_context(_config: object) -> tuple[str, None]:
    return "shop", None


def _input(attribution: AttributionResult) -> AfterRunInput:
    return AfterRunInput(
        evaluator_name="shop",
        evaluator_request=_Request(),
        attribution_response=None,
        attribution=attribution,
    )
