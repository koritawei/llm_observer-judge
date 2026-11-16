from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from benchmark.domain.notification_dispatch_resolve import AgentNotificationSnapshot
from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.attribution.badcase.schemas import (
    NotificationDispatchResult,
    NotificationDispatchStatus,
    AttributionResult,
)
from benchmark.attribution.service.service import AfterRunInput, NotificationAfterRunHook


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregate", [False, True])
async def test_non_safety_dispatch_defaults_and_opt_in(monkeypatch, aggregate):
    create = AsyncMock(
        return_value=NotificationDispatchResult(status=NotificationDispatchStatus.CREATED, workitem_id="test")
    )
    monkeypatch.setattr("benchmark.attribution.badcase.publisher.create_notification_task", create)
    hook = NotificationAfterRunHook(
        notification_config=NotificationConfig(), **({"aggregate_non_safety": True} if aggregate else {})
    )
    outcome = await hook(
        AfterRunInput(
            evaluator_name="amap_agent",
            evaluator_request=SimpleNamespace(),
            attribution_response=None,
            attribution=AttributionResult(
                root_cause="test", root_cause_code="other", should_dispatch=True
            ),
        )
    )
    assert outcome.status == ("skipped" if aggregate else "created")
    assert create.await_count == (0 if aggregate else 1)


@pytest.mark.asyncio
async def test_agent_notification_disabled_flag_ignored_still_dispatches(monkeypatch) -> None:
    """Turning dispatch off in the registration document is deprecated; the realtime hook can still create a ticket."""
    create = AsyncMock(
        return_value=NotificationDispatchResult(status=NotificationDispatchStatus.CREATED, workitem_id="test")
    )
    monkeypatch.setattr("benchmark.attribution.badcase.publisher.create_notification_task", create)
    monkeypatch.setattr(
        "benchmark.attribution.service.service.load_agent_notification_context",
        AsyncMock(
            return_value=(
                "amap_agent",
                AgentNotificationSnapshot(name="amap_agent", notification_dispatch_enabled=False),
            )
        ),
    )
    monkeypatch.setattr(
        "benchmark.attribution.service.service.resolve_effective_notification_targets",
        lambda *_args, **_kwargs: ("2165583", "529504", [], True),
    )
    hook = NotificationAfterRunHook(notification_config=NotificationConfig(enabled=True))
    outcome = await hook(
        AfterRunInput(
            evaluator_name="amap_agent",
            evaluator_request=SimpleNamespace(
                metadata={"agent_registration_id": "ba-1"},
            ),
            attribution_response=None,
            attribution=AttributionResult(
                root_cause="test",
                root_cause_code="other",
                should_dispatch=True,
            ),
            fused_score=0.0,
        )
    )
    assert outcome.status == "created"
    create.assert_awaited_once()
