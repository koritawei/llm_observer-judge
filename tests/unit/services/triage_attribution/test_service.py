"""Unit tests for TriageAttributionService — the before / after aspects plus dispatching the agent in the middle.

It does not depend on a business service; AgentManager / TriageClassifier / AoneConfig are all mocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.libs.agenthost.agents.manager import AgentManager
from benchmark.libs.agenthost.agents.request import AgentRequest, AgentResponse, ContentPart, ResponseStatus
from benchmark.services import TraceRecordService
from benchmark.attribution.badcase.publisher import NotificationConfig
from benchmark.attribution.badcase.schemas import AttributionResult
from benchmark.services.trace_record_service import TraceRecordKey
from benchmark.attribution.service import (
    NotificationAfterRunHook,
    BeforeRunDecision,
    TriageAttributionService,
    TriageBeforeRunHook,
    TriageConfig,
)
from benchmark.attribution.service.service import AfterRunInput, TriageRunInput, TriageServiceDeps


@dataclass
class _FakeEvaluatorRequest:
    model: str = "example"
    trace_id: str = "tr-1"
    session_id: str = "sess-1"
    agent_name: str | None = "amap-agent"
    project_id: str | None = "proj-x"


@dataclass
class _FakeEvaluatorResult:
    score: float | None = 2.5
    reason: str | None = "loop too long"


def _make_manager_with(
    agent_name: str | None = "attribution_example",
    *,
    root_cause_code: str = "safety_violation",
) -> tuple[AgentManager, MagicMock | None]:
    manager = AgentManager()
    agent: MagicMock | None = None
    if agent_name is not None:
        agent = MagicMock()
        agent.name = agent_name
        result = AttributionResult(
            trace_id="tr-1",
            case_id="case-1",
            category="prompt",
            root_cause="test",
            root_cause_code=root_cause_code,
            confidence=0.9,
            reasoning="test",
            evidence_refs=[],
            action_hint="test",
            needs_human_review=False,
        )
        agent.call = AsyncMock(
            return_value=AgentResponse(
                traceid="tr-1",
                status=ResponseStatus.SUCCESS,
                content_parts=[ContentPart(type="text", data=result.model_dump_json())],
                metadata={},
            )
        )
        manager.register(agent_name, agent)
    return manager, agent


def _make_classifier_should_attribute(should_attribute: bool) -> MagicMock:
    """Build a mock that satisfies the TriageClassifier protocol, and classify returns the given should_attribute."""

    from benchmark.records.enums import TriageLevel
    from benchmark.attribution.service.schema import TriageVerdict

    classifier = MagicMock()
    level = TriageLevel.MILD_BAD if should_attribute else TriageLevel.GOOD
    classifier.classify = MagicMock(
        return_value=TriageVerdict(level=level, should_attribute=should_attribute),
    )
    return classifier


def _service_with_classifier(
    *,
    manager: AgentManager,
    should_attribute: bool,
    notification: NotificationConfig | None = None,
) -> TriageAttributionService:
    """Build a service, and the before aspect injects a mock classifier with the given should_attribute."""

    from benchmark.services import TraceRecordService

    return TriageAttributionService(
        TriageServiceDeps(
            agent_manager=manager,
            before_hook=TriageBeforeRunHook(
                classifiers={"example": _make_classifier_should_attribute(should_attribute)},
                triage_config=TriageConfig(),
            ),
            after_hook=NotificationAfterRunHook(
                notification_config=notification or NotificationConfig(enabled=False, dry_run=True),
            ),
            trace_record_service=TraceRecordService(),
        ),
    )


@pytest.mark.asyncio
async def test_run_calls_attribution_agent_when_should_attribute_true() -> None:
    manager, agent = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(trace_id="tr-99", session_id="sess-99"),
            evaluator_result=_FakeEvaluatorResult(score=1.5),
        ),
    )

    assert outcome.status in ("created", "skipped")
    assert agent is not None
    agent.call.assert_awaited_once()
    agent_request: AgentRequest = agent.call.await_args.args[0]
    assert agent_request.traceid == "tr-99"
    assert agent_request.session_id == "sess-99"


@pytest.mark.asyncio
async def test_run_does_not_prefetch_raw_trace_or_resolve_trace() -> None:
    """The platform layer no longer holds any trace-resolution capability.

    Fetching a trace is pushed down to the business agent (``AttributionAgent``) to manage itself; the platform service
    does not import or hold a ``TraceResolver``, and does not call resolve. ``_build_attribution_request``
    only passes ``trace_id`` / ``project_id`` through to the business agent as strings, and after the business agent
    receives them it calls ``bound_from_metadata`` → ``bind_trace_request`` /
    ``call_tool`` on its own (a ContextVar or ``bound=``).
    """

    manager, agent = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    # Sanity: after the service is constructed it should not hold any trace_resolver shadow field
    # (this is the new contract — the platform does not hold the ability to fetch data).
    for attr in dir(service):
        assert "resolver" not in attr.lower() or attr.startswith("_"), (
            f"TriageAttributionService still exposes trace-resolver-like attribute: {attr}"
        )

    await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(trace_id="tr-99", project_id="proj-x"),
            evaluator_result=_FakeEvaluatorResult(score=1.5),
        ),
    )

    # What is passed through to the business agent in metadata should be the ids, not any resolved trace shadow.
    assert agent is not None
    agent_request: AgentRequest = agent.call.await_args.args[0]
    metadata = agent_request.metadata or {}
    assert metadata.get("raw_trace") is None
    assert "resolved_trace" not in metadata
    assert "trace_resolver" not in metadata
    # What the business agent receives is exactly the entry fields it needs.
    assert metadata.get("trace_id") == "tr-99"
    assert metadata.get("project_id") == "proj-x"


@pytest.mark.asyncio
async def test_run_skips_attribution_when_should_attribute_false() -> None:
    manager, agent = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=False)

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(),
            evaluator_result=_FakeEvaluatorResult(),
        ),
    )

    assert outcome.status == "skipped"
    assert agent is not None
    agent.call.assert_not_called()


@pytest.mark.asyncio
async def test_run_skips_when_classifier_raises() -> None:
    manager, agent = _make_manager_with("attribution_example")
    classifier = _make_classifier_should_attribute(False)
    classifier.classify.side_effect = RuntimeError("classifier down")
    service = TriageAttributionService(
        TriageServiceDeps(
            agent_manager=manager,
            before_hook=TriageBeforeRunHook(
                classifiers={"example": classifier},
                triage_config=TriageConfig(),
            ),
            after_hook=NotificationAfterRunHook(notification_config=NotificationConfig(enabled=False, dry_run=True)),
            trace_record_service=TraceRecordService(),
        ),
    )

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(),
            evaluator_result=_FakeEvaluatorResult(),
        ),
    )

    assert outcome.status == "skipped"
    assert agent is not None
    agent.call.assert_not_called()


@pytest.mark.asyncio
async def test_run_returns_attribution_failed_when_agent_missing() -> None:
    manager, _ = _make_manager_with(None)
    service = _service_with_classifier(manager=manager, should_attribute=True)

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(),
            evaluator_result=_FakeEvaluatorResult(),
        ),
    )

    assert outcome.status == "attribution_failed"
    assert "example" in (outcome.error or "")


@pytest.mark.asyncio
async def test_run_returns_attribution_failed_when_agent_call_raises() -> None:
    manager, agent = _make_manager_with("attribution_example")
    assert agent is not None
    agent.call.side_effect = RuntimeError("agent crashed")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(),
            evaluator_result=_FakeEvaluatorResult(),
        ),
    )

    assert outcome.status == "attribution_failed"
    assert "agent crashed" in (outcome.error or "")


@pytest.mark.asyncio
async def test_set_before_replaces_hook() -> None:
    manager, agent = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    service.set_before(_always_skip_hook())

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(),
            evaluator_result=_FakeEvaluatorResult(),
        ),
    )

    assert outcome.status == "skipped"
    assert agent is not None
    agent.call.assert_not_called()


@pytest.mark.asyncio
async def test_set_after_replaces_hook() -> None:
    manager, _ = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    after_called = {"hit": False}

    async def custom_after(input: Any) -> Any:
        after_called["hit"] = True
        from benchmark.attribution.service import TriageOutcome

        return TriageOutcome(status="created")

    service.set_after(custom_after)  # type: ignore[arg-type]

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(),
            evaluator_result=_FakeEvaluatorResult(),
        ),
    )

    assert outcome.status == "created"
    assert after_called["hit"] is True


@pytest.mark.asyncio
async def test_run_returns_attribution_failed_when_after_hook_raises() -> None:
    manager, _ = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    async def bad_after(input: Any) -> Any:
        raise RuntimeError("after crashed")

    service.set_after(bad_after)  # type: ignore[arg-type]

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(),
            evaluator_result=_FakeEvaluatorResult(),
        ),
    )

    assert outcome.status == "dispatch_failed"
    assert "after crashed" in (outcome.error or "")


def _always_skip_hook() -> object:
    """A hook that always returns should_proceed=False, used to test replacing set_before."""

    class _Skip:
        async def __call__(self, input: Any) -> BeforeRunDecision:
            return BeforeRunDecision(should_proceed=False, reason="test-skip")

    return _Skip()


# ── business-module write assertions ──


@pytest.mark.asyncio
async def test_run_writes_triage_level_to_record_with_seed() -> None:
    from benchmark.records.models import TraceRecord
    from benchmark.services import TraceRecordService

    seed = TraceRecordService()
    await seed.upsert(
        TraceRecordKey(
            trace_id="tr-1",
            project_id="proj-x",
            agent_name="amap-agent",
            evaluator_name="example",
        ),
        status="scored",
    )

    manager, _ = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=_FakeEvaluatorRequest(),
            evaluator_result=_FakeEvaluatorResult(),
        ),
    )

    rec = await TraceRecord.filter(trace_id="tr-1").first()
    assert rec is not None
    assert rec.triage_level == "mild_bad"
    assert rec.judge_json["triage_level"] == "mild_bad"


@pytest.mark.asyncio
async def test_run_writes_attribution_to_record_with_seed() -> None:
    from unittest.mock import AsyncMock, patch

    from benchmark.records.models import TraceRecord
    from benchmark.services import TraceRecordService

    seed = TraceRecordService()
    await seed.upsert(
        TraceRecordKey(
            trace_id="tr-1",
            project_id="proj-x",
            agent_name="amap-agent",
            evaluator_name="example",
        ),
        status="scored",
    )

    manager, _ = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    with (
        patch(
            "benchmark.attribution.badcase.publisher.create_notification_task",
            new_callable=AsyncMock,
        ) as mock_create,
        patch(
            "benchmark.domain.owner_mapping.fetch_owner_mappings",
            return_value=[],
        ),
    ):
        from benchmark.attribution.badcase.schemas import (
            NotificationDispatchResult,
            NotificationDispatchStatus,
        )

        mock_create.return_value = NotificationDispatchResult(
            status=NotificationDispatchStatus.FAILED,
            workitem_id=None,
            message="dry-run skip",
            payload=None,
        )
        await service.run(
            TriageRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(),
                evaluator_result=_FakeEvaluatorResult(),
            ),
        )

    rec = await TraceRecord.filter(trace_id="tr-1").first()
    assert rec is not None
    assert rec.root_cause == "test"
    bucket = rec.attribution_detail_json
    assert bucket["root_cause"] == "test"
    assert bucket["owner_domain"] == "prompt"
    assert bucket["confidence"] == 0.9
    assert bucket["reasoning_raw"] == "test"
    assert bucket["evidence_raw"] == "[]"
    assert bucket["action_hint"] == "test"
    assert bucket["needs_human_review"] is False
    assert bucket["attribution_raw"]  # When raw_response is empty, persist the full snapshot
    assert rec.owner_group is None  # RemoteConfig is not configured → None


@pytest.mark.asyncio
async def test_run_writes_dispatch_and_notification_task() -> None:
    """update_dispatch + enqueue_aone_task dual write, end to end.

    Every previous case patched create_aone_task so payload=None, skipping the enqueue_aone_task path;
    this case returns a CREATED result that carries a payload, and checks that both writes land: notification_dispatch_json.notification_ref +
    notification_job_json (idempotency_key / status=pending).
    """

    from unittest.mock import patch

    from benchmark.records.models import TraceRecord
    from benchmark.services import TraceRecordService

    seed = TraceRecordService()
    await seed.upsert(
        TraceRecordKey(
            trace_id="tr-dispatch",
            project_id="proj-dispatch",
            agent_name="amap-agent",
            evaluator_name="example",
        ),
        status="scored",
    )

    manager, _ = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        from benchmark.attribution.badcase.schemas import (
            NotificationDispatchResult,
            NotificationDispatchStatus,
            NotificationWorkitemPayload,
        )

        mock_create.return_value = NotificationDispatchResult(
            status=NotificationDispatchStatus.CREATED,
            workitem_id="W-12345",
            workitem_url="https://project.aone.alibaba-inc.com/v2/project/1/bug#openWorkitemIdentifier=W-12345",
            message="Aone workitem created.",
            payload=NotificationWorkitemPayload(
                project="agent-eval",
                category="bug",
                type="bug",
                title="[badcase] t-dispatch",
                body="root_cause=test",
                assignee="",
                tags=[],
            ),
        )
        await service.run(
            TriageRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(
                    trace_id="tr-dispatch", project_id="proj-dispatch"
                ),
                evaluator_result=_FakeEvaluatorResult(score=0.2),
            ),
        )

    rec = await TraceRecord.filter(trace_id="tr-dispatch").first()
    assert rec is not None
    # Section 3: update_dispatch — write notification_dispatch_json
    assert rec.notification_dispatch_json is not None
    assert rec.notification_dispatch_json.get("notification_ref") == "W-12345"
    assert (
        rec.notification_dispatch_json.get("notification_url")
        == "https://project.aone.alibaba-inc.com/v2/project/1/bug#openWorkitemIdentifier=W-12345"
    )
    assert rec.notification_dispatch_json.get("notification_title") == "Aone workitem created."
    assert rec.notification_dispatch_json.get("notification_dispatched_at") is not None
    # Section 4: enqueue_aone_task — write notification_job_json
    assert rec.notification_job_json is not None
    assert rec.notification_job_json.get("notification_job_idempotency_key") == "W-12345"
    assert rec.notification_job_json.get("notification_job_status") == "pending"
    assert rec.notification_job_json.get("notification_job_id") is not None
    assert rec.notification_job_json.get("notification_job_payload") is not None


def test_extract_attribution_result_strips_json_fence() -> None:
    from benchmark.attribution.service.service import _extract_attribution_result

    result = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="model",
        root_cause="hallucination",
        confidence=0.8,
        reasoning="test",
        evidence_refs=[],
        action_hint="review",
        needs_human_review=False,
    )
    fenced = f"```json\n{result.model_dump_json()}\n```"
    response = AgentResponse(
        traceid="tr-1",
        status=ResponseStatus.SUCCESS,
        content_parts=[ContentPart(type="text", data=fenced)],
    )

    extracted = _extract_attribution_result(response)
    assert extracted is not None
    assert extracted.root_cause == "hallucination"


def test_extract_attribution_result_repairs_unescaped_quotes() -> None:
    from benchmark.attribution.service.service import _extract_attribution_result

    response = AgentResponse(
        traceid="tr-1",
        status=ResponseStatus.SUCCESS,
        content_parts=[
            ContentPart(
                type="text",
                data=(
                    '{"trace_id":"tr-1","case_id":"case-1","confidence":0.8,'
                    '"reasoning":"reported "missing poi""}'
                ),
            ),
        ],
    )

    extracted = _extract_attribution_result(response)

    assert extracted is not None
    assert extracted.reasoning == 'reported "missing poi"'


# ── OwnerMapping passthrough to AoneAfterRunHook ──


@pytest.mark.asyncio
async def test_notification_after_run_hook_passes_mapping_assigned_to_priority_project_url() -> None:
    """AoneAfterRunHook.__call__ should pass OwnerMapping through to create_aone_task."""
    from benchmark.domain.enums import CasePriority
    from benchmark.domain.owner_mapping import OwnerMapping

    captured: dict[str, Any] = {}

    async def _fake_create_notification_task(**kwargs: Any) -> Any:
        captured.update(kwargs)
        from benchmark.attribution.badcase.schemas import (
            NotificationDispatchResult,
            NotificationDispatchStatus,
            NotificationWorkitemPayload,
        )

        return NotificationDispatchResult(
            status=NotificationDispatchStatus.SKIPPED,
            payload=NotificationWorkitemPayload(
                project="2165583",
                category="bug",
                title="t",
                body="b",
            ),
        )

    config = NotificationConfig(
        enabled=False,
        project="2165583",
        category="bug",
        assignee="fallback-user",
    )
    hook = NotificationAfterRunHook(notification_config=config)

    mapping = OwnerMapping(
        mapping_id="m1",
        match_type="agent_cause",
        match_key="amap-agent/tool_error",
        assigned_to="alice",
        priority=CasePriority.P0,
        project_url="https://example.com/p/123",
        owner_group="platform-team",
        enabled=True,
    )

    attribution = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="prompt",
        root_cause="tool_error",
        root_cause_code="safety_violation",
        confidence=0.9,
        reasoning="x",
        evidence_refs=[],
        action_hint="x",
        needs_human_review=False,
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        AsyncMock(side_effect=_fake_create_notification_task),
    ):
        await hook(
            AfterRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
                resolved_route=mapping,
            ),
        )

    # Core assertion: the hook passes the whole mapping through, and create_aone_task owns the expansion
    assert captured.get("resolved_route") is mapping


@pytest.mark.asyncio
async def test_notification_after_run_hook_passes_none_when_no_route() -> None:
    """When resolved_route is not passed, the hook should pass None to create_aone_task (the fallback is handled by the latter)."""
    captured: dict[str, Any] = {}

    async def _fake_create_notification_task(**kwargs: Any) -> Any:
        captured.update(kwargs)
        from benchmark.attribution.badcase.schemas import (
            NotificationDispatchResult,
            NotificationDispatchStatus,
            NotificationWorkitemPayload,
        )

        return NotificationDispatchResult(
            status=NotificationDispatchStatus.SKIPPED,
            payload=NotificationWorkitemPayload(
                project="2165583",
                category="bug",
                title="t",
                body="b",
            ),
        )

    config = NotificationConfig(
        enabled=False,
        project="2165583",
        category="bug",
        assignee="fallback-user",
    )
    hook = NotificationAfterRunHook(notification_config=config)

    attribution = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="x",
        root_cause="x",
        root_cause_code="safety_violation",
        confidence=0.9,
        reasoning="x",
        evidence_refs=[],
        action_hint="x",
        needs_human_review=False,
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        AsyncMock(side_effect=_fake_create_notification_task),
    ):
        await hook(
            AfterRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
            ),
        )

    assert captured.get("resolved_route") is None


# ── should_dispatch: the agent controls whether a ticket is dispatched ──


def _make_manager_with_should_dispatch(
    should_dispatch: bool,
    agent_name: str = "attribution_example",
    *,
    root_cause_code: str = "safety_violation",
) -> tuple[AgentManager, MagicMock]:
    """Build an attribution agent manager that produces the given should_dispatch value."""

    manager = AgentManager()
    agent = MagicMock()
    agent.name = agent_name
    result = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="prompt",
        root_cause="test",
        root_cause_code=root_cause_code,
        confidence=0.9,
        reasoning="test",
        evidence_refs=[],
        action_hint="test",
        needs_human_review=False,
        should_dispatch=should_dispatch,
    )
    agent.call = AsyncMock(
        return_value=AgentResponse(
            traceid="tr-1",
            status=ResponseStatus.SUCCESS,
            content_parts=[ContentPart(type="text", data=result.model_dump_json())],
            metadata={},
        )
    )
    manager.register(agent_name, agent)
    return manager, agent


@pytest.mark.asyncio
async def test_run_skips_dispatch_when_agent_says_should_dispatch_false() -> None:
    """AttributionResult.should_dispatch=False → after_hook returns skipped, and does not call create_aone_task."""

    from unittest.mock import patch

    from benchmark.records.models import TraceRecord
    from benchmark.services import TraceRecordService

    seed = TraceRecordService()
    await seed.upsert(
        TraceRecordKey(
            trace_id="tr-skip",
            project_id="proj-skip",
            agent_name="amap-agent",
            evaluator_name="example",
        ),
        status="scored",
    )

    manager, _ = _make_manager_with_should_dispatch(should_dispatch=False)
    service = _service_with_classifier(manager=manager, should_attribute=True)

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        outcome = await service.run(
            TriageRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(trace_id="tr-skip", project_id="proj-skip"),
                evaluator_result=_FakeEvaluatorResult(),
            ),
        )

    assert outcome.status == "skipped"
    assert outcome.reason == "agent_should_dispatch_false"
    assert outcome.notification_task is not None
    assert outcome.notification_task.status.value == "skipped"
    mock_create.assert_not_awaited()

    rec = await TraceRecord.filter(trace_id="tr-skip").first()
    assert rec is not None
    assert rec.notification_dispatch_json is not None
    assert rec.notification_dispatch_json.get("notification_skip_reason") == "agent_should_dispatch_false"
    assert rec.notification_dispatch_json.get("notification_ref") is None
    assert rec.notification_job_json is None


@pytest.mark.asyncio
async def test_run_candidate_does_not_route_or_dispatch_and_persists_snapshot() -> None:
    """An `other:` candidate does not enter owner routing or Aone, and still lands in the attribution bucket in full."""
    from benchmark.records.models import TraceRecord
    from benchmark.services import TraceRecordService
    from benchmark.attribution.components.candidate_root_cause import (
        CandidateRootCause,
    )

    seed = TraceRecordService()
    await seed.upsert(
        TraceRecordKey(
            trace_id="tr-candidate",
            project_id="proj-candidate",
            agent_name="amap-agent",
            evaluator_name="example",
        ),
        status="scored",
    )
    candidate = CandidateRootCause(
        raw_code="other:tool_schema_gap",
        title="工具 schema 缺口",
        surface="tool_data",
        symptom="missing_price_attribute",
        evidence_refs=["extra:failure_modes"],
        agent_scope="vehicle",
    )
    result = AttributionResult(
        trace_id="tr-candidate",
        case_id="case-candidate",
        category="工具能力|功能覆盖缺口",
        root_cause="未注册工具缺口",
        candidate_root_cause=candidate,
        should_dispatch=True,
    )
    manager = AgentManager()
    agent = MagicMock(name="attribution_example")
    agent.name = "attribution_example"
    agent.call = AsyncMock(
        return_value=AgentResponse(
            traceid="tr-candidate",
            status=ResponseStatus.SUCCESS,
            content_parts=[ContentPart(type="text", data=result.model_dump_json())],
            metadata={},
        )
    )
    manager.register(agent.name, agent)
    service = _service_with_classifier(manager=manager, should_attribute=True)

    with (
        patch(
            "benchmark.attribution.badcase.publisher.create_notification_task",
            new_callable=AsyncMock,
        ) as mock_create,
        patch(
            "benchmark.attribution.service.service.resolve_owner_route",
        ) as mock_route,
    ):
        outcome = await service.run(
            TriageRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(
                    trace_id="tr-candidate",
                    project_id="proj-candidate",
                ),
                evaluator_result=_FakeEvaluatorResult(),
            )
        )

    assert outcome.status == "skipped"
    mock_create.assert_not_awaited()
    mock_route.assert_not_called()
    record = await TraceRecord.filter(trace_id="tr-candidate").first()
    assert record is not None
    assert record.owner_group is None
    assert record.attribution_detail_json["candidate_root_cause"] == candidate.model_dump(
        mode="json"
    )


@pytest.mark.asyncio
async def test_run_dispatches_when_agent_says_should_dispatch_true() -> None:
    """AttributionResult.should_dispatch=True → take the original create_aone_task path."""

    from unittest.mock import patch

    from benchmark.services import TraceRecordService

    seed = TraceRecordService()
    await seed.upsert(
        TraceRecordKey(
            trace_id="tr-go",
            project_id="proj-go",
            agent_name="amap-agent",
            evaluator_name="example",
        ),
        status="scored",
    )

    manager, _ = _make_manager_with_should_dispatch(should_dispatch=True)
    service = _service_with_classifier(manager=manager, should_attribute=True)

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        from benchmark.attribution.badcase.schemas import (
            NotificationDispatchResult,
            NotificationDispatchStatus,
        )

        mock_create.return_value = NotificationDispatchResult(
            status=NotificationDispatchStatus.CREATED,
            workitem_id="W-999",
            message="created",
        )
        outcome = await service.run(
            TriageRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(trace_id="tr-go", project_id="proj-go"),
                evaluator_result=_FakeEvaluatorResult(score=0.2),
            ),
        )

    assert outcome.status == "created"
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_dispatches_when_attribution_result_omits_should_dispatch() -> None:
    """AttributionResult has no should_dispatch field (an old agent) → default True → take the original path."""

    manager, _ = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        from benchmark.attribution.badcase.schemas import (
            NotificationDispatchResult,
            NotificationDispatchStatus,
        )

        mock_create.return_value = NotificationDispatchResult(
            status=NotificationDispatchStatus.CREATED,
            workitem_id="W-old",
            message="created",
        )
        outcome = await service.run(
            TriageRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(),
                evaluator_result=_FakeEvaluatorResult(score=0.2),
            ),
        )

    assert outcome.status == "created"
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_notification_after_run_hook_returns_skipped_when_should_dispatch_false() -> None:
    """Call AoneAfterRunHook directly, and verify the gate behavior when should_dispatch=False."""

    config = NotificationConfig(enabled=True, project="proj", category="bug")
    hook = NotificationAfterRunHook(notification_config=config)

    attribution = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="prompt",
        root_cause="test",
        confidence=0.9,
        should_dispatch=False,
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        outcome = await hook(
            AfterRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
            ),
        )

    assert outcome.status == "skipped"
    assert outcome.reason == "agent_should_dispatch_false"
    mock_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_notification_hook_skips_when_grade_not_poor_or_fail() -> None:
    """On a 0–2 scale, ACCEPTABLE (for example 1.2) → score_threshold, and no ticket is created."""
    config = NotificationConfig(enabled=True, project="proj", category="bug")
    hook = NotificationAfterRunHook(notification_config=config)
    attribution = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="prompt",
        root_cause="test",
        confidence=0.9,
        should_dispatch=True,
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        outcome = await hook(
            AfterRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
                fused_score=1.2,
                score_scale_max=2.0,
            ),
        )

    assert outcome.status == "skipped"
    assert outcome.reason == "score_threshold"
    mock_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_notification_hook_dispatches_low_ten_point_score() -> None:
    """A low score on a 0–10 scale (3.0=POOR) must create a ticket, and must not be misread as rubric_v2 EXCELLENT."""
    from benchmark.attribution.badcase.schemas import (
        NotificationDispatchResult,
        NotificationDispatchStatus,
    )

    config = NotificationConfig(enabled=True, project="proj", category="bug")
    hook = NotificationAfterRunHook(notification_config=config)
    attribution = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="prompt",
        root_cause="test",
        confidence=0.9,
        should_dispatch=True,
    )
    created = NotificationDispatchResult(
        status=NotificationDispatchStatus.CREATED,
        workitem_id="W-ten",
        message="ok",
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
        return_value=created,
    ) as mock_create:
        outcome = await hook(
            AfterRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
                fused_score=3.0,
                score_scale_max=10.0,
            ),
        )

    assert outcome.status == "created"
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_notification_hook_infers_ten_point_when_scale_missing() -> None:
    """When score_scale_max is not carried, a score >2 is still inferred as 0–10, so a dispatch is not silently skipped."""
    from benchmark.attribution.badcase.schemas import (
        NotificationDispatchResult,
        NotificationDispatchStatus,
    )

    config = NotificationConfig(enabled=True, project="proj", category="bug")
    hook = NotificationAfterRunHook(notification_config=config)
    attribution = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="prompt",
        root_cause="test",
        confidence=0.9,
        should_dispatch=True,
    )
    created = NotificationDispatchResult(
        status=NotificationDispatchStatus.CREATED,
        workitem_id="W-infer",
        message="ok",
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
        return_value=created,
    ) as mock_create:
        outcome = await hook(
            AfterRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
                fused_score=3.0,
            ),
        )

    assert outcome.status == "created"
    mock_create.assert_awaited_once()


def _make_manager_with_corrected_triage(
    *,
    corrected_triage_level: str,
    should_dispatch: bool = True,
    agent_name: str = "attribution_example",
) -> tuple[AgentManager, MagicMock]:
    """Build an attribution agent that carries corrected_triage_level (simulating an override of the judge)."""

    manager = AgentManager()
    agent = MagicMock()
    agent.name = agent_name
    result = AttributionResult(
        trace_id="tr-1",
        case_id="case-1",
        category="unknown",
        root_cause="Judge 误判：preview 截断导致无法核实。",
        confidence=0.9,
        reasoning="TOOL output 与回答一致",
        evidence_refs=["obs-1"],
        action_hint="improve_judge_preview",
        needs_human_review=False,
        should_dispatch=should_dispatch,
        corrected_triage_level=corrected_triage_level,
    )
    agent.call = AsyncMock(
        return_value=AgentResponse(
            traceid="tr-1",
            status=ResponseStatus.SUCCESS,
            content_parts=[ContentPart(type="text", data=result.model_dump_json())],
            metadata={},
        )
    )
    manager.register(agent_name, agent)
    return manager, agent


@pytest.mark.asyncio
async def test_run_rewrites_triage_and_skips_notification_when_corrected_triage_level() -> None:
    """Attribution overrides the judge: write triage_level back, refuse to dispatch Aone, and root_cause keeps the attribution conclusion."""

    from unittest.mock import patch

    from benchmark.records.models import TraceRecord
    from benchmark.services import TraceRecordService

    seed = TraceRecordService()
    await seed.upsert(
        TraceRecordKey(
            trace_id="tr-overturn",
            project_id="proj-overturn",
            agent_name="amap-agent",
            evaluator_name="example",
        ),
        status="scored",
    )

    manager, _ = _make_manager_with_corrected_triage(
        corrected_triage_level="good",
        should_dispatch=True,  # Even if the agent forgets to refuse dispatch, the platform should force a skip
    )
    service = _service_with_classifier(manager=manager, should_attribute=True)

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        outcome = await service.run(
            TriageRunInput(
                evaluator_name="example",
                evaluator_request=_FakeEvaluatorRequest(
                    trace_id="tr-overturn",
                    project_id="proj-overturn",
                ),
                evaluator_result=_FakeEvaluatorResult(),
            ),
        )

    assert outcome.status == "skipped"
    assert outcome.reason == "agent_should_dispatch_false"
    mock_create.assert_not_awaited()

    rec = await TraceRecord.filter(trace_id="tr-overturn").first()
    assert rec is not None
    assert rec.triage_level == "attribution_overridden"
    assert rec.root_cause == "Judge 误判：preview 截断导致无法核实。"
    assert rec.attribution_detail_json is not None
    assert rec.attribution_detail_json.get("original_triage_level") == "mild_bad"
    assert rec.attribution_detail_json.get("corrected_triage_level") == "good"
    assert rec.notification_job_json is None


def test_validate_notification_dispatch_json_accepts_skip_reason() -> None:
    """The validator accepts the notification_skip_reason field (backward compatible)."""

    from benchmark.records.models.validators import validate_notification_dispatch_json

    bucket = {"notification_skip_reason": "agent_should_dispatch_false"}
    result = validate_notification_dispatch_json(bucket)
    assert result is not None
    assert result["notification_skip_reason"] == "agent_should_dispatch_false"

    legacy = {"notification_ref": "W-1"}
    result = validate_notification_dispatch_json(legacy)
    assert result is not None
    assert result.get("notification_skip_reason") is None


@pytest.mark.asyncio
async def test_run_short_circuits_on_missing_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The required trio is missing → fail fast at the run entry, return attribution_failed, and neither the classifier nor the agent is called."""
    import logging

    manager, agent = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    bad_req = _FakeEvaluatorRequest(
        trace_id=None,
        agent_name=None,
        project_id=None,
    )

    with caplog.at_level(logging.ERROR, logger="agent_eval.evaluator"):
        outcome = await service.run(
            TriageRunInput(
                evaluator_name="example",
                evaluator_request=bad_req,
                evaluator_result=_FakeEvaluatorResult(score=1.5),
            ),
        )

    assert outcome.status == "attribution_failed"
    assert outcome.error is not None
    assert "metadata_missing" in outcome.error
    assert "trace_id" in outcome.error
    assert "project_id" in outcome.error
    assert "agent_name" in outcome.error
    # What matters: the agent is never called, and no LLM/tool is started
    if agent is not None:
        agent.call.assert_not_called()
    # An ERROR log is visible, which helps debugging
    err = [r for r in caplog.records if "triage.metadata_missing" in r.message]
    assert len(err) == 1


@pytest.mark.asyncio
async def test_run_partial_missing_lists_only_missing_fields() -> None:
    """Only project_id is missing → the error message lists only project_id."""
    manager, agent = _make_manager_with("attribution_example")
    service = _service_with_classifier(manager=manager, should_attribute=True)

    bad_req = _FakeEvaluatorRequest(
        trace_id="tr-1",
        agent_name="amap-agent",
        project_id=None,
    )

    outcome = await service.run(
        TriageRunInput(
            evaluator_name="example",
            evaluator_request=bad_req,
            evaluator_result=_FakeEvaluatorResult(score=1.5),
        ),
    )

    assert outcome.status == "attribution_failed"
    assert outcome.error == "metadata_missing: project_id"
    if agent is not None:
        agent.call.assert_not_called()


# ── AoneAfterRunHook safety-class gate ──


@pytest.mark.asyncio
async def test_notification_hook_dispatches_safety_root_cause() -> None:
    """safety_violation + should_dispatch=True → create a ticket immediately for the single case.

    Even when the fused score is ACCEPTABLE (1.2) it does not take score_threshold; the safety class is exempt.
    """
    config = NotificationConfig(enabled=True, project="proj", category="bug")
    hook = NotificationAfterRunHook(notification_config=config)

    attribution = AttributionResult(
        trace_id="tr-safety",
        case_id="case-s",
        category="model",
        root_cause_code="safety_violation",
        confidence=0.95,
        should_dispatch=True,
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        from benchmark.attribution.badcase.schemas import (
            NotificationDispatchResult,
            NotificationDispatchStatus,
        )

        mock_create.return_value = NotificationDispatchResult(
            status=NotificationDispatchStatus.CREATED,
            workitem_id="W-SAFETY",
            message="created",
        )
        outcome = await hook(
            AfterRunInput(
                evaluator_name="vehicle_agent",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
                fused_score=1.2,
                score_scale_max=2.0,
            )
        )

    assert outcome.status == "created"
    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_notification_hook_skips_non_safety_aggregate_only() -> None:
    """A non-safety class + should_dispatch=True → do not create a ticket, and wait for the cluster_report aggregation."""
    config = NotificationConfig(enabled=True, project="proj", category="bug")
    hook = NotificationAfterRunHook(notification_config=config, aggregate_non_safety=True)

    attribution = AttributionResult(
        trace_id="tr-nonsafety",
        case_id="case-ns",
        category="model",
        root_cause_code="tool_cmd_tag_missing",
        confidence=0.9,
        should_dispatch=True,
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        outcome = await hook(
            AfterRunInput(
                evaluator_name="vehicle_agent",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
            )
        )

    assert outcome.status == "skipped"
    assert outcome.reason == "non_safety_aggregate_only"
    mock_create.assert_not_awaited()


@pytest.mark.asyncio
async def test_notification_hook_medical_boundary_is_safety() -> None:
    """medical_boundary_violation is also a safety class, and a single case creates a ticket immediately (exempt from the score threshold)."""
    config = NotificationConfig(enabled=True, project="proj", category="bug")
    hook = NotificationAfterRunHook(notification_config=config)

    attribution = AttributionResult(
        trace_id="tr-med",
        case_id="case-m",
        category="model",
        root_cause_code="medical_boundary_violation",
        confidence=0.9,
        should_dispatch=True,
    )

    with patch(
        "benchmark.attribution.badcase.publisher.create_notification_task",
        new_callable=AsyncMock,
    ) as mock_create:
        from benchmark.attribution.badcase.schemas import (
            NotificationDispatchResult,
            NotificationDispatchStatus,
        )

        mock_create.return_value = NotificationDispatchResult(
            status=NotificationDispatchStatus.CREATED,
            workitem_id="W-MED",
            message="created",
        )
        outcome = await hook(
            AfterRunInput(
                evaluator_name="vehicle_agent",
                evaluator_request=_FakeEvaluatorRequest(),
                attribution_response=None,
                attribution=attribution,
                fused_score=1.2,
                score_scale_max=2.0,
            )
        )

    assert outcome.status == "created"
    mock_create.assert_awaited_once()
