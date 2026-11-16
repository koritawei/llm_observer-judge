"""End-to-end behavior lock for one project with multiple agents (plan C).

    Ownership injection for a nameless trace (S3') and "the entry declaration is the source of truth":
- an entry's declared ``agentName`` does not have to be in the registry candidate set (S4d removed that);
- when metadata.agent_name is empty/unknown and the observed name hits an entry → inject the
  agentName the entry declared (passed by webUI); only an undeclared entry falls back to registry candidates;
- an entry with no agentName and a registry with no candidates → discard, because there is nothing to attribute to;
- an agent reported by the SDK must be covered by evaluators (S4e); a nameless trace is not checked.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorDeps,
    resolve_missing_agent_name,
)
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.libs.mq.consumer import ConsumedMessage

_PROJECT = "proj-multi"
_TRACE_ID = "trace-m1"


def _make_evaluator(configs: list[EvalConfigEntry]) -> BenchmarkStreamingEvaluator:
    config_store = MagicMock()
    config_store.get_configs.return_value = configs
    config_file = MagicMock()
    config_file.configs = {_PROJECT: configs} if configs else {}
    config_store.get_config_file.return_value = config_file
    evaluator_service = AsyncMock()

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        # Aligned with production to_evaluator_request: extract agent_name from observation.metadata
        obs_meta = dict(getattr(observation, "metadata", None) or {})
        return EvaluatorRequest(
            metadata=obs_meta,
            agent_name=obs_meta.get("agent_name"),
            skip_trace_resolve=True,
        )

    return BenchmarkStreamingEvaluator(
        deps=BenchmarkStreamingEvaluatorDeps(
            config_store=config_store,
            evaluator_service=evaluator_service,
            request_converter=_converter,
        ),
    )


def _make_msg(
    trace_id: str = _TRACE_ID,
    name: str = "amap-agent.stream.ai_chat.standard",
    metadata: dict | None = None,
) -> ConsumedMessage:
    record = TraceRecord(
        id=trace_id,
        project_id=_PROJECT,
        name=name,
        environment="prod",
        input={"q": "test"},
        output={"a": "resp"},
        metadata=metadata or {},
        tags=["prod"],
    )
    return ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record=record.model_dump()),
        key=f"{_PROJECT}:{trace_id}",
        offset=0,
        partition=0,
    )


def _registry(names: list[str]) -> list:
    from benchmark.domain.agent_config import AgentConfig

    return [AgentConfig(agent_name=n, project_id=_PROJECT) for n in names]


@pytest.mark.asyncio
async def test_multi_candidate_admitted_and_entry_agent_name_overrides(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Multi-candidate admission + entry.agentName override: the trace is owned by the agent the entry declares."""
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="example",
                agentName="search_agent",
                name=["search_agent.stream.overview"],
                sampling=1.0,
            ),
            EvalConfigEntry(
                evaluatorName="example2",
                agentName="amap-agent",
                name=["amap-agent.stream.ai_chat.standard"],
                sampling=1.0,
            ),
        ],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured[name] = [request]
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(name="amap-agent.stream.ai_chat.standard")

    with (
        patch(
            "benchmark.streaming.consumer.get_agent_names_by_project_id",
            return_value=["amap-agent", "search_agent"],
        ),
        caplog.at_level(logging.INFO),
    ):
        await evaluator(msg)

    # S3b did not discard (the funnel has no filter_agent_name); entry2 hits the name and is evaluated as amap-agent
    assert evaluator._funnel.filter_agent_name == 0
    assert "example2" in captured
    request = captured["example2"][0]
    assert request.agent_name == "amap-agent"
    assert request.metadata["agent_name"] == "amap-agent"
    # An entry whose name does not match (the search_agent stream) is not evaluated
    assert "example" not in captured


@pytest.mark.asyncio
async def test_entry_agent_name_not_in_registry_still_evaluated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """S4d has been removed: an entry-declared agentName is still evaluated when it is not in the registry candidate set.

    Production case: the agent registered for the project is vui_agent, but the webUI task declares
    amap-agent (the owner of the traces that actually flow under that project) — the declaration is the source of truth,
    and whatever the registry returns does not affect whether the entry survives or is evaluated.
    """
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="typo_entry",
                agentName="amap-agnet",
                sampling=1.0,
            ),
        ],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured[name] = [request]
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(metadata={"agent_name": "amap-agnet"})

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["amap-agent"],
    ):
        await evaluator(msg)

    # The entry is not skipped: it is evaluated as usual and persisted under the value the entry declared
    assert "typo_entry" in captured
    request = captured["typo_entry"][0]
    assert request.agent_name == "amap-agnet"
    evaluator._evaluator_service.evaluate.assert_called()


@pytest.mark.asyncio
async def test_webui_declared_agent_wins_over_registry_mapping(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lock for the trace f22d0868 case: the registry maps the project to vui_agent,

    the webUI entry declares amap-agent and the trace reports amap-agent → evaluate and own it as
    amap-agent; a registry candidate set that disagrees with the declaration no longer blocks (S4d removed).
    """
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["amap-agent.stream.ai_chat.standard"],
                sampling=1.0,
            ),
        ],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured[name] = [request]
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(metadata={"agent_name": "amap-agent"})

    with (
        patch(
            "benchmark.streaming.consumer.get_agent_names_by_project_id",
            return_value=["vui_agent"],
        ),
        caplog.at_level(logging.INFO),
    ):
        await evaluator(msg)

    # Evaluation happens and is not blocked by S4e; the persisted owner is the entry-declared amap-agent
    assert evaluator._funnel.filter_agent_name == 0
    assert "amap_agent" in captured
    request = captured["amap_agent"][0]
    assert request.agent_name == "amap-agent"


@pytest.mark.asyncio
async def test_declared_vui_agent_still_blocks_amap_agent_trace() -> None:
    """Reverse lock: when the single entry declares vui_agent, an amap-agent trace is still blocked by that entry."""
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="vui_agent",
                name=["amap-agent.stream.ai_chat.standard"],
                sampling=1.0,
            ),
        ],
    )
    msg = _make_msg(metadata={"agent_name": "amap-agent"})

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["vui_agent"],
    ):
        await evaluator(msg)

    assert evaluator._funnel.filter_agent_name == 1
    evaluator._evaluator_service.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_same_trace_fans_out_to_independent_benchmark_entries() -> None:
    """The same Kafka payload is copied and fanned out: each entry applies its own filter, and they do not drop each other by mistake.

    Historical bug: S4e used the **union** of every entry's agentName as a global gate — when entry A declared
    amap-agent and entry B declared nothing, a trace reporting search_ai_card was dropped entirely,
    so B could never evaluate it. After the fix each entry is decided on its own: A rejects, B accepts → only B evaluates.
    """
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["shared.stream.event"],
                sampling=1.0,
            ),
            EvalConfigEntry(
                evaluatorName="example",
                name=["shared.stream.event"],
                sampling=1.0,
            ),
        ],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured.setdefault(name, []).append(request)
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(
        name="shared.stream.event",
        metadata={"agent_name": "search_ai_card"},
    )

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["amap-agent", "search_ai_card"],
    ):
        await evaluator(msg)

    assert "amap_agent" not in captured
    assert "example" in captured
    assert captured["example"][0].agent_name == "search_ai_card"


@pytest.mark.asyncio
async def test_empty_name_entry_does_not_steal_sibling_agent_traffic() -> None:
    """With several entries in one project, an entry with no name must not consume a neighboring agent's traffic by mistake.

    History: an empty name matched every trace; when example had no name it evaluated the amap-agent
    stream. Repair: multiple entries require a name, and a nameless entry is removed at S4.
    """
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["amap-agent.stream.ai_chat.standard"],
                sampling=1.0,
            ),
            EvalConfigEntry(
                evaluatorName="example",
                agentName="example",
                sampling=1.0,  # Name is left unset on purpose
            ),
        ],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured.setdefault(name, []).append(request)
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(
        name="amap-agent.stream.ai_chat.standard",
        metadata={"agent_name": "amap-agent"},
    )

    await evaluator(msg)

    assert set(captured) == {"amap_agent"}
    assert "example" not in captured


@pytest.mark.asyncio
async def test_same_stem_prompt_versions_fan_out_independently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """example_v56 / v57 share a stem and coexist: the same trace is fanned out twice, and logs are distinguished by registrationId."""
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="example",
                agentName="amap-agent",
                agentRegistrationId="reg-v56",
                internalAgentName="agx_example_0000000056",
                name=["shared.stream.event"],
                sampling=1.0,
            ),
            EvalConfigEntry(
                evaluatorName="example",
                agentName="amap-agent",
                agentRegistrationId="reg-v57",
                internalAgentName="agx_example_0000000057",
                name=["shared.stream.event"],
                sampling=1.0,
            ),
        ],
    )
    calls: list[object] = []

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        calls.append(request)
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(
        name="shared.stream.event",
        metadata={"agent_name": "amap-agent"},
    )

    with caplog.at_level(logging.INFO):
        await evaluator(msg)

    assert len(calls) == 2
    assert evaluator._funnel.eval_ok == 2
    joined = " ".join(r.message for r in caplog.records)
    assert "example@reg-v56" in joined
    assert "example@reg-v57" in joined


@pytest.mark.asyncio
async def test_transient_failure_does_not_abort_sibling_fan_out() -> None:
    """A transient failure of entry A must not interrupt evaluation of entry B (early-commit does not retry the whole message)."""
    from benchmark.utils.exceptions import LLM_FAILED_CODE, EvaluatorError

    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["shared.stream.event"],
                sampling=1.0,
            ),
            EvalConfigEntry(
                evaluatorName="example",
                agentName="amap-agent",
                name=["shared.stream.event"],
                sampling=1.0,
            ),
        ],
    )
    captured: list[str] = []

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured.append(name)
        if name == "amap_agent":
            raise EvaluatorError(
                code=502,
                error_type=LLM_FAILED_CODE,
                message="llm down",
            )
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(
        name="shared.stream.event",
        metadata={"agent_name": "amap-agent"},
    )

    await evaluator(msg)

    assert captured == ["amap_agent", "example"]
    assert evaluator._funnel.eval_fail_transient == 1
    assert evaluator._funnel.eval_ok == 1


@pytest.mark.asyncio
async def test_same_trace_fans_out_when_both_entries_accept() -> None:
    """When both entries pass their filters, the same trace enters each one's evaluation flow."""
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["shared.stream.event"],
                sampling=1.0,
            ),
            EvalConfigEntry(
                evaluatorName="example",
                agentName="amap-agent",
                name=["shared.stream.event"],
                sampling=1.0,
            ),
        ],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured.setdefault(name, []).append(request)
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(
        name="shared.stream.event",
        metadata={"agent_name": "amap-agent"},
    )

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["amap-agent"],
    ):
        await evaluator(msg)

    assert set(captured) == {"amap_agent", "example"}
    assert all(req.agent_name == "amap-agent" for reqs in captured.values() for req in reqs)


@pytest.mark.asyncio
async def test_undeclared_agent_name_dimension_not_filtered() -> None:
    """The S4e fallback has been removed: an entry that does not declare agentName → that dimension is not filtered (no longer falling

    back to the registry candidate set). The trace is still evaluated even when its agent is not in the registry candidates.
    """
    evaluator = _make_evaluator(
        [EvalConfigEntry(evaluatorName="example", sampling=1.0)],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured[name] = [request]
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    # The SDK reports amap-agent, but the registry candidates are only vui_agent and the entry declared nothing
    msg = _make_msg(metadata={"agent_name": "amap-agent"})

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["vui_agent"],
    ):
        await evaluator(msg)

    assert evaluator._funnel.filter_agent_name == 0
    assert "example" in captured
    request = captured["example"][0]
    # The entry did not declare agentName, so the persisted owner stays the value the trace reported
    assert request.agent_name == "amap-agent"


@pytest.mark.asyncio
async def test_tags_only_hit_does_not_stamp_registered_agent_name() -> None:
    """A tags-only hit with a name miss must not stamp the registered agentName into the evaluation request.

    Aligned with the production log: when name=amap-vui-agent.voice_ws.vui_agent is not in the entry name
    list, the registered amap-agent is not set even if tags contain amap-vui-agent.
    """
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["amap-agent.stream.ai_chat.standard"],
                tags=["amap-vui-agent"],
                sampling=1.0,
            ),
        ],
    )
    msg = _make_msg(
        name="amap-vui-agent.voice_ws.vui_agent",
        metadata={},
    )
    msg.payload.record["tags"] = [
        "33.35.28.201",
        "amap-vui-agent",
        "sa128",
        "vui",
        "vui-online-eval-00062",
    ]

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["vui_agent"],
    ):
        await evaluator(msg)

    # The name filter missed → S4c excludes every config, and nothing is evaluated
    assert evaluator._funnel.filter_attrs >= 1
    evaluator._evaluator_service.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_missing_agent_name_injected_from_entry_declaration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A nameless trace (the observed name hits an entry) → inject the agentName passed by webUI.

    The production root cause this fixes: traces reported by amap-agent carry no metadata.agent_name,
    and the registry maps that project to vui_agent — the old logic stamped the wrong owner.
    New logic: the observed name already matched the entry name filter, so the owner is the entry-declared amap-agent.
    """
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["amap-agent.stream.ai_chat.standard"],
                sampling=1.0,
            ),
        ],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured[name] = [request]
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    # Nameless: metadata has no agent_name key
    msg = _make_msg(name="amap-agent.stream.ai_chat.standard", metadata={})
    msg.payload.record["tags"] = ["ai_chat", "amap-agent"]

    with (
        patch(
            "benchmark.streaming.consumer.get_agent_names_by_project_id",
            return_value=["vui_agent"],
        ),
        caplog.at_level(logging.INFO),
    ):
        await evaluator(msg)

    # Inject the entry-declared value rather than the registry's vui_agent; evaluation happens and is not blocked
    assert evaluator._funnel.filter_agent_name == 0
    assert "amap_agent" in captured
    request = captured["amap_agent"][0]
    assert request.agent_name == "amap-agent"
    assert request.metadata["agent_name"] == "amap-agent"
    # A successful injection is not a filter action: INFO does not log every injection (the final owner is on the Sampling hit line)
    assert not any("filled from entry" in r.getMessage() for r in caplog.records)
    # Trace received is after the S3' backfill: instance identity uses slots (registrationId),
    # and no longer prints agent_name=; the injection result is asserted via request.agent_name.
    assert any(
        "Trace received:" in r.getMessage()
        and "tags=['ai_chat', 'amap-agent']" in r.getMessage()
        and "slots=" in r.getMessage()
        and "agent_name=" not in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_missing_agent_name_no_source_discarded() -> None:
    """Nameless + entry undeclared + registry has no candidates → discard, because there is nothing to attribute to."""
    evaluator = _make_evaluator(
        [EvalConfigEntry(evaluatorName="example", sampling=1.0)],
    )
    msg = _make_msg(metadata={})

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=[],
    ):
        await evaluator(msg)

    assert evaluator._funnel.filter_agent_name == 1
    evaluator._evaluator_service.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_single_candidate_legacy_behavior_preserved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single candidate + an entry with no agentName → behavior matches the old version (inject the candidate + evaluate normally)."""
    evaluator = _make_evaluator(
        [EvalConfigEntry(evaluatorName="example", sampling=1.0)],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured[name] = [request]
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(metadata={})  # OTel shape: no top-level agent_name

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["amap-agent"],
    ):
        await evaluator(msg)

    assert evaluator._funnel.filter_agent_name == 0
    request = captured["example"][0]
    assert request.agent_name == "amap-agent"
    assert request.metadata["agent_name"] == "amap-agent"


@pytest.mark.asyncio
async def test_no_candidates_still_discarded() -> None:
    """No registry candidates + no metadata value → still discarded (admission protection is unchanged)."""
    evaluator = _make_evaluator(
        [EvalConfigEntry(evaluatorName="example", sampling=1.0)],
    )
    msg = _make_msg(metadata={})

    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=[],
    ):
        await evaluator(msg)

    assert evaluator._funnel.filter_agent_name == 1
    evaluator._evaluator_service.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_agent_not_declared_in_evaluators_blocked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The SDK reports search_ai_card but evaluators declare only amap-agent → blocked."""
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["amap-agent.stream.ai_chat.standard"],
                sampling=1.0,
            ),
        ],
    )
    msg = _make_msg(
        name="amap-agent.stream.ai_chat.standard",
        metadata={"agent_name": "search_ai_card"},
    )

    with (
        patch(
            "benchmark.streaming.consumer.get_agent_names_by_project_id",
            return_value=["amap-agent"],
        ),
        caplog.at_level(logging.INFO),
    ):
        await evaluator(msg)

    assert evaluator._funnel.filter_agent_name == 1
    assert any(
        "no evaluator entry accepts reported agent" in r.message and "slots=" in r.message
        for r in caplog.records
    )
    evaluator._evaluator_service.evaluate.assert_not_called()


def test_resolve_missing_agent_name_prefers_benchmark_entry() -> None:
    """When metadata is missing: use the registered agentName only when trace.name hits the entry name."""
    configs = [
        EvalConfigEntry(
            evaluatorName="amap_agent",
            agentName="amap-agent",
            name=["amap-agent.stream.ai_chat.standard"],
            sampling=1.0,
        ),
    ]
    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["vui_agent"],
    ):
        name, source = resolve_missing_agent_name(
            configs,
            project_id=_PROJECT,
            tags=["ai-chat", "ai_chat", "amap-agent", "amap_agent_strategy__default"],
            trace_name="amap-agent.stream.ai_chat.standard",
        )
    assert name == "amap-agent"
    assert source == "entry"


def test_resolve_missing_agent_name_ignores_entry_without_name_hit() -> None:
    """A name miss (or a tags-only hit) must not stamp in the registered agentName; fall back to the registry."""
    configs = [
        EvalConfigEntry(
            evaluatorName="amap_agent",
            agentName="amap-agent",
            name=["amap-agent.stream.ai_chat.standard"],
            tags=["amap-vui-agent"],
            sampling=1.0,
        ),
    ]
    with patch(
        "benchmark.streaming.consumer.get_agent_names_by_project_id",
        return_value=["vui_agent"],
    ):
        # Matches the production log: the vui stream name is not in the entry name list
        name, source = resolve_missing_agent_name(
            configs,
            project_id=_PROJECT,
            tags=["33.35.28.201", "amap-vui-agent", "sa128", "vui"],
            trace_name="amap-vui-agent.voice_ws.vui_agent",
        )
    assert name == "vui_agent"
    assert source == "registry"


@pytest.mark.asyncio
async def test_sdk_underscore_name_matches_hyphen_scope_declaration() -> None:
    """A page scope that declares amap-agent (hyphen) must not filter out amap_agent reported by the SDK.

    Historical bug: S4e matched the string exactly, so after amap_agent_benchmark was registered most
    traces were dropped by filter_agent_name before sampling, and the sampling base collapsed.
    """
    evaluator = _make_evaluator(
        [
            EvalConfigEntry(
                evaluatorName="amap_agent",
                agentName="amap-agent",
                name=["amap-agent.stream.ai_chat.standard"],
                sampling=1.0,
            ),
        ],
    )
    captured: dict[str, list] = {}

    async def _fake_evaluate(name: str, request: object, **kwargs: object) -> object:
        captured[name] = [request]
        return MagicMock()

    evaluator._evaluator_service.evaluate = AsyncMock(side_effect=_fake_evaluate)
    msg = _make_msg(metadata={"agent_name": "amap_agent"})

    await evaluator(msg)

    assert evaluator._funnel.filter_agent_name == 0
    assert evaluator._funnel.sample_hit == 1
    assert "amap_agent" in captured
