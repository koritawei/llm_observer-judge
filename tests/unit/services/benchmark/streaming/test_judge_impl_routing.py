"""Unit tests for plugin-based streaming judge execution routing (§5.1: a registered judge_class takes effect).

Covers the fix at three breakpoints:
1. the ``StreamingJudgeImpl`` protocol check (execution_classes);
2. evaluator S8 dispatch (the plugin only takes over scoring; persistence / attribution still go through the unified layer);
3. the runner resolves and injects from ``tasks[].judge_class``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.domain.eval_config import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult
from benchmark.runs.common import execution_classes as ec_mod
from benchmark.runs.common.execution_classes import (
    ExecutionClassResolutionError,
    resolve_streaming_judge_class,
    validate_streaming_judge_protocol,
)
from benchmark.streaming.consumer import (
    BenchmarkStreamingEvaluator,
    BenchmarkStreamingEvaluatorConfig,
    BenchmarkStreamingEvaluatorDeps,
)
from benchmark.streaming.runner import StreamingBenchmarkRunner
from benchmark.streaming.schemas import RecordStreamMessage, TraceRecord
from benchmark.streaming.tasks import StreamingTaskSpec
from benchmark.libs.mq.consumer import ConsumedMessage


class _DemoStreamingJudge:
    """A business-specific streaming judge plugin (the target shape: registering it is enough for it to take effect)."""

    execution = "sync"
    provider = "local"

    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    async def evaluate(self, trace: object, config: object) -> EvaluatorResult:
        self.calls.append((trace, config))
        return EvaluatorResult(score=1.0, reason="demo")


class _InvalidAsyncJudgeNoSubmit:
    execution = "async"
    provider = "external"

    async def evaluate(self, trace: object, config: object) -> EvaluatorResult:
        return EvaluatorResult(score=0.0)


class _ValidAsyncJudge:
    execution = "async"
    provider = "external"

    async def submit(self, trace: object, config: object) -> str:
        return "job-1"


class _MissingProviderJudge:
    execution = "sync"
    provider = ""

    async def evaluate(self, trace: object, config: object) -> EvaluatorResult:
        return EvaluatorResult(score=0.0)


# ── 1. execution_classes protocol check ──────────────────────────────────────────


class TestValidateStreamingJudgeProtocol:
    def test_valid_class_with_evaluate(self) -> None:
        validate_streaming_judge_protocol(_DemoStreamingJudge, alias="demo")

    def test_marker_class_without_evaluate_is_allowed(self) -> None:
        validate_streaming_judge_protocol(
            ec_mod.EvaluatorServiceJudge, alias="evaluator_service_judge"
        )

    def test_async_without_submit_rejected(self) -> None:
        with pytest.raises(ExecutionClassResolutionError, match="must implement submit"):
            validate_streaming_judge_protocol(_InvalidAsyncJudgeNoSubmit, alias="bad")

    def test_async_with_submit_allowed(self) -> None:
        validate_streaming_judge_protocol(_ValidAsyncJudge, alias="ok")

    def test_empty_provider_rejected(self) -> None:
        with pytest.raises(ExecutionClassResolutionError, match="provider"):
            validate_streaming_judge_protocol(_MissingProviderJudge, alias="bad")


class TestResolveStreamingJudgeClass:
    def test_evaluator_service_judge_has_no_target(self) -> None:
        spec = resolve_streaming_judge_class("evaluator_service_judge")
        assert spec.alias == "evaluator_service_judge"
        assert spec.execution == "sync"
        assert spec.target is None  # Default = the production EvaluatorService route

    def test_custom_class_resolves_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(
            ec_mod.STREAMING_JUDGE_CLASS_REFS,
            "demo_judge",
            "benchmark.runs.common.execution_classes:EvaluatorServiceJudge",
        )
        monkeypatch.setattr(ec_mod, "load_plugin_class", lambda ref, *, what: _DemoStreamingJudge)
        spec = resolve_streaming_judge_class("demo_judge")
        assert (
            spec.target == "benchmark.runs.common.execution_classes:EvaluatorServiceJudge"
        )

    def test_unknown_alias_rejected(self) -> None:
        with pytest.raises(ExecutionClassResolutionError, match="unknown streaming judge_class"):
            resolve_streaming_judge_class("ieval_external_judge")  # A batch alias is rejected

    def test_batch_alias_not_in_streaming_registry(self) -> None:
        assert "ieval_external_judge" not in ec_mod.STREAMING_JUDGE_CLASS_REFS
        assert "search_eval_judge" not in ec_mod.STREAMING_JUDGE_CLASS_REFS


# ── 2. evaluator S8 dispatch (plugin scoring + unified-layer persistence and attribution)─────────────────────────


def _msg() -> ConsumedMessage[RecordStreamMessage]:
    record = TraceRecord(
        id="t-plugin",
        project_id="proj-1",
        environment="prod",
        metadata={"agent_name": "amap_agent"},
        input={"q": "test"},
        output={"a": "resp"},
    )
    return ConsumedMessage(
        topic="langfuse.records",
        payload=RecordStreamMessage(record_type="trace", record=record.model_dump()),
        key="proj-1:t-plugin",
        offset=1,
        partition=0,
    )


def _deps(evaluator_service: AsyncMock) -> BenchmarkStreamingEvaluatorDeps:
    config_store = MagicMock()
    config_store.get_configs.return_value = [
        EvalConfigEntry(evaluatorName="example", sampling=1.0),
    ]

    def _converter(observation: object, config: object) -> EvaluatorRequest:
        return EvaluatorRequest(metadata={}, skip_trace_resolve=True)

    return BenchmarkStreamingEvaluatorDeps(
        config_store=config_store,
        evaluator_service=evaluator_service,
        request_converter=_converter,
    )


@pytest.mark.asyncio
async def test_judge_impl_takes_over_scoring_and_unified_after_eval_runs() -> None:
    judge = _DemoStreamingJudge()
    evaluator_service = AsyncMock()
    evaluator = BenchmarkStreamingEvaluator(
        deps=_deps(evaluator_service),
        config=BenchmarkStreamingEvaluatorConfig(judge_impl=judge),
    )

    await evaluator(_msg())

    # The plugin takes over scoring
    assert len(judge.calls) == 1
    trace, cfg = judge.calls[0]
    assert trace.trace_id == "t-plugin"  # type: ignore[attr-defined]
    assert cfg.evaluator_name == "example"  # type: ignore[attr-defined]
    # It does not go through the production evaluate; persistence / attribution still go through the unified-layer after_eval
    evaluator_service.evaluate.assert_not_awaited()
    evaluator_service.after_eval.assert_awaited_once()
    args = evaluator_service.after_eval.await_args
    assert args.args[0] == "example"
    assert args.args[2].score == 1.0
    assert args.kwargs["options"] is None
    assert evaluator._funnel.eval_ok == 1


@pytest.mark.asyncio
async def test_judge_impl_with_skip_triage_passes_options() -> None:
    judge = _DemoStreamingJudge()
    evaluator_service = AsyncMock()
    evaluator = BenchmarkStreamingEvaluator(
        deps=_deps(evaluator_service),
        config=BenchmarkStreamingEvaluatorConfig(judge_impl=judge, skip_triage=True),
    )

    await evaluator(_msg())

    options = evaluator_service.after_eval.await_args.kwargs["options"]
    assert options is not None
    assert options.skip_triage is True


@pytest.mark.asyncio
async def test_no_judge_impl_keeps_legacy_routing() -> None:
    evaluator_service = AsyncMock()
    evaluator = BenchmarkStreamingEvaluator(deps=_deps(evaluator_service))

    await evaluator(_msg())

    evaluator_service.evaluate.assert_awaited_once()
    evaluator_service.after_eval.assert_not_awaited()


# ── 3. runner resolves and injects by judge_class ──────────────────────────────────────


class TestRunnerResolveJudgeImpl:
    def _runner(self) -> StreamingBenchmarkRunner:
        return StreamingBenchmarkRunner(
            evaluator_service=AsyncMock(),
            config_store=MagicMock(),
        )

    def test_null_judge_class_returns_none(self) -> None:
        impl = StreamingBenchmarkRunner._resolve_judge_impl(
            StreamingTaskSpec(name="t", topics=["a"], group_id="g"),
        )
        assert impl is None

    def test_evaluator_service_judge_returns_none(self) -> None:
        impl = StreamingBenchmarkRunner._resolve_judge_impl(
            StreamingTaskSpec(
                name="t",
                topics=["a"],
                group_id="g",
                judge_class="evaluator_service_judge",
            ),
        )
        assert impl is None

    def test_custom_class_loads_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from benchmark.streaming import runner as runner_mod

        monkeypatch.setitem(
            ec_mod.STREAMING_JUDGE_CLASS_REFS,
            "demo_judge",
            "x:Y",
        )
        monkeypatch.setattr(ec_mod, "load_plugin_class", lambda ref, *, what: _DemoStreamingJudge)
        monkeypatch.setattr(
            runner_mod,
            "load_execution_instance",
            lambda class_ref, *, what: _DemoStreamingJudge(),
        )
        impl = StreamingBenchmarkRunner._resolve_judge_impl(
            StreamingTaskSpec(
                name="t",
                topics=["a"],
                group_id="g",
                judge_class="demo_judge",
            ),
        )
        assert isinstance(impl, _DemoStreamingJudge)
