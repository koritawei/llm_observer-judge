"""Unit tests for EvaluatorService orchestration (select/before/eval/after).

The service speaks the domain types (EvaluatorRequest → EvaluatorResult); the
request/verdict cross the agent boundary as JSON in the message / response content
(not metadata), so the fake agent round-trips with the same models.
"""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.libs.agenthost.agents.request import AgentRequest, AgentResponse, ResponseStatus
from benchmark.domain.schemas import ScoreWritebackPolicy
from benchmark.evaluators.base import (
    EvaluatorRequest,
    EvaluatorResult,
    decode_eval_text,
    encode_eval_error,
    encode_eval_json,
)
from benchmark.services.evaluation import EvalRunOptions, EvaluatorService
from benchmark.trace.http_backend import ObservationListQuery
from benchmark.trace.resolver import TraceResolver
from benchmark.trace.tools import TraceToolProvider
from benchmark.utils.exceptions import EvaluatorError, evaluator_status


class _StubAgentManager:
    """Minimal stand-in for runtime.agent_manager (get by name; KeyError on miss)."""

    def __init__(self, agents: dict[str, "_FakeAgent"]) -> None:
        self._agents = agents

    def get(self, name: str) -> "_FakeAgent":
        if name not in self._agents:
            raise KeyError(name)
        return self._agents[name]


class _FakeAgent:
    def __init__(self, response: AgentResponse) -> None:
        self._response = response
        self.seen: list[EvaluatorRequest] = []

    async def call(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        parts = request.messages[0].content if request.messages else []
        self.seen.append(EvaluatorRequest.model_validate_json(decode_eval_text(parts) or "{}"))
        return self._response


def _success_agent(
    score: float = 7.0,
    reason: str = "ok",
    *,
    extra: dict[str, object] | None = None,
    sub_scores: dict[str, float] | None = None,
) -> _FakeAgent:
    return _FakeAgent(
        AgentResponse(
            traceid="t",
            status=ResponseStatus.SUCCESS,
            content_parts=encode_eval_json(
                EvaluatorResult(
                    score=score,
                    reason=reason,
                    extra=extra or {},
                    sub_scores=sub_scores,
                )
            ),
        )
    )


def _failed_agent(error_type: str) -> _FakeAgent:
    return _FakeAgent(
        AgentResponse(
            traceid="t",
            status=ResponseStatus.FAILED,
            content_parts=encode_eval_error(
                EvaluatorError(
                    code=evaluator_status(error_type), error_type=error_type, message="boom"
                )
            ),
        )
    )


def _request(*, trace_id: str = "t-1") -> EvaluatorRequest:
    return EvaluatorRequest(
        input="你是谁",
        output="我是助手",
        metadata={"trace_id": trace_id},
        trace_id=trace_id,
        agent_name="vehicle-agent",
        project_id="p-9",
    )


def _service(
    agent: _FakeAgent, *, source_score_client_provider: object | None = None
) -> EvaluatorService:
    from benchmark.services import TraceRecordService

    class _NoTools:
        def get_provider(self, name: str) -> object:
            raise KeyError(name)

    runtime = SimpleNamespace(
        agent_manager=_StubAgentManager({"judge_example": agent}),
        tool_manager=_NoTools(),
    )
    return EvaluatorService(
        runtime=runtime,
        trace_record_service=TraceRecordService(),
        source_score_client_provider=source_score_client_provider,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_after_eval_writes_score_to_configured_source_project() -> None:
    client = MagicMock()
    provider = MagicMock()
    provider.get_client.return_value = client
    svc = _service(
        _success_agent(score=1.0, reason="partial"),
        source_score_client_provider=provider,
    )

    await svc.evaluate(
        "example",
        _request(trace_id="source-trace-1"),
        options=EvalRunOptions(
            score_writeback=ScoreWritebackPolicy(enabled=True, scoreName="vehicle_quality")
        ),
    )

    provider.get_client.assert_called_once_with("p-9")
    client.create_score.assert_called_once_with(
        trace_id="source-trace-1",
        name="vehicle_quality",
        value=1.0,
        data_type="NUMERIC",
        comment="partial",
        metadata={"stage": "judge"},
    )


@pytest.mark.asyncio
async def test_after_eval_skips_score_writeback_when_policy_disabled() -> None:
    provider = MagicMock()
    svc = _service(_success_agent(), source_score_client_provider=provider)

    await svc.evaluate(
        "example",
        _request(),
        options=EvalRunOptions(
            score_writeback=ScoreWritebackPolicy(enabled=False, scoreName="unused")
        ),
    )

    provider.get_client.assert_not_called()


@pytest.mark.asyncio
async def test_after_eval_keeps_scoring_when_source_project_is_unconfigured() -> None:
    provider = MagicMock()
    provider.get_client.return_value = None
    svc = _service(_success_agent(), source_score_client_provider=provider)

    result = await svc.evaluate(
        "example",
        _request(),
        options=EvalRunOptions(
            score_writeback=ScoreWritebackPolicy(enabled=True, scoreName="vehicle_quality")
        ),
    )

    assert result.score == 7.0
    provider.get_client.assert_called_once_with("p-9")


@pytest.mark.asyncio
async def test_after_eval_persists_judge_json() -> None:
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=7.0, reason="looks good"))

    await svc.evaluate("example", _request())

    rec = await TraceRecord.filter(trace_id="t-1").first()
    assert rec is not None
    assert rec.status == "scored"
    assert rec.judge_json["score"] == "7.0"
    assert rec.judge_json["reason"] == "looks good"
    assert "extra" not in rec.judge_json
    assert "sub_scores" not in rec.judge_json
    # Kafka/HTTP-like request: no dataset keys → no dataset block in meta
    assert not (rec.trace_meta_json or {}).get("dataset")
    # Full IO is always stamped when present (even without pipeline_source).
    assert rec.trace_meta_json["input"] == "你是谁"
    assert rec.trace_meta_json["output"] == "我是助手"


@pytest.mark.asyncio
async def test_after_eval_skips_stream_table_for_benchmark_batch() -> None:
    """When a batch evaluation passes no writer, it must not write trace_pipeline_records (the streaming table)."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=2.0, reason="batch"))
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={"pipeline_source": "benchmark", "dataset_item_id": "search_eval-1:1"},
        trace_id="search-batch-trace",
        agent_name="search_ai_card",
        project_id="cmpwln0qs04fo6j8eryfdgznb",
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="search-batch-trace").first()
    assert rec is None


@pytest.mark.asyncio
async def test_after_eval_persists_dataset_trace_meta() -> None:
    """Dataset-eval metadata → trace_meta_json.source = dataset-name-run."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=1.0, reason="ok"))
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={
            "pipeline_source": "dataset-vehicle-baseline-v1-run-abc",
            "dataset_name": "vehicle-baseline-v1",
            "dataset_item_id": "item-9",
            "dataset_run_name": "run-abc",
        },
        trace_id="dataset:item-9",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="dataset:item-9").first()
    assert rec is not None
    assert rec.trace_meta_json["source"] == "dataset-vehicle-baseline-v1-run-abc"
    dataset = (rec.trace_meta_json or {}).get("dataset")
    assert dataset == {
        "name": "vehicle-baseline-v1",
        "item_id": "item-9",
        "run_name": "run-abc",
    }
    assert rec.trace_meta_json["input"] == "q"
    assert rec.trace_meta_json["output"] == "a"


@pytest.mark.asyncio
async def test_after_eval_infers_composite_dataset_source() -> None:
    """Missing pipeline_source but dataset_* keys → composite source stamp."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=1.0, reason="ok"))
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={
            "dataset_name": "ds-infer",
            "dataset_item_id": "item-1",
            "dataset_run_name": "run-z",
        },
        trace_id="dataset:item-1",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="dataset:item-1").first()
    assert rec is not None
    assert rec.trace_meta_json["source"] == "dataset-ds-infer-run-z"


@pytest.mark.asyncio
async def test_after_eval_rewrites_bare_and_plus_dataset_source() -> None:
    """Bare ``dataset`` / legacy ``dataset+…`` are rewritten to composite form."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=1.0, reason="ok"))
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={
            "pipeline_source": "dataset",
            "dataset_name": "ds-rewrite",
            "dataset_item_id": "item-r",
            "dataset_run_name": "run-r",
        },
        trace_id="dataset:item-r",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="dataset:item-r").first()
    assert rec is not None
    assert rec.trace_meta_json["source"] == "dataset-ds-rewrite-run-r"

    svc2 = _service(_success_agent(score=1.0, reason="ok"))
    req2 = EvaluatorRequest(
        input="q",
        output="a",
        metadata={
            "pipeline_source": "dataset+old+r0",
            "dataset_name": "old",
            "dataset_item_id": "item-plus",
            "dataset_run_name": "r0",
        },
        trace_id="dataset:item-plus",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc2.evaluate("example", req2)

    rec2 = await TraceRecord.filter(trace_id="dataset:item-plus").first()
    assert rec2 is not None
    assert rec2.trace_meta_json["source"] == "dataset-old-r0"


@pytest.mark.asyncio
async def test_after_eval_persists_http_source() -> None:
    """HTTP path stamps trace_meta_json.source=http."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=1.0, reason="ok"))
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={"pipeline_source": "http", "scene": "non_voice"},
        trace_id="t-http-1",
        agent_name="vehicle-agent",
        project_id="p-9",
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="t-http-1").first()
    assert rec is not None
    assert rec.trace_meta_json["source"] == "http"
    assert "dataset" not in (rec.trace_meta_json or {})
    assert rec.trace_meta_json["input"] == "q"
    assert rec.trace_meta_json["output"] == "a"


@pytest.mark.asyncio
async def test_after_eval_persists_metaq_source() -> None:
    """MetaQ (luoshu) path stamps trace_meta_json.source=metaq (not conflated with kafka)."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=1.0, reason="ok"))
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={"pipeline_source": "metaq", "scene": "non_voice"},
        trace_id="t-metaq-1",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="t-metaq-1").first()
    assert rec is not None
    assert rec.trace_meta_json["source"] == "metaq"
    assert "dataset" not in (rec.trace_meta_json or {})
    assert rec.trace_meta_json["input"] == "q"
    assert rec.trace_meta_json["output"] == "a"


@pytest.mark.asyncio
async def test_after_eval_persists_sidecar_source() -> None:
    """sidecar-trace path stamps trace_meta_json.source=sidecar (not kafka/metaq)."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=1.0, reason="ok"))
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={"pipeline_source": "sidecar", "scene": "non_voice"},
        trace_id="t-sidecar-1",
        agent_name="amap-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="t-sidecar-1").first()
    assert rec is not None
    assert rec.trace_meta_json["source"] == "sidecar"
    assert "dataset" not in (rec.trace_meta_json or {})


@pytest.mark.asyncio
async def test_after_eval_persists_kafka_message_meta() -> None:
    """Kafka transport metadata is copied into trace_meta_json.kafka."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=1.0, reason="ok"))
    req = EvaluatorRequest(
        input={"query": "去哪"},
        output={"answer": "附近"},
        metadata={
            "pipeline_source": "kafka",
            "agentName": "vehicle-agent",
            "kafka": {
                "topic": "langfuse.records",
                "partition": 1,
                "offset": 88,
                "key": "p:t",
            },
        },
        trace_id="t-kafka-meta-1",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="t-kafka-meta-1").first()
    assert rec is not None
    assert rec.trace_meta_json["source"] == "kafka"
    assert rec.trace_meta_json["kafka"] == {
        "topic": "langfuse.records",
        "partition": 1,
        "offset": 88,
        "key": "p:t",
    }
    assert rec.trace_meta_json["input"] == {"query": "去哪"}
    assert rec.trace_meta_json["output"] == {"answer": "附近"}
    assert (
        rec.trace_meta_json["trace_url"]
        == "https://agentfuse-v4.amap-inc.com/project/p-9/traces/t-kafka-meta-1"
        "?traceId=t-kafka-meta-1"
    )


@pytest.mark.asyncio
async def test_after_eval_no_dataset_meta_without_dataset_keys() -> None:
    """No pipeline_source / dataset_* → no source/dataset, but IO still stamps."""
    from benchmark.records.models import TraceRecord

    svc = _service(_success_agent(score=1.0, reason="ok"))
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={"scene": "non_voice"},
        trace_id="t-kafka-1",
        agent_name="vehicle-agent",
        project_id="p-9",
    )

    await svc.evaluate("example", req)

    rec = await TraceRecord.filter(trace_id="t-kafka-1").first()
    assert rec is not None
    assert rec.trace_meta_json is not None
    assert "dataset" not in rec.trace_meta_json
    assert "source" not in rec.trace_meta_json
    assert rec.trace_meta_json["input"] == "q"
    assert rec.trace_meta_json["output"] == "a"


@pytest.mark.asyncio
async def test_after_eval_persists_extra_and_sub_scores() -> None:
    """amap-style nested ``extra`` + flat ``sub_scores`` both land in judge_json."""
    from benchmark.records.models import TraceRecord

    extra = {
        "completion": {"score": 1, "reason": "ok"},
        "tool_call": {"score": 1, "reason": "ok"},
        "expression": {"score": 5, "reason": "natural"},
    }
    svc = _service(
        _success_agent(
            score=1.0,
            reason="aggregated",
            extra=extra,
            sub_scores={"completion": 1.0, "tool_call": 1.0, "expression": 5.0},
        )
    )

    await svc.evaluate("example", _request(trace_id="t-extra-1"))

    rec = await TraceRecord.filter(trace_id="t-extra-1").first()
    assert rec is not None
    assert rec.judge_json["score"] == "1.0"
    assert rec.judge_json["reason"] == "aggregated"
    assert rec.judge_json["extra"] == extra
    assert rec.judge_json["sub_scores"] == {
        "completion": 1.0,
        "tool_call": 1.0,
        "expression": 5.0,
    }


@pytest.mark.asyncio
async def test_metadata_missing_short_circuits_before_agent_and_db(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The required trio is missing → fail fast at the entry, the LLM is not called, the DB is not written, and an ERROR log is visible.

    Replaces the old ``test_after_eval_warns_and_skips_when_three_tuple_incomplete``:
    the guard is in front, and the branch where after_eval silently skips is unreachable on the production path.
    """
    from benchmark.records.models import TraceRecord
    from benchmark.utils.exceptions import EvaluatorError

    agent = _success_agent(score=0.0, reason="虚构 POI")
    svc = _service(agent)

    req_all_missing = EvaluatorRequest(
        input=None,
        output=None,
        metadata={},
        trace_id=None,
        agent_name=None,
        project_id=None,
    )

    with caplog.at_level("ERROR", logger="agent_eval.evaluator"):
        with pytest.raises(EvaluatorError) as exc:
            await svc.evaluate("example", req_all_missing)
    assert exc.value.error_type == "metadata_missing"
    assert exc.value.code == 400
    # The LLM entry is blocked
    assert agent.seen == []
    # The DB is not written
    rec = await TraceRecord.filter(trace_id=None).first()
    assert rec is None
    # An ERROR log exists, and it is observable
    err_records = [r for r in caplog.records if "evaluator.metadata_missing" in r.message]
    assert len(err_records) == 1


@pytest.mark.asyncio
async def test_evaluate_returns_result() -> None:
    svc = _service(_success_agent(score=7.0, reason="ok"))

    result = await svc.evaluate("example", _request())

    assert isinstance(result, EvaluatorResult)
    assert result.score == 7.0
    assert result.reason == "ok"


@pytest.mark.asyncio
async def test_evaluate_bridges_expected_output_to_output_when_output_none() -> None:
    """Dataset-eval: expected_output bridges to output for judges that read output."""
    agent = _success_agent(score=1.0, reason="bridged")
    svc = _service(agent)
    req = EvaluatorRequest(
        input="q",
        output=None,
        expected_output={"a": "ref"},
        metadata={},
        trace_id="t-bridge-1",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc.evaluate("example", req)

    assert len(agent.seen) == 1
    assert agent.seen[0].output == {"a": "ref"}
    assert agent.seen[0].expected_output == {"a": "ref"}


@pytest.mark.asyncio
async def test_evaluate_does_not_overwrite_explicit_output_with_expected_output() -> None:
    agent = _success_agent(score=1.0, reason="keep")
    svc = _service(agent)
    req = EvaluatorRequest(
        input="q",
        output="scored-reply",
        expected_output="should-not-replace",
        metadata={},
        trace_id="t-bridge-2",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    await svc.evaluate("example", req)

    assert agent.seen[0].output == "scored-reply"
    assert agent.seen[0].expected_output == "should-not-replace"


@pytest.mark.asyncio
async def test_evaluate_wraps_resolve_and_judge_in_one_report_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.services import evaluation

    events: list[tuple[str, object]] = []
    root = MagicMock()
    root.trace_id = "abcdef0123456789abcdef0123456789"
    root_active = False

    @contextmanager
    def fake_root(**kwargs: object):
        nonlocal root_active
        events.append(("root_open", kwargs))
        root_active = True
        try:
            yield root
        finally:
            root_active = False
            events.append(("root_close", None))

    agent = _success_agent()
    svc = _service(agent)
    original_before_eval = svc.before_eval

    async def tracked_before_eval(request: EvaluatorRequest) -> EvaluatorRequest:
        assert root_active
        events.append(("resolve", None))
        return await original_before_eval(request)

    original_eval = svc._eval

    async def tracked_eval(
        selected_agent: _FakeAgent,
        request: EvaluatorRequest,
    ) -> EvaluatorResult:
        assert root_active
        events.append(("judge", None))
        return await original_eval(selected_agent, request)

    monkeypatch.setattr(evaluation, "eval_report_root", fake_root)
    monkeypatch.setattr(svc, "before_eval", tracked_before_eval)
    monkeypatch.setattr(svc, "_eval", tracked_eval)

    result = await svc.evaluate("example", _request(trace_id="trace-123456789"))

    assert result.score == 7.0
    assert "trace_id" not in result.model_dump()
    assert [event[0] for event in events] == [
        "root_open",
        "resolve",
        "judge",
        "root_close",
    ]
    root_kwargs = events[0][1]
    assert isinstance(root_kwargs, dict)
    from benchmark.services._stage_key import STAGE_JUDGE
    from benchmark.utils.trace_naming import build_eval_trace_name

    assert root_kwargs["name"] == build_eval_trace_name("vehicle-agent")
    assert root_kwargs["input_value"] == {
        "evaluator_name": "example",
        "trace_id": "trace-123456789",
        "project_id": "p-9",
        "agent_name": "vehicle-agent",
        "stage": STAGE_JUDGE,
    }
    root.update.assert_called_once_with(
        output={
            "status": "completed",
            "score": 7.0,
        }
    )


@pytest.mark.asyncio
async def test_evaluate_skips_trace_report_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark path: enable_langfuse_report=False must not open eval_report_root."""
    from benchmark.services import evaluation

    called = False

    @contextmanager
    def fake_root(**_kwargs: object):
        nonlocal called
        called = True
        yield MagicMock()

    monkeypatch.setattr(evaluation, "eval_report_root", fake_root)
    svc = _service(_success_agent(score=1.0, reason="ok"))

    result = await svc.evaluate(
        "example",
        _request(),
        options=EvalRunOptions(
            resolve_trace=False,
            enable_trace_report=False,
        ),
    )

    assert result.score == 1.0
    assert "trace_id" not in result.model_dump()
    assert called is False


@pytest.mark.asyncio
async def test_shared_evaluation_returns_original_verdict(monkeypatch) -> None:
    svc = _service(_success_agent())
    verdict = EvaluatorResult(score=7.0, reason="ok")
    monkeypatch.setattr(svc, "_eval", AsyncMock(return_value=verdict))
    after_eval = AsyncMock()
    monkeypatch.setattr(svc, "after_eval", after_eval)
    result = await svc.evaluate("example", _request(), options=EvalRunOptions(resolve_trace=False))
    assert result is verdict
    assert after_eval.call_args.args[2] is verdict
    assert "trace_id" not in result.model_dump()


@pytest.mark.asyncio
async def test_evaluate_still_opens_report_root_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset-eval / HTTP default: eval_report_root still opens."""
    from benchmark.services import evaluation

    opened = False

    @contextmanager
    def fake_root(**_kwargs: object):
        nonlocal opened
        opened = True
        yield MagicMock()

    monkeypatch.setattr(evaluation, "eval_report_root", fake_root)
    svc = _service(_success_agent())

    await svc.evaluate(
        "example",
        _request(),
        options=EvalRunOptions(resolve_trace=False),
    )

    assert opened is True


@pytest.mark.asyncio
async def test_eval_serializes_request_to_agent_content() -> None:
    agent = _success_agent()
    svc = _service(agent)

    await svc.evaluate("example", _request())

    # the agent received the request as content JSON and deserialized it back
    req = agent.seen[0]
    assert isinstance(req, EvaluatorRequest)
    assert req.input == "你是谁"
    assert req.output == "我是助手"
    assert req.trace_id == "t-1"
    assert req.agent_name == "vehicle-agent"
    assert req.project_id == "p-9"


@pytest.mark.asyncio
async def test_before_eval_skips_without_trace_provider() -> None:
    svc = _service(_success_agent())
    req = _request()
    out = await svc.before_eval(req)
    assert out.trace_id == req.trace_id


@pytest.mark.asyncio
async def test_before_eval_rewrites_to_trace_primary_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.services import TraceRecordService, evaluation
    from benchmark.trace.tools import TraceToolProvider

    agent = _success_agent()
    fake_resolver = AsyncMock()
    fake_resolver.resolve_primary_id_for.return_value = "lf-primary-99"
    resolve_span = MagicMock()
    start_span = MagicMock(return_value=resolve_span)
    finish_span = MagicMock()
    provider = TraceToolProvider(fake_resolver)

    class _Tools:
        def get_provider(self, name: str) -> object:
            assert name == "trace"
            return provider

    runtime = SimpleNamespace(
        agent_manager=_StubAgentManager({"judge_example": agent}),
        tool_manager=_Tools(),
    )
    svc = EvaluatorService(runtime=runtime, trace_record_service=TraceRecordService())

    monkeypatch.setattr(evaluation, "start_report_child_span", start_span)
    monkeypatch.setattr(evaluation, "finish_report_child_span", finish_span)
    out = await svc.before_eval(_request(trace_id="biz-meta-1"))

    assert out.trace_id == "lf-primary-99"
    assert out.metadata["business_trace_id"] == "biz-meta-1"
    assert out.metadata["trace_id"] == "lf-primary-99"
    fake_resolver.resolve_primary_id_for.assert_awaited_once_with("p-9", "biz-meta-1")
    start_span.assert_called_once_with(
        name="trace.resolve_primary_id",
        input_value={"business_trace_id": "biz-meta-1", "project_id": "p-9"},
    )
    finish_span.assert_called_once_with(
        resolve_span,
        output={
            "business_trace_id": "biz-meta-1",
            "project_id": "p-9",
            "primary_id": "lf-primary-99",
            "status": "resolved",
        },
    )


@pytest.mark.asyncio
async def test_before_eval_miss_raises_trace_not_found_400() -> None:
    from unittest.mock import AsyncMock

    from benchmark.services import TraceRecordService
    from benchmark.trace.tools import TraceToolProvider

    agent = _success_agent()
    fake_resolver = AsyncMock()
    fake_resolver.resolve_primary_id_for.return_value = None
    provider = TraceToolProvider(fake_resolver)

    class _Tools:
        def get_provider(self, name: str) -> object:
            return provider

    runtime = SimpleNamespace(
        agent_manager=_StubAgentManager({"judge_example": agent}),
        tool_manager=_Tools(),
    )
    svc = EvaluatorService(runtime=runtime, trace_record_service=TraceRecordService())

    with pytest.raises(EvaluatorError) as exc:
        await svc.before_eval(_request(trace_id="biz-missing"))
    assert exc.value.error_type == "trace_not_found"
    assert exc.value.code == 400
    assert agent.seen == []


@pytest.mark.asyncio
async def test_before_eval_trace_http_error_raises_502() -> None:
    from unittest.mock import AsyncMock

    from benchmark.services import TraceRecordService
    from benchmark.trace.http_backend import TraceHttpError
    from benchmark.trace.tools import TraceToolProvider

    agent = _success_agent()
    fake_resolver = AsyncMock()
    fake_resolver.resolve_primary_id_for.side_effect = TraceHttpError(
        "trace filter request failed (status=422)",
        status_code=422,
    )
    provider = TraceToolProvider(fake_resolver)

    class _Tools:
        def get_provider(self, name: str) -> object:
            return provider

    runtime = SimpleNamespace(
        agent_manager=_StubAgentManager({"judge_example": agent}),
        tool_manager=_Tools(),
    )
    svc = EvaluatorService(runtime=runtime, trace_record_service=TraceRecordService())

    with pytest.raises(EvaluatorError) as exc:
        await svc.before_eval(_request(trace_id="biz-timeout"))
    assert exc.value.error_type == "trace_unavailable"
    assert exc.value.code == 502
    assert agent.seen == []


@pytest.mark.asyncio
async def test_before_eval_cancelled_finishes_span_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.services import TraceRecordService, evaluation
    from benchmark.trace.tools import TraceToolProvider

    cancelled = asyncio.CancelledError()
    fake_resolver = AsyncMock()
    fake_resolver.resolve_primary_id_for.side_effect = cancelled
    provider = TraceToolProvider(fake_resolver)
    resolve_span = MagicMock()
    finish_span = MagicMock()

    class _Tools:
        def get_provider(self, name: str) -> object:
            assert name == "trace"
            return provider

    runtime = SimpleNamespace(
        agent_manager=_StubAgentManager({"judge_example": _success_agent()}),
        tool_manager=_Tools(),
    )
    svc = EvaluatorService(runtime=runtime, trace_record_service=TraceRecordService())
    monkeypatch.setattr(
        evaluation,
        "start_report_child_span",
        MagicMock(return_value=resolve_span),
    )
    monkeypatch.setattr(evaluation, "finish_report_child_span", finish_span)

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await svc.before_eval(_request(trace_id="biz-cancelled"))

    assert exc_info.value is cancelled
    finish_span.assert_called_once_with(resolve_span, exc=cancelled)


@pytest.mark.asyncio
async def test_evaluate_short_circuits_on_trace_resolve_miss() -> None:
    from unittest.mock import AsyncMock

    from benchmark.records.models import TraceRecord
    from benchmark.services import TraceRecordService
    from benchmark.trace.tools import TraceToolProvider

    agent = _success_agent()
    fake_resolver = AsyncMock()
    fake_resolver.resolve_primary_id_for.return_value = None
    provider = TraceToolProvider(fake_resolver)

    class _Tools:
        def get_provider(self, name: str) -> object:
            return provider

    runtime = SimpleNamespace(
        agent_manager=_StubAgentManager({"judge_example": agent}),
        tool_manager=_Tools(),
    )
    svc = EvaluatorService(runtime=runtime, trace_record_service=TraceRecordService())

    with pytest.raises(EvaluatorError) as exc:
        await svc.evaluate("example", _request(trace_id="biz-gone"))
    assert exc.value.error_type == "trace_not_found"
    assert exc.value.code == 400
    assert agent.seen == []
    assert await TraceRecord.filter(trace_id="biz-gone").first() is None


@pytest.mark.asyncio
async def test_unknown_model_raises_evaluator_error() -> None:
    svc = _service(_success_agent())
    with pytest.raises(EvaluatorError) as exc:
        await svc.evaluate("ghost", _request())
    assert exc.value.error_type == "model_not_found"


@pytest.mark.asyncio
async def test_prompt_not_found_raises_evaluator_error() -> None:
    svc = _service(_failed_agent("evaluator_prompt_not_found"))
    with pytest.raises(EvaluatorError) as exc:
        await svc.evaluate("example", _request())
    assert exc.value.error_type == "evaluator_prompt_not_found"


@pytest.mark.asyncio
async def test_llm_failure_raises_evaluator_error() -> None:
    svc = _service(_failed_agent("llm_failed"))
    with pytest.raises(EvaluatorError) as exc:
        await svc.evaluate("example", _request())
    assert exc.value.error_type == "llm_failed"


@pytest.mark.asyncio
async def test_metadata_missing_raises_evaluator_error_with_listed_fields() -> None:
    """The required trio is missing → fail fast at the entry, 400 metadata_missing, the LLM is not called, and the DB is not written."""
    agent = _success_agent()
    svc = _service(agent)

    req = EvaluatorRequest(
        input="你是谁",
        output="我是助手",
        metadata={},
        trace_id=None,
        agent_name=None,
        project_id=None,
    )

    with pytest.raises(EvaluatorError) as exc:
        await svc.evaluate("example", req)
    assert exc.value.error_type == "metadata_missing"
    assert exc.value.code == 400
    msg = exc.value.message
    assert "trace_id" in msg
    assert "project_id" in msg
    assert "agent_name" in msg
    # The LLM entry is blocked, and the agent should not be called
    assert agent.seen == []


@pytest.mark.asyncio
async def test_metadata_missing_partial_returns_listed_field() -> None:
    """When only one item is missing, the error message lists only the missing field name."""
    agent = _success_agent()
    svc = _service(agent)

    req = EvaluatorRequest(
        input="x",
        output="y",
        metadata={},
        trace_id="t-1",
        agent_name=None,
        project_id="p-9",
    )

    with pytest.raises(EvaluatorError) as exc:
        await svc.evaluate("example", req)
    assert exc.value.error_type == "metadata_missing"
    assert exc.value.message == "required metadata missing: agent_name"
    assert agent.seen == []


@pytest.mark.asyncio
async def test_metadata_whitespace_treated_as_missing() -> None:
    """An empty string and pure whitespace also count as missing."""
    agent = _success_agent()
    svc = _service(agent)

    req = EvaluatorRequest(
        input="x",
        output="y",
        metadata={},
        trace_id="   ",
        agent_name="a",
        project_id="p",
    )

    with pytest.raises(EvaluatorError) as exc:
        await svc.evaluate("example", req)
    assert exc.value.error_type == "metadata_missing"
    assert "trace_id" in exc.value.message


@pytest.mark.asyncio
async def test_metadata_complete_passes_guard() -> None:
    """The required trio is complete → the guard passes, the agent is called, and a normal EvaluatorResult is returned."""
    agent = _success_agent(score=7.0, reason="ok")
    svc = _service(agent)

    result = await svc.evaluate("example", _request(trace_id="t-guard"))

    assert result.score == 7.0
    assert len(agent.seen) == 1


@pytest.mark.asyncio
async def test_enrich_request_io_preserves_json_value_types() -> None:
    """When backfilling a dict from the full trace, keep the original JsonValue types."""
    from benchmark.trace.schema import TraceDocument
    from benchmark.trace.tools import TraceToolProvider

    dict_input = {"request_body": {"query": "西湖怎么走"}}
    dict_output = {"answer": {"text": "沿西湖步行"}}
    resolver = AsyncMock()
    resolver.fetch_full_trace.return_value = TraceDocument(
        input=dict_input,
        output=dict_output,
    )

    class _Tools:
        def get_provider(self, name: str) -> object:
            assert name == "trace"
            return TraceToolProvider(resolver)  # type: ignore[arg-type]

    svc = _service(_success_agent(score=7.0, reason="ok"))
    svc._runtime.tool_manager = _Tools()
    req = EvaluatorRequest(
        input=None,
        output=None,
        metadata={"trace_id": "lf-primary-1"},
        trace_id="lf-primary-1",
        agent_name="vehicle-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )

    enriched = await svc._enrich_request_io(req)

    assert isinstance(enriched.input, dict)
    assert enriched.input == dict_input
    assert isinstance(enriched.output, dict)
    assert enriched.output == dict_output
    resolver.fetch_full_trace.assert_awaited_once_with(
        "p-9",
        "lf-primary-1",
        fields="core,io",
    )


@pytest.mark.asyncio
async def test_enrich_request_io_keeps_kafka_field_and_backfills_only_none() -> None:
    """IO that Kafka already carried is not overwritten by the full trace; only a ``None`` field is filled."""
    from benchmark.trace.schema import TraceDocument
    from benchmark.trace.tools import TraceToolProvider

    resolver = AsyncMock()
    resolver.fetch_full_trace.return_value = TraceDocument(
        input={"stale": "must-not-overwrite"},
        output={"answer": "已从 Langfuse 补齐"},
    )

    class _Tools:
        def get_provider(self, name: str) -> object:
            assert name == "trace"
            return TraceToolProvider(resolver)  # type: ignore[arg-type]

    svc = _service(_success_agent(score=7.0, reason="ok"))
    svc._runtime.tool_manager = _Tools()
    req = _request(trace_id="lf-primary-2").model_copy(
        update={"input": {"request_body": {"query": "Kafka 原始输入"}}, "output": None}
    )

    enriched = await svc._enrich_request_io(req)

    assert enriched.input == {"request_body": {"query": "Kafka 原始输入"}}
    assert enriched.output == {"answer": "已从 Langfuse 补齐"}
    resolver.fetch_full_trace.assert_awaited_once_with(
        "p-9",
        "lf-primary-2",
        fields="core,io",
    )


@pytest.mark.asyncio
async def test_enrich_request_io_incomplete_trace_raises_without_calling_agent() -> None:
    """When the full trace is still missing IO, do not enter the evaluator with an empty context."""
    from benchmark.trace.schema import TraceDocument
    from benchmark.trace.tools import TraceToolProvider
    from benchmark.utils.exceptions import INCOMPLETE_TRACE_IO_CODE

    agent = _success_agent(score=7.0, reason="ok")
    resolver = AsyncMock()
    resolver.fetch_full_trace.return_value = TraceDocument(
        input={"request_body": {"query": "仅有输入"}},
        output=None,
    )

    class _Tools:
        def get_provider(self, name: str) -> object:
            assert name == "trace"
            return TraceToolProvider(resolver)  # type: ignore[arg-type]

    svc = _service(agent)
    svc._runtime.tool_manager = _Tools()
    req = _request(trace_id="lf-primary-3").model_copy(
        update={
            "input": None,
            "output": None,
            "skip_trace_resolve": True,
            "metadata": {"pipeline_source": "kafka"},
        }
    )

    with pytest.raises(EvaluatorError) as exc_info:
        await svc.evaluate("example", req)

    assert exc_info.value.error_type == INCOMPLETE_TRACE_IO_CODE
    assert agent.seen == []
    resolver.fetch_full_trace.assert_awaited_once_with(
        "p-9",
        "lf-primary-3",
        fields="core,io",
    )


@pytest.mark.asyncio
async def test_enrich_request_io_skipped_when_io_present() -> None:
    """When IO is already present, do not look it up again and do not overwrite the fields."""
    from benchmark.trace.tools import TraceToolProvider

    resolver = AsyncMock()

    class _Tools:
        def get_provider(self, name: str) -> object:
            assert name == "trace"
            return TraceToolProvider(resolver)  # type: ignore[arg-type]

    req = _request(trace_id="t-1")
    svc = _service(_success_agent(score=7.0, reason="ok"))
    svc._runtime.tool_manager = _Tools()

    enriched = await svc._enrich_request_io(req)

    assert enriched is req
    assert enriched.input == "你是谁"
    assert enriched.output == "我是助手"
    resolver.fetch_full_trace.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_binds_sidecar_observations_for_tools() -> None:
    """During the sidecar entry, create_bound / tools can read the local tree."""
    from benchmark.trace.tools import current_sidecar_observations

    seen: list[object] = []

    class _CaptureAgent(_FakeAgent):
        async def call(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
            seen.append(current_sidecar_observations())
            return await super().call(request, **kwargs)

    observations = [{"span_id": "s-1", "name": "root"}]
    agent = _CaptureAgent(
        AgentResponse(
            traceid="t",
            status=ResponseStatus.SUCCESS,
            content_parts=encode_eval_json(EvaluatorResult(score=1.0, reason="ok")),
        )
    )
    svc = _service(agent)
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={
            "pipeline_source": "sidecar",
            "sidecar_observations": observations,
        },
        trace_id="t-sidecar",
        agent_name="amap-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )
    await svc.evaluate("example", req, options=EvalRunOptions(resolve_trace=False))
    assert seen == [observations]
    assert current_sidecar_observations() is None


@pytest.mark.asyncio
async def test_vui_observation_context_fallback_uses_one_list_request() -> None:
    agent = _success_agent()
    resolver = SimpleNamespace(
        fetch_observations_once=AsyncMock(
            return_value=[
                {
                    "name": "lite.AmapChatModel",
                    "input": "<history>用户: 我要回家</history>"
                    "<page_detail><snapshot>首页</snapshot></page_detail>",
                }
            ]
        )
    )
    provider = TraceToolProvider(
        resolver,
        enable_vui_observation_context_fallback=True,
    )
    runtime = SimpleNamespace(
        agent_manager=_StubAgentManager({"judge_vui_agent": agent}),
        tool_manager=SimpleNamespace(get_provider=lambda name: provider),
    )
    from benchmark.services import TraceRecordService

    svc = EvaluatorService(runtime=runtime, trace_record_service=TraceRecordService())
    request = EvaluatorRequest(
        input={"request_body": {"query": "回家"}},
        output={"component": {"text": "已为您导航回家"}},
        trace_id="lf-trace-1",
        agent_name="vui-agent",
        project_id="p-1",
    )

    enriched = await svc._enrich_vui_observation_context("vui_agent", request)

    resolver.fetch_observations_once.assert_awaited_once_with("p-1", primary_id="lf-trace-1")
    assert enriched.input == request.input
    assert enriched.output == request.output
    assert enriched.observations == resolver.fetch_observations_once.return_value
    assert "_enriched_history" not in request.input
    await svc._eval(agent, enriched)
    assert agent.seen[-1].observations == enriched.observations


@pytest.mark.asyncio
@pytest.mark.parametrize("observations", [[], [{"name": "response", "output": "原始证据"}]])
async def test_explicit_vui_observations_skip_fallback(observations: list[dict]) -> None:
    svc = _service(_success_agent())
    resolver = SimpleNamespace(fetch_observations_once=AsyncMock())
    svc._resolve_trace_provider = lambda: SimpleNamespace(
        enable_vui_observation_context_fallback=True, resolver=resolver
    )
    request = _request().model_copy(
        update={"input": {"request_body": {"query": "回家"}}, "observations": observations}
    )

    enriched = await svc._enrich_vui_observation_context("vui_agent", request)

    assert enriched is request
    resolver.fetch_observations_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_vui_observation_page_requests_timestamps_without_detail_fanout() -> None:
    client = SimpleNamespace(list_observations_for_trace=AsyncMock(return_value=[]))
    resolver = TraceResolver({"p-1": SimpleNamespace()})
    resolver._make_lf_client = lambda instance: client

    assert await resolver.fetch_observations_once("p-1", primary_id="trace-1") == []

    client.list_observations_for_trace.assert_awaited_once_with(
        "trace-1", ObservationListQuery(limit=100, fields="core,basic,io,time")
    )


@pytest.mark.asyncio
async def test_vui_observation_context_fallback_respects_switch() -> None:
    resolver = SimpleNamespace(fetch_observations_once=AsyncMock())
    provider = SimpleNamespace(enable_vui_observation_context_fallback=False, resolver=resolver)
    runtime = SimpleNamespace(
        agent_manager=_StubAgentManager({}),
        tool_manager=SimpleNamespace(get_provider=lambda name: provider),
    )
    from benchmark.services import TraceRecordService

    svc = EvaluatorService(runtime=runtime, trace_record_service=TraceRecordService())
    request = EvaluatorRequest(
        input={"request_body": {"query": "回家"}},
        output={},
        trace_id="lf-trace-1",
        agent_name="vui-agent",
        project_id="p-1",
    )

    assert await svc._enrich_vui_observation_context("vui_agent", request) is request
    resolver.fetch_observations_once.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_binds_sidecar_observations_from_options() -> None:
    from benchmark.trace.tools import current_sidecar_observations

    seen: list[object] = []

    class _CaptureAgent(_FakeAgent):
        async def call(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
            seen.append(current_sidecar_observations())
            return await super().call(request, **kwargs)

    observations = [{"span_id": "s-opt", "name": "root"}]
    agent = _CaptureAgent(
        AgentResponse(
            traceid="t",
            status=ResponseStatus.SUCCESS,
            content_parts=encode_eval_json(EvaluatorResult(score=1.0, reason="ok")),
        )
    )
    svc = _service(agent)
    req = EvaluatorRequest(
        input="q",
        output="a",
        metadata={"pipeline_source": "sidecar"},
        trace_id="t-sidecar-opt",
        agent_name="amap-agent",
        project_id="p-9",
        skip_trace_resolve=True,
    )
    await svc.evaluate(
        "example",
        req,
        options=EvalRunOptions(resolve_trace=False, sidecar_observations=observations),
    )
    assert seen == [observations]
    assert current_sidecar_observations() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kafka", [True, False])
async def test_review_io_transport_error_retryability(kafka):
    from benchmark.utils.exceptions import TRACE_UNAVAILABLE_CODE

    svc = _service(_success_agent())
    resolver = SimpleNamespace(fetch_full_trace=AsyncMock(side_effect=TimeoutError()))
    svc._resolve_trace_provider = lambda: SimpleNamespace(resolver=resolver)
    req = _request().model_copy(
        update={"input": None, "metadata": {"pipeline_source": "kafka" if kafka else "benchmark"}}
    )
    if kafka:
        with pytest.raises(EvaluatorError) as err:
            await svc._enrich_request_io(req)
        assert err.value.error_type == TRACE_UNAVAILABLE_CODE
    else:
        # A benchmark other than search_eval still takes the best-effort backfill; a timeout does not block
        assert await svc._enrich_request_io(req) is req
        resolver.fetch_full_trace.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark_name", ["search_eval", "search_eval_v2"])
async def test_enrich_request_io_skips_trace_for_search_eval(benchmark_name: str) -> None:
    """Only search_eval / v2 skip the Langfuse IO backfill by name; other benchmarks are unaffected."""
    from benchmark.trace.tools import TraceToolProvider

    resolver = AsyncMock()

    class _Tools:
        def get_provider(self, name: str) -> object:
            assert name == "trace"
            return TraceToolProvider(resolver)  # type: ignore[arg-type]

    svc = _service(_success_agent())
    svc._runtime.tool_manager = _Tools()
    req = _request().model_copy(
        update={
            "input": None,
            "output": None,
            "metadata": {
                "pipeline_source": "benchmark",
                "benchmark_name": benchmark_name,
            },
        },
    )
    enriched = await svc._enrich_request_io(req)
    assert enriched is req
    resolver.fetch_full_trace.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_vui_compaction_error_is_nonfatal(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _service(_success_agent())
    svc._resolve_trace_provider = lambda: SimpleNamespace(
        enable_vui_observation_context_fallback=True,
        resolver=SimpleNamespace(fetch_observations_once=AsyncMock(return_value=[])),
    )
    monkeypatch.setattr(
        "benchmark.services.evaluation.compact_observations",
        lambda *a: (_ for _ in ()).throw(ValueError("malformed")),
    )
    req = _request().model_copy(update={"input": {"request_body": {"query": "回家"}}})
    assert await svc._enrich_vui_observation_context("vui_agent", req) is req
