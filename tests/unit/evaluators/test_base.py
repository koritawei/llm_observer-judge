"""Unit tests for the BaseEvaluatorAgent contract (request parsing lives in the
service layer — see test_evaluator_service.py)."""
# ruff: noqa: PLR2004  (test-only: literal asserts / fixture params / local imports)

from types import SimpleNamespace

import pytest
from benchmark.libs.agenthost.agents.card import AgentCard
from benchmark.libs.agenthost.agents.request import AgentRequest, ChatMessage, ResponseStatus
from benchmark.evaluators.base import (
    BaseEvaluatorAgent,
    EvaluatorRequest,
    EvaluatorResult,
    decode_eval_text,
    encode_eval_json,
)
from benchmark.utils.exceptions import EvaluatorError


class _DummyEvaluator(BaseEvaluatorAgent):
    default_description = "dummy evaluator"

    def __init__(
        self,
        *,
        name: str,
        result: EvaluatorResult | None = None,
        error: EvaluatorError | None = None,
    ) -> None:
        super().__init__(name=name)
        self._result = result
        self._error = error
        self.seen: list[EvaluatorRequest] = []

    @classmethod
    def from_config(cls, cfg: object, ctx: object) -> "_DummyEvaluator":
        return cls(name=getattr(ctx, "name", "dummy"))

    async def evaluate(self, request: EvaluatorRequest) -> EvaluatorResult:
        self.seen.append(request)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _agent_request(req: EvaluatorRequest) -> AgentRequest:
    return AgentRequest(
        traceid="rid",
        session_id="s",
        messages=[ChatMessage(role="user", content=encode_eval_json(req))],
    )


def test_result_defaults() -> None:
    r = EvaluatorResult(score=7.0)
    assert r.score == 7.0
    assert r.reason is None
    assert r.extra == {}
    assert r.sub_scores is None  # added


@pytest.mark.asyncio
async def test_call_unwraps_request_and_projects_result() -> None:
    agent = _DummyEvaluator(
        name="example",
        result=EvaluatorResult(score=8.0, reason="good", extra={"used_tool": ["x"]}),
    )
    req = EvaluatorRequest(input="你是谁")

    resp = await agent.call(_agent_request(req))

    assert resp.status == ResponseStatus.SUCCESS
    result = EvaluatorResult.model_validate_json(decode_eval_text(resp.content_parts))
    assert result.score == 8.0
    assert result.reason == "good"
    assert result.extra == {"used_tool": ["x"]}
    # the business saw the request deserialized from the message content
    assert agent.seen[0].input == "你是谁"


@pytest.mark.asyncio
async def test_call_maps_evaluator_error_to_failed() -> None:
    agent = _DummyEvaluator(
        name="example",
        error=EvaluatorError(
            code=400, error_type="evaluator_prompt_not_found", message="no prompt"
        ),
    )
    req = EvaluatorRequest()
    resp = await agent.call(_agent_request(req))
    assert resp.status == ResponseStatus.FAILED
    err = EvaluatorError.from_payload(decode_eval_text(resp.content_parts))
    assert err.error_type == "evaluator_prompt_not_found"
    assert err.code == 400
    assert err.message == "no prompt"


@pytest.mark.asyncio
async def test_run_delegates_to_call() -> None:
    agent = _DummyEvaluator(name="example", result=EvaluatorResult(score=5.0, reason="r"))
    resp = await agent.run(_agent_request(EvaluatorRequest()))
    assert resp.status == ResponseStatus.SUCCESS
    result = EvaluatorResult.model_validate_json(decode_eval_text(resp.content_parts))
    assert result.score == 5.0


def test_get_agent_card_uses_name_and_description() -> None:
    agent = _DummyEvaluator(name="example", result=EvaluatorResult(score=1.0))
    card = agent.get_agent_card()
    assert isinstance(card, AgentCard)
    assert card.name == "example"
    assert card.description == "dummy evaluator"


def test_from_config_builds_instance() -> None:
    agent = _DummyEvaluator.from_config(cfg=None, ctx=SimpleNamespace(name="example"))
    assert isinstance(agent, _DummyEvaluator)
    assert agent.name == "example"


def test_result_sub_scores_roundtrip_via_model_validate_json() -> None:
    """The sub_scores field round-trips through JSON serialization and deserialization."""
    r = EvaluatorResult(
        score=8.0,
        reason="good",
        sub_scores={"准确度": 8.0, "相关性": 9.0, "完整性": 7.0},
    )
    payload = r.model_dump_json()
    r2 = EvaluatorResult.model_validate_json(payload)
    assert r2.sub_scores == {"准确度": 8.0, "相关性": 9.0, "完整性": 7.0}
    assert r2.score == 8.0


def test_result_sub_scores_serialized_as_null_when_none() -> None:
    """Pydantic v2 default serializes None as ``null``.

    The legacy ``omit key`` contract is enforced at the ``judge_json`` persistence point,
    not via this serialization path.
    """
    r = EvaluatorResult(score=7.0)
    payload = r.model_dump_json()
    assert '"sub_scores":null' in payload
    # After deserialization sub_scores is still None (the default-value contract)
    r2 = EvaluatorResult.model_validate_json(payload)
    assert r2.sub_scores is None
