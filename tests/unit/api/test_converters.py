"""Unit tests for the OpenAI ↔ domain conversions (in benchmark.routes.schema).

The message content carries a JSON :class:`EvalEnvelope`
(``input`` / ``output`` / ``metadata`` + top-level ``agent_name`` /
``project_id``). The top-level shape matches the real Langfuse payload;
see :data:`tests.fixtures.real_trace_envelope` for the golden fixture.
"""
# ruff: noqa: PLR2004  (test-only: literal asserts)

import json
from pathlib import Path

import pytest
from benchmark.routes.schema import ChatCompletionRequest, to_chat_completion, to_evaluator_request
from benchmark.evaluators.base import EvaluatorResult

_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "real_trace_envelope.json"


def _request(
    *,
    model: str = "example",
    trace_id: str = "t-1",
    agent_name: str = "vehicle-agent",
    project_id: str = "p-9",
) -> ChatCompletionRequest:
    envelope = {
        "input": {"request_body": {"query": "你是谁"}},
        "output": {"component": {"text": "我是助手"}},
        "metadata": {"trace_id": trace_id},
        "agent_name": agent_name,
        "project_id": project_id,
    }
    return ChatCompletionRequest(
        model=model, messages=[{"role": "user", "content": json.dumps(envelope)}]
    )


def test_to_evaluator_request_copies_prompt_texts() -> None:
    envelope = {
        "input": {"request_body": {"query": "你是谁"}},
        "output": {"component": {"text": "我是助手"}},
        "metadata": {"trace_id": "t-1"},
        "judge_prompt_text": "JUDGE_CUSTOM",
        "attribution_prompt_text": "ATTR_CUSTOM",
        "tool_check": {"required": ["get_order"], "must_be_empty": False},
    }
    payload = ChatCompletionRequest(
        model="example",
        messages=[{"role": "user", "content": json.dumps(envelope)}],
    )

    req = to_evaluator_request(payload)

    assert req.judge_prompt_text == "JUDGE_CUSTOM"
    assert req.attribution_prompt_text == "ATTR_CUSTOM"
    assert req.tool_check == {"required": ["get_order"], "must_be_empty": False}


def test_to_evaluator_request_parses_stringified_tool_check() -> None:
    envelope = {
        "input": {"request_body": {"query": "你是谁"}},
        "output": {"component": {"text": "我是助手"}},
        "metadata": {"trace_id": "t-1"},
        "tool_check": json.dumps({"forbidden": ["get_order"]}),
    }
    payload = ChatCompletionRequest(
        model="example",
        messages=[{"role": "user", "content": json.dumps(envelope)}],
    )

    req = to_evaluator_request(payload)

    assert req.tool_check == {"forbidden": ["get_order"]}


def test_to_evaluator_request_parses_all_fields() -> None:
    req = to_evaluator_request(_request(trace_id="t-1"))
    assert req.input == {"request_body": {"query": "你是谁"}}
    assert req.output == {"component": {"text": "我是助手"}}
    # to_evaluator_request always stamps pipeline_source=http (the basis for trace_meta filtering)
    assert req.metadata == {"trace_id": "t-1", "pipeline_source": "http"}
    assert req.trace_id == "t-1"
    assert req.agent_name == "vehicle-agent"
    assert req.project_id == "p-9"


def test_to_evaluator_request_accepts_plain_text_output() -> None:
    # output format is not fixed — a plain string is carried as a JSON string.
    envelope = {"input": {"request_body": {"query": "q"}}, "output": "纯文本回答"}
    payload = ChatCompletionRequest(
        model="x", messages=[{"role": "user", "content": json.dumps(envelope)}]
    )
    req = to_evaluator_request(payload)
    assert req.output == "纯文本回答"


def test_to_evaluator_request_coerces_trace_stringified_fields() -> None:
    """Langfuse LLM-as-a-Judge injects variables as text: input/output/metadata get serialized into
    (escaped) JSON strings stuffed into the envelope (rather than objects). The evaluator wants objects — this case locks the restore behavior,
    otherwise a metadata validation failure degrades to the proxy fallback (scoring becomes free text and Langfuse records a failure).
    """
    envelope = {
        "input": json.dumps({"request_body": {"query": "浪花屿洗浴的整体评价怎么样"}}),
        "output": json.dumps({"component": {"type": "markdown", "text": "评分 4.5"}}),
        "metadata": json.dumps(
            {
                "trace_id": "21c860b217848059764594210d0995",
                "biz_trace_id": "ee04e010e81a20de3e80266d96b010c2",
                "graph_key": "ai_chat.standard",
            }
        ),
        "agent_name": "memory_pairwise",
        "project_id": "cmpwln0qs04fo6j8eryfdgznb",
    }
    payload = ChatCompletionRequest(
        model="memory_pairwise",
        messages=[{"role": "user", "content": json.dumps(envelope)}],
    )
    req = to_evaluator_request(payload)
    # Stringified objects are restored into structures
    assert req.input == {"request_body": {"query": "浪花屿洗浴的整体评价怎么样"}}
    assert req.output == {"component": {"type": "markdown", "text": "评分 4.5"}}
    # After metadata is restored, trace_id can be extracted, and extra fields (biz_trace_id / graph_key) are kept
    assert req.trace_id == "21c860b217848059764594210d0995"
    assert req.metadata["biz_trace_id"] == "ee04e010e81a20de3e80266d96b010c2"
    assert req.metadata["graph_key"] == "ai_chat.standard"
    assert req.agent_name == "memory_pairwise"
    assert req.project_id == "cmpwln0qs04fo6j8eryfdgznb"


def test_has_eval_envelope_true_for_stringified_fields() -> None:
    """After stringified fields are restored, the envelope must be recognized as a real evaluation envelope (not the proxy fallback)."""
    from benchmark.routes.schema import has_eval_envelope

    envelope = {
        "input": json.dumps({"request_body": {"query": "q"}}),
        "output": json.dumps({"component": {"text": "a"}}),
        "metadata": json.dumps({"trace_id": "t-1"}),
        "agent_name": "memory_pairwise",
        "project_id": "p-9",
    }
    payload = ChatCompletionRequest(
        model="memory_pairwise",
        messages=[{"role": "user", "content": json.dumps(envelope)}],
    )
    assert has_eval_envelope(payload) is True


@pytest.mark.parametrize("model", ["vehicle_agent", "vui_agent"])
@pytest.mark.parametrize("io", [{}, {"input": None, "output": None}])
def test_trace_identity_routes_to_evaluation(model: str, io: dict) -> None:
    from benchmark.routes.schema import has_eval_envelope, validate_evaluator_request

    envelope = {
        **io,
        "metadata": {"trace_id": "langfuse-primary-id"},
        "project_id": "project-key",
        "agent_name": model.replace("_", "-"),
    }
    payload = ChatCompletionRequest(
        model=model,
        skip_trace_resolve=True,
        messages=[{"role": "user", "content": json.dumps(envelope)}],
    )
    assert has_eval_envelope(payload)
    request = to_evaluator_request(payload)
    assert request.skip_trace_resolve is True
    assert request.input is None and request.output is None
    assert request.trace_id == "langfuse-primary-id"
    assert validate_evaluator_request(request) == []


def test_trace_identity_missing_project_routes_to_metadata_validation() -> None:
    from benchmark.routes.schema import has_eval_envelope, validate_evaluator_request

    payload = ChatCompletionRequest(
        model="vehicle_agent",
        messages=[{"role": "user", "content": '{"metadata":{"trace_id":"t-1"}}'}],
    )
    assert has_eval_envelope(payload)
    assert validate_evaluator_request(to_evaluator_request(payload)) == ["project_id", "agent_name"]


@pytest.mark.parametrize(
    "content", ["hello", "{}", '{"input":null,"output":null}', '{"metadata":{}}']
)
def test_non_eval_content_keeps_proxy_fallback(content: str) -> None:
    from benchmark.routes.schema import has_eval_envelope

    payload = ChatCompletionRequest(
        model="example", messages=[{"role": "user", "content": content}]
    )
    assert not has_eval_envelope(payload)


def test_to_evaluator_request_tolerates_non_json_content() -> None:
    payload = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "not json"}])
    req = to_evaluator_request(payload)
    assert req.input is None
    assert req.output is None
    assert req.metadata == {"pipeline_source": "http"}
    assert req.trace_id is None
    assert req.agent_name is None
    assert req.project_id is None


def test_to_evaluator_request_parses_real_trace_fixture() -> None:
    """Golden fixture: real Langfuse Numeric payload (vehicle-agent, score=0 attribution case).

    Regression guard for the ``extra.{agent_name,project_id}`` → top-level rename.
    Before the rename, this fixture parsed with ``agent_name=None`` and
    ``project_id=None``, which silently routed the verdict past ``after_eval``
    (no DB row, no triage, no warning).
    """
    envelope = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    payload = ChatCompletionRequest(
        model="vehicle-agent",
        messages=[{"role": "user", "content": json.dumps(envelope)}],
    )
    req = to_evaluator_request(payload)
    assert req.agent_name == "vehicle-agent"
    assert req.project_id == "cmpwlnaot04it5ycqvm1lqnc4"
    assert req.trace_id == "1f6f566b7dea475db21b871ffce7f40a"
    assert req.metadata["workflow_name"] == "vehicle-agent.stream.chat.max"


def test_to_chat_completion_projects_result() -> None:
    body = to_chat_completion("example", EvaluatorResult(score=7.0, reason="ok"))
    assert body.object == "chat.completion"
    assert body.model == "example"
    content = json.loads(body.choices[0].message.content)
    assert content["score"] == 7.0
    assert content["reasoning"] == "ok"


@pytest.mark.parametrize("model", ["vehicle_agent", "vui_agent"])
def test_completion_returns_evaluation_report_trace_id(model: str) -> None:
    result = EvaluatorResult(
        score=7.0,
        reason="ok",
    )
    body = to_chat_completion(
        model, result, trace_id="abcdef0123456789abcdef0123456789"
    ).model_dump()
    assert body["trace_id"] == "abcdef0123456789abcdef0123456789"
    assert json.loads(body["choices"][0]["message"]["content"]) == {"score": 7.0, "reasoning": "ok"}


def test_completion_without_report_returns_null_trace_id() -> None:
    body = to_chat_completion("vui_agent", EvaluatorResult(score=7.0)).model_dump()
    assert body["trace_id"] is None


def test_validate_evaluator_request_accepts_complete() -> None:
    from benchmark.routes.schema import validate_evaluator_request
    from benchmark.evaluators.base import EvaluatorRequest

    assert (
        validate_evaluator_request(
            EvaluatorRequest(
                input="x",
                output="y",
                metadata={},
                trace_id="t-1",
                agent_name="a",
                project_id="p",
            )
        )
        == []
    )


def test_validate_evaluator_request_lists_missing_fields() -> None:
    """Missing fields are returned in _REQUIRED_METADATA_FIELDS order; None, empty, and whitespace all count as missing."""
    from benchmark.routes.schema import validate_evaluator_request
    from benchmark.evaluators.base import EvaluatorRequest

    missing = validate_evaluator_request(
        EvaluatorRequest(
            input="x",
            output="y",
            metadata={},
            trace_id=None,
            agent_name="",
            project_id="   ",
        )
    )
    assert missing == ["trace_id", "project_id", "agent_name"]


def test_validate_evaluator_request_partial_missing() -> None:
    from benchmark.routes.schema import validate_evaluator_request
    from benchmark.evaluators.base import EvaluatorRequest

    missing = validate_evaluator_request(
        EvaluatorRequest(
            input="x",
            output="y",
            metadata={},
            trace_id="t-1",
            agent_name="a",
            project_id=None,
        )
    )
    assert missing == ["project_id"]


@pytest.mark.parametrize("version", [None, "vui_agent_evidence_boundary", "unknown", 3])
def test_request_prompt_version_is_forwarded(version: str | int | None) -> None:
    payload = ChatCompletionRequest(
        model="vehicle_agent",
        prompt_version=version,
        messages=[{"role": "user", "content": '{"input":{},"output":{}}'}],
    )
    assert to_evaluator_request(payload).prompt_version == version
