"""BaseTaskMessage abstract-contract tests plus a round trip of a concrete subclass.

After the framework dropped the default protocol, :class:`BaseTaskMessage` has no business fields, and ``evaluator_name`` /
``to_evaluator_request`` are abstract contracts. Only ``decode_error`` / ``rejected`` plumbing is built in.
Normalization belongs to each source's concrete schema (see the luoshu cases in test_protocol). Here the generic
:class:`~tests.unit.mq_consumer.stubs.StubTaskMessage` covers the contract and the JsonCodec round trip.
"""

import pytest
from benchmark.libs.mq import JsonCodec

from benchmark.ingest.mq.protocol import BaseTaskMessage
from tests.unit.mq_consumer.stubs import StubTaskMessage


def test_base_is_abstract_contract() -> None:
    # The base class has no business fields; evaluator_name is a shared field (default empty string), and to_evaluator_request is not implemented.
    base = BaseTaskMessage()
    assert base.evaluator_name == ""
    with pytest.raises(NotImplementedError):
        base.to_evaluator_request()
    # source defaults to None (= do not enrich); decode_error defaults to None (= a normal message).
    assert base.source is None
    assert base.decode_error is None
    # The evaluability gate allows by default (a source that does not override it keeps its behavior).
    assert base.unevaluatable_reason() is None


def test_rejected_carries_decode_error() -> None:
    msg = BaseTaskMessage.rejected("非 JSON 结构")
    assert msg.decode_error == "非 JSON 结构"


def test_stub_to_evaluator_request() -> None:
    msg = StubTaskMessage(
        evaluator_name="judge_example",
        input={"query": "x"},
        output={"answer": "y"},
        metadata={"scene": "voice"},
        trace_id="t-1",
        agent_name="vehicle_agent",
        project_id="demo",
    )
    assert msg.evaluator_name == "judge_example"
    req = msg.to_evaluator_request()
    assert req.trace_id == "t-1"
    assert req.agent_name == "vehicle_agent"
    assert req.project_id == "demo"
    assert req.input == {"query": "x"}
    assert req.output == {"answer": "y"}
    # trace_id is also kept on metadata (consistent with the HTTP entry).
    assert req.metadata["trace_id"] == "t-1"
    assert req.metadata["scene"] == "voice"


def test_stub_source_from_metadata() -> None:
    assert StubTaskMessage(metadata={"source": "stub"}).source == "stub"
    assert StubTaskMessage(metadata={}).source is None


def test_jsoncodec_round_trip() -> None:
    codec = JsonCodec()
    original = StubTaskMessage(
        evaluator_name="judge_example",
        input={"query": "x"},
        output="plain text",
        trace_id="t-1",
        project_id="p",
        agent_name="a",
    )
    data = codec.encode(original)
    assert isinstance(data, bytes)
    decoded = codec.decode(data, StubTaskMessage)
    assert decoded == original
