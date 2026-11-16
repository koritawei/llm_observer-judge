"""Inbound protocol-layer unit tests — the luoshu converter, convert_for_source dispatch, and the tolerant codec.

Routing is **topic → source → converter** (no longer discriminated by ``eventType`` content):
``convert_for_source(source, raw)`` picks the converter by source. An unregistered source, or a required-field check
that fails → reject (``BaseTaskMessage.rejected``, carrying decode_error so the handler can skip). luoshu's concrete
normalization belongs to :class:`LuoshuTaskMessage` (fields model the raw JSON; the evaluation input is projected by ``to_evaluator_request``).
"""

import json

from business.evaluators.example.schema import parse_input, parse_output
from benchmark.ingest.mq.consumers.luoshu.converter import LuoshuConverter
from benchmark.ingest.mq.consumers.luoshu.schema import SOURCE as LUOSHU_SOURCE, LuoshuTaskMessage
from benchmark.ingest.mq.protocol import (
    BaseTaskMessage,
    TolerantEvalCodec,
    convert_for_source,
)

_LUOSHU_RAW = {
    # Messages are distinguished only by topic (the luoshu topic is enough); eventType is only archived/forwarded and does not take part in routing.
    # Provided directly at the top level: project_id selects the Langfuse instance, and trace_id looks up the primary key.
    "eventType": "GUI_CHAT_SAMPLE_RESULT",
    "project_id": "cmpwln0qs04fo6j8eryfdgznb",
    "trace_id": "94db371ce84a5fb290ccdead4dfa412e",
    "app": "baidu",
    "query": "这附近挑个公园逛会儿吧，坐公交怎么过去，现在人多不多",
    "coordinate": "30.5928,114.3055",
    "current_location": "山西省太原市万柏林区滨河西路辅路",
    "generate_time_b": "1789092215409",
    "conversationHistoryMessages": [
        {"role": "user", "content": "这附近挑个公园逛会儿吧", "turn_index": 1},
        {"role": "assistant", "content": "首选江滩沿线，走走就到。", "turn_index": 1},
    ],
    "screenshots": ["https://atap.amap.com/x.jpg"],
}


def test_luoshu_converter_maps_core_fields() -> None:
    msg = LuoshuConverter().convert(_LUOSHU_RAW)
    assert isinstance(msg, LuoshuTaskMessage)
    assert msg.evaluator_name == "competitor_pairwise"
    assert msg.event_type == "GUI_CHAT_SAMPLE_RESULT"
    assert msg.trace_id == "94db371ce84a5fb290ccdead4dfa412e"
    assert msg.project_id == "cmpwln0qs04fo6j8eryfdgznb"
    # Extra context goes into metadata.
    assert msg.metadata["source"] == "luoshu"
    assert msg.metadata["coordinate"] == "30.5928,114.3055"
    assert msg.metadata["generate_time_b"] == "1789092215409"
    assert msg.metadata["screenshots"] == ["https://atap.amap.com/x.jpg"]
    # The evaluation input is projected by to_evaluator_request into a side-by-side comparison structure.
    req = msg.to_evaluator_request()
    assert req.agent_name == "luoshu"  # fixed SOURCE
    assert req.project_id == "cmpwln0qs04fo6j8eryfdgznb"
    # The sample generation time is passed down the existing trace_timestamp channel (no new field).
    assert req.trace_timestamp == "1789092215409"
    # request_body: query is required; location comes from coordinate, and address comes from current_location.
    assert req.input == {
        "request_body": {
            "query": _LUOSHU_RAW["query"],
            "location": "30.5928,114.3055",
            "address": "山西省太原市万柏林区滨河西路辅路",
        }
    }
    # output contains only eval_mode + candidates (a flat {candidate name: raw content}).
    assert req.output["eval_mode"] == "horizontal_compare"
    # The gaode key is always present (a side-by-side comparison always has two candidates); on a Langfuse miss it is an empty string ("").
    assert set(req.output["candidates"]) == {"baidu", "gaode"}
    assert req.output["candidates"]["gaode"] == ""
    # key1 (baidu) = the full MQ payload (as-is, except Langfuse enrichment; assistant is not extracted).
    baidu = req.output["candidates"]["baidu"]
    assert baidu["app"] == "baidu"
    assert baidu["query"] == _LUOSHU_RAW["query"]
    assert baidu["conversationHistoryMessages"] == _LUOSHU_RAW["conversationHistoryMessages"]
    assert baidu["trace_id"] == "94db371ce84a5fb290ccdead4dfa412e"
    # generate_time_b is forwarded into the baidu candidate with the full MQ payload.
    assert baidu["generate_time_b"] == "1789092215409"
    # eventType is forwarded into the baidu candidate with the full MQ payload (by_alias returns it to camelCase).
    assert baidu["eventType"] == "GUI_CHAT_SAMPLE_RESULT"
    assert "langfuse" not in baidu


def test_luoshu_converted_message_parses_in_evaluator() -> None:
    # The projected result goes through the example evaluator's parse_input: query is usable directly; the answer has moved into output.candidates.
    req = LuoshuConverter().convert(_LUOSHU_RAW).to_evaluator_request()
    assert parse_input(req.input).request_body.query.startswith("这附近挑个公园")
    # output no longer has component, so parse_output gets an empty component; the answer is in candidates.
    assert parse_output(req.output).component.text == ""
    assert req.output["candidates"]["baidu"]["conversationHistoryMessages"]


def test_luoshu_adds_gaode_candidate_when_trace_present() -> None:
    # A Langfuse hit (langfuse backfilled) → add the fixed gaode candidate, forming a two-candidate side-by-side comparison.
    msg = LuoshuConverter().convert(_LUOSHU_RAW)
    enriched = msg.model_copy(update={"langfuse": {"primary_id": "pk-1", "spans": []}})
    req = enriched.to_evaluator_request()
    assert set(req.output["candidates"]) == {"baidu", "gaode"}
    # gaode content = the raw Langfuse data (a dict, not processed).
    assert req.output["candidates"]["gaode"] == {"primary_id": "pk-1", "spans": []}
    # baidu is still the full MQ payload, and the enrichment data is not mixed into baidu.
    assert req.output["candidates"]["baidu"]["app"] == "baidu"
    assert "langfuse" not in req.output["candidates"]["baidu"]


def test_luoshu_query_falls_back_to_last_user_message() -> None:
    raw = dict(_LUOSHU_RAW)
    raw.pop("query")
    req = LuoshuConverter().convert(raw).to_evaluator_request()
    assert req.input["request_body"]["query"] == "这附近挑个公园逛会儿吧"


def test_luoshu_is_evaluatable_with_or_without_trace() -> None:
    # After the "store however many were sent" change: an id is enough to let it through to the evaluator (an empty candidate is stored as UNEVALUATED_SCORE).
    msg = LuoshuConverter().convert(_LUOSHU_RAW)
    assert msg.trace_id is not None
    assert msg.project_id is not None
    assert msg.unevaluatable_reason() is None  # Let it through even with no langfuse

    # After enrichment hits, let it through the same way.
    enriched = msg.model_copy(update={"langfuse": {"output": "高德回答"}})
    assert enriched.unevaluatable_reason() is None


def test_convert_for_source_dispatches_luoshu() -> None:
    msg = convert_for_source(LUOSHU_SOURCE, _LUOSHU_RAW)
    assert msg.evaluator_name == "competitor_pairwise"
    assert msg.trace_id == "94db371ce84a5fb290ccdead4dfa412e"


def test_convert_for_source_rejects_unknown_source() -> None:
    # A source with no registered converter (an unknown source was configured, or that source package was not imported) → reject, carrying decode_error so the handler can skip.
    msg = convert_for_source("nope", _LUOSHU_RAW)
    assert msg.decode_error is not None


def test_luoshu_convert_missing_required_fields_raises() -> None:
    # Routing is already fixed as luoshu by topic (not discriminated by any event-type field); the required fields are still validated.
    raw = {k: v for k, v in _LUOSHU_RAW.items() if k not in ("trace_id", "project_id")}
    try:
        LuoshuConverter().convert(raw)
    except ValueError as exc:
        assert "trace_id" in str(exc) and "project_id" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing required fields")


def test_convert_for_source_rejects_missing_required_fields() -> None:
    # A missing required field → convert raises → convert_for_source falls back to rejected (carrying the reason).
    raw = {k: v for k, v in _LUOSHU_RAW.items() if k != "trace_id"}
    msg = convert_for_source(LUOSHU_SOURCE, raw)
    assert msg.decode_error is not None


def test_codec_decodes_luoshu_bytes_through_protocol() -> None:
    codec = TolerantEvalCodec(LUOSHU_SOURCE)
    data = json.dumps(_LUOSHU_RAW).encode("utf-8")
    msg = codec.decode(data, BaseTaskMessage)
    assert isinstance(msg, LuoshuTaskMessage)
    assert msg.decode_error is None
    assert msg.evaluator_name == "competitor_pairwise"
    assert msg.to_evaluator_request().agent_name == "luoshu"


def test_codec_rejects_non_json_bytes() -> None:
    codec = TolerantEvalCodec(LUOSHU_SOURCE)
    msg = codec.decode(b"not-json-at-all", BaseTaskMessage)
    assert msg.decode_error is not None


def test_codec_rejects_non_object_json() -> None:
    codec = TolerantEvalCodec(LUOSHU_SOURCE)
    msg = codec.decode(b"[1, 2, 3]", BaseTaskMessage)
    assert msg.decode_error is not None
