"""Sidecar observation tree → TRACE-level fields and tool projections."""

# observations → schema runs trace_resolver.__init__ → tools → observations.
from benchmark.trace.tools import TraceToolProvider as _TraceTools  # noqa: F401
from benchmark.ingest.sidecar.observations import (
    extract_io,
    extract_metadata,
    filter_observations,
    find_observation,
    to_observation_detail,
    to_span_skeleton,
)

_TREE = [
    {
        "span_id": "gen-1",
        "parent_span_id": "root",
        "name": "AmapChatModel.chat",
        "type": "GENERATION",
        "level": "DEFAULT",
        "end_time": 300,
        "input": [{"role": "user", "content": "内部 messages"}],
        "output": {"text": "generation 输出"},
        "metadata": {"span": "generation"},
    },
    {
        "span_id": "root",
        "parent_span_id": "",
        "name": "amap-agent.stream.ai_chat.standard",
        "type": "SPAN",
        "level": "DEFAULT",
        "end_time": 200,
        "input": {"query": "附近咖啡"},
        "output": {"text": "去星巴克"},
        "metadata": {"request_id": "r-1", "scene": "navi"},
    },
]


def test_extract_io_prefers_io_root_over_newer_child() -> None:
    inp, out = extract_io(_TREE)
    assert inp == {"query": "附近咖啡"}
    assert out == {"text": "去星巴克"}


def test_extract_metadata_prefers_io_root() -> None:
    assert extract_metadata(_TREE) == {"request_id": "r-1", "scene": "navi"}


def test_extract_io_falls_back_when_root_has_no_io() -> None:
    tree = [
        {"span_id": "root", "parent_span_id": "", "end_time": 400, "name": "root"},
        {
            "span_id": "gen-1",
            "parent_span_id": "root",
            "end_time": 300,
            "input": {"q": 1},
            "output": "a",
        },
    ]
    assert extract_io(tree) == ({"q": 1}, "a")


def test_extract_metadata_empty_when_root_has_no_metadata() -> None:
    tree = [
        {"span_id": "root", "parent_span_id": "", "end_time": 2, "input": {"q": 1}},
        {
            "span_id": "child",
            "parent_span_id": "root",
            "end_time": 1,
            "metadata": {"span": "child"},
        },
    ]
    assert extract_metadata(tree) == {}


def test_tool_projections_share_span_id_and_filters() -> None:
    detail = to_observation_detail(_TREE[1], "t-1")
    assert detail.id == "root"
    assert detail.trace_id == "t-1"
    assert detail.input == {"query": "附近咖啡"}
    skeleton = to_span_skeleton(_TREE[1], 80)
    assert skeleton.id == "root"
    assert skeleton.parent_observation_id is None or skeleton.parent_observation_id == ""
    gens = filter_observations(_TREE, name=None, type="GENERATION", status=None)
    assert [find_observation(gens, "gen-1")] == [gens[0]]
    assert gens[0]["span_id"] == "gen-1"
