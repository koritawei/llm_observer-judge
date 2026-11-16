"""ExampleParseInterceptor: payload normalization plus promotion of the trace / project identity."""

from benchmark.attribution.badcase.attribution_trigger_policy import (
    EXAMPLE_TRIGGER_MAX_EXCLUSIVE,
    RubricV2TriggerPolicy,
    resolve_trigger_policy,
)
from benchmark.runs.common.parse_interceptor import (
    DataParseContext,
    validate_business_item,
)
from benchmark.runs.example.parse import ExampleParseInterceptor
from benchmark.runs.example.rubric import EXAMPLE_PASS_THRESHOLD, ExampleRubric


def _raw(row: dict[str, object], **extra: object) -> dict:
    item = {
        "id": extra.get("id", "case-1"),
        "item_id": "run:0",
        "source_trace_id": extra.get("source_trace_id"),
        "metadata": {"csv_row": dict(row), "title_columns": list(row)},
    }
    return item


def test_rubric_mounts_example_parse_interceptor() -> None:
    assert ExampleRubric.parse_interceptor_ref == (
        "benchmark.runs.example.parse:ExampleParseInterceptor"
    )


def test_rubric_is_holistic_with_online_sub_score_axes() -> None:
    """One criterion plus three-axis stat_dimensions, aligned with production judge_example.sub_score_axes."""
    criteria = ExampleRubric.criteria()
    assert len(criteria) == 1
    assert criteria[0].id == "example"
    assert criteria[0].evaluator_ref == "example"
    assert criteria[0].weight == 1.0
    assert [d.key for d in ExampleRubric.stat_dimensions] == [
        "准确度",
        "相关性",
        "完整性",
    ]
    assert ExampleRubric.judge_execution_policy == "local"
    assert ExampleRubric.pass_threshold == EXAMPLE_PASS_THRESHOLD == 3.0
    assert ExampleRubric.default_aggregation().pass_threshold == 3.0
    assert EXAMPLE_PASS_THRESHOLD == EXAMPLE_TRIGGER_MAX_EXCLUSIVE == 3.0
    policy = resolve_trigger_policy("example_benchmark")
    assert isinstance(policy, RubricV2TriggerPolicy)
    assert policy.max_exclusive == EXAMPLE_PASS_THRESHOLD


def test_parse_normalizes_plain_text_payload_and_identity() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[
                _raw(
                    {
                        "input": "附近咖啡",
                        "output": "CHAT芝田",
                        "expected_output": "特色咖啡",
                        "trace_id": "trace-1",
                        "project": "proj-1",
                    }
                )
            ],
        )
    )[0]

    assert validate_business_item(item) is None
    assert item["input"] == {"request_body": {"query": "附近咖啡"}}
    assert item["output"] == {
        "component": {"type": "markdown", "text": "CHAT芝田"},
        "response_text": "CHAT芝田",
    }
    assert item["expected_output"] == "特色咖啡"
    assert item["source_trace_id"] == "trace-1"
    assert item["metadata"]["project_id"] == "proj-1"
    assert item["metadata"]["csv_row"]["trace_id"] == "trace-1"


def test_parse_decodes_and_normalizes_recommended_json_shapes() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[
                _raw(
                    {
                        "input": ('{"request_body":{"query":"这周边到处乱碰，看碰得出去不？"}}'),
                        "output": (
                            '{"component":{"type":"markdown","text":"先减速再观察"},'
                            '"response_text":"先减速再观察"}'
                        ),
                        "expected_output": '{"reference":"应提醒减速"}',
                    }
                )
            ],
        )
    )[0]

    assert item["input"] == {"request_body": {"query": "这周边到处乱碰，看碰得出去不？"}}
    assert item["output"] == {
        "component": {"type": "markdown", "text": "先减速再观察"},
        "response_text": "先减速再观察",
    }
    assert item["expected_output"] == {"reference": "应提醒减速"}


def test_parse_fills_response_text_from_component_only_json() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[
                _raw(
                    {
                        "input": '{"request_body":{"query":"q"}}',
                        "output": '{"component":{"text":"仅 component"}}',
                    }
                )
            ],
        )
    )[0]
    assert item["output"]["component"]["text"] == "仅 component"
    assert item["output"]["component"]["type"] == "markdown"
    assert item["output"]["response_text"] == "仅 component"


def test_parse_fills_component_from_response_text_only() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[
                _raw(
                    {
                        "input": '{"request_body":{"query":"q"}}',
                        "output": '{"response_text":"仅 response_text"}',
                    }
                )
            ],
        )
    )[0]
    assert item["output"]["response_text"] == "仅 response_text"
    assert item["output"]["component"] == {
        "type": "markdown",
        "text": "仅 response_text",
    }


def test_parse_keeps_dict_payload_columns() -> None:
    raw = _raw({"trace_id": "t-1"})
    raw["metadata"]["csv_row"]["input"] = {"request_body": {"query": "附近咖啡"}}
    raw["metadata"]["csv_row"]["output"] = {
        "component": {"type": "markdown", "text": "CHAT"},
        "response_text": "CHAT",
    }
    item = ExampleParseInterceptor().parse(
        DataParseContext(benchmark_name="example_benchmark", raw_items=[raw])
    )[0]
    assert item["input"] == {"request_body": {"query": "附近咖啡"}}
    assert item["output"]["response_text"] == "CHAT"


def test_parse_lifts_query_alias_when_input_missing() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[_raw({"query": "别名问题", "answer": "别名回答"})],
        )
    )[0]
    assert item["input"] == {"request_body": {"query": "别名问题"}}
    assert item["output"]["response_text"] == "别名回答"


def test_parse_accepts_traceid_alias() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[_raw({"input": "q", "output": "a", "traceid": "tr-alias"})],
        )
    )[0]
    assert item["source_trace_id"] == "tr-alias"


def test_parse_prefers_project_column_over_project_id() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[
                _raw(
                    {
                        "input": "q",
                        "output": "a",
                        "project": "from-project",
                        "project_id": "from-id",
                    }
                )
            ],
        )
    )[0]
    assert item["metadata"]["project_id"] == "from-project"


def test_parse_uses_project_id_column_when_project_absent() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[_raw({"input": "q", "output": "a", "project_id": "from-id"})],
        )
    )[0]
    assert item["metadata"]["project_id"] == "from-id"


def test_parse_keeps_existing_source_trace_id() -> None:
    item = ExampleParseInterceptor().parse(
        DataParseContext(
            benchmark_name="example_benchmark",
            raw_items=[
                _raw(
                    {"input": "q", "output": "a", "trace_id": "from-csv"},
                    source_trace_id="already-set",
                )
            ],
        )
    )[0]
    assert item["source_trace_id"] == "already-set"


def test_parse_normalizes_top_level_without_csv_row() -> None:
    raw = {"id": "case-1", "item_id": "run:0", "input": "q", "output": "a", "metadata": {}}
    item = ExampleParseInterceptor().parse(
        DataParseContext(benchmark_name="example_benchmark", raw_items=[raw])
    )[0]
    assert item["input"] == {"request_body": {"query": "q"}}
    assert item["output"]["response_text"] == "a"
    assert item["metadata"] == {}
