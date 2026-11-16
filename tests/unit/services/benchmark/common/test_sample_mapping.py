"""Tests for the generic mapping from a dataset sample to EvaluatorRequest."""

from types import SimpleNamespace

from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY, PipelineSource
from benchmark.runs.common.sample_mapping import RunSample, build_evaluator_request


def _services() -> SimpleNamespace:
    metadata = SimpleNamespace(
        service_name="example",
        name="example_benchmark",
        dataset=SimpleNamespace(project_id="default-project"),
    )
    return SimpleNamespace(get_metadata=lambda: metadata)


def test_inline_sample_preserves_output_and_ground_truth() -> None:
    sample = SimpleNamespace(
        id="item-1",
        input="query",
        output="answer",
        expected_output="ground-truth",
        source_trace_id=None,
    )
    request = SimpleNamespace(project_id=None)

    result = build_evaluator_request(
        _services(),
        RunSample(sample=sample, run_id="run-1"),
        request,
        extra_metadata={"scene": "inline"},
    )

    assert result.output == "answer"
    assert result.expected_output == "ground-truth"
    assert result.trace_id == "dataset:item-1"
    assert result.project_id == "default-project"
    assert result.metadata[PIPELINE_SOURCE_METADATA_KEY] == PipelineSource.BENCHMARK.value
    assert result.metadata["benchmark_run_id"] == "run-1"
    assert result.metadata["scene"] == "inline"


def test_trace_sample_maps_expected_output_only_to_output() -> None:
    sample = SimpleNamespace(
        id="item-2",
        input="query",
        output=None,
        expected_output="ground-truth",
        source_trace_id="trace-2",
    )
    request = SimpleNamespace(project_id="request-project")

    result = build_evaluator_request(
        _services(),
        RunSample(sample=sample, run_id="run-2"),
        request,
    )

    assert result.output == "ground-truth"
    assert result.expected_output is None
    assert result.trace_id == "trace-2"
    assert result.project_id == "request-project"
