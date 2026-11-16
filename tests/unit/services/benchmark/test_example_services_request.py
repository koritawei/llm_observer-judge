"""Unit tests for ExampleBenchmarkServices._build_evaluator_request mapping."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from benchmark.runs.example.services import ExampleBenchmarkServices


def test_build_evaluator_request_maps_expected_output_to_output() -> None:
    """Benchmark path: Langfuse expected_output → EvaluatorRequest.output only."""
    services = object.__new__(ExampleBenchmarkServices)
    services.get_metadata = MagicMock(
        return_value=SimpleNamespace(
            service_name="example",
            name="example_benchmark",
            dataset=SimpleNamespace(project_id="proj-1"),
        )
    )
    sample = SimpleNamespace(
        id="item-42",
        input="where is X?",
        expected_output="X is at Y",
        source_trace_id=None,
        metadata={},
    )

    req = services._build_evaluator_request(sample, run_id="run-1", request=None)

    assert req.output == "X is at Y"
    assert req.expected_output is None
    assert req.input == "where is X?"
    assert req.trace_id == "dataset:item-42"
    assert req.metadata["pipeline_source"] == "benchmark"
    assert req.metadata["dataset_item_id"] == "item-42"
    assert req.metadata["benchmark_run_id"] == "run-1"


@pytest.mark.parametrize(
    "metadata",
    [None, "some-string", [1, 2, 3], 42],
    ids=["none", "str", "list", "int"],
)
def test_extract_react_rounds_non_dict_metadata_returns_none(metadata: object) -> None:
    """metadata that is not a dict (the Langfuse SDK types it as Any) does not blow up, and returns None.

    Regression guard: the react_rounds branch used to call ``metadata.get`` bare — metadata that was neither a dict
    nor None raised AttributeError and blew up criterion stats aggregation.
    """
    services = object.__new__(ExampleBenchmarkServices)

    assert services._extract_react_rounds(SimpleNamespace(metadata=metadata)) is None


def test_extract_react_rounds_reads_turns_and_react_rounds() -> None:
    """Happy path for dict metadata: the turns list wins, then the react_rounds integer."""
    services = object.__new__(ExampleBenchmarkServices)

    turns_item = SimpleNamespace(metadata={"turns": ["a", "b", "c"]})
    rounds_item = SimpleNamespace(metadata={"react_rounds": 4})
    empty_item = SimpleNamespace(metadata={})

    assert services._extract_react_rounds(turns_item) == 3
    assert services._extract_react_rounds(rounds_item) == 4
    assert services._extract_react_rounds(empty_item) is None
