"""Generic mapping from a dataset sample to ``EvaluatorRequest`` (duck-typing).

Shared by the example component and dynamic components. Compatible with a Langfuse
``DatasetItem`` (no ``output``; fall back ``expected_output → output``) and with inline /
pydantic model views (``output`` is set explicitly, and ``expected_output`` is mapped
separately as ground truth). Business metadata carries the run / component context.
"""

from dataclasses import dataclass

from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY, PipelineSource
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.runs.schema import BenchmarkRequest
from benchmark.reporting.framework.base import BaseBenchmarkServices


@dataclass(frozen=True)
class RunSample:
    """One dataset-sample reference in an evaluation run (the sample itself + its run_id)."""

    sample: object
    run_id: str


def build_evaluator_request(
    services: BaseBenchmarkServices,
    run_sample: RunSample,
    request: BenchmarkRequest,
    *,
    extra_metadata: dict | None = None,
) -> EvaluatorRequest:
    """Turn a dataset item into an ``EvaluatorRequest`` (component metadata comes from ``services``).

    ``services`` supplies ``get_metadata()`` (service_name / dataset fallback);
    ``request`` may be None (unit tests / direct calls), in which case project_id uses the
    component default. ``extra_metadata`` holds extra business fields (for example ``scene``
    on example) merged into the base metadata; on a key collision the business fields win.
    """
    sample = run_sample.sample
    meta = services.get_metadata()
    item_id = getattr(sample, "id", "") or ""
    trace_id = getattr(sample, "source_trace_id", None) or f"dataset:{item_id}"

    input_value = getattr(sample, "input", None)

    # Prefer output (inline); fall back to expected_output (Langfuse)
    raw_output = getattr(sample, "output", None)
    if raw_output is None:
        raw_output = getattr(sample, "expected_output", None)

    # Map expected_output on its own: fill it as ground truth only when inline sets output explicitly;
    # on the Langfuse path expected_output is already mapped to output, so do not fill it again.
    raw_expected = getattr(sample, "expected_output", None)
    has_explicit_output = getattr(sample, "output", None) is not None

    return EvaluatorRequest(
        input=input_value,
        output=raw_output,
        expected_output=raw_expected if has_explicit_output else None,
        trace_id=trace_id,
        agent_name=meta.service_name,
        project_id=(request.project_id if request is not None else None) or meta.dataset.project_id,
        metadata={
            PIPELINE_SOURCE_METADATA_KEY: PipelineSource.BENCHMARK.value,
            "dataset_item_id": item_id,
            "benchmark_run_id": run_sample.run_id,
            "benchmark_name": meta.name,
            **(extra_metadata or {}),
        },
    )


__all__ = ["RunSample", "build_evaluator_request"]
