"""Evaluator identity mapping for Cluster Report.

The ``evaluatorName`` of a single report trigger is the only source of business identity.
All business metadata written to Benchmark and shown downstream is derived from this
restricted mapping; a separate runtime config must not override it.
"""

from dataclasses import dataclass

VehicleAgentRubric = None
VuiAgentRubric = None
from benchmark.runs.common.rubric_base import BaseRubric


@dataclass(frozen=True)
class EvaluatorReportProfile:
    """Downstream identity and scoring contract for one evaluator in Cluster Report."""

    evaluator_name: str
    service_name: str
    benchmark_name: str
    dataset_name: str
    source: str
    rubric: type[BaseRubric]
    rubric_ref: str


_PROFILES: dict[str, EvaluatorReportProfile] = {}


def resolve_evaluator_report_profile(evaluator_name: str) -> EvaluatorReportProfile:
    """Return the evaluator's downstream identity; unknown values must fail explicitly so the wrong business domain is not written silently."""

    try:
        return _PROFILES[evaluator_name]
    except KeyError as exc:
        raise ValueError(f"unsupported cluster report evaluator: {evaluator_name}") from exc


__all__ = ["EvaluatorReportProfile", "resolve_evaluator_report_profile"]
