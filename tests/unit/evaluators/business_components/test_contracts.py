"""Cross-business data-contract tests for evaluation."""

from benchmark.pipeline.contracts import EvaluationSnapshot, EvidenceFact
from benchmark.pipeline.signals import EvaluationSignal


def test_evidence_fact_id_is_stable_across_reference_order() -> None:
    first = EvidenceFact(
        extractor_name="facts",
        fact_type="tool_result",
        evidence_refs=["tool:a", "tool:b"],
        neutral_value={"value": 1},
    )
    second = EvidenceFact(
        extractor_name="facts",
        fact_type="tool_result",
        evidence_refs=["tool:b", "tool:a"],
        neutral_value={"value": 1},
    )

    assert first.fact_id == second.fact_id


def test_evaluation_snapshot_keeps_business_dimensions_open() -> None:
    snapshot = EvaluationSnapshot(
        schema_version="eval_snapshot.v1",
        logic_name="custom_agent",
        checker_signals=[
            EvaluationSignal(
                source_type="checker", source_name="custom_checker", dimensions=["CUSTOM_DIMENSION"]
            )
        ],
        hard_rule_hits=["rule-a", "rule-a"],
    )

    assert snapshot.checker_signals[0].dimensions == ["CUSTOM_DIMENSION"]
    assert snapshot.hard_rule_hits == ["rule-a"]
