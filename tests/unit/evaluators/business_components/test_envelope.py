"""Contract tests for the unified evaluation envelope and business extensions."""

import pytest
from benchmark.pipeline.envelope import (
    AgentEvalEnvelope,
    BusinessExtension,
    DiagnosticFinding,
    FailureRelation,
    MetricObservation,
)
from benchmark.pipeline.extensions import (
    VUI_EXTENSION_TYPE,
    VUI_EXTENSION_VERSION,
    VuiEvalExtensionV2,
)
from pydantic import ValidationError


def test_envelope_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValidationError):
        AgentEvalEnvelope(business_id="vui_agent", schema_version="vui_eval_extra.v1")


def test_envelope_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        AgentEvalEnvelope(business_id="vui_agent", route_plan={})


def test_envelope_accepts_independent_observations() -> None:
    envelope = AgentEvalEnvelope(
        business_id="vui_agent",
        metric_observations=[
            MetricObservation(
                metric_key="first_markdown_latency_ms",
                value=820,
                unit="ms",
                status="within_target",
                collector_name="vui_latency",
            )
        ],
        diagnostic_findings=[
            DiagnosticFinding(
                finding_key="llm_timeout_recovered",
                severity="warning",
                status="recovered",
                source_name="vui_completion",
            )
        ],
        failure_relations=[
            FailureRelation(
                failure_mode="cmd_shape_missing",
                responsibility="model",
                impact="blocking",
            )
        ],
    )

    assert envelope.metric_observations[0].status == "within_target"
    assert envelope.diagnostic_findings[0].status == "recovered"
    assert envelope.failure_relations[0].impact == "blocking"


def test_vui_extension_type_and_version() -> None:
    extension = VuiEvalExtensionV2(
        judge_ran=True, user_query="附近有什么好吃的"
    ).to_business_extension()

    assert isinstance(extension, BusinessExtension)
    assert extension.extension_type == VUI_EXTENSION_TYPE
    assert extension.extension_version == VUI_EXTENSION_VERSION
    assert extension.payload["judge_ran"] is True
