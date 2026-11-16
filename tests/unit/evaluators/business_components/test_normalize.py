"""Compatible read-view tests for the unified envelope."""

import json

from benchmark.pipeline.envelope import AgentEvalEnvelope
from benchmark.pipeline.extensions import (
    VehicleEvalExtensionV3,
    VuiEvalExtensionV2,
)
from benchmark.pipeline.normalize import (
    normalize_eval_extra,
    normalize_vehicle_extra,
    normalize_vui_extra,
)
from benchmark.pipeline.signals import EvaluationSignal


def test_legacy_payload_passes_through_unchanged() -> None:
    legacy = {"schema_version": "vehicle_eval_extra.v3", "score": 1.5}

    assert normalize_vehicle_extra(legacy) == legacy
    assert normalize_eval_extra(legacy) == legacy


def test_vehicle_envelope_restores_compatibility_view() -> None:
    extension = VehicleEvalExtensionV3(
        score=1.5, reason="ok", checker_results=[{"source_name": "shape"}]
    ).to_business_extension()
    envelope = AgentEvalEnvelope(
        business_id="vehicle_agent",
        primary_score=1.5,
        summary_reason="ok",
        business_extension=extension,
    )

    view = normalize_vehicle_extra(envelope.model_dump(mode="json"))

    assert view["score"] == 1.5
    assert view["reason"] == "ok"
    assert view["checker_results"] == [{"source_name": "shape"}]


def test_vui_envelope_restores_json_snapshot() -> None:
    extension = VuiEvalExtensionV2(
        judge_ran=True, output_snapshot={"text": "好的"}
    ).to_business_extension()
    envelope = AgentEvalEnvelope(business_id="vui_agent", business_extension=extension)

    view = normalize_vui_extra(envelope.model_dump(mode="json"))

    assert json.loads(view["output_snapshot"]) == {"text": "好的"}
    assert normalize_eval_extra(envelope.model_dump(mode="json")) == view


def test_vehicle_envelope_respects_explicit_empty_list() -> None:
    """An explicit empty list in the payload must not fall through to the evaluation_signals fallback."""

    extension = VehicleEvalExtensionV3(
        score=1.0, reason="ok", checker_results=[], risk_flags=[]
    ).to_business_extension()
    envelope = AgentEvalEnvelope(
        business_id="vehicle_agent",
        primary_score=1.0,
        evaluation_signals=[
            EvaluationSignal(
                source_type="checker",
                source_name="fallback",
                risk_flags=["should_not_appear"],
            )
        ],
        business_extension=extension,
    )

    view = normalize_vehicle_extra(envelope.model_dump(mode="json"))

    assert view["checker_results"] == []
    assert view["risk_flags"] == []
