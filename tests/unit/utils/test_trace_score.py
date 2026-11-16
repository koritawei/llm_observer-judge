"""Unit tests for attribution score writeback — override on judge_false_positive."""

from dataclasses import dataclass, field

import pytest
from benchmark.records.enums import TriageLevel
from benchmark.attribution.badcase.schemas import AttributionResult
from benchmark.utils.trace_score import (
    AttributionScoreWrite,
    JudgeScoreWrite,
    _resolve_score,
    write_attribution_score,
    write_judge_score,
)


class TestResolveScore:
    def test_keeps_original_when_no_correction(self) -> None:
        metadata: dict = {}
        value = _resolve_score(AttributionResult(), 0.0, metadata)

        assert value == 0.0
        assert "original_judge_score" not in metadata

    def test_overrides_to_good_on_judge_false_positive(self) -> None:
        # Attribution contradicts the evaluation as a misjudgment → score is rewritten to 2, and the original score is kept on metadata.
        metadata: dict = {}
        value = _resolve_score(AttributionResult(corrected_triage_level="good"), 0.0, metadata)

        assert value == 2.0
        assert metadata["original_judge_score"] == 0.0
        assert metadata["score_overridden_by_attribution"] is True

    def test_overrides_to_watch(self) -> None:
        metadata: dict = {}
        value = _resolve_score(AttributionResult(corrected_triage_level="watch"), 0.0, metadata)

        assert value == 1.0

    def test_unknown_level_keeps_original(self) -> None:
        # An unknown grade is not rewritten, so score is not written to a wrong value.
        metadata: dict = {}
        value = _resolve_score(AttributionResult(corrected_triage_level="weird"), 1.0, metadata)

        assert value == 1.0
        assert "original_judge_score" not in metadata

    def test_missing_original_score_falls_back_to_zero(self) -> None:
        metadata: dict = {}
        assert _resolve_score(AttributionResult(), None, metadata) == 0.0


@dataclass
class _Result:
    score: float = 1.0
    reason: str = "partial"
    sub_scores: dict[str, float] | None = None
    extra: object = field(default_factory=dict)


def _envelope(payload: object) -> dict:
    return {"schema_version": "agent_eval_extra.v1", "business_extension": {"payload": payload}}


class _Client:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def create_score(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class TestRubricDetails:
    @pytest.mark.asyncio
    async def test_write_judge_score_adds_existing_rubric_details(self) -> None:
        dimensions = {
            "C1_intent": {"score": 2, "reasoning": "表现正常"},
            "C7_output": {"score": 1, "reasoning": "落地页应直达详情"},
        }
        payload = {"evaluation_detail": {"dimensions": dimensions}}
        client = _Client()

        await write_judge_score(
            JudgeScoreWrite(
                trace_id="trace-1",
                evaluator_name="vui_agent",
                result=_Result(extra=_envelope(payload)),
                client=client,  # type: ignore[arg-type]
            )
        )

        assert client.kwargs["value"] == 1.0
        assert client.kwargs["comment"] == "partial"
        assert client.kwargs["metadata"] == {
            "stage": "judge",
            "rubricDetails": dimensions,
        }

    @pytest.mark.asyncio
    async def test_write_judge_score_keeps_metadata_shape_without_dimensions(self) -> None:
        client = _Client()

        await write_judge_score(
            JudgeScoreWrite(
                trace_id="trace-1",
                evaluator_name="vehicle_agent",
                result=_Result(extra=_envelope({"evaluation_detail": {}})),
                client=client,  # type: ignore[arg-type]
            )
        )

        assert client.kwargs["metadata"] == {"stage": "judge"}


@pytest.mark.parametrize("level", list(TriageLevel))
async def test_write_attribution_score_uses_real_model_levels(level: TriageLevel) -> None:
    client = _Client()
    attribution = AttributionResult(root_cause="checked", corrected_triage_level=level)
    original = 0.25
    expected = {
        "good": 2.0,
        "suspicious_good": 2.0,
        "watch": 1.0,
        "mild_bad": 1.0,
        "severe_bad": 0.0,
    }
    await write_attribution_score(
        AttributionScoreWrite(
            trace_id="trace-1",
            evaluator_name="vehicle_agent",
            attribution=attribution,
            original_score=original,
            client=client,
        )
    )
    assert client.kwargs["value"] == expected.get(level.value, original)
    if level.value in expected:
        assert client.kwargs["metadata"]["original_judge_score"] == original
        assert client.kwargs["metadata"]["score_overridden_by_attribution"] is True
    else:
        assert "score_overridden_by_attribution" not in client.kwargs["metadata"]


@pytest.mark.parametrize("invalid_level", ["partial", "bad"])
def test_real_model_rejects_legacy_correction_words(invalid_level: str) -> None:
    attribution = AttributionResult(corrected_triage_level=invalid_level)
    assert attribution.corrected_triage_level is None
    metadata = {}
    assert _resolve_score(attribution, 0.25, metadata) == 0.25
    assert "score_overridden_by_attribution" not in metadata
