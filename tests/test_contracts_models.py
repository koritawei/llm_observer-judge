"""Contract model validation against Frozen schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jsonschema
import pytest
from jsonschema import Draft202012Validator

from llmobserverjudge.contracts.idempotency import eval_job_key
from llmobserverjudge.contracts.models import (
    EvalConfig,
    EvalJob,
    Priority,
    ScoreResult,
    Span,
)

SCHEMA_DIR = (
    Path(__file__).resolve().parents[1] / "docs" / "sdd" / "04-contracts" / "schemas"
)


def _load(name: str) -> dict:
    import json

    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_span_model_and_schema() -> None:
    span = Span(
        tenant_id="t1",
        trace_id="0123456789abcdef0123456789abcdef",
        span_id="0123456789abcdef",
        name="chat",
        timestamp=datetime.now(UTC),
        kind="LLM",
        status="OK",
    )
    schema = _load("span.schema.json")
    Draft202012Validator.check_schema(schema)
    jsonschema.validate(span.model_dump(mode="json", exclude_none=True), schema)


def test_span_rejects_bad_trace_id() -> None:
    with pytest.raises(Exception):
        Span(
            tenant_id="t1",
            trace_id="not-hex",
            span_id="0123456789abcdef",
            name="x",
            timestamp=datetime.now(UTC),
        )


def test_eval_job_schema() -> None:
    configs = [EvalConfig(agent_id="judge_quality", model="m1")]
    job = EvalJob(
        trace_id="tr1",
        tenant_id="t1",
        trace_summary={"span_id": "s1"},
        eval_configs=configs,
        priority=Priority.P2,
        idempotency_key=eval_job_key("tr1", [c.model_dump() for c in configs]),
        created_at=datetime.now(UTC),
    )
    schema = _load("eval-job.schema.json")
    jsonschema.validate(job.model_dump(mode="json", exclude_none=True), schema)


def test_score_result_schema() -> None:
    result = ScoreResult(
        tenant_id="t1",
        trace_id="tr1",
        eval_scores=[{"agent_id": "a", "score_status": "scored", "normalized_score": 0.8}],
        final_score=0.8,
        bad_case_detected=False,
    )
    schema = _load("score-result.schema.json")
    jsonschema.validate(result.model_dump(mode="json", exclude_none=True), schema)
