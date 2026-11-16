"""EvalConfigEntry.ingest defaults to kafka; sidecar is explicit."""

import pytest
from benchmark.domain.schemas import INGEST_KAFKA, INGEST_SIDECAR, EvalConfigEntry
from pydantic import ValidationError


def test_ingest_defaults_to_kafka() -> None:
    entry = EvalConfigEntry.model_validate(
        {"evaluatorName": "example", "sampling": 1.0}
    )
    assert entry.ingest == INGEST_KAFKA
    assert entry.is_sidecar_ingest is False


def test_ingest_sidecar_is_explicit() -> None:
    entry = EvalConfigEntry.model_validate(
        {"evaluatorName": "amap_agent", "sampling": 1.0, "ingest": "sidecar"}
    )
    assert entry.ingest == INGEST_SIDECAR
    assert entry.is_sidecar_ingest is True


def test_ingest_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        EvalConfigEntry.model_validate(
            {"evaluatorName": "example", "sampling": 1.0, "ingest": "both"}
        )
