"""Unit tests for ExternalDailyBenchmarkConfig (D5 require_case_id / D9 rubric tightened)."""

import pytest
from benchmark.domain.external_daily_config import ExternalDailyBenchmarkConfig
from pydantic import ValidationError


def _payload(**updates: object) -> dict:
    payload: dict[str, object] = {
        "benchmark_name": "ieval_external",
        "judge_class": "ieval_external_judge",
        "attribution_class": "ieval_external_attribution",
        "rubric_class": "amap_chat_rubric_v2",
    }
    payload.update(updates)
    return payload


def test_require_case_id_defaults_false() -> None:
    cfg = ExternalDailyBenchmarkConfig.model_validate(_payload())
    assert cfg.require_case_id is False


def test_require_case_id_accepts_true() -> None:
    cfg = ExternalDailyBenchmarkConfig.model_validate(_payload(require_case_id=True))
    assert cfg.require_case_id is True


def test_rubric_rejected_in_judge_attribution_mode() -> None:
    with pytest.raises(ValidationError, match="attribution_only"):
        ExternalDailyBenchmarkConfig.model_validate(
            _payload(mode="judge_attribution", rubric="custom-rubric"),
        )


def test_rubric_allowed_in_attribution_only_mode() -> None:
    cfg = ExternalDailyBenchmarkConfig.model_validate(
        _payload(mode="attribution_only", rubric="custom-rubric"),
    )
    assert cfg.rubric == "custom-rubric"


def test_rubric_none_is_always_allowed() -> None:
    cfg = ExternalDailyBenchmarkConfig.model_validate(_payload())
    assert cfg.rubric is None
