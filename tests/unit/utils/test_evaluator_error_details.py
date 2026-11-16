"""EvaluatorError.details keeps prompt_customization across the boundary."""

from benchmark.utils.exceptions import EVALUATION_FAILED_CODE, EvaluatorError, evaluator_status


def test_evaluator_error_details_round_trip() -> None:
    err = EvaluatorError(
        code=evaluator_status(EVALUATION_FAILED_CODE),
        error_type=EVALUATION_FAILED_CODE,
        message="llm boom",
    )
    err.details = {
        "prompt_customization": {
            "stage": "judge",
            "content_version": 56,
            "git_commit": "9ff5ffd28883",
            "git_path": "benchmark-customizations/.../judge/v56",
        }
    }
    restored = EvaluatorError.from_payload(err.to_payload())
    assert restored.message == "llm boom"
    assert restored.details is not None
    assert restored.details["prompt_customization"]["content_version"] == 56


def test_evaluator_error_without_details_omits_key() -> None:
    err = EvaluatorError(code=500, error_type=EVALUATION_FAILED_CODE, message="x")
    payload = err.to_payload()
    assert "details" not in payload
    restored = EvaluatorError.from_payload(payload)
    assert restored.details is None
