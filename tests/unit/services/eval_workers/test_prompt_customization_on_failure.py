"""The judge failure path still extracts prompt_customization so it can be persisted."""

from benchmark.evaluators.base import EvaluatorRequest, PromptBundleRef
from benchmark.workers.benchmark_worker import _prompt_customization_on_failure
from benchmark.utils.exceptions import EVALUATION_FAILED_CODE, EvaluatorError, evaluator_status


def test_prefers_error_details() -> None:
    req = EvaluatorRequest(input="q", output="a")
    err = EvaluatorError(
        code=evaluator_status(EVALUATION_FAILED_CODE),
        error_type=EVALUATION_FAILED_CODE,
        message="boom",
    )
    # details is assigned after construction (FUNC-PARAMS-001; consistent with the EvaluatorError docstring contract)
    err.details = {"prompt_customization": {"content_version": 56, "git_commit": "abc"}}
    meta = _prompt_customization_on_failure(err, req)
    assert meta == {"content_version": 56, "git_commit": "abc"}


def test_falls_back_to_request_bundle_ref() -> None:
    req = EvaluatorRequest(
        input="q",
        output="a",
        prompt_bundle_ref=PromptBundleRef(
            stage="judge",
            git_path="benchmark-customizations/example/biz/v1/judge/v56",
            git_commit="9ff5ffd288832ca148ade5d2ff2f1381a562d637",
            bundle_sha256="c0ab" * 16,
            content_version=56,
            slots=["prompt"],
            files=[],
        ),
    )
    meta = _prompt_customization_on_failure(RuntimeError("x"), req)
    assert meta is not None
    assert meta["content_version"] == 56
    assert meta["git_path"].endswith("/v56")


def test_returns_none_without_refs_or_details() -> None:
    req = EvaluatorRequest(input="q", output="a")
    assert _prompt_customization_on_failure(RuntimeError("x"), req) is None
