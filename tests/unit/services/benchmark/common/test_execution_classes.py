"""Tests for the benchmark execution-class name registry and per-class routing.

Core case: two execution classes with the same execution/provider (async/aone) but
different logic route submit and callback parsing through their own class_ref and do not interfere.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.runs.common import execution_classes
from benchmark.runs.common.execution_classes import (
    AsyncJudgeSubmitRequest,
    ExecutionClassResolutionError,
    LocalAttributionRequest,
    normalize_judge_callback,
    resolve_attribution_class,
    resolve_judge_class,
)
registry_module = None
AttributionProviderRegistry = None
class IevalExternalJudge:
    pass
from benchmark.runs.schema import CriterionScore, ExternalBenchmarkRequest, JudgeCallback
from benchmark.workers import external_worker
from benchmark.workers.external_runner import (
    _normalize_judge_payload,
    profile_from_external_request,
    run_custom_local_attribution,
)

_IEVAL_JUDGE_REF = "business.runs.ieval.external_execution:IevalExternalJudge"
_IEVAL_ATTR_REF = "business.runs.ieval.external_execution:IevalExternalAttribution"


class OtherExternalJudge:
    """Another Aone judge: a custom request body plus a callback ``scores`` list shape."""

    execution = "async"
    provider = "external"

    def __init__(self) -> None:
        self.requests: list[AsyncJudgeSubmitRequest] = []

    async def submit(self, req: AsyncJudgeSubmitRequest) -> str:
        self.requests.append(req)
        return f"other-{req.item_id}"

    @classmethod
    def normalize_callback(cls, payload: JudgeCallback) -> JudgeCallback:
        extra = payload.model_extra or {}
        scores = {s["dim"]: CriterionScore(score=s["value"]) for s in extra.get("scores", [])}
        return payload.model_copy(update={"criterion_scores": scores})


class AsyncJudgeWithoutSubmit:
    execution = "async"
    provider = "external"


class OtherExternalAttribution:
    execution = "async"
    provider = "external"

    def __init__(self, *, client_factory: object = None, external_client: object = None) -> None:
        self.client_factory = client_factory

    async def submit(self, request: object) -> object:
        return SimpleNamespace(job_id="other-attr")

    async def query_job_status(self, job_id: str) -> object:
        return SimpleNamespace(status="running")


def _patch_loader(monkeypatch: pytest.MonkeyPatch, module: Any, classes: dict[str, type]) -> None:  # noqa: ANN401
    original = module.load_plugin_class

    def _load(ref: str, *, what: str = "class_ref") -> type:
        return classes[ref] if ref in classes else original(ref, what=what)

    monkeypatch.setattr(module, "load_plugin_class", _load)


# ── Alias resolution ──────────────────────────────────────────────────────────────


def test_resolve_ieval_external_classes() -> None:
    judge = resolve_judge_class("ieval_external_judge")
    attribution = resolve_attribution_class("ieval_external_attribution")

    assert (judge.execution, judge.provider, judge.target) == ("async", "external", None)
    assert judge.class_ref == _IEVAL_JUDGE_REF
    assert (attribution.execution, attribution.provider, attribution.target) == (
        "async",
        "external",
        None,
    )
    assert attribution.class_ref == _IEVAL_ATTR_REF


def test_resolve_search_eval_local_judge_and_no_attribution() -> None:
    judge = resolve_judge_class("search_eval_judge")
    attribution = resolve_attribution_class("no_attribution")

    assert judge.execution == "sync"
    assert judge.provider == "local"
    assert judge.target == ("business.runs.search_eval.local_judge:SearchEvalLocalJudge")
    assert (attribution.execution, attribution.provider, attribution.target) == (
        "none",
        "none",
        None,
    )


def test_resolve_unified_evaluator_service_aliases_for_batch() -> None:
    """The batch registry accepts evaluator_service_* names shared with streaming (target=None)."""
    judge = resolve_judge_class("evaluator_service_judge")
    attribution = resolve_attribution_class("evaluator_service_attribution")

    assert (judge.execution, judge.provider, judge.target) == ("sync", "local", None)
    assert judge.class_ref.endswith(":EvaluatorServiceJudge")
    assert (attribution.execution, attribution.provider, attribution.target) == (
        "sync",
        "local",
        None,
    )
    assert attribution.class_ref.endswith(":EvaluatorServiceAttributionMarker")


def test_unknown_alias_is_rejected() -> None:
    with pytest.raises(ExecutionClassResolutionError, match="available aliases"):
        resolve_judge_class("unknown")


def test_async_class_without_submit_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = "benchmark.runs.x:AsyncJudgeWithoutSubmit"
    monkeypatch.setitem(execution_classes.JUDGE_CLASS_REFS, "broken_judge", ref)
    _patch_loader(monkeypatch, execution_classes, {ref: AsyncJudgeWithoutSubmit})

    with pytest.raises(ExecutionClassResolutionError, match="must implement"):
        resolve_judge_class("broken_judge")


def test_two_async_external_judges_resolve_to_distinct_class_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = "benchmark.runs.x:OtherAoneJudge"
    monkeypatch.setitem(execution_classes.JUDGE_CLASS_REFS, "other_external_judge", ref)
    _patch_loader(monkeypatch, execution_classes, {ref: OtherExternalJudge})

    ieval = resolve_judge_class("ieval_external_judge")
    other = resolve_judge_class("other_external_judge")

    assert (ieval.execution, ieval.provider) == (other.execution, other.provider)
    assert ieval.class_ref != other.class_ref


def test_external_request_profile_carries_class_refs() -> None:
    profile = profile_from_external_request(
        ExternalBenchmarkRequest(
            benchmark_name="ieval_external",
            judge_class="ieval_external_judge",
            attribution_class="ieval_external_attribution",
            rubric_class="amap_chat_rubric_v2",
        ),
    )

    assert profile.judge.class_ref == _IEVAL_JUDGE_REF
    assert profile.attribution.class_ref == _IEVAL_ATTR_REF


# ── async judge: dispatch submit by class ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_dispatches_to_configured_judge_impl() -> None:
    client = MagicMock()
    client.submit_evaluation = AsyncMock()
    other = OtherExternalJudge()
    submit = external_worker._ExternalJudgeSubmit(
        run_id="bench-20260923",
        item_id="i1",
        data={"query": "q"},
        dims="D0",
    )

    job_id = await external_worker._submit_judge_to_external(client, submit, other)

    assert job_id == "other-i1"
    assert other.requests[0].dims == "D0"
    assert other.requests[0].client is client
    client.submit_evaluation.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_defaults_to_ieval_judge_without_class_ref() -> None:
    client = MagicMock()
    client.submit_evaluation = AsyncMock(return_value=SimpleNamespace(job_id="aone-1"))
    submit = external_worker._ExternalJudgeSubmit(
        run_id="bench-20260923",
        item_id="i1",
        data={"query": "q", "response_text": "r"},
    )

    job_id = await external_worker._submit_judge_to_external(client, submit)

    assert job_id == "aone-1"
    request = client.submit_evaluation.await_args.args[0]
    assert request.params.task_id == "i1"
    assert request.params.ds == "20260923"


# ── async judge: dispatch callback parsing by class ──────────────────────────────────────────


def _raw_callback() -> JudgeCallback:
    return JudgeCallback.model_validate(
        {"run_id": "r1", "job_id": "j1", "scores": [{"dim": "D1", "value": 2.0}]},
    )


def test_normalize_callback_uses_run_judge_class(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = "benchmark.runs.x:OtherAoneJudge"
    _patch_loader(monkeypatch, execution_classes, {ref: OtherExternalJudge})

    other = normalize_judge_callback(ref, _raw_callback())
    ieval = normalize_judge_callback(_IEVAL_JUDGE_REF, _raw_callback())

    assert other.criterion_scores["D1"].score == 2.0
    assert ieval.criterion_scores == {}


def test_normalize_failure_becomes_error_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(ref: str, *, what: str = "class_ref") -> type:
        raise RuntimeError("bad hook")

    monkeypatch.setattr(execution_classes, "load_plugin_class", _boom)
    config = {"judge": {"class_ref": "benchmark.runs.x:Broken"}}

    payload = _normalize_judge_payload(config, _raw_callback())

    assert payload.error is not None
    assert not payload.is_success


def test_ieval_judge_normalize_is_identity() -> None:
    payload = _raw_callback()
    assert IevalExternalJudge.normalize_callback(payload) is payload


# ── async attribution: route provider by class ───────────────────────────────────


def test_registry_resolves_class_ref_before_provider_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = "benchmark.runs.x:OtherAoneAttribution"
    _patch_loader(monkeypatch, registry_module, {ref: OtherExternalAttribution})
    factory = MagicMock()
    registry = AttributionProviderRegistry.from_client_factory(factory)

    other = registry.resolve_for_job({"attribution_provider": "external", "attribution_class_ref": ref})
    ieval = registry.resolve_for_job(
        {"attribution_provider": "external", "attribution_class_ref": _IEVAL_ATTR_REF},
    )
    legacy = registry.resolve_for_job({"attribution_provider": "external"})

    assert isinstance(other, OtherExternalAttribution)
    assert other.client_factory is factory
    assert type(ieval).__name__ == "IevalExternalAttribution"
    assert legacy is registry.get_provider("external")
    assert registry.resolve_for_job({"attribution_class_ref": ref}) is other


def test_registry_without_client_returns_none_for_class_ref() -> None:
    registry = AttributionProviderRegistry.from_external_client(None)
    assert registry.resolve_for_job({"attribution_class_ref": _IEVAL_ATTR_REF}) is None


# ── sync attribution: custom-class count reconciliation ─────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_local_attribution_counts_received_even_on_failure() -> None:
    service = MagicMock()
    service.increment_attribution_received_once = AsyncMock()
    impl = MagicMock()
    impl.attribute_item = AsyncMock(side_effect=RuntimeError("llm down"))
    req = LocalAttributionRequest(run_id="r1", item_id="i1", fused_score=0.5, provider="local")

    await run_custom_local_attribution(service, impl, req)

    impl.attribute_item.assert_awaited_once_with(req)
    service.increment_attribution_received_once.assert_awaited_once_with("r1", "i1")
