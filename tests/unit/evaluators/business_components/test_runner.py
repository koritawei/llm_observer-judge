"""Public evaluation Runner contract tests: nine-node topology, short-circuit, skipped, exception isolation, and the observation channel."""

from typing import TypedDict

import pytest
from benchmark.pipeline.profiles import (
    ExtensionBinding,
    ProfileSpec,
)
from benchmark.pipeline.runner import (
    ENVELOPE_IDENTITY,
    ERROR,
    ERROR_CODE,
    STAGE_NODES,
    STAGE_REPORT,
    EvaluationPipelineHooks,
    build_evaluation_graph,
    envelope_identity_from_profile,
    skipped,
)
from benchmark.utils.exceptions import EVALUATION_FAILED_CODE
from pydantic import ValidationError


class _RunnerTestState(TypedDict, total=False):
    trace: list[str]
    result: str
    error: str | None
    error_code: str | None
    stage_report: dict[str, dict[str, object]]
    envelope_identity: object


def _make_test_profile() -> ProfileSpec:
    """Build the smallest usable ProfileSpec, so the test does not depend on a business plugin module that is not merged yet."""

    return ProfileSpec(
        profile_id="test.eval",
        profile_version="v1",
        plugin="test_plugin",
        catalog_version="2026-01-01",
        extension=ExtensionBinding(extension_type="test_eval"),
        enabled_dimensions=("D0", "D1"),
    )


def _hooks_factory(**overrides) -> EvaluationPipelineHooks:
    """Build Hooks that record the execution trace; overrides can replace a handler per stage."""

    def _make(stage: str):
        async def _handler(state: dict) -> dict:
            trace = list(state.get("trace") or [])
            trace.append(stage)
            return {"trace": trace}

        return _handler

    handlers = {stage: overrides.get(stage, _make(stage)) for stage in STAGE_NODES}
    return EvaluationPipelineHooks(**handlers)


async def _invoke(hooks: EvaluationPipelineHooks, initial: dict | None = None) -> dict:
    graph = build_evaluation_graph(_RunnerTestState, hooks)
    return await graph.ainvoke(initial or {})


async def test_runner_executes_all_nine_stages_in_fixed_order():
    final = await _invoke(_hooks_factory())

    assert final["trace"] == list(STAGE_NODES)
    report = final[STAGE_REPORT]
    assert set(report) == set(STAGE_NODES)
    for stage in STAGE_NODES:
        assert report[stage]["status"] == "executed"
        assert report[stage]["duration_ms"] >= 0
        assert "reason" not in report[stage]


async def test_runner_short_circuits_to_end_on_error_channel():
    async def _failing_stage(state: dict) -> dict:
        return {ERROR: "boom", ERROR_CODE: EVALUATION_FAILED_CODE}

    final = await _invoke(_hooks_factory(run_checkers=_failing_stage))

    # After run_checkers writes an error it goes straight to END, and no later stage runs.
    assert final["trace"] == ["build_context", "resolve_profile", "collect_evidence"]
    assert final[ERROR] == "boom"
    report = final[STAGE_REPORT]
    assert report["run_checkers"]["status"] == "failed"
    assert "plan_judges" not in report


async def test_runner_records_skipped_stage_without_leaking_marker():
    async def _noop_stage(state: dict) -> dict:
        return skipped("本业务无此阶段工作")

    final = await _invoke(_hooks_factory(collect_metrics=_noop_stage))

    # trace is maintained by the handlers: _noop_stage does not record a trace, so there are only 8 nodes.
    assert final["trace"] == [s for s in STAGE_NODES if s != "collect_metrics"]
    # stage_report is maintained by the Runner: it must contain all nine nodes, with collect_metrics marked skipped.
    report = final[STAGE_REPORT]
    assert set(report) == set(STAGE_NODES)
    assert report["collect_metrics"]["status"] == "skipped"
    assert report["collect_metrics"]["reason"] == "本业务无此阶段工作"
    # The skipped marker key must not leak into the business channel.
    assert "__stage_skipped_reason__" not in final


async def test_runner_isolates_unexpected_exception_into_error_channel():
    async def _raising_stage(state: dict) -> dict:
        raise ValueError("stage blew up")

    final = await _invoke(_hooks_factory(plan_judges=_raising_stage))

    assert "plan_judges 阶段执行异常" in final[ERROR]
    assert final[ERROR_CODE] == EVALUATION_FAILED_CODE
    assert final[STAGE_REPORT]["plan_judges"]["status"] == "failed"
    # Once an exception is isolated, later stages do not run.
    assert "run_judges" not in final[STAGE_REPORT]


async def test_runner_enforces_stage_timeout():
    async def _slow_stage(state: dict) -> dict:
        import asyncio  # noqa: PLC0415 — imported locally inside the test to avoid module-level side effects

        await asyncio.sleep(0.5)
        return {}

    hooks = _hooks_factory(run_judges=_slow_stage)
    hooks = EvaluationPipelineHooks(
        **{stage: getattr(hooks, stage) for stage in STAGE_NODES},
        stage_timeouts={"run_judges": 0.01},
    )
    final = await _invoke(hooks)

    assert "run_judges 阶段执行超时" in final[ERROR]
    assert final[ERROR_CODE] == EVALUATION_FAILED_CODE
    assert final[STAGE_REPORT]["run_judges"]["status"] == "failed"


def test_envelope_identity_from_profile_carries_digest_and_scenario():
    profile = _make_test_profile()

    identity = envelope_identity_from_profile(profile, scenario="poi_search")

    assert identity.profile_id == profile.profile_id
    assert identity.profile_version == profile.profile_version
    assert identity.plugin == profile.plugin
    assert identity.catalog_version == profile.catalog_version
    assert identity.config_digest  # non-empty fingerprint
    assert identity.scenario == "poi_search"
    with pytest.raises(ValidationError):  # frozen: once the identity is written it is immutable
        identity.scenario = "other"


async def test_runner_writes_envelope_identity_channel():
    async def _resolve(state: dict) -> dict:
        return {ENVELOPE_IDENTITY: envelope_identity_from_profile(_make_test_profile())}

    final = await _invoke(_hooks_factory(resolve_profile=_resolve))

    assert final[ENVELOPE_IDENTITY].plugin == "test_plugin"
