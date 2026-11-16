"""Unit tests for streaming triage / attribution aliases and sync/async combinations."""

import pytest
from benchmark.runs.common import execution_classes as ec_mod
from benchmark.runs.common.execution_classes import (
    ExecutionClassResolutionError,
    effective_streaming_attribution_execution,
    effective_streaming_judge_execution,
    resolve_streaming_attribution_class,
    resolve_streaming_triage_class,
    validate_streaming_execution_combo,
)
from benchmark.streaming.execution_plugins import (
    AmapAgentStreamingAsyncAttribution,
    AmapAgentStreamingAttribution,
    AmapAgentStreamingTriage,
)


class TestStreamingTriageAliases:
    def test_amap_agent_triage_resolves(self) -> None:
        spec = resolve_streaming_triage_class("amap_agent_triage")
        assert spec.alias == "amap_agent_triage"
        assert spec.execution == "sync"
        assert spec.target is not None
        plugin = AmapAgentStreamingTriage()
        clf = plugin.build_classifier()
        assert clf.classify  # type: ignore[truthy-function]

    def test_unknown_triage_rejected(self) -> None:
        with pytest.raises(ExecutionClassResolutionError, match="unknown streaming triage_class"):
            resolve_streaming_triage_class("not_a_real_triage")


class TestStreamingAttributionAliases:
    def test_amap_sync_and_async_aliases(self) -> None:
        sync = resolve_streaming_attribution_class("amap_agent_attribution")
        assert sync.execution == "sync"
        assert AmapAgentStreamingAttribution.agent_key == "attribution_amap_agent"

        async_spec = resolve_streaming_attribution_class("amap_agent_async_attribution")
        assert async_spec.execution == "async"
        assert AmapAgentStreamingAsyncAttribution.agent_key == "attribution_amap_agent"

    def test_no_attribution_is_none(self) -> None:
        spec = resolve_streaming_attribution_class("no_attribution")
        assert spec.execution == "none"

    def test_batch_alias_rejected(self) -> None:
        with pytest.raises(ExecutionClassResolutionError, match="unknown"):
            resolve_streaming_attribution_class("ieval_external_attribution")


class TestExecutionOverridesAndCombo:
    def test_judge_override_wins(self) -> None:
        assert (
            effective_streaming_judge_execution(class_execution="sync", override="async") == "async"
        )
        assert effective_streaming_judge_execution(class_execution=None, override=None) == "sync"

    def test_attribution_no_attribution_forces_none(self) -> None:
        assert (
            effective_streaming_attribution_execution(
                class_execution="sync",
                override="async",
                attribution_class="no_attribution",
            )
            == "none"
        )

    def test_combo_sync_judge_async_attribution_ok(self) -> None:
        validate_streaming_execution_combo(
            judge_execution="sync",
            attribution_execution="async",
            judge_class="evaluator_service_judge",
            attribution_class="amap_agent_attribution",
        )

    def test_async_judge_without_submit_rejected(self) -> None:
        with pytest.raises(ExecutionClassResolutionError, match="does not implement submit"):
            validate_streaming_execution_combo(
                judge_execution="async",
                attribution_execution="sync",
                judge_class="evaluator_service_judge",
                attribution_class=None,
            )

    def test_async_judge_null_class_rejected(self) -> None:
        with pytest.raises(ExecutionClassResolutionError, match="requires an explicit judge_class"):
            validate_streaming_execution_combo(
                judge_execution="async",
                attribution_execution="sync",
                judge_class=None,
                attribution_class=None,
            )

    def test_invalid_override_rejected(self) -> None:
        with pytest.raises(ExecutionClassResolutionError, match="invalid judge_execution"):
            effective_streaming_judge_execution(class_execution=None, override="maybe")


class TestStreamingConfigFields:
    def test_task_config_parses_execution_and_triage(self) -> None:
        from benchmark.domain.streaming_config import StreamingTaskConfig

        task = StreamingTaskConfig.model_validate(
            {
                "name": "amap_agent_daily",
                "topics": ["langfuse.records"],
                "judge_class": "evaluator_service_judge",
                "judge_execution": "SYNC",
                "triage_class": "amap_agent_triage",
                "attribution_class": "amap_agent_async_attribution",
                "attribution_execution": "Async",
            },
        )
        assert task.judge_execution == "sync"
        assert task.triage_class == "amap_agent_triage"
        assert task.attribution_class == "amap_agent_async_attribution"
        assert task.attribution_execution == "async"


def test_streaming_registries_registered() -> None:
    assert "amap_agent_triage" in ec_mod.STREAMING_TRIAGE_CLASS_REFS
    assert "amap_agent_attribution" in ec_mod.STREAMING_ATTRIBUTION_CLASS_REFS
    assert "amap_agent_async_attribution" in ec_mod.STREAMING_ATTRIBUTION_CLASS_REFS
