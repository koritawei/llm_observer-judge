"""rerun payload: OSS snapshot replay and sparse-snapshot enrichment."""

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "rerun.py"


def _load_module() -> ModuleType:
    name = "rerun_under_test"
    stub_bootstrap = types.ModuleType("_bootstrap_quiet")
    stub_main = types.ModuleType("benchmark.service.main")
    stub_main.create_app = lambda **_kwargs: None  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"_bootstrap_quiet": stub_bootstrap, "benchmark.service.main": stub_main},
    ):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


rerun = _load_module()


def _run(*, run_id: str = "search_eval_v2-20261103", **cfg: Any) -> SimpleNamespace:
    config: dict[str, object] = {
        "benchmark_services": "search_eval_v2",
        "generate_report": True,
    }
    config.update(cfg)
    return SimpleNamespace(
        run_id=run_id,
        benchmark_name="search_eval_v2",
        dataset_name="",
        project_id="",
        config_json=config,
    )


def test_build_rerun_payload_replays_oss_object_key() -> None:
    payload = rerun._build_rerun_payload(
        _run(
            judge_class="search_eval_v2_judge",
            attribution_class="no_attribution",
            rubric_class="search_eval_v2_rubric",
            oss_prefix="search-eval/benchmark_result/20261103",
            data_date="20261103",
            oss_object_key=("search-eval/benchmark_result/20261103/benchmark_final_results.csv"),
        ),
    )
    assert payload["oss_prefix"] == "search-eval/benchmark_result/20261103"
    assert payload["oss_object_key"] == (
        "search-eval/benchmark_result/20261103/benchmark_final_results.csv"
    )
    assert payload["date"] == "20261103"


def test_build_rerun_payload_pins_dated_prefix_when_object_key_missing() -> None:
    payload = rerun._build_rerun_payload(
        _run(
            judge_class="search_eval_v2_judge",
            attribution_class="no_attribution",
            rubric_class="search_eval_v2_rubric",
            oss_prefix="search-eval/benchmark_result/20261103",
        ),
    )
    assert payload["oss_object_key"] == (
        "search-eval/benchmark_result/20261103/benchmark_final_results.csv"
    )


def test_apply_profile_snapshot_skips_rubric_json_in_judge_mode() -> None:
    payload: dict[str, object] = {"generate_report": True}
    rerun._apply_profile_snapshot(
        payload,
        {
            "rubric_ref": None,
            "rubric": [{"id": "search_ai_card_v2"}],
            "aggregation": {"method": "weighted_avg"},
        },
    )
    assert "rubric" not in payload
    assert payload["aggregation"] == {"method": "weighted_avg"}

    attr_only: dict[str, object] = {"mode": "attribution_only"}
    rerun._apply_profile_snapshot(
        attr_only,
        {"rubric": [{"id": "dim"}]},
    )
    assert attr_only["rubric"] == [{"id": "dim"}]


def test_build_rerun_payload_allows_sparse_snapshot() -> None:
    """A thin placeholder left by a previous failed init no longer hard-fails at the build stage."""
    payload = rerun._build_rerun_payload(
        _run(
            external=True,
            pull_attempt_count=1,
            attribution_provider=None,
        ),
    )
    assert payload["benchmark_name"] == "search_eval_v2"
    assert payload["date"] == "20261103"
    assert payload.get("judge_class") is None
    assert payload.get("rubric_class") is None
    assert "oss_prefix" not in payload


@pytest.mark.asyncio
async def test_enrich_overrides_placeholder_component_rubric() -> None:
    """rubric_alias_for_ref(None)=component_rubric must not hide the real alias in meta."""
    from benchmark.runs.common.agent_capabilities import CapabilityBindingView

    services = MagicMock()
    services.get_metadata.return_value = SimpleNamespace(
        rubric_class=("business.runs.search_eval_v2.services:SearchEvalV2Rubric"),
    )
    services.benchmark_capabilities = (
        CapabilityBindingView(
            pipeline="batch",
            kind="judge",
            execution="sync",
            provider="local",
            description="v2 judge",
            alias="search_eval_v2_judge",
        ),
        CapabilityBindingView(
            pipeline="batch",
            kind="attribution",
            execution="sync",
            provider="none",
            description="no attribution",
            alias="no_attribution",
        ),
    )
    app = MagicMock()
    app.state.runtime.agent_manager.get.return_value = services

    payload = await rerun._enrich_rerun_payload(
        app,
        {
            "benchmark_name": "search_eval_v2",
            "date": "20261103",
            "rubric_class": "component_rubric",
            "oss_prefix": "search-eval/benchmark_result",
            "generate_report": True,
        },
    )
    assert payload["rubric_class"] == "search_eval_v2_rubric"
    assert payload["parse_interceptor_ref"].endswith(":SearchEvalParseInterceptor")


def test_pin_dated_oss_object_key() -> None:
    payload = {
        "oss_prefix": "search-eval/benchmark_result/20261103",
        "date": "20261103",
    }
    rerun._pin_dated_oss_object_key(payload)
    assert payload["oss_object_key"] == (
        "search-eval/benchmark_result/20261103/benchmark_final_results.csv"
    )

    # An offset-date tail ≠ date is pinned too
    offset = {
        "oss_prefix": "search-eval/benchmark_result/20261102",
        "date": "20261103",
    }
    rerun._pin_dated_oss_object_key(offset)
    assert offset["oss_object_key"] == (
        "search-eval/benchmark_result/20261102/benchmark_final_results.csv"
    )


@pytest.mark.asyncio
async def test_enrich_rerun_payload_fills_aliases_and_dataset_oss() -> None:
    from benchmark.runs.common.agent_capabilities import CapabilityBindingView

    services = MagicMock()
    services.get_metadata.return_value = SimpleNamespace(
        rubric_class=("business.runs.search_eval_v2.services:SearchEvalV2Rubric"),
    )
    services.benchmark_capabilities = (
        CapabilityBindingView(
            pipeline="batch",
            kind="judge",
            execution="sync",
            provider="local",
            description="v2 judge",
            alias="search_eval_v2_judge",
        ),
        CapabilityBindingView(
            pipeline="batch",
            kind="attribution",
            execution="sync",
            provider="none",
            description="no attribution",
            alias="no_attribution",
        ),
    )
    app = MagicMock()
    app.state.runtime.agent_manager.get.return_value = services

    with patch.object(
        rerun,
        "_lookup_dataset_for_benchmark",
        new=AsyncMock(
            return_value=(
                "search-eval/benchmark_result",
                "ds-1",
                "search-daily",
                "task-1",
            ),
        ),
    ):
        payload = await rerun._enrich_rerun_payload(
            app,
            {
                "benchmark_name": "search_eval_v2",
                "date": "20261103",
                "generate_report": True,
            },
        )

    assert payload["judge_class"] == "search_eval_v2_judge"
    assert payload["attribution_class"] == "no_attribution"
    assert payload["rubric_class"] == "search_eval_v2_rubric"
    assert payload["oss_prefix"] == "search-eval/benchmark_result"
    assert payload["dataset_id"] == "ds-1"
    assert payload["eval_task_id"] == "task-1"


@pytest.mark.asyncio
async def test_enrich_rerun_payload_expands_date_macro() -> None:
    from benchmark.runs.common.agent_capabilities import CapabilityBindingView

    services = MagicMock()
    services.get_metadata.return_value = SimpleNamespace(
        rubric_class=("business.runs.search_eval_v2.services:SearchEvalV2Rubric"),
    )
    services.benchmark_capabilities = (
        CapabilityBindingView(
            pipeline="batch",
            kind="judge",
            execution="sync",
            provider="local",
            description="v2 judge",
            alias="search_eval_v2_judge",
        ),
        CapabilityBindingView(
            pipeline="batch",
            kind="attribution",
            execution="sync",
            provider="none",
            description="no attribution",
            alias="no_attribution",
        ),
    )
    app = MagicMock()
    app.state.runtime.agent_manager.get.return_value = services

    payload = await rerun._enrich_rerun_payload(
        app,
        {
            "benchmark_name": "search_eval_v2",
            "date": "20261103",
            "oss_prefix": "search-eval/benchmark_result/${yyyymmdd}",
            "generate_report": True,
        },
    )
    assert payload["oss_prefix"] == "search-eval/benchmark_result/20261103"
    assert payload["oss_object_key"] == (
        "search-eval/benchmark_result/20261103/benchmark_final_results.csv"
    )
