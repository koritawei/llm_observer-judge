"""T13 comparison tests: performance definitions match generate-eval-report.py item by item.

The three-bucket map and the aggregation logic on the service side are a **copy** of the script side (the script
lives under services/vehicle-agent/skills/ and hardcodes paths, so it cannot be imported across services into production code).
Two copies can drift when one side is edited and the other is forgotten. This test really imports the script module, runs both
sides on the same fixture, and fails when the results differ.

Three definitions are covered (Spec Q5 / AC-10):
1. The three-bucket map matches item by item, and bucketed aggregation matches
2. End-to-end latency extraction: trace.latency × 1000
3. First-body latency extraction: metadata.first_text_response_time_ms
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from benchmark.reporting.cluster.constants import (
    BUCKET_LABELS,
    MODEL_BUCKETS,
    OBSERVATION_NAME_TO_BUCKET,
)
from benchmark.reporting.cluster_service.collector import aggregate_buckets

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[4]
    / "vehicle-agent"
    / "skills"
    / "vehicle-agent-eval"
    / "scripts"
    / "generate-eval-report.py"
)


def _load_script_module(path: Path, module_name: str) -> Any:
    """Load the script module by path (the filename contains a hyphen, so a normal import is impossible).

    When a dependency is missing the script calls ``sys.exit(1)`` itself (rather than raising ImportError), so besides
    ``ImportError`` this must also catch ``SystemExit``. Otherwise, with no local openpyxl, the whole
    test file errors instead of skipping. CI has openpyxl, so the comparison test really runs in CI.
    """
    if not path.is_file():
        pytest.skip(f"script not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"cannot load spec: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (ImportError, SystemExit) as exc:
        pytest.skip(f"script import failed (missing deps?): {path.name} {exc}")
    return module


@pytest.fixture(scope="module")
def script() -> Any:
    return _load_script_module(_SCRIPT_PATH, "_gen_eval_report")


@pytest.fixture(scope="module")
def compare_script() -> Any:
    return _load_script_module(_SCRIPT_PATH.parent / "eval-compare.py", "_eval_compare")


def _obs(
    name: str,
    prompt: int,
    completion: int,
    total: int | None = None,
    latency: float | None = None,
    cost: float = 0.0,
) -> dict:
    entry: dict = {
        "name": name,
        "promptTokens": prompt,
        "completionTokens": completion,
        "totalTokens": total if total is not None else prompt + completion,
        "calculatedTotalCost": cost,
    }
    if latency is not None:
        entry["latency"] = latency
    return entry


#: Covers six shapes: multiple buckets, repeated calls, zero tokens, an unknown name, missing totalTokens, and missing latency.
_FIXTURE_OBSERVATIONS: list[dict] = [
    _obs("AmapChatModel.chat", 100, 20, latency=1.5, cost=0.01),
    _obs("AmapChatModel.chat", 50, 10, latency=0.5, cost=0.005),
    _obs("cgi_classify", 10, 2, latency=0.1),
    _obs("model_call.verifyAgent", 8, 1, latency=0.05),
    _obs("model_call.unifiedIntentAgent", 12, 3, latency=0.08),
    _obs("unknown.node", 9, 1, latency=1.0),  # Unknown name: both sides should ignore it
    _obs("AmapChatModel.chat", 0, 0, total=0),  # Zero tokens: both sides should skip it
    {"name": "AmapChatModel.chat", "promptTokens": 7, "completionTokens": 3},  # Missing total/latency
]


class TestBucketMappingParity:
    def test_mapping_tables_are_identical(self, script: Any) -> None:
        """The service's original three-bucket map aligns with the main script item by item, and keeps the script's extended nodes."""
        assert all(
            script._OBS_NAME_TO_BUCKET.get(name) == bucket
            for name, bucket in OBSERVATION_NAME_TO_BUCKET.items()
        )

    def test_bucket_tuple_identical(self, script: Any) -> None:
        # main's offline script has grown to seven buckets; the service report keeps the original three-bucket display contract.
        assert all(bucket in script.MODEL_BUCKETS for bucket in MODEL_BUCKETS)

    def test_bucket_labels_match_compare_script(self, compare_script: Any) -> None:
        """The Chinese bucket labels the xlsx column names depend on must match BUCKET_LABELS in eval-compare.py."""
        assert BUCKET_LABELS == compare_script.BUCKET_LABELS


class TestBucketAggregationParity:
    """The same fixture: service-side aggregate_buckets vs script-side aggregate_tokens_per_trace."""

    @pytest.fixture()
    def script_buckets(self, script: Any) -> dict[str, dict]:
        result = script.aggregate_tokens_per_trace({"t-1": _FIXTURE_OBSERVATIONS})
        assert "t-1" in result, "script should produce an entry for the fixture trace"
        return result["t-1"]

    @pytest.fixture()
    def service_buckets(self) -> dict:
        return aggregate_buckets(_FIXTURE_OBSERVATIONS)

    def test_calls_per_bucket_identical(self, script_buckets: dict, service_buckets: dict) -> None:
        for bucket in MODEL_BUCKETS:
            script_calls = script_buckets[bucket]["llm_calls"]
            service_calls = service_buckets[bucket].calls if bucket in service_buckets else 0
            assert service_calls == script_calls, f"{bucket} calls mismatch"

    def test_prompt_tokens_identical(self, script_buckets: dict, service_buckets: dict) -> None:
        for bucket in MODEL_BUCKETS:
            expected = script_buckets[bucket]["prompt_tokens"]
            actual = service_buckets[bucket].prompt_tokens if bucket in service_buckets else 0
            assert actual == expected, f"{bucket} prompt_tokens mismatch"

    def test_completion_tokens_identical(self, script_buckets: dict, service_buckets: dict) -> None:
        for bucket in MODEL_BUCKETS:
            expected = script_buckets[bucket]["completion_tokens"]
            actual = service_buckets[bucket].completion_tokens if bucket in service_buckets else 0
            assert actual == expected, f"{bucket} completion_tokens mismatch"

    def test_total_tokens_identical_including_fallback(
        self, script_buckets: dict, service_buckets: dict
    ) -> None:
        """Includes the prompt+completion fallback path when totalTokens is missing."""
        for bucket in MODEL_BUCKETS:
            expected = script_buckets[bucket]["total_tokens"]
            actual = service_buckets[bucket].total_tokens if bucket in service_buckets else 0
            assert actual == expected, f"{bucket} total_tokens mismatch"

    def test_latency_ms_identical(self, script_buckets: dict, service_buckets: dict) -> None:
        """The accumulated definition of latency (seconds) × 1000 matches."""
        for bucket in MODEL_BUCKETS:
            expected = script_buckets[bucket]["total_latency_ms"]
            actual = service_buckets[bucket].latency_ms if bucket in service_buckets else 0.0
            assert actual == pytest.approx(expected), f"{bucket} latency mismatch"

    def test_cost_identical(self, script_buckets: dict, service_buckets: dict) -> None:
        for bucket in MODEL_BUCKETS:
            expected = script_buckets[bucket]["total_cost"]
            actual = service_buckets[bucket].cost if bucket in service_buckets else 0.0
            assert actual == pytest.approx(expected), f"{bucket} cost mismatch"

    def test_unknown_name_excluded_by_both(
        self, script_buckets: dict, service_buckets: dict
    ) -> None:
        """The 10 tokens of unknown.node must not enter any bucket (same definition on both sides)."""
        script_total = sum(script_buckets[b]["total_tokens"] for b in MODEL_BUCKETS)
        service_total = sum(
            service_buckets[b].total_tokens for b in MODEL_BUCKETS if b in service_buckets
        )
        assert script_total == service_total
        assert service_total > 0  # Guard against a fake agreement where both sides compute 0


class TestTraceLevelMetricParity:
    """How end-to-end latency and first-body latency are extracted."""

    @pytest.mark.parametrize(
        ("latency_s", "expected_ms"),
        [
            (2.5, 2500.0),
            (0.123, 123.0),
            (1.0, 1000.0),
            # round() is banker's rounding (round(0.5)==0) — both sides use Python round,
            # so matching definitions is enough; do not invent another rounding rule.
            (0.0005, 0.0),
            (0.0015, 2.0),
        ],
    )
    async def test_service_e2e_matches_script_rounding(
        self, latency_s: float, expected_ms: float
    ) -> None:
        """Service-side e2e_ms == round(latency × 1000), matching the script's round() behavior."""
        from benchmark.reporting.cluster_service.collector import TracePerfCollector

        # Both formulas are self-consistent: the script side is round(lat * 1000) (the generate-eval-report main flow)
        assert float(round(latency_s * 1000)) == expected_ms

        class _Client:
            async def resolve_primary_id(self, trace_id: str) -> str:
                return trace_id

            async def get_trace(
                self, primary_id: str, *, fields: str = "core", expand_metadata: str | None = None
            ) -> dict:
                return {"latency": latency_s, "metadata": {}}

            async def list_observations_for_trace(self, primary_id: str, query) -> list:
                return []

        perf = (await TracePerfCollector(client=_Client()).fetch_perf(["t-1"]))["t-1"]
        assert perf.e2e_ms == expected_ms

    async def test_first_text_metadata_key(self) -> None:
        """First-body latency reads metadata.first_text_response_time_ms (the same key name as the script)."""
        from benchmark.reporting.cluster_service.collector import TracePerfCollector

        class _Client:
            async def resolve_primary_id(self, trace_id: str) -> str:
                return trace_id

            async def get_trace(
                self, primary_id: str, *, fields: str = "core", expand_metadata: str | None = None
            ) -> dict:
                return {
                    "latency": 1.0,
                    "metadata": {
                        "first_text_response_time_ms": 850,
                        # A 2.0-only field; this service does not consume it
                        "first_valid_response_time_ms": 300,
                    },
                }

            async def list_observations_for_trace(self, primary_id: str, query) -> list:
                return []

        perf = (await TracePerfCollector(client=_Client()).fetch_perf(["t-1"]))["t-1"]
        assert perf.first_text_ms == 850.0
