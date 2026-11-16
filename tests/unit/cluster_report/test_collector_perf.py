"""T12 unit tests: the Langfuse performance collector (three buckets / end-to-end / first body / degradation).

A fake client stands in for real Langfuse. It depends only on the contract shape of ``get_trace`` /
``list_observations_for_trace`` (same signatures as
app/services/trace_resolver/langfuse_client.py).
The collector no longer calls ``resolve_primary_id`` — trace_id is already the primary id when it is stored.
"""

import asyncio
from typing import Any

import pytest
from benchmark.reporting.cluster.schema import CasePerf
from benchmark.reporting.cluster_service.collector import (
    TracePerfCollector,
    aggregate_buckets,
    aggregate_models,
)
from benchmark.trace.http_backend import ObservationListQuery


class _FakeTrace:
    """A pydantic-like trace object (the collector must accept both attribute access and dict form)."""

    def __init__(self, latency: float | None, metadata: dict | None = None) -> None:
        self.latency = latency
        self.metadata = metadata or {}


class _FakeClient:
    def __init__(
        self,
        *,
        traces: dict[str, Any] | None = None,
        observations: dict[str, list[dict]] | None = None,
        raise_on: set[str] | None = None,
        delay_on: set[str] | None = None,
    ) -> None:
        self._traces = traces or {}
        self._observations = observations or {}
        self._raise_on = raise_on or set()
        self._delay_on = delay_on or set()
        self.observation_calls: list[dict] = []

    async def get_trace(
        self, primary_id: str, *, fields: str = "core", expand_metadata: str | None = None
    ) -> Any:
        if primary_id in self._raise_on:
            raise RuntimeError("get_trace boom")
        return self._traces.get(primary_id)

    async def list_observations_for_trace(
        self,
        primary_id: str,
        query: ObservationListQuery,
    ) -> list[dict]:
        self.observation_calls.append(
            {"primary_id": primary_id, "limit": query.limit, "type": query.type}
        )
        if primary_id in self._delay_on:
            await asyncio.sleep(10)
        return self._observations.get(primary_id, [])


def _gen_obs(
    name: str,
    prompt: int,
    completion: int,
    latency: float = 0.0,
    cost: float = 0.0,
    model: str = "",
) -> dict:
    return {
        "name": name,
        "promptTokens": prompt,
        "completionTokens": completion,
        "totalTokens": prompt + completion,
        "latency": latency,
        "calculatedTotalCost": cost,
        **({"model": model} if model else {}),
    }


class TestAggregateBuckets:
    def test_maps_names_to_buckets(self) -> None:
        buckets = aggregate_buckets(
            [
                _gen_obs("AmapChatModel.chat", 100, 20, latency=1.5, cost=0.01, model="qwen-max"),
                _gen_obs("AmapChatModel.chat", 50, 10, latency=0.5, model="qwen-max"),
                _gen_obs("cgi_classify", 10, 2, latency=0.1),
            ]
        )

        assert buckets["react"].calls == 2
        assert buckets["react"].prompt_tokens == 150
        assert buckets["react"].completion_tokens == 30
        assert buckets["react"].total_tokens == 180
        assert buckets["react"].latency_ms == 2000.0  # (1.5 + 0.5) × 1000
        assert buckets["react"].cost == 0.01
        assert buckets["react"].models == {"qwen-max": 2}
        assert buckets["cgi"].calls == 1
        assert "summary" not in buckets

    def test_legacy_2_0_names_supported(self) -> None:
        """2.0 names map the same way (the mapping table matches the script side item by item)."""
        buckets = aggregate_buckets(
            [
                _gen_obs("model_call.unifiedIntentAgent", 10, 1),
                _gen_obs("model_call.toolAgentReact", 20, 2),
                _gen_obs("model_call.verifyAgent", 5, 1),
            ]
        )

        assert set(buckets) == {"cgi", "react", "verify"}

    def test_unknown_name_kept_in_other_bucket(self) -> None:
        """A newly added VUI Generation name still counts toward total tokens, so the performance card is not blank."""
        buckets = aggregate_buckets([_gen_obs("unknown.node", 9, 1, model="qwen-plus")])

        assert buckets["other"].total_tokens == 10
        assert buckets["other"].models == {"qwen-plus": 1}

    def test_groups_metrics_by_actual_model(self) -> None:
        models = aggregate_models(
            [
                _gen_obs("vui.reason", 100, 20, latency=1.5, model="qwen-max"),
                _gen_obs("vui.reply", 50, 10, latency=0.5, model="qwen-plus"),
                _gen_obs("vui.reply", 10, 2, latency=0.1),
            ]
        )

        assert models["qwen-max"].total_tokens == 120
        assert models["qwen-plus"].calls == 1
        assert models["未采集"].total_tokens == 12

    def test_zero_token_observation_skipped(self) -> None:
        """An observation with no tokens is skipped (definition aligned with the script side)."""
        assert aggregate_buckets([_gen_obs("AmapChatModel.chat", 0, 0)]) == {}

    def test_total_tokens_fallback_to_sum(self) -> None:
        """When totalTokens is missing, fall back to prompt + completion."""
        buckets = aggregate_buckets(
            [
                {"name": "AmapChatModel.chat", "promptTokens": 7, "completionTokens": 3},
            ]
        )
        assert buckets["react"].total_tokens == 10

    def test_missing_latency_does_not_break(self) -> None:
        buckets = aggregate_buckets(
            [
                {"name": "AmapChatModel.chat", "promptTokens": 1, "completionTokens": 1},
            ]
        )
        assert buckets["react"].latency_ms == 0.0

    def test_non_dict_entries_ignored(self) -> None:
        assert aggregate_buckets(["bad", None, 1]) == {}  # type: ignore[list-item]

    def test_empty_input(self) -> None:
        assert aggregate_buckets([]) == {}


class TestFetchPerf:
    async def test_collects_trace_and_observation_metrics(self) -> None:
        client = _FakeClient(
            traces={"lf-1": _FakeTrace(2.5, {"first_text_response_time_ms": 800})},
            observations={"lf-1": [_gen_obs("AmapChatModel.chat", 100, 20, latency=1.0)]},
        )
        collector = TracePerfCollector(client=client)

        # A passed-in trace_id is treated as the primary id (the DB path already rewrote it); query by it directly.
        perf = (await collector.fetch_perf(["lf-1"]))["lf-1"]

        assert perf.e2e_ms == 2500.0  # latency (2.5s) × 1000, aligned with the script's definition
        assert perf.first_text_ms == 800.0
        assert perf.buckets["react"].calls == 1
        assert perf.error == ""

    async def test_uses_primary_id_and_generation_filter(self) -> None:
        """Use the passed-in trace_id as the primary id to query observations, and take GENERATION only."""
        client = _FakeClient(
            traces={"lf-xyz": _FakeTrace(1.0)},
            observations={"lf-xyz": []},
        )
        collector = TracePerfCollector(client=client)

        await collector.fetch_perf(["lf-xyz"])

        assert client.observation_calls[0]["primary_id"] == "lf-xyz"
        assert client.observation_calls[0]["type"] == "GENERATION"

    async def test_does_not_reresolve_primary_id(self) -> None:
        """Regression: on the DB path trace_id is already the primary id, and the collector must not call resolve_primary_id again.

        If resolve were called by mistake (treating the primary id as a business id and filtering metadata), the lookup
        would miss → the whole batch would be trace_not_found (the live failure where perf collected=0). This supplies a
        resolve_primary_id that maps the id wrongly: if it is called, get_trace receives the wrong id and finds nothing.
        Asserting that perf is still collected under the original id proves there was no second resolution.
        """

        class _PoisonResolveClient(_FakeClient):
            async def resolve_primary_id(self, trace_id: str) -> str:
                return "WRONG-" + trace_id  # If it is called, later lookups will miss

        client = _PoisonResolveClient(
            traces={"lf-1": _FakeTrace(2.0)},
            observations={"lf-1": []},
        )
        collector = TracePerfCollector(client=client)

        perf = (await collector.fetch_perf(["lf-1"]))["lf-1"]

        assert perf.error == ""
        assert perf.e2e_ms == 2000.0
        assert client.observation_calls[0]["primary_id"] == "lf-1"

    async def test_dict_shaped_trace_supported(self) -> None:
        """A trace in dict form exposes the same fields."""
        client = _FakeClient(
            traces={"biz-1": {"latency": 3.0, "metadata": {"first_text_response_time_ms": 900}}},
            observations={"biz-1": []},
        )
        collector = TracePerfCollector(client=client)

        perf = (await collector.fetch_perf(["biz-1"]))["biz-1"]

        assert perf.e2e_ms == 3000.0
        assert perf.first_text_ms == 900.0

    async def test_trace_not_found_marks_error(self) -> None:
        # Langfuse has no trace / observation for this primary id → not found.
        client = _FakeClient()
        collector = TracePerfCollector(client=client)

        perf = (await collector.fetch_perf(["missing-1"]))["missing-1"]

        assert perf.error == "trace_not_found"
        assert perf.e2e_ms is None

    async def test_missing_metadata_leaves_first_text_none(self) -> None:
        client = _FakeClient(
            traces={"biz-1": _FakeTrace(1.0, {})},
            observations={"biz-1": []},
        )
        collector = TracePerfCollector(client=client)

        perf = (await collector.fetch_perf(["biz-1"]))["biz-1"]

        assert perf.e2e_ms == 1000.0
        assert perf.first_text_ms is None

    async def test_null_latency_leaves_e2e_none(self) -> None:
        client = _FakeClient(traces={"biz-1": _FakeTrace(None)}, observations={"biz-1": []})
        collector = TracePerfCollector(client=client)

        perf = (await collector.fetch_perf(["biz-1"]))["biz-1"]

        assert perf.e2e_ms is None

    async def test_per_case_failure_isolated(self) -> None:
        """A single failure affects only that item; the rest are produced normally (AC-13 degradation)."""
        client = _FakeClient(
            traces={"ok-1": _FakeTrace(1.0)},
            observations={"ok-1": [_gen_obs("AmapChatModel.chat", 10, 2)]},
            raise_on={"bad-1"},
        )
        collector = TracePerfCollector(client=client, min_interval_s=0)

        result = await collector.fetch_perf(["ok-1", "bad-1"])

        assert result["ok-1"].error == ""
        assert result["ok-1"].e2e_ms == 1000.0
        assert result["bad-1"].error == "RuntimeError"

    async def test_timeout_marks_error_without_raising(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "benchmark.reporting.cluster_service.collector._PERF_TIMEOUT_S",
            0.01,
        )
        client = _FakeClient(
            traces={"slow-1": _FakeTrace(1.0)},
            delay_on={"slow-1"},
        )
        collector = TracePerfCollector(client=client)

        perf = (await collector.fetch_perf(["slow-1"]))["slow-1"]

        assert perf.error == "timeout"

    async def test_client_without_optional_methods_degrades(self) -> None:
        """When the client is missing a method, degrade to empty performance data and do not raise."""

        class _Bare:
            pass

        collector = TracePerfCollector(client=_Bare())

        perf = (await collector.fetch_perf(["biz-1"]))["biz-1"]

        assert isinstance(perf, CasePerf)
        assert perf.buckets == {}
        assert perf.e2e_ms is None

    async def test_concurrency_is_bounded(self) -> None:
        """A Semaphore limits concurrency so a large batch does not overwhelm Langfuse."""
        in_flight = 0
        peak = 0

        class _CountingClient(_FakeClient):
            async def list_observations_for_trace(self, primary_id, query):  # type: ignore[override]
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1
                return []

        client = _CountingClient(traces={f"t-{i}": _FakeTrace(1.0) for i in range(10)})
        collector = TracePerfCollector(client=client, concurrency=2, min_interval_s=0)

        await collector.fetch_perf([f"t-{i}" for i in range(10)])

        assert peak <= 2

    async def test_startup_pacing_spreads_requests(self) -> None:
        """T7 start throttle: adjacent per-trace fetches start at least min_interval_s apart."""
        client = _FakeClient(traces={f"t-{i}": _FakeTrace(1.0) for i in range(4)})
        collector = TracePerfCollector(client=client, concurrency=10, min_interval_s=0.05)

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        result = await collector.fetch_perf([f"t-{i}" for i in range(4)])

        assert len(result) == 4
        # 4 traces start 0.05s apart, so at least (4-1) × 0.05s is required.
        assert loop.time() - started_at >= 0.15

    async def test_zero_interval_disables_pacing(self) -> None:
        """min_interval_s=0 turns the throttle off and adds no extra wait."""
        client = _FakeClient(traces={f"t-{i}": _FakeTrace(1.0) for i in range(4)})
        collector = TracePerfCollector(client=client, concurrency=10, min_interval_s=0)

        loop = asyncio.get_running_loop()
        started_at = loop.time()
        result = await collector.fetch_perf([f"t-{i}" for i in range(4)])

        assert len(result) == 4
        assert loop.time() - started_at < 0.1

    async def test_empty_trace_id_list(self) -> None:
        collector = TracePerfCollector(client=_FakeClient())
        assert await collector.fetch_perf([]) == {}

    @pytest.mark.parametrize("concurrency", [0, -1])
    async def test_concurrency_floor_is_one(self, concurrency: int) -> None:
        """An illegal concurrency value falls back to 1; Semaphore(0) must not be constructed or it deadlocks."""
        client = _FakeClient(traces={"t-1": _FakeTrace(1.0)}, observations={"t-1": []})
        collector = TracePerfCollector(client=client, concurrency=concurrency)

        result = await collector.fetch_perf(["t-1"])

        assert result["t-1"].e2e_ms == 1000.0
