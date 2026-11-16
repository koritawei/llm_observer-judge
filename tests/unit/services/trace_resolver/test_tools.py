"""Unit tests for TraceToolProvider (an AsyncMock fake_resolver, no real Langfuse)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.trace import ObservationDetail, TraceSkeletonResult, TraceSpanSkeleton
from benchmark.trace.http_backend import TraceHttpError
from benchmark.trace.tools import (
    GenerationMessageRange,
    TraceRequestBound,
    TraceToolProvider,
)

# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_resolver() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def provider(fake_resolver: AsyncMock) -> TraceToolProvider:
    return TraceToolProvider(fake_resolver)


PRIMARY_ID = "lf-primary-123"


@pytest.fixture
def bound() -> TraceRequestBound:
    """Bound already holds Langfuse primary id (post entry rewrite)."""
    return TraceRequestBound(trace_id=PRIMARY_ID, project_id="proj-7", primary_id=PRIMARY_ID)


def _make_skeleton(
    *,
    primary_id: str = "lf-primary-123",
    trace_id: str = PRIMARY_ID,
    project_id: str = "proj-7",
    spans: list[TraceSpanSkeleton] | None = None,
) -> TraceSkeletonResult:
    return TraceSkeletonResult(
        primary_id=primary_id,
        business_trace_id=trace_id,
        project_id=project_id,
        spans=spans or [],
    )


def _make_obs(
    *,
    obs_id: str = "obs-1",
    trace_id: str = "lf-primary-123",
) -> ObservationDetail:
    return ObservationDetail(id=obs_id, trace_id=trace_id, name="llm-call")


# ── 1. list_tools ────────────────────────────────────────────────────────────


async def test_list_tools_exposes_three_names(provider: TraceToolProvider) -> None:
    tools = await provider.list_tools()
    names = {t["name"] for t in tools}
    assert names == {"fetch_trace_skeleton", "fetch_observation_detail", "find_observations"}


@pytest.mark.asyncio
async def test_list_tools_default_excludes_fetch_full_trace(
    provider: TraceToolProvider,
) -> None:
    tools = await provider.list_tools()
    names = {t["name"] for t in tools}
    assert names == {
        "fetch_trace_skeleton",
        "fetch_observation_detail",
        "find_observations",
    }
    assert "fetch_full_trace" not in names


@pytest.mark.asyncio
async def test_list_tools_includes_fetch_full_trace_when_enabled(
    fake_resolver: AsyncMock,
) -> None:
    provider = TraceToolProvider(fake_resolver, enable_fetch_full_trace=True)
    names = {t["name"] for t in await provider.list_tools()}
    assert "fetch_full_trace" in names


@pytest.mark.asyncio
async def test_fetch_full_trace_disabled_errors_without_resolver_call(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    result = await provider.call_tool("fetch_full_trace", {}, bound=bound)
    assert not result.success
    fake_resolver.fetch_full_trace.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_full_trace_returns_raw_payload_without_char_cap(
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    from benchmark.trace.schema import TraceDocument

    huge = "x" * 50_000
    fake_resolver.fetch_full_trace.return_value = TraceDocument.model_validate(
        {
            "id": PRIMARY_ID,
            "observations": [{"id": "obs-1", "input": huge}],
            "scores": [],
        }
    )
    provider = TraceToolProvider(
        fake_resolver,
        max_observation_chars=100,
        enable_fetch_full_trace=True,
    )
    result = await provider.call_tool("fetch_full_trace", {}, bound=bound)
    assert result.success
    body = result.to_text()
    assert huge in body
    assert "body_truncated" not in body
    fake_resolver.fetch_full_trace.assert_called_once_with(
        bound.project_id,
        PRIMARY_ID,
        fields=None,
    )


@pytest.mark.asyncio
async def test_fetch_full_trace_passes_fields_arg(
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    from benchmark.trace.schema import TraceDocument

    fake_resolver.fetch_full_trace.return_value = TraceDocument(id=PRIMARY_ID)
    provider = TraceToolProvider(fake_resolver, enable_fetch_full_trace=True)
    await provider.call_tool("fetch_full_trace", {"fields": "core"}, bound=bound)
    fake_resolver.fetch_full_trace.assert_called_once_with(
        bound.project_id,
        PRIMARY_ID,
        fields="core",
    )


@pytest.mark.asyncio
async def test_from_config_reads_enable_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from benchmark.libs.agenthost.tools.component_metadata import ToolProviderContext
    from benchmark.settings import Settings
    from benchmark.trace.schema import TraceEndpoint
    from benchmark.trace.tools import TraceToolExt, TraceToolProvider
    from pydantic import SecretStr

    inst = TraceEndpoint(
        base_url="http://lf",
        public_key="pk",
        secret_key=SecretStr("sk"),
    )
    monkeypatch.setattr(
        "benchmark.settings.load_trace_instances",
        lambda _encoded: {"proj": inst},
    )
    runtime = SimpleNamespace(settings=Settings(trace_instances="x"))
    ctx = ToolProviderContext(service_root=Path.cwd(), runtime=runtime)
    provider = TraceToolProvider.from_config(TraceToolExt(enable_fetch_full_trace=True), ctx)
    assert provider is not None
    assert provider._enable_fetch_full_trace is True


# ── 2. skeleton uses bound ids ───────────────────────────────────────────────


async def test_fetch_skeleton_uses_bound_ids(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_skeleton.return_value = _make_skeleton()

    result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)

    assert result.success
    fake_resolver.fetch_skeleton.assert_called_once_with(
        PRIMARY_ID, "proj-7", max_spans=300, preview_chars=180
    )
    assert bound.primary_id == PRIMARY_ID


async def test_fetch_skeleton_stores_primary_id_on_success(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    bound = TraceRequestBound(trace_id="lf-xyz", project_id="proj-7", primary_id="lf-xyz")
    fake_resolver.fetch_skeleton.return_value = _make_skeleton(
        primary_id="lf-xyz", trace_id="lf-xyz"
    )

    await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)

    assert bound.primary_id == "lf-xyz"


async def test_fetch_skeleton_unknown_project_returns_error(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_skeleton.return_value = None

    result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)

    assert not result.success
    assert result.error is not None
    assert "unknown project_id" in result.error


async def test_fetch_skeleton_observations_failure_returns_error(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    from benchmark.trace.http_backend import TraceObservationsError

    fake_resolver.fetch_skeleton.side_effect = TraceObservationsError(
        "observations request failed (status=500)",
        status_code=500,
    )

    result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)

    assert not result.success
    assert result.error is not None
    assert "observations" in result.error.lower()
    assert bound.primary_id == PRIMARY_ID


async def test_find_observations_uses_bound_primary_id(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.find_observations.return_value = []

    result = await provider.call_tool("find_observations", {"name": "planner"}, bound=bound)

    assert result.success
    fake_resolver.find_observations.assert_awaited_once()
    kwargs = fake_resolver.find_observations.await_args.kwargs
    assert kwargs["primary_id"] == PRIMARY_ID
    fake_resolver.resolve_primary_id_for.assert_not_awaited()


async def test_find_observations_list_failure_returns_error(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    from benchmark.trace.http_backend import TraceObservationsError

    bound.primary_id = PRIMARY_ID
    fake_resolver.find_observations.side_effect = TraceObservationsError(
        "observations request failed (status=401)",
        status_code=401,
    )

    result = await provider.call_tool("find_observations", {"name": "planner"}, bound=bound)

    assert not result.success
    assert result.error is not None
    assert "observations" in result.error.lower()


async def test_fetch_observation_detail_http_error_not_masqueraded_as_not_found(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    from benchmark.trace.http_backend import TraceHttpError

    fake_resolver.fetch_observation.side_effect = TraceHttpError(
        "observation detail request failed (status=500)",
        status_code=500,
    )

    result = await provider.call_tool(
        "fetch_observation_detail",
        {"observation_id": "obs-1"},
        bound=bound,
    )

    assert not result.success
    assert result.error is not None
    assert "status=500" in result.error
    assert "not found" not in result.error.lower()


# ── 3. missing bound → error ──────────────────────────────────────────────────


async def test_call_tool_without_bound_returns_error(
    provider: TraceToolProvider,
) -> None:
    result = await provider.call_tool("fetch_trace_skeleton", {})

    assert not result.success
    assert result.error is not None
    assert "bound" in result.error.lower()


async def test_call_observation_detail_without_bound_returns_error(
    provider: TraceToolProvider,
) -> None:
    result = await provider.call_tool("fetch_observation_detail", {"observation_id": "obs-1"})

    assert not result.success
    assert result.error is not None


# ── 4. observation detail cached ─────────────────────────────────────────────


async def test_observation_detail_cached_second_call_skips_resolver(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.return_value = _make_obs()

    result1 = await provider.call_tool(
        "fetch_observation_detail", {"observation_id": "obs-1"}, bound=bound
    )
    result2 = await provider.call_tool(
        "fetch_observation_detail", {"observation_id": "obs-1"}, bound=bound
    )

    assert result1.success
    assert result2.success
    fake_resolver.fetch_observation.assert_called_once()
    fake_resolver.resolve_primary_id_for.assert_not_awaited()
    assert bound.primary_id == PRIMARY_ID


async def test_observation_detail_different_ids_each_fetched_once(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.side_effect = [
        _make_obs(obs_id="obs-1"),
        _make_obs(obs_id="obs-2"),
    ]

    await provider.call_tool("fetch_observation_detail", {"observation_id": "obs-1"}, bound=bound)
    await provider.call_tool("fetch_observation_detail", {"observation_id": "obs-2"}, bound=bound)
    await provider.call_tool("fetch_observation_detail", {"observation_id": "obs-1"}, bound=bound)

    assert fake_resolver.fetch_observation.call_count == 2
    fake_resolver.resolve_primary_id_for.assert_not_awaited()


# ── bound_from_metadata ───────────────────────────────────────────────────────


async def test_bound_from_metadata_extracts_ids(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    fake_resolver.fetch_skeleton.return_value = _make_skeleton(trace_id="t-42", project_id="p-9")
    bound = provider.bound_from_metadata({"trace_id": "t-42", "project_id": "p-9"})
    assert bound.primary_id == "t-42"

    result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)

    assert result.success
    fake_resolver.fetch_skeleton.assert_called_once_with(
        "t-42", "p-9", max_spans=300, preview_chars=180
    )


async def test_bound_from_metadata_trace_id_override(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    fake_resolver.fetch_skeleton.return_value = _make_skeleton(trace_id="override-id")
    bound = provider.bound_from_metadata(
        {"trace_id": "meta-id", "project_id": "proj-7"},
        trace_id="override-id",
    )

    await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)

    fake_resolver.fetch_skeleton.assert_called_once_with(
        "override-id", "proj-7", max_spans=300, preview_chars=180
    )


async def test_bound_from_metadata_nested_evaluator_request(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    fake_resolver.fetch_skeleton.return_value = _make_skeleton(
        trace_id="er-tid", project_id="er-pid"
    )
    bound = provider.bound_from_metadata(
        {
            "evaluator_request": {
                "trace_id": "er-tid",
                "extra": {"project_id": "er-pid"},
            }
        }
    )

    await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)

    fake_resolver.fetch_skeleton.assert_called_once_with(
        "er-tid", "er-pid", max_spans=300, preview_chars=180
    )


async def test_bound_from_metadata_missing_project_raises(
    provider: TraceToolProvider,
) -> None:
    with pytest.raises(ValueError, match="project_id"):
        provider.bound_from_metadata({"trace_id": "t-1"})


# ── health_check ─────────────────────────────────────────────────────────────


async def test_health_check_returns_true(provider: TraceToolProvider) -> None:
    assert await provider.health_check() is True


# ── find_observations ─────────────────────────────────────────────────────────


async def test_find_observations_rejects_missing_filters(
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    """Unfiltered listing belongs to skeleton — find requires name/type/status."""

    bound.primary_id = PRIMARY_ID
    result = await provider.call_tool("find_observations", {}, bound=bound)
    assert not result.success
    assert "name/type/status" in (result.error or "")


async def test_find_observations_uses_primary_id_when_cached(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.find_observations.return_value = []

    result = await provider.call_tool("find_observations", {"name": "planner"}, bound=bound)

    assert result.success
    fake_resolver.find_observations.assert_called_once_with(
        "proj-7",
        primary_id=PRIMARY_ID,
        name="planner",
        type=None,
        status=None,
    )
    fake_resolver.resolve_primary_id_for.assert_not_awaited()


async def test_find_observations_falls_back_to_bound_trace_id(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    bound = TraceRequestBound(trace_id=PRIMARY_ID, project_id="proj-7")
    fake_resolver.find_observations.return_value = []

    result = await provider.call_tool("find_observations", {"type": "GENERATION"}, bound=bound)

    assert result.success
    fake_resolver.find_observations.assert_called_once_with(
        "proj-7",
        primary_id=PRIMARY_ID,
        name=None,
        type="GENERATION",
        status=None,
    )
    assert bound.primary_id == PRIMARY_ID
    fake_resolver.resolve_primary_id_for.assert_not_awaited()


async def test_find_observations_reuses_primary_id_across_calls(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.find_observations.return_value = []

    await provider.call_tool("find_observations", {"name": "a"}, bound=bound)
    await provider.call_tool("find_observations", {"name": "planner"}, bound=bound)

    fake_resolver.resolve_primary_id_for.assert_not_awaited()
    assert fake_resolver.find_observations.await_count == 2
    second_kwargs = fake_resolver.find_observations.await_args_list[1].kwargs
    assert second_kwargs["primary_id"] == PRIMARY_ID


# ── trace attribution enforcement (bug #4) ────────────────────────────────────


async def test_detail_without_prior_skeleton_validates_against_bound_primary(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.return_value = ObservationDetail(
        id="obs-1", trace_id=PRIMARY_ID, name="llm-call"
    )

    result = await provider.call_tool(
        "fetch_observation_detail", {"observation_id": "obs-1"}, bound=bound
    )

    assert result.success
    fake_resolver.resolve_primary_id_for.assert_not_awaited()
    assert bound.primary_id == PRIMARY_ID


async def test_detail_rejects_observation_from_other_trace(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.return_value = ObservationDetail(
        id="obs-leak", trace_id="lf-other-trace", name="planner"
    )

    result = await provider.call_tool(
        "fetch_observation_detail", {"observation_id": "obs-leak"}, bound=bound
    )

    assert not result.success
    assert "belongs to trace 'lf-other-trace'" in (result.error or "")
    assert "expected 'lf-primary-123'" in (result.error or "")
    assert "obs-leak" not in bound.detail_cache


async def test_detail_rejects_when_observation_omits_trace_id(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.return_value = ObservationDetail(
        id="obs-no-trace", trace_id=None, name="planner"
    )

    result = await provider.call_tool(
        "fetch_observation_detail", {"observation_id": "obs-no-trace"}, bound=bound
    )

    assert not result.success
    assert "cannot validate observation attribution" in (result.error or "")
    assert "has no trace_id" in (result.error or "")
    assert "obs-no-trace" not in bound.detail_cache


async def test_detail_succeeds_after_skeleton_keeps_primary_id(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_skeleton.return_value = _make_skeleton(primary_id=PRIMARY_ID)
    fake_resolver.fetch_observation.return_value = _make_obs()

    skeleton_result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)
    detail_result = await provider.call_tool(
        "fetch_observation_detail", {"observation_id": "obs-1"}, bound=bound
    )

    assert skeleton_result.success
    assert detail_result.success
    fake_resolver.resolve_primary_id_for.assert_not_awaited()
    fake_resolver.fetch_observation.assert_awaited_once()


# ── response body size cap (bug #10) ──────────────────────────────────────────


async def test_observation_detail_body_truncated_at_provider_cap(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    # bug #10: an oversized detail body is truncated to the provider cap
    cap = 512
    provider = TraceToolProvider(fake_resolver, max_observation_chars=cap)
    huge_input = "X" * (cap * 4)  # 4x over the cap
    fake_resolver.fetch_observation.return_value = ObservationDetail(
        id="obs-1",
        trace_id=PRIMARY_ID,
        name="llm-call",
        input={"messages": [{"role": "user", "content": huge_input}]},
        output={"text": huge_input},
    )

    result = await provider.call_tool(
        "fetch_observation_detail", {"observation_id": "obs-1"}, bound=bound
    )

    assert result.success
    body = result.to_text()
    parsed = json.loads(body)
    assert len(body) <= cap or parsed.get("body_truncated") is True
    assert parsed.get("body_truncated") is True
    assert "obs-1" in bound.known_refs


async def test_skeleton_body_truncated_at_provider_cap(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    cap = 600
    provider = TraceToolProvider(fake_resolver, max_observation_chars=cap)
    # Several spans with a large preview → the skeleton body is truncated
    spans = [
        TraceSpanSkeleton(
            id=f"obs-{i}",
            name=f"step-{i}",
            type="SPAN",
            input_preview="Y" * 200,
            output_preview="Z" * 200,
        )
        for i in range(20)
    ]
    fake_resolver.fetch_skeleton.return_value = _make_skeleton(spans=spans)

    result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)

    assert result.success
    body = result.to_text()
    parsed = json.loads(body)
    assert parsed.get("body_truncated") is True
    emitted = {s["id"] for s in parsed["spans"]}
    assert emitted == bound.known_refs
    assert len(emitted) < 20
    assert len(body) <= cap


async def test_skeleton_known_refs_match_emitted_span_ids(
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    """known_refs must equal ids in the (possibly truncated) body — never more."""

    provider = TraceToolProvider(fake_resolver, max_observation_chars=400)
    spans = [
        TraceSpanSkeleton(
            id=f"obs-{i}",
            name=f"step-{i}",
            input_preview="P" * 80,
            output_preview="Q" * 80,
        )
        for i in range(10)
    ]
    fake_resolver.fetch_skeleton.return_value = _make_skeleton(spans=spans)

    result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)
    parsed = json.loads(result.to_text())
    assert {s["id"] for s in parsed["spans"]} == bound.known_refs
    assert "lf-primary-123" not in bound.known_refs  # primary_id is not evidence


@pytest.mark.asyncio
async def test_close_aliases_aclose(fake_resolver: AsyncMock) -> None:
    """AppRuntime.shutdown calls ``close()``; it must close the TraceResolver httpx client."""

    provider = TraceToolProvider(fake_resolver)
    await provider.close()
    fake_resolver.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_call_tool_keeps_bound_isolation(
    fake_resolver: AsyncMock,
) -> None:
    """A singleton provider + an explicit bound: concurrent requests must not cross trace_id / primary_id."""

    provider = TraceToolProvider(fake_resolver)

    async def _skeleton(trace_id: str, project_id: str, **kwargs: object) -> TraceSkeletonResult:
        await asyncio.sleep(0.02)
        return _make_skeleton(
            primary_id=f"lf-{trace_id}",
            trace_id=trace_id,
            project_id=project_id,
        )

    fake_resolver.fetch_skeleton.side_effect = _skeleton
    bound_a = provider.create_bound(trace_id="trace-A", project_id="proj")
    bound_b = provider.create_bound(trace_id="trace-B", project_id="proj")

    result_a, result_b = await asyncio.gather(
        provider.call_tool("fetch_trace_skeleton", {}, bound=bound_a),
        provider.call_tool("fetch_trace_skeleton", {}, bound=bound_b),
    )

    assert result_a.success
    assert result_b.success
    assert bound_a.primary_id == "lf-trace-A"
    assert bound_b.primary_id == "lf-trace-B"
    assert not hasattr(provider, "_bound")


# ── CR fixes: ContextVar / args validation / cache max_chars / fit fallback ──


@pytest.mark.asyncio
async def test_call_tool_via_bind_trace_request_contextvar(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    from benchmark.trace.tools import bind_trace_request

    fake_resolver.fetch_skeleton.return_value = _make_skeleton()
    bound = TraceRequestBound(trace_id=PRIMARY_ID, project_id="proj-7", primary_id=PRIMARY_ID)
    with bind_trace_request(bound):
        result = await provider.call_tool("fetch_trace_skeleton", {})
    assert result.success
    fake_resolver.fetch_skeleton.assert_called_once_with(
        PRIMARY_ID, "proj-7", max_spans=300, preview_chars=180
    )


@pytest.mark.asyncio
async def test_tool_manager_call_tool_with_bound_contextvar(
    fake_resolver: AsyncMock,
) -> None:
    """Standard ToolManager.call_tool path: list tools, then call under bind."""

    from benchmark.libs.agenthost.tools.manager import ToolManager
    from benchmark.trace.tools import bind_trace_request

    fake_resolver.fetch_skeleton.return_value = _make_skeleton()
    provider = TraceToolProvider(fake_resolver)
    manager = ToolManager()
    manager.register_provider(provider)
    await manager.list_tools()

    bound = TraceRequestBound(trace_id=PRIMARY_ID, project_id="proj-7", primary_id=PRIMARY_ID)
    with bind_trace_request(bound):
        result = await manager.call_tool("fetch_trace_skeleton", {})
    assert result.success
    assert bound.primary_id == PRIMARY_ID


@pytest.mark.asyncio
async def test_invalid_skeleton_args_return_tool_error_not_raise(
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    result = await provider.call_tool(
        "fetch_trace_skeleton",
        {"max_spans": "many"},
        bound=bound,
    )
    assert not result.success
    assert "invalid" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_negative_max_spans_rejected(
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    result = await provider.call_tool(
        "fetch_trace_skeleton",
        {"max_spans": -1},
        bound=bound,
    )
    assert not result.success


@pytest.mark.asyncio
async def test_detail_cache_preserves_full_and_applies_max_chars_per_call(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    full = ObservationDetail(
        id="obs-1",
        trace_id=PRIMARY_ID,
        name="llm-call",
        input="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        output="0123456789",
    )
    fake_resolver.fetch_observation.return_value = full

    r1 = await provider.call_tool(
        "fetch_observation_detail",
        {"observation_id": "obs-1", "max_chars": 5},
        bound=bound,
    )
    assert r1.success
    body1 = json.loads(r1.to_text())
    assert isinstance(body1.get("input"), str)
    assert body1["input"].startswith("ABCDE")

    r2 = await provider.call_tool(
        "fetch_observation_detail",
        {"observation_id": "obs-1"},
        bound=bound,
    )
    assert r2.success
    body2 = json.loads(r2.to_text())
    assert body2.get("input") == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    fake_resolver.fetch_observation.assert_called_once()
    assert fake_resolver.fetch_observation.await_args.kwargs.get("max_chars") is None


@pytest.mark.asyncio
async def test_fit_detail_falls_back_to_minimal_stable_fields(
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    cap = 120
    provider = TraceToolProvider(fake_resolver, max_observation_chars=cap)
    fake_resolver.fetch_observation.return_value = ObservationDetail(
        id="obs-1",
        trace_id=PRIMARY_ID,
        name="n" * 80,
        type="GENERATION",
        metadata={"blob": "M" * 500},
        status_message="S" * 500,
    )

    result = await provider.call_tool(
        "fetch_observation_detail",
        {"observation_id": "obs-1"},
        bound=bound,
    )
    assert result.success
    body = result.to_text()
    parsed = json.loads(body)
    assert len(body) <= cap
    assert parsed.get("body_truncated") is True
    assert parsed.get("id") == "obs-1"
    assert "metadata" not in parsed


@pytest.mark.asyncio
async def test_from_config_prefers_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from benchmark.libs.agenthost.tools.component_metadata import ToolProviderContext
    from benchmark.settings import Settings
    from benchmark.trace.tools import TraceToolExt, TraceToolProvider

    custom = Settings(trace_instances="from-runtime")
    runtime = SimpleNamespace(settings=custom)
    ctx = ToolProviderContext(service_root=Path.cwd(), runtime=runtime)

    called: dict[str, str] = {}

    def _load(encoded: str) -> dict:
        called["encoded"] = encoded
        return {}

    monkeypatch.setattr("benchmark.settings.load_trace_instances", _load)

    out = TraceToolProvider.from_config(TraceToolExt(), ctx)
    assert out is None
    assert called.get("encoded") == "from-runtime"


@pytest.mark.asyncio
async def test_from_config_passes_timeout_and_lookback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from benchmark.libs.agenthost.tools.component_metadata import ToolProviderContext
    from benchmark.settings import Settings
    from benchmark.trace.schema import TraceEndpoint
    from benchmark.trace.tools import TraceToolExt, TraceToolProvider
    from pydantic import SecretStr

    inst = TraceEndpoint(
        name="amap",
        base_url="https://lf.example.com",
        public_key="pk",
        secret_key=SecretStr("sk"),
    )
    monkeypatch.setattr(
        "benchmark.settings.load_trace_instances",
        lambda _encoded: {"proj": inst},
    )
    captured: dict[str, object] = {}

    class _FakeResolver:
        def __init__(self, instances: object, **kwargs: object) -> None:
            captured.update(kwargs)
            self.instances = instances

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "benchmark.trace.tools.TraceResolver",
        _FakeResolver,
    )

    runtime = SimpleNamespace(settings=Settings(trace_instances="x"))
    ctx = ToolProviderContext(service_root=Path.cwd(), runtime=runtime)
    provider = TraceToolProvider.from_config(
        TraceToolExt(
            timeout_seconds=45,
            max_retries=1,
            resolve_lookback_days=7,
        ),
        ctx,
    )
    assert provider is not None
    assert captured["timeout_seconds"] == 45
    assert captured["max_retries"] == 1
    assert captured["resolve_lookback_days"] == 7
    await provider.aclose()


# ── report-credential SPAN (LANGFUSE_* , not INSTANCES) ─────────────────────
# Keep BEFORE importlib.reload tests — reload mutates module identity mid-suite.


def test_start_trace_tool_span_skips_without_active_parent_trace(
    provider: TraceToolProvider,
    bound: TraceRequestBound,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No parent workflow trace → skip; must not open a second root."""

    class _FakeClient:
        def get_current_trace_id(self) -> None:
            return None

        def start_observation(self, **_kwargs: object) -> object:
            raise AssertionError("must not start observation without parent")

    monkeypatch.setattr(
        "benchmark.utils.trace_report.get_report_client",
        _FakeClient,
    )
    assert provider._start_trace_tool_span("fetch_trace_skeleton", {}, bound) is None


def test_start_trace_tool_span_nests_under_get_client_parent(
    provider: TraceToolProvider,
    bound: TraceRequestBound,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse get_report_client() + start_observation so SPAN nests under Middleware root."""

    from contextlib import contextmanager

    captured: dict[str, object] = {}
    node_parent_used = {"entered": False}

    class _FakeSpan:
        def update(self, **kwargs: object) -> None:
            captured["update_kwargs"] = kwargs

        def end(self) -> None:
            captured["ended"] = True

    class _FakeClient:
        def get_current_trace_id(self) -> str:
            return "parent-trace-1"

        def start_observation(self, **kwargs: object) -> _FakeSpan:
            assert node_parent_used["entered"] is True
            captured["observation_kwargs"] = kwargs
            return _FakeSpan()

    @contextmanager
    def fake_node_parent() -> object:
        node_parent_used["entered"] = True
        yield

    monkeypatch.setattr(
        "benchmark.utils.trace_report.get_report_client",
        _FakeClient,
    )
    monkeypatch.setattr(
        "benchmark.utils.trace_report.with_current_node_otel_parent",
        fake_node_parent,
    )

    span_info = provider._start_trace_tool_span(
        "fetch_trace_skeleton",
        {"max_spans": 10},
        bound,
    )
    assert span_info is not None
    span_context, span = span_info
    assert span_context is None
    assert span is not None
    obs = captured["observation_kwargs"]
    assert isinstance(obs, dict)
    assert obs["name"] == "trace.fetch_trace_skeleton"
    assert obs["as_type"] == "span"
    assert obs["input"]["bound"] == {
        "trace_id": PRIMARY_ID,
        "project_id": "proj-7",
    }

    provider._finish_trace_tool_span(
        span_context,
        span,
        result=type("R", (), {"success": True, "body": '{"name":"全聚德"}'})(),
    )
    assert captured["update_kwargs"] == {
        "output": {
            "success": True,
            "data": {"name": "全聚德"},
            "body_chars": 14,
            "body_truncated": False,
        }
    }
    assert captured.get("ended") is True


def test_finish_trace_tool_span_truncates_large_result(
    fake_resolver: AsyncMock,
) -> None:
    provider = TraceToolProvider(fake_resolver, max_observation_chars=5)
    span = MagicMock()

    provider._finish_trace_tool_span(
        None,
        span,
        result=type("R", (), {"success": True, "body": "123456789"})(),
    )

    span.update.assert_called_once_with(
        output={
            "success": True,
            "data": "12345…[truncated]",
            "body_chars": 9,
            "body_truncated": True,
        }
    )
    span.end.assert_called_once_with()


@pytest.mark.asyncio
async def test_call_tool_still_reads_via_resolver_when_span_skipped(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing parent report trace must not block read path through resolver."""

    class _FakeClient:
        def get_current_trace_id(self) -> None:
            return None

    monkeypatch.setattr(
        "benchmark.utils.trace_report.get_report_client",
        _FakeClient,
    )
    fake_resolver.fetch_skeleton.return_value = _make_skeleton()
    result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)
    assert result.success
    fake_resolver.fetch_skeleton.assert_awaited_once()


# ── sidecar local tree (no Langfuse HTTP) ─────────────────────────────────────

_SIDECAR_OBS = [
    {
        "span_id": "tool-1",
        "parent_span_id": None,
        "name": "search_poi",
        "type": "TOOL",
        "level": "DEFAULT",
        "input": {"keywords": "咖啡"},
        "output": {"name": "星巴克"},
        "start_time": 1724486400000,
    },
    {
        "span_id": "gen-1",
        "parent_span_id": "tool-1",
        "name": "AmapChatModel.chat",
        "type": "GENERATION",
        "level": "DEFAULT",
        "input": [{"role": "user", "content": "附近咖啡"}],
        "output": {"text": "去星巴克"},
        "start_time": 1724486400100,
    },
]


def _sidecar_bound() -> TraceRequestBound:
    return TraceRequestBound(
        trace_id=PRIMARY_ID,
        project_id="proj-7",
        primary_id=PRIMARY_ID,
        sidecar_observations=_SIDECAR_OBS,
    )


async def test_bound_from_metadata_copies_sidecar_observations(
    provider: TraceToolProvider,
) -> None:
    bound = provider.bound_from_metadata(
        {
            "trace_id": PRIMARY_ID,
            "project_id": "proj-7",
            "sidecar_observations": _SIDECAR_OBS,
        }
    )
    assert bound.sidecar_observations == _SIDECAR_OBS


async def test_bound_from_metadata_nested_evaluator_sidecar_observations(
    provider: TraceToolProvider,
) -> None:
    bound = provider.bound_from_metadata(
        {
            "evaluator_request": {
                "trace_id": PRIMARY_ID,
                "project_id": "proj-7",
                "metadata": {"sidecar_observations": _SIDECAR_OBS},
            }
        }
    )
    assert bound.sidecar_observations == _SIDECAR_OBS


async def test_create_bound_copies_sidecar_contextvar(
    provider: TraceToolProvider,
) -> None:
    from benchmark.trace.tools import bind_sidecar_observations

    with bind_sidecar_observations(_SIDECAR_OBS):
        bound = provider.create_bound(trace_id=PRIMARY_ID, project_id="proj-7")
    assert bound.sidecar_observations == _SIDECAR_OBS


async def test_sidecar_skeleton_does_not_call_trace(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    result = await provider.call_tool("fetch_trace_skeleton", {}, bound=_sidecar_bound())
    assert result.success
    body = json.loads(result.to_text())
    ids = {span["id"] for span in body["spans"]}
    assert ids == {"tool-1", "gen-1"}
    fake_resolver.fetch_skeleton.assert_not_called()


async def test_sidecar_detail_does_not_call_trace(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    result = await provider.call_tool(
        "fetch_observation_detail",
        {"observation_id": "gen-1"},
        bound=_sidecar_bound(),
    )
    assert result.success
    detail = json.loads(result.to_text())
    assert detail["id"] == "gen-1"
    assert detail["trace_id"] == PRIMARY_ID
    assert detail["name"] == "AmapChatModel.chat"
    fake_resolver.fetch_observation.assert_not_called()


async def test_sidecar_detail_miss_does_not_fallback_to_trace(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    result = await provider.call_tool(
        "fetch_observation_detail",
        {"observation_id": "missing-obs"},
        bound=_sidecar_bound(),
    )
    assert not result.success
    assert "missing-obs" in (result.error or "")
    fake_resolver.fetch_observation.assert_not_called()


async def test_sidecar_find_filters_locally(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    result = await provider.call_tool(
        "find_observations",
        {"type": "GENERATION"},
        bound=_sidecar_bound(),
    )
    assert result.success
    spans = json.loads(result.to_text())
    assert [span["id"] for span in spans] == ["gen-1"]
    fake_resolver.find_observations.assert_not_called()


async def test_sidecar_full_trace_works_when_disabled(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    result = await provider.call_tool("fetch_full_trace", {}, bound=_sidecar_bound())
    assert result.success
    payload = json.loads(result.to_text())
    assert payload["id"] == PRIMARY_ID
    obs_ids = {item.get("id") or item.get("span_id") for item in payload["observations"]}
    assert obs_ids == {"tool-1", "gen-1"}
    fake_resolver.fetch_full_trace.assert_not_called()


async def test_sidecar_contextvar_serves_even_without_bound_copy(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
) -> None:
    from benchmark.trace.tools import bind_sidecar_observations

    bound = TraceRequestBound(trace_id=PRIMARY_ID, project_id="proj-7", primary_id=PRIMARY_ID)
    with bind_sidecar_observations(_SIDECAR_OBS):
        result = await provider.call_tool("fetch_trace_skeleton", {}, bound=bound)
    assert result.success
    fake_resolver.fetch_skeleton.assert_not_called()


def test_tool_component_adapter_registered_on_import() -> None:
    """Importing tools.py must register tool metadata adapter (no manual re-register)."""

    import importlib

    from benchmark.libs.agenthost.app.providers import ProviderRegistry
    from benchmark.trace import tools as tools_mod

    # Simulate reset wiping adapters; reload must restore register side-effect.
    ProviderRegistry._metadata_adapters.pop("tool", None)
    importlib.reload(tools_mod)

    assert "tool" in ProviderRegistry._metadata_adapters
    assert hasattr(tools_mod, "_build_tool_component_provider")
    coerced = ProviderRegistry._coerce_to_component_provider(tools_mod.TraceToolProvider)
    assert coerced.kind == "tool"
    assert coerced.name == "trace"


# ── generation snapshot Python API (Final-Chat-First lossless read)────────────


def _gen_detail(
    *,
    obs_id: str = "obs-gen-1",
    input_value: object = None,
    output: object = None,
    trace_id: str = PRIMARY_ID,
) -> ObservationDetail:
    return ObservationDetail(
        id=obs_id,
        trace_id=trace_id,
        name="AmapChatModel.chat",
        type="GENERATION",
        input=input_value,
        output=output,
        level="DEFAULT",
        startTime="2026-01-01T00:00:00Z",
        endTime="2026-01-01T00:00:05Z",
        parentObservationId="parent-chat",
    )


@pytest.mark.asyncio
async def test_find_generation_snapshots_stable_sorted_and_generation_only(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.find_observations.return_value = [
        TraceSpanSkeleton(id="b", start_time="2026-01-01T00:01:00Z"),
        TraceSpanSkeleton(id="a", start_time="2026-01-01T00:00:00Z"),
        TraceSpanSkeleton(id="c", start_time="2026-01-01T00:01:00Z"),
    ]

    spans = await provider.find_generation_snapshots(bound, name="AmapChatModel.chat")

    assert [span.id for span in spans] == ["a", "b", "c"]
    fake_resolver.find_observations.assert_awaited_once_with(
        "proj-7",
        primary_id=PRIMARY_ID,
        name="AmapChatModel.chat",
        type="GENERATION",
        status=None,
    )


@pytest.mark.asyncio
async def test_fetch_generation_snapshot_missing_returns_incomplete_not_raises(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.return_value = None

    snapshot = await provider.fetch_generation_snapshot(bound, "obs-gone")

    assert snapshot.evidence_incomplete is True
    assert snapshot.incomplete_reasons == ["missing_generation"]
    assert "obs-gone" not in bound.known_refs


@pytest.mark.asyncio
async def test_fetch_generation_snapshot_cross_trace_rejected(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.return_value = _gen_detail(
        obs_id="obs-other", trace_id="another-trace"
    )

    snapshot = await provider.fetch_generation_snapshot(bound, "obs-other")

    assert snapshot.evidence_incomplete is True
    assert snapshot.incomplete_reasons == ["cross_trace_rejected"]


@pytest.mark.asyncio
async def test_fetch_generation_snapshot_full_read_marks_complete(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    messages = [
        {"role": "system", "content": "final chat prompt"},
        {"role": "user", "content": "query"},
    ]
    fake_resolver.fetch_observation.return_value = _gen_detail(
        input_value=messages, output="最终回复"
    )

    snapshot = await provider.fetch_generation_snapshot(bound, "obs-gen-1")

    assert snapshot.evidence_incomplete is False
    assert snapshot.input_message_count == 2
    assert snapshot.read_message_start == 0
    assert snapshot.read_message_end == 2  # When the read finishes, end == the total count
    assert snapshot.input == messages
    assert snapshot.output == "最终回复"
    assert snapshot.parent_observation_id == "parent-chat"
    assert "obs-gen-1" in bound.known_refs


@pytest.mark.asyncio
async def test_fetch_generation_snapshot_independent_budget_not_8k(
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
) -> None:
    """The independent budget defaults to 60_000: an output larger than max_observation_chars (8000) is not trimmed a second time."""
    provider = TraceToolProvider(fake_resolver, max_observation_chars=100)
    big_output = "x" * 20_000
    fake_resolver.fetch_observation.return_value = _gen_detail(output=big_output)

    snapshot = await provider.fetch_generation_snapshot(bound, "obs-gen-1")

    assert snapshot.output == big_output
    assert snapshot.output_truncated is False
    assert snapshot.evidence_incomplete is False


@pytest.mark.asyncio
async def test_fetch_generation_snapshot_budget_aligns_message_boundary(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    messages = [
        {"role": "system", "content": "a" * 60},
        {"role": "user", "content": "b" * 60},
        {"role": "tool", "content": "c" * 60},
    ]
    fake_resolver.fetch_observation.return_value = _gen_detail(
        input_value=messages, output="ok"
    )

    # Budget 200: output("ok") takes 2, the first two messages of about 90 characters each fit, and the third overflows → keep only whole messages.
    snapshot = await provider.fetch_generation_snapshot(bound, "obs-gen-1", max_chars=200)

    assert snapshot.evidence_incomplete is True
    assert "over_budget" in snapshot.incomplete_reasons
    assert snapshot.input_message_count == 3
    assert snapshot.read_message_end == 2
    assert len(snapshot.input) == 2


@pytest.mark.asyncio
async def test_fetch_generation_snapshot_single_message_over_budget_kept_whole(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    huge_message = {"role": "user", "content": "z" * 500}
    fake_resolver.fetch_observation.return_value = _gen_detail(input_value=[huge_message])

    snapshot = await provider.fetch_generation_snapshot(bound, "obs-gen-1", max_chars=100)

    # A single message over the budget is still kept whole (an unmarked truncation is forbidden), but it is explicitly marked over_budget.
    assert snapshot.input == [huge_message]
    assert snapshot.evidence_incomplete is True
    assert "over_budget" in snapshot.incomplete_reasons


@pytest.mark.asyncio
async def test_fetch_generation_snapshot_output_truncated_marks_incomplete(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.return_value = _gen_detail(output="y" * 500)

    snapshot = await provider.fetch_generation_snapshot(bound, "obs-gen-1", max_chars=50)

    assert snapshot.output_truncated is True
    assert snapshot.output.endswith("…[truncated]")
    assert "over_budget" in snapshot.incomplete_reasons


@pytest.mark.asyncio
async def test_fetch_generation_message_range_reads_remaining(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "tool", "content": "third"},
    ]
    fake_resolver.fetch_observation.return_value = _gen_detail(input_value=messages)

    snapshot = await provider.fetch_generation_message_range(
        bound, "obs-gen-1", GenerationMessageRange(message_start=1, message_end=3)
    )

    assert snapshot.read_message_start == 1
    assert snapshot.read_message_end == 3  # offset(1) + 2 messages == the total → the read is finished
    assert snapshot.input == messages[1:3]
    assert snapshot.evidence_incomplete is False


@pytest.mark.asyncio
async def test_fetch_generation_message_range_rejects_invalid_args(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    with pytest.raises(ValueError, match="field"):
        await provider.fetch_generation_message_range(
            bound,
            "obs-gen-1",
            GenerationMessageRange(field="output", message_start=0, message_end=1),
        )
    with pytest.raises(ValueError, match="invalid message range"):
        await provider.fetch_generation_message_range(
            bound, "obs-gen-1", GenerationMessageRange(message_start=3, message_end=1)
        )
    fake_resolver.fetch_observation.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_generation_message_range_missing_input(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    fake_resolver.fetch_observation.return_value = _gen_detail(input_value="not-a-list")

    snapshot = await provider.fetch_generation_message_range(
        bound, "obs-gen-1", GenerationMessageRange(message_start=0, message_end=1)
    )

    assert snapshot.evidence_incomplete is True
    assert snapshot.incomplete_reasons == ["missing_input"]


@pytest.mark.asyncio
async def test_fetch_generation_message_range_honors_request_budget(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
) -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    fake_resolver.fetch_observation.return_value = _gen_detail(input_value=messages)

    snapshot = await provider.fetch_generation_message_range(
        bound,
        "obs-gen-1",
        GenerationMessageRange(message_start=0, message_end=2, max_chars=1),
    )

    assert snapshot.input == messages[:1]
    assert snapshot.read_message_end == 1
    assert snapshot.evidence_incomplete is True
    assert "over_budget" in snapshot.incomplete_reasons


@pytest.mark.parametrize("read_range", [False, True])
@pytest.mark.parametrize("observation_id", ["gen-1", "missing"])
async def test_sidecar_generation_reads_never_fall_back_to_http(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    read_range: bool,
    observation_id: str,
) -> None:
    bound = _sidecar_bound()
    if read_range:
        snapshot = await provider.fetch_generation_message_range(
            bound, observation_id, GenerationMessageRange(message_start=0, message_end=1)
        )
    else:
        snapshot = await provider.fetch_generation_snapshot(bound, observation_id)
    fake_resolver.fetch_observation.assert_not_awaited()
    if observation_id == "gen-1":
        assert snapshot.input == [{"role": "user", "content": "附近咖啡"}]
        assert snapshot.evidence_incomplete is False
    else:
        assert snapshot.incomplete_reasons == ["missing_generation"]


async def test_sidecar_generation_rejects_cross_trace_detail(
    provider: TraceToolProvider, fake_resolver: AsyncMock
) -> None:
    bound = _sidecar_bound()
    bound.sidecar_observations = [{**_SIDECAR_OBS[1], "traceId": "other-trace"}]
    snapshot = await provider.fetch_generation_snapshot(bound, "gen-1")
    assert snapshot.incomplete_reasons == ["cross_trace_rejected"]
    fake_resolver.fetch_observation.assert_not_awaited()


@pytest.mark.parametrize(
    ("start", "end", "budget", "expected_end", "incomplete"),
    [(1, 2, None, 2, True), (1, 3, None, 3, False), (0, 3, 1, 1, True), (1, 20, None, 3, False)],
)
async def test_message_range_preserves_total_and_absolute_progress(
    fake_resolver: AsyncMock,
    provider: TraceToolProvider,
    bound: TraceRequestBound,
    start: int,
    end: int,
    budget: int | None,
    expected_end: int,
    incomplete: bool,
) -> None:
    messages = [{"role": "user", "content": str(i)} for i in range(3)]
    fake_resolver.fetch_observation.return_value = _gen_detail(input_value=messages)
    snapshot = await provider.fetch_generation_message_range(
        bound,
        "obs-gen-1",
        GenerationMessageRange(message_start=start, message_end=end, max_chars=budget),
    )
    assert snapshot.input_message_count == len(messages)
    assert snapshot.read_message_start == start
    assert snapshot.read_message_end == expected_end
    assert snapshot.evidence_incomplete is incomplete
    if incomplete:
        assert set(snapshot.incomplete_reasons) & {"message_range_unread", "over_budget"}


@pytest.mark.parametrize("read_range", [False, True])
@pytest.mark.parametrize("budget", [10000, 20])
async def test_generation_preserves_input_envelope_siblings(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
    read_range: bool,
    budget: int,
) -> None:
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"name": "route", "description": "x" * 100}],
        "temperature": 0,
    }
    fake_resolver.fetch_observation.return_value = _gen_detail(input_value=payload)
    if read_range:
        snapshot = await provider.fetch_generation_message_range(
            bound, "obs-gen-1", GenerationMessageRange(message_start=0, message_end=1, max_chars=budget)
        )
    else:
        snapshot = await provider.fetch_generation_snapshot(bound, "obs-gen-1", max_chars=budget)
    assert snapshot.input == payload
    assert snapshot.evidence_incomplete is (budget == 20)
    if budget == 20:
        assert "over_budget" in snapshot.incomplete_reasons
    assert bound.detail_cache["obs-gen-1"].input == payload


@pytest.mark.parametrize("read_range", [False, True])
@pytest.mark.parametrize("http_status", [None, 401, 503])
async def test_generation_read_error_is_distinct_from_missing(
    provider: TraceToolProvider,
    fake_resolver: AsyncMock,
    bound: TraceRequestBound,
    read_range: bool,
    http_status: int | None,
) -> None:
    error = TraceHttpError("upstream read failed", status_code=http_status)
    fake_resolver.fetch_observation.side_effect = error
    if read_range:
        snapshot = await provider.fetch_generation_message_range(
            bound, "obs-gen-1", GenerationMessageRange(message_start=0, message_end=1)
        )
    else:
        snapshot = await provider.fetch_generation_snapshot(bound, "obs-gen-1")
    assert snapshot.incomplete_reasons == ["generation_read_failed"]
    assert snapshot.read_error == str(error)
    assert snapshot.evidence_incomplete is True
    assert not bound.detail_cache


@pytest.mark.parametrize("read_range", [False, True])
async def test_generation_not_found_keeps_missing_reason(
    provider: TraceToolProvider, fake_resolver: AsyncMock,
    bound: TraceRequestBound, read_range: bool,
) -> None:
    fake_resolver.fetch_observation.return_value = None
    if read_range:
        snapshot = await provider.fetch_generation_message_range(
            bound, "obs-gone", GenerationMessageRange(message_start=0, message_end=1)
        )
    else:
        snapshot = await provider.fetch_generation_snapshot(bound, "obs-gone")
    assert snapshot.incomplete_reasons == ["missing_generation"]
    assert "obs-gone" in snapshot.read_error
