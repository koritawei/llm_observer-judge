"""Unit tests for the trace_url factory — identity-extractor registry + shared main flow (resolve / join / fall back).

Regression anchor: a production incident — the old implementation called ``resolve_primary_id_for(start=..., end=...)``
with kwargs the real signature does not have, so building trace_url on the judge callback always raised TypeError.
This test locks the real call shape: ``resolve_primary_id_for(project_id, trace_id)``
(positional arguments, no kwargs).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.common.trace_url_builder import (
    build_trace_url,
    build_trace_url_for_item,
    clear_item_trace_identity_extractors,
    clear_trace_identity_extractors,
    default_item_trace_identity,
    default_trace_identity,
    register_item_trace_identity_extractor,
    register_trace_identity_extractor,
    resolve_item_trace_identity_extractor,
    resolve_trace_identity_extractor,
)
from benchmark.runs.schema import JudgeCallback
from benchmark.trace.http_backend import TraceHttpError
from benchmark.trace.schema import TraceEndpoint

MOD = "benchmark.runs.common.trace_url_builder"


@pytest.fixture(autouse=True)
def _isolate_registry():
    clear_trace_identity_extractors()
    clear_item_trace_identity_extractors()
    yield
    clear_trace_identity_extractors()
    clear_item_trace_identity_extractors()


def _payload(
    *,
    project_id: str | None = "proj-1",
    trace_id: str | None = "biz-trace-1",
    trace_url: str | None = "https://pass.example/judge",
) -> JudgeCallback:
    return JudgeCallback(
        run_id="bench-20260907",
        item_id="bench-20260907:0",
        criterion_scores={},
        project_id=project_id,
        trace_id=trace_id,
        trace_url=trace_url,
    )


def _instances() -> dict[str, TraceEndpoint]:
    return {
        "proj-1": TraceEndpoint(
            base_url="https://lf.example.com",
            public_key="pk",
            secret_key="sk",
        ),
    }


def _resolver_cls(primary_id: str | None) -> tuple[MagicMock, MagicMock]:
    """Patched TraceResolver class mock; returns (cls, instance)."""
    resolver_inst = MagicMock()
    resolver_inst.resolve_primary_id_for = AsyncMock(return_value=primary_id)
    resolver_inst.aclose = AsyncMock()
    resolver_cls = MagicMock(return_value=resolver_inst)
    return resolver_cls, resolver_inst


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════


def test_default_extractor_reads_payload_fields() -> None:
    """Default extractor (ieval shape): read payload.project_id / payload.trace_id directly."""
    payload = _payload(project_id="p-x", trace_id="t-y")
    assert default_trace_identity(payload) == ("p-x", "t-y")


def test_resolve_falls_back_to_default_when_unregistered() -> None:
    assert resolve_trace_identity_extractor("unknown_bench") is default_trace_identity
    assert resolve_trace_identity_extractor(None) is default_trace_identity
    assert resolve_trace_identity_extractor("") is default_trace_identity


def test_registered_extractor_wins() -> None:
    """After a business registers an extractor it is hit by component name; a same-name override is idempotent."""
    extractor = lambda payload: ("custom-proj", "custom-trace")  # noqa: E731
    register_trace_identity_extractor("my_bench", extractor)
    register_trace_identity_extractor("my_bench", extractor)  # Idempotent override
    assert resolve_trace_identity_extractor("my_bench") is extractor
    # Other businesses are not affected by the registration
    assert resolve_trace_identity_extractor("other_bench") is default_trace_identity


# ════════════════════════════════════════════════════════════════════════════
# build_trace_url main flow
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_builds_url_from_payload_identity() -> None:
    """Default extractor → resolver resolution → an agentfuse panel link."""
    instances = _instances()
    resolver_cls, resolver_inst = _resolver_cls("primary-abc")
    with (
        patch(f"{MOD}.load_trace_instances", return_value=instances),
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url(
            _payload(),
            benchmark_name="ieval_external",
            fallback="https://pass.example/judge",
        )

    assert (
        url
        == "https://agentfuse-v4.amap-inc.com/project/proj-1/traces/primary-abc?traceId=primary-abc"
    )
    resolver_cls.assert_called_once_with(instances)
    # Regression lock: positional arguments (project_id, trace_id), no start/end kwargs
    call = resolver_inst.resolve_primary_id_for.await_args
    assert call.args == ("proj-1", "biz-trace-1")
    assert call.kwargs == {}
    resolver_inst.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_business_extractor_overrides_identity_source() -> None:
    """A business-registered extractor: the identity source is custom (not read straight off the payload), and the main flow is unchanged."""
    register_trace_identity_extractor(
        "custom_bench",
        lambda payload: ("proj-1", "trace-from-custom-logic"),
    )
    resolver_cls, resolver_inst = _resolver_cls("primary-xyz")
    with (
        patch(f"{MOD}.load_trace_instances", return_value=_instances()),
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url(
            _payload(trace_id="ignored-by-custom-extractor"),
            benchmark_name="custom_bench",
            fallback=None,
        )

    assert (
        url
        == "https://agentfuse-v4.amap-inc.com/project/proj-1/traces/primary-xyz?traceId=primary-xyz"
    )
    call = resolver_inst.resolve_primary_id_for.await_args
    assert call.args == ("proj-1", "trace-from-custom-logic")


@pytest.mark.asyncio
async def test_missing_project_id_returns_fallback() -> None:
    resolver_cls, _ = _resolver_cls("primary-abc")
    with (
        patch(f"{MOD}.load_trace_instances"),
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url(
            _payload(project_id=None),
            benchmark_name="ieval_external",
            fallback="https://pass.example/judge",
        )

    assert url == "https://pass.example/judge"
    resolver_cls.assert_not_called()


@pytest.mark.asyncio
async def test_missing_trace_id_returns_fallback() -> None:
    resolver_cls, _ = _resolver_cls("primary-abc")
    with (
        patch(f"{MOD}.load_trace_instances"),
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url(
            _payload(trace_id=""),
            benchmark_name="ieval_external",
            fallback="https://pass.example/judge",
        )

    assert url == "https://pass.example/judge"
    resolver_cls.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_project_returns_fallback() -> None:
    resolver_cls, _ = _resolver_cls("primary-abc")
    with (
        patch(f"{MOD}.load_trace_instances", return_value={}),
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url(
            _payload(),
            benchmark_name="ieval_external",
            fallback="https://pass.example/judge",
        )

    assert url == "https://pass.example/judge"
    resolver_cls.assert_not_called()


@pytest.mark.asyncio
async def test_no_primary_hit_returns_fallback_and_closes() -> None:
    resolver_cls, resolver_inst = _resolver_cls(None)
    with (
        patch(f"{MOD}.load_trace_instances", return_value=_instances()),
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url(
            _payload(),
            benchmark_name="ieval_external",
            fallback="https://pass.example/judge",
        )

    assert url == "https://pass.example/judge"
    resolver_inst.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolver_error_returns_fallback() -> None:
    resolver_cls, resolver_inst = _resolver_cls(None)
    resolver_inst.resolve_primary_id_for = AsyncMock(
        side_effect=TraceHttpError("upstream 500", status_code=500),
    )
    with (
        patch(f"{MOD}.load_trace_instances", return_value=_instances()),
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url(
            _payload(),
            benchmark_name="ieval_external",
            fallback="https://pass.example/judge",
        )

    assert url == "https://pass.example/judge"
    resolver_inst.aclose.assert_awaited_once()


# ════════════════════════════════════════════════════════════════════════════
# item shape (sync judge path: identity is extracted from business item fields)
# ════════════════════════════════════════════════════════════════════════════


def _item(
    *,
    project_id: str | None = "proj-1",
    trace_id: str | None = "biz-trace-1",
) -> dict:
    """Business item view (the minimal slice of a parse-interceptor product)."""
    return {
        "item_id": "run:0",
        "source_trace_id": trace_id,
        "metadata": {"project_id": project_id, "csv_row": {}},
    }


def test_default_item_extractor_reads_metadata_and_source_trace_id() -> None:
    """Default item extractor: metadata.project_id + source_trace_id."""
    assert default_item_trace_identity(_item(project_id="p", trace_id="t")) == ("p", "t")
    # Legacy sync business: metadata has no project_id → (None, trace_id)
    assert default_item_trace_identity(_item(project_id=None)) == (None, "biz-trace-1")
    # A non-dict dirty metadata value → None
    assert default_item_trace_identity({"metadata": "bad", "source_trace_id": "t"}) == (
        None,
        "t",
    )


def test_resolve_item_falls_back_to_default_when_unregistered() -> None:
    assert resolve_item_trace_identity_extractor("unknown") is default_item_trace_identity
    assert resolve_item_trace_identity_extractor(None) is default_item_trace_identity


def test_registered_item_extractor_wins() -> None:
    extractor = lambda item: ("custom-p", "custom-t")  # noqa: E731
    register_item_trace_identity_extractor("search_eval", extractor)
    assert resolve_item_trace_identity_extractor("search_eval") is extractor
    assert resolve_item_trace_identity_extractor("other") is default_item_trace_identity


@pytest.mark.asyncio
async def test_item_builds_url_from_item_identity() -> None:
    """search_eval shape: csv project/trace_id → item → an agentfuse panel link."""
    resolver_cls, resolver_inst = _resolver_cls("primary-item-1")
    with (
        patch(f"{MOD}.load_trace_instances", return_value=_instances()),
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url_for_item(
            _item(),
            benchmark_name="search_eval",
        )

    assert (
        url
        == "https://agentfuse-v4.amap-inc.com/project/proj-1/traces/primary-item-1?traceId=primary-item-1"
    )
    call = resolver_inst.resolve_primary_id_for.await_args
    assert call.args == ("proj-1", "biz-trace-1")
    assert call.kwargs == {}
    resolver_inst.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_item_missing_project_id_no_network() -> None:
    """Legacy sync business (metadata has no project_id): do not touch the network, and fall back to None."""
    resolver_cls, _ = _resolver_cls("primary-x")
    with (
        patch(f"{MOD}.load_trace_instances") as load_mock,
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url_for_item(
            _item(project_id=None),
            benchmark_name="example_benchmark",
        )

    assert url is None
    load_mock.assert_not_called()
    resolver_cls.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════
# Direct-join mode (trace_id is already the Langfuse primary key, so skip primary_id resolution)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_direct_mode_builds_url_without_resolver() -> None:
    """search_eval registers direct-join: do not touch the network or look up instance config; join the panel link directly."""
    from benchmark.runs.common.trace_url_builder import (
        register_direct_trace_url,
    )

    register_direct_trace_url("search_eval")
    resolver_cls, resolver_inst = _resolver_cls("primary-should-not-be-called")
    with (
        patch(f"{MOD}.load_trace_instances") as load_mock,
        patch(f"{MOD}.TraceResolver", resolver_cls),
    ):
        url = await build_trace_url_for_item(
            _item(project_id="proj-1", trace_id="trace-primary-id"),
            benchmark_name="search_eval",
        )

    assert (
        url == "https://agentfuse-v4.amap-inc.com/project/proj-1/traces/trace-primary-id"
        "?traceId=trace-primary-id"
    )
    load_mock.assert_not_called()
    resolver_cls.assert_not_called()
    resolver_inst.resolve_primary_id_for.assert_not_called()


@pytest.mark.asyncio
async def test_direct_mode_missing_identity_returns_fallback() -> None:
    """Direct-join mode: a missing identity still falls back (do not join an empty link)."""
    from benchmark.runs.common.trace_url_builder import (
        register_direct_trace_url,
    )

    register_direct_trace_url("search_eval")
    url = await build_trace_url_for_item(
        _item(project_id=None, trace_id="t"),
        benchmark_name="search_eval",
        fallback="fb",
    )
    assert url == "fb"
