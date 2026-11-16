"""Unit tests for the trace_resolver on-demand API (httpx mock, no real Langfuse)."""

import json

import httpx
import pytest
from benchmark.trace import (
    TraceEndpoint,
    TraceObservationsError,
    ObservationDetail,
    TraceResolver,
    TraceSkeletonResult,
    TraceSpanSkeleton,
)


@pytest.fixture(autouse=True)
def _pin_backoff_sleep_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin exponential-backoff sleep to zero so retry tests run instantly."""

    async def _noop(_delay: float) -> None:
        return None

    monkeypatch.setattr(
        "benchmark.trace.http_backend._backoff_sleep", _noop
    )


INSTANCE = TraceEndpoint(
    base_url="https://lf.example.com",
    public_key="pk-test",
    secret_key="sk-test",
    metadata_key="trace_id",
    name="proj-7-lf",
)
PRIMARY_ID = "lf-primary-123"
BIZ_TRACE_ID = "biz-1"

OBS_1 = {
    "id": "obs-1",
    "traceId": PRIMARY_ID,
    "name": "llm-call",
    "type": "GENERATION",
    "startTime": "2024-01-01T00:00:00Z",
    "endTime": "2024-01-01T00:01:00Z",
    "input": {"messages": [{"role": "user", "content": "hello"}]},
    "output": {"text": "world"},
    "parentObservationId": None,
    "level": "DEFAULT",
    "statusMessage": None,
}
OBS_2 = {
    "id": "obs-2",
    "traceId": PRIMARY_ID,
    "name": "tool-call",
    "type": "SPAN",
    "startTime": "2024-01-01T00:00:10Z",
    "endTime": "2024-01-01T00:00:30Z",
    "input": None,
    "output": None,
    "parentObservationId": "obs-1",
    "level": "DEFAULT",
    "statusMessage": None,
}
OBS_DETAIL = {
    "id": "obs-1",
    "traceId": PRIMARY_ID,
    "name": "llm-call",
    "type": "GENERATION",
    "input": {"messages": [{"role": "user", "content": "hello"}]},
    "output": {"text": "world"},
    "metadata": {"model": "gpt-4"},
    "statusMessage": None,
}


def _resolver(handler: object, *, max_retries: int = 3) -> TraceResolver:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    client = httpx.AsyncClient(transport=transport)
    return TraceResolver({"proj-7": INSTANCE}, client=client, max_retries=max_retries)


def _skeleton_hit_handler(request: httpx.Request) -> httpx.Response:
    """Happy-path: observations list only (no GET /traces/{id})."""
    if request.url.path == "/api/public/v2/observations":
        assert request.url.params.get("traceId") == PRIMARY_ID
        return httpx.Response(200, json={"data": [OBS_1, OBS_2]})
    return httpx.Response(404)


# ─── fetch_skeleton ────────────────────────────────────────────────────────────


async def test_fetch_skeleton_returns_skeleton():
    resolver = _resolver(_skeleton_hit_handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert result is not None
    assert isinstance(result, TraceSkeletonResult)
    assert result.primary_id == PRIMARY_ID
    assert result.business_trace_id == PRIMARY_ID
    assert result.project_id == "proj-7"
    assert result.instance_name == "proj-7-lf"
    assert len(result.spans) == 2
    assert not result.truncated

    span = result.spans[0]
    assert isinstance(span, TraceSpanSkeleton)
    assert span.id == "obs-1"
    assert span.name == "llm-call"
    assert span.type == "GENERATION"
    assert span.status == "DEFAULT"
    assert span.input_preview is not None
    assert span.output_preview is not None
    await resolver.aclose()


async def test_fetch_skeleton_null_io_gives_null_preview():
    resolver = _resolver(_skeleton_hit_handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert result is not None
    span2 = result.spans[1]
    assert span2.input_preview is None
    assert span2.output_preview is None
    await resolver.aclose()


async def test_fetch_skeleton_unknown_project_returns_none():
    resolver = _resolver(_skeleton_hit_handler)
    assert await resolver.fetch_skeleton(PRIMARY_ID, "proj-unknown") is None
    await resolver.aclose()


async def test_fetch_skeleton_skips_heavy_trace_detail_endpoint():
    """Skeleton must not call GET /traces/{id} (default fields → nested obs → 500)."""
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(200, json={"data": [OBS_1]})
        return httpx.Response(500, json={"error": "should not be called"})

    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert result is not None
    assert len(result.spans) == 1
    assert f"/api/public/traces/{PRIMARY_ID}" not in paths
    await resolver.aclose()


async def test_fetch_skeleton_observations_non200_raises():
    """Observations HTTP failure must not masquerade as an empty span list."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(500, json={"error": "obs error"})
        return httpx.Response(404)

    resolver = _resolver(handler)
    with pytest.raises(TraceObservationsError) as exc_info:
        await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert exc_info.value.status_code == 500
    await resolver.aclose()


async def test_fetch_skeleton_observations_empty_data_returns_empty_spans():
    """HTTP 200 with data=[] is a real empty result, not a failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert result is not None
    assert result.spans == []
    assert not result.truncated
    await resolver.aclose()


async def test_fetch_skeleton_observations_transport_exhausted_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            raise httpx.ConnectError("boom")
        return httpx.Response(404)

    resolver = _resolver(handler)
    with pytest.raises(TraceObservationsError) as exc_info:
        await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert exc_info.value.status_code is None
    await resolver.aclose()


async def test_find_observations_non200_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(403, json={"error": "forbidden"})
        return httpx.Response(404)

    resolver = _resolver(handler)
    with pytest.raises(TraceObservationsError) as exc_info:
        await resolver.find_observations("proj-7", primary_id=PRIMARY_ID)
    assert exc_info.value.status_code == 403
    await resolver.aclose()


async def test_fetch_skeleton_truncated():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            obs_list = [{"id": f"obs-{i}", "type": "SPAN"} for i in range(3)]
            return httpx.Response(200, json={"data": obs_list})
        return httpx.Response(404)

    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7", max_spans=2)
    assert result is not None
    assert result.truncated is True
    assert len(result.spans) == 2
    await resolver.aclose()


async def test_fetch_skeleton_preview_truncation():
    long_input = "x" * 500

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(
                200, json={"data": [{"id": "obs-1", "input": long_input, "output": "short"}]}
            )
        return httpx.Response(404)

    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7", preview_chars=50)
    assert result is not None
    span = result.spans[0]
    assert span.input_preview is not None
    assert len(span.input_preview) <= 52  # 50 chars + "…" (1 char)
    assert span.output_preview == "short"
    await resolver.aclose()


# ─── fetch_observation ─────────────────────────────────────────────────────────


async def test_fetch_observation_returns_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            flt = json.loads(request.url.params["filter"])
            assert flt[0]["column"] == "id"
            assert flt[0]["value"] == "obs-1"
            return httpx.Response(200, json={"data": [OBS_DETAIL]})
        return httpx.Response(404)

    resolver = _resolver(handler)
    detail = await resolver.fetch_observation("proj-7", "obs-1")
    assert detail is not None
    assert isinstance(detail, ObservationDetail)
    assert detail.id == "obs-1"
    assert detail.trace_id == PRIMARY_ID
    assert detail.name == "llm-call"
    assert detail.input is not None
    assert detail.output is not None
    await resolver.aclose()


async def test_fetch_observation_parses_json_io_string():
    payload = {
        **OBS_DETAIL,
        "input": '{"messages":[{"role":"user","content":"hi"}]}',
        "output": '{"text":"ok"}',
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(200, json={"data": [payload]})
        return httpx.Response(404)

    resolver = _resolver(handler)
    detail = await resolver.fetch_observation("proj-7", "obs-1")
    assert detail is not None
    assert detail.input == {"messages": [{"role": "user", "content": "hi"}]}
    assert detail.output == {"text": "ok"}
    await resolver.aclose()


async def test_fetch_observation_unknown_project_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OBS_DETAIL)

    resolver = _resolver(handler)
    assert await resolver.fetch_observation("proj-unknown", "obs-1") is None
    await resolver.aclose()


async def test_fetch_observation_not_found_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    resolver = _resolver(handler)
    assert await resolver.fetch_observation("proj-7", "obs-1") is None
    await resolver.aclose()


async def test_fetch_observation_non404_failure_raises():
    from benchmark.trace import TraceHttpError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    resolver = _resolver(handler)
    with pytest.raises(TraceHttpError) as exc_info:
        await resolver.fetch_observation("proj-7", "obs-1")
    assert exc_info.value.status_code == 500
    await resolver.aclose()


async def test_get_trace_defaults_to_fields_core():
    """Smoke/light detail must request fields=core (or caller override), never default-all."""
    from benchmark.trace.http_backend import TraceHttpBackend

    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            seen["fields"] = request.url.params.get("fields")
            assert request.url.params.get("traceId") == PRIMARY_ID
            assert request.url.params.get("isRootObservation") == "true"
            return httpx.Response(
                200,
                json={"data": [{"id": "obs-root", "traceId": PRIMARY_ID, "name": "t"}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        lf = TraceHttpBackend(INSTANCE, client, max_retries=0)
        trace = await lf.get_trace(PRIMARY_ID)
    assert trace is not None
    assert trace.id == PRIMARY_ID
    assert seen["fields"] == "core"


async def test_get_trace_empty_fields_requests_wide_set_and_rows():
    from benchmark.trace.http_backend import (
        _TRACE_FULL_FIELDS,
        TraceHttpBackend,
    )

    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            seen["fields"] = request.url.params.get("fields")
            assert request.url.params.get("isRootObservation") is None
            return httpx.Response(
                200,
                json={"data": [{"id": "obs-1", "traceId": PRIMARY_ID, "name": "t"}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        lf = TraceHttpBackend(INSTANCE, client, max_retries=0)
        trace = await lf.get_trace(PRIMARY_ID, fields=None)
    assert trace is not None
    assert seen["fields"] == _TRACE_FULL_FIELDS
    assert trace.model_dump()["observations"] == [
        {"id": "obs-1", "traceId": PRIMARY_ID, "name": "t"}
    ]


# ─── fetch_full_trace ──────────────────────────────────────────────────────────


async def test_fetch_full_trace_empty_fields_includes_observation_rows():
    """fields is empty → the wide field group, and the observation rows are attached to the trace."""
    from benchmark.trace.http_backend import _TRACE_FULL_FIELDS

    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            seen["fields"] = request.url.params.get("fields")
            return httpx.Response(
                200,
                json={"data": [{"id": "obs-1", "traceId": PRIMARY_ID}]},
            )
        return httpx.Response(404)

    resolver = _resolver(handler)
    result = await resolver.fetch_full_trace("proj-7", PRIMARY_ID)
    assert result is not None
    assert seen["fields"] == _TRACE_FULL_FIELDS
    assert result.id == PRIMARY_ID
    dumped = result.model_dump()
    assert dumped["observations"] == [{"id": "obs-1", "traceId": PRIMARY_ID}]
    await resolver.aclose()


async def test_fetch_full_trace_passes_fields_core():
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            seen["fields"] = request.url.params.get("fields")
            return httpx.Response(
                200,
                json={"data": [{"id": "obs-1", "traceId": PRIMARY_ID, "name": "t"}]},
            )
        return httpx.Response(404)

    resolver = _resolver(handler)
    result = await resolver.fetch_full_trace("proj-7", PRIMARY_ID, fields="core")
    assert result is not None
    assert result.id == PRIMARY_ID
    assert seen["fields"] == "core"
    await resolver.aclose()


async def test_fetch_full_trace_unknown_project_returns_none():
    resolver = TraceResolver({})
    assert await resolver.fetch_full_trace("nope", PRIMARY_ID) is None


async def test_get_trace_non404_failure_raises():
    from benchmark.trace import TraceHttpError
    from benchmark.trace.http_backend import TraceHttpBackend

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        lf = TraceHttpBackend(INSTANCE, client, max_retries=0)
        with pytest.raises(TraceHttpError) as exc_info:
            await lf.get_trace(PRIMARY_ID)
    assert exc_info.value.status_code == 500


async def test_fetch_observation_max_chars_truncates():
    long_text = "a" * 500
    detail_payload = {**OBS_DETAIL, "input": long_text, "output": long_text}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(200, json={"data": [detail_payload]})
        return httpx.Response(404)

    resolver = _resolver(handler)
    detail = await resolver.fetch_observation("proj-7", "obs-1", max_chars=100)
    assert detail is not None
    assert isinstance(detail.input, str) and len(detail.input) <= 102  # 100 + "…" (1 char)
    assert isinstance(detail.output, str) and len(detail.output) <= 102
    await resolver.aclose()


# ─── find_observations ─────────────────────────────────────────────────────────


async def test_find_observations_by_primary_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            assert request.url.params.get("traceId") == PRIMARY_ID
            return httpx.Response(200, json={"data": [OBS_1]})
        return httpx.Response(404)

    resolver = _resolver(handler)
    spans = await resolver.find_observations("proj-7", primary_id=PRIMARY_ID)
    assert len(spans) == 1
    assert isinstance(spans[0], TraceSpanSkeleton)
    assert spans[0].id == "obs-1"
    await resolver.aclose()


async def test_resolve_primary_id_for_uses_metadata_filter():
    """Entry helper filters by metadata.trace_id + bounded timestamp lookback."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations" and "filter" in request.url.params:
            flt = json.loads(request.url.params["filter"])
            assert flt == [
                {
                    "type": "stringObject",
                    "column": "metadata",
                    "key": "trace_id",
                    "operator": "=",
                    "value": BIZ_TRACE_ID,
                }
            ]
            assert request.url.params.get("fromStartTime")
            assert request.url.params.get("toStartTime")
            assert request.url.params.get("fields") == "core"
            return httpx.Response(
                200,
                json={"data": [{"id": "obs-1", "traceId": PRIMARY_ID}]},
            )
        return httpx.Response(404)

    resolver = _resolver(handler)
    assert await resolver.resolve_primary_id_for("proj-7", BIZ_TRACE_ID) == PRIMARY_ID
    await resolver.aclose()


async def test_resolve_primary_id_for_non200_raises():
    """422 / transport must not collapse into a not-found miss."""
    from benchmark.trace import TraceHttpError

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(
                422,
                json={"message": "narrow your request", "error": "Unprocessable Content"},
            )
        return httpx.Response(404)

    resolver = _resolver(handler)
    with pytest.raises(TraceHttpError) as exc_info:
        await resolver.resolve_primary_id_for("proj-7", BIZ_TRACE_ID)
    assert exc_info.value.status_code == 422
    await resolver.aclose()


async def test_resolve_primary_id_for_empty_200_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    resolver = _resolver(handler)
    assert await resolver.resolve_primary_id_for("proj-7", BIZ_TRACE_ID) is None
    await resolver.aclose()


async def test_find_observations_unknown_project_returns_empty():
    resolver = _resolver(_skeleton_hit_handler)
    spans = await resolver.find_observations("proj-unknown", primary_id=PRIMARY_ID)
    assert spans == []
    await resolver.aclose()


async def test_find_observations_filter_by_name():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            assert request.url.params.get("name") == "llm-call"
            return httpx.Response(200, json={"data": [OBS_1]})
        return httpx.Response(404)

    resolver = _resolver(handler)
    spans = await resolver.find_observations("proj-7", primary_id=PRIMARY_ID, name="llm-call")
    assert len(spans) == 1
    assert spans[0].name == "llm-call"
    await resolver.aclose()


async def test_find_observations_status_maps_to_level_query():
    """status is forwarded as Langfuse ``level`` query param (server-side filter)."""
    obs_error = {**OBS_2, "id": "obs-err", "level": "ERROR"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            assert request.url.params.get("level") == "ERROR"
            return httpx.Response(200, json={"data": [obs_error]})
        return httpx.Response(404)

    resolver = _resolver(handler)
    spans = await resolver.find_observations("proj-7", primary_id=PRIMARY_ID, status="ERROR")
    assert len(spans) == 1
    assert spans[0].id == "obs-err"
    await resolver.aclose()


# ─── observation list pagination ───────────────────────────────────────────────


def _obs_meta(*, page: int, limit: int, total_items: int) -> dict:
    total_pages = max(1, (total_items + limit - 1) // limit) if limit else 1
    return {
        "page": page,
        "limit": limit,
        "totalItems": total_items,
        "totalPages": total_pages,
    }


async def test_list_observations_pages_until_limit():
    """Client follows meta.cursor with fixed limit=100 until target count is reached."""
    total = 150
    all_obs = [{"id": f"obs-{i}", "type": "SPAN"} for i in range(total)]
    cursors_seen: list[str | None] = []
    limits_seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            cursor = request.url.params.get("cursor")
            limit = int(request.url.params["limit"])
            cursors_seen.append(cursor)
            limits_seen.append(limit)
            assert limit == 100
            start = 0 if not cursor else 100
            chunk = all_obs[start : start + limit]
            next_cursor = None if cursor else "c2"
            return httpx.Response(
                200,
                json={"data": chunk, "meta": {"cursor": next_cursor}},
            )
        return httpx.Response(404)

    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7", max_spans=120)
    assert result is not None
    assert len(result.spans) == 120
    assert result.truncated is True  # asked for 121, more remain
    assert cursors_seen == [None, "c2"]
    assert limits_seen == [100, 100]
    await resolver.aclose()


async def test_list_observations_not_truncated_when_exhausted():
    """When all pages fit under max_spans, truncated stays False."""
    total = 50
    all_obs = [{"id": f"obs-{i}", "type": "SPAN"} for i in range(total)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            page = int(request.url.params.get("page", "1"))
            limit = int(request.url.params["limit"])
            start = (page - 1) * 100
            chunk = all_obs[start : start + limit]
            return httpx.Response(
                200,
                json={
                    "data": chunk,
                    "meta": _obs_meta(page=page, limit=limit, total_items=total),
                },
            )
        return httpx.Response(404)

    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7", max_spans=300)
    assert result is not None
    assert len(result.spans) == 50
    assert result.truncated is False
    await resolver.aclose()


async def test_find_observations_error_on_second_page():
    """ERROR spans beyond the first page are still found via pagination + level."""
    error_obs = {"id": "obs-err-late", "type": "SPAN", "level": "ERROR"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            assert request.url.params.get("level") == "ERROR"
            cursor = request.url.params.get("cursor")
            limit = int(request.url.params["limit"])
            assert limit <= 100
            if not cursor:
                dummy = [
                    {"id": f"err-{i}", "type": "SPAN", "level": "ERROR"}
                    for i in range(100)
                ]
                return httpx.Response(
                    200,
                    json={"data": dummy, "meta": {"cursor": "c2"}},
                )
            return httpx.Response(
                200,
                json={"data": [error_obs], "meta": {"cursor": None}},
            )
        return httpx.Response(404)

    resolver = _resolver(handler)
    spans = await resolver.find_observations("proj-7", primary_id=PRIMARY_ID, status="ERROR")
    assert len(spans) == 101
    assert spans[-1].id == "obs-err-late"
    await resolver.aclose()


# ─── schema / lifecycle tests ─────────────────────────────────────────────────


def test_instance_parses_camelcase_config():
    inst = TraceEndpoint.model_validate(
        {
            "name": "amap",
            "baseUrl": "https://lf.example.com",
            "publicKey": "pk-x",
            "secretKey": "sk-x",
        }
    )
    assert inst.base_url == "https://lf.example.com"
    assert inst.public_key == "pk-x"
    assert inst.secret_key.get_secret_value() == "sk-x"
    assert inst.metadata_key == "trace_id"


async def test_aclose_closes_owned_client():
    resolver = TraceResolver({"proj-7": INSTANCE})
    await resolver.aclose()


# ─── retry tests (via observations list used by fetch_skeleton) ───────────────


def _obs_plan_handler(plan: list[object]) -> tuple[object, dict[str, int]]:
    """Serve sequential responses / transport errors to GET /observations."""
    calls: dict[str, int] = {"obs": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/public/v2/observations":
            item = plan[calls["obs"]]
            calls["obs"] += 1
            if isinstance(item, Exception):
                raise item
            return item  # type: ignore[return-value]
        return httpx.Response(404)

    return handler, calls


async def test_retries_observations_5xx_then_succeeds():
    plan: list[object] = [
        httpx.Response(503),
        httpx.Response(503),
        httpx.Response(200, json={"data": [OBS_1]}),
    ]
    handler, calls = _obs_plan_handler(plan)
    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert result is not None
    assert len(result.spans) == 1
    assert calls["obs"] == 3
    await resolver.aclose()


async def test_observations_5xx_exhausts_retries_raises():
    plan: list[object] = [httpx.Response(500)] * 4  # initial attempt + 3 retries
    handler, calls = _obs_plan_handler(plan)
    resolver = _resolver(handler)
    with pytest.raises(TraceObservationsError) as exc_info:
        await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert exc_info.value.status_code == 500
    assert calls["obs"] == 4
    await resolver.aclose()


async def test_observations_4xx_not_retried():
    handler, calls = _obs_plan_handler([httpx.Response(403)])
    resolver = _resolver(handler)
    with pytest.raises(TraceObservationsError) as exc_info:
        await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert exc_info.value.status_code == 403
    assert calls["obs"] == 1
    await resolver.aclose()


async def test_retries_observations_429_then_succeeds():
    plan: list[object] = [
        httpx.Response(429),
        httpx.Response(200, json={"data": [OBS_1]}),
    ]
    handler, calls = _obs_plan_handler(plan)
    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert result is not None
    assert calls["obs"] == 2
    await resolver.aclose()


async def test_retries_observations_transport_error_then_succeeds():
    plan: list[object] = [
        httpx.ConnectError("boom"),
        httpx.ConnectError("boom"),
        httpx.Response(200, json={"data": [OBS_1]}),
    ]
    handler, calls = _obs_plan_handler(plan)
    resolver = _resolver(handler)
    result = await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert result is not None
    assert calls["obs"] == 3
    await resolver.aclose()


async def test_observations_read_timeout_not_retried():
    """ReadTimeout must fail fast — retrying a slow Langfuse query is useless."""
    handler, calls = _obs_plan_handler([httpx.ReadTimeout("slow")])
    resolver = _resolver(handler)
    with pytest.raises(TraceObservationsError) as exc_info:
        await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert exc_info.value.status_code is None
    assert calls["obs"] == 1
    await resolver.aclose()


async def test_observations_transport_error_exhausts_retries_raises():
    plan: list[object] = [httpx.ConnectError("boom")] * 4
    handler, calls = _obs_plan_handler(plan)
    resolver = _resolver(handler)
    with pytest.raises(TraceObservationsError) as exc_info:
        await resolver.fetch_skeleton(PRIMARY_ID, "proj-7")
    assert exc_info.value.status_code is None
    assert calls["obs"] == 4
    await resolver.aclose()


async def test_fetch_metric_inputs_requests_bounded_trace_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/v2/observations"
        if request.url.params.get("type") == "GENERATION":
            assert request.url.params.get("fields") == "core,basic,io,usage,metrics"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            **OBS_1,
                            "inputUsage": 40,
                            "outputUsage": 10,
                            "usageDetails": {"total": 50},
                        }
                    ]
                },
            )
        assert request.url.params.get("fields") == "core,io,metadata,metrics"
        assert request.url.params.get("expandMetadata") == "first_text_response_time_ms"
        assert request.url.params.get("traceId") == PRIMARY_ID
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "obs-root",
                        "traceId": PRIMARY_ID,
                        "isRootObservation": True,
                        "latency": 1.5,
                        "metadata": {"first_text_response_time_ms": 123},
                        "input": {"query": "hello"},
                    }
                ]
            },
        )

    resolver = _resolver(handler)
    trace, observations = await resolver.fetch_metric_inputs("proj-7", primary_id=PRIMARY_ID)
    assert trace is not None
    assert trace.input == {"query": "hello"}
    assert trace.metadata == {"first_text_response_time_ms": 123}
    assert observations[0]["promptTokens"] == 40
    assert observations[0]["completionTokens"] == 10
    assert observations[0]["totalTokens"] == 50
    assert observations[0]["id"] == "obs-1"
