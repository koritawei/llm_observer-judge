"""Single-instance Langfuse HTTP client (no fan-out, no concurrency).

Reads use Observations API v2 (``GET /api/public/v2/observations``).
Every call is bounded by ``fromStartTime`` / ``toStartTime``.

1. ``resolve_primary_id(trace_id)`` — **entry-only**: filter
   ``metadata.<key> == trace_id``. The hit's ``traceId`` is the primary id.
   Downstream (tools / DB) must already hold that id.
2. ``list_observations_for_trace(primary_id, ...)`` — cursor pages for one
   ``traceId``.
3. ``get_observation(observation_id)`` — filter ``id == observation_id``.
   An empty page is a miss; there is no get-by-id route.
4. ``get_trace(primary_id, fields=...)`` — project the root observation onto
   :class:`LangfuseTrace`. Empty ``fields`` also attaches the observation rows.
"""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx  # noqa: TID251 — Langfuse HTTP client wrapper; keeps the main transport boundary
from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from benchmark.trace.schema import TraceEndpoint, TraceDocument

logger = get_logger("agent_eval.trace_resolver")

_HTTP_OK = 200

# Retry policy: transient failures (transport/timeout + HTTP 429/5xx) are
# retried with exponential backoff + jitter via tenacity. Non-retryable 4xx
# (auth / not-found / bad-request) are returned immediately.
DEFAULT_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, *range(500, 600)})
#: Metadata resolve and later reads share this window. ``toStartTime`` is
#: exclusive, so the client adds one second past ``now``.
DEFAULT_RESOLVE_LOOKBACK_DAYS = 14
#: Default httpx timeout for all Langfuse Public API calls (resolve + tools).
DEFAULT_TIMEOUT_SECONDS = 30.0

#: Type alias for the Langfuse HTTP transport pool — reused by TraceResolver injection signatures (httpx is closed over in this module).
TraceHttpClient = httpx.AsyncClient


def build_shared_http_client(timeout: float) -> TraceHttpClient:
    """Build a Langfuse HTTP connection pool shared across instances (TraceResolver owns the lifecycle)."""
    return httpx.AsyncClient(
        timeout=timeout,
        trust_env=False,
        follow_redirects=True,
    )


def _utc_z(dt: datetime) -> str:
    """Format ``dt`` as ISO-8601 UTC with ``Z`` (Langfuse datetime filter)."""

    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


class TraceHttpError(Exception):
    """Raised when a Langfuse Public API call fails (HTTP / transport / parse).

    ``status_code`` is set for HTTP failures; ``None`` means transport
    exhaustion or JSON parse failure. Callers must not collapse this into a
    not-found miss — especially for 5xx.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TraceObservationsError(TraceHttpError):
    """Raised when listing observations fails (HTTP / transport / parse).

    Distinct from a successful empty list: callers must not treat this as
    ``spans=[]``.
    """


#: Observations API v2. Legacy ``/api/public/traces`` and
#: ``/api/public/observations`` are unavailable in v4 events_only mode.
_OBS_V2_PATH = "/api/public/v2/observations"
#: One cursor page. v2 allows up to 1000; 100 matches the previous page size.
_OBS_PAGE_SIZE = 100
_OBS_LIST_FIELDS = "core,basic,io"
_OBS_DETAIL_FIELDS = "core,basic,io,metadata,time"
_TRACE_FULL_FIELDS = "core,basic,io,metadata,time,usage,metrics,trace_context"
_METRIC_TRACE_FIELDS = "core,io,metadata,metrics"
_METRIC_GENERATION_FIELDS = "core,basic,io,usage,metrics"
_FIRST_RESPONSE_METADATA_KEY = "first_text_response_time_ms"
_FULL_TRACE_OBS_LIMIT = 1000
#: v2 requires a time bound on every request. Entry metadata lookup uses 14 days;
#: a point lookup of a known traceId / observation id uses a wide window so a historical trace is not judged a miss.
_ID_LOOKUP_LOOKBACK_DAYS = 3650
_METADATA_TRUNCATE_LEN = 200


@dataclass(frozen=True)
class ObservationListQuery:
    """One ``list_observations_for_trace`` call.

    Bundles the filters so the method stays under the parameter gate.
    ``fields`` empty means ``core,basic,io``.
    """

    limit: int
    name: str | None = None
    type: str | None = None
    level: str | None = None
    fields: str | None = None
    is_root: bool = False
    expand_metadata: str | None = None


class _RetryableHTTPStatusError(Exception):
    """Wrap a retryable HTTP response (429/5xx) to trigger tenacity retry.

    Carries the original ``httpx.Response`` so the caller can recover it
    when retries are exhausted.
    """

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"retryable HTTP {response.status_code}")


async def _backoff_sleep(delay: float) -> None:
    """Module-level indirection so tests can pin backoff sleep to zero.

    Passed as ``sleep=`` to tenacity so test monkeypatch still works.
    """
    await asyncio.sleep(delay)


def _metadata_filter(metadata_key: str, trace_id: str) -> str:
    """``filter`` for ``metadata.<key> == trace_id``. Time bounds are query params."""
    return json.dumps(
        [
            {
                "type": "stringObject",
                "column": "metadata",
                "key": metadata_key,
                "operator": "=",
                "value": trace_id,
            },
        ]
    )


def _observation_id_filter(observation_id: str) -> str:
    """``filter`` for a single observation. v2 has no get-by-id route."""
    return json.dumps(
        [
            {
                "type": "string",
                "column": "id",
                "operator": "=",
                "value": observation_id,
            },
        ]
    )


def _observation_page(response: httpx.Response) -> tuple[list[dict], str | None]:
    """Return ``(rows, next_cursor)``. ``next_cursor`` is ``None`` at the last page."""
    body = response.json()
    data = body.get("data") if isinstance(body, dict) else None
    rows = [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    meta = body.get("meta") if isinstance(body, dict) else None
    cursor = meta.get("cursor") if isinstance(meta, dict) else None
    if not isinstance(cursor, str) or not cursor:
        return rows, None
    return rows, cursor


def _parse_json_text(value: object) -> object:
    """v2 fixes input/output as raw strings. Restore them when they parse as an object or array."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _normalize_observation(row: dict) -> dict:
    normalized = dict(row)
    for key in ("input", "output", "metadata"):
        if key in normalized:
            normalized[key] = _parse_json_text(normalized[key])
    _fill_token_aliases(normalized)
    return normalized


def _as_int(value: object) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fill_token_aliases(row: dict) -> None:
    """Copy v2 usage fields onto ``promptTokens`` / ``completionTokens`` / ``totalTokens``.

    ``metrics.py`` still reads the v1 names. Existing v1 values are left as-is.
    """

    details = row.get("usageDetails")
    detail_map = details if isinstance(details, dict) else {}
    _set_token_alias(row, detail_map, ("promptTokens", "inputUsage", "input"))
    _set_token_alias(row, detail_map, ("completionTokens", "outputUsage", "output"))
    _set_token_alias(row, detail_map, ("totalTokens", "totalUsage", "total"))


def _set_token_alias(
    row: dict,
    detail_map: dict,
    keys: tuple[str, str, str],
) -> None:
    legacy, usage_key, detail_key = keys
    if _as_int(row.get(legacy)) is not None:
        return
    number = _as_int(row.get(usage_key))
    if number is None:
        number = _as_int(detail_map.get(detail_key))
    if number is not None:
        row[legacy] = number


@dataclass(frozen=True)
class _ListPage:
    window: dict[str, str]
    cursor: str | None
    expand_keys: list[str]
    request_fields: str


def _observation_page_params(
    primary_id: str,
    query: ObservationListQuery,
    page: _ListPage,
) -> dict[str, object]:
    params: dict[str, object] = {
        **page.window,
        "traceId": primary_id,
        "limit": _OBS_PAGE_SIZE,
        "fields": page.request_fields,
    }
    if page.expand_keys:
        params["expandMetadata"] = ",".join(page.expand_keys)
    if query.name is not None:
        params["name"] = query.name
    if query.type is not None:
        params["type"] = query.type
    if query.level is not None:
        params["level"] = query.level
    if query.is_root:
        params["isRootObservation"] = "true"
    if page.cursor is not None:
        params["cursor"] = page.cursor
    return params


def _expand_retry_keys(
    request_fields: str,
    expand_keys: list[str],
    rows: list[dict],
) -> list[str] | None:
    """Keys to refetch when this page still has truncated metadata. ``None`` means keep the page."""

    if "metadata" not in request_fields.split(","):
        return None
    extra = _truncated_metadata_keys(rows)
    merged = sorted(set(expand_keys).union(extra))
    if extra and merged != expand_keys:
        return merged
    return None


def _truncated_metadata_keys(rows: list[dict]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        metadata = _parse_json_text(row.get("metadata"))
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            if isinstance(value, str) and len(value) >= _METADATA_TRUNCATE_LEN:
                keys.append(str(key))
    return sorted(set(keys))


def _trace_from_observations(
    primary_id: str,
    rows: list[dict],
    *,
    include_observations: bool,
) -> TraceDocument | None:
    """Project observation rows onto a trace. ``id`` is the Langfuse trace id."""
    if not rows:
        return None
    root = next((row for row in rows if row.get("isRootObservation") is True), rows[0])
    payload: dict[str, object] = {
        "id": primary_id,
        "name": root.get("name"),
        "timestamp": root.get("startTime") or root.get("timestamp"),
        "environment": root.get("environment"),
        "sessionId": root.get("sessionId"),
        "userId": root.get("userId"),
        "tags": root.get("tags"),
        "input": root.get("input"),
        "output": root.get("output"),
        "metadata": root.get("metadata"),
        "latency": root.get("latency"),
        "totalCost": root.get("totalCost"),
    }
    if include_observations:
        payload["observations"] = rows
    try:
        return TraceDocument.model_validate(payload)
    except ValidationError as exc:
        logger.warning("trace_resolver.trace_parse_failed error=%s", exc)
        return None


class TraceHttpBackend:
    """Query one Langfuse instance for a trace by its business id.

    The httpx client is borrowed (owned by the caller / ``TraceResolver``); this
    class does not open or close it.
    """

    def __init__(
        self,
        instance: TraceEndpoint,
        client: httpx.AsyncClient,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        resolve_lookback_days: int = DEFAULT_RESOLVE_LOOKBACK_DAYS,
    ) -> None:
        self._instance = instance
        self._client = client
        self._max_retries = max_retries
        self._resolve_lookback_days = resolve_lookback_days

    def _window_params(
        self,
        *,
        lookback_days: int | None = None,
        now: datetime | None = None,
    ) -> dict[str, str]:
        """``fromStartTime`` / ``toStartTime`` for one v2 read.

        ``toStartTime`` is exclusive (``startTime < value``).
        When ``lookback_days`` is omitted, use the entry resolve window.
        """
        days = self._resolve_lookback_days if lookback_days is None else lookback_days
        end = now or datetime.now(UTC)
        start = end - timedelta(days=max(1, days))
        return {
            "fromStartTime": _utc_z(start),
            "toStartTime": _utc_z(end + timedelta(seconds=1)),
        }

    async def _get_with_retry(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None,
        auth: tuple[str, str],
    ) -> httpx.Response | None:
        """GET ``url`` with selective retry on transient failures.

        Retries on transport errors (``httpx.HTTPError``) and on HTTP 429 / 5xx
        (via ``_RetryableHTTPStatus``) — **except** timeouts
        (``httpx.TimeoutException``, e.g. ``ReadTimeout``): a slow Langfuse
        query stays slow, so retrying it fails fast instead of compounding
        load with more slow requests. A non-retryable response (e.g. 4xx) and
        the final response after retries are exhausted are returned as-is;
        ``None`` means the attempt failed at the transport layer.

        Uses ``tenacity`` for retry logic with ``_backoff_sleep`` as the sleep
        function (so tests can pin it to zero).
        """
        retrying = retry(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential_jitter(
                initial=_BACKOFF_BASE, max=30, jitter=0.5
            ),
            retry=(
                retry_if_exception_type((httpx.HTTPError, _RetryableHTTPStatusError))
                & retry_if_not_exception_type(httpx.TimeoutException)
            ),
            sleep=_backoff_sleep,
            reraise=True,
        )

        async def _do_get() -> httpx.Response:
            resp = await self._client.get(url, params=params, auth=auth)
            if resp.status_code in _RETRYABLE_STATUS:
                raise _RetryableHTTPStatusError(resp)
            return resp

        try:
            return await retrying(_do_get)()
        except _RetryableHTTPStatusError as exc:
            # Retries exhausted on HTTP status — return last response
            return exc.response
        except httpx.HTTPError:
            # Transport error retries exhausted
            logger.warning(
                "trace_resolver.transport_exhausted instance=%s url=%s",
                self._instance.name or self._instance.base_url,
                url,
            )
            return None

    async def resolve_primary_id(self, trace_id: str) -> str | None:
        """Entry-only: map business ``metadata.<key>`` value → Langfuse primary id.

        Do **not** call this from tool / observation paths — those must already
        hold the primary id rewritten at eval entry.

        Returns ``None`` only on HTTP 200 with an empty hit list. Transport
        exhaustion and non-200 responses raise :class:`LangfuseHttpError` so
        callers do not collapse upstream failures into ``trace_not_found``.
        """
        instance = self._instance
        base = instance.base_url.rstrip("/")
        auth = (instance.public_key, instance.secret_key.get_secret_value())
        params: dict[str, object] = {
            **self._window_params(),
            "limit": 1,
            "fields": "core",
            "filter": _metadata_filter(instance.metadata_key, trace_id),
        }

        filter_resp = await self._get_with_retry(
            f"{base}{_OBS_V2_PATH}",
            params=params,
            auth=auth,
        )
        if filter_resp is None or filter_resp.status_code != _HTTP_OK:
            status = filter_resp.status_code if filter_resp is not None else None
            logger.warning(
                "trace_resolver.filter_failed instance=%s status=%s trace_id=%s "
                "lookback_days=%s",
                instance.name or base,
                status,
                trace_id,
                self._resolve_lookback_days,
            )
            raise TraceHttpError(
                f"trace filter request failed (status={status})",
                status_code=status,
            )

        try:
            rows, _cursor = _observation_page(filter_resp)
        except Exception as exc:
            logger.warning("trace_resolver.filter_parse_failed error=%s", exc)
            raise TraceHttpError(
                f"trace filter response parse failed: {exc}",
                status_code=_HTTP_OK,
            ) from exc
        if not rows:
            logger.info(
                "trace_resolver.miss instance=%s trace_id=%s", instance.name or base, trace_id
            )
            return None

        primary_id = rows[0].get("traceId") or rows[0].get("trace_id")
        if not isinstance(primary_id, str) or not primary_id:
            logger.warning(
                "trace_resolver.no_primary_id instance=%s trace_id=%s",
                instance.name or base,
                trace_id,
            )
            return None
        return primary_id

    async def get_trace(
        self,
        primary_id: str,
        *,
        fields: str | None = "core",
        expand_metadata: str | None = None,
    ) -> TraceDocument | None:
        """Return trace detail by Langfuse primary id, or ``None`` when no row matches.

        ``fields`` selects v2 field groups. Default ``core``. Empty ``fields``
        requests the wide set and attaches every fetched observation under
        ``observations`` (``fetch_full_trace`` only).

        Non-200 responses raise :class:`LangfuseHttpError`.
        """
        full = not (fields and fields.strip())
        request_fields = fields.strip() if fields and fields.strip() else _TRACE_FULL_FIELDS
        rows = await self.list_observations_for_trace(
            primary_id,
            ObservationListQuery(
                limit=_FULL_TRACE_OBS_LIMIT if full else 1,
                fields=request_fields,
                is_root=not full,
                expand_metadata=expand_metadata,
            ),
        )
        trace = _trace_from_observations(
            primary_id,
            rows,
            include_observations=full,
        )
        if trace is None:
            logger.info(
                "trace_resolver.trace_miss instance=%s primary_id=%s",
                self._instance.name or self._instance.base_url,
                primary_id,
            )
        return trace

    async def list_observations_for_trace(
        self,
        primary_id: str,
        query: ObservationListQuery,
    ) -> list[dict]:
        """Return up to ``query.limit`` observation dicts for one Langfuse trace id.

        Pages with ``meta.cursor`` (v2). Each request asks for ``_OBS_PAGE_SIZE``
        rows. ``fields`` defaults to ``core,basic,io``.

        A successful empty result returns ``[]``. Transport / HTTP / parse
        failures on any page raise :class:`LangfuseObservationsError`.
        """
        if query.limit <= 0:
            return []

        instance = self._instance
        url = f"{instance.base_url.rstrip('/')}{_OBS_V2_PATH}"
        auth = (instance.public_key, instance.secret_key.get_secret_value())
        request_fields = query.fields or _OBS_LIST_FIELDS
        collected = await self._collect_observation_pages(
            primary_id,
            query,
            (url, auth, request_fields),
        )
        return collected[: query.limit]

    async def _collect_observation_pages(
        self,
        primary_id: str,
        query: ObservationListQuery,
        transport: tuple[str, tuple[str, str], str],
    ) -> list[dict]:
        """Page until ``query.limit`` or the cursor ends.

        Refetch a page once when metadata is truncated.
        """

        url, auth, request_fields = transport
        collected: list[dict] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        page = 0
        expand_keys = [part for part in (query.expand_metadata or "").split(",") if part]
        retried_expand = False
        window = self._window_params(lookback_days=_ID_LOOKUP_LOOKBACK_DAYS)
        while len(collected) < query.limit:
            page += 1
            params = _observation_page_params(
                primary_id,
                query,
                _ListPage(window, cursor, expand_keys, request_fields),
            )
            rows, next_cursor = await self._read_observation_rows(
                url,
                params,
                (auth, page, len(collected), primary_id),
            )
            retry_keys = None
            if not retried_expand:
                retry_keys = _expand_retry_keys(request_fields, expand_keys, rows)
            if retry_keys is not None:
                expand_keys = retry_keys
                retried_expand = True
                page -= 1
                continue
            retried_expand = False
            collected.extend(_normalize_observation(row) for row in rows)
            if not next_cursor or next_cursor in seen_cursors or len(rows) < _OBS_PAGE_SIZE:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return collected

    async def _read_observation_rows(
        self,
        url: str,
        params: dict[str, object],
        call: tuple[tuple[str, str], int, int, str],
    ) -> tuple[list[dict], str | None]:
        auth, page, collected, primary_id = call
        resp = await self._get_with_retry(url, params=params, auth=auth)
        if resp is None or resp.status_code != _HTTP_OK:
            status = resp.status_code if resp is not None else None
            logger.warning(
                "trace_resolver.observations_failed instance=%s status=%s "
                "primary_id=%s page=%s collected=%s",
                self._instance.name or self._instance.base_url,
                status,
                primary_id,
                page,
                collected,
            )
            raise TraceObservationsError(
                f"observations request failed (status={status}, page={page})",
                status_code=status,
            )
        try:
            return _observation_page(resp)
        except Exception as exc:
            logger.warning(
                "trace_resolver.observations_parse_failed instance=%s page=%s error=%s",
                self._instance.name or self._instance.base_url,
                page,
                exc,
            )
            raise TraceObservationsError(
                f"observations response parse failed (page={page}): {exc}",
                status_code=_HTTP_OK,
            ) from exc

    async def get_observation(self, observation_id: str) -> dict | None:
        """Return one observation dict, or ``None`` when the id filter matches nothing.

        Non-200 responses raise :class:`LangfuseHttpError`.
        """
        instance = self._instance
        base = instance.base_url.rstrip("/")
        auth = (instance.public_key, instance.secret_key.get_secret_value())
        params: dict[str, object] = {
            **self._window_params(lookback_days=_ID_LOOKUP_LOOKBACK_DAYS),
            "limit": 1,
            "fields": _OBS_DETAIL_FIELDS,
            "filter": _observation_id_filter(observation_id),
        }

        resp = await self._get_with_retry(
            f"{base}{_OBS_V2_PATH}",
            params=params,
            auth=auth,
        )
        if resp is None or resp.status_code != _HTTP_OK:
            status = resp.status_code if resp is not None else None
            logger.warning(
                "trace_resolver.observation_failed instance=%s status=%s obs_id=%s",
                instance.name or base,
                status,
                observation_id,
            )
            raise TraceHttpError(
                f"observation detail request failed (status={status})",
                status_code=status,
            )

        try:
            rows, _cursor = _observation_page(resp)
        except Exception as exc:
            logger.warning(
                "trace_resolver.observation_parse_failed instance=%s error=%s",
                instance.name or base,
                exc,
            )
            raise TraceHttpError(
                f"observation detail parse failed: {exc}",
                status_code=_HTTP_OK,
            ) from exc
        if not rows:
            logger.info(
                "trace_resolver.observation_miss instance=%s obs_id=%s",
                instance.name or base,
                observation_id,
            )
            return None
        extra = _truncated_metadata_keys(rows)
        if extra:
            params["expandMetadata"] = ",".join(extra)
            resp = await self._get_with_retry(
                f"{base}{_OBS_V2_PATH}",
                params=params,
                auth=auth,
            )
            if resp is None or resp.status_code != _HTTP_OK:
                status = resp.status_code if resp is not None else None
                raise TraceHttpError(
                    f"observation detail request failed (status={status})",
                    status_code=status,
                )
            try:
                expanded, _cursor = _observation_page(resp)
            except Exception as exc:
                raise TraceHttpError(
                    f"observation detail parse failed: {exc}",
                    status_code=_HTTP_OK,
                ) from exc
            if expanded:
                rows = expanded
        return _normalize_observation(rows[0])

    async def fetch(self, trace_id: str) -> TraceDocument | None:
        """Smoke / debug: business id → primary (filter) → light detail.

        Uses ``fields=core,io`` so smoke can read envelope I/O without pulling
        nested observations (which 500s on large traces). Production tools use
        observations APIs with the primary id directly.
        """
        primary_id = await self.resolve_primary_id(trace_id)
        if primary_id is None:
            return None
        return await self.get_trace(primary_id, fields="core,io")
