"""TraceResolver facade — the module's public orchestration entry point.

Provides on-demand access to Langfuse observations via four methods (three default + one opt-in):

- ``resolve_primary_id_for``: **entry-only** business metadata id → Langfuse primary.
- ``fetch_skeleton``: Langfuse primary id → lightweight span list via observations
  API only (no heavy ``GET /traces/{id}``).
- ``fetch_observation``: observation id → full detail.
- ``find_observations``: list observations for a primary id with optional filters.
- ``fetch_full_trace``: Langfuse primary id → full ``GET /traces/{id}`` (opt-in;
  default omits ``fields`` query for official all-fields payload).

Tools and DB paths must already hold the Langfuse primary id (rewritten at
eval entry). Do not call ``resolve_primary_id_for`` from those paths.
"""

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.trace.http_backend import (
    _FIRST_RESPONSE_METADATA_KEY,
    _METRIC_GENERATION_FIELDS,
    _METRIC_TRACE_FIELDS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RESOLVE_LOOKBACK_DAYS,
    DEFAULT_TIMEOUT_SECONDS,
    TraceHttpBackend,
    TraceHttpClient,
    ObservationListQuery,
    build_shared_http_client,
)
from benchmark.trace.schema import (
    TraceEndpoint,
    TraceDocument,
    ObservationDetail,
    ResolvedTrace,
    TraceSkeletonResult,
    TraceSpanSkeleton,
    _to_preview,
)

logger = get_logger("agent_eval.trace_resolver")

# Safety cap for find_observations pagination (Langfuse page size is 100).
FIND_MAX_OBSERVATIONS = 1000


def _span_time(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(int(value))
    return str(value)


def _build_span(obs: dict, preview_chars: int) -> TraceSpanSkeleton:
    """Map a raw API / sidecar observation dict to a ``TraceSpanSkeleton``."""
    raw_id = obs.get("id") or obs.get("span_id") or ""
    if not isinstance(raw_id, str):
        raw_id = str(raw_id)
    parent = (
        obs.get("parentObservationId")
        or obs.get("parent_span_id")
        or obs.get("parent_observation_id")
    )
    if parent is not None and not isinstance(parent, str):
        parent = str(parent)
    return TraceSpanSkeleton(
        id=raw_id,
        name=obs.get("name"),
        type=obs.get("type") or obs.get("observation_type"),
        status=obs.get("level") or obs.get("status"),
        parent_observation_id=parent,
        start_time=_span_time(obs.get("startTime") or obs.get("start_time")),
        end_time=_span_time(obs.get("endTime") or obs.get("end_time")),
        input_preview=_to_preview(obs.get("input"), preview_chars),
        output_preview=_to_preview(obs.get("output"), preview_chars),
    )


class TraceResolver:
    """On-demand Langfuse reader for the single instance owning ``project_id``."""

    def __init__(
        self,
        instances: dict[str, TraceEndpoint],
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: TraceHttpClient | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        resolve_lookback_days: int = DEFAULT_RESOLVE_LOOKBACK_DAYS,
    ) -> None:
        self._instances = instances
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._resolve_lookback_days = resolve_lookback_days
        # An injected client (tests) is borrowed; an owned one is created lazily
        # and closed on aclose().
        self._client = client
        self._owns_client = client is None

    def _ensure_client(self) -> TraceHttpClient:
        if self._client is None:
            self._client = build_shared_http_client(self._timeout)
        return self._client

    def _make_lf_client(self, instance: TraceEndpoint) -> TraceHttpBackend:
        from benchmark.trace.backend import open_trace_backend

        return open_trace_backend(
            instance,
            self._ensure_client(),
            max_retries=self._max_retries,
            resolve_lookback_days=self._resolve_lookback_days,
        )

    async def resolve(self, trace_id: str, project_id: str) -> ResolvedTrace | None:
        """Resolve business trace_id → full trace + instance context.

        Combines ``resolve_primary_id_for`` and ``fetch_full_trace`` into a
        single call, returning a :class:`ResolvedTrace` that carries both the
        trace data and the originating ``LangfuseInstance``.

        Returns ``None`` when the project_id is unknown, the primary id cannot
        be resolved, or the trace is not found (404).
        """
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning("trace_resolver.unknown_project project_id=%s", project_id)
            return None

        lf = self._make_lf_client(instance)
        primary_id = await lf.resolve_primary_id(trace_id)
        if primary_id is None:
            logger.warning("trace_resolver.resolve_no_primary trace_id=%s", trace_id)
            return None

        trace = await lf.get_trace(primary_id, fields="")
        if trace is None:
            logger.warning("trace_resolver.resolve_no_trace primary_id=%s", primary_id)
            return None

        return ResolvedTrace(trace=trace, instance=instance)

    async def resolve_primary_id_for(self, project_id: str, trace_id: str) -> str | None:
        """Entry-only: business ``metadata.trace_id`` → Langfuse primary id.

        Used by ``EvaluatorService.before_eval`` to rewrite the request. Tool
        paths must not call this — they receive the primary id already.
        """
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning("trace_resolver.unknown_project project_id=%s", project_id)
            return None
        lf = self._make_lf_client(instance)
        return await lf.resolve_primary_id(trace_id)

    async def fetch_skeleton(
        self,
        trace_id: str,
        project_id: str,
        *,
        max_spans: int = 300,
        preview_chars: int = 180,
        business_trace_id: str | None = None,
    ) -> TraceSkeletonResult | None:
        """Return a lightweight skeleton for a Langfuse **primary** ``trace_id``.

        Lists observations via ``GET /observations?traceId=...`` only — does
        **not** call ``GET /traces/{id}`` (default fields include nested
        observations and 500s on large traces). ``trace_id`` must already be
        the Langfuse primary id (rewritten at eval entry).

        Returns ``None`` only when ``project_id`` is unknown. A successful empty
        observation list yields ``spans=[]`` (not ``None``). Observation HTTP
        failures raise :class:`LangfuseObservationsError`.
        ``business_trace_id`` is optional audit echo (defaults to ``trace_id``).
        """
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning("trace_resolver.unknown_project project_id=%s", project_id)
            return None

        lf = self._make_lf_client(instance)
        primary_id = trace_id

        raw_obs = await lf.list_observations_for_trace(
            primary_id,
            ObservationListQuery(limit=max_spans + 1),
        )
        truncated = len(raw_obs) > max_spans
        obs_page = raw_obs[:max_spans]

        spans = [_build_span(obs, preview_chars) for obs in obs_page]
        return TraceSkeletonResult(
            primary_id=primary_id,
            business_trace_id=business_trace_id or trace_id,
            project_id=project_id,
            spans=spans,
            truncated=truncated,
            instance_name=instance.name,
        )

    async def fetch_observations_once(
        self,
        project_id: str,
        *,
        primary_id: str,
    ) -> list[dict]:
        """Fetch one bounded observations page for a legacy Judge-context fallback.

        This intentionally uses the existing paginated client with ``limit=100``:
        the client requests one page only, preventing the synchronous Kafka judge
        path from issuing an N+1 observation-detail fan-out.
        """

        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning("trace_resolver.unknown_project project_id=%s", project_id)
            return []
        return await self._make_lf_client(instance).list_observations_for_trace(
            primary_id,
            # VUI terminal-state selection needs an end time; only this fallback page extends the fields, and the public client default is unchanged.
            ObservationListQuery(limit=100, fields="core,basic,io,time"),
        )

    async def fetch_observation(
        self,
        project_id: str,
        observation_id: str,
        *,
        max_chars: int | None = None,
    ) -> ObservationDetail | None:
        """Return full detail for a single observation, or ``None`` on miss.

        If ``max_chars`` is given, both ``input`` and ``output`` are converted
        to truncated string previews.
        """
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning("trace_resolver.unknown_project project_id=%s", project_id)
            return None

        lf = self._make_lf_client(instance)
        raw = await lf.get_observation(observation_id)
        if raw is None:
            return None

        detail = ObservationDetail.model_validate(raw)
        if max_chars is not None:
            detail = detail.model_copy(
                update={
                    "input": _to_preview(detail.input, max_chars),
                    "output": _to_preview(detail.output, max_chars),
                }
            )
        return detail

    async def find_observations(
        self,
        project_id: str,
        *,
        primary_id: str,
        name: str | None = None,
        type: str | None = None,
        status: str | None = None,
    ) -> list[TraceSpanSkeleton]:
        """Return observations for a Langfuse primary id, with optional filters.

        ``name``, ``type``, and ``status`` (mapped to API ``level``) are sent
        as query params. Results are paged up to ``FIND_MAX_OBSERVATIONS``.
        """
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning("trace_resolver.unknown_project project_id=%s", project_id)
            return []

        lf = self._make_lf_client(instance)
        raw_obs = await lf.list_observations_for_trace(
            primary_id,
            ObservationListQuery(
                limit=FIND_MAX_OBSERVATIONS,
                name=name,
                type=type,
                level=status,
            ),
        )

        return [_build_span(obs, 180) for obs in raw_obs]

    async def fetch_full_trace(
        self,
        project_id: str,
        primary_id: str,
        *,
        fields: str | None = None,
    ) -> TraceDocument | None:
        """Use with care: fetch v2 observations by primary id and project them into a trace.

        An empty ``fields`` attaches observation rows (at most 1000). Framework tools are not exposed by default.
        Unknown project_id → None; no hit → None; HTTP failure raises LangfuseHttpError.
        """
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning("trace_resolver.unknown_project project_id=%s", project_id)
            return None
        lf = self._make_lf_client(instance)
        return await lf.get_trace(primary_id, fields=fields or "")

    async def fetch_metric_inputs(
        self,
        project_id: str,
        *,
        primary_id: str,
    ) -> tuple[TraceDocument | None, list[dict]]:
        """Return the raw performance-metric inputs for the same trace, with no business normalization."""
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning("trace_resolver.unknown_project project_id=%s", project_id)
            return None, []
        client = self._make_lf_client(instance)
        trace = await client.get_trace(
            primary_id,
            fields=_METRIC_TRACE_FIELDS,
            expand_metadata=_FIRST_RESPONSE_METADATA_KEY,
        )
        observations = await client.list_observations_for_trace(
            primary_id,
            ObservationListQuery(
                limit=FIND_MAX_OBSERVATIONS,
                type="GENERATION",
                fields=_METRIC_GENERATION_FIELDS,
            ),
        )
        return trace, observations

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
