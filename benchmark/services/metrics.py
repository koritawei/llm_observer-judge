"""Runtime-metric collection and normalization for a Vehicle trace."""

from collections.abc import Mapping

from benchmark.pipeline.envelope import MetricObservation
from benchmark.pipeline.profiles import MetricBinding
from benchmark.trace.resolver import TraceResolver

METRIC_COLLECTOR_NAME = "vehicle_trace"
METRIC_SOURCE_TRACE_LATENCY = "trace.latency"
METRIC_SOURCE_FIRST_RESPONSE = "trace.metadata.first_text_response_time_ms"
METRIC_SOURCE_GENERATION = "observations[type=GENERATION]"
OBSERVATION_NAME_TO_SCOPE: dict[str, str] = {
    "model_call.unifiedIntentAgent": "cgi",
    "model_call.toolAgentReact": "react",
    "model_call.toolAgent": "react",
    "react.tool_select": "react",
    "model_call.verifyAgent": "verify",
    "cgi_classify": "cgi",
    "AmapChatModel.chat": "react",
}


class TraceMetricCollector:
    """Collect runtime metrics from a finished Vehicle trace for reuse by attribution and reports."""

    @staticmethod
    async def collect(
        *,
        resolver: TraceResolver,
        project_id: str,
        trace_id: str,
        bindings: Mapping[str, MetricBinding],
    ) -> list[MetricObservation]:
        trace, observations = await resolver.fetch_metric_inputs(
            project_id=project_id,
            primary_id=trace_id,
        )
        values = _metric_values(trace, observations)
        return [
            _observation(
                metric_key,
                values.get((metric_key, scope)),
                binding,
                sample_scope=scope,
            )
            for metric_key, binding in bindings.items()
            for scope in _scopes_for(metric_key, values)
        ]


def _metric_values(
    trace: object | None,
    observations: list[dict[str, object]],
) -> dict[tuple[str, str | None], float]:
    values: dict[tuple[str, str | None], float] = {}
    latency = _number(_value(trace, "latency"))
    if latency is not None:
        values[("total_latency_ms", None)] = float(round(latency * 1000))
    metadata = _value(trace, "metadata")
    if isinstance(metadata, Mapping):
        first_response = _number(metadata.get("first_text_response_time_ms"))
        if first_response is not None:
            values[("first_packet_latency_ms", None)] = first_response

    for item in observations:
        if not isinstance(item, Mapping):
            continue
        scope = OBSERVATION_NAME_TO_SCOPE.get(str(item.get("name") or ""))
        if scope is None:
            continue
        prompt = int(item.get("promptTokens") or 0)
        completion = int(item.get("completionTokens") or 0)
        total = int(item.get("totalTokens") or 0) or (prompt + completion)
        values[("llm_call_count", scope)] = values.get(("llm_call_count", scope), 0.0) + 1
        values[("llm_token_count", scope)] = values.get(("llm_token_count", scope), 0.0) + total
        observation_latency = _number(item.get("latency"))
        if observation_latency is not None:
            values[("llm_latency_ms", scope)] = values.get(("llm_latency_ms", scope), 0.0) + (
                observation_latency * 1000
            )
    return values


def _scopes_for(
    metric_key: str,
    values: Mapping[tuple[str, str | None], float],
) -> tuple[str | None, ...]:
    scopes = tuple(scope for key, scope in values if key == metric_key)
    return scopes or (None,)


def _observation(
    metric_key: str,
    value: float | None,
    binding: MetricBinding,
    *,
    sample_scope: str | None,
) -> MetricObservation:
    if value is None or binding.threshold is None:
        status = "unknown"
    elif value > binding.threshold:
        status = "exceeded"
    else:
        status = "within_target"
    source_field = _source_field(metric_key)
    return MetricObservation(
        metric_key=metric_key,
        value=value,
        unit="ms" if metric_key.endswith("_ms") else "count",
        status=status,
        threshold=binding.threshold,
        threshold_ref=binding.threshold_ref,
        evidence_refs=(
            [f"metric:{metric_key}:{sample_scope or 'case'}"]
            if value is not None
            else []
        ),
        collector_name=METRIC_COLLECTOR_NAME,
        source_field=source_field,
        sample_scope=sample_scope,
    )


def _source_field(metric_key: str) -> str:
    if metric_key == "total_latency_ms":
        return METRIC_SOURCE_TRACE_LATENCY
    if metric_key == "first_packet_latency_ms":
        return METRIC_SOURCE_FIRST_RESPONSE
    return METRIC_SOURCE_GENERATION


def _value(value: object | None, field: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(field)
    return getattr(value, field, None)


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def unknown_observations(
    bindings: Mapping[str, MetricBinding],
) -> list[MetricObservation]:
    """Keep an explicit unknown for a metric that could not be collected, instead of substituting zero."""
    return [
        _observation(metric_key, None, binding, sample_scope=None)
        for metric_key, binding in bindings.items()
    ]


__all__ = ["TraceMetricCollector", "unknown_observations"]
