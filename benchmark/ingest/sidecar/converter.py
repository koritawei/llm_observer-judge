"""Build EvaluatorRequest from sidecar notify + unpacked observations."""

from typing import Any

from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY, PipelineSource
from benchmark.domain.schemas import EvalConfigEntry
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.streaming.metadata import set_metadata_agent_name
from benchmark.streaming.request import to_evaluator_request
from benchmark.streaming.schemas import ObservationForEval
from benchmark.ingest.sidecar.filter import declared_name_hit, resolved_agent_name
from benchmark.ingest.sidecar.notify import SidecarTraceNotify
from benchmark.ingest.sidecar.observations import (
    extract_agent_name,
    extract_project_id,
    extract_trace_fields,
)


class SidecarIdentityError(ValueError):
    """Unpacked tree is missing project_id or agent_name."""


def to_sidecar_evaluator_request(
    notify: SidecarTraceNotify,
    observations: list[Any],
    config: EvalConfigEntry,
) -> EvaluatorRequest:
    """Convert notify + observation tree into an evaluator request.

    TRACE-level ``input`` / ``output`` / ``metadata`` come from the IO-root
    span (empty ``parent_span_id``), same role as Kafka ``TraceRecord``.
    Stamps ``pipeline_source=sidecar``. Observation tree is passed via
    ``EvalRunOptions.sidecar_observations`` / ContextVar, not metadata.
    """
    project_id = extract_project_id(notify.project_id, observations)
# An entry's agentName is overwritten only when the name filter matches the whole string (same as kafka S3'/S8);
# a tags-only hit, or an unconfigured name, must not stamp the registered benchmark agent name.
    if declared_name_hit(notify, config):
        agent_name = (config.agent_name or "").strip()
    else:
        agent_name = resolved_agent_name(
            extract_agent_name(notify.agent_name, observations),
            project_id or "",
        )
    if not project_id or agent_name is None:
        raise SidecarIdentityError(
            f"missing eval identity project_id={project_id!r} agent_name={agent_name!r}"
        )
    inp, out, metadata = extract_trace_fields(observations)
    set_metadata_agent_name(metadata, agent_name)
    observation = ObservationForEval(
        span_id=notify.trace_id,
        trace_id=notify.trace_id,
        project_id=project_id,
        type="TRACE",
        name=notify.name or "",
        environment=notify.environment or "default",
        trace_name=notify.name,
        trace_timestamp=notify.timestamp,
        user_id=notify.user_id,
        session_id=notify.session_id,
        tags=list(notify.tags or []),
        input=inp,
        output=out,
        metadata=metadata,
    )
    request = to_evaluator_request(observation, config)
    stamped = dict(request.metadata) if request.metadata else {}
    stamped[PIPELINE_SOURCE_METADATA_KEY] = PipelineSource.SIDECAR.value
    stamped.pop("kafka", None)
    return request.model_copy(
        update={
            "metadata": stamped,
            "agent_name": agent_name,
            "project_id": project_id,
            "skip_trace_resolve": True,
        }
    )


__all__ = ["SidecarIdentityError", "to_sidecar_evaluator_request"]
