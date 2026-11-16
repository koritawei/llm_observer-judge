"""Streaming judge / triage / attribution plugin implementations (§5.1 extension).

Aliases are registered in ``execution_classes.STREAMING_*_CLASS_REFS``; this module holds
the default implementations for businesses such as amap / vehicle:

- triage: wraps the existing ``TriageClassifier`` (``classifier_key`` aligned with
  ``classifiers_registry``);
- attribution: declares ``execution`` + ``agent_key`` (overrides the
  ``attribution_<evaluator_name>`` default route); an ``async`` implementation returns immediately after ``submit``
  on the message path, and the real agent.call finishes in a background task.
"""

from typing import Any, Literal

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.runs.common.execution_classes import Execution
from benchmark.attribution.service.classifier import TriageClassifier
AmapAgentTriageClassifier = None
VehicleAgentTriageClassifier = None

logger = get_logger("agent_eval.benchmark.streaming.execution_plugins")

StreamingExecutionOverride = Literal["sync", "async", "none"]


class StreamingTriagePlugin:
    """Base class for streaming triage aliases: exposes a ``classifier`` instance for hooks to override."""

    provider: str = "local"
    classifier_key: str = ""

    def build_classifier(self) -> TriageClassifier:
        raise NotImplementedError


class AmapAgentStreamingTriage(StreamingTriagePlugin):
    """amap-agent triage level classifier (≡ classifiers_registry ``amap_agent``)."""

    classifier_key = "amap_agent"

    def build_classifier(self) -> TriageClassifier:
        return AmapAgentTriageClassifier()


class VehicleAgentStreamingTriage(StreamingTriagePlugin):
    """vehicle-agent triage level classifier."""

    classifier_key = "vehicle_agent"

    def build_classifier(self) -> TriageClassifier:
        return VehicleAgentTriageClassifier()


class StreamingAttributionPlugin:
    """Base class for streaming attribution aliases.

    Attributes:
        execution: ``sync`` / ``async`` / ``none`` (may be overridden by tasks[].attribution_execution).
        provider: execution-provider id.
        agent_key: when non-empty, overrides the ``attribution_<evaluator_name>`` route;
            ``None`` = still derived from evaluator_name (``evaluator_service_attribution``).
    """

    execution: Execution = "sync"
    provider: str = "local"
    agent_key: str | None = None

    async def submit(self, ctx: dict[str, Any]) -> str:
        """Submit entry for the async shape; the default raises (only ``execution=async`` classes must implement it)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support async submit",
        )


class EvaluatorServiceAttribution(StreamingAttributionPlugin):
    """Explicit form of the current synchronous attribution chain (route the attribution agent by evaluator_name)."""

    execution = "sync"
    provider = "local"
    agent_key = None


class AmapAgentStreamingAttribution(StreamingAttributionPlugin):
    """amap-agent synchronous attribution (fixed ``attribution_amap_agent``)."""

    execution = "sync"
    provider = "local"
    agent_key = "attribution_amap_agent"


class AmapAgentStreamingAsyncAttribution(StreamingAttributionPlugin):
    """amap-agent asynchronous attribution: return immediately after submit on the message path; the agent and dispatch run in the background.

    ``submit`` is scheduled as a background task by :class:`TriageAttributionService`;
    this class only declares ``execution=async`` + agent_key for the config side to compose.
    """

    execution = "async"
    provider = "local"
    agent_key = "attribution_amap_agent"

    async def submit(self, ctx: dict[str, Any]) -> str:
        """Return the correlation key (trace_id); the caller ``create_task`` does the real work."""
        trace_id = str(ctx.get("trace_id") or "")
        logger.info(
            "streaming.async_attribution_submitted trace_id=%s agent_key=%s",
            trace_id,
            self.agent_key,
        )
        return trace_id or "submitted"


class VehicleAgentStreamingAttribution(StreamingAttributionPlugin):
    """vehicle-agent synchronous attribution."""

    execution = "sync"
    provider = "local"
    agent_key = "attribution_vehicle_agent"


__all__ = [
    "AmapAgentStreamingAsyncAttribution",
    "AmapAgentStreamingAttribution",
    "AmapAgentStreamingTriage",
    "EvaluatorServiceAttribution",
    "StreamingAttributionPlugin",
    "StreamingExecutionOverride",
    "StreamingTriagePlugin",
    "VehicleAgentStreamingAttribution",
    "VehicleAgentStreamingTriage",
]
