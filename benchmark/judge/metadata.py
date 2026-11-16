"""Agent identity helpers for Judge Trace / Observation metadata.

Shared by :mod:`benchmark.judge.consumer` (Kafka B1.5/B1.6) and
:mod:`benchmark.judge.service` (``to_evaluator_request``) so both paths
normalize ``agent_name`` the same way without importing consumer
private helpers.
"""

from typing import Any

from benchmark.judge.schemas import ObservationForEval

#: Metadata key for business agent identity on Kafka Trace / Observation.
#: Prefer snake_case ``agent_name`` (aligned with EvaluatorRequest / RemoteConfig registry).
AGENT_NAME_METADATA_KEY = "agent_name"

#: Legacy Langfuse-style key; still accepted when reading upstream traces.
LEGACY_AGENT_NAME_METADATA_KEY = "agentName"

#: Agent names that mean "identity unknown" — discard on Kafka path (no judge).
_UNKNOWN_AGENT_NAME_SENTINELS = frozenset({"unknown"})


def metadata_agent_name(metadata: dict[str, Any] | None) -> str | None:
    """Read agent name from metadata; prefer ``agent_name``, fall back to ``agentName``.

    Non-string dirty data is treated as missing (an agent name must be a str).
    """
    if not metadata:
        return None
    if AGENT_NAME_METADATA_KEY in metadata:
        value = metadata[AGENT_NAME_METADATA_KEY]
    elif LEGACY_AGENT_NAME_METADATA_KEY in metadata:
        value = metadata[LEGACY_AGENT_NAME_METADATA_KEY]
    else:
        return None
    return value if isinstance(value, str) else None


def set_metadata_agent_name(metadata: dict[str, Any], agent_name: str) -> None:
    """Write ``agent_name`` into metadata; drop legacy ``agentName`` if present."""
    metadata[AGENT_NAME_METADATA_KEY] = agent_name
    metadata.pop(LEGACY_AGENT_NAME_METADATA_KEY, None)


def observation_with_agent_name(
    observation: ObservationForEval,
    agent_name: str,
) -> ObservationForEval:
    """Return a copy of ``observation`` with ``metadata.agent_name`` stamped.

    Uses :meth:`~pydantic.BaseModel.model_copy` so the update is reliable —
    in-place mutation of ``observation.metadata`` is not always visible to
    downstream readers when the field was empty / shared / reassigned.
    """
    stamped = agent_name.strip()
    meta = dict(observation.metadata) if observation.metadata else {}
    if meta.get(AGENT_NAME_METADATA_KEY) == stamped and LEGACY_AGENT_NAME_METADATA_KEY not in meta:
        return observation
    set_metadata_agent_name(meta, stamped)
    return observation.model_copy(update={"metadata": meta})


def is_discard_agent_name(agent_name: object) -> bool:
    """True when agent name is missing/blank or the sentinel ``unknown``.

    Matches the Langfuse naming fallback in :func:`build_eval_trace_name`
    (empty → ``agent-eval.unknown``) and upstream ``metadata.agent_name=unknown``.
    """
    if not isinstance(agent_name, str):
        return True
    normalized = agent_name.strip().lower()
    return not normalized or normalized in _UNKNOWN_AGENT_NAME_SENTINELS


__all__ = [
    "AGENT_NAME_METADATA_KEY",
    "LEGACY_AGENT_NAME_METADATA_KEY",
    "is_discard_agent_name",
    "metadata_agent_name",
    "observation_with_agent_name",
    "set_metadata_agent_name",
]
