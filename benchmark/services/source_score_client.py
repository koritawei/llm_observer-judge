"""Project-scoped Langfuse clients for source-trace score writeback."""

from typing import Protocol

from benchmark.libs.agenthost.observability.logger import get_logger
from langfuse import Langfuse

from benchmark.settings.trace_endpoints import TraceEndpoint, load_trace_instances

logger = get_logger("agent_eval.source_score_client")


class SourceScoreClientProvider(Protocol):
    """Resolves a Langfuse SDK client for the source trace's project."""

    def get_client(self, project_id: str) -> Langfuse | None: ...


class ProjectScoreClientProvider:
    """Build and cache Langfuse score clients keyed by source project ID."""

    def __init__(
        self,
        *,
        trace_instances: str,
        timeout_seconds: float,
    ) -> None:
        self._instances: dict[str, TraceEndpoint] = load_trace_instances(
            trace_instances
        )
        self._timeout_seconds = timeout_seconds
        self._clients: dict[str, Langfuse] = {}

    def get_client(self, project_id: str) -> Langfuse | None:
        client = self._clients.get(project_id)
        if client is not None:
            return client
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning(
                "langfuse_score.source_project_unconfigured project_id=%s", project_id
            )
            return None
        client = Langfuse(
            public_key=instance.public_key,
            secret_key=instance.secret_key.get_secret_value(),
            base_url=instance.base_url,
            timeout=self._timeout_seconds,
        )
        self._clients[project_id] = client
        return client


__all__ = ["ProjectScoreClientProvider", "SourceScoreClientProvider"]
