"""Luoshu-only: treat ``trace_id`` as the Langfuse trace primary key and fetch the root observation's input/output/metadata.

The ``trace_id`` written by the producer is itself the Langfuse primary key. When ``resolve_primary=False``, skip
metadata filtering and read Observations API v2 via :meth:`LangfuseClient.get_trace`.
"""

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.trace.http_backend import TraceHttpBackend, TraceHttpError
from benchmark.trace.resolver import TraceResolver
from benchmark.trace.schema import TraceEndpoint, TraceDocument

logger = get_logger("agent_eval.mq.consumers.luoshu.trace_fetch")


class _LuoshuTraceClient(TraceHttpBackend):
    """Add a "fetch top-level trace detail by primary key" method on top of the platform ``LangfuseClient``."""

    async def fetch_trace_detail(self, primary_id: str) -> TraceDocument | None:
        """Read the root observation by Langfuse primary key and project a trace with input / output / metadata.

        Return ``None`` on a miss or an HTTP failure.
        """
        try:
            return await self.get_trace(primary_id, fields="core,basic,io,metadata")
        except TraceHttpError:
            logger.warning(
                "luoshu.trace_detail_failed instance=%s primary_id=%s",
                self._instance.name or self._instance.base_url,
                primary_id,
                exc_info=True,
            )
            return None


class LuoshuTraceResolver(TraceResolver):
    """Luoshu-only ``TraceResolver``: adds :meth:`fetch_trace` (fetch top-level trace I/O/metadata by primary key).

    Reuses the platform ``TraceResolver`` httpx client lifecycle and instance routing (``_instances`` /
    ``_ensure_client`` / ``aclose``, and similar), and only swaps the created ``LangfuseClient`` for
    :class:`_LuoshuLangfuseClient` so :meth:`fetch_trace_detail` is available.
    """

    def _make_lf_client(self, instance: TraceEndpoint) -> _LuoshuTraceClient:
        return _LuoshuTraceClient(
            instance, self._ensure_client(), max_retries=self._max_retries
        )

    async def fetch_trace(
        self,
        trace_id: str,
        project_id: str,
        *,
        resolve_primary: bool = True,
    ) -> TraceDocument | None:
        """Return top-level trace detail (input/output/metadata), or ``None`` on a miss.

        When ``resolve_primary`` is True (the default), first resolve the business ``trace_id`` to a
        Langfuse primary key via a ``metadata`` filter; for Luoshu (the producer ``trace_id`` **is** the Langfuse trace primary key) pass
        ``resolve_primary=False`` to fetch by primary key and skip the metadata filter. Unknown project or a miss returns
        ``None``.
        """
        instance = self._instances.get(project_id)
        if instance is None:
            logger.warning(
                "luoshu.trace_fetch.unknown_project project_id=%s", project_id
            )
            return None

        lf = self._make_lf_client(instance)
        if resolve_primary:
            primary_id = await lf.resolve_primary_id(trace_id)
            if primary_id is None:
                return None
        else:
            primary_id = trace_id

        return await lf.fetch_trace_detail(primary_id)


__all__ = ["LuoshuTraceResolver"]
