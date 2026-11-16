"""On-demand Langfuse observation access for a project.

Facade: :class:`TraceResolver`. Agent-facing tools: :class:`TraceToolProvider`
(+ :class:`TraceRequestBound` / :func:`bind_trace_request`).
See package README for API details.
"""

from benchmark.trace.http_backend import (
    TraceHttpError,
    TraceObservationsError,
)
from benchmark.trace.resolver import TraceResolver
from benchmark.trace.schema import (
    GenerationSnapshot,
    TraceEndpoint,
    TraceDocument,
    ObservationDetail,
    TraceSkeletonResult,
    TraceSpanSkeleton,
)
from benchmark.trace.tools import (
    TraceRequestBound,
    TraceToolProvider,
    bind_trace_request,
)

__all__ = [
    "GenerationSnapshot",
    "TraceHttpError",
    "TraceEndpoint",
    "TraceObservationsError",
    "TraceDocument",
    "TraceToolProvider",
    "ObservationDetail",
    "TraceRequestBound",
    "TraceResolver",
    "TraceSkeletonResult",
    "TraceSpanSkeleton",
    "bind_trace_request",
]
