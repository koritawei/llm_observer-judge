"""agent-eval service layer.

Holds business logic, including data access and domain services.
ORM models live in :mod:`benchmark.records.models`.

Usage:
    from benchmark.services import TraceRecordService

``BenchmarkService`` / ``TraceRecordService`` are exported lazily so that
``from benchmark.runs...`` does not eagerly load the run orchestration chain and form a
circular import with ``benchmark.domain.benchmark_console``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.runs.run_service import BenchmarkService as BenchmarkService
    from benchmark.services.trace_record_service import TraceRecordService as TraceRecordService

__all__ = [
    "BenchmarkService",
    "TraceRecordService",
]


def __getattr__(
    name: str,
) -> type["BenchmarkService"] | type["TraceRecordService"]:
    if name == "BenchmarkService":
        from benchmark.runs.run_service import (  # noqa: PLC0415
            BenchmarkService,
        )

        return BenchmarkService
    if name == "TraceRecordService":
        from benchmark.services.trace_record_service import (  # noqa: PLC0415
            TraceRecordService,
        )

        return TraceRecordService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
