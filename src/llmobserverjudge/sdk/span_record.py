"""In-memory span record before serialization to ingest contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class SpanRecord:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    sampled: bool = True
    status: str = "OK"
    kind: str = "OTHER"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    duration_ms: int = 0
    input: Any = None
    output: Any = None
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_ingest_dict(self, tenant_id: str = "unknown") -> dict[str, Any]:
        return {
            "tenant_id": tenant_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "timestamp": self.started_at.isoformat().replace("+00:00", "Z"),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "sampled": self.sampled,
            "attributes": {
                **self.attributes,
                **({"input": repr(self.input)[:2000]} if self.input is not None else {}),
                **({"output": repr(self.output)[:2000]} if self.output is not None else {}),
                **({"error": self.error} if self.error else {}),
            },
            "traceparent": (
                f"00-{self.trace_id}-{self.span_id}-{'01' if self.sampled else '00'}"
            ),
        }
