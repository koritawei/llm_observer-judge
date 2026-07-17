"""Span durable store abstraction (ClickHouse later)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SpanStore(Protocol):
    async def write_span(self, span: dict[str, Any]) -> None: ...

    async def get_span(self, tenant_id: str, span_id: str) -> dict[str, Any] | None: ...

    async def list_traces(
        self, *, tenant_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    async def list_spans_by_trace(
        self, trace_id: str, *, tenant_id: str | None = None
    ) -> list[dict[str, Any]]: ...


@dataclass
class InMemorySpanStore:
    _rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _key(self, tenant_id: str, span_id: str) -> str:
        return f"{tenant_id}:{span_id}"

    async def write_span(self, span: dict[str, Any]) -> None:
        key = self._key(str(span["tenant_id"]), str(span["span_id"]))
        self._rows[key] = dict(span)

    async def get_span(self, tenant_id: str, span_id: str) -> dict[str, Any] | None:
        return self._rows.get(self._key(tenant_id, span_id))

    async def list_spans_by_trace(
        self, trace_id: str, *, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self._rows.values():
            if str(row.get("trace_id")) != trace_id:
                continue
            if tenant_id is not None and str(row.get("tenant_id")) != tenant_id:
                continue
            out.append(dict(row))
        out.sort(key=lambda r: str(r.get("timestamp") or r.get("span_id") or ""))
        return out

    async def list_traces(
        self, *, tenant_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Aggregate spans into trace summaries (newest first).

        Includes Langfuse-style input/output previews from the latest root-ish span.
        """
        by_trace: dict[str, dict[str, Any]] = {}
        for row in self._rows.values():
            tid = str(row.get("trace_id") or "")
            if not tid:
                continue
            ten = str(row.get("tenant_id") or "")
            if tenant_id is not None and ten != tenant_id:
                continue
            prompt = str(row.get("prompt") or "")
            response = str(row.get("response") or "")
            tags = row.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            cur = by_trace.get(tid)
            if cur is None:
                by_trace[tid] = {
                    "trace_id": tid,
                    "tenant_id": ten,
                    "span_count": 1,
                    "status": row.get("status"),
                    "name": row.get("name"),
                    "kind": row.get("kind"),
                    "updated_at": row.get("timestamp"),
                    "sampled": row.get("sampled"),
                    "sample_reason": row.get("sample_reason"),
                    "duration_ms": row.get("duration_ms"),
                    "tags": list(tags),
                    "input_preview": _preview(prompt),
                    "output_preview": _preview(response),
                    "token_input": int(row.get("token_input") or 0),
                    "token_output": int(row.get("token_output") or 0),
                }
            else:
                cur["span_count"] = int(cur["span_count"]) + 1
                cur["token_input"] = int(cur.get("token_input") or 0) + int(
                    row.get("token_input") or 0
                )
                cur["token_output"] = int(cur.get("token_output") or 0) + int(
                    row.get("token_output") or 0
                )
                ts = row.get("timestamp")
                if ts and (not cur.get("updated_at") or str(ts) > str(cur["updated_at"])):
                    cur["updated_at"] = ts
                    cur["status"] = row.get("status") or cur.get("status")
                    cur["name"] = row.get("name") or cur.get("name")
                    cur["kind"] = row.get("kind") or cur.get("kind")
                    cur["duration_ms"] = row.get("duration_ms") or cur.get("duration_ms")
                    cur["tags"] = list(tags) or cur.get("tags") or []
                    if prompt:
                        cur["input_preview"] = _preview(prompt)
                    if response:
                        cur["output_preview"] = _preview(response)
        items = sorted(
            by_trace.values(),
            key=lambda r: str(r.get("updated_at") or ""),
            reverse=True,
        )
        return items[: max(1, min(limit, 500))]


def _preview(text: str, *, limit: int = 120) -> str:
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= limit:
        return t
    return t[: limit - 1] + "…"
