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

    def _key(self, tenant_id: str, span_id: str, project_id: str = "default") -> str:
        return f"{tenant_id}:{project_id}:{span_id}"

    async def write_span(self, span: dict[str, Any]) -> None:
        project_id = str(span.get("project_id") or "default")
        key = self._key(str(span["tenant_id"]), str(span["span_id"]), project_id)
        row = dict(span)
        row["project_id"] = project_id
        self._rows[key] = row

    async def get_span(
        self, tenant_id: str, span_id: str, *, project_id: str = "default"
    ) -> dict[str, Any] | None:
        return self._rows.get(self._key(tenant_id, span_id, project_id))

    async def list_spans_by_trace(
        self,
        trace_id: str,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self._rows.values():
            if str(row.get("trace_id")) != trace_id:
                continue
            if tenant_id is not None and str(row.get("tenant_id")) != tenant_id:
                continue
            if project_id is not None and str(row.get("project_id") or "default") != project_id:
                continue
            out.append(dict(row))
        out.sort(key=lambda r: str(r.get("timestamp") or r.get("span_id") or ""))
        return out

    async def list_traces(
        self,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Aggregate spans into trace summaries (newest first)."""
        by_trace: dict[str, dict[str, Any]] = {}
        for row in self._rows.values():
            tid = str(row.get("trace_id") or "")
            if not tid:
                continue
            ten = str(row.get("tenant_id") or "")
            pid = str(row.get("project_id") or "default")
            if tenant_id is not None and ten != tenant_id:
                continue
            if project_id is not None and pid != project_id:
                continue
            prompt = str(row.get("prompt") or "")
            response = str(row.get("response") or "")
            tags = row.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            # Scope aggregation key by tenant+project+trace
            agg_key = f"{ten}:{pid}:{tid}"
            cur = by_trace.get(agg_key)
            if cur is None:
                by_trace[agg_key] = {
                    "trace_id": tid,
                    "tenant_id": ten,
                    "project_id": pid,
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
