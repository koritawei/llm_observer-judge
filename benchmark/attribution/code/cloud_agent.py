"""CloudAgent Open API skeleton. Unverified paths are concentrated here."""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from benchmark.attribution.code.models import ModuleConfig


class AuthError(RuntimeError):
    pass


class TransportError(RuntimeError):
    pass


@dataclass
class Approval:
    id: str
    kind: str = ""


@dataclass
class Events:
    is_processing: bool
    pending_approvals: list[Approval] = field(default_factory=list)
    final_text: str = ""
    has_more: bool = False
    next_cursor: str = ""


def is_readonly(approval: Approval) -> bool:
    """Read/write fields are unconfirmed. Only a standalone read word is approved; everything else is rejected."""
    kind = approval.kind.lower()
    if not kind:
        return False
    if any(word in kind for word in ("write", "edit", "delete", "bash", "exec", "patch")):
        return False
    return "read" in re.findall(r"[a-z]+", kind)


def _as_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and payload and isinstance(payload[-1], dict):
        return payload[-1]
    return {}


def session_id_from(payload: Any) -> str:
    data = _as_dict(payload)
    for key in ("id", "session_id"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = data.get("data")
    if isinstance(nested, dict):
        return session_id_from(nested)
    raise TransportError("session id missing")


def events_from(payload: Any) -> Events:
    data = _as_dict(payload)
    approvals: list[Approval] = []
    raw_approvals = data.get("pending_approvals") or []
    if isinstance(raw_approvals, list):
        for item in raw_approvals:
            if not isinstance(item, dict):
                continue
            approval_id = str(
                item.get("requestId") or item.get("id") or item.get("approval_id") or ""
            ).strip()
            if not approval_id:
                continue
            kind = str(item.get("kind") or item.get("type") or item.get("action") or "")
            approvals.append(Approval(id=approval_id, kind=kind))
    final_text = _last_assistant_text(data.get("events"))
    if not final_text:
        raw_text = data.get("final_text") or data.get("assistant_text") or ""
        final_text = raw_text if isinstance(raw_text, str) else ""
    if not final_text.strip():
        final_text = _last_assistant_text(data.get("messages"))
    next_cursor = ""
    raw_events = data.get("events")
    if isinstance(raw_events, list) and raw_events and isinstance(raw_events[-1], dict):
        cursor = raw_events[-1].get("cursor")
        if cursor is not None:
            next_cursor = str(cursor)
    return Events(
        is_processing=bool(data.get("is_processing")),
        pending_approvals=approvals,
        final_text=final_text if isinstance(final_text, str) else "",
        has_more=bool(data.get("has_more")),
        next_cursor=next_cursor,
    )


def _last_assistant_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    text = ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "") != "assistant":
            continue
        if str(item.get("type") or "text") != "text":
            continue
        content = item.get("content")
        if isinstance(content, str):
            text = content
    return text


class HttpTransport:
    def __init__(self, config: ModuleConfig, *, opener: Any | None = None) -> None:
        self._config = config
        self._opener = opener or urllib.request.urlopen

    def create_session(self, *, prompt: str) -> str:
        project = self._config.project_path
        body = {
            "provider": self._config.provider,
            "prompt": prompt,
            "project_path": project,
        }
        payload = self._request("POST", "/sessions", body)
        return session_id_from(payload)

    def poll_events(self, session_id: str) -> Events:
        cursor = ""
        final_text = ""
        approvals: list[Approval] = []
        processing = True
        while True:
            path = f"/sessions/{session_id}/events"
            if cursor:
                path += "?cursor=" + urllib.parse.quote(cursor, safe="")
            page = events_from(self._request("GET", path, None))
            processing = page.is_processing
            if page.pending_approvals:
                approvals = page.pending_approvals
            if page.final_text:
                final_text = page.final_text
            if not page.has_more or not page.next_cursor or page.next_cursor == cursor:
                break
            cursor = page.next_cursor
        return Events(
            is_processing=processing,
            pending_approvals=approvals,
            final_text=final_text,
        )

    def answer_approval(self, session_id: str, approval_id: str, *, approve: bool) -> None:
        self._request(
            "POST",
            f"/sessions/{session_id}/ask-user/{approval_id}/answer",
            {"approve": approve},
        )

    def stop(self, session_id: str) -> None:
        try:
            self._request("POST", f"/sessions/{session_id}/stop", {})
        except TransportError:
            return

    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", f"/sessions/{session_id}", None)

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> Any:
        base = self._config.base_url.rstrip("/")
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            base + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._config.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener(request, timeout=30) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise AuthError(f"aoned auth {exc.code}") from exc
            raise TransportError(f"aoned http {exc.code} {path}") from exc
        except urllib.error.URLError as exc:
            raise TransportError(f"aoned unreachable: {exc.reason}") from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode())
        except json.JSONDecodeError as exc:
            raise TransportError(f"aoned json {path}") from exc
