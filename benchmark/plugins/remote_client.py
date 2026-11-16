"""Unified remote-plugin HTTP client (async + sync).

契约见 :mod:`benchmark.domain.plugin_binding`。各 action 响应约定：

- ``fetch_output`` → ``{"output": <any>}``
- ``parse`` → ``{"items": [ ... ]}``
- ``generate`` → ``{"report_url"}`` / ``{"html"}``
- ``should_trigger`` / ``should_dispatch`` → ``{"result": true|false}``
- ``collect``（归因统计）→ ``{"result": { ... }}``
- ``extract``（trace 身份）→ ``{"project_id": "...", "trace_id": "..."}``
- ``fuse``（item 级融合）→ ``{"score": <float 0~1>}``
- ``build_payload``（统一日报 ReportPayload）→ ``{"payload": { ... }}``
"""

from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.plugin_binding import PluginBinding, PluginKind
from benchmark.utils.http_transport import AsyncClient, Client, HTTPError

logger = get_logger(__name__)

_HTTP_ERROR_STATUS_MIN = 400


class RemotePluginError(Exception):
    """A remote plugin call failed."""


def _build_request(
    binding: PluginBinding,
    *,
    kind: PluginKind | str,
    action: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], float]:
    if not binding.endpoint_url:
        raise RemotePluginError("endpoint_url is empty")
    body = {
        "plugin_kind": kind,
        "plugin_name": binding.name,
        "action": action,
        "payload": payload,
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if binding.bearer_token:
        headers["Authorization"] = f"Bearer {binding.bearer_token}"
    timeout_s = max(binding.timeout_ms, 1000) / 1000.0
    return body, headers, timeout_s


class _ResponseLike(Protocol):
    """Minimal read surface of httpx.Response (http_transport loads httpx dynamically and does not reference its type directly)."""

    status_code: int
    text: str

    def json(self) -> object: ...


def _parse_response(resp: _ResponseLike) -> dict[str, Any]:
    if resp.status_code >= _HTTP_ERROR_STATUS_MIN:
        raise RemotePluginError(
            f"http status={resp.status_code} body={resp.text[:500]!r}",
        )
    try:
        data = resp.json()
    except Exception as exc:
        raise RemotePluginError(f"invalid json response: {exc}") from exc
    if not isinstance(data, dict):
        raise RemotePluginError("response must be a JSON object")
    return data


class RemotePluginClient:
    """Issue one unified POST for any plugin_kind/action."""

    async def invoke(
        self,
        *,
        binding: PluginBinding,
        kind: PluginKind | str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        body, headers, timeout_s = _build_request(
            binding,
            kind=kind,
            action=action,
            payload=payload,
        )
        logger.info(
            "remote_plugin.request kind=%s name=%s action=%s url=%s timeout_s=%.1f",
            kind,
            binding.name,
            action,
            binding.endpoint_url,
            timeout_s,
        )
        try:
            async with AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(
                    binding.endpoint_url,
                    json=body,
                    headers=headers,
                )
        except HTTPError as exc:
            raise RemotePluginError(f"http error: {exc}") from exc
        return _parse_response(resp)

    def invoke_sync(
        self,
        *,
        binding: PluginBinding,
        kind: PluginKind | str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronous call (used when a policy Protocol method is synchronous)."""
        body, headers, timeout_s = _build_request(
            binding,
            kind=kind,
            action=action,
            payload=payload,
        )
        logger.info(
            "remote_plugin.sync_request kind=%s name=%s action=%s url=%s timeout_s=%.1f",
            kind,
            binding.name,
            action,
            binding.endpoint_url,
            timeout_s,
        )
        try:
            with Client(timeout=timeout_s) as client:
                resp = client.post(
                    binding.endpoint_url,
                    json=body,
                    headers=headers,
                )
        except HTTPError as exc:
            raise RemotePluginError(f"http error: {exc}") from exc
        return _parse_response(resp)


__all__ = [
    "RemotePluginClient",
    "RemotePluginError",
]
