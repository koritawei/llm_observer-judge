"""Shared OpenAI/vLLM-compatible chat completions client."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx

from llmobserverjudge.attribution_worker.ssrf import ToolEndpointValidator


class OpenAIChatClient:
    """POST {base_url}/v1/chat/completions."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model_name: str = "default",
        timeout_s: float = 30.0,
        allowed_hosts: list[str] | None = None,
        client: httpx.AsyncClient | None = None,
        enforce_ssrf: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_s = timeout_s
        host = urlparse(self.base_url).hostname or ""
        hosts = list(allowed_hosts or [])
        if host and host not in hosts:
            hosts.append(host)
        if enforce_ssrf:
            ToolEndpointValidator(allowed_hosts=hosts).validate(
                self.base_url if "://" in self.base_url else f"https://{host}/"
            )
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=self.timeout_s,
                headers=headers,
                follow_redirects=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": model or self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        url = f"{self.base_url}/v1/chat/completions"
        client = await self._get_client()
        try:
            resp = await client.post(url, json=payload)
        except httpx.HTTPError as e:
            raise RuntimeError(f"chat_transport_error: {e}") from e
        if resp.status_code >= 400:
            raise RuntimeError(f"chat_http_{resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise RuntimeError(f"invalid_chat_response: {e}") from e


def chat_client_from_env(
    *,
    env: dict[str, str] | None = None,
    prefix: str = "LLMJUDGE_JUDGE",
    client: httpx.AsyncClient | None = None,
    enforce_ssrf: bool | None = None,
) -> OpenAIChatClient | None:
    """Build from LLMJUDGE_JUDGE_* or LLMJUDGE_REACT_* env vars."""
    e = env if env is not None else dict(os.environ)
    base = (e.get(f"{prefix}_BASE_URL") or "").strip()
    if not base:
        return None
    if enforce_ssrf is None:
        # Prefer prefix-specific flag, then JUDGE_* fallback.
        raw = (
            e.get(f"{prefix}_ENFORCE_SSRF")
            or e.get("LLMJUDGE_JUDGE_ENFORCE_SSRF")
            or "1"
        )
        enforce_ssrf = raw.strip().lower() not in ("0", "false", "no")
    return OpenAIChatClient(
        base_url=base,
        api_key=e.get(f"{prefix}_API_KEY") or None,
        model_name=e.get(f"{prefix}_MODEL_NAME") or "default",
        timeout_s=float(e.get(f"{prefix}_TIMEOUT") or 30),
        client=client,
        enforce_ssrf=enforce_ssrf,
    )


def extract_json_object(content: str) -> dict[str, Any] | None:
    """Reuse judge parse helper for JSON blobs."""
    from llmobserverjudge.eval_worker.parse import extract_json_object as _extract

    return _extract(content)
