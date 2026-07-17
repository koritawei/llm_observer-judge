"""HTTP / vLLM OpenAI-compatible Judge adapter (design §6.4)."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

import httpx

from llmobserverjudge.attribution_worker.ssrf import ToolEndpointValidator
from llmobserverjudge.eval_worker.parse import parse_judge_response
from llmobserverjudge.eval_worker.prompt import (
    DEFAULT_JUDGE_PROMPT,
    InMemoryPromptService,
    build_prompt_variables,
)

logger = logging.getLogger(__name__)


class HttpJudgeAdapter:
    """POST {base_url}/v1/chat/completions — OpenAI / vLLM compatible."""

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
            # Fail closed: whitelist + private-IP / DNS checks
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

    async def judge(
        self,
        *,
        agent_id: str,
        cfg: dict[str, Any],
        trace: dict[str, Any],
        upstream: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = cfg.get("resolved_prompt")
        if not prompt:
            vars_ = build_prompt_variables(
                agent_id=agent_id, cfg=cfg, trace=trace, upstream=upstream
            )
            prompt = InMemoryPromptService.render_template(DEFAULT_JUDGE_PROMPT, **vars_)

        model = cfg.get("model_name") or self.model_name
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict JSON-only LLM judge. Output one JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(cfg.get("temperature", 0.0)),
        }
        url = f"{self.base_url}/v1/chat/completions"
        client = await self._get_client()
        try:
            resp = await client.post(url, json=payload)
        except httpx.HTTPError as e:
            raise RuntimeError(f"http_judge_transport_error: {e}") from e

        if resp.status_code >= 500:
            raise RuntimeError(f"http_judge_5xx: {resp.status_code}")
        if resp.status_code >= 400:
            raise RuntimeError(f"http_judge_4xx: {resp.status_code} {resp.text[:200]}")

        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            return {
                "agent_id": agent_id,
                "score_status": "model_error",
                "error": f"invalid_chat_response: {e}",
                "error_type": "ResponseShapeError",
                "label": "PARSE_ERROR",
            }

        result = parse_judge_response(content, agent_id=agent_id, cfg=cfg)
        if result.get("score_status") == "scored":
            result["model_used"] = model
        return result


def http_judge_from_env(
    *,
    env: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> HttpJudgeAdapter | None:
    """Build adapter if LLMJUDGE_JUDGE_BASE_URL is set."""
    e = env if env is not None else dict(os.environ)
    base = (e.get("LLMJUDGE_JUDGE_BASE_URL") or "").strip()
    if not base:
        return None
    # Local transparent proxies often map public hosts to 198.18.0.0/15;
    # set LLMJUDGE_JUDGE_ENFORCE_SSRF=0 to allow those endpoints.
    enforce = (e.get("LLMJUDGE_JUDGE_ENFORCE_SSRF") or "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    return HttpJudgeAdapter(
        base_url=base,
        api_key=(e.get("LLMJUDGE_JUDGE_API_KEY") or None),
        model_name=e.get("LLMJUDGE_JUDGE_MODEL_NAME") or "default",
        timeout_s=float(e.get("LLMJUDGE_JUDGE_TIMEOUT") or 30),
        client=client,
        enforce_ssrf=enforce,
    )
