"""Judge adapters + fallback router (SDD-EV-22 / design §6.4)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class JudgeAdapter(Protocol):
    async def judge(
        self,
        *,
        agent_id: str,
        cfg: dict[str, Any],
        trace: dict[str, Any],
        upstream: dict[str, Any],
    ) -> dict[str, Any]: ...


class RuleJudgeAdapter:
    """Deterministic Code Eval — heuristic scores from trace_summary (no LLM)."""

    async def judge(
        self,
        *,
        agent_id: str,
        cfg: dict[str, Any],
        trace: dict[str, Any],
        upstream: dict[str, Any],
    ) -> dict[str, Any]:
        _ = upstream
        scale_max = float(cfg.get("scale_max", 10) or 10)
        threshold = float(cfg.get("threshold", 0.5))
        weight = float(cfg.get("weight", 1.0))
        raw, label, reasoning = _heuristic_score(agent_id, trace, scale_max)
        normalized = raw / scale_max if scale_max > 0 else 0.0
        return {
            "agent_id": agent_id,
            "score_status": "scored",
            "score": raw,
            "scale_max": scale_max,
            "threshold": threshold,
            "weight": weight,
            "normalized_score": round(normalized, 4),
            "label": label,
            "reasoning": reasoning,
            "model_used": cfg.get("model", "rule_judge"),
        }


class ScriptedJudgeAdapter:
    """Test double: fixed raw scores / statuses by agent_id."""

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        *,
        statuses: dict[str, str] | None = None,
        errors: dict[str, Exception] | None = None,
        latency_s: float = 0.0,
    ) -> None:
        self.scores = scores or {}
        self.statuses = statuses or {}
        self.errors = errors or {}
        self.latency_s = latency_s

    async def judge(
        self,
        *,
        agent_id: str,
        cfg: dict[str, Any],
        trace: dict[str, Any],
        upstream: dict[str, Any],
    ) -> dict[str, Any]:
        _ = trace, upstream
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if agent_id in self.errors:
            raise self.errors[agent_id]
        status = self.statuses.get(agent_id, "scored")
        if status != "scored":
            return {
                "agent_id": agent_id,
                "score_status": status,
                "error": "scripted",
                "error_type": "Scripted",
            }
        scale_max = float(cfg.get("scale_max", 10) or 10)
        raw = float(self.scores.get(agent_id, 8.0))
        normalized = raw / scale_max if scale_max > 0 else 0.0
        return {
            "agent_id": agent_id,
            "score_status": "scored",
            "score": raw,
            "scale_max": scale_max,
            "threshold": float(cfg.get("threshold", 0.5)),
            "weight": float(cfg.get("weight", 1.0)),
            "normalized_score": round(normalized, 4),
            "label": "scripted",
            "reasoning": "scripted judge",
            "model_used": cfg.get("model", "scripted"),
        }


class FailingJudgeAdapter:
    """Always raises — used as primary in fallback tests."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or TimeoutError("simulated judge timeout")

    async def judge(
        self,
        *,
        agent_id: str,
        cfg: dict[str, Any],
        trace: dict[str, Any],
        upstream: dict[str, Any],
    ) -> dict[str, Any]:
        _ = agent_id, cfg, trace, upstream
        raise self.exc


class JudgeModelRouter:
    """Primary model + fallback_models chain; ultimate infra_error if all fail."""

    def __init__(
        self,
        registry: dict[str, JudgeAdapter] | None = None,
        *,
        default_adapter: JudgeAdapter | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.registry = dict(registry or {})
        self.default_adapter = default_adapter or RuleJudgeAdapter()
        self.timeout_s = timeout_s

    def resolve(self, model_id: str) -> JudgeAdapter:
        return self.registry.get(model_id, self.default_adapter)

    async def invoke_with_fallback(
        self,
        *,
        agent_id: str,
        cfg: dict[str, Any],
        trace: dict[str, Any],
        upstream: dict[str, Any],
    ) -> dict[str, Any]:
        primary = cfg.get("model") or "default-judge"
        chain = [primary, *list(cfg.get("fallback_models") or [])]
        # Ultimate deterministic fallback if not already listed
        if "code_eval" not in chain:
            chain.append("code_eval")

        last_error: Exception | None = None
        for model_id in chain:
            adapter = self.resolve(model_id)
            try:
                timeout = float(cfg.get("timeout", self.timeout_s))
                result = await asyncio.wait_for(
                    adapter.judge(
                        agent_id=agent_id, cfg=cfg, trace=trace, upstream=upstream
                    ),
                    timeout=timeout,
                )
                if result.get("score_status") == "scored":
                    result["model_used"] = model_id
                    return result
                if result.get("score_status") in ("infra_error", "model_error"):
                    last_error = RuntimeError(result.get("error") or result["score_status"])
                    logger.warning(
                        "judge_status_not_scored model=%s status=%s",
                        model_id,
                        result.get("score_status"),
                    )
                    continue
                return result
            except TimeoutError as e:
                last_error = e
                logger.warning("model_timeout model=%s agent=%s", model_id, agent_id)
            except Exception as e:
                last_error = e
                logger.warning(
                    "model_error model=%s agent=%s err=%s", model_id, agent_id, e
                )

        return {
            "agent_id": agent_id,
            "score_status": "infra_error",
            "error": str(last_error) if last_error else "all_models_failed",
            "error_type": type(last_error).__name__ if last_error else "JudgeError",
        }


def default_judge_router() -> JudgeModelRouter:
    rule = RuleJudgeAdapter()
    return JudgeModelRouter(
        registry={
            "default-judge": rule,
            "code_eval": rule,
            "rule": rule,
        },
        default_adapter=rule,
    )


def build_judge_router(
    *,
    http_adapters: dict[str, JudgeAdapter] | None = None,
    include_env_http: bool = False,
) -> JudgeModelRouter:
    """Rule defaults + optional HTTP/vLLM model ids."""
    rule = RuleJudgeAdapter()
    registry: dict[str, JudgeAdapter] = {
        "default-judge": rule,
        "code_eval": rule,
        "rule": rule,
    }
    if http_adapters:
        registry.update(http_adapters)
    if include_env_http:
        from llmobserverjudge.eval_worker.http_judge import http_judge_from_env

        adapter = http_judge_from_env()
        if adapter is not None:
            model_id = (
                __import__("os").environ.get("LLMJUDGE_JUDGE_MODEL_ID") or "http-judge"
            )
            registry[model_id] = adapter
            # Prefer HTTP as default-judge when env configured
            registry["default-judge"] = adapter
    return JudgeModelRouter(registry=registry, default_adapter=rule)


def _heuristic_score(
    agent_id: str, trace: dict[str, Any], scale_max: float
) -> tuple[float, str, str]:
    status = str(trace.get("status") or "OK").upper()
    tags = trace.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    response = str(trace.get("response") or "")
    prompt = str(trace.get("prompt") or "")

    if status == "ERROR":
        return scale_max * 0.2, "ERROR_TRACE", "trace status=ERROR"

    aid = agent_id.lower()
    if "safety" in aid:
        risky = any(t in tags for t in ("toxicity", "jailbreak", "pii_leak"))
        if risky or _contains_risky(response):
            return scale_max * 0.3, "SAFETY_RISK", "safety heuristic flagged risk"
        return scale_max * 0.9, "SAFETY_OK", "safety heuristic clear"

    if "code" in aid:
        if "tool_error" in tags or "exception" in response.lower():
            return scale_max * 0.25, "CODE_FAIL", "code heuristic saw tool/exception"
        return scale_max * 0.85, "CODE_OK", "code heuristic ok"

    if "quality" in aid or "fast" in aid:
        length = len(response) or len(prompt)
        if length == 0:
            return scale_max * 0.4, "EMPTY", "empty response/prompt"
        if length < 8:
            return scale_max * 0.55, "SHORT", "very short output"
        return scale_max * 0.8, "QUALITY_OK", "quality heuristic ok"

    return scale_max * 0.75, "DEFAULT_OK", "default heuristic"


_RISKY_MARKERS = ("ignore previous", "jailbreak", "ssn:", "api_key=")


def _contains_risky(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in _RISKY_MARKERS)
