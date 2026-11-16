"""LLM seam for the example evaluator — adapt a LangChain ChatModel to ``complete``.

The ChatModel is built by the runtime (``runtime.get_llm(model_key)``); this only
adapts it to the tiny :class:`EvaluatorLLM` seam (prompt in, text out) that tests
fake. Fetched lazily on first :meth:`complete`, so building the evaluator / booting
the runtime never needs a live model.

When a RemoteConfig ``model`` override is present, :func:`build_remote_config_llm` constructs
a ChatModel from the inline config (mirroring ``AppRuntime._build_llm_from_model_cfg``)
and wraps it in a :class:`ChatModelLLM`. Built instances are cached per config hash
so repeated evaluations with the same config reuse the same ChatModel.
"""

import hashlib
import os
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from benchmark.domain.schemas import RemoteConfigModelConfig

logger = get_logger("agent_eval.evaluator.example")


@runtime_checkable
class EvaluatorLLM(Protocol):
    """The minimal LLM seam the evaluator graph needs: prompt in, text out."""

    async def complete(self, prompt: str) -> str: ...


class ChatModelLLM:
    """Adapt a lazily-fetched LangChain ``BaseChatModel`` to :class:`EvaluatorLLM`.

    ``model_factory`` (typically ``lambda: runtime.get_llm(model_key)``) is called once,
    on the first :meth:`complete`.
    """

    def __init__(self, model_factory: Callable[[], BaseChatModel]) -> None:
        self._factory = model_factory
        self._model: BaseChatModel | None = None

    def _model_or_build(self) -> BaseChatModel:
        if self._model is None:
            self._model = self._factory()
        return self._model

    async def complete(self, prompt: str) -> str:
        response = await self._model_or_build().ainvoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", response)
        text = _coerce_llm_text(content)
        response_metadata = getattr(response, "response_metadata", {}) or {}
        usage_metadata = getattr(response, "usage_metadata", None)
        finish_reason = response_metadata.get("finish_reason")
        logger.info(
            "evaluator.example.llm_response content_len=%d content_type=%s "
            "finish_reason=%r usage_metadata=%r",
            len(text),
            type(content).__name__,
            finish_reason,
            usage_metadata,
        )
        if not text.strip() and finish_reason == "length":
            logger.warning(
                "evaluator.example.llm_truncated_empty finish_reason=length "
                "usage_metadata=%r — raise max_tokens if this persists",
                usage_metadata,
            )
        return text


def _coerce_llm_text(content: object) -> str:
    """Normalize ChatModel content to a plain string (str / multimodal list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


# ── RemoteConfig model override ────────────────────────────────────────────────

_REMOTE_CONFIG_LLM_CACHE_MAXSIZE = 32
_remote_config_llm_cache: OrderedDict[str, ChatModelLLM] = OrderedDict()


def _config_cache_key(cfg: RemoteConfigModelConfig) -> str:
    """Stable cache key from RemoteConfigModelConfig fields."""
    raw = cfg.model_dump_json(exclude_none=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_remote_config_llm(
    cfg: RemoteConfigModelConfig,
    *,
    business_id: str = "agent-eval",
    default_model_name: str = "",
) -> ChatModelLLM:
    """Build a :class:`ChatModelLLM` from a RemoteConfig ``model`` override.

    Mirrors ``AppRuntime._build_llm_from_model_cfg``: reads ``api_key_env``
    from the environment, assembles a provider config dict, and delegates to
    ``agentgraph.models.factory.build_chat_model``. Results are cached per
    config hash so repeated evaluations reuse the same ChatModel.

    ``default_model_name`` is the ``model_name`` from the YAML config
    (``models.<key>.model_name``); used as fallback when the RemoteConfig
    config omits ``modelName``.
    """
    key = _config_cache_key(cfg)
    cached = _remote_config_llm_cache.get(key)
    if cached is not None:
        return cached

    # Lazy import: agentgraph.models.factory pulls in langchain + tiktoken,
    # which are heavy and only needed when a RemoteConfig override is actually used.
    from benchmark.libs.agenthost.models.factory import (  # noqa: PLC0415 — import is delayed on purpose (see the comment above)
        build_chat_model,
    )

    api_key_env = cfg.api_key_env or "AGENT_EVAL_MODEL_API_KEY"
    api_key = os.environ.get(api_key_env, "") or ""  # noqa: TID251 — the evaluator plugin reads the secret env directly by api_key_env name (no config-layer injection channel)
    model_name = cfg.model_name or default_model_name
    provider_type = cfg.provider or "openai"
    base_url = cfg.base_url or ""
    timeout = float(cfg.timeout or 60.0)

    cfg_dict: dict[str, dict] = {
        "default_model": model_name,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "global": {"default_timeout": timeout},
        "providers": [
            {
                "name": provider_type,
                "type": provider_type,
                "api_key": api_key,
                "base_url": base_url,
                "models": [model_name],
            }
        ],
    }
    chat_model = build_chat_model(
        cfg_dict,
        business_id=business_id,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    llm = ChatModelLLM(lambda: chat_model)
    # Evict oldest entry if cache is full (FIFO eviction)
    if len(_remote_config_llm_cache) >= _REMOTE_CONFIG_LLM_CACHE_MAXSIZE:
        _remote_config_llm_cache.popitem(last=False)
    _remote_config_llm_cache[key] = llm
    return llm
