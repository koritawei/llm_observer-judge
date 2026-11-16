"""agent-eval runtime: inject rate-limit backoff for every named model by default."""

from benchmark.libs.agenthost.app import AppRuntime
from benchmark.libs.agenthost.models.schemas import ModelConfigSchema

from benchmark.models import build_anthropic_chat_model
from benchmark.utils.model_retry import RetryingChatModel


class AgentEvalRuntime(AppRuntime):
    """Wrap ChatModel uniformly at the AppRuntime model-cache entry point."""

    def _build_llm_from_model_cfg(
        self,
        model_name: str,
        model_cfg: ModelConfigSchema,
        business_id: str,
    ) -> RetryingChatModel:
        if model_cfg.provider == "anthropic":
            model = build_anthropic_chat_model(model_cfg)
        else:
            model = super()._build_llm_from_model_cfg(
                model_name,
                model_cfg,
                business_id,
            )
        return RetryingChatModel(model)


__all__ = ["AgentEvalRuntime"]
