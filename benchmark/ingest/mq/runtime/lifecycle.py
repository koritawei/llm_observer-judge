"""MQ consume lifecycle assembly — extracted from the ``benchmark.service.main`` lifespan to keep main small.

Wraps three things so ``benchmark.service.main`` only needs two call lines:

1. Read ``custom.mq_consumer`` from ``conf/app.<env>.yaml`` plus env overrides (``AGENT_EVAL_MQ_ENABLED`` /
   ``AGENT_EVAL_MQ_NAMESRV_DOMAIN``) and parse :class:`EvalMQConfig`.
2. Decide whether to start via :func:`should_start_consumer`; on start failure only warn and return ``None``,
   **without blocking HTTP startup** (health checks stay available).
3. Graceful shutdown (no-op when ``None``; a failed stop only warns and does not affect the later runtime / DB close).
"""

from typing import TYPE_CHECKING

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.ingest.mq.runtime.config import load_mq_config, should_start_consumer
from benchmark.ingest.mq.runtime.service import EvalMQConsumerService
from benchmark.paths import app_config_path

if TYPE_CHECKING:
    from benchmark.settings import Settings
    from benchmark.services.evaluation import EvaluatorService

logger = get_logger("agent_eval.mq.lifecycle")


async def start_eval_mq_consumer(
    *,
    evaluator_service: "EvaluatorService",
    settings: "Settings",
) -> EvalMQConsumerService | None:
    """Start eval-message consumption from conf plus env overrides.

    Return ``None`` when disabled (``enabled=false`` or no enabled subscription) or when start fails — the caller uses that
    to decide whether to attach it to ``app.state``, without knowing internal config or exception details.
    """

    try:
        cfg = load_mq_config(
            app_config_path(settings.env),
            namesrv_env=settings.mq_namesrv_domain,
            enabled_override=settings.mq_enabled,
        )
    except Exception:
        logger.error(
            "EvalMQConsumerService config load failed; HTTP serving continues",
            exc_info=True,
        )
        return None

    if not should_start_consumer(cfg):
        logger.info("EvalMQConsumerService not started (disabled or no consumers)")
        return None

    try:
        service = EvalMQConsumerService(
            evaluator_service=evaluator_service,
            config=cfg,
            env=settings.env,
        )
        await service.start()
    except Exception:
        logger.error(
            "EvalMQConsumerService failed to start; HTTP serving continues",
            exc_info=True,
        )
        return None
    return service


async def stop_eval_mq_consumer(service: EvalMQConsumerService | None) -> None:
    """Gracefully stop consumption (no-op when ``None``); a failed stop only warns and does not affect the later runtime / DB close."""

    if service is None:
        return
    try:
        await service.stop()
    except Exception:
        logger.warning("EvalMQConsumerService stop failed", exc_info=True)


__all__ = ["start_eval_mq_consumer", "stop_eval_mq_consumer"]
