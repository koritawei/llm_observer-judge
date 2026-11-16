"""Sidecar ingest lifespan helpers — keep ``benchmark.service.main`` thin."""

from typing import TYPE_CHECKING

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.paths import app_config_path
from benchmark.ingest.sidecar.config import load_sidecar_config
from benchmark.ingest.sidecar.oss_blob import OssBlob
from benchmark.ingest.sidecar.service import SidecarIngestDeps, SidecarIngestService

if TYPE_CHECKING:
    from benchmark.settings import Settings
    from benchmark.ingest.sidecar.processor import BenchmarkJudgeProtocol

logger = get_logger("agent_eval.sidecar_ingest.lifecycle")


async def start_sidecar_ingest(
    *,
    benchmark_judge: "BenchmarkJudgeProtocol",
    settings: "Settings",
) -> SidecarIngestService | None:
    try:
        cfg = load_sidecar_config(
            app_config_path(settings.env),
            namesrv_env=settings.sidecar_trace_namesrv_domain or settings.mq_namesrv_domain,
            enabled_override=settings.sidecar_trace_enabled,
        )
    except Exception:
        logger.error("SidecarIngestService config load failed", exc_info=True)
        return None
    if not cfg.enabled:
        logger.info("SidecarIngestService not started (disabled)")
        return None
    if settings.sidecar_trace_lindorm_table:
        cfg = cfg.model_copy(update={"lindorm_table": settings.sidecar_trace_lindorm_table})
    cfg = cfg.model_copy(update={"evaluate_timeout_s": settings.judge_evaluate_timeout_s})
    try:
        service = SidecarIngestService(
            SidecarIngestDeps(
                benchmark_judge=benchmark_judge,
                config=cfg,
                env=settings.env,
                lindorm_dsn=settings.sidecar_trace_lindorm_dsn,
                oss=OssBlob(
                    endpoint=settings.sidecar_trace_oss_endpoint,
                    access_key_id=settings.sidecar_trace_oss_access_key_id,
                    access_key_secret=settings.sidecar_trace_oss_access_key_secret,
                ),
            )
        )
        await service.start()
    except Exception:
        logger.error(
            "SidecarIngestService failed to start; HTTP serving continues",
            exc_info=True,
        )
        return None
    logger.info(
        "SidecarIngestService ready enabled=true active_consumers=%d lindorm_table=%s",
        service.active_consumer_count,
        cfg.lindorm_table,
    )
    return service


async def stop_sidecar_ingest(service: SidecarIngestService | None) -> None:
    if service is None:
        return
    try:
        await service.stop()
    except Exception:
        logger.warning("SidecarIngestService stop failed", exc_info=True)


__all__ = ["start_sidecar_ingest", "stop_sidecar_ingest"]
