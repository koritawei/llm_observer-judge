"""sidecar_trace yaml + env overlay (independent of custom.mq_consumer)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_DEFAULT_NAMESRV = "http://jmenv.tbsite.net:8080/rocketmq/nsaddr?nofix=1"
_PROD_NAMESRV = "http://jmenv.tbsite.net:8080/rocketmq/nsaddr-sh?nofix=1"


class SidecarTraceConfig(BaseModel):
    """``custom.sidecar_trace`` — sidecar-trace notify consumer."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    topic_prefix: str = "sidecar_trace_notify_"
    lindorm_table: str = "sidecar_trace"
    consumer_group: str = "CID_sidecar_trace_notify"
    namesrv_domain: str = ""
    max_buffer_size: int = Field(default=1000, gt=0)
    poll_timeout_s: float = Field(default=1.0, gt=0)
    handler_timeout_s: float = Field(default=300.0, gt=0)
    evaluate_timeout_s: float = Field(
        default=0.0,
        ge=0,
        description="0 = 不掐断评测（与 Kafka JUDGE_EVALUATE_TIMEOUT_S=0 相同）。",
    )


def load_sidecar_config(
    yaml_path: Path,
    *,
    namesrv_env: str = "",
    enabled_override: bool | None = None,
) -> SidecarTraceConfig:
    """Load ``custom.sidecar_trace`` from ``conf/app.<env>.yaml``."""
    with yaml_path.open() as handle:
        data = yaml.safe_load(handle) or {}
    raw = (data.get("custom") or {}).get("sidecar_trace") or {}
    cfg = SidecarTraceConfig.model_validate(raw)
    updates: dict[str, object] = {}
    if namesrv_env:
        updates["namesrv_domain"] = namesrv_env
    if enabled_override is not None:
        updates["enabled"] = enabled_override
    return cfg.model_copy(update=updates) if updates else cfg


def resolve_namesrv_domain(cfg: SidecarTraceConfig, env: str) -> str:
    if cfg.namesrv_domain:
        return cfg.namesrv_domain
    return _PROD_NAMESRV if env == "prod" else _DEFAULT_NAMESRV


__all__ = [
    "SidecarTraceConfig",
    "load_sidecar_config",
    "resolve_namesrv_domain",
]
