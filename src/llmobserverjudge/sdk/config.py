"""SDK global configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SDKConfig:
    api_key: str = ""
    endpoint: str = "http://127.0.0.1:8080"
    buffer_size: int = 4096
    flush_interval_s: float = 10.0
    flush_max_batch: int = 50
    sample_rate: float = 1.0
    telemetry_interval_s: float = 30.0
    enabled: bool = True


_config = SDKConfig(
    api_key=os.environ.get("LLMJUDGE_API_KEY", ""),
    endpoint=os.environ.get("LLMJUDGE_ENDPOINT", "http://127.0.0.1:8080"),
)


def get_config() -> SDKConfig:
    return _config


def configure(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    buffer_size: int | None = None,
    flush_interval_s: float | None = None,
    flush_max_batch: int | None = None,
    sample_rate: float | None = None,
    telemetry_interval_s: float | None = None,
    enabled: bool | None = None,
) -> SDKConfig:
    global _config
    if api_key is not None:
        _config.api_key = api_key
    if endpoint is not None:
        _config.endpoint = endpoint.rstrip("/")
    if buffer_size is not None:
        _config.buffer_size = buffer_size
    if flush_interval_s is not None:
        _config.flush_interval_s = flush_interval_s
    if flush_max_batch is not None:
        _config.flush_max_batch = flush_max_batch
    if sample_rate is not None:
        _config.sample_rate = sample_rate
    if telemetry_interval_s is not None:
        _config.telemetry_interval_s = telemetry_interval_s
    if enabled is not None:
        _config.enabled = enabled
    return _config
