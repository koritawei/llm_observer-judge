"""Unit tests for load_benchmark_config."""

from pathlib import Path

import pytest
import yaml
from benchmark.runs.config import (
    WEBDEV_AGENT_URL_ENV,
    BenchmarkConfig,
    load_benchmark_config,
)
from pydantic import ValidationError


def test_load_benchmark_config_defaults_when_section_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WEBDEV_AGENT_URL_ENV, raising=False)
    path = tmp_path / "app.yaml"
    path.write_text("custom: {}\n", encoding="utf-8")
    cfg = load_benchmark_config(path)
    assert cfg.stale_heartbeat_timeout_s == 600
    assert cfg.default_max_concurrency == 10


def test_load_benchmark_config_from_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(WEBDEV_AGENT_URL_ENV, raising=False)
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.dump(
            {
                "custom": {
                    "benchmark": {
                        "evaluate_timeout_s": 60,
                        "stale_heartbeat_timeout_s": 90,
                        "webdev_agent_url": "http://example:9880/",
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    cfg = load_benchmark_config(path)
    assert cfg.stale_heartbeat_timeout_s == 90
    assert cfg.webdev_agent_url == "http://example:9880/"
    assert isinstance(cfg, BenchmarkConfig)


def test_load_benchmark_config_env_overrides_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BENCHMARK_WEBDEV_AGENT_URL (if set) overrides webdev_agent_url from YAML."""
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.dump(
            {"custom": {"benchmark": {"webdev_agent_url": "http://yaml:9880/"}}},
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(WEBDEV_AGENT_URL_ENV, "http://env:9880/")
    cfg = load_benchmark_config(path)
    assert cfg.webdev_agent_url == "http://env:9880/"
    # env must not affect the other fields
    assert cfg.evaluate_timeout_s == 300


def test_benchmark_config_stale_timeout_must_exceed_evaluate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When stale <= evaluate, reject at startup (fail fast, so an in-progress run is not falsely marked dead)."""
    monkeypatch.delenv(WEBDEV_AGENT_URL_ENV, raising=False)
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.dump(
            {
                "custom": {
                    "benchmark": {
                        "evaluate_timeout_s": 300,
                        "stale_heartbeat_timeout_s": 300,
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="stale_heartbeat_timeout_s"):
        load_benchmark_config(path)


def test_benchmark_config_stale_timeout_direct_construction() -> None:
    """Constructing the model directly is also blocked by model_validator (covers construction paths other than load)."""
    with pytest.raises(ValidationError, match="stale_heartbeat_timeout_s"):
        BenchmarkConfig(evaluate_timeout_s=600, stale_heartbeat_timeout_s=599)
