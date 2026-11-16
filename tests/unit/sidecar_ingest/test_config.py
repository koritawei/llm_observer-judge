"""sidecar_trace yaml loader."""

from pathlib import Path

import pytest
from benchmark.ingest.sidecar.config import (
    SidecarTraceConfig,
    load_sidecar_config,
    resolve_namesrv_domain,
)
from pydantic import ValidationError


def test_load_sidecar_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "app.daily.yaml"
    path.write_text(
        """
custom:
  sidecar_trace:
    enabled: true
    topic_prefix: sidecar_trace_notify_
    lindorm_table: sidecar_trace
    consumer_group: CID_sidecar_trace_notify
""",
        encoding="utf-8",
    )
    cfg = load_sidecar_config(path)
    assert cfg.enabled is True
    assert cfg.topic_prefix == "sidecar_trace_notify_"
    assert cfg.lindorm_table == "sidecar_trace"


def test_load_sidecar_config_env_override(tmp_path: Path) -> None:
    path = tmp_path / "app.daily.yaml"
    path.write_text("custom:\n  sidecar_trace:\n    enabled: true\n", encoding="utf-8")
    cfg = load_sidecar_config(path, enabled_override=False)
    assert cfg.enabled is False


def test_missing_section_defaults(tmp_path: Path) -> None:
    path = tmp_path / "app.daily.yaml"
    path.write_text("custom: {}\n", encoding="utf-8")
    cfg = load_sidecar_config(path)
    assert cfg == SidecarTraceConfig()
    assert cfg.enabled is False
    assert cfg.topic_prefix == "sidecar_trace_notify_"
    assert cfg.lindorm_table == "sidecar_trace"


def test_old_topic_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "app.daily.yaml"
    path.write_text(
        "custom:\n  sidecar_trace:\n    topic: sidecar_trace_notify\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_sidecar_config(path)


def test_resolve_namesrv_domain_prod_fallback() -> None:
    cfg = SidecarTraceConfig()
    assert "nsaddr-sh" in resolve_namesrv_domain(cfg, "prod")
    assert "nsaddr-sh" not in resolve_namesrv_domain(cfg, "daily")


def test_resolve_namesrv_domain_explicit() -> None:
    cfg = SidecarTraceConfig(namesrv_domain="http://namesrv.test")
    assert resolve_namesrv_domain(cfg, "prod") == "http://namesrv.test"


@pytest.mark.parametrize(
    "name",
    ["app.daily.yaml", "app.pre.yaml", "app.prod.yaml"],
)
def test_committed_yaml_uses_sidecar_defaults(name: str) -> None:
    path = Path(__file__).resolve().parents[3] / "conf" / name
    cfg = load_sidecar_config(path)
    assert cfg == SidecarTraceConfig()
    assert cfg.enabled is False
    assert cfg.lindorm_table == "sidecar_trace"
