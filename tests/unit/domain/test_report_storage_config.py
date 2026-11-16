"""oss_config domain service — parsing RemoteConfig OSS config."""

from unittest.mock import patch

from benchmark.domain.report_storage_config import (
    BENCHMARK_STORAGE_DATA_ID,
    BENCHMARK_STORAGE_GROUP,
    BenchmarkStorageConfig,
    fetch_benchmark_storage_config,
)


def test_remote_config_constants() -> None:
    assert BENCHMARK_STORAGE_DATA_ID == "agent-eval.benchmark.oss"
    assert BENCHMARK_STORAGE_GROUP == "agent-eval.service.group"


def test_benchmark_oss_config_defaults() -> None:
    cfg = BenchmarkStorageConfig()
    assert cfg.report_upload_to_oss is False
    assert cfg.bucket == ""
    assert cfg.upload_endpoint == ""
    assert cfg.cdn_base_url == ""
    assert cfg.object_key_prefix == "agent-eval/benchmark-reports"
    assert cfg.upload_timeout_ms == 60_000
    assert cfg.download_timeout_ms == 2_000
    assert cfg.credential_mode == "provider"


def test_benchmark_oss_config_ignores_extra_fields() -> None:
    cfg = BenchmarkStorageConfig.model_validate(
        {
            "report_upload_to_oss": True,
            "bucket": "b1",
            "unknown_field": "should_be_ignored",
        },
    )
    assert cfg.report_upload_to_oss is True
    assert cfg.bucket == "b1"


def test_benchmark_oss_config_ignores_legacy_ak_fields() -> None:
    """Leftover access_key_id/secret on RemoteConfig (static AK is retired) is ignored silently."""
    cfg = BenchmarkStorageConfig.model_validate(
        {
            "credential_mode": "static",
            "access_key_id": "AKID",
            "access_key_secret": "SKSEC",
        },
    )
    assert not hasattr(cfg, "access_key_id")
    assert not hasattr(cfg, "access_key_secret")
    assert "SKSEC" not in repr(cfg)


def test_fetch_returns_default_when_remote_config_empty() -> None:
    with patch(
        "benchmark.domain.report_storage_config.fetch_cached_settings",
        return_value={},
    ):
        cfg = fetch_benchmark_storage_config(
            data_id="test.id",
            group="test.group",
            use_cache=False,
        )

    assert cfg.report_upload_to_oss is False
    assert cfg.bucket == ""


def test_fetch_parses_remote_config_payload() -> None:
    raw = {
        "report_upload_to_oss": True,
        "bucket": "agent-eval-oss",
        "upload_endpoint": "oss-cn-zhangjiakou-internal.aliyuncs.com",
        "cdn_base_url": "https://agent-eval-oss.oss-cn-zhangjiakou.aliyuncs.com",
        "object_key_prefix": "custom/prefix",
        "upload_timeout_ms": 120000,
        "download_timeout_ms": 5000,
        "credential_mode": "provider",
    }
    with patch(
        "benchmark.domain.report_storage_config.fetch_cached_settings",
        return_value=raw,
    ):
        cfg = fetch_benchmark_storage_config(
            data_id="test.id",
            group="test.group",
            use_cache=False,
        )

    assert cfg.report_upload_to_oss is True
    assert cfg.bucket == "agent-eval-oss"
    assert cfg.upload_endpoint == "oss-cn-zhangjiakou-internal.aliyuncs.com"
    assert cfg.cdn_base_url == "https://agent-eval-oss.oss-cn-zhangjiakou.aliyuncs.com"
    assert cfg.object_key_prefix == "custom/prefix"
    assert cfg.upload_timeout_ms == 120000
    assert cfg.download_timeout_ms == 5000
    assert cfg.credential_mode == "provider"


def test_fetch_returns_default_on_parse_error() -> None:
    """RemoteConfig returns non-dict → graceful degradation to default config."""
    with patch(
        "benchmark.domain.report_storage_config.fetch_cached_settings",
        return_value={"report_upload_to_oss": "not-a-bool"},
    ):
        cfg = fetch_benchmark_storage_config(
            data_id="test.id",
            group="test.group",
            use_cache=False,
        )

    # Pydantic coerces "not-a-bool" to... actually it will raise ValueError
    # which is caught → default config returned
    assert cfg.report_upload_to_oss is False
