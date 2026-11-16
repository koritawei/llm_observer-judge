"""Unit tests for benchmark report storage factory (uses domain layer config)."""

from unittest.mock import MagicMock, patch

from benchmark.reporting.framework.report_storage import (
    build_data_storage_client,
    build_report_storage_client,
)


def test_build_none_when_remote_config_empty() -> None:
    """Domain returns default config (upload disabled) → None."""
    with patch(
        "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
        return_value=MagicMock(report_upload_to_oss=False),
    ):
        assert build_report_storage_client() is None


def test_build_none_when_upload_disabled() -> None:
    """Domain returns config with report_upload_to_oss=false → None."""
    cfg = MagicMock(report_upload_to_oss=False)
    with patch(
        "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
        return_value=cfg,
    ):
        assert build_report_storage_client() is None


def test_build_none_when_config_incomplete() -> None:
    """Domain returns config with upload=true but missing required fields → None.

    cdn_base_url is not required (when it is omitted the upload URL falls back to the public bucket domain);
    the config is incomplete only when bucket / upload_endpoint is missing.
    """
    cfg = MagicMock(
        report_upload_to_oss=True,
        bucket="b1",
        upload_endpoint="",
        cdn_base_url="https://cdn.example.com",
    )
    with patch(
        "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
        return_value=cfg,
    ):
        assert build_report_storage_client() is None


def test_build_wires_storage() -> None:
    """provider (a credential bound to a Normandy resource) → upload and the 30d sign_url both go through Normandy."""
    cfg = MagicMock(
        report_upload_to_oss=True,
        bucket="b1",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="https://cdn.example.com",
        object_key_prefix="agent-eval/benchmark-reports",
        upload_timeout_ms=60000,
        download_timeout_ms=2000,
        credential_mode="provider",
    )
    sentinel = MagicMock(name="storage_client")

    with (
        patch(
            "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
            return_value=cfg,
        ),
        patch(
            "benchmark.reporting.framework.report_storage.parse_storage_config",
            return_value={"ok": True},
        ) as mock_parse,
        patch(
            "benchmark.reporting.framework.report_storage.build_signed_client",
            return_value=sentinel,
        ) as mock_create,
    ):
        client = build_report_storage_client()

    assert client is sentinel
    mock_create.assert_called_once()
    raw = mock_parse.call_args.args[0]
    assert raw["bucket"] == "b1"
    assert raw["credential_mode"] == "provider"
    assert "access_key_id" not in raw
    assert "access_key_secret" not in raw
    assert raw["backend"] == "oss"


def test_build_without_cdn_base_url() -> None:
    """An omitted cdn_base_url does not block construction — the URL returned by upload falls back to the public bucket domain."""
    cfg = MagicMock(
        report_upload_to_oss=True,
        bucket="b1",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="",
        object_key_prefix="agent-eval/benchmark-reports",
        upload_timeout_ms=60000,
        download_timeout_ms=2000,
        credential_mode="provider",
    )
    sentinel = MagicMock(name="storage_client")

    with (
        patch(
            "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
            return_value=cfg,
        ),
        patch(
            "benchmark.reporting.framework.report_storage.parse_storage_config",
            return_value={"ok": True},
        ) as mock_parse,
        patch(
            "benchmark.reporting.framework.report_storage.build_signed_client",
            return_value=sentinel,
        ),
    ):
        client = build_report_storage_client()

    assert client is sentinel
    assert mock_parse.call_args.args[0]["cdn_base_url"] == ""


def test_build_normalizes_unsupported_mode_to_provider() -> None:
    """A leftover RemoteConfig credential_mode=static (already retired) → normalize to provider, and do not abort."""
    cfg = MagicMock(
        report_upload_to_oss=True,
        bucket="b1",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="https://cdn.example.com",
        object_key_prefix="p",
        upload_timeout_ms=60000,
        download_timeout_ms=2000,
        credential_mode="static",
    )
    sentinel = MagicMock(name="storage_client")

    with (
        patch(
            "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
            return_value=cfg,
        ),
        patch(
            "benchmark.reporting.framework.report_storage.parse_storage_config",
            return_value={"ok": True},
        ) as mock_parse,
        patch(
            "benchmark.reporting.framework.report_storage.build_signed_client",
            return_value=sentinel,
        ),
    ):
        client = build_report_storage_client()

    assert client is sentinel
    assert mock_parse.call_args.args[0]["credential_mode"] == "provider"


def test_build_none_when_storage_package_missing() -> None:
    """storage_ext / libs/storage is not installed (the module-level import degrades to None) → None."""
    cfg = MagicMock(
        report_upload_to_oss=True,
        bucket="b1",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="https://cdn.example.com",
        object_key_prefix="p",
        upload_timeout_ms=60000,
        download_timeout_ms=2000,
        credential_mode="provider",
    )
    with (
        patch(
            "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
            return_value=cfg,
        ),
        patch("benchmark.reporting.framework.report_storage.build_signed_client", None),
        patch("benchmark.reporting.framework.report_storage.parse_storage_config", None),
    ):
        assert build_report_storage_client() is None


def test_build_env_mode_passes_through() -> None:
    """credential_mode=env (local/CI) → passed through to libs/storage unchanged."""
    cfg = MagicMock(
        report_upload_to_oss=True,
        bucket="b1",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="https://cdn.example.com",
        object_key_prefix="p",
        upload_timeout_ms=60000,
        download_timeout_ms=2000,
        credential_mode="env",
    )
    sentinel = MagicMock(name="storage_client")

    with (
        patch(
            "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
            return_value=cfg,
        ),
        patch(
            "benchmark.reporting.framework.report_storage.parse_storage_config",
            return_value={"ok": True},
        ) as mock_parse,
        patch(
            "benchmark.reporting.framework.report_storage.build_signed_client",
            return_value=sentinel,
        ),
    ):
        client = build_report_storage_client()

    assert client is sentinel
    raw = mock_parse.call_args.args[0]
    assert raw["credential_mode"] == "env"
    assert "access_key_id" not in raw
    assert "access_key_secret" not in raw


# ════════════════════════════════════════════════════════════════════════════
# build_data_storage_client — the data-file download client (isolated from the report-upload prefix)
# ════════════════════════════════════════════════════════════════════════════


def test_data_client_none_when_config_incomplete() -> None:
    """Domain returns a config with incomplete fields → None (the data client does not check report_upload_to_oss)."""
    cfg = MagicMock(bucket="", upload_endpoint="", cdn_base_url="")
    with patch(
        "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
        return_value=cfg,
    ):
        assert build_data_storage_client() is None


def test_data_client_built_even_when_upload_disabled() -> None:
    """report_upload_to_oss=false does not block the data client — download is independent of the upload switch."""
    cfg = MagicMock(
        report_upload_to_oss=False,
        bucket="b1",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="https://cdn.example.com",
        upload_timeout_ms=60000,
        download_timeout_ms=2000,
        credential_mode="provider",
    )
    sentinel = MagicMock(name="storage_client")
    with (
        patch(
            "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
            return_value=cfg,
        ),
        patch(
            "benchmark.reporting.framework.report_storage.parse_storage_config",
            return_value={"ok": True},
        ),
        patch(
            "benchmark.reporting.framework.report_storage.build_signed_client",
            return_value=sentinel,
        ),
    ):
        assert build_data_storage_client() is sentinel


def test_data_client_wires_empty_object_key_prefix() -> None:
    """object_key_prefix is always the empty string — the caller (DailyDataLoader) passes the full key.

    Locked contract: the download prefix takes effect when the caller assembles the key via get_default_download_prefix();
    a prefix preset on the client would double the prefix and make every download 404.
    """
    cfg = MagicMock(
        bucket="b1",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="https://cdn.example.com",
        upload_timeout_ms=60000,
        download_timeout_ms=2000,
        credential_mode="provider",
    )
    with (
        patch(
            "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
            return_value=cfg,
        ),
        patch(
            "benchmark.reporting.framework.report_storage.parse_storage_config",
            return_value={"ok": True},
        ) as mock_parse,
        patch(
            "benchmark.reporting.framework.report_storage.build_signed_client",
            return_value=MagicMock(name="storage_client"),
        ),
    ):
        build_data_storage_client()

    assert mock_parse.call_args.args[0]["object_key_prefix"] == ""


def test_data_client_none_when_storage_package_missing() -> None:
    """libs/storage is not installed (the module-level import degrades to None) → None."""
    cfg = MagicMock(
        bucket="b1",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="https://cdn.example.com",
        upload_timeout_ms=60000,
        download_timeout_ms=2000,
        credential_mode="provider",
    )
    with (
        patch(
            "benchmark.reporting.framework.report_storage.fetch_benchmark_storage_config",
            return_value=cfg,
        ),
        patch("benchmark.reporting.framework.report_storage.build_signed_client", None),
        patch("benchmark.reporting.framework.report_storage.parse_storage_config", None),
    ):
        assert build_data_storage_client() is None
