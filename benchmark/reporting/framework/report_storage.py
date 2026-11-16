"""Benchmark report storage client factory — uses RemoteConfig config from domain layer.

Storage config is loaded from RemoteConfig by :func:`fetch_benchmark_storage_config` in
:mod:`benchmark.domain.report_storage_config`. This module uses that config to build the
signed extension client in ``benchmark.libs.storage_ext`` (the main ``libs/storage`` does not
yet provide ``sign_url`` / ``upload(content_type=)``).

The default ``credential_mode=provider`` sends both upload and ``sign_url`` through Normandy
resource-bound credentials (long-lived AK, cached and renewed inside the SDK), so a 30-day
presigned URL works directly. Static AK/SK are not used.

``cdn_base_url`` may be omitted. When omitted, the access URL returned by ``upload`` falls
back in ``libs/storage`` to the public bucket domain
(``https://{bucket}.{public endpoint}/{key}``, an external URL). When set, CDN concatenation
is kept (consumer acceleration / custom domain).

Returns ``None`` when RemoteConfig is unavailable, required config is missing, or
``libs/storage`` is not installed, so the report has no durable URL (graceful degradation).
"""

from typing import TYPE_CHECKING, Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.report_storage_config import fetch_benchmark_storage_config

if TYPE_CHECKING:
    from benchmark.libs.storage import StorageClient

# libs/storage is an optional dependency: module-level try-import, degrade to None if unavailable.
# storage_ext transitively depends on storage symbols, so it is try-imported in the same batch.
try:
    from benchmark.libs.storage import parse_storage_config

    from benchmark.libs.storage_ext.signed_client import build_signed_client
except ImportError:  # pragma: no cover - optional dependency absent
    build_signed_client = None  # type: ignore[assignment]
    parse_storage_config = None  # type: ignore[assignment]

logger = get_logger("agent_eval.benchmark.report_storage")


def build_report_storage_client() -> "StorageClient | None":
    """Build ``StorageClient`` for benchmark HTML upload, or ``None``.

    Reads storage config from RemoteConfig (via domain :func:`fetch_benchmark_storage_config`).
    Returns ``None`` when ``report_upload_to_oss`` is ``False`` or a required field is missing
    (graceful degradation; the report has no durable URL).
    """
    cfg = fetch_benchmark_storage_config()

    if not cfg.report_upload_to_oss:
        return None

    if not (cfg.bucket and cfg.upload_endpoint):
        logger.warning(
            "RemoteConfig storage config: report_upload_to_oss=true but required "
            "fields incomplete (need bucket / upload_endpoint); "
            "reports will have no persistent URL",
        )
        return None

    mode = (cfg.credential_mode or "provider").strip() or "provider"
    # Tolerate leftover RemoteConfig static (the static AK/SK form is retired): normalize to provider,
    # so Normandy resource-bound credentials back the 30-day sign_url.
    if mode not in ("provider", "env"):
        logger.warning(
            "RemoteConfig storage config: unsupported credential_mode=%s; "
            "falling back to provider (Normandy)",
            mode,
        )
        mode = "provider"

    if build_signed_client is None or parse_storage_config is None:
        logger.warning(
            "benchmark.report_storage storage package unavailable; "
            "install libs/storage + oss2 + normandy-credential-sdk",
        )
        return None

    try:
        raw: dict[str, Any] = {
            "backend": "oss",
            "bucket": cfg.bucket,
            "upload_endpoint": cfg.upload_endpoint,
            "cdn_base_url": cfg.cdn_base_url,
            "object_key_prefix": cfg.object_key_prefix,
            "upload_timeout_ms": cfg.upload_timeout_ms,
            "download_timeout_ms": cfg.download_timeout_ms,
            "credential_mode": mode,
        }
        client = build_signed_client(parse_storage_config(raw))
    except Exception:
        logger.warning(
            "benchmark.report_storage client construction failed; reports will have no URL",
            exc_info=True,
        )
        return None

    logger.info(
        "benchmark.report_storage client ready bucket=%s endpoint=%s credential_mode=%s prefix=%s",
        cfg.bucket,
        cfg.upload_endpoint,
        mode,
        cfg.object_key_prefix or "(none)",
    )
    return client


def build_data_storage_client() -> "StorageClient | None":
    """Build ``StorageClient`` for OSS data-file download, or ``None``.

    Shares the same RemoteConfig (bucket / endpoint / credential) as
    :func:`build_report_storage_client`, but ``object_key_prefix`` is empty —
    data files (``ieval-service/benchmark/benchmark_result/{date}/
    benchmark_final_results.csv``) live outside the report prefix, so downloads must not
    preset the ``agent-eval/benchmark-reports`` prefix.
    """
    cfg = fetch_benchmark_storage_config()

    if not (cfg.bucket and cfg.upload_endpoint):
        logger.warning(
            "RemoteConfig storage config: data client required but required "
            "fields incomplete (need bucket / upload_endpoint)",
        )
        return None

    mode = (cfg.credential_mode or "provider").strip() or "provider"
    if mode not in ("provider", "env"):
        mode = "provider"

    if build_signed_client is None or parse_storage_config is None:
        logger.warning(
            "benchmark.data_storage storage package unavailable; "
            "install libs/storage + oss2 + normandy-credential-sdk",
        )
        return None

    try:
        raw: dict[str, Any] = {
            "backend": "oss",
            "bucket": cfg.bucket,
            "upload_endpoint": cfg.upload_endpoint,
            "cdn_base_url": cfg.cdn_base_url,
            "object_key_prefix": "",  # data-file paths do not include the report prefix
            "upload_timeout_ms": cfg.upload_timeout_ms,
            "download_timeout_ms": cfg.download_timeout_ms,
            "credential_mode": mode,
        }
        client = build_signed_client(parse_storage_config(raw))
    except Exception:
        logger.warning(
            "benchmark.data_storage client construction failed",
            exc_info=True,
        )
        return None

    logger.info(
        "benchmark.data_storage client ready bucket=%s endpoint=%s "
        "credential_mode=%s prefix=(none)",
        cfg.bucket,
        cfg.upload_endpoint,
        mode,
    )
    return client


__all__ = ["build_data_storage_client", "build_report_storage_client"]
