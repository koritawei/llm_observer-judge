"""Benchmark report storage config — loaded from RemoteConfig, no DB storage.

The backend is object storage (OSS). The medium is a fact, not part of the name (§4.6.3).

Domain layer. Contains:
  - :class:`BenchmarkStorageConfig` — OSS upload config model (Pydantic)
  - :func:`fetch_benchmark_storage_config` — domain service that loads OSS config from RemoteConfig

The RemoteConfig client singleton and cache live in :mod:`benchmark.settings.remote_config_store`.

RemoteConfig JSON shape::

    {
        "report_upload_to_oss": true,
        "bucket": "agent-eval-oss",
        "upload_endpoint": "oss-cn-zhangjiakou-internal.aliyuncs.com",
        "cdn_base_url": "https://agent-eval-oss.oss-cn-zhangjiakou.aliyuncs.com",
        "report_object_key_prefix": "agent-eval/benchmark-reports",
        "download_object_key_prefix": "ieval-service/benchmark/benchmark_result/",
        "upload_timeout_ms": 60000,
        "download_timeout_ms": 2000,
        "credential_mode": "provider"
    }

``report_object_key_prefix`` is the report upload prefix. ``download_object_key_prefix``
is the data-file download prefix. The old field ``object_key_prefix`` is still accepted:
when ``report_object_key_prefix`` is missing, fall back to ``object_key_prefix``.

``cdn_base_url`` may be omitted. When it is, the access URL returned by ``upload`` falls
back via ``build_oss_url`` in ``libs/storage`` to the public bucket host
(``https://{bucket}.{public endpoint}/{key}``, the external URL with ``-internal``
stripped) and no longer depends on CDN config. When set, CDN concatenation is kept
(client acceleration / custom domain).

Default ``credential_mode=provider``: upload, download, and ``sign_url`` all use the
Normandy resource-binding credential (a long-lived AK issued by normandy-credential-sdk
per bucket URN, cached and renewed inside the SDK), so a 30-day presigned URL works
directly. ``env`` reads process env vars (local/CI). Static AK/SK are not used.

Usage::

    from benchmark.domain.report_storage_config import (
        BenchmarkStorageConfig,
        fetch_benchmark_storage_config,
    )

    cfg = fetch_benchmark_storage_config()
    if not cfg.report_upload_to_oss:
        # upload disabled
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, model_validator

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.settings.setting import settings
from benchmark.domain._cache import get_or_compute

logger = get_logger(__name__)

#: RemoteConfig data_id for the benchmark OSS config.
BENCHMARK_STORAGE_DATA_ID = "agent-eval.benchmark.oss"
#: RemoteConfig group for the benchmark OSS config.
BENCHMARK_STORAGE_GROUP = "agent-eval.service.group"

_DEFAULT_REPORT_OBJECT_KEY_PREFIX = "agent-eval/benchmark-reports"
_DEFAULT_DOWNLOAD_OBJECT_KEY_PREFIX = "ieval-service/benchmark/benchmark_result"
_DEFAULT_UPLOAD_TIMEOUT_MS = 60_000
_DEFAULT_DOWNLOAD_TIMEOUT_MS = 2_000
#: Upload and sign_url both use the Normandy resource-binding credential (long-lived AK).
_DEFAULT_CREDENTIAL_MODE = "provider"


class BenchmarkStorageConfig(BaseModel):
    """OSS upload config — loaded from RemoteConfig and validated by Pydantic.

    Every field has a default. ``report_upload_to_oss`` defaults to ``False``, so upload
    is disabled automatically when RemoteConfig is unset or unavailable.

    Credentials are not part of the config: ``provider`` mode is issued by the Normandy
    SDK per bucket, and ``env`` mode reads process env vars. Leftover ``access_key_id`` /
    ``access_key_secret`` fields in RemoteConfig are silently ignored (``extra=ignore``).
    """

    model_config = {"extra": "ignore"}

    report_upload_to_oss: bool = False
    bucket: str = ""
    upload_endpoint: str = ""
    cdn_base_url: str = ""
    #: Report upload prefix (legacy ``object_key_prefix`` is still accepted).
    report_object_key_prefix: str = _DEFAULT_REPORT_OBJECT_KEY_PREFIX
    #: Data-file download prefix.
    download_object_key_prefix: str = _DEFAULT_DOWNLOAD_OBJECT_KEY_PREFIX
    upload_timeout_ms: int = _DEFAULT_UPLOAD_TIMEOUT_MS
    download_timeout_ms: int = _DEFAULT_DOWNLOAD_TIMEOUT_MS
    credential_mode: str = _DEFAULT_CREDENTIAL_MODE

    @model_validator(mode="before")
    @classmethod
    def _fallback_legacy_prefix(cls, data: object) -> object:
        """Legacy ``object_key_prefix`` fallback (see the module docstring)."""
        if (
            isinstance(data, dict)
            and "report_object_key_prefix" not in data
            and data.get("object_key_prefix") is not None
        ):
            data["report_object_key_prefix"] = data["object_key_prefix"]
        return data

    @property
    def object_key_prefix(self) -> str:
        """Backward-compat alias for ``report_object_key_prefix``."""
        return self.report_object_key_prefix


def fetch_benchmark_storage_config(
    *,
    data_id: str | None = None,
    group: str | None = None,
    use_cache: bool = True,
) -> BenchmarkStorageConfig:
    """Load and parse OSS config from RemoteConfig. Unavailable RemoteConfig returns defaults.

    Results use the parsed-result cache by default (TTL 300s, see :mod:`benchmark.domain._cache`).
    Pass ``use_cache=False`` to force a refresh.

    Args:
        data_id: RemoteConfig data_id, default read from settings.
        group: RemoteConfig group, default read from settings.
        use_cache: Whether to use the cache, default ``True``.
    """
    data_id = data_id or settings.benchmark_oss_remote_config_data_id
    group = group or settings.benchmark_oss_remote_config_group

    def _load_and_parse() -> BenchmarkStorageConfig:
        raw = fetch_cached_settings(data_id, group, use_cache=use_cache)
        if not raw:
            return BenchmarkStorageConfig()
        try:
            return BenchmarkStorageConfig.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "RemoteConfig OSS config parse failed (%s/%s): %s",
                data_id,
                group,
                exc,
            )
            return BenchmarkStorageConfig()

    return get_or_compute(
        (data_id, group),
        _load_and_parse,
        use_cache=use_cache,
    )


def get_default_download_prefix() -> str:
    """Construct ``oss://{bucket}/{download_object_key_prefix}`` from RemoteConfig config.

    Used as the default ``oss_prefix`` for the API / scheduler, so the path is not hardcoded.
    Falls back to the built-in default when RemoteConfig is unavailable.
    """
    cfg = fetch_benchmark_storage_config()
    prefix = cfg.download_object_key_prefix.strip("/")
    if not prefix:
        prefix = _DEFAULT_DOWNLOAD_OBJECT_KEY_PREFIX
    bucket = cfg.bucket or "agent-eval-oss"
    return f"oss://{bucket}/{prefix}"


__all__ = [
    "BENCHMARK_STORAGE_DATA_ID",
    "BENCHMARK_STORAGE_GROUP",
    "BenchmarkStorageConfig",
    "fetch_benchmark_storage_config",
    "get_default_download_prefix",
]
