"""Read-only OSS GET for sidecar-trace observation blobs."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_RETRYABLE_CLIENT_STATUSES = frozenset({401, 403})
_HTTP_SERVER_ERROR = 500
_RETRY_BACKOFF_S = 0.2


@dataclass(frozen=True)
class _BucketCreds:
    access_key_id: str
    access_key_secret: str
    endpoint: str
    security_token: str | None = None


def _normalize_endpoint(endpoint: str) -> str:
    ep = (endpoint or "").strip()
    if not ep:
        return ""
    if "://" not in ep:
        return f"https://{ep}"
    return ep


def _creds_from_normandy(bucket_name: str, *, endpoint_override: str) -> _BucketCreds:
    try:
        from benchmark.libs.credential import factory, names  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "normandy-credential-sdk is not installed; cannot resolve OSS credentials"
        ) from exc

    try:
        provider = factory.get_default_credential_provider()
        urn = names.of_aliyun_oss_bucket_name(bucket_name)
        credential = provider.get_credential(urn)
    except Exception as exc:
        raise RuntimeError(f"Normandy OSS credential failed bucket={bucket_name}: {exc}") from exc

    ak = getattr(credential, "access_key_id", None) or ""
    sk = getattr(credential, "access_key_secret", None) or ""
    token = getattr(credential, "security_token", None) or None
    if not ak or not sk:
        raise RuntimeError(f"Normandy returned empty AK bucket={bucket_name}")
    endpoint = endpoint_override
    if not endpoint:
        try:
            from benchmark.libs.credential import helpers  # noqa: PLC0415

            endpoint = _normalize_endpoint(str(helpers.get_endpoint(urn) or ""))
        except Exception:
            endpoint = ""
    if not endpoint:
        raise RuntimeError(
            f"OSS endpoint missing bucket={bucket_name}; "
            "set SIDECAR_TRACE_OSS_ENDPOINT or enable Normandy endpoint delivery"
        )
    return _BucketCreds(
        access_key_id=str(ak),
        access_key_secret=str(sk),
        endpoint=endpoint,
        security_token=str(token) if token else None,
    )


class OssBlob:
    """GET zstd observation objects. Bucket comes from the Lindorm row."""

    def __init__(
        self,
        *,
        endpoint: str = "",
        access_key_id: str = "",
        access_key_secret: str = "",
        timeout: float = 60.0,
    ) -> None:
        explicit_ak = access_key_id.strip()
        explicit_sk = access_key_secret.strip()
        self._use_normandy = not (explicit_ak and explicit_sk)
        self._static_endpoint = _normalize_endpoint(endpoint)
        self._static_ak = explicit_ak
        self._static_sk = explicit_sk
        self._timeout = timeout
        self._cached: dict[str, object] = {}
        self._cache_lock = threading.Lock()
        if not self._use_normandy and not self._static_endpoint:
            raise RuntimeError("static OSS mode needs endpoint")

    def _resolve_creds(self, bucket: str) -> _BucketCreds:
        if self._use_normandy:
            return _creds_from_normandy(bucket, endpoint_override=self._static_endpoint)
        return _BucketCreds(
            access_key_id=self._static_ak,
            access_key_secret=self._static_sk,
            endpoint=self._static_endpoint,
        )

    def _oss_bucket(self, bucket_name: str) -> object:
        from benchmark.libs.storage import osscompat as oss2

        with self._cache_lock:
            cached = self._cached.get(bucket_name)
            if cached is not None:
                return cached
            creds = self._resolve_creds(bucket_name)
            if creds.security_token:
                auth = oss2.StsAuth(
                    creds.access_key_id,
                    creds.access_key_secret,
                    creds.security_token,
                )
            else:
                auth = oss2.Auth(creds.access_key_id, creds.access_key_secret)
            bucket = oss2.Bucket(auth, creds.endpoint, bucket_name, connect_timeout=self._timeout)
            self._cached[bucket_name] = bucket
            return bucket

    def _retry_oss(self, bucket_name: str, op: Callable[[object], object]) -> object:
        from benchmark.libs.storage import osscompat as oss2

        try:
            return op(self._oss_bucket(bucket_name))
        except oss2.exceptions.OssError as exc:
            if not _should_retry_oss(exc):
                raise
            with self._cache_lock:
                self._cached.pop(bucket_name, None)
            time.sleep(_RETRY_BACKOFF_S)
            return op(self._oss_bucket(bucket_name))

    def _get_bytes(self, bucket_name: str, key: str) -> bytes:
        from benchmark.libs.storage import osscompat as oss2

        logger.debug(
            "OSS download bucket=%s key=%s mode=%s",
            bucket_name,
            key,
            "normandy" if self._use_normandy else "static",
        )
        try:
            result = self._retry_oss(bucket_name, lambda bucket: bucket.get_object(key))
            data = result.read()
        except oss2.exceptions.OssError as exc:
            raise RuntimeError(f"OSS download failed oss://{bucket_name}/{key}: {exc}") from exc
        if not isinstance(data, bytes | bytearray):
            data = bytes(data)
        return bytes(data)

    async def get(self, bucket: str, key: str) -> bytes:
        name = bucket.strip()
        oss_key = key.strip()
        if not name:
            raise RuntimeError("OSS bucket is required")
        if not oss_key:
            raise RuntimeError("OSS key is required")
        return await asyncio.to_thread(self._get_bytes, name, oss_key)


def _should_retry_oss(exc: BaseException) -> bool:
    status = getattr(exc, "status", None)
    if not isinstance(status, int):
        return True
    if status >= _HTTP_SERVER_ERROR:
        return True
    return status in _RETRYABLE_CLIENT_STATUSES


__all__ = ["OssBlob"]
