"""Report OSS adapter over shared storage's optional signed-upload capability.

Shared storage owns credentials, SDK upload/signing, executor dispatch and timeout handling.
Reports own endpoint/config adaptation, execution budget and process-local same-key ordering.
Application logging initialization retains SDK diagnostic suppression.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Lock
from time import monotonic
from weakref import WeakValueDictionary

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.storage.credentials import CredentialProvider
from benchmark.libs.storage.providers.oss.client import (
    CREDENTIAL_PROVIDER_CREATORS,
    OssSigningOptions,
    OSSStorageClient,
    oss_execution_scope,
)
from benchmark.libs.storage.providers.oss.config import OSSStorageConfig

logger = get_logger(__name__)

#: Default signed-URL lifetime is 7d. Production credentials are a long-lived AK (oss_cred_acquired has_sts=False),
#: so the signature lifetime is not capped by an STS token lifetime and a report shared across days or weeks does not expire early.
#: The actual value is injected by ``Settings.cluster_report_oss_sign_expires_s``; this constant is only
#: the fallback when that setting is unset.
DEFAULT_SIGN_EXPIRES_S = 604800

#: Default credential mode: production uses Normandy platform resource binding and needs no explicit AK/SK.
DEFAULT_CREDENTIAL_MODE = "provider"

#: Per-connection timeout. Intranet connect is <100ms, so 5s is enough and fails fast (upload failure only degrades and does not block the report).
OSS_CONNECT_TIMEOUT_S = 5.0

#: Overall deadline for one operation: covers oss2 internal retries plus STS credential fetch.
#: An xlsx may reach megabytes, so this is looser than the same timeout on the online request path (this path is an async job).
OSS_OP_DEADLINE_S = 60.0

# Shared only by report OSS operations, with a fixed worker budget for this process.
_OSS_UPLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cluster-report-oss")
# Keep the same key exclusive until the actual SDK call finishes, including after timeout.
_UPLOAD_LOCKS_GUARD = Lock()
_UPLOAD_LOCKS: "WeakValueDictionary[tuple[str, str], Lock]" = WeakValueDictionary()


def _object_upload_lock(bucket: str, object_key: str) -> "Lock":
    with _UPLOAD_LOCKS_GUARD:
        key = (bucket, object_key)
        lock = _UPLOAD_LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _UPLOAD_LOCKS[key] = lock
        return lock


def _make_credential_provider(credential_mode: str) -> CredentialProvider:
    """Get the libs/storage credential provider by ``credential_mode``; an unknown mode falls back to provider."""

    if credential_mode not in CREDENTIAL_PROVIDER_CREATORS:
        logger.warning(
            "cluster_report.unknown_oss_credential_mode mode=%r fallback=%r",
            credential_mode,
            DEFAULT_CREDENTIAL_MODE,
        )
    creator = CREDENTIAL_PROVIDER_CREATORS.get(
        credential_mode, CREDENTIAL_PROVIDER_CREATORS[DEFAULT_CREDENTIAL_MODE]
    )
    return creator()


def _with_scheme(endpoint: str) -> str:
    """An oss2 endpoint may or may not include a scheme; always prefix https:// so a signed URL is not an http link."""

    if endpoint.startswith(("http://", "https://")):
        return endpoint
    return f"https://{endpoint}"


def _derive_public_endpoint(endpoint: str) -> str:
    """Derive the public domain from the endpoint by dropping the ``-internal`` segment of the host.

    Upload uses the intranet domain (faster inside the VPC and free of public traffic
    fees), but the signed URL is for people outside the VPC, and an ``-internal`` domain
    does not resolve on the public network. An OSS signature does not include the host, so
    the same signature is valid on the public domain (provided the bucket does not block
    public access). Only the domain is swapped and re-signed; the object is not re-uploaded.
    """

    scheme_split = endpoint.split("://", 1)
    has_scheme = len(scheme_split) > 1
    host = scheme_split[1] if has_scheme else scheme_split[0]
    if not host.endswith("-internal.aliyuncs.com"):
        return endpoint  # Already a public domain (or a nonstandard domain that cannot be derived) → use as-is
    # -internal sits in the middle of the domain, so removesuffix does not apply; drop the segment by replacement
    public_host = host.replace("-internal.aliyuncs.com", ".aliyuncs.com", 1)
    scheme = scheme_split[0] + "://" if has_scheme else ""
    return f"{scheme}{public_host}"


@dataclass(frozen=True)
class OssUploaderConfig:
    """Resource, credential, signing and timeout inputs for a report uploader."""

    bucket: str
    endpoint: str
    public_endpoint: str = ""
    credential_mode: str = DEFAULT_CREDENTIAL_MODE
    sign_expires_s: int = DEFAULT_SIGN_EXPIRES_S
    connect_timeout_s: float = OSS_CONNECT_TIMEOUT_S
    op_deadline_s: float = OSS_OP_DEADLINE_S


class OssArtifactUploader(OSSStorageClient):
    """Artifact uploader — satisfies the :class:`ObjectUploader` protocol (``upload`` returns an accessible URL).

    A successful upload returns a 7d signed URL. Any failure returns ``None`` (the same
    degradation convention as libs/storage), and the sink layer falls back to the local path.
    """

    def __init__(
        self,
        config: OssUploaderConfig,
    ) -> None:
        bucket, endpoint, public_endpoint = config.bucket, config.endpoint, config.public_endpoint
        credential_mode, sign_expires_s = config.credential_mode, config.sign_expires_s
        connect_timeout_s, op_deadline_s = config.connect_timeout_s, config.op_deadline_s
        self._bucket_name = bucket
        self._endpoint = _with_scheme(endpoint)
        # Domain used only for signed URLs: when not set explicitly, derive it from the upload endpoint (drop -internal).
        # When the upload domain is already public, the derived result is itself and behavior is unchanged.
        self._sign_endpoint = _with_scheme(
            public_endpoint.strip() or _derive_public_endpoint(self._endpoint)
        )
        self._credential_mode = credential_mode
        self._credentials = _make_credential_provider(credential_mode)
        self._sign_expires_s = sign_expires_s
        self._connect_timeout_s = connect_timeout_s
        self._op_deadline_s = op_deadline_s
        super().__init__(
            config=OSSStorageConfig(
                bucket=bucket,
                upload_endpoint=endpoint.split("://", 1)[-1],
                cdn_base_url="https://" + self._sign_endpoint.split("://", 1)[-1].rstrip("/"),
                upload_timeout_ms=max(1, round(op_deadline_s * 1000)),
                connect_timeout_s=connect_timeout_s,
                credential_mode=credential_mode
                if credential_mode in ("provider", "env")
                else DEFAULT_CREDENTIAL_MODE,
            ),
            credential_provider=self._credentials,
        )

    @contextmanager
    def _signed_operation_scope(self, object_key: str, deadline: float) -> Iterator[None]:
        lock = _object_upload_lock(self._bucket_name, object_key)
        remaining = deadline - monotonic()
        if remaining <= 0 or not lock.acquire(timeout=remaining):
            raise TimeoutError
        try:
            yield
        finally:
            lock.release()

    async def upload(self, object_key: str, data: bytes) -> str | None:
        with oss_execution_scope(_OSS_UPLOAD_EXECUTOR):
            return await self.upload_signed(
                object_key,
                data,
                OssSigningOptions(
                    sign_endpoint=self._sign_endpoint,
                    expires_s=self._sign_expires_s,
                    upload_endpoint=self._endpoint,
                ),
            )


def build_oss_uploader(
    config: OssUploaderConfig,
) -> OssArtifactUploader | None:
    """Build from config; missing bucket or endpoint selects local-only reporting."""
    bucket, endpoint, public_endpoint = config.bucket, config.endpoint, config.public_endpoint
    credential_mode, sign_expires_s = config.credential_mode, config.sign_expires_s

    if not bucket.strip() or not endpoint.strip():
        logger.info("cluster_report.oss_not_configured falling back to local-only sink")
        return None
    uploader = OssArtifactUploader(
        replace(
            config,
            bucket=bucket.strip(),
            endpoint=endpoint.strip(),
            public_endpoint=public_endpoint.strip(),
        )
    )
    logger.info(
        "cluster_report.oss_uploader_ready bucket=%s endpoint=%s sign_endpoint=%s "
        "credential_mode=%s sign_expires_s=%d",
        bucket.strip(),
        _with_scheme(endpoint.strip()),
        uploader._sign_endpoint,
        credential_mode,
        sign_expires_s,
    )
    return uploader


__all__ = [
    "DEFAULT_CREDENTIAL_MODE",
    "DEFAULT_SIGN_EXPIRES_S",
    "OssArtifactUploader",
    "OssUploaderConfig",
    "_derive_public_endpoint",
    "build_oss_uploader",
]
