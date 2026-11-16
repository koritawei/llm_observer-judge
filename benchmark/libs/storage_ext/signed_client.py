"""SignedStorageClient — sign_url 与 content_type 扩展（基于 libs/storage OSS）。

main 版 ``libs/storage`` 的 ``OSSStorageClient``
尚未提供 agent-eval 报告链路需要的两个能力：

1. ``sign_url`` — 私有 bucket 的 30 天 GET 预签名 URL（剥离 ``-internal``
   走公网 endpoint，浏览器可直接访问）；
2. ``upload(content_type=...)`` — 上传时显式设置 ``Content-Type``
   （HTML 报告防止浏览器按默认编码渲染乱码）。

本模块以子类承载这两个扩展，共享库零改动。退出路径：main 版
``libs/storage`` 合入等价能力后，删除本包，调用点换回
``storage.create_client``。

Design（对齐 libs/storage 的 OSS 客户端约定）：

1. ``oss2`` 是同步 SDK，经 ``asyncio.to_thread`` + ``asyncio.wait_for`` 包装
2. 失败返回 ``None``（优雅降级，不重试）
3. 每次 OSS 操作新建 ``oss2.Bucket`` 实例（避免共享状态线程问题）
4. 凭证防泄漏：WARNING 日志仅用 ``type(exc).__name__``
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from benchmark.libs.storage.credentials import CredentialProvider, StorageCredential
from benchmark.libs.storage.exceptions import StorageCredentialError, StorageError, StorageUploadError
from benchmark.libs.storage.providers.oss.client import OSSStorageClient
from benchmark.libs.storage.providers.oss.config import OSSStorageConfig
from benchmark.libs.storage.providers.oss.credential import (
    EnvCredentialProvider,
    NormandyCredentialProvider,
)
from benchmark.libs.storage.utils import build_object_key, build_oss_url

logger = logging.getLogger(__name__)


def public_endpoint(upload_endpoint: str) -> str:
    """Map VPC/internal upload host to the public OSS endpoint.

    Example: ``oss-cn-zhangjiakou-internal.aliyuncs.com``
    → ``oss-cn-zhangjiakou.aliyuncs.com``。

    内联于本模块：主干 ``libs/storage.utils`` 未提供该函数，为保持
    agent-eval 对主干 libs 的兼容，不再跨包引用。
    """
    return upload_endpoint.replace("-internal.", ".")


#: 默认 GET 预签名 URL 有效期 — 30 天（报告下载链接）。
DEFAULT_PRESIGNED_EXPIRES_SECONDS = 30 * 24 * 60 * 60  # 2_592_000

CredentialProviderCreator = Callable[[], CredentialProvider]


@dataclass(frozen=True)
class _PutObjectPayload:
    """``_put_object_with_content_type`` 参数包（FUNC-PARAMS-001 参数收敛）。"""

    credential: StorageCredential
    object_key: str
    data: bytes
    content_type: str | None = None


@dataclass(frozen=True)
class _SignUrlPayload:
    """``_sync_sign_url`` 参数包（FUNC-PARAMS-001 参数收敛）。"""

    credential: StorageCredential
    endpoint: str
    object_key: str
    expires_seconds: int


CREDENTIAL_PROVIDER_CREATORS: dict[str, CredentialProviderCreator] = {
    "provider": NormandyCredentialProvider,
    "env": EnvCredentialProvider,
}


class ReportStorageClient(Protocol):
    """报告上传链路消费的最小存储接口。

    结构化协议：``SignedStorageClient`` 天然满足，测试可用任意
    提供同签名方法的对象注入，不要求继承。
    """

    async def upload(
        self,
        object_key: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> str | None:
        """Upload *data* and return an accessible URL, or ``None`` on failure."""
        ...

    async def sign_url(
        self,
        object_key: str,
        *,
        expires_seconds: int = DEFAULT_PRESIGNED_EXPIRES_SECONDS,
    ) -> str | None:
        """Return a time-limited GET URL for *object_key*, or ``None`` on failure."""
        ...


def build_signed_client(config: object) -> "SignedStorageClient":
    """Create a sign_url-capable OSS storage client from OSS-specific config.

    与 ``storage.create_oss_client`` 同构（``config: object`` + isinstance
    校验），仅支持 ``provider`` / ``env`` 两种凭证模式（main 版
    ``OSSStorageConfig`` 的 ``credential_mode`` Literal 范围一致）。
    """
    if not isinstance(config, OSSStorageConfig):
        raise StorageError("signed OSS client requires OSSStorageConfig")

    provider_creator = CREDENTIAL_PROVIDER_CREATORS.get(config.credential_mode)
    if provider_creator is None:
        raise StorageError(f"Unknown OSS credential mode: {config.credential_mode!r}")
    return SignedStorageClient(
        config=config,
        credential_provider=provider_creator(),
    )


class SignedStorageClient(OSSStorageClient):
    """OSSStorageClient + ``sign_url`` + ``upload(content_type=...)``.

    ``content_type=None`` 的 ``upload`` 直接委托父类（main 版行为，
    未来父类演进自动继承）；仅显式携带 ``content_type`` 时才走本类
    的完整上传流程。
    """

    async def upload(
        self,
        object_key: str,
        data: bytes,
        *,
        content_type: str | None = None,
    ) -> str | None:
        """Upload data to OSS and return the CDN URL, or ``None`` on failure.

        If *content_type* is set, it overrides OSS auto-detection so the
        object is served with the correct ``Content-Type`` header (e.g.
        ``text/html; charset=utf-8`` to prevent browser encoding 乱码).
        """
        if content_type is None:
            return await super().upload(object_key, data)

        config = self._config
        storage_object_key = build_object_key(config, object_key)
        logger.info(
            "oss upload start: bucket=%s key=%s bytes=%d endpoint=%s "
            "credential_mode=%s cdn_base_url=%s content_type=%s",
            config.bucket,
            storage_object_key,
            len(data) if data else 0,
            config.upload_endpoint,
            config.credential_mode,
            config.cdn_base_url,
            content_type,
        )
        try:
            credential = self._credential_provider.acquire(bucket=config.bucket)
        except StorageCredentialError as exc:
            logger.warning(
                "oss credential acquire failed: bucket=%s key=%s err=%s",
                config.bucket,
                storage_object_key,
                exc,
            )
            return None

        try:
            await asyncio.wait_for(
                self._put_object_with_content_type(
                    _PutObjectPayload(
                        credential=credential,
                        object_key=storage_object_key,
                        data=data,
                        content_type=content_type,
                    ),
                ),
                timeout=config.upload_timeout_ms / 1000,
            )
        except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041  # py310 compat
            logger.warning(
                "oss upload timeout: bucket=%s key=%s timeout_ms=%d",
                config.bucket,
                storage_object_key,
                config.upload_timeout_ms,
            )
            return None
        except StorageUploadError as exc:
            logger.warning(
                "oss upload failed: bucket=%s key=%s err=%s",
                config.bucket,
                storage_object_key,
                exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "oss upload unexpected error: bucket=%s key=%s err_type=%s",
                config.bucket,
                storage_object_key,
                type(exc).__name__,
            )
            logger.debug(
                "oss upload unexpected error detail: bucket=%s key=%s",
                config.bucket,
                storage_object_key,
                exc_info=True,
            )
            return None

        url = build_oss_url(config, object_key)
        logger.info(
            "oss upload ok: bucket=%s key=%s url=%s",
            config.bucket,
            storage_object_key,
            url,
        )
        return url

    async def _put_object_with_content_type(
        self,
        payload: _PutObjectPayload,
    ) -> None:
        """Wrap sync ``oss2.put_object`` as async; *content_type* sets ``Content-Type``.

        父类 ``upload`` 多态进入父类 ``_do_put_object``（不带
        ``content_type``），行为与 main 版一致（不发送 ``Content-Type`` 头）。
        """
        credential = payload.credential
        object_key = payload.object_key
        data = payload.data
        content_type = payload.content_type
        config = self._config

        def _sync_put() -> None:
            try:
                import oss2
            except ImportError as exc:  # pragma: no cover
                raise StorageUploadError("oss2 not installed; check requirements_lock.txt") from exc
            try:
                if credential.security_token:
                    auth: object = oss2.StsAuth(
                        credential.access_key_id,
                        credential.access_key_secret,
                        credential.security_token,
                    )
                else:
                    auth = oss2.Auth(credential.access_key_id, credential.access_key_secret)
                bucket = oss2.Bucket(auth, config.upload_endpoint, config.bucket)
                headers: dict[str, str] = {}
                if content_type:
                    headers["Content-Type"] = content_type
                bucket.put_object(object_key, data, headers=headers or None)
            except StorageUploadError:
                raise
            except Exception as exc:
                raise StorageUploadError(
                    f"oss2 put_object error: bucket={config.bucket!r} key={object_key!r}"
                ) from exc

        await asyncio.to_thread(_sync_put)

    async def sign_url(
        self,
        object_key: str,
        *,
        expires_seconds: int = DEFAULT_PRESIGNED_EXPIRES_SECONDS,
    ) -> str | None:
        """Generate a GET presigned URL via ``oss2.Bucket.sign_url``.

        Uses the **public** endpoint (internal ``-internal`` suffix stripped)
        so the URL is reachable outside the VPC. Returns ``None`` on
        credential or signing failure.

        凭证来自构造时注入的 ``credential_provider``（生产为 Normandy
        资源绑定的长期 AK，30 天 TTL 直接生效）。

        Note: STS credentials may cap effective TTL below *expires_seconds*.
        """
        if expires_seconds < 1:
            logger.warning(
                "oss sign_url rejected: expires_seconds must be >= 1, got %s",
                expires_seconds,
            )
            return None

        config = self._config
        storage_object_key = build_object_key(config, object_key)

        try:
            credential = self._credential_provider.acquire(bucket=config.bucket)
        except StorageCredentialError as exc:
            logger.warning(
                "oss sign_url credential acquire failed: bucket=%s key=%s err=%s",
                config.bucket,
                storage_object_key,
                exc,
            )
            return None

        if credential.security_token:
            logger.warning(
                "oss sign_url with STS: requested_expires=%ss may exceed token TTL "
                "bucket=%s key=%s",
                expires_seconds,
                config.bucket,
                storage_object_key,
            )

        endpoint = public_endpoint(config.upload_endpoint)
        try:
            url = await asyncio.to_thread(
                self._sync_sign_url,
                _SignUrlPayload(
                    credential=credential,
                    endpoint=endpoint,
                    object_key=storage_object_key,
                    expires_seconds=expires_seconds,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "oss sign_url failed: bucket=%s key=%s err_type=%s",
                config.bucket,
                storage_object_key,
                type(exc).__name__,
            )
            logger.debug(
                "oss sign_url failed detail: bucket=%s key=%s",
                config.bucket,
                storage_object_key,
                exc_info=True,
            )
            return None

        logger.info(
            "oss sign_url ok: bucket=%s key=%s expires=%ss endpoint=%s",
            config.bucket,
            storage_object_key,
            expires_seconds,
            endpoint,
        )
        return url

    def _sync_sign_url(
        self,
        payload: _SignUrlPayload,
    ) -> str:
        """Blocking ``sign_url``; wrapped by :meth:`sign_url` via ``to_thread``."""
        credential = payload.credential
        endpoint = payload.endpoint
        object_key = payload.object_key
        expires_seconds = payload.expires_seconds
        config = self._config
        try:
            import oss2
        except ImportError as exc:  # pragma: no cover
            raise StorageError("oss2 not installed; check requirements_lock.txt") from exc
        if credential.security_token:
            auth: object = oss2.StsAuth(
                credential.access_key_id,
                credential.access_key_secret,
                credential.security_token,
            )
        else:
            auth = oss2.Auth(credential.access_key_id, credential.access_key_secret)
        # Prefer https so signed links are browser-safe over the public internet.
        bucket = oss2.Bucket(auth, f"https://{endpoint}", config.bucket)
        return bucket.sign_url("GET", object_key, expires_seconds, slash_safe=True)


__all__ = [
    "DEFAULT_PRESIGNED_EXPIRES_SECONDS",
    "ReportStorageClient",
    "SignedStorageClient",
    "build_signed_client",
    "public_endpoint",
]
