"""storage_ext — agent-eval 专属 OSS 存储扩展（sign_url + content_type）。

main 版 ``libs/storage`` 的 OSS 客户端尚未提供报告链路所需的
``sign_url``（私有 bucket 预签名 URL）与 ``upload(content_type=...)``
能力；本包以子类形式承载扩展，共享库保持零改动。
"""

from benchmark.libs.storage_ext.signed_client import (
    DEFAULT_PRESIGNED_EXPIRES_SECONDS,
    ReportStorageClient,
    SignedStorageClient,
    build_signed_client,
    public_endpoint,
)

__all__ = [
    "DEFAULT_PRESIGNED_EXPIRES_SECONDS",
    "ReportStorageClient",
    "SignedStorageClient",
    "build_signed_client",
    "public_endpoint",
]
