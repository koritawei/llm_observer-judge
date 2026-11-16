from contextlib import contextmanager
from dataclasses import dataclass, field

from benchmark.libs.storage import OSSStorageClient, StorageClient
from benchmark.libs.storage.credentials import CredentialProvider
from benchmark.libs.storage.providers.oss.credential import (
    EnvCredentialProvider,
    NormandyCredentialProvider,
)

try:
    import oss2
except ImportError:  # pragma: no cover - tests inject a fake module
    oss2 = None


@dataclass
class OssSigningOptions:
    expires_s: int = 3600
    headers: dict | None = None
    sign_endpoint: str = ""
    upload_endpoint: str = ""
    extra: dict = field(default_factory=dict)

    def __init__(self, expires_s: int = 3600, headers: dict | None = None, **extra):
        self.expires_s = int(extra.pop("expires_s", expires_s) or expires_s)
        self.headers = headers
        self.sign_endpoint = str(extra.pop("sign_endpoint", "") or "")
        self.upload_endpoint = str(extra.pop("upload_endpoint", "") or "")
        self.extra = extra


CREDENTIAL_PROVIDER_CREATORS: dict[str, type[CredentialProvider]] = {
    "provider": NormandyCredentialProvider,
    "env": EnvCredentialProvider,
}


@contextmanager
def oss_execution_scope(*_args, **_kwargs):
    yield


__all__ = [
    "CREDENTIAL_PROVIDER_CREATORS",
    "OSSStorageClient",
    "OssSigningOptions",
    "StorageClient",
    "oss_execution_scope",
]
