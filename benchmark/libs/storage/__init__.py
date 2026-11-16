"""Local blob store with the storage client names the service already calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


class StorageError(Exception):
    pass


class StorageCredentialError(StorageError):
    pass


class StorageUploadError(StorageError):
    pass


@dataclass
class StorageCredential:
    access_key_id: str = ""
    access_key_secret: str = ""
    security_token: str = ""


class CredentialProvider:
    def get(self) -> StorageCredential:
        return StorageCredential(
            access_key_id=os.environ.get("BENCHMARK_STORAGE_ACCESS_KEY", ""),
            access_key_secret=os.environ.get("BENCHMARK_STORAGE_SECRET", ""),
            security_token=os.environ.get("BENCHMARK_STORAGE_TOKEN", ""),
        )

    def acquire(self, *, bucket: str = "") -> StorageCredential:
        del bucket
        return self.get()


class EnvCredentialProvider(CredentialProvider):
    pass


class NormandyCredentialProvider(CredentialProvider):
    """Kept so existing call sites import. Credentials come from the environment."""

    def __init__(self, *_args, **_kwargs):
        pass


class OSSStorageConfig:
    def __init__(
        self,
        endpoint: str = "",
        bucket: str = "local",
        prefix: str = "",
        root: str = "",
        **extra,
    ):
        self.endpoint = endpoint or str(extra.get("upload_endpoint") or "")
        self.bucket = bucket
        self.prefix = prefix or str(extra.get("object_key_prefix") or "")
        self.root = root
        self.upload_endpoint = str(extra.get("upload_endpoint") or self.endpoint)
        self.cdn_base_url = str(extra.get("cdn_base_url") or "")
        self.upload_timeout_ms = int(extra.get("upload_timeout_ms") or 60000)
        self.connect_timeout_s = float(extra.get("connect_timeout_s") or 5)
        self.credential_mode = str(extra.get("credential_mode") or "provider")
        self.object_key_prefix = str(extra.get("object_key_prefix") or self.prefix)
        for key, value in extra.items():
            setattr(self, key, value)


def parse_storage_config(raw: dict | None) -> OSSStorageConfig:
    raw = raw or {}
    return OSSStorageConfig(
        endpoint=str(raw.get("endpoint") or ""),
        bucket=str(raw.get("bucket") or "local"),
        prefix=str(raw.get("prefix") or ""),
        root=str(raw.get("root") or os.environ.get("BENCHMARK_OBJECT_STORE", "var/objects")),
    )


def build_object_key(*parts) -> str:
    if parts and not isinstance(parts[0], str):
        config = parts[0]
        key = str(parts[1] if len(parts) > 1 else "")
        prefix = str(getattr(config, "object_key_prefix", "") or getattr(config, "prefix", "") or "")
        return "/".join(part.strip("/") for part in (prefix, key) if part)
    return "/".join(str(part).strip("/") for part in parts if part)


def build_oss_url(*args) -> str:
    if args and not isinstance(args[0], str):
        config = args[0]
        key = str(args[1] if len(args) > 1 else "").lstrip("/")
        cdn = str(getattr(config, "cdn_base_url", "") or "")
        if cdn:
            return f"{cdn.rstrip('/')}/{key}"
        bucket = str(getattr(config, "bucket", "") or "local")
        return f"file://{bucket}/{key}"
    bucket = str(args[0] if args else "local")
    key = str(args[1] if len(args) > 1 else "")
    endpoint = str(args[2] if len(args) > 2 else "")
    if endpoint.startswith("file:"):
        return f"{endpoint.rstrip('/')}/{key}"
    return f"file://{bucket}/{key}"


class OSSStorageClient:
    def __init__(self, config: OSSStorageConfig, credential_provider: CredentialProvider | None = None):
        self.config = config
        self.credential_provider = credential_provider or EnvCredentialProvider()
        self._config = config
        self._credential_provider = self.credential_provider
        self.backend_name = "oss"
        root = Path(getattr(config, "root", None) or "var/objects") / config.bucket
        prefix = getattr(config, "prefix", "") or ""
        if prefix:
            root = root / prefix
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    async def upload(self, object_key: str, data: bytes, **_kwargs) -> str | None:
        key = build_object_key(self.config, object_key)
        try:
            import oss2
        except ImportError:
            oss2 = None
        if oss2 is not None and getattr(self.config, "upload_endpoint", ""):
            credential = self._credential_provider.acquire(bucket=self.config.bucket)
            if getattr(credential, "security_token", None):
                auth = oss2.StsAuth(
                    credential.access_key_id,
                    credential.access_key_secret,
                    credential.security_token,
                )
            else:
                auth = oss2.Auth(credential.access_key_id, credential.access_key_secret)
            endpoint = self.config.upload_endpoint
            if endpoint and not str(endpoint).startswith("http"):
                endpoint = "https://" + endpoint
            bucket = oss2.Bucket(auth, endpoint, self.config.bucket)
            bucket.put_object(key, data)
        else:
            self.put_object(key, data)
        return build_oss_url(self.config, object_key)

    def put_object(self, key: str, body: bytes | str) -> str:
        data = body.encode("utf-8") if isinstance(body, str) else body
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return build_oss_url(self.config.bucket, key)

    def get_object(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise StorageError(f"object not found: {key}")
        return path.read_bytes()

    def sign_url(self, method: str, key: str, expires: int = 3600) -> str:
        del method, expires
        return self._path(key).resolve().as_uri()

    def _path(self, key: str) -> Path:
        parsed = urlparse(key)
        relative = parsed.path if parsed.scheme else key
        path = (self.root / relative.lstrip("/")).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise StorageError("object key escapes the store")
        return path


class StorageClient(OSSStorageClient):
    pass
