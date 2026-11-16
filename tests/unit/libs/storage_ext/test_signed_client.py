"""SignedStorageClient unit tests — sign_url / content_type / factory.

A fake ``oss2`` is injected via ``sys.modules`` (monkeypatch restores it), and credentials are
injected with ``_FakeProvider``. There is no real network / OSS / Normandy dependency.
"""

import logging
import sys
import types
from unittest.mock import MagicMock

import pytest
from benchmark.libs.storage_ext.signed_client import (
    DEFAULT_PRESIGNED_EXPIRES_SECONDS,
    SignedStorageClient,
    build_signed_client,
    public_endpoint,
)
from benchmark.libs.storage.credentials import StorageCredential
from benchmark.libs.storage.exceptions import StorageCredentialError
from benchmark.libs.storage.providers.oss.config import OSSStorageConfig

LOGGER_NAME = "benchmark.libs.storage_ext.signed_client"


# ── Fixtures ────────────────────────────────────────────────────────


def _oss_cfg(**override: object) -> OSSStorageConfig:
    base: dict[str, object] = dict(
        bucket="test-bucket",
        upload_endpoint="oss-cn-test-internal.aliyuncs.com",
        cdn_base_url="https://cdn.example.com",
        upload_timeout_ms=2000,
        credential_mode="env",
    )
    base.update(override)
    return OSSStorageConfig(**base)


class _FakeProvider:
    """Sync acquire returning fixed credentials; supports injecting errors."""

    def __init__(
        self,
        *,
        ak: str = "AK",
        sk: str = "SK",
        security_token: str | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._ak = ak
        self._sk = sk
        self._sts = security_token
        self._raise = raise_exc
        self.calls: list[str] = []

    def acquire(self, *, bucket: str) -> StorageCredential:
        self.calls.append(bucket)
        if self._raise:
            raise self._raise
        return StorageCredential(
            access_key_id=self._ak,
            access_key_secret=self._sk,
            security_token=self._sts,
        )


def _install_fake_oss2(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock]:
    """Install a fake ``oss2`` module into ``sys.modules``.

    ``put_object`` accepts a ``headers`` keyword argument (same signature as real oss2),
    and call-argument assertions are done from the MagicMock call record.
    """
    fake_oss2 = types.ModuleType("oss2")
    fake_exceptions = types.ModuleType("oss2.exceptions")

    class FakeOssError(Exception):
        pass

    class FakeNoSuchKeyError(FakeOssError):
        pass

    fake_exceptions.OssError = FakeOssError
    fake_exceptions.NoSuchKey = FakeNoSuchKeyError

    fake_bucket = MagicMock()

    def _put_object(
        object_key: str,
        data: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        pass

    fake_bucket.put_object = MagicMock(side_effect=_put_object)

    _default_obj = MagicMock()
    _default_obj.read = MagicMock(return_value=b"")
    fake_bucket.get_object = MagicMock(return_value=_default_obj)
    fake_bucket.sign_url = MagicMock(
        return_value="https://oss-cn-test.aliyuncs.com/test-bucket/signed?Expires=1&Signature=x",
    )

    fake_oss2.Bucket = MagicMock(return_value=fake_bucket)
    fake_oss2.Auth = MagicMock(return_value="auth_obj")
    fake_oss2.StsAuth = MagicMock(return_value="sts_auth_obj")
    fake_oss2.exceptions = fake_exceptions

    # Shared storage imports the optional SDK at module load; patch that binding too.
    from benchmark.libs.storage.providers.oss import client as shared_oss

    monkeypatch.setattr(shared_oss, "oss2", fake_oss2)
    monkeypatch.setitem(sys.modules, "oss2", fake_oss2)
    monkeypatch.setitem(sys.modules, "oss2.exceptions", fake_exceptions)
    return fake_oss2, fake_bucket


# ── public_endpoint ──────────────────────────────────────────────────


def test_public_endpoint_strips_internal() -> None:
    assert (
        public_endpoint("oss-cn-zhangjiakou-internal.aliyuncs.com")
        == "oss-cn-zhangjiakou.aliyuncs.com"
    )
    assert public_endpoint("oss-cn-hangzhou.aliyuncs.com") == "oss-cn-hangzhou.aliyuncs.com"


# ── sign_url ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sign_url_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_oss2, fake_bucket = _install_fake_oss2(monkeypatch)
    provider = _FakeProvider(ak="AK1", sk="SK1")
    client = SignedStorageClient(config=_oss_cfg(), credential_provider=provider)

    url = await client.sign_url("reports/r1/report.html")

    assert url is not None
    assert "signed" in url
    assert provider.calls == ["test-bucket"]
    fake_oss2.Auth.assert_called_once_with("AK1", "SK1")
    fake_oss2.Bucket.assert_called_once_with(
        "auth_obj",
        "https://oss-cn-test.aliyuncs.com",
        "test-bucket",
    )
    fake_bucket.sign_url.assert_called_once_with(
        "GET",
        "reports/r1/report.html",
        DEFAULT_PRESIGNED_EXPIRES_SECONDS,
        slash_safe=True,
    )


@pytest.mark.asyncio
async def test_sign_url_applies_prefix_and_custom_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fake_bucket = _install_fake_oss2(monkeypatch)
    client = SignedStorageClient(
        config=_oss_cfg(object_key_prefix="agent-eval/benchmark-reports"),
        credential_provider=_FakeProvider(),
    )

    await client.sign_url("run1/report.html", expires_seconds=3600)

    fake_bucket.sign_url.assert_called_once_with(
        "GET",
        "agent-eval/benchmark-reports/run1/report.html",
        3600,
        slash_safe=True,
    )


@pytest.mark.asyncio
async def test_sign_url_credential_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_oss2(monkeypatch)
    client = SignedStorageClient(
        config=_oss_cfg(),
        credential_provider=_FakeProvider(
            raise_exc=StorageCredentialError("no creds"),
        ),
    )

    assert await client.sign_url("x.html") is None


@pytest.mark.asyncio
async def test_sign_url_invalid_expires_returns_none() -> None:
    client = SignedStorageClient(
        config=_oss_cfg(),
        credential_provider=_FakeProvider(),
    )
    assert await client.sign_url("x.html", expires_seconds=0) is None


@pytest.mark.asyncio
async def test_sign_url_sts_uses_sts_auth(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_oss2, _ = _install_fake_oss2(monkeypatch)
    client = SignedStorageClient(
        config=_oss_cfg(),
        credential_provider=_FakeProvider(security_token="TOKEN"),
    )

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        url = await client.sign_url("x.html", expires_seconds=100)

    assert url is not None
    fake_oss2.StsAuth.assert_called_once_with("AK", "SK", "TOKEN")
    assert any("STS" in r.message for r in caplog.records)


# ── upload(content_type=...) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_with_content_type_sets_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fake_bucket = _install_fake_oss2(monkeypatch)
    client = SignedStorageClient(
        config=_oss_cfg(),
        credential_provider=_FakeProvider(),
    )

    url = await client.upload(
        "run1/report.html",
        b"<html></html>",
        content_type="text/html; charset=utf-8",
    )

    assert url == "https://cdn.example.com/run1/report.html"
    fake_bucket.put_object.assert_called_once_with(
        "run1/report.html",
        b"<html></html>",
        headers={"Content-Type": "text/html; charset=utf-8"},
    )


@pytest.mark.asyncio
async def test_upload_without_content_type_uses_parent_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """content_type=None delegates to the parent upload; when polymorphism enters the subclass _do_put_object, no header is passed."""
    _, fake_bucket = _install_fake_oss2(monkeypatch)
    client = SignedStorageClient(
        config=_oss_cfg(),
        credential_provider=_FakeProvider(),
    )

    url = await client.upload("run1/report.html", b"<html></html>")

    assert url == "https://cdn.example.com/run1/report.html"
    # content_type=None takes the parent implementation (headers is not passed explicitly; oss2 defaults it to None)
    fake_bucket.put_object.assert_called_once_with(
        "run1/report.html",
        b"<html></html>",
    )


# ── build_signed_client factory ────────────────────────────────────


def test_build_signed_client_env_mode() -> None:
    client = build_signed_client(_oss_cfg(credential_mode="env"))
    assert isinstance(client, SignedStorageClient)
    assert client.backend_name == "oss"


def test_build_signed_client_rejects_non_oss_config() -> None:
    from benchmark.libs.storage.exceptions import StorageError

    with pytest.raises(StorageError, match="requires OSSStorageConfig"):
        build_signed_client({"backend": "oss"})
