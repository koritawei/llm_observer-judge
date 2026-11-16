"""OSS uploader unit tests (credentials / signing / degradation), with no network.

A real upload needs a Normandy binding or AK/SK. Here we only check: the assembly switch, endpoint
normalization, credential-provider selection, and failure/timeout degrading to None (the sink then falls back to a local path).
"""

import asyncio
import threading
from contextvars import ContextVar
from unittest.mock import Mock

import pytest
from benchmark.reporting.cluster_service.oss_uploader import (
    DEFAULT_CREDENTIAL_MODE,
    DEFAULT_SIGN_EXPIRES_S,
    OssArtifactUploader,
    OssUploaderConfig,
    _derive_public_endpoint,
    _with_scheme,
    build_oss_uploader,
)

_BUCKET = "vehicle-agent-eval-data"
_ENDPOINT = "oss-cn-zhangjiakou-internal.aliyuncs.com"
_PUBLIC_ENDPOINT = "oss-cn-zhangjiakou.aliyuncs.com"


class TestBuildSwitch:
    def test_returns_none_when_unconfigured(self) -> None:
        """Missing bucket/endpoint → None, and the sink degrades to local-only (must not raise)."""
        assert build_oss_uploader(OssUploaderConfig(bucket="", endpoint=_ENDPOINT)) is None
        assert build_oss_uploader(OssUploaderConfig(bucket=_BUCKET, endpoint="")) is None
        assert build_oss_uploader(OssUploaderConfig(bucket="   ", endpoint="  ")) is None

    def test_builds_when_configured(self) -> None:
        uploader = build_oss_uploader(
            OssUploaderConfig(bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env")
        )
        assert isinstance(uploader, OssArtifactUploader)

    def test_default_sign_expiry_is_7d(self) -> None:
        """The default signature lifetime is 7d: production credentials are a long-lived AK (has_sts=False) and are not capped by STS,
        so a report shared across days or weeks does not expire early."""
        assert DEFAULT_SIGN_EXPIRES_S == 7 * 24 * 3600

    def test_all_params_come_from_config(self) -> None:
        """Every resource parameter must be configurable: bucket/endpoint/lifetime must not be hardcoded.

        Regression anchor: an early version wrote the bucket name and the resource unikey (including an account ID)
        into a docstring, and the signature lifetime was not exposed as a config item.
        """
        from benchmark.settings.setting import Settings

        for field in (
            "cluster_report_oss_bucket",
            "cluster_report_oss_endpoint",
            "cluster_report_oss_public_endpoint",
            "cluster_report_oss_credential_mode",
            "cluster_report_oss_key_prefix",
            "cluster_report_oss_sign_expires_s",
        ):
            assert field in Settings.model_fields, field
        # bucket/endpoint must default to empty (no real resource is built in)
        assert Settings.model_fields["cluster_report_oss_bucket"].default == ""
        assert Settings.model_fields["cluster_report_oss_endpoint"].default == ""
        assert Settings.model_fields["cluster_report_oss_public_endpoint"].default == ""

    def test_no_resource_identifier_hardcoded(self) -> None:
        """The code (including comments) must not contain a concrete bucket name, account ID, or unikey."""
        from pathlib import Path

        import benchmark.reporting.cluster_service.oss_uploader as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        for forbidden in ("vehicle-agent-eval-data", "1155002547732120", "ACS::OSS::Bucket"):
            assert forbidden not in source, f"资源标识不得入代码: {forbidden}"

    def test_sign_expiry_overridable(self) -> None:
        """The signature lifetime can be overridden by a parameter (which ultimately comes from a config item)."""
        uploader = build_oss_uploader(
            OssUploaderConfig(
                bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env", sign_expires_s=600
            )
        )
        assert uploader is not None
        assert uploader._sign_expires_s == 600

    def test_default_credential_mode_is_provider(self) -> None:
        """Production uses a Normandy resource binding and does not depend on an explicit AK/SK."""
        assert DEFAULT_CREDENTIAL_MODE == "provider"

    def test_unknown_credential_mode_falls_back(self) -> None:
        """An unknown mode falls back to the provider instead of crashing (a bad config value must not keep the service from starting)."""
        uploader = build_oss_uploader(
            OssUploaderConfig(bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="nonsense")
        )
        assert isinstance(uploader, OssArtifactUploader)


class TestEndpointNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (_ENDPOINT, f"https://{_ENDPOINT}"),
            (f"https://{_ENDPOINT}", f"https://{_ENDPOINT}"),
            # Do not rewrite a value that already has http: an internal test environment may have set it that way on purpose
            (f"http://{_ENDPOINT}", f"http://{_ENDPOINT}"),
        ],
    )
    def test_scheme_prefixed(self, raw: str, expected: str) -> None:
        assert _with_scheme(raw) == expected


class TestPublicEndpointDerivation:
    """Signed-URL hostname: when not configured explicitly, derive it by stripping -internal from the upload endpoint."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (_ENDPOINT, _PUBLIC_ENDPOINT),
            (f"https://{_ENDPOINT}", f"https://{_PUBLIC_ENDPOINT}"),
            # Already a public hostname → keep it (upload and sign share the hostname, behavior unchanged)
            (_PUBLIC_ENDPOINT, _PUBLIC_ENDPOINT),
            (f"https://{_PUBLIC_ENDPOINT}", f"https://{_PUBLIC_ENDPOINT}"),
            # A non-standard hostname cannot be derived → return it as-is, and do not guess
            ("oss.example.com", "oss.example.com"),
        ],
    )
    def test_derive(self, raw: str, expected: str) -> None:
        assert _derive_public_endpoint(raw) == expected

    def test_sign_endpoint_derived_when_public_not_configured(self) -> None:
        uploader = build_oss_uploader(
            OssUploaderConfig(bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env")
        )
        assert uploader is not None
        assert uploader._sign_endpoint == f"https://{_PUBLIC_ENDPOINT}"

    def test_sign_endpoint_explicit_public_wins(self) -> None:
        uploader = build_oss_uploader(
            OssUploaderConfig(
                bucket=_BUCKET,
                endpoint=_ENDPOINT,
                public_endpoint="oss.example.com",
                credential_mode="env",
            )
        )
        assert uploader is not None
        assert uploader._sign_endpoint == "https://oss.example.com"

    def test_sign_endpoint_same_as_upload_when_already_public(self) -> None:
        uploader = build_oss_uploader(
            OssUploaderConfig(bucket=_BUCKET, endpoint=_PUBLIC_ENDPOINT, credential_mode="env")
        )
        assert uploader is not None
        assert uploader._sign_endpoint == uploader._endpoint


class TestUploadDegradation:
    async def test_returns_none_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A credential-fetch error → None (the sink then falls back to a local path), and it is not raised outward."""
        uploader = OssArtifactUploader(
            OssUploaderConfig(bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env")
        )
        monkeypatch.setattr(
            uploader, "_acquire_auth", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        assert await uploader.upload("k", b"data") is None

    async def test_returns_none_on_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A timeout → None (an xlsx upload must not stall the whole report flow)."""

        class _SlowBucket:
            def put_object(self, key: str, data: bytes) -> None:
                import time

                time.sleep(5)

            def sign_url(self, method: str, key: str, expires: int) -> str:
                return "never"

        uploader = OssArtifactUploader(
            OssUploaderConfig(
                bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env", op_deadline_s=0.05
            )
        )
        monkeypatch.setattr(uploader, "_acquire_auth", object)
        monkeypatch.setattr(uploader, "_make_bucket", lambda auth, endpoint: _SlowBucket())

        assert await uploader.upload("k", b"data") is None

    async def test_returns_signed_url_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Success path: upload on the internal hostname, sign on the public hostname (the URL can be shared outside the VPC)."""
        seen: dict[str, object] = {}
        request_marker = ContextVar("oss_request_marker", default="")
        loop = asyncio.get_running_loop()
        submit = Mock(wraps=loop.run_in_executor)
        monkeypatch.setattr(loop, "run_in_executor", submit)

        class _FakeBucket:
            def __init__(self, endpoint: str) -> None:
                self.endpoint = endpoint

            def put_object(self, key: str, data: bytes) -> None:
                seen["request_marker"] = request_marker.get()
                seen["upload_endpoint"] = self.endpoint
                seen["key"] = key
                seen["size"] = len(data)

            def sign_url(self, method: str, key: str, expires: int) -> str:
                seen["sign_endpoint"] = self.endpoint
                seen["method"] = method
                seen["expires"] = expires
                # The endpoint already has a scheme (_make_bucket receives the normalized full hostname)
                return f"{self.endpoint}/{key}?Expires={expires}"

        uploader = OssArtifactUploader(
            OssUploaderConfig(bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env")
        )
        monkeypatch.setattr(uploader, "_acquire_auth", object)
        monkeypatch.setattr(uploader, "_make_bucket", lambda auth, endpoint: _FakeBucket(endpoint))

        token = request_marker.set("request-1")
        try:
            url = await uploader.upload("eval-reports/run-1/report.json", b"{}")
        finally:
            request_marker.reset(token)
        assert submit.call_args.args[0] is not None
        assert seen["request_marker"] == "request-1"

        # Upload uses the internal network and signing uses the public network (regression anchor: the old version used -internal for both, so the link would not open off the internal network)
        assert seen["upload_endpoint"] == f"https://{_ENDPOINT}"
        assert seen["sign_endpoint"] == f"https://{_PUBLIC_ENDPOINT}"
        assert url == (
            f"https://{_PUBLIC_ENDPOINT}/eval-reports/run-1/report.json"
            f"?Expires={DEFAULT_SIGN_EXPIRES_S}"
        )
        assert seen["method"] == "GET"
        assert seen["expires"] == DEFAULT_SIGN_EXPIRES_S
        assert seen["key"] == "eval-reports/run-1/report.json"

    async def test_satisfies_object_uploader_protocol(self) -> None:
        """It must satisfy the sink side's ObjectUploader protocol (signatures have to match before it can be injected)."""
        from benchmark.reporting.cluster_service.sink import build_sink

        uploader = build_oss_uploader(
            OssUploaderConfig(bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env")
        )
        # Being accepted by build_sink means the protocol matches (runtime duck typing + a static Protocol)
        sink = build_sink(local_root="/tmp/x", oss_client=uploader, oss_key_prefix="p")
        assert sink is not None
        assert asyncio.iscoroutinefunction(uploader.upload)


async def test_timed_out_upload_cannot_overwrite_new_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    writes: list[bytes] = []

    class ControlledBucket:
        def put_object(self, key: str, data: bytes) -> None:
            if data == b"old-report":
                started.set()
                assert release.wait(3)
            writes.append(data)
            if data == b"old-report":
                completed.set()

        def sign_url(self, method: str, key: str, expires: int) -> str:
            return "https://example.com/latest"

    old = OssArtifactUploader(
        OssUploaderConfig(
            bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env", op_deadline_s=0.1
        )
    )
    new = OssArtifactUploader(
        OssUploaderConfig(
            bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env", op_deadline_s=2
        )
    )
    for uploader in (old, new):
        monkeypatch.setattr(uploader, "_acquire_auth", object)
        monkeypatch.setattr(uploader, "_make_bucket", lambda auth, endpoint: ControlledBucket())
    try:
        assert await old.upload("eval/vui_agent/latest.json", b"old-report") is None
        assert started.is_set()
        task = asyncio.create_task(new.upload("eval/vui_agent/latest.json", b"new-report"))
        # A new success must not precede the old worker's completion.
        await asyncio.sleep(0.05)
        assert not task.done()
    finally:
        release.set()
    assert await task == "https://example.com/latest"
    assert completed.is_set()
    assert writes == [b"old-report", b"new-report"]


async def test_expired_lock_waiter_releases_worker_for_other_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import benchmark.reporting.cluster_service.oss_uploader as mod

    release = threading.Event()
    started = threading.Event()
    waiting = threading.Event()
    writes: list[str] = []
    original_lock = mod._object_upload_lock
    lock_calls: list[str] = []

    def track_lock(bucket: str, key: str) -> object:
        lock_calls.append(key)
        if lock_calls.count("latest.json") == 2:
            waiting.set()
        return original_lock(bucket, key)

    class ControlledBucket:
        def put_object(self, key: str, data: bytes) -> None:
            if data == b"old":
                started.set()
                assert release.wait(5)
            writes.append(key)

        def sign_url(self, method: str, key: str, expires: int) -> str:
            return f"https://example.com/{key}"

    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(mod, "_OSS_UPLOAD_EXECUTOR", pool)
    monkeypatch.setattr(mod, "_object_upload_lock", track_lock)
    blocked = OssArtifactUploader(
        OssUploaderConfig(
            bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env", op_deadline_s=0.2
        )
    )
    healthy = OssArtifactUploader(
        OssUploaderConfig(
            bucket=_BUCKET, endpoint=_ENDPOINT, credential_mode="env", op_deadline_s=1
        )
    )
    for uploader in (blocked, healthy):
        monkeypatch.setattr(uploader, "_acquire_auth", object)
        monkeypatch.setattr(uploader, "_make_bucket", lambda auth, endpoint: ControlledBucket())
    try:
        assert await blocked.upload("latest.json", b"old") is None
        assert started.is_set()
        assert await blocked.upload("latest.json", b"expired") is None
        assert waiting.is_set()
        assert await healthy.upload("healthy.json", b"ok") == "https://example.com/healthy.json"
        assert not release.is_set()
        assert writes == ["healthy.json"]
    finally:
        release.set()
        pool.shutdown(wait=True)
    assert writes == ["healthy.json", "latest.json"]
