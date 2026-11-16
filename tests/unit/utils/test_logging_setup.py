"""All process logs → stdout; VIP root FileHandler is stripped."""

import io
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.utils.logging_setup import (
    _APP_STDIO_MARKER,
    app_namespaces_lack_stdio,
    ensure_application_logging,
    resolve_app_log_level,
)


def _hijack_root_like_vip(tmp_path: Path) -> None:
    """Simulate vip-py-client: sole root handler is a RotatingFileHandler."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = RotatingFileHandler(tmp_path / "vip-client.log", maxBytes=1_000_000, backupCount=1)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _swap_root_stdio_stream(buf: io.StringIO) -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, _APP_STDIO_MARKER, False):
            handler.stream = buf


def test_before_bind_business_logs_go_to_vip_file(tmp_path: Path) -> None:
    _hijack_root_like_vip(tmp_path)
    assert app_namespaces_lack_stdio() is True

    get_logger("benchmark.attribution.service.service").error("should_land_in_vip_file")
    assert "should_land_in_vip_file" in (tmp_path / "vip-client.log").read_text(encoding="utf-8")


def test_after_bind_all_logs_on_stdout_vip_file_unused(tmp_path: Path) -> None:
    _hijack_root_like_vip(tmp_path)

    assert ensure_application_logging(reason="unit_test") is True
    assert app_namespaces_lack_stdio() is False

    root = logging.getLogger()
    assert not any(isinstance(h, logging.FileHandler) for h in root.handlers)
    assert any(getattr(h, _APP_STDIO_MARKER, False) for h in root.handlers)

    buf = io.StringIO()
    _swap_root_stdio_stream(buf)

    get_logger("agent_eval.test_after").error("biz_on_stdout")
    get_logger("agentgraph.test_after").warning("graph_on_stdout")
    get_logger("benchmark.attribution.service.service").warning("app_pkg_on_stdout")
    logging.getLogger("vipserver_client.host_reactor").warning("vip_also_on_stdout")

    stdout_text = buf.getvalue()
    vip_text = (tmp_path / "vip-client.log").read_text(encoding="utf-8")

    assert "biz_on_stdout" in stdout_text
    assert "graph_on_stdout" in stdout_text
    assert "app_pkg_on_stdout" in stdout_text
    assert "vip_also_on_stdout" in stdout_text

    # VIP file handler was closed/removed — no new lines after reclaim.
    assert "biz_on_stdout" not in vip_text
    assert "vip_also_on_stdout" not in vip_text


def test_ensure_strips_file_handler_again_if_vip_reattaches(tmp_path: Path) -> None:
    """SDK init after create_app may re-hang FileHandler; re-call must strip again."""
    ensure_application_logging(reason="first")
    assert app_namespaces_lack_stdio() is False

    _hijack_root_like_vip(tmp_path)
    # Keep our stdout if hijack cleared it — hijack clears all; simulate reattach only.
    root = logging.getLogger()
    # _hijack cleared handlers; add VIP file then call ensure again.
    assert ensure_application_logging(reason="after_sdk") is True
    assert not any(isinstance(h, logging.FileHandler) for h in root.handlers)
    assert any(getattr(h, _APP_STDIO_MARKER, False) for h in root.handlers)


def test_ensure_application_logging_is_idempotent() -> None:
    ensure_application_logging(reason="first")
    assert app_namespaces_lack_stdio() is False
    assert ensure_application_logging(reason="second") is False


def test_teardown_legacy_stdout_handler_does_not_close_sys_stdout() -> None:
    """Legacy per-namespace StreamHandler(sys.stdout) must not close stdout."""
    from benchmark.utils.logging_setup import (
        _APP_STDIO_MARKER,
        _make_app_stdio_handler,
    )

    legacy = logging.getLogger("agent_eval")
    legacy.handlers.clear()
    handler = _make_app_stdio_handler()
    legacy.addHandler(handler)
    legacy.propagate = False

    ensure_application_logging(reason="teardown_legacy")

    assert handler not in legacy.handlers
    assert not sys.stdout.closed  # type: ignore[attr-defined]
    # The stdout handler newly attached to root is still writable
    buf = io.StringIO()
    _swap_root_stdio_stream(buf)
    get_logger("agent_eval.after_teardown").error("still_writable")
    assert "still_writable" in buf.getvalue()
    assert any(getattr(h, _APP_STDIO_MARKER, False) for h in logging.getLogger().handlers)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"LOG_LEVEL": "WARNING"}, logging.WARNING),
        ({"AGENT_EVAL_LOG_LEVEL": "ERROR"}, logging.ERROR),
        ({"LOG_LEVEL": "WARNING", "AGENT_EVAL_LOG_LEVEL": "DEBUG"}, logging.WARNING),
        ({}, logging.INFO),
    ],
)
def test_resolve_app_log_level(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: int
) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("AGENT_EVAL_LOG_LEVEL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert resolve_app_log_level() == expected


def test_ensure_application_logging_refreshes_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    ensure_application_logging(reason="info")
    assert logging.getLogger().level == logging.INFO

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert ensure_application_logging(reason="warning") is False
    assert logging.getLogger().level == logging.WARNING
    ours = [h for h in logging.getLogger().handlers if getattr(h, _APP_STDIO_MARKER, False)]
    assert ours and all(h.level == logging.WARNING for h in ours)


def test_warning_level_filters_info_from_child_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Child loggers created under INFO must not leak INFO after refresh to WARNING."""
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    ensure_application_logging(reason="seed")
    child = get_logger("agent_eval.child_level_test")
    # Simulate early get_logger pinning the child to INFO.
    logging.getLogger("agent_eval.child_level_test").setLevel(logging.INFO)

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    ensure_application_logging(reason="raise")

    buf = io.StringIO()
    _swap_root_stdio_stream(buf)
    child.info("should_be_filtered")
    child.warning("should_pass")
    text = buf.getvalue()
    assert "should_be_filtered" not in text
    assert "should_pass" in text


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_real_oss_sdk_error_does_not_expose_auth_material(
    monkeypatch: pytest.MonkeyPatch, level: str
) -> None:
    from benchmark.libs.storage import osscompat as oss2

    marker_signature = "FAKE_SIGNATURE_MUST_NOT_BE_LOGGED"
    marker_token = "FAKE_STS_TOKEN_MUST_NOT_BE_LOGGED"
    body = (
        "<Error><Code>SignatureDoesNotMatch</Code><Message>signature mismatch</Message>"
        f"<SignatureProvided>{marker_signature}</SignatureProvided>"
        f"<StringToSign>x-oss-security-token:{marker_token}</StringToSign></Error>"
    ).encode()

    class FakeResponse:
        status = 403
        headers = {"x-oss-request-id": "fake-request"}

        def read(self, amount: int) -> bytes:
            return body[:amount]

    class FakeSession:
        def do_request(self, request: object, timeout: float) -> FakeResponse:
            return FakeResponse()

    # Exercise even SDK children explicitly configured before app logging setup.
    for name in ("oss2", "oss2.api", "oss2.auth", "oss2.http"):
        monkeypatch.setattr(logging.getLogger(name), "level", logging.DEBUG)
    monkeypatch.setenv("LOG_LEVEL", level)
    ensure_application_logging(reason="oss-error-regression")
    buf = io.StringIO()
    _swap_root_stdio_stream(buf)
    bucket = oss2.Bucket(
        oss2.StsAuth("fake-access-id", "fake-secret", marker_token),
        "https://oss-cn-test.aliyuncs.com",
        "fake-bucket",
        session=FakeSession(),
    )
    with pytest.raises(oss2.exceptions.SignatureDoesNotMatch):
        bucket.put_object("latest.json", b"{}")
    logging.getLogger("app.oss_regression").warning("safe_business_diagnostic")
    text = buf.getvalue()
    assert "safe_business_diagnostic" in text
    assert marker_signature not in text
    assert marker_token not in text
    assert "fake-secret" not in text
