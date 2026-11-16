import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from benchmark.utils import trace_report
from benchmark.utils.trace_report import (
    eval_report_root,
    finish_report_child_span,
    get_report_client,
    report_child_scope,
    start_report_child_span,
    with_current_node_otel_parent,
)

_REPORT_PK = "pk-agent-eval-report"


def _install_trace(monkeypatch: pytest.MonkeyPatch, client: Any) -> MagicMock:
    get_client = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(get_client=get_client))
    monkeypatch.setenv("TRACE_PUBLIC_KEY", _REPORT_PK)
    return get_client


def test_get_report_client_pins_trace_public_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    get_client = _install_trace(monkeypatch, client)

    assert get_report_client() is client
    get_client.assert_called_once_with(public_key=_REPORT_PK)


def test_get_report_client_reads_public_key_on_each_call(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    get_client = _install_trace(monkeypatch, client)
    for value in ("pk-other-report", "", None):
        if value is None:
            monkeypatch.delenv("TRACE_PUBLIC_KEY", raising=False)
        else:
            monkeypatch.setenv("TRACE_PUBLIC_KEY", value)
        assert get_report_client() is client
        get_client.assert_called_with(public_key=value or None)


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE ", " On "])
def test_local_span_logging_reads_truthy_environment_on_use(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    caplog.set_level("INFO", logger="agent_eval.trace_report")
    monkeypatch.delenv("AGENT_EVAL_LOCAL_SPAN_LOG", raising=False)
    trace_report._log_local_span_event(event="start", name="disabled")
    assert caplog.text == ""
    monkeypatch.setenv("AGENT_EVAL_LOCAL_SPAN_LOG", value)
    trace_report._log_local_span_event(event="start", name="enabled")
    assert "event=start name=enabled" in caplog.text


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "enabled"])
def test_local_span_logging_rejects_other_environment_values(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    caplog.set_level("INFO", logger="agent_eval.trace_report")
    monkeypatch.setenv("AGENT_EVAL_LOCAL_SPAN_LOG", value)
    trace_report._log_local_span_event(event="start", name="disabled")
    assert caplog.text == ""


@pytest.mark.parametrize("value", [None, "", "  ", "invalid", "0", "-1", "1.5"])
def test_local_span_preview_uses_default_for_invalid_limit(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("AGENT_EVAL_LOCAL_SPAN_LOG_MAX_CHARS", raising=False)
    else:
        monkeypatch.setenv("AGENT_EVAL_LOCAL_SPAN_LOG_MAX_CHARS", value)
    assert trace_report._safe_json_preview("x" * 12_001) == ('"' + "x" * 11_999 + "…<truncated>")


def test_local_span_preview_reads_limit_on_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_EVAL_LOCAL_SPAN_LOG_MAX_CHARS", " 3 ")
    assert trace_report._safe_json_preview("abc") == '"ab…<truncated>'
    monkeypatch.setenv("AGENT_EVAL_LOCAL_SPAN_LOG_MAX_CHARS", "5")
    assert trace_report._safe_json_preview("abc") == '"abc"'


def _inner_span(span: Any) -> Any:
    return getattr(span, "_inner", span)


def test_eval_report_root_reuses_get_client_and_opens_one_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = object()
    root_context = MagicMock()
    root_context.__enter__.return_value = observation
    client = MagicMock()
    client.start_as_current_observation.return_value = root_context
    get_client = _install_trace(monkeypatch, client)

    with eval_report_root(
        name="evaluation",
        input_value={"query": "nearby coffee"},
        metadata={"eval_id": "eval-1"},
    ) as actual:
        assert _inner_span(actual) is observation

    get_client.assert_called_once_with(public_key=_REPORT_PK)
    client.start_as_current_observation.assert_called_once_with(
        name="evaluation",
        as_type="span",
        input={"query": "nearby coffee"},
        metadata={"eval_id": "eval-1"},
    )


def test_child_span_requires_active_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    client.get_current_trace_id.return_value = None
    _install_trace(monkeypatch, client)

    result = start_report_child_span(name="resolve", input_value={"trace_id": "trace-1"})

    assert result is not None
    assert _inner_span(result) is None
    client.start_observation.assert_not_called()


def test_child_span_uses_start_observation_not_new_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = object()
    client = MagicMock()
    client.get_current_trace_id.return_value = "trace-1"
    client.start_observation.return_value = span
    get_client = _install_trace(monkeypatch, client)
    monkeypatch.setattr(
        "benchmark.utils.trace_report.with_current_node_otel_parent",
        contextmanager(lambda: (yield)),
    )

    result = start_report_child_span(name="resolve", input_value={"eval_id": "eval-1"})

    assert _inner_span(result) is span
    get_client.assert_called_once_with(public_key=_REPORT_PK)
    client.start_observation.assert_called_once_with(
        name="resolve",
        as_type="span",
        input={"eval_id": "eval-1"},
    )
    client.start_as_current_observation.assert_not_called()


def test_with_current_node_otel_parent_attaches_node_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentelemetry import context as otel_context_api
    from opentelemetry import trace as otel_trace_api

    node_span = object()
    calls: list[tuple[str, object]] = []

    class _FakeMw:
        @staticmethod
        def get_current_node_otel_span() -> object:
            return node_span

    monkeypatch.setattr(
        "benchmark.libs.agenthost.observability.middleware.trace_middleware.TraceMiddleware",
        _FakeMw,
    )
    monkeypatch.setattr(
        otel_trace_api,
        "set_span_in_context",
        lambda span: ("ctx", span),
    )
    monkeypatch.setattr(
        otel_context_api,
        "attach",
        lambda ctx: calls.append(("attach", ctx)) or "token-1",
    )
    monkeypatch.setattr(
        otel_context_api,
        "detach",
        lambda token: calls.append(("detach", token)),
    )

    with with_current_node_otel_parent():
        assert calls == [("attach", ("ctx", node_span))]

    assert calls == [
        ("attach", ("ctx", node_span)),
        ("detach", "token-1"),
    ]


def test_with_current_node_otel_parent_noop_without_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMw:
        @staticmethod
        def get_current_node_otel_span() -> None:
            return None

    monkeypatch.setattr(
        "benchmark.libs.agenthost.observability.middleware.trace_middleware.TraceMiddleware",
        _FakeMw,
    )

    with with_current_node_otel_parent():
        pass


def test_child_scope_sets_current_observation_for_nested_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = object()
    child_context = MagicMock()
    child_context.__enter__.return_value = observation
    client = MagicMock()
    client.get_current_trace_id.return_value = "trace-1"
    client.start_as_current_observation.return_value = child_context
    _install_trace(monkeypatch, client)

    with report_child_scope(
        name="triage_attribution",
        input_value={"trace_id": "trace-1"},
    ) as actual:
        assert _inner_span(actual) is observation

    client.start_as_current_observation.assert_called_once_with(
        name="triage_attribution",
        as_type="span",
        input={"trace_id": "trace-1"},
    )
    client.start_observation.assert_not_called()


@pytest.mark.parametrize(
    ("output", "exc", "expected_update"),
    [
        ({"status": "ok"}, None, {"output": {"status": "ok"}}),
        (None, ValueError("failed"), {"level": "ERROR", "status_message": "failed"}),
    ],
)
def test_finish_child_span_updates_and_ends(
    output: dict[str, Any] | None,
    exc: BaseException | None,
    expected_update: dict[str, Any],
) -> None:
    span = MagicMock()

    finish_report_child_span(span, output=output, exc=exc)

    span.update.assert_called_once_with(**expected_update)
    span.end.assert_called_once_with()


def test_eval_report_root_degrades_when_trace_open_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_client = MagicMock(side_effect=RuntimeError("unavailable"))
    monkeypatch.setitem(sys.modules, "langfuse", SimpleNamespace(get_client=get_client))

    with eval_report_root(name="evaluation") as observation:
        assert _inner_span(observation) is None


def test_eval_report_root_does_not_swallow_business_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def root_context() -> Any:
        yield object()

    client = MagicMock()
    client.start_as_current_observation.return_value = root_context()
    _install_trace(monkeypatch, client)

    with pytest.raises(ValueError, match="business failed"):
        with eval_report_root(name="evaluation"):
            raise ValueError("business failed")


def test_local_span_log_records_input_update_and_end(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MagicMock()
    client.get_current_trace_id.return_value = None
    _install_trace(monkeypatch, client)
    monkeypatch.setenv("AGENT_EVAL_LOCAL_SPAN_LOG", "1")
    caplog.set_level("INFO", logger="agent_eval.trace_report")

    span = start_report_child_span(name="resolve", input_value={"trace_id": "trace-1"})
    finish_report_child_span(span, output={"status": "ok"})

    assert "trace_report.local_span event=start name=resolve" in caplog.text
    assert "trace-1" in caplog.text
    assert "trace_report.local_span event=update name=resolve" in caplog.text
    assert '"status": "ok"' in caplog.text
    assert "trace_report.local_span event=end name=resolve" in caplog.text


def test_finish_child_span_is_noop_for_none() -> None:
    finish_report_child_span(None, output={"status": "ok"})


@pytest.mark.parametrize(
    "trace_id", [None, "", "0" * 32, "invalid", "abcdef0123456789abcdef0123456789"]
)
def test_get_report_trace_id_validates_root_id(trace_id):
    from benchmark.utils.trace_report import get_report_trace_id

    root = SimpleNamespace(
        trace_id=trace_id,
        _otel_span=SimpleNamespace(is_recording=lambda: True),
    )
    expected = trace_id if trace_id == "abcdef0123456789abcdef0123456789" else None
    assert get_report_trace_id(root) == expected


def test_get_report_trace_id_rejects_noop_inherited_parent():
    from benchmark.utils.trace_report import get_report_trace_id
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    parent_id = "abcdef0123456789abcdef0123456789"
    span = NonRecordingSpan(
        SpanContext(
            trace_id=int(parent_id, 16),
            span_id=1,
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )
    assert get_report_trace_id(SimpleNamespace(trace_id=parent_id, _otel_span=span)) is None


def test_get_report_trace_id_tolerates_sdk_failure():
    from benchmark.utils.trace_report import get_report_trace_id

    class BrokenSpan:
        @property
        def trace_id(self):
            raise RuntimeError("report SDK unavailable")

    assert get_report_trace_id(BrokenSpan()) is None
    assert get_report_trace_id(None) is None
    assert get_report_trace_id(SimpleNamespace(trace_id="a" * 32)) is None


@pytest.mark.parametrize("recording", [True, False])
def test_capture_report_trace_id_only_records_root(monkeypatch, recording):
    from benchmark.utils.trace_report import capture_report_trace_id

    client = MagicMock()
    root = SimpleNamespace(
        trace_id="a" * 32, _otel_span=SimpleNamespace(is_recording=lambda: recording)
    )
    client.start_as_current_observation.return_value.__enter__.return_value = root
    _install_trace(monkeypatch, client)
    with capture_report_trace_id() as report:
        with eval_report_root(name="outer"):
            root.trace_id = "b" * 32
            with eval_report_root(name="nested"):
                pass
        assert report.trace_id == ("a" * 32 if recording else None)


def test_capture_report_trace_id_resets_after_exception(monkeypatch):
    from benchmark.utils.trace_report import capture_report_trace_id

    root = MagicMock()
    root.trace_id = "a" * 32
    root._otel_span.is_recording.return_value = True
    client = MagicMock()
    client.start_as_current_observation.return_value.__enter__.return_value = root
    _install_trace(monkeypatch, client)
    with capture_report_trace_id() as outer:
        with pytest.raises(ValueError), capture_report_trace_id() as inner:
            with eval_report_root(name="failed"):
                raise ValueError("evaluation failed")
        assert inner.trace_id == "a" * 32
        assert outer.trace_id is None
        root.trace_id = "b" * 32
        with eval_report_root(name="next"):
            pass
    assert outer.trace_id == "b" * 32
    with capture_report_trace_id() as fresh:
        assert fresh.trace_id is None


@pytest.mark.asyncio
async def test_capture_report_trace_id_isolates_concurrent_requests(monkeypatch):
    import asyncio

    from benchmark.utils.trace_report import capture_report_trace_id

    @contextmanager
    def root_context(**kwargs):
        yield SimpleNamespace(
            trace_id=kwargs["name"] * 32,
            _otel_span=SimpleNamespace(is_recording=lambda: True),
        )

    client = SimpleNamespace(start_as_current_observation=root_context)
    _install_trace(monkeypatch, client)
    both_started = asyncio.Event()
    started = 0

    async def request(name):
        nonlocal started
        with capture_report_trace_id() as report:
            with eval_report_root(name=name):
                started += 1
                if started == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=1)
            return report.trace_id

    assert await asyncio.gather(request("a"), request("b")) == ["a" * 32, "b" * 32]


def test_eval_report_root_binds_exported_parent_and_fresh_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = SimpleNamespace(_otel_span="otel-root")
    root_context = MagicMock()
    root_context.__enter__.return_value = observation
    client = MagicMock()
    client.start_as_current_observation.return_value = root_context
    _install_trace(monkeypatch, client)

    bound: list[object] = []
    scope_events: list[str] = []

    @contextmanager
    def fake_bind(obs: object) -> Any:
        bound.append(obs)
        yield

    class _FakeScope:
        @contextmanager
        def bind(self) -> Any:
            scope_events.append("enter")
            yield
            scope_events.append("exit")

    monkeypatch.setattr(
        "benchmark.utils.trace_report.bind_trace_parent_observation",
        fake_bind,
    )
    monkeypatch.setattr(
        "benchmark.utils.trace_report.TraceRequestScope",
        _FakeScope,
    )

    with eval_report_root(name="evaluation") as actual:
        assert actual._inner is observation
        assert len(bound) == 1
        assert bound[0] is observation
        assert scope_events == ["enter"]

    assert len(bound) == 1
    assert bound[0] is observation
    assert scope_events == ["enter", "exit"]


def test_eval_report_root_skips_parent_bind_when_observation_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_context = MagicMock()
    root_context.__enter__.return_value = None
    client = MagicMock()
    client.start_as_current_observation.return_value = root_context
    _install_trace(monkeypatch, client)

    bind = MagicMock()
    monkeypatch.setattr(
        "benchmark.utils.trace_report.bind_trace_parent_observation",
        bind,
    )
    monkeypatch.setattr(
        "benchmark.utils.trace_report.TraceRequestScope",
        MagicMock(),
    )

    with eval_report_root(name="evaluation") as actual:
        assert actual._inner is None

    bind.assert_not_called()


def test_child_scope_rebinds_exported_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = SimpleNamespace(_otel_span="otel-triage")
    child_context = MagicMock()
    child_context.__enter__.return_value = observation
    client = MagicMock()
    client.get_current_trace_id.return_value = "trace-1"
    client.start_as_current_observation.return_value = child_context
    _install_trace(monkeypatch, client)

    bound: list[object] = []

    @contextmanager
    def fake_bind(obs: object) -> Any:
        bound.append(obs)
        yield

    monkeypatch.setattr(
        "benchmark.utils.trace_report.bind_trace_parent_observation",
        fake_bind,
    )

    with report_child_scope(name="triage_attribution") as actual:
        assert actual._inner is observation
        assert len(bound) == 1
        assert bound[0] is observation

    assert len(bound) == 1
    assert bound[0] is observation


def test_child_scope_does_not_swallow_business_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def child_context() -> Any:
        yield SimpleNamespace(_otel_span="otel-triage")

    client = MagicMock()
    client.get_current_trace_id.return_value = "trace-1"
    client.start_as_current_observation.return_value = child_context()
    _install_trace(monkeypatch, client)
    monkeypatch.setattr(
        "benchmark.utils.trace_report.bind_trace_parent_observation",
        contextmanager(lambda _obs: (yield)),
    )

    with pytest.raises(ValueError, match="triage failed"):
        with report_child_scope(name="triage_attribution"):
            raise ValueError("triage failed")


def test_report_scopes_bind_real_otel_parent_and_restore_context(monkeypatch):
    from benchmark.libs.agenthost.observability import activate_trace_parent_context
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    def span(trace_id):
        return NonRecordingSpan(
            SpanContext(
                trace_id=trace_id,
                span_id=1,
                is_remote=False,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
        )

    outer, root_span, child_span = span(1), span(2), span(3)
    root = SimpleNamespace(_otel_span=root_span)
    child = SimpleNamespace(_otel_span=child_span)
    root_cm, child_cm = MagicMock(), MagicMock()
    root_cm.__enter__.return_value = root
    child_cm.__enter__.return_value = child
    client = MagicMock()
    client.get_current_trace_id.return_value = "2" * 32
    client.start_as_current_observation.side_effect = [root_cm, child_cm]
    _install_trace(monkeypatch, client)

    with otel_trace.use_span(outer, end_on_exit=False):
        with eval_report_root(name="root"):
            with activate_trace_parent_context():
                assert otel_trace.get_current_span() is root_span
                with report_child_scope(name="child"):
                    with activate_trace_parent_context():
                        assert otel_trace.get_current_span() is child_span
                with activate_trace_parent_context():
                    assert otel_trace.get_current_span() is root_span
        assert otel_trace.get_current_span() is outer
