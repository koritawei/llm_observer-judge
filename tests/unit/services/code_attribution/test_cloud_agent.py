"""aoned polling state machine and HTTP skeleton."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from benchmark.settings.setting import settings
from benchmark.attribution.code.cloud_agent import (
    Approval,
    Events,
    HttpTransport,
    events_from,
    is_readonly,
)
from benchmark.attribution.code.models import ModuleConfig, TaskPack
from benchmark.attribution.code.runner import (
    CodeAttributionModule,
    ModuleRuntime,
    module_config_from_settings,
)

_JSON = json.dumps(
    {
        "label_id": "plan.tool_orchestration",
        "tool_name": "should-clear",
        "module_root_cause": "编排策略选错了工具",
        "evidence_chain": (
            "症状是答非所问。读到 tool_orchestration_policy 选了错误工具，"
            "所以定 plan.tool_orchestration。"
        ),
        "fix_plan": "1. 调整编排策略\n2. 风险是误拒\n3. 验收看工具选择",
        "gate": None,
    },
    ensure_ascii=False,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeTransport:
    def __init__(
        self,
        polls: list[Events],
        *,
        auth_on_create: bool = False,
        repeat_last: bool = False,
    ) -> None:
        self.polls = list(polls)
        self.auth_on_create = auth_on_create
        self.repeat_last = repeat_last
        self.sessions = 0
        self.stops = 0
        self.deletes: list[str] = []
        self.answers: list[tuple[str, bool]] = []
        self.prompts: list[str] = []
        self._last: Events | None = None

    def create_session(self, *, prompt: str) -> str:
        self.sessions += 1
        self.prompts.append(prompt)
        if self.auth_on_create:
            from benchmark.attribution.code.cloud_agent import AuthError

            raise AuthError("aoned auth 401")
        return f"s{self.sessions}"

    def delete_session(self, session_id: str) -> None:
        self.deletes.append(session_id)

    def poll_events(self, session_id: str) -> Events:
        if self.polls:
            self._last = self.polls.pop(0)
            return self._last
        if self.repeat_last and self._last is not None:
            return self._last
        return Events(is_processing=False, final_text=_JSON)

    def answer_approval(self, session_id: str, approval_id: str, *, approve: bool) -> None:
        self.answers.append((approval_id, approve))

    def stop(self, session_id: str) -> None:
        self.stops += 1


def _task() -> TaskPack:
    return TaskPack(
        case_id="case-1",
        query="q",
        symptom_layer="planning",
        symptom_category="tool_selection",
        symptom_root_cause="选错工具",
        symptom_chain="选了不该用的工具",
        symptom_suggestion="改策略",
        trace_summary="trace",
    )


def _module(transport: FakeTransport, clock: _Clock | None = None) -> CodeAttributionModule:
    clock = clock or _Clock()
    return CodeAttributionModule(
        ModuleConfig(live_enabled=True, timeout_sec=5, poll_interval_sec=2),
        ModuleRuntime(transport=transport, clock=clock, sleep=clock.sleep),
    )


def test_readonly_unknown_kind_is_rejected() -> None:
    assert is_readonly(Approval(id="1", kind="read_file")) is True
    assert is_readonly(Approval(id="2", kind="write")) is False
    assert is_readonly(Approval(id="3", kind="")) is False
    assert is_readonly(Approval(id="4", kind="unread")) is False
    assert is_readonly(Approval(id="5", kind="readiness")) is False


def test_events_skip_approval_without_id() -> None:
    events = events_from(
        {
            "is_processing": True,
            "pending_approvals": [
                {"kind": "read_file"},
                {"id": "r", "kind": "read_file"},
            ],
        }
    )
    assert [item.id for item in events.pending_approvals] == ["r"]


def test_poll_approves_read_rejects_write_then_parses() -> None:
    transport = FakeTransport(
        [
            Events(
                is_processing=True,
                pending_approvals=[
                    Approval(id="r", kind="read_file"),
                    Approval(id="w", kind="edit_file"),
                ],
            ),
            Events(is_processing=False, final_text="```json\n" + _JSON + "\n```"),
        ]
    )
    clock = _Clock()
    result = _module(transport, clock).run(_task())
    assert result.status == "ok"
    assert result.label_id == "plan.tool_orchestration"
    assert result.tool_name == ""
    assert transport.answers == [("r", True), ("w", False)]
    assert transport.sessions == 1
    assert transport.deletes == []
    assert transport.stops == 0
    assert "case-1" in transport.prompts[0]


def test_timeout_retries_once() -> None:
    transport = FakeTransport([Events(is_processing=True)], repeat_last=True)
    clock = _Clock()
    module = _module(transport, clock)
    module.config.timeout_sec = 0
    result = module.run(_task())
    assert result.status == "timeout"
    assert transport.sessions == 2
    assert transport.stops == 2
    assert transport.deletes == []


def test_auth_error_does_not_retry() -> None:
    transport = FakeTransport([], auth_on_create=True)
    result = _module(transport).run(_task())
    assert result.status == "failed"
    assert "401" in result.error_message
    assert transport.sessions == 1
    assert transport.stops == 0
    assert transport.deletes == []


def test_live_disabled_skips_transport() -> None:
    transport = FakeTransport([])
    module = CodeAttributionModule(
        ModuleConfig(live_enabled=False),
        ModuleRuntime(transport=transport),
    )
    result = module.run(_task())
    assert result.status == "failed"
    assert result.error_message == "aoned live call disabled"
    assert transport.sessions == 0


def test_http_create_omits_unconfirmed_fields_and_maps_401() -> None:
    seen: dict[str, object] = {}

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b'{"session_id":"s1"}'

    def opener(request: urllib.request.Request, timeout: float = 30) -> _Resp:
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode())
        seen["auth"] = request.get_header("Authorization")
        return _Resp()

    config = ModuleConfig(
        base_url="https://sandbox/open-api/v1",
        token="ck_x",
        provider="claude",
        project_path="/home/admin/workspace/amap-ai-agent",
    )
    transport = HttpTransport(config, opener=opener)
    assert transport.create_session(prompt="ping") == "s1"
    assert seen["body"] == {
        "provider": "claude",
        "prompt": "ping",
        "project_path": "/home/admin/workspace/amap-ai-agent",
    }
    assert seen["url"] == "https://sandbox/open-api/v1/sessions"
    assert seen["auth"] == "Bearer ck_x"

    def denied(request: urllib.request.Request, timeout: float = 30) -> _Resp:
        raise urllib.error.HTTPError(request.full_url, 401, "no", hdrs=None, fp=io.BytesIO(b""))

    denied_transport = HttpTransport(config, opener=denied)
    try:
        denied_transport.poll_events("s1")
    except Exception as exc:  # noqa: BLE001
        assert type(exc).__name__ == "AuthError"
    else:
        raise AssertionError("expected auth error")


def test_load_config_reads_settings_fields(monkeypatch) -> None:
    monkeypatch.setattr(settings, "code_attribution_enabled", True)
    monkeypatch.setattr(settings, "cloud_agent_base_url", "https://sandbox/open-api/v1")
    monkeypatch.setattr(settings, "cloud_agent_token", "ck_x")
    monkeypatch.setattr(settings, "cloud_agent_project_path", "/home/admin/workspace/amap-ai-agent")

    config = module_config_from_settings()

    assert config.live_enabled is True
    assert config.base_url == "https://sandbox/open-api/v1"
    assert config.token == "ck_x"
    assert config.project_path == "/home/admin/workspace/amap-ai-agent"
    assert config.provider == "claude"
    assert config.poll_interval_sec == 2.0
    assert config.timeout_sec == 900.0
    assert not hasattr(config, "repo_path")
    assert not hasattr(config, "send_cwd")
    assert not hasattr(config, "send_mode")
