from unittest.mock import AsyncMock

import pytest
from benchmark.attribution.badcase.publisher import (
    NotificationConfig,
    NotificationCreateError,
    _parse_workitem_id,
    _parse_workitem_url,
    _sanitize_workitem_type,
    create_notification_task,
    create_workitem_reference,
    map_priority_to_notification,
    update_workitem_body,
)
from benchmark.attribution.badcase.notification_content import (
    NOTIFICATION_AI_CONTEXT_SCHEMA_VERSION,
    NotificationWorkitemOverrides,
    build_workitem_payload,
    render_ai_context,
    render_human_summary,
)
from benchmark.attribution.badcase.schemas import (
    NotificationDispatchStatus,
    NotificationWorkitemPayload,
    AttributionResult,
)


def _result() -> AttributionResult:
    return AttributionResult(
        trace_id="trace-1",
        case_id="case-1",
        category="prompt",
        root_cause="final answer ignored required JSON format",
        confidence=0.91,
        reasoning="The failure node returned text instead of JSON.",
        evidence_refs=["evidence:n3"],
        action_hint="fix prompt output contract",
    )


class _FakeProcess:
    """Capture-only process test double; always returns workitem_id=123."""

    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return (b'{"workitem_id": "123"}', b"")


def install_subprocess_capture(monkeypatch) -> dict[str, list[str]]:
    """Monkeypatch asyncio.create_subprocess_exec and store cmd on the returned dict."""

    captured: dict[str, list[str]] = {}

    async def _fake_exec(*cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakeProcess()

    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_exec)
    return captured


@pytest.mark.asyncio
async def test_create_notification_task_disabled_is_skipped() -> None:
    result = await create_notification_task(
        notification_config=NotificationConfig(enabled=False),
        attribution_result=_result(),
    )

    assert result.status == NotificationDispatchStatus.SKIPPED
    assert result.payload is not None
    assert result.payload.title.startswith("[AgentEval Badcase] prompt")


@pytest.mark.asyncio
async def test_create_notification_task_missing_project_is_blocked() -> None:
    result = await create_notification_task(
        notification_config=NotificationConfig(enabled=True),
        attribution_result=_result(),
    )

    assert result.status == NotificationDispatchStatus.BLOCKED
    assert "project" in result.message.lower()


@pytest.mark.asyncio
async def test_create_notification_task_dry_run_is_skipped() -> None:
    result = await create_notification_task(
        notification_config=NotificationConfig(
            enabled=True,
            dry_run=True,
            dispatch_profile="agent-eval",
            project="2165583",
            category="bug",
        ),
        attribution_result=_result(),
    )

    assert result.status == NotificationDispatchStatus.SKIPPED
    assert "dry_run" in result.message


def _enabled_config() -> NotificationConfig:
    return NotificationConfig(
        enabled=True,
        dry_run=False,
        dispatch_profile="agent-eval",
        project="2165583",
        category="bug",
        workitem_type="bug",
        title_prefix="[badcase]",
        assignee="alice",
    )


@pytest.mark.asyncio
async def test_create_notification_task_created_when_workitem_succeeds(monkeypatch) -> None:
    from benchmark.attribution.badcase.publisher import NotificationWorkitemReference

    async def _fake_create_workitem_reference(
        _payload: object, _config: object
    ) -> NotificationWorkitemReference:
        return NotificationWorkitemReference(
            workitem_id="W-12345",
            url="https://project.aone.alibaba-inc.com/v2/project/2165583/bug#openWorkitemIdentifier=W-12345",
        )

    monkeypatch.setattr(
        "benchmark.attribution.badcase.publisher.create_workitem_reference",
        _fake_create_workitem_reference,
    )
    result = await create_notification_task(
        notification_config=_enabled_config(),
        attribution_result=_result(),
    )

    assert result.status == NotificationDispatchStatus.CREATED
    assert result.workitem_id == "W-12345"
    assert result.workitem_url.endswith("openWorkitemIdentifier=W-12345")
    assert result.payload is not None


@pytest.mark.asyncio
async def test_create_notification_task_failed_when_workitem_raises(monkeypatch) -> None:
    async def _boom(_payload: object, _config: object) -> object:
        raise RuntimeError("a1 cli unavailable")

    monkeypatch.setattr(
        "benchmark.attribution.badcase.publisher.create_workitem_reference",
        _boom,
    )
    result = await create_notification_task(
        notification_config=_enabled_config(),
        attribution_result=_result(),
    )

    assert result.status == NotificationDispatchStatus.FAILED
    assert "a1 cli unavailable" in (result.message or "")


@pytest.mark.asyncio
async def test_create_notification_task_blocked_when_assignee_missing() -> None:
    """Empty assignee (missing on both the per-run config and RemoteConfig) → block the black-hole ticket and do not create one."""
    config = _enabled_config().model_copy(update={"assignee": ""})

    result = await create_notification_task(
        notification_config=config,
        attribution_result=_result(),
    )

    assert result.status == NotificationDispatchStatus.BLOCKED
    assert "assignee" in (result.message or "")


@pytest.mark.asyncio
async def test_create_notification_task_dispatches_when_attribution_category_unknown(
    monkeypatch,
) -> None:
    """An attribution category of unknown (judge did not send it back) still creates a ticket.

    The only dispatch criterion is the score threshold (should_dispatch_attribution);
    unknown appears only in the ticket title and does not block ticket creation.
    """

    from benchmark.attribution.badcase.publisher import NotificationWorkitemReference

    async def _fake_create_workitem_reference(
        _payload: object, _config: object
    ) -> NotificationWorkitemReference:
        return NotificationWorkitemReference(workitem_id="W-99999", url="https://aone.example/W-99999")

    monkeypatch.setattr(
        "benchmark.attribution.badcase.publisher.create_workitem_reference",
        _fake_create_workitem_reference,
    )
    result = await create_notification_task(
        notification_config=_enabled_config(),
        attribution_result=_result().model_copy(update={"category": "unknown"}),
    )

    assert result.status == NotificationDispatchStatus.CREATED
    assert result.workitem_id == "W-99999"
    assert result.workitem_url == "https://aone.example/W-99999"
    assert "unknown" in (result.payload.title if result.payload else "")


def test_parse_workitem_id_from_cli_json() -> None:
    text = '{"workitem_id": "67890123"}'
    assert _parse_workitem_id(text) == "67890123"


def test_parse_workitem_id_rejects_non_json_output() -> None:
    """When the CLI fails silently and prints non-JSON text, raise, and do not let a regex mistake a number for a ticket id."""
    with pytest.raises(NotificationCreateError, match="not JSON"):
        _parse_workitem_id("created workitem 54321098 ok")


def test_parse_workitem_id_rejects_json_without_id() -> None:
    """JSON that is valid but has no id field (for example an error body) → raise rather than return the raw text."""
    with pytest.raises(NotificationCreateError, match="no workitem id"):
        _parse_workitem_id('{"success": false, "errorCode": 85703548}')


def test_parse_workitem_url_normalizes_legacy_path() -> None:
    text = '{"workitem_id": "67890123", "url": "https://project.aone.alibaba-inc.com/v2/project/2165583/task/67890123"}'
    assert _parse_workitem_url(text) == (
        "https://project.aone.alibaba-inc.com/v2/project/2165583/task"
        "#openWorkitemIdentifier=67890123"
    )


def test_parse_workitem_url_keeps_notification_view_identifier() -> None:
    url = (
        "https://project.aone.alibaba-inc.com/v2/project/2174105/req"
        "#viewIdentifier=d7f112f9d023e2108fa1b0d8&openWorkitemIdentifier=85881892"
    )

    assert _parse_workitem_url(f'{{"workitem_id": "85881892", "url": "{url}"}}') == url


@pytest.mark.asyncio
async def test_create_workitem_reference_falls_back_to_configured_project_url(monkeypatch) -> None:
    install_subprocess_capture(monkeypatch)
    payload = NotificationWorkitemPayload(project="2165583", category="task", title="t", body="b")

    reference = await create_workitem_reference(
        payload, NotificationConfig(enabled=True, project="2165583")
    )

    assert reference.workitem_id == "123"
    assert reference.url == (
        "https://project.aone.alibaba-inc.com/v2/project/2165583/task#openWorkitemIdentifier=123"
    )


@pytest.mark.asyncio
async def test_create_workitem_reference_uses_configured_view_url_template(monkeypatch) -> None:
    install_subprocess_capture(monkeypatch)
    monkeypatch.setenv(
        "NOTIFICATION_WORKITEM_URL_TEMPLATE",
        "https://project.aone.alibaba-inc.com/v2/project/{project_id}/{category}"
        "#viewIdentifier=project-view&openWorkitemIdentifier={workitem_id}",
    )
    payload = NotificationWorkitemPayload(project="2174105", category="req", title="t", body="b")

    reference = await create_workitem_reference(
        payload, NotificationConfig(enabled=True, project="2174105")
    )

    assert reference.url == (
        "https://project.aone.alibaba-inc.com/v2/project/2174105/req"
        "#viewIdentifier=project-view&openWorkitemIdentifier=123"
    )


@pytest.mark.asyncio
async def test_update_workitem_body_uses_a1_update_command(monkeypatch) -> None:
    captured = install_subprocess_capture(monkeypatch)

    await update_workitem_body("90000001", "# updated", NotificationConfig(enabled=True, project="2165583"))

    cmd = captured["cmd"]
    assert cmd[:5] == [cmd[0], "project", "workitem", "update", "90000001"]
    assert cmd[cmd.index("--body") + 1] == "# updated"
    # a1 update does not accept --project (the workitem id already locates it)
    assert "--project" not in cmd


@pytest.mark.asyncio
async def test_update_workitem_assignee_uses_a1_assignee_flag(monkeypatch) -> None:
    from benchmark.attribution.badcase.publisher import update_workitem_assignee

    captured = install_subprocess_capture(monkeypatch)
    await update_workitem_assignee(
        "90000001",
        "八鹤",
        NotificationConfig(enabled=True, project="2165583"),
    )
    cmd = captured["cmd"]
    assert cmd[:5] == [cmd[0], "project", "workitem", "update", "90000001"]
    assert cmd[cmd.index("--assignee") + 1] == "八鹤"
    assert "--project" not in cmd


def test_build_workitem_payload_contains_core_attribution_fields() -> None:
    payload = build_workitem_payload(
        _result(),
        project="123",
        category="task",
    )

    assert payload.project == "123"
    assert "trace-1" in payload.title
    assert "final answer ignored required JSON format" in payload.body
    assert "evidence:n3" in payload.body
    assert "## 归因摘要（给人看）" in payload.body
    assert "## AI Context（给后续自动化读取）" in payload.body


def test_notification_content_has_human_and_ai_sections() -> None:
    human = render_human_summary(_result())
    ai_context = render_ai_context(_result())

    assert "归因类别: prompt" in human
    assert "fix prompt output contract" in human
    assert "- Trace URL: -" in human  # Missing empty value → '-'
    assert "- Query: -" in human
    assert f'"schema_version": "{NOTIFICATION_AI_CONTEXT_SCHEMA_VERSION}"' in ai_context
    assert '"evidence_refs": [' in ai_context
    assert '"trace_url": ""' in ai_context
    assert '"query": ""' in ai_context


def test_render_human_summary_trace_url_populated() -> None:
    """When trace_url is set → the summary renders a Trace URL line, and AI Context carries the link."""
    result = _result().model_copy(
        update={"trace_url": "https://trace.example.com/t/1"},
    )
    human = render_human_summary(result)
    ai_context = render_ai_context(result)

    assert "- Trace URL: https://trace.example.com/t/1" in human
    assert '"trace_url": "https://trace.example.com/t/1"' in ai_context


def test_render_query_same_level_as_trace_id() -> None:
    """query is written at the same level as trace_id into the ticket summary and the AI Context JSON."""
    result = _result().model_copy(update={"query": "附近加油站"})
    human = render_human_summary(result)
    ai_context = render_ai_context(result)

    assert "- Query: 附近加油站" in human
    assert '"query": "附近加油站"' in ai_context
    assert '"trace_id": "trace-1"' in ai_context


@pytest.mark.parametrize(
    "raw_type,category,expected",
    [
        ("bug", "bug", ""),  # Misconfiguration: same value as category → skip
        ("BUG", "bug", ""),  # Case-insensitive
        ("task", "task", ""),  # Same value as task
        ("req", "req", ""),  # Same value as req
        ("", "bug", ""),  # Empty → skip
        ("  ", "bug", ""),
        ("缺陷(BUG)", "bug", "缺陷(BUG)"),  # Legal project subtype → keep
        ("故事(STO)", "req", "故事(STO)"),
    ],
)
def test_sanitize_workitem_type_filters_misconfigured(raw_type, category, expected) -> None:
    assert _sanitize_workitem_type(raw_type, category) == expected


@pytest.mark.asyncio
async def test_create_workitem_omits_type_when_misconfigured(monkeypatch) -> None:
    """When --type is mis-set to category, omit the whole flag from the a1 CLI."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem
    from benchmark.attribution.badcase.schemas import NotificationWorkitemPayload

    payload = NotificationWorkitemPayload(
        project="2165583",
        category="bug",
        type="bug",  # Misconfigured
        title="t",
        body="b",
    )
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    cmd = captured["cmd"]
    assert "--type" not in cmd
    assert "--category" in cmd


@pytest.mark.asyncio
async def test_create_workitem_participants_comma_joined(monkeypatch) -> None:
    """Multiple participants → join them with commas for a1 --participant (Aone CC role)."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem
    from benchmark.attribution.badcase.schemas import NotificationWorkitemPayload

    payload = NotificationWorkitemPayload(
        project="2165583",
        category="bug",
        title="t",
        body="b",
        assignee="alice",
        participants=["bob", "carol"],
    )
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    cmd = captured["cmd"]
    assert cmd[cmd.index("--participant") + 1] == "bob,carol"
    assert cmd[cmd.index("--assignee") + 1] == "alice"


@pytest.mark.asyncio
async def test_create_workitem_omits_participants_when_empty(monkeypatch) -> None:
    """Empty participants → do not pass --participant."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem
    from benchmark.attribution.badcase.schemas import NotificationWorkitemPayload

    payload = NotificationWorkitemPayload(
        project="2165583",
        category="bug",
        title="t",
        body="b",
        assignee="alice",
    )
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    assert "--participant" not in captured["cmd"]


# ---------------------------------------------------------------------------
# priority mapping + payload passthrough + a1 --priority command construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "priority,expected",
    [
        ("P0", "urgent"),
        ("P1", "high"),
        ("P2", "medium"),
        (None, "medium"),
    ],
)
def test_map_priority_to_notification(priority, expected) -> None:
    from benchmark.domain.enums import CasePriority

    input_value = CasePriority(priority) if priority else None
    assert map_priority_to_notification(input_value) == expected


def test_build_workitem_payload_propagates_priority_and_project_url() -> None:
    payload = build_workitem_payload(
        _result(),
        project="123",
        category="bug",
        overrides=NotificationWorkitemOverrides(
            priority="urgent",
            project_url="https://example.com/p/123",
        ),
    )

    assert payload.priority == "urgent"
    assert "https://example.com/p/123" in payload.body
    assert "[关联看板]" in payload.body


def test_build_workitem_payload_omits_priority_and_url_when_empty() -> None:
    payload = build_workitem_payload(
        _result(),
        project="123",
        category="bug",
    )

    assert payload.priority == ""
    assert payload.project_url == ""
    assert "[关联看板]" not in payload.body


def test_build_workitem_payload_propagates_assignee() -> None:
    payload = build_workitem_payload(
        _result(),
        project="123",
        category="bug",
        overrides=NotificationWorkitemOverrides(assignee="WB001"),
    )

    assert payload.assignee == "WB001"


def test_build_workitem_payload_reads_style_from_options() -> None:
    """Ticket style / CC / tags are read from options (AoneConfig is duck-compatible with AoneRuntimeOptions)."""
    from benchmark.attribution.badcase.schemas import NotificationRuntimeOptions

    payload = build_workitem_payload(
        _result(),
        project="123",
        category="bug",
        options=NotificationRuntimeOptions(
            workitem_type="缺陷(BUG)",
            participants=["bob", "carol"],
            tags=["eval"],
        ),
    )

    assert payload.type == "缺陷(BUG)"
    assert payload.participants == ["bob", "carol"]
    assert payload.tags == ["eval"]
    assert payload.title.startswith("[AgentEval Badcase]")


def test_build_workitem_payload_defaults_without_options() -> None:
    """options omitted → empty style / default prefix / no CC and no tags."""
    payload = build_workitem_payload(
        _result(),
        project="123",
        category="bug",
    )

    assert payload.type == ""
    assert payload.participants == []
    assert payload.tags == []
    assert payload.title.startswith("[AgentEval Badcase]")


@pytest.mark.asyncio
async def test_create_workitem_passes_priority_flag(monkeypatch) -> None:
    """When priority is non-empty, cmd must contain --priority urgent."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem

    payload = NotificationWorkitemPayload(
        project="2165583",
        category="bug",
        type="",
        title="t",
        body="b",
        priority="urgent",
    )
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    cmd = captured["cmd"]
    assert "--priority" in cmd
    idx = cmd.index("--priority")
    assert cmd[idx + 1] == "urgent"


@pytest.mark.asyncio
async def test_create_workitem_omits_priority_when_empty(monkeypatch) -> None:
    """When priority is an empty string, do not pass --priority."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem

    payload = NotificationWorkitemPayload(
        project="2165583",
        category="bug",
        type="",
        title="t",
        body="b",
        priority="",
    )
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    assert "--priority" not in captured["cmd"]


@pytest.mark.asyncio
async def test_create_workitem_passes_attachment_flags(monkeypatch) -> None:
    """When attachments is non-empty, pass each local path once as --attachment (repeatable)."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem

    payload = NotificationWorkitemPayload(
        project="2165583",
        category="bug",
        title="t",
        body="b",
        attachments=["/tmp/reports/cluster_report.md", "/tmp/reports/eval_summary.xlsx"],
    )
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    cmd = captured["cmd"]
    indices = [i for i, arg in enumerate(cmd) if arg == "--attachment"]
    assert len(indices) == 2
    assert cmd[indices[0] + 1] == "/tmp/reports/cluster_report.md"
    assert cmd[indices[1] + 1] == "/tmp/reports/eval_summary.xlsx"


@pytest.mark.asyncio
async def test_create_workitem_omits_attachment_when_empty(monkeypatch) -> None:
    """When attachments is omitted (empty list), do not pass --attachment, keeping the old behavior."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem

    payload = NotificationWorkitemPayload(project="2165583", category="bug", title="t", body="b")
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    assert "--attachment" not in captured["cmd"]


@pytest.mark.asyncio
async def test_create_workitem_passes_parent_relation(monkeypatch) -> None:
    """When parent_workitem_id is non-empty, pass --relation parent:<id> so the child ticket hangs under the parent."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem

    payload = NotificationWorkitemPayload(
        project="2165583",
        category="req",
        title="t",
        body="b",
        parent_workitem_id="90000001",
    )
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    cmd = captured["cmd"]
    idx = cmd.index("--relation")
    assert cmd[idx + 1] == "parent:90000001"


@pytest.mark.asyncio
async def test_create_workitem_omits_relation_without_parent(monkeypatch) -> None:
    """When parent_workitem_id is omitted (empty), do not pass --relation, keeping the old sibling-ticket behavior."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem

    payload = NotificationWorkitemPayload(project="2165583", category="req", title="t", body="b")
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    assert "--relation" not in captured["cmd"]


@pytest.mark.asyncio
async def test_create_workitem_body_contains_project_url(monkeypatch) -> None:
    """project_url is embedded in the body via build_workitem_payload; create_workitem does not take it separately."""
    captured = install_subprocess_capture(monkeypatch)
    from benchmark.attribution.badcase.publisher import create_workitem

    body_with_link = "## 归因摘要\n\n### 根因\nx\n\n[关联看板](https://example.com/p/123)"
    payload = NotificationWorkitemPayload(
        project="2165583",
        category="bug",
        type="",
        title="t",
        body=body_with_link,
    )
    await create_workitem(payload, NotificationConfig(enabled=True, project="2165583"))

    cmd = captured["cmd"]
    body_idx = cmd.index("--body")
    assert "https://example.com/p/123" in cmd[body_idx + 1]
    assert "[关联看板]" in cmd[body_idx + 1]


# ---------------------------------------------------------------------------
# Real-path end to end: create_aone_task(resolved_route) expands internally → payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_notification_task_with_resolved_route_expands_to_payload(monkeypatch) -> None:
    """create_aone_task real path: OwnerMapping.assigned_to / priority / project_url expand into the payload."""
    from benchmark.domain.enums import CasePriority
    from benchmark.domain.owner_mapping import OwnerMapping
    from benchmark.attribution.badcase.publisher import NotificationWorkitemReference, create_notification_task

    captured: dict[str, Any] = {}

    async def _fake_create_workitem_reference(
        payload: object, _config: object
    ) -> NotificationWorkitemReference:
        captured["payload"] = payload
        return NotificationWorkitemReference(
            workitem_id="W-99999",
            url="https://aone.example/W-99999",
        )

    monkeypatch.setattr(
        "benchmark.attribution.badcase.publisher.create_workitem_reference",
        _fake_create_workitem_reference,
    )

    config = NotificationConfig(
        enabled=True,
        dry_run=False,
        project="2165583",
        category="bug",
        assignee="fallback-user",
    )
    mapping = OwnerMapping(
        mapping_id="m1",
        match_type="agent_cause",
        match_key="amap-agent/tool_error",
        assigned_to="alice",
        priority=CasePriority.P0,
        project_url="https://example.com/p/123",
        owner_group="platform-team",
        enabled=True,
    )

    result = await create_notification_task(
        notification_config=config,
        attribution_result=_result(),
        resolved_route=mapping,
    )

    # Real-path assertion: mapping.assigned_to overrides the config.assignee fallback
    assert captured["payload"].assignee == "alice"
    # Real priority mapping: P0 → urgent
    assert captured["payload"].priority == "urgent"
    # project_url is really embedded in the body
    assert captured["payload"].project_url == "https://example.com/p/123"
    assert "https://example.com/p/123" in captured["payload"].body
    assert "[关联看板]" in captured["payload"].body
    # Ticket created successfully, and the URL is returned along with it
    assert result.status == NotificationDispatchStatus.CREATED
    assert result.workitem_id == "W-99999"
    assert result.workitem_url == "https://aone.example/W-99999"


@pytest.mark.asyncio
async def test_create_notification_task_assigned_to_list_random_picks_one(monkeypatch) -> None:
    """assign_type=random: pick one person among many as the assignee, and do not add the rest as participants."""
    from benchmark.domain.owner_mapping import OwnerMapping
    from benchmark.attribution.badcase.publisher import NotificationWorkitemReference, create_notification_task

    captured: dict[str, Any] = {}

    async def _fake_create_workitem_reference(
        payload: object, _config: object
    ) -> NotificationWorkitemReference:
        captured["payload"] = payload
        return NotificationWorkitemReference(workitem_id="W-1", url="https://aone.example/W-1")

    monkeypatch.setattr(
        "benchmark.attribution.badcase.publisher.create_workitem_reference",
        _fake_create_workitem_reference,
    )
    monkeypatch.setattr(
        "benchmark.domain.owner_mapping.random.choice",
        lambda pool: pool[1],
    )

    result = await create_notification_task(
        notification_config=NotificationConfig(
            enabled=True,
            project="2165583",
            assignee="fallback-user",
            participants=["dave"],
        ),
        attribution_result=_result(),
        resolved_route=OwnerMapping(
            mapping_id="m1",
            assignees=("alice", "bob", "carol"),
        ),
    )

    assert result.status == NotificationDispatchStatus.CREATED
    assert captured["payload"].assignee == "bob"
    assert captured["payload"].participants == ["dave"]


@pytest.mark.asyncio
async def test_create_notification_task_assigned_to_list_all_ccs_rest(monkeypatch) -> None:
    """assign_type=all: the first person is the assignee, and the rest are appended as participants on the same ticket."""
    from benchmark.domain.owner_mapping import OwnerMapping
    from benchmark.attribution.badcase.publisher import NotificationWorkitemReference, create_notification_task

    captured: dict[str, Any] = {}

    async def _fake_create_workitem_reference(
        payload: object, _config: object
    ) -> NotificationWorkitemReference:
        captured["payload"] = payload
        return NotificationWorkitemReference(workitem_id="W-1", url="https://aone.example/W-1")

    monkeypatch.setattr(
        "benchmark.attribution.badcase.publisher.create_workitem_reference",
        _fake_create_workitem_reference,
    )
    monkeypatch.setattr(
        "benchmark.domain.owner_mapping.random.choice",
        lambda pool: pool[-1],
    )

    result = await create_notification_task(
        notification_config=NotificationConfig(
            enabled=True,
            project="2165583",
            assignee="fallback-user",
            participants=["dave"],
        ),
        attribution_result=_result(),
        resolved_route=OwnerMapping(
            mapping_id="m1",
            assignees=("alice", "bob", "carol"),
            assign_type="all",
        ),
    )

    assert result.status == NotificationDispatchStatus.CREATED
    assert captured["payload"].assignee == "alice"
    assert captured["payload"].participants == ["dave", "bob", "carol"]


@pytest.mark.asyncio
async def test_create_notification_task_without_resolved_route_uses_fallback(monkeypatch) -> None:
    """When resolved_route=None, the payload falls back to aone_config (empty priority / empty project_url)."""
    from benchmark.attribution.badcase.publisher import (
        NotificationWorkitemReference,
        create_notification_task,
    )

    captured: dict[str, Any] = {}

    async def _fake_create_workitem_reference(
        payload: object, _config: object
    ) -> NotificationWorkitemReference:
        captured["payload"] = payload
        return NotificationWorkitemReference(workitem_id="W-99999", url="https://aone.example/W-99999")

    monkeypatch.setattr(
        "benchmark.attribution.badcase.publisher.create_workitem_reference",
        _fake_create_workitem_reference,
    )

    config = NotificationConfig(
        enabled=True,
        dry_run=False,
        project="2165583",
        category="bug",
        assignee="fallback-user",
    )

    await create_notification_task(
        notification_config=config,
        attribution_result=_result(),
        # resolved_route is not passed
    )

    assert captured["payload"].assignee == "fallback-user"
    assert captured["payload"].priority == ""
    assert captured["payload"].project_url == ""
    assert "[关联看板]" not in captured["payload"].body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template", ["{view}", "{", "{0}", "{project_id.missing}", "{project_id[bad]}"]
)
async def test_created_workitem_survives_invalid_url_template(monkeypatch, template) -> None:
    monkeypatch.setenv("NOTIFICATION_WORKITEM_URL_TEMPLATE", template)
    run = AsyncMock(return_value='{"workitem_id": "123"}')
    monkeypatch.setattr("benchmark.attribution.badcase.publisher._run_a1_command", run)
    reference = await create_workitem_reference(
        NotificationWorkitemPayload(project="2165583", category="task", title="t", body="b"),
        _enabled_config(),
    )
    assert reference.workitem_id == "123"
    assert reference.url == (
        "https://project.aone.alibaba-inc.com/v2/project/2165583/task#openWorkitemIdentifier=123"
    )
    run.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_stays_created_when_url_template_is_invalid(monkeypatch) -> None:
    monkeypatch.setenv("NOTIFICATION_WORKITEM_URL_TEMPLATE", "{view}")
    run = AsyncMock(return_value='{"workitem_id": "123"}')
    monkeypatch.setattr("benchmark.attribution.badcase.publisher._run_a1_command", run)
    result = await create_notification_task(notification_config=_enabled_config(), attribution_result=_result())
    assert result.status == NotificationDispatchStatus.CREATED
    assert result.workitem_id == "123"
    assert "openWorkitemIdentifier=123" in result.workitem_url
    run.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "project,env_project", [("configured", "fallback"), ("", "fallback"), ("", "")]
)
async def test_workitem_url_project_config_precedence(monkeypatch, project, env_project) -> None:
    monkeypatch.delenv("NOTIFICATION_WORKITEM_URL_TEMPLATE", raising=False)
    monkeypatch.setenv("NOTIFICATION_PROJECT_ID", env_project)
    run = AsyncMock(return_value='{"workitem_id": "123"}')
    monkeypatch.setattr("benchmark.attribution.badcase.publisher._run_a1_command", run)
    reference = await create_workitem_reference(
        NotificationWorkitemPayload(project=project, category="task", title="t", body="b"),
        NotificationConfig(project=project),
    )
    assert reference.workitem_id == "123"
    effective = project or env_project
    assert reference.url == (
        f"https://project.aone.alibaba-inc.com/v2/project/{effective}/task#openWorkitemIdentifier=123"
        if effective
        else ""
    )
