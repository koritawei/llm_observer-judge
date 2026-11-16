"""Envelope columns of runtime_config_records: ownership key, soft delete, and revive."""

from benchmark.records.models.runtime_config_record import (
    KIND_EVALUATOR,
    SCHEMA_VERSION,
    SCOPE_GLOBAL,
    RuntimeConfigRecord,
    evaluator_record_key,
    project_scope_key,
)
from benchmark.domain.runtime_config_store import (
    refresh_eval_config_snapshot,
    replace_eval_config_file,
    restore_runtime_config,
    retire_runtime_config,
    retire_sibling_evaluator_entries,
    upsert_runtime_config,
)
from benchmark.domain.schemas import EvalConfigFile


def test_project_scope_key_uses_prefix() -> None:
    assert project_scope_key(" proj ") == "project:proj"


def test_empty_project_is_global() -> None:
    assert project_scope_key("  ") == SCOPE_GLOBAL


async def test_retire_hides_row_and_upsert_revives_same_key() -> None:
    record_key = evaluator_record_key("proj", "example", "")
    payload = {"evaluatorName": "example", "sampling": 1.0}
    try:
        await upsert_runtime_config(
            kind=KIND_EVALUATOR,
            record_key=record_key,
            payload=payload,
            scope_key=project_scope_key("proj"),
        )
        loaded = await refresh_eval_config_snapshot()
        assert list(loaded.configs) == ["proj"]

        assert await retire_runtime_config(KIND_EVALUATOR, record_key) is True
        retired_row = await RuntimeConfigRecord.filter(record_key=record_key).first()
        assert retired_row is not None
        assert retired_row.is_deleted is True
        assert retired_row.deleted_at is not None
        hidden = await refresh_eval_config_snapshot()
        assert hidden.configs == {}

        await upsert_runtime_config(
            kind=KIND_EVALUATOR,
            record_key=record_key,
            payload=payload,
            scope_key=project_scope_key("proj"),
        )
        rows = await RuntimeConfigRecord.filter(record_key=record_key)
        assert len(rows) == 1
        row = rows[0]
        assert row.deleted_at is None
        assert row.is_deleted is False
        assert row.revision == 2
        assert row.schema_version == SCHEMA_VERSION
        assert row.scope_key == "project:proj"
        revived = await refresh_eval_config_snapshot()
        assert "proj" in revived.configs
    finally:
        replace_eval_config_file(EvalConfigFile())


async def _seed_evaluator(project: str, evaluator: str, agent: str) -> str:
    key = evaluator_record_key(project, evaluator, agent)
    await upsert_runtime_config(
        kind=KIND_EVALUATOR,
        record_key=key,
        payload={"evaluatorName": evaluator, "sampling": 0.1, **({"agentName": agent} if agent else {})},
        scope_key=project_scope_key(project),
    )
    return key


async def test_retire_sibling_evaluator_entries_replaces_variants() -> None:
    """No registration slot: other unregistered agent_name variants of the same (project, evaluator) are cleared.

    Multi-prompt slots that carry a registration are not harmed by an unregistered activation.
    """
    try:
        legacy_empty = await _seed_evaluator("proj", "amap_agent", "")
        keep_key = await _seed_evaluator("proj", "amap_agent", "amap-agent")
        other_agent = await _seed_evaluator("proj", "amap_agent", "vui_agent")
        other_evaluator = await _seed_evaluator("proj", "example", "")
        other_project = await _seed_evaluator("proj2", "amap_agent", "")
        versioned = evaluator_record_key("proj", "example", "amap-agent", "example--v56")
        await upsert_runtime_config(
            kind=KIND_EVALUATOR,
            record_key=versioned,
            payload={
                "evaluatorName": "example",
                "sampling": 0.1,
                "agentName": "amap-agent",
                "agentRegistrationId": "example--v56",
            },
            scope_key=project_scope_key("proj"),
        )

        retired = await retire_sibling_evaluator_entries("proj", "amap_agent", "amap-agent")
        assert retired == 2  # empty string + vui_agent

        async def alive(key: str) -> bool:
            row = await RuntimeConfigRecord.filter(record_key=key, deleted_at__isnull=True).first()
            return row is not None

        assert not await alive(legacy_empty)
        assert not await alive(other_agent)
        assert await alive(keep_key)
        assert await alive(other_evaluator)
        assert await alive(other_project)
        assert await alive(versioned)

        snapshot = await refresh_eval_config_snapshot()
        amap_entries = [
            e for e in snapshot.configs["proj"] if e.evaluator_name == "amap_agent"
        ]
        assert len(amap_entries) == 1
        assert amap_entries[0].agent_name == "amap-agent"

        assert await retire_sibling_evaluator_entries("proj", "amap_agent", "amap-agent") == 0
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_registration_slots_coexist_and_retire_is_scoped() -> None:
    """example_v56 and v57 coexist for the same project and stem; retire clears only this registration slot."""
    try:
        key_v56 = evaluator_record_key("proj", "example", "amap-agent", "reg-v56")
        key_v57 = evaluator_record_key("proj", "example", "amap-agent", "reg-v57")
        for key, reg, internal in (
            (key_v56, "reg-v56", "agx_example_0000000056"),
            (key_v57, "reg-v57", "agx_example_0000000057"),
        ):
            await upsert_runtime_config(
                kind=KIND_EVALUATOR,
                record_key=key,
                payload={
                    "evaluatorName": "example",
                    "sampling": 0.5,
                    "agentName": "amap-agent",
                    "agentRegistrationId": reg,
                    "internalAgentName": internal,
                },
                scope_key=project_scope_key("proj"),
            )

        # Reactivating v56 (same agent) must not delete v57
        retired = await retire_sibling_evaluator_entries(
            "proj",
            "example",
            "amap-agent",
            keep_registration_id="reg-v56",
        )
        assert retired == 0

        snapshot = await refresh_eval_config_snapshot()
        examples = [e for e in snapshot.configs["proj"] if e.evaluator_name == "example"]
        assert len(examples) == 2
        regs = {e.agent_registration_id for e in examples}
        assert regs == {"reg-v56", "reg-v57"}

        # Same registration, different agentName variant: clear only that registration slot
        dirty = evaluator_record_key("proj", "example", "", "reg-v56")
        await upsert_runtime_config(
            kind=KIND_EVALUATOR,
            record_key=dirty,
            payload={
                "evaluatorName": "example",
                "sampling": 0.5,
                "agentRegistrationId": "reg-v56",
            },
            scope_key=project_scope_key("proj"),
        )
        retired = await retire_sibling_evaluator_entries(
            "proj",
            "example",
            "amap-agent",
            keep_registration_id="reg-v56",
        )
        assert retired == 1
        row = await RuntimeConfigRecord.filter(
            record_key=dirty, deleted_at__isnull=True
        ).first()
        assert row is None
        assert (
            await RuntimeConfigRecord.filter(
                record_key=key_v57, deleted_at__isnull=True
            ).first()
            is not None
        )
    finally:
        replace_eval_config_file(EvalConfigFile())


def test_evaluator_record_key_roundtrip_with_registration() -> None:
    from benchmark.records.models.runtime_config_record import split_evaluator_record_key

    key = evaluator_record_key("proj", "example", "amap-agent", "example--biz--v--v56")
    project, evaluator, agent, reg = split_evaluator_record_key(key)
    assert (project, evaluator, agent, reg) == (
        "proj",
        "example",
        "amap-agent",
        "example--biz--v--v56",
    )
    legacy = evaluator_record_key("proj", "example", "")
    p, e, a, r = split_evaluator_record_key(legacy)
    assert (p, e, a, r) == ("proj", "example", "", "")


async def test_restore_runtime_config_revives_retired_row() -> None:
    """restore is symmetric with retire: revive a soft-deleted row, idempotent, and safe for a missing key."""
    key = await _seed_evaluator("proj", "example", "")
    try:
        assert await retire_runtime_config(KIND_EVALUATOR, key) is True
        assert await restore_runtime_config(KIND_EVALUATOR, key) is True
        row = await RuntimeConfigRecord.filter(record_key=key).first()
        assert row is not None
        assert row.deleted_at is None
        assert row.is_deleted is False
        # Already active: return False idempotently
        assert await restore_runtime_config(KIND_EVALUATOR, key) is False
        # A key that was never registered: return False safely
        assert await restore_runtime_config(KIND_EVALUATOR, "evaluator:ghost:none:") is False
        revived = await refresh_eval_config_snapshot()
        assert "proj" in revived.configs
    finally:
        replace_eval_config_file(EvalConfigFile())
