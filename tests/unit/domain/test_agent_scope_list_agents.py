"""L1: list_agents filtered by evaluation target (the main seam)."""

import pytest
from benchmark.records.models.runtime_config_record import (
    KIND_BENCHMARK_AGENT,
    SCOPE_GLOBAL,
    benchmark_agent_record_key,
)
from benchmark.domain.benchmark_console import (
    AgentDocument,
    ConsoleError,
    list_agents,
    list_all_agent_registrations,
    save_agent,
)
from benchmark.domain.runtime_config_store import upsert_runtime_config
from pydantic import ValidationError


def _plain(doc_id: str, name: str, *, agent_scope: str) -> AgentDocument:
    return AgentDocument.model_validate(
        {
            "id": doc_id,
            "name": name,
            "owner": "eval-team",
            "registeredAt": "2026-11-02 10:00:00",
            "agentScope": agent_scope,
        }
    )


def _customized_without_scope_payload() -> dict[str, object]:
    """Shape of dirty pre-env data: customization is enabled but agentScope is missing."""
    return {
        "id": "example_benchmark--biz--v1--v1",
        "name": "example_benchmark",
        "owner": "eval-team",
        "registeredAt": "2026-10-12 10:00:00",
        "business": "biz",
        "variantKey": "v1",
        "generateAttributionReport": False,
        "customization": {
            "judge": {
                "stage": "judge",
                "contentVersion": 1,
                "gitBranch": "master",
                "gitPath": "benchmark-customizations/example/biz/v1/judge/v1",
                "gitCommit": "deadbeef",
                "bundleSha256": "a" * 64,
                "prompt": {
                    "slots": ["prompt"],
                    "files": [{"path": "prompt/prompts/prompt.md", "sha256": "b" * 64, "bytes": 1}],
                },
                "createdAt": "2026-10-12 10:00:00",
            }
        },
    }


@pytest.mark.asyncio
async def test_list_agents_requires_agent_scope() -> None:
    with pytest.raises(ConsoleError) as exc_info:
        await list_agents(None)
    assert exc_info.value.status_code == 400
    with pytest.raises(ConsoleError) as exc_info:
        await list_agents("  ")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_list_agents_filters_by_amap_whitelist() -> None:
    await save_agent(_plain("ba-ex", "example_benchmark", agent_scope="amap-agent"))
    await save_agent(_plain("ba-search", "search_eval", agent_scope="amap-agent"))
    await upsert_runtime_config(
        kind=KIND_BENCHMARK_AGENT,
        record_key=benchmark_agent_record_key("ba-stray"),
        payload=_plain("ba-stray", "some_other_agent", agent_scope="amap-agent").model_dump(
            by_alias=True,
            exclude_none=True,
        ),
        scope_key=SCOPE_GLOBAL,
    )

    amap_names = {item["name"] for item in await list_agents("amap-agent")}
    assert "example_benchmark" in amap_names
    assert "search_eval" in amap_names
    assert "some_other_agent" not in amap_names

    # Not registered under vehicle → the completed list is empty (example is only globally selectable in the dropdown catalog)
    vehicle_names = {item["name"] for item in await list_agents("vehicle-agent")}
    assert vehicle_names == set()


@pytest.mark.asyncio
async def test_list_agents_empty_when_scope_has_no_registration() -> None:
    await save_agent(_plain("ba-ex-amap", "example_benchmark", agent_scope="amap-agent"))
    assert await list_agents("vehicle-agent") == []
    assert await list_agents("vui-agent") == []


@pytest.mark.asyncio
async def test_list_agents_includes_unbound_legacy_under_amap() -> None:
    await upsert_runtime_config(
        kind=KIND_BENCHMARK_AGENT,
        record_key=benchmark_agent_record_key("ba-legacy"),
        payload={
            "id": "ba-legacy",
            "name": "example_benchmark",
            "owner": "eval-team",
            "registeredAt": "2026-10-01 10:00:00",
        },
        scope_key=SCOPE_GLOBAL,
    )
    amap_ids = {item["id"] for item in await list_agents("amap-agent")}
    assert "ba-legacy" in amap_ids
    assert await list_agents("vehicle-agent") == []


def test_customized_agent_without_scope_still_rejected_on_write() -> None:
    """The write path stays strict: a customized registration missing agentScope must not pass model_validate."""
    with pytest.raises(ValidationError, match="agentScope"):
        AgentDocument.model_validate(_customized_without_scope_payload())


@pytest.mark.asyncio
async def test_list_agents_skips_invalid_customized_without_scope() -> None:
    """Dirty pre-env data: one customized registration missing agentScope must not take down the agents list on the datasets page."""
    await save_agent(_plain("ba-ok", "example_benchmark", agent_scope="amap-agent"))
    await upsert_runtime_config(
        kind=KIND_BENCHMARK_AGENT,
        record_key=benchmark_agent_record_key("example_benchmark--biz--v1--v1"),
        payload=_customized_without_scope_payload(),
        scope_key=SCOPE_GLOBAL,
    )

    items = await list_agents("amap-agent")
    ids = {item["id"] for item in items}
    assert "ba-ok" in ids
    assert "example_benchmark--biz--v1--v1" not in ids

    overview_ids = {item["id"] for item in await list_all_agent_registrations()}
    assert "ba-ok" in overview_ids
    assert "example_benchmark--biz--v1--v1" not in overview_ids
