"""Registration-page evaluationCount: per-chain definition (batch + streaming) and example version ownership."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from benchmark.records.enums import TraceStatus
from benchmark.records.models.runtime_config_record import RuntimeConfigRecord
from benchmark.records.models.trace_record import TraceRecord
from benchmark.domain.benchmark_console import (
    AgentRegistrationMatch,
    content_version_from_bucket,
    evaluator_name_candidates,
    fill_agent_evaluation_counts,
    registration_content_versions,
    row_matches_agent_registration,
)
from benchmark.runs.run_service import BenchmarkService


def test_evaluator_name_candidates_strips_agent_and_benchmark_suffix() -> None:
    assert evaluator_name_candidates("example_agent") == frozenset({"example_agent", "example"})
    assert evaluator_name_candidates("example-agent") == frozenset({"example-agent", "example"})
    assert evaluator_name_candidates("example_benchmark") == frozenset(
        {"example_benchmark", "example"}
    )
    assert evaluator_name_candidates("example") == frozenset({"example"})
    assert evaluator_name_candidates("  ") == frozenset()


def test_content_version_from_judge_extra_prompt_customization() -> None:
    bucket = {
        "score": "9",
        "extra": {"prompt_customization": {"content_version": 56, "stage": "judge"}},
    }
    assert (
        content_version_from_bucket(
            bucket,
            (
                ("extra", "prompt_customization", "content_version"),
                ("content_version",),
            ),
        )
        == "56"
    )


def test_content_version_from_attribution_prompt_bucket() -> None:
    bucket = {"prompt_customization": {"content_version": "3"}, "confidence": 0.9}
    assert (
        content_version_from_bucket(
            bucket,
            (
                ("prompt_customization", "content_version"),
                ("content_version",),
            ),
        )
        == "3"
    )


def test_registration_content_versions_reads_camel_case() -> None:
    item = {
        "customization": {
            "judge": {"contentVersion": 56},
            "attribution": {"contentVersion": 7},
        }
    }
    assert registration_content_versions(item) == ("56", "7")
    assert registration_content_versions({"name": "example"}) == (None, None)


def test_row_matches_prefers_judge_version() -> None:
    candidates = frozenset({"example"})
    assert row_matches_agent_registration(
        AgentRegistrationMatch(
            evaluator_name="example",
            judge_cv="56",
            attr_cv="7",
            candidates=candidates,
            want_judge_cv="56",
            want_attr_cv="7",
        ),
    )
    assert not row_matches_agent_registration(
        AgentRegistrationMatch(
            evaluator_name="example",
            judge_cv="55",
            attr_cv="7",
            candidates=candidates,
            want_judge_cv="56",
            want_attr_cv="7",
        ),
    )


def test_row_matches_legacy_registration_only_unstamped_rows() -> None:
    candidates = frozenset({"example"})
    assert row_matches_agent_registration(
        AgentRegistrationMatch(
            evaluator_name="example",
            judge_cv=None,
            attr_cv=None,
            candidates=candidates,
            want_judge_cv=None,
            want_attr_cv=None,
        ),
    )
    assert not row_matches_agent_registration(
        AgentRegistrationMatch(
            evaluator_name="example",
            judge_cv="56",
            attr_cv=None,
            candidates=candidates,
            want_judge_cv=None,
            want_attr_cv=None,
        ),
    )


@pytest.mark.asyncio
async def test_fill_agent_evaluation_counts_applies_run_counts_split() -> None:
    """agents API cards: the total and the batch/streaming components are both returned."""
    items: list[dict[str, Any]] = [
        {
            "id": "reg-v56",
            "name": "example_benchmark",
            "evaluationCount": 0,
            "batchCount": 0,
            "streamingCount": 0,
        },
    ]
    await fill_agent_evaluation_counts(
        items,
        [
            {
                "benchmarkName": "example_benchmark",
                "registrationId": "reg-v56",
                "evaluationCount": 35,
                "batchCount": 0,
                "streamingCount": 35,
            }
        ],
    )
    assert items[0]["evaluationCount"] == 35
    assert items[0]["batchCount"] == 0
    assert items[0]["streamingCount"] == 35


@pytest.mark.asyncio
async def test_run_counts_streaming_by_registration_id_column() -> None:
    """Newly stored evaluator_name=stem@registrationId; bare registrationId rows from before are still accepted."""
    await TraceRecord.all().delete()
    await RuntimeConfigRecord.all().delete()
    anchor = datetime(2026, 10, 20, tzinfo=UTC)
    for registration_id in ("reg-v56", "reg-v57"):
        await RuntimeConfigRecord.create(
            kind="benchmark_agent",
            record_key=f"benchmark_agent:{registration_id}",
            payload={
                "id": registration_id,
                "name": "example_benchmark",
            },
            scope_key=f"agent:{registration_id}",
        )
        await RuntimeConfigRecord.filter(record_key=f"benchmark_agent:{registration_id}").update(
            created_at=anchor
        )

    for index in range(2):
        await TraceRecord.create(
            trace_id=f"t-reg-v56-{index}",
            project_id="proj",
            agent_name="amap-agent",
            evaluator_name="example@reg-v56",
            status=TraceStatus.SCORED,
            judge_json={"score": "9"},
        )
    # A bare registrationId is still accepted (rows stored by the previous version)
    await TraceRecord.create(
        trace_id="t-reg-v57",
        project_id="proj",
        agent_name="amap-agent",
        evaluator_name="reg-v57",
        status=TraceStatus.SCORED,
        judge_json={"score": "8"},
    )

    entries = {
        (item["benchmarkName"], item.get("registrationId")): item
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert entries[("example_benchmark", "reg-v56")]["streamingCount"] == 2
    assert entries[("example_benchmark", "reg-v57")]["streamingCount"] == 1
    assert ("example", None) not in entries


@pytest.mark.asyncio
async def test_run_counts_example_streaming_by_content_version() -> None:
    """The example realtime stream is attributed to the matching registrationId by the judge content_version."""
    await TraceRecord.all().delete()
    await RuntimeConfigRecord.all().delete()
    anchor = datetime(2026, 10, 20, tzinfo=UTC)
    for registration_id, version in (("reg-v56", 56), ("reg-v57", 57)):
        await RuntimeConfigRecord.create(
            kind="benchmark_agent",
            record_key=f"benchmark_agent:{registration_id}",
            payload={
                "id": registration_id,
                "name": "example_benchmark",
                "customization": {
                    "judge": {
                        "stage": "judge",
                        "contentVersion": version,
                        "prompt": {
                            "business": "example",
                            "variantKey": "default",
                            "contentVersion": version,
                            "gitCommit": "abc",
                        },
                    }
                },
            },
            scope_key=f"agent:{registration_id}",
        )
        await RuntimeConfigRecord.filter(record_key=f"benchmark_agent:{registration_id}").update(
            created_at=anchor
        )

    for index in range(3):
        await TraceRecord.create(
            trace_id=f"t-v56-{index}",
            project_id="proj",
            agent_name="amap-agent",
            evaluator_name="example",
            status=TraceStatus.SCORED,
            judge_json={
                "score": "9",
                "extra": {"prompt_customization": {"content_version": 56}},
            },
        )
    await TraceRecord.create(
        trace_id="t-v57",
        project_id="proj",
        agent_name="amap-agent",
        evaluator_name="example",
        status=TraceStatus.SCORED,
        judge_json={
            "score": "8",
            "extra": {"prompt_customization": {"content_version": 57}},
        },
    )
    # A misspelled execution-side name does not count
    await TraceRecord.create(
        trace_id="t-bad-name",
        project_id="proj",
        agent_name="amap-agent",
        evaluator_name="example_benchmark",
        status=TraceStatus.SCORED,
        judge_json={
            "score": "7",
            "extra": {"prompt_customization": {"content_version": 56}},
        },
    )

    entries = {
        (item["benchmarkName"], item.get("registrationId")): item
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    v56 = entries[("example_benchmark", "reg-v56")]
    v57 = entries[("example_benchmark", "reg-v57")]
    assert v56["evaluationCount"] == 3
    assert v56["batchCount"] == 0
    assert v56["streamingCount"] == 3
    assert v57["evaluationCount"] == 1
    assert v57["streamingCount"] == 1
    # The example stem is not shared by name
    assert ("example", None) not in entries
    assert ("example_benchmark", None) not in entries


@pytest.mark.asyncio
async def test_run_counts_example_stem_uses_trace_meta_registration() -> None:
    """An old stem-name row is attributed to the matching version by trace_meta.agent_registration_id first."""
    await TraceRecord.all().delete()
    await RuntimeConfigRecord.all().delete()
    anchor = datetime(2026, 10, 20, tzinfo=UTC)
    for registration_id, version in (("reg-v56", 56), ("reg-v57", 57)):
        await RuntimeConfigRecord.create(
            kind="benchmark_agent",
            record_key=f"benchmark_agent:{registration_id}",
            payload={
                "id": registration_id,
                "name": "example_benchmark",
                "customization": {
                    "judge": {
                        "stage": "judge",
                        "contentVersion": version,
                        "prompt": {
                            "business": "example",
                            "variantKey": "default",
                            "contentVersion": version,
                            "gitCommit": "abc",
                        },
                    }
                },
            },
            scope_key=f"agent:{registration_id}",
        )
        await RuntimeConfigRecord.filter(record_key=f"benchmark_agent:{registration_id}").update(
            created_at=anchor
        )

    # No content_version, but meta carries a registration → it should still land in v56
    await TraceRecord.create(
        trace_id="t-meta-v56",
        project_id="proj",
        agent_name="amap-agent",
        evaluator_name="example",
        status=TraceStatus.SCORED,
        judge_json={"score": "9"},
        trace_meta_json={"source": "kafka", "agent_registration_id": "reg-v56"},
    )
    await TraceRecord.create(
        trace_id="t-meta-v57",
        project_id="proj",
        agent_name="amap-agent",
        evaluator_name="example",
        status=TraceStatus.SCORED,
        judge_json={"score": "8"},
        trace_meta_json={"source": "kafka", "agent_registration_id": "reg-v57"},
    )

    entries = {
        (item["benchmarkName"], item.get("registrationId")): item
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert entries[("example_benchmark", "reg-v56")]["streamingCount"] == 1
    assert entries[("example_benchmark", "reg-v57")]["streamingCount"] == 1


@pytest.mark.asyncio
async def test_run_counts_example_unversioned_ambiguous_not_shared() -> None:
    """When several uncustomized example registrations exist, old rows with no version stamp are not counted — so versions do not share one number."""
    await TraceRecord.all().delete()
    await RuntimeConfigRecord.all().delete()
    anchor = datetime(2026, 10, 20, tzinfo=UTC)
    for registration_id in ("reg-a", "reg-b"):
        await RuntimeConfigRecord.create(
            kind="benchmark_agent",
            record_key=f"benchmark_agent:{registration_id}",
            payload={"id": registration_id, "name": "example_benchmark"},
            scope_key=f"agent:{registration_id}",
        )
        await RuntimeConfigRecord.filter(record_key=f"benchmark_agent:{registration_id}").update(
            created_at=anchor
        )

    for index in range(3):
        await TraceRecord.create(
            trace_id=f"t-ambig-{index}",
            project_id="proj",
            agent_name="amap-agent",
            evaluator_name="example",
            status=TraceStatus.SCORED,
            judge_json={"score": "9"},
        )

    entries = {
        (item["benchmarkName"], item.get("registrationId")): item.get("streamingCount")
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert entries.get(("example_benchmark", "reg-a"), 0) == 0
    assert entries.get(("example_benchmark", "reg-b"), 0) == 0
    assert ("example", None) not in entries


@pytest.mark.asyncio
async def test_run_counts_emits_batch_and_streaming_split_for_shared_stem() -> None:
    """A shared-stem realtime stream emits streamingCount; the component fields are always present."""
    await TraceRecord.all().delete()
    for index in range(2):
        await TraceRecord.create(
            trace_id=f"amap-split-{index}",
            evaluator_name="amap_agent",
            agent_name="amap-agent",
            status=TraceStatus.SCORED,
        )
    counts = {
        item["benchmarkName"]: item for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert counts["amap_agent"]["evaluationCount"] == 2
    assert counts["amap_agent"]["batchCount"] == 0
    assert counts["amap_agent"]["streamingCount"] == 2
    assert counts["amap_agent_benchmark"]["streamingCount"] == 2
