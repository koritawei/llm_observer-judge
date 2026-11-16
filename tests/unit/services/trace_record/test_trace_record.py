"""Unit tests for TraceRecordService against real SQLite + PostgreSQL."""

import json
from datetime import UTC, datetime

import pytest
from benchmark.records.enums import TriageLevel
from benchmark.records.models import TraceRecord
from benchmark.services import TraceRecordService
from benchmark.services.trace_record_service import (
    NotificationTaskEnqueue,
    AttributionUpdate,
    DispatchUpdate,
    TraceRecordKey,
)

# ── fixtures ──


@pytest.fixture()
def repo() -> TraceRecordService:
    return TraceRecordService()


async def _seed_record(
    repo: TraceRecordService,
    trace_id: str = "t-1",
    agent_name: str = "agent-a",
    project_id: str = "test-project",
    evaluator_name: str = "default",
    status: str = "new",
    **kwargs,
) -> TraceRecord:
    return await repo.upsert(
        TraceRecordKey(
            trace_id=trace_id,
            project_id=project_id,
            agent_name=agent_name,
            evaluator_name=evaluator_name,
        ),
        status=status,
        **kwargs,
    )


_NON_EXISTENT_UUID = "00000000-0000-0000-0000-000000000000"


# ── 1. upsert ──


class TestUpsert:
    async def test_insert_new_record(self, repo: TraceRecordService) -> None:
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-100",
                project_id="test-project",
                agent_name="agent-x",
                evaluator_name="default",
            ),
            status="scored",
        )

        assert isinstance(rec, TraceRecord)
        assert rec.trace_id == "t-100"
        assert rec.agent_name == "agent-x"
        assert rec.status == "scored"
        assert rec.created_at.tzinfo is not None

    async def test_update_existing_record(self, repo: TraceRecordService) -> None:
        original = await repo.upsert(
            TraceRecordKey(
                trace_id="t-200",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="new",
        )
        original_created = original.created_at

        updated = await repo.upsert(
            TraceRecordKey(
                trace_id="t-200",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="scored",
            judge_json={"score": "0.85", "reason": "looks good"},
        )

        assert updated.status == "scored"
        assert updated.judge_json["score"] == "0.85"
        assert updated.judge_json["reason"] == "looks good"
        assert updated.created_at == original_created
        assert updated.updated_at >= original.updated_at

    async def test_trace_meta_json_merges_on_reupsert(self, repo: TraceRecordService) -> None:
        """When trace_meta_json is passed in, shallow-merge it, and keep the existing digest / source / dataset."""
        await repo.upsert(
            TraceRecordKey(
                trace_id="t-meta-merge",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="scored",
            trace_meta_json={
                "source": "kafka",
                "digest": {"trace_id": "lf-1"},
                "fetched_at": "2026-01-01T00:00:00Z",
            },
        )
        updated = await repo.upsert(
            TraceRecordKey(
                trace_id="t-meta-merge",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="scored",
            trace_meta_json={
                "source": "dataset-nav/hard-r1",
                "dataset": {"name": "nav/hard", "item_id": "i1"},
            },
        )
        meta = updated.trace_meta_json or {}
        assert meta["source"] == "dataset-nav/hard-r1"
        assert meta["dataset"] == {"name": "nav/hard", "item_id": "i1"}
        assert meta["digest"] == {"trace_id": "lf-1"}
        assert meta["fetched_at"] == "2026-01-01T00:00:00Z"
        assert "updated_at" in meta

    async def test_trace_meta_json_preserved_when_omitted_on_reupsert(
        self,
        repo: TraceRecordService,
    ) -> None:
        """Reinserting without trace_meta_json does not clear the whole bucket."""
        await repo.upsert(
            TraceRecordKey(
                trace_id="t-meta-keep",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="scored",
            trace_meta_json={"source": "http"},
        )
        updated = await repo.upsert(
            TraceRecordKey(
                trace_id="t-meta-keep",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="scored",
            judge_json={"score": "1.0", "reason": "ok"},
        )
        assert updated.trace_meta_json is not None
        assert updated.trace_meta_json["source"] == "http"

    async def test_upsert_with_any_agent_name(self, repo: TraceRecordService) -> None:
        # agent_name is a plain string; any value is accepted without registration.
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-bad",
                project_id="p-auto",
                agent_name="unregistered-agent",
                evaluator_name="default",
            ),
            status="new",
        )
        assert rec.agent_name == "unregistered-agent"
        # idempotent: a second upsert with the same agent_name works fine
        again = await repo.upsert(
            TraceRecordKey(
                trace_id="t-bad2",
                project_id="p-auto",
                agent_name="unregistered-agent",
                evaluator_name="default",
            ),
            status="new",
        )
        assert again.agent_name == "unregistered-agent"

    async def test_upsert_clips_long_root_cause_and_scene(
        self,
        repo: TraceRecordService,
    ) -> None:
        """When an optional upsert scalar is too long, truncate it, so create/save does not fail on max_length."""
        long_cause = "c" * 300
        long_scene = "s" * 100
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-clip-upsert",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="triaged",
            root_cause=long_cause,
            scene=long_scene,
        )
        assert rec.root_cause == long_cause[:256]
        assert rec.scene == long_scene[:64]


# ── 1c. deterministic PK ──


class TestScalarSync:
    """Verify that ``TraceRecordObserver.on_save`` syncs the notification_job_status scalar.

    triage_level / root_cause are already written directly by the service layer (update_triage_level /
    update_attribution) and are no longer synced from the bucket — the observer has only notification_job_status left.
    """

    async def test_notification_job_json_syncs_notification_job_status(self, repo: TraceRecordService) -> None:
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-sync-3",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="dispatched",
            notification_job_json={"notification_job_status": "pending", "notification_job_id": "job-1"},
        )
        assert rec.notification_job_status == "pending"

    async def test_long_notification_job_status_clipped_on_sync(
        self,
        repo: TraceRecordService,
    ) -> None:
        """An over-long notification_job_status in the bucket is truncated to 32 when synced to the scalar."""
        long_status = "pending-" + ("z" * 40)
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-sync-long-status",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="dispatched",
            notification_job_json={"notification_job_status": long_status, "notification_job_id": "job-long"},
        )
        assert rec.notification_job_status == long_status[:32]
        # The bucket still keeps the original text
        assert rec.notification_job_json["notification_job_status"] == long_status

    async def test_no_bucket_leaves_scalar_null(self, repo: TraceRecordService) -> None:
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-sync-5",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="new",
        )
        assert rec.triage_level is None
        assert rec.root_cause is None
        assert rec.notification_job_status is None


# ── 2. get_by_X ──


async def test_get_by_trace_id_found(repo: TraceRecordService) -> None:
    await _seed_record(repo, trace_id="t-find")
    result = await repo.get_by_trace_id(
        TraceRecordKey(
            trace_id="t-find",
            project_id="test-project",
            agent_name="agent-a",
            evaluator_name="default",
        ),
    )
    assert result is not None
    assert result.trace_id == "t-find"


async def test_get_by_trace_id_not_found(repo: TraceRecordService) -> None:
    assert (
        await repo.get_by_trace_id(
            TraceRecordKey(
                trace_id="nonexistent",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
        )
        is None
    )


# ── 3. list_by_status ──


class TestListByStatus:
    async def test_returns_matching_records(self, repo: TraceRecordService) -> None:
        await _seed_record(repo, trace_id="t-s1", status="scored")
        await _seed_record(repo, trace_id="t-s2", status="scored")
        await _seed_record(repo, trace_id="t-s3", status="new")

        records, total = await repo.list_by_status("scored")
        assert total == 2
        assert len(records) == 2
        assert all(r.status == "scored" for r in records)

    async def test_pagination(self, repo: TraceRecordService) -> None:
        for i in range(5):
            await _seed_record(repo, trace_id=f"t-pg-{i}", status="scored")

        page1, total = await repo.list_by_status("scored", page=1, page_size=2)
        page2, _ = await repo.list_by_status("scored", page=2, page_size=2)
        page3, _ = await repo.list_by_status("scored", page=3, page_size=2)

        assert total == 5
        assert len(page1) == 2
        assert len(page2) == 2
        assert len(page3) == 1

    async def test_empty_result(self, repo: TraceRecordService) -> None:
        records, total = await repo.list_by_status("nonexistent_status")
        assert total == 0
        assert records == []


# ── 3b. list_dataset_records ──


class TestListDatasetRecords:
    async def test_filters_dataset_rows_only(self, repo: TraceRecordService) -> None:
        await _seed_record(
            repo,
            trace_id="dataset:item-1",
            status="scored",
            trace_meta_json={
                "source": "dataset-nav/hard-r1",
                "dataset": {"name": "nav/hard", "item_id": "item-1", "run_name": "r1"},
            },
        )
        await _seed_record(
            repo,
            trace_id="t-kafka",
            status="scored",
            trace_meta_json={"source": "kafka"},
        )
        # Bare "dataset" / legacy "dataset+…" must NOT match
        await _seed_record(
            repo,
            trace_id="dataset:legacy-bare",
            status="scored",
            trace_meta_json={
                "source": "dataset",
                "dataset": {"name": "old", "item_id": "legacy"},
            },
        )
        await _seed_record(
            repo,
            trace_id="dataset:legacy-plus",
            status="scored",
            trace_meta_json={
                "source": "dataset+old+r0",
                "dataset": {"name": "old", "item_id": "legacy-plus"},
            },
        )

        records, total = await repo.list_dataset_records()
        assert total == 1
        assert records[0].trace_meta_json["source"] == "dataset-nav/hard-r1"

    async def test_filter_by_dataset_name(self, repo: TraceRecordService) -> None:
        await _seed_record(
            repo,
            trace_id="dataset:a",
            status="scored",
            trace_meta_json={
                "source": "dataset-nav/hard-r1",
                "dataset": {"name": "nav/hard", "item_id": "a", "run_name": "r1"},
            },
        )
        await _seed_record(
            repo,
            trace_id="dataset:b",
            status="scored",
            trace_meta_json={
                "source": "dataset-nav/easy-r2",
                "dataset": {"name": "nav/easy", "item_id": "b", "run_name": "r2"},
            },
        )

        records, total = await repo.list_dataset_records(dataset_name="nav/hard")
        assert total == 1
        assert records[0].trace_id == "dataset:a"

    async def test_list_by_pipeline_source_http(self, repo: TraceRecordService) -> None:
        await _seed_record(
            repo,
            trace_id="t-http",
            status="scored",
            trace_meta_json={"source": "http"},
        )
        await _seed_record(
            repo,
            trace_id="t-bench",
            status="scored",
            trace_meta_json={"source": "benchmark"},
        )
        records, total = await repo.list_by_pipeline_source("http")
        assert total == 1
        assert records[0].trace_id == "t-http"

    async def test_list_by_pipeline_source_metaq_not_conflated_with_kafka(
        self,
        repo: TraceRecordService,
    ) -> None:
        await _seed_record(
            repo,
            trace_id="t-metaq",
            status="scored",
            trace_meta_json={"source": "metaq"},
        )
        await _seed_record(
            repo,
            trace_id="t-kafka",
            status="scored",
            trace_meta_json={"source": "kafka"},
        )
        records, total = await repo.list_by_pipeline_source("metaq")
        assert total == 1
        assert records[0].trace_id == "t-metaq"

    async def test_list_by_pipeline_source_sidecar_not_conflated_with_metaq(
        self, repo: TraceRecordService,
    ) -> None:
        await _seed_record(
            repo,
            trace_id="t-sidecar",
            status="scored",
            trace_meta_json={"source": "sidecar"},
        )
        await _seed_record(
            repo,
            trace_id="t-metaq",
            status="scored",
            trace_meta_json={"source": "metaq"},
        )
        records, total = await repo.list_by_pipeline_source("sidecar")
        assert total == 1
        assert records[0].trace_id == "t-sidecar"

    async def test_empty_when_no_dataset_meta(self, repo: TraceRecordService) -> None:
        await _seed_record(
            repo,
            trace_id="t-only-kafka",
            status="scored",
            trace_meta_json={"source": "kafka"},
        )
        records, total = await repo.list_dataset_records()
        assert total == 0
        assert records == []


class TestUpdateReasoning:
    async def test_updates_reasoning_json(self, repo: TraceRecordService) -> None:
        rec = await _seed_record(repo, trace_id="t-rsn")
        reasoning = json.dumps({"chain": ["step1", "step2"], "confidence": "high"})

        result = await repo.update_reasoning(
            record_id=str(rec.id),
            reasoning_json=reasoning,
        )

        assert result is not None
        assert result.attribution_detail_json["reasoning_raw"] == reasoning
        parsed = json.loads(result.attribution_detail_json["reasoning_raw"])
        assert parsed["confidence"] == "high"

    async def test_returns_none_for_missing_trace(self, repo: TraceRecordService) -> None:
        assert (
            await repo.update_reasoning(
                record_id=_NON_EXISTENT_UUID,
                reasoning_json="{}",
            )
            is None
        )


# ── 5. update_dispatch ──


class TestUpdateDispatch:
    async def test_updates_dispatch_fields(self, repo: TraceRecordService) -> None:
        rec = await _seed_record(repo, trace_id="t-dsp")
        dispatched_time = datetime(2026, 6, 5, 10, 0, 0, tzinfo=UTC)

        result = await repo.update_dispatch(
            update=DispatchUpdate(
                dispatch_ref="AONE-12345",
                dispatch_url="https://aone.example.com/task/12345",
                dispatch_assigned_to="user@example.com",
                dispatch_title="Fix hallucination in tool call",
                dispatched_at=dispatched_time,
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert result.notification_dispatch_json["notification_ref"] == "AONE-12345"
        assert result.notification_dispatch_json["notification_url"] == "https://aone.example.com/task/12345"
        assert result.notification_dispatch_json["notification_assigned_to"] == "user@example.com"
        assert result.notification_dispatch_json["notification_title"] == "Fix hallucination in tool call"
        assert "notification_dispatched_at" in result.notification_dispatch_json

    async def test_dispatch_without_datetime(self, repo: TraceRecordService) -> None:
        rec = await _seed_record(repo, trace_id="t-dsp2")
        result = await repo.update_dispatch(
            update=DispatchUpdate(
                dispatch_ref="REF-99",
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert result.notification_dispatch_json["notification_ref"] == "REF-99"
        assert "notification_dispatched_at" not in result.notification_dispatch_json

    async def test_returns_none_for_missing_trace(self, repo: TraceRecordService) -> None:
        assert (
            await repo.update_dispatch(
                update=DispatchUpdate(
                    dispatch_ref="REF-0",
                ),
                record_id=_NON_EXISTENT_UUID,
            )
            is None
        )


# ── 6. update_triage_level ──


class TestUpdateTriageLevel:
    async def test_writes_scalar_and_bucket(self, repo: TraceRecordService) -> None:
        rec = await _seed_record(repo, trace_id="t-tl", status="scored")

        result = await repo.update_triage_level(
            record_id=str(rec.id),
            triage_level="severe_bad",
        )

        assert result is not None
        assert result.triage_level == "severe_bad"
        assert result.judge_json["triage_level"] == "severe_bad"

    async def test_merges_into_existing_judge_json(self, repo: TraceRecordService) -> None:
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-tl2",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="scored",
            judge_json={"score": "0.9", "reason": "bad output"},
        )

        result = await repo.update_triage_level(
            record_id=str(rec.id),
            triage_level="mild_bad",
        )

        assert result is not None
        assert result.triage_level == "mild_bad"
        # The original score/reason must be kept, and triage_level is merged in
        assert result.judge_json["score"] == "0.9"
        assert result.judge_json["reason"] == "bad output"
        assert result.judge_json["triage_level"] == "mild_bad"

    async def test_returns_none_for_missing_trace(self, repo: TraceRecordService) -> None:
        assert (
            await repo.update_triage_level(
                record_id=_NON_EXISTENT_UUID,
                triage_level="good",
            )
            is None
        )

    async def test_rejects_unknown_triage_level(self, repo: TraceRecordService) -> None:
        """A string that is not in the TriageLevel enum should be rejected by an active check in the service layer.

        Tortoise CharEnumField does not validate that a string is legal on save (SQLite writes the raw
        string, and PG is a character column too), and only coerces it when reading from the DB → the service layer must
        actively fall back with ``TriageLevel(...)``.
        """
        rec = await _seed_record(repo, trace_id="t-tl-bad", status="scored")
        with pytest.raises(ValueError) as exc_info:
            await repo.update_triage_level(
                record_id=str(rec.id),
                triage_level="not_a_real_level",  # type: ignore[arg-type]
            )
        assert "TriageLevel" in str(exc_info.value)


# ── 7. update_attribution ──


class TestUpdateAttribution:
    async def test_writes_root_cause_scalar_and_bucket(self, repo: TraceRecordService) -> None:
        rec = await _seed_record(repo, trace_id="t-at", status="triaged")

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="hallucination",
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert result.root_cause == "hallucination"
        assert result.attribution_detail_json["root_cause"] == "hallucination"

    async def test_writes_full_bucket_fields(self, repo: TraceRecordService) -> None:
        rec = await _seed_record(repo, trace_id="t-at2", status="triaged")

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="tool_error",
                owner_domain="search",
                confidence=0.87,
                attribution_raw='{"raw": "x"}',
                evidence_raw='["ev:y"]',
                reasoning_raw="because tool failed",
                action_hint="retry with timeout",
                needs_human_review=True,
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert result.root_cause == "tool_error"
        bucket = result.attribution_detail_json
        assert bucket["root_cause"] == "tool_error"
        assert bucket["owner_domain"] == "search"
        assert bucket["confidence"] == 0.87
        assert bucket["attribution_raw"] == '{"raw": "x"}'
        assert bucket["evidence_raw"] == '["ev:y"]'
        assert bucket["reasoning_raw"] == "because tool failed"
        assert bucket["action_hint"] == "retry with timeout"
        assert bucket["needs_human_review"] is True

    async def test_persists_and_clears_candidate_root_cause(
        self, repo: TraceRecordService
    ) -> None:
        """A candidate is stored only in the bucket, and a later official attribution must clear the old candidate snapshot."""
        rec = await _seed_record(repo, trace_id="t-at-candidate", status="triaged")
        candidate = {
            "raw_code": "other:tool_schema_gap",
            "title": "工具 schema 缺口",
            "surface": "tool_data",
            "symptom": "missing_price_attribute",
            "evidence_refs": ["extra:failure_modes"],
            "agent_scope": "vehicle",
        }

        result = await repo.update_attribution(
            record_id=str(rec.id),
            update=AttributionUpdate(
                root_cause="未注册工具缺口",
                candidate_root_cause=candidate,
            ),
        )

        assert result is not None
        assert result.attribution_detail_json["candidate_root_cause"] == candidate
        updated = await repo.update_attribution(
            record_id=str(rec.id),
            update=AttributionUpdate(
                root_cause="已注册工具缺口",
                root_cause_code="tool_data_missing",
                candidate_root_cause=None,
            ),
        )
        assert updated is not None
        assert updated.attribution_detail_json["candidate_root_cause"] is None

    async def test_writes_triage_level_in_same_update(self, repo: TraceRecordService) -> None:
        """An overridden grade must be persisted in the same UPDATE as the bucket (the scalar + judge_json stay in sync)."""

        rec = await _seed_record(
            repo,
            trace_id="t-at-override",
            status="triaged",
            judge_json={"score": "0.1", "reason": "looks bad"},
        )
        await repo.update_triage_level(
            record_id=str(rec.id),
            triage_level=TriageLevel.SEVERE_BAD,
        )

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="unknown",
                corrected_triage_level=TriageLevel.WATCH,
                original_triage_level="severe_bad",
                triage_level=TriageLevel.ATTRIBUTION_OVERRIDDEN,
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert result.triage_level == TriageLevel.ATTRIBUTION_OVERRIDDEN
        assert result.judge_json["triage_level"] == "attribution_overridden"
        # The judge's original fields must not be wiped by the merge
        assert result.judge_json["score"] == "0.1"
        bucket = result.attribution_detail_json
        assert bucket["corrected_triage_level"] == TriageLevel.WATCH
        assert bucket["original_triage_level"] == "severe_bad"

        # Read it again after persisting, and confirm the scalar was really persisted (rather than only the in-memory object changing)
        fresh = await TraceRecord.filter(id=rec.id).first()
        assert fresh is not None
        assert fresh.triage_level == TriageLevel.ATTRIBUTION_OVERRIDDEN

    async def test_keeps_triage_level_when_not_overridden(self, repo: TraceRecordService) -> None:
        """When it is not overridden, triage_level is not passed, and the original grade stays unchanged."""

        rec = await _seed_record(repo, trace_id="t-at-keep", status="triaged")
        await repo.update_triage_level(
            record_id=str(rec.id),
            triage_level=TriageLevel.MILD_BAD,
        )

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="tool_error",
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert result.triage_level == TriageLevel.MILD_BAD

    async def test_merges_without_overwriting_reasoning_raw(self, repo: TraceRecordService) -> None:
        """update_attribution must not overwrite the reasoning_raw that update_reasoning already wrote."""
        rec = await _seed_record(repo, trace_id="t-at3", status="triaged")
        await repo.update_reasoning(
            record_id=str(rec.id),
            reasoning_json='{"chain": ["s1"]}',
        )

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="logic_bug",
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert result.root_cause == "logic_bug"
        # reasoning_raw must be kept (update_attribution did not pass reasoning_raw)
        assert result.attribution_detail_json["reasoning_raw"] == '{"chain": ["s1"]}'

    async def test_returns_none_for_missing_trace(self, repo: TraceRecordService) -> None:
        assert (
            await repo.update_attribution(
                update=AttributionUpdate(
                    root_cause="x",
                ),
                record_id=_NON_EXISTENT_UUID,
            )
            is None
        )

    async def test_writes_owner_group_scalar_and_bucket(self, repo: TraceRecordService) -> None:
        rec = await _seed_record(repo, trace_id="t-at-og", status="triaged")

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="hallucination",
                owner_group="nlp-platform",
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert result.owner_group == "nlp-platform"
        assert result.attribution_detail_json["owner_group"] == "nlp-platform"

    async def test_owner_group_omitted_leaves_existing_value(
        self, repo: TraceRecordService
    ) -> None:
        """When owner_group is not passed, the value already written is kept (it is not overwritten with None)."""
        rec = await _seed_record(repo, trace_id="t-at-og2", status="triaged")
        await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="tool_error",
                owner_group="search-team",
            ),
            record_id=str(rec.id),
        )

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="logic_bug",
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        # owner_group should keep the value written the first time
        assert result.owner_group == "search-team"
        # root_cause has been updated
        assert result.root_cause == "logic_bug"

    async def test_long_root_cause_clips_scalar_keeps_bucket_full(
        self,
        repo: TraceRecordService,
    ) -> None:
        """An over-long root_cause: the scalar is truncated to 256, the bucket keeps the full text, and the write succeeds."""
        rec = await _seed_record(repo, trace_id="t-at-long-rc", status="triaged")
        long_cause = "根因详情-" + ("x" * 300)

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause=long_cause,
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert len(result.root_cause) == 256
        assert result.root_cause == long_cause[:256]
        assert result.attribution_detail_json["root_cause"] == long_cause

    async def test_long_owner_group_clips_scalar_keeps_bucket_full(
        self,
        repo: TraceRecordService,
    ) -> None:
        """An over-long owner_group: the scalar is truncated to 128, and the bucket keeps the full text."""
        rec = await _seed_record(repo, trace_id="t-at-long-og", status="triaged")
        long_group = "team-" + ("y" * 200)

        result = await repo.update_attribution(
            update=AttributionUpdate(
                root_cause="tool_error",
                owner_group=long_group,
            ),
            record_id=str(rec.id),
        )

        assert result is not None
        assert len(result.owner_group) == 128
        assert result.owner_group == long_group[:128]
        assert result.attribution_detail_json["owner_group"] == long_group


# ══════════════════════════════════════════════════════════════════════════════
# PostgreSQL tests — requires PG_HOST/PG_USER/PG_PASSWORD/PG_DATABASE
# ══════════════════════════════════════════════════════════════════════════════


async def _pg_seed(repo: TraceRecordService, trace_id: str = "t-1", **kw) -> TraceRecord:
    agent = kw.pop("agent_name", "agent-a")
    return await repo.upsert(
        TraceRecordKey(
            trace_id=trace_id,
            project_id=kw.pop("project_id", "test-project"),
            agent_name=agent,
            evaluator_name=kw.pop("evaluator_name", "default"),
        ),
        status=kw.pop("status", "new"),
        **kw,
    )


class TestPgUpsert:
    async def test_insert_new(self, pg_trace_repo: TraceRecordService) -> None:
        rec = await pg_trace_repo.upsert(
            TraceRecordKey(
                trace_id="t-100",
                project_id="test-project",
                agent_name="agent-x",
                evaluator_name="default",
            ),
            status="scored",
        )
        assert isinstance(rec, TraceRecord)
        assert rec.trace_id == "t-100"
        assert rec.status == "scored"
        assert rec.created_at.tzinfo is not None

    async def test_update_existing(self, pg_trace_repo: TraceRecordService) -> None:
        original = await pg_trace_repo.upsert(
            TraceRecordKey(
                trace_id="t-200",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="new",
        )
        updated = await pg_trace_repo.upsert(
            TraceRecordKey(
                trace_id="t-200",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="scored",
            judge_json={"score": "0.85", "reason": "good"},
        )
        assert updated.status == "scored"
        assert updated.judge_json["score"] == "0.85"
        assert updated.updated_at >= original.updated_at

    async def test_upsert_with_any_agent_name(self, pg_trace_repo: TraceRecordService) -> None:
        rec = await pg_trace_repo.upsert(
            TraceRecordKey(
                trace_id="t-auto",
                project_id="p-auto",
                agent_name="ghost",
                evaluator_name="default",
            ),
            status="new",
        )
        assert rec.agent_name == "ghost"


class TestPgGetBy:
    async def test_get_by_trace_id(self, pg_trace_repo: TraceRecordService) -> None:
        await _pg_seed(pg_trace_repo, trace_id="t-find")
        assert (
            await pg_trace_repo.get_by_trace_id(
                TraceRecordKey(
                    trace_id="t-find",
                    project_id="test-project",
                    agent_name="agent-a",
                    evaluator_name="default",
                ),
            )
            is not None
        )
        assert (
            await pg_trace_repo.get_by_trace_id(
                TraceRecordKey(
                    trace_id="no-such",
                    project_id="test-project",
                    agent_name="agent-a",
                    evaluator_name="default",
                ),
            )
            is None
        )


class TestPgListByStatus:
    async def test_paginated(self, pg_trace_repo: TraceRecordService) -> None:
        for i in range(5):
            await pg_trace_repo.upsert(
                TraceRecordKey(
                    trace_id=f"t-{i}",
                    project_id="test-project",
                    agent_name="agent-a",
                    evaluator_name="default",
                ),
                status="scored",
            )
        records, total = await pg_trace_repo.list_by_status("scored", page=1, page_size=3)
        assert total == 5
        assert len(records) == 3


class TestPgUpdateReasoning:
    async def test_update_reasoning(self, pg_trace_repo: TraceRecordService) -> None:
        rec = await _pg_seed(pg_trace_repo, trace_id="t-reason")
        reasoning = json.dumps({"model": "qwen", "output": "test"})
        updated = await pg_trace_repo.update_reasoning(
            record_id=str(rec.id),
            reasoning_json=reasoning,
        )
        assert updated is not None and updated.attribution_detail_json["reasoning_raw"] == reasoning


class TestPgUpdateDispatch:
    async def test_update_dispatch(self, pg_trace_repo: TraceRecordService) -> None:
        rec = await _pg_seed(pg_trace_repo, trace_id="t-disp")
        now = datetime.now(UTC)
        updated = await pg_trace_repo.update_dispatch(
            update=DispatchUpdate(
                dispatch_ref="AONE-123",
                dispatch_url="https://aone.example.com/123",
                dispatch_assigned_to="owner@test",
                dispatch_title="fix bug",
                dispatched_at=now,
            ),
            record_id=str(rec.id),
        )
        assert updated is not None
        assert updated.notification_dispatch_json["notification_ref"] == "AONE-123"
        assert "notification_dispatched_at" in updated.notification_dispatch_json


class TestPgTimestamptz:
    async def test_datetime_roundtrip(self, pg_trace_repo: TraceRecordService) -> None:
        rec = await pg_trace_repo.upsert(
            TraceRecordKey(
                trace_id="t-tz",
                project_id="test-project",
                agent_name="agent-a",
                evaluator_name="default",
            ),
            status="new",
        )
        assert isinstance(rec.created_at, datetime)
        assert rec.created_at.tzinfo is not None


class TestPgUpdateTriageLevel:
    async def test_update_triage_level(self, pg_trace_repo: TraceRecordService) -> None:
        rec = await _pg_seed(pg_trace_repo, trace_id="t-tl")
        updated = await pg_trace_repo.update_triage_level(
            record_id=str(rec.id),
            triage_level="severe_bad",
        )
        assert updated is not None
        assert updated.triage_level == "severe_bad"
        assert updated.judge_json["triage_level"] == "severe_bad"


class TestPgUpdateAttribution:
    async def test_update_attribution(self, pg_trace_repo: TraceRecordService) -> None:
        rec = await _pg_seed(pg_trace_repo, trace_id="t-at")
        updated = await pg_trace_repo.update_attribution(
            update=AttributionUpdate(
                root_cause="hallucination",
                owner_domain="nlp",
                confidence=0.9,
            ),
            record_id=str(rec.id),
        )
        assert updated is not None
        assert updated.root_cause == "hallucination"
        assert updated.attribution_detail_json["owner_domain"] == "nlp"
        assert updated.attribution_detail_json["confidence"] == 0.9

    async def test_update_attribution_with_owner_group(
        self, pg_trace_repo: TraceRecordService
    ) -> None:
        rec = await _pg_seed(pg_trace_repo, trace_id="t-at-og")
        updated = await pg_trace_repo.update_attribution(
            update=AttributionUpdate(
                root_cause="hallucination",
                owner_group="nlp",
            ),
            record_id=str(rec.id),
        )
        assert updated is not None
        assert updated.owner_group == "nlp"
        assert updated.attribution_detail_json["owner_group"] == "nlp"


# ── enqueue_aone_task ──


class TestEnqueueNotificationTask:
    async def test_enqueue_creates_pending_job(self, repo: TraceRecordService) -> None:
        rec = await _seed_record(
            repo,
            trace_id="t-enq",
            agent_name="amap-agent",
            evaluator_name="example",
            status="scored",
        )
        payload = {"title": "badcase"}
        result = await repo.enqueue_notification_task(
            task=NotificationTaskEnqueue(
                idempotency_key="key-1",
                payload=payload,
            ),
            record_id=str(rec.id),
        )
        assert result is not None
        assert result.notification_job_json is not None
        assert result.notification_job_json["notification_job_status"] == "pending"
        assert result.notification_job_json["notification_job_idempotency_key"] == "key-1"
        assert result.notification_job_json["notification_job_payload"] == payload

    async def test_enqueue_same_key_force_pending_resets_failed_job(
        self, repo: TraceRecordService
    ) -> None:
        rec = await _seed_record(
            repo,
            trace_id="t-retry",
            agent_name="amap-agent",
            evaluator_name="example",
            status="scored",
            notification_job_json={
                "notification_job_status": "failed",
                "notification_job_idempotency_key": "key-retry",
                "notification_job_last_error": "timeout",
            },
        )
        new_payload = {"title": "retry"}
        result = await repo.enqueue_notification_task(
            task=NotificationTaskEnqueue(
                idempotency_key="key-retry",
                payload=new_payload,
                force_pending=True,
            ),
            record_id=str(rec.id),
        )
        assert result is not None
        assert result.notification_job_json["notification_job_status"] == "pending"
        assert result.notification_job_json["notification_job_payload"] == new_payload
        assert result.notification_job_json["notification_job_last_error"] is None

    async def test_enqueue_returns_none_when_record_missing(self, repo: TraceRecordService) -> None:
        assert (
            await repo.enqueue_notification_task(
                task=NotificationTaskEnqueue(
                    idempotency_key="k",
                    payload={},
                ),
                record_id=_NON_EXISTENT_UUID,
            )
            is None
        )
