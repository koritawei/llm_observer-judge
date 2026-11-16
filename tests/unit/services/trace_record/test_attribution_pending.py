"""T14/T15 unit tests: the attribution-accept pending mark + persisting attribution's structured fields.

Covers (review regression anchors):
1. mark_attribution_pending writes by merge — it does not affect the same row's judge_json or
   the ``TraceRecord.status`` scalar column (review round 1 #1: upsert reinsert semantics would clear each other's buckets).
2. After update_attribution finishes, attribution_status is set to completed and root_cause exists,
   and the three-state decision uses root_cause as the completion baseline.
3. The fields added in T15 (category/root_cause_code/conclusion_type/needs_human_review/
   evidence_details/action_hint) can all be written and read back unchanged (F7/F8).
4. Reading old data (missing the new fields) does not raise.
"""

import pytest
from benchmark.records.models import TraceRecord
from benchmark.services import TraceRecordService
from benchmark.services.trace_record_service import AttributionUpdate, TraceRecordKey


@pytest.fixture()
def repo() -> TraceRecordService:
    return TraceRecordService()


async def _seed_scored_record(
    repo: TraceRecordService,
    trace_id: str,
    judge_json: dict | None = None,
) -> TraceRecord:
    return await repo.upsert(
        TraceRecordKey(
            trace_id=trace_id,
            project_id="test-project",
            agent_name="agent-a",
            evaluator_name="vehicle_agent",
        ),
        status="scored",
        judge_json=judge_json
        or {"score": "0.0", "reason": "bad", "extra": {"batch_tag": "run-20260729-120000"}},
    )


class TestMarkAttributionPending:
    async def test_pending_visible_after_accept(self, repo: TraceRecordService) -> None:
        rec = await _seed_scored_record(repo, "t-pending-1")

        updated = await repo.mark_attribution_pending(record_id=str(rec.id))

        assert updated is not None
        assert updated.attribution_detail_json["attribution_status"] == "pending"
        # Three states: in progress = pending and no root_cause
        assert "root_cause" not in updated.attribution_detail_json

    async def test_pending_does_not_touch_judge_json_or_status(
        self, repo: TraceRecordService
    ) -> None:
        """Review #1 regression anchor: a merge write must not clear the other data on the same row."""
        rec = await _seed_scored_record(repo, "t-pending-2")

        updated = await repo.mark_attribution_pending(record_id=str(rec.id))

        assert updated.judge_json["score"] == "0.0"
        assert updated.judge_json["extra"]["batch_tag"] == "run-20260729-120000"
        assert updated.status == "scored"  # The TraceRecord.status scalar column is unaffected

    async def test_pending_overwritten_by_completion(self, repo: TraceRecordService) -> None:
        rec = await _seed_scored_record(repo, "t-pending-3")
        await repo.mark_attribution_pending(record_id=str(rec.id))

        done = await repo.update_attribution(
            record_id=str(rec.id), update=AttributionUpdate(root_cause="幻觉")
        )

        assert done.attribution_detail_json["attribution_status"] == "completed"
        assert done.attribution_detail_json["root_cause"] == "幻觉"

    async def test_missing_record_returns_none(self, repo: TraceRecordService) -> None:
        result = await repo.mark_attribution_pending(
            record_id="00000000-0000-0000-0000-000000000000",
        )
        assert result is None


class TestAttributionStructuredFields:
    async def test_all_new_fields_roundtrip(self, repo: TraceRecordService) -> None:
        """T15: after every new field is written it can be read back unchanged from the bucket (F7)."""
        rec = await _seed_scored_record(repo, "t-fields-1")
        evidence = [
            {
                "field": "poi.name",
                "expected": "A店",
                "actual": "B店",
                "source": "trace_tool",
                "observation_id": "obs-1",
            },
        ]

        done = await repo.update_attribution(
            record_id=str(rec.id),
            update=AttributionUpdate(
                root_cause="POI标签嫁接",
                owner_domain="model",
                owner_group="model-team",
                confidence=0.9,
                corrected_triage_level="watch",
                category="model",
                root_cause_code="poi_tag_grafting",
                conclusion_type="agent_badcase",
                needs_human_review=True,
                evidence_details=evidence,
                action_hint="复查 D3 标签嫁接 checker",
            ),
        )

        bucket = done.attribution_detail_json
        assert bucket["category"] == "model"
        assert bucket["root_cause_code"] == "poi_tag_grafting"
        assert bucket["conclusion_type"] == "agent_badcase"
        assert bucket["needs_human_review"] is True
        assert bucket["evidence_details"] == evidence
        assert bucket["action_hint"] == "复查 D3 标签嫁接 checker"
        # The old key is kept for compatibility (F8: owner_domain historically stored the category value)
        assert bucket["owner_domain"] == "model"
        # A direct write of the scalar column is unaffected
        assert done.root_cause == "POI标签嫁接"
        assert done.owner_group == "model-team"

    async def test_old_data_without_new_fields_reads_fine(self, repo: TraceRecordService) -> None:
        """Reading old data (missing the new fields) does not raise, and the three-state decision is completed."""
        rec = await _seed_scored_record(repo, "t-fields-2")
        done = await repo.update_attribution(
            record_id=str(rec.id), update=AttributionUpdate(root_cause="旧格式归因")
        )

        bucket = done.attribution_detail_json
        assert bucket.get("root_cause_code") is None
        assert bucket.get("evidence_details") is None
        # Completed = root_cause is present
        assert bucket["root_cause"] == "旧格式归因"
        assert bucket["attribution_status"] == "completed"


@pytest.mark.parametrize("change", ["omitted", "clear", "replace"])
async def test_candidate_update_distinguishes_omission_and_clear(repo, change):
    rec = await _seed_scored_record(repo, "candidate-" + change)
    old = {"root_cause_code": "other:old"}
    await repo.update_attribution(
        record_id=str(rec.id), update=AttributionUpdate(root_cause="old", candidate_root_cause=old)
    )
    kwargs = (
        {}
        if change == "omitted"
        else {
            "candidate_root_cause": None if change == "clear" else {"root_cause_code": "other:new"}
        }
    )
    done = await repo.update_attribution(
        record_id=str(rec.id), update=AttributionUpdate(root_cause="replay", **kwargs)
    )
    expected = old if change == "omitted" else kwargs["candidate_root_cause"]
    assert done.attribution_detail_json["candidate_root_cause"] == expected


@pytest.mark.parametrize("completed", [False, True])
async def test_failure_marker_only_transitions_pending(repo, completed):
    rec = await _seed_scored_record(repo, "failure-" + str(completed))
    await repo.mark_attribution_pending(record_id=str(rec.id))
    if completed:
        await repo.update_attribution(
            record_id=str(rec.id), update=AttributionUpdate(root_cause="done")
        )
    result = await repo.mark_attribution_failed(record_id=str(rec.id))
    assert result.attribution_detail_json["attribution_status"] == (
        "completed" if completed else "failed"
    )


@pytest.mark.parametrize("execution", ["sync", "async"])
async def test_agent_failure_closes_pending_in_pipeline(repo, execution):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from benchmark.attribution.service.service import (
        BeforeRunDecision,
        TriageAttributionService,
        TriageRunInput,
        TriageServiceDeps,
    )

    rec = await _seed_scored_record(repo, "pipeline-failure-" + execution)
    manager = MagicMock()
    manager.get.side_effect = KeyError("missing")
    service = TriageAttributionService(
        TriageServiceDeps(
            agent_manager=manager,
            before_hook=AsyncMock(return_value=BeforeRunDecision(should_proceed=True)),
            after_hook=AsyncMock(),
            trace_record_service=repo,
        )
    )
    done = asyncio.Event()
    original = service._close_pending_attribution

    async def close(record_id):
        await original(record_id)
        done.set()

    service._close_pending_attribution = close
    await service.run(
        TriageRunInput(
            evaluator_name="vehicle_agent",
            evaluator_request=SimpleNamespace(
                trace_id="pipeline-failure-" + execution,
                project_id="test-project",
                agent_name="agent-a",
                metadata={},
            ),
            evaluator_result=SimpleNamespace(score=0),
            attribution_execution=execution,
        )
    )
    await asyncio.wait_for(done.wait(), 2)
    await rec.refresh_from_db()
    assert rec.attribution_detail_json["attribution_status"] == "failed"
