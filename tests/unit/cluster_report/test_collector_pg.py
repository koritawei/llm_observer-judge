"""T11 unit tests: the PG collector (batch recall / time window / evaluator narrowing / three-state counts).

Two layers:
- ``TestSqlite*`` — runs locally and covers the SQLite dialect branch (time window + Python-side filtering),
  field mapping, three-state decisions, and multi-evaluator dedupe.
- ``TestPg*`` — runs when the PG environment variables are present (CI), and checks that the real PG dialect
  follows the same "time window + Python-side filtering" path as SQLite. With no local PG env, conftest skips it.

Note: the fixture name contains ``repo`` so conftest initializes SQLite Tortoise.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio
from benchmark.reporting.cluster.constants import DEFAULT_EVALUATOR_NAME
from benchmark.records.models import TraceRecord
from benchmark.pipeline.envelope import (
    AgentEvalEnvelope,
    DiagnosticFinding,
    MetricObservation,
)
from benchmark.pipeline.extensions import VehicleEvalExtensionV3
from benchmark.pipeline.signals import DimensionResult
from benchmark.services import TraceRecordService
from benchmark.reporting.cluster_service.collector import (
    _PROGRESS_COLUMNS,
    BatchCollectionOptions,
    PgCollector,
    _BatchQuery,
    parse_batch_window,
)
from benchmark.services.trace_record_service import AttributionUpdate, TraceRecordKey

_UTC_PLUS_8 = timezone(timedelta(hours=8))
_BATCH = "run-20260729-153000"
_BATCH_START = datetime(2026, 7, 29, 15, 30, tzinfo=_UTC_PLUS_8)


@pytest.fixture()
def repo() -> TraceRecordService:
    """SQLite TraceRecordService (the fixture name triggers conftest's ORM initialization)."""
    return TraceRecordService()


@pytest.fixture()
def collector() -> PgCollector:
    return PgCollector()


def _judge_json(
    *,
    batch_tag: str | None = _BATCH,
    score: str = "1.0",
    suite: str = "车机-Standard",
    case_id: str = "standard-1",
    tag: str = "navigation.direct",
    query: str = "导航回家",
    turn_index: str = "",
) -> dict:
    extra: dict = {
        "schema_version": "vehicle_eval_extra.v2",
        "score": float(score),
        "reason": "ok",
        "dimension_results": [{"dimension": "D4", "status": "fail"}],
        "checker_results": [{"source_name": "cmd_checker", "status": "fail"}],
        "hard_rule_hits": ["cmd_missing"],
        "risk_flags": ["cmd_sensitive"],
        "failure_summary": {"failure_modes": [{"mode": "cmd_tag_missing", "group": "cmd"}]},
        "confidence": 0.8,
        "tag": tag,
        "query": query,
    }
    if turn_index:
        extra["turn_index"] = turn_index
    if batch_tag is not None:
        extra["batch_tag"] = batch_tag
        extra["case_meta"] = {
            "source": "golden_testset",
            "suite": suite,
            "case_id": case_id,
        }
    return {"score": score, "reason": "ok", "extra": extra}


async def _seed(
    repo: TraceRecordService,
    *,
    trace_id: str,
    evaluator_name: str = DEFAULT_EVALUATOR_NAME,
    created_at: datetime | None = None,
    **judge_kwargs,
) -> TraceRecord:
    rec = await repo.upsert(
        TraceRecordKey(
            trace_id=trace_id,
            project_id="p-1",
            agent_name="vehicle-agent",
            evaluator_name=evaluator_name,
        ),
        status="scored",
        judge_json=_judge_json(**judge_kwargs),
    )
    if created_at is not None:
        # created_at is auto_now_add, so the time-window case has to bypass the ORM and write it directly.
        await TraceRecord.filter(id=rec.id).update(created_at=created_at)
        rec = await TraceRecord.filter(id=rec.id).first()
    return rec


class TestParseBatchWindow:
    def test_window_anchored_to_utc8(self) -> None:
        start, end = parse_batch_window(_BATCH)
        # The left edge looks back 10min, and the right edge is +6h (the default window)
        assert start == _BATCH_START - timedelta(minutes=10)
        assert end == _BATCH_START + timedelta(hours=6)
        assert start.utcoffset() == timedelta(hours=8)

    def test_custom_window(self) -> None:
        _, end = parse_batch_window(_BATCH, window_s=3600)
        assert end == _BATCH_START + timedelta(hours=1)

    def test_cross_midnight_batch(self) -> None:
        """A batch that crosses midnight: the window is the interval [start + window], and natural day boundaries do not affect it."""
        start, end = parse_batch_window("run-20260729-235000", window_s=3600)
        assert start.day == 29
        assert end.day == 30 and end.hour == 0 and end.minute == 50

    def test_unparsable_returns_none(self) -> None:
        """An invalid tag returns None → the caller degrades to no time window (it must not silently return empty)."""
        for bad in ["", "run-bad", "run-20260729", "manual-tag", "run-20261332-153000"]:
            assert parse_batch_window(bad) is None, bad


class TestNoQuerySetExtra:
    """Regression guard: tortoise 1.1.7 has no QuerySet.extra(). That Django-style API once caused
    /progress to 500 in the PG environment (SQLite unit tests did not cover it). Lock the source so it is not reintroduced."""

    def test_collector_source_has_no_queryset_extra(self) -> None:
        import re
        from pathlib import Path

        import benchmark.reporting.cluster_service.collector as collector_mod

        source = Path(collector_mod.__file__).read_text(encoding="utf-8")
        # Check only real calls of `<expr>.extra(` on non-comment lines, and do not flag
        # the text "QuerySet.extra()" inside comments.
        call_pattern = re.compile(r"[\w\)\]]\.extra\(")
        offending = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith("#") and call_pattern.search(line)
        ]
        assert not offending, (
            f"tortoise QuerySet 无 .extra()，PG 上会 AttributeError→500：{offending}"
        )

    def test_queryset_really_lacks_extra(self) -> None:
        from tortoise.queryset import QuerySet

        assert not hasattr(QuerySet, "extra"), "若 tortoise 日后提供 .extra() 可重评下推优化"


class TestSqliteFetchBatchRecords:
    async def test_recalls_only_matching_batch(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        await _seed(repo, trace_id="t-1", case_id="standard-1", created_at=_BATCH_START)
        await _seed(repo, trace_id="t-2", case_id="standard-2", created_at=_BATCH_START)
        # Other batches, and rows with no batch (live traffic), must not be recalled
        await _seed(repo, trace_id="t-3", batch_tag="run-20260728-100000", created_at=_BATCH_START)
        await _seed(repo, trace_id="t-4", batch_tag=None, created_at=_BATCH_START)

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert {r.trace_id for r in records} == {"t-1", "t-2"}

    def test_preserves_project_identity_for_private_trace_rendering(self) -> None:
        row = SimpleNamespace(
            trace_id="raw-primary-id",
            project_id="p-1",
            id="row-1",
            evaluator_name=DEFAULT_EVALUATOR_NAME,
            judge_json=_judge_json(),
            trace_meta_json={},
            triage_level="",
            attribution_detail_json={},
            root_cause="",
            owner_group="",
            notification_dispatch_json={},
        )

        record = PgCollector._to_record(row)

        assert record.project_id == "p-1"

    async def test_extracts_notification_workitem_from_dispatch_reference(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        seeded = await _seed(repo, trace_id="t-aone", created_at=_BATCH_START)
        await TraceRecord.filter(id=seeded.id).update(notification_dispatch_json={"notification_ref": "85881892"})

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert records[0].notification_workitem_id == "85881892"

    async def test_time_window_recalls_online_records_without_batch_tag(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        await _seed(
            repo,
            trace_id="t-online",
            batch_tag=None,
            evaluator_name="vui_agent",
            created_at=_BATCH_START,
        )
        await _seed(
            repo,
            trace_id="t-other-evaluator",
            batch_tag=None,
            evaluator_name="other_judge",
            created_at=_BATCH_START,
        )
        await _seed(
            repo,
            trace_id="t-outside-window",
            batch_tag=None,
            evaluator_name="vui_agent",
            created_at=_BATCH_START + timedelta(hours=3),
        )
        await _seed(
            repo,
            trace_id="t-window-end",
            batch_tag=None,
            evaluator_name="vui_agent",
            created_at=_BATCH_START + timedelta(minutes=1),
        )

        records = await collector.fetch_time_window_records(
            start_at=_BATCH_START - timedelta(minutes=1),
            end_at=_BATCH_START + timedelta(minutes=1),
            evaluator_name="vui_agent",
        )

        assert [record.trace_id for record in records] == ["t-online"]

    async def test_evaluator_name_narrows_count(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """One trace with multiple evaluators means multiple rows; failing to narrow inflates judged_count (F10)."""
        await _seed(repo, trace_id="t-same", created_at=_BATCH_START)
        await _seed(repo, trace_id="t-same", evaluator_name="other_judge", created_at=_BATCH_START)

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert len(records) == 1
        assert records[0].evaluator_name == DEFAULT_EVALUATOR_NAME

    async def test_time_window_narrows_but_tag_match_decides(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """The time window is only a scan optimization: a hit inside the window is returned directly and does not trigger a full scan."""
        await _seed(repo, trace_id="t-in", created_at=_BATCH_START + timedelta(hours=1))
        # Outside the default 6h window: when something inside the window hits, it does not join the recall (no degrade-to-full-scan)
        await _seed(repo, trace_id="t-late", created_at=_BATCH_START + timedelta(hours=7))

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert {r.trace_id for r in records} == {"t-in"}

    async def test_retry_mode_merges_original_window_and_late_records(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """A late retry that reuses batch_tag merges with the original batch, and the same case keeps the latest record."""
        await _seed(
            repo,
            trace_id="t-original",
            case_id="standard-1",
            score="0.0",
            created_at=_BATCH_START + timedelta(hours=1),
        )
        await _seed(
            repo,
            trace_id="t-untouched",
            case_id="standard-2",
            score="1.0",
            created_at=_BATCH_START + timedelta(hours=2),
        )
        await _seed(
            repo,
            trace_id="t-retry",
            case_id="standard-1",
            score="2.0",
            created_at=_BATCH_START + timedelta(hours=7),
        )
        await _seed(
            repo,
            trace_id="t-other-batch",
            batch_tag="run-20260728-100000",
            case_id="standard-3",
            created_at=_BATCH_START + timedelta(hours=8),
        )

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
            options=BatchCollectionOptions(window_s=72 * 3600, include_late_records=True),
        )
        progress = await collector.count_progress(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
            options=BatchCollectionOptions(window_s=72 * 3600, include_late_records=True),
        )

        assert {record.trace_id for record in records} == {"t-retry", "t-untouched"}
        assert next(record for record in records if record.trace_id == "t-retry").score == 2.0
        assert progress["judged_count"] == 2

    async def test_window_miss_returns_incomplete_recall_without_full_scan(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """An old record outside the bounded window must not trigger a full-table scan on the evaluator dimension."""
        await _seed(repo, trace_id="t-late", created_at=_BATCH_START + timedelta(days=16))
        await _seed(
            repo,
            trace_id="t-other",
            batch_tag="run-20260728-100000",
            created_at=_BATCH_START + timedelta(days=16),
        )

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
            options=BatchCollectionOptions(window_s=72 * 3600),
        )

        assert records == []

    async def test_window_override_widens_range(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        await _seed(repo, trace_id="t-late", created_at=_BATCH_START + timedelta(hours=7))

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
            options=BatchCollectionOptions(window_s=12 * 3600),
        )

        assert {r.trace_id for r in records} == {"t-late"}

    async def test_maps_judge_fields(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        await _seed(repo, trace_id="t-map", created_at=_BATCH_START)

        rec = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0]

        assert rec.score == 1.0
        assert rec.hard_rule_hits == ["cmd_missing"]
        assert rec.risk_flags == ["cmd_sensitive"]
        assert rec.dimension_results[0]["dimension"] == "D4"
        assert rec.checker_results[0]["source_name"] == "cmd_checker"
        assert rec.failure_modes[0]["mode"] == "cmd_tag_missing"
        assert rec.judge_confidence == 0.8
        assert rec.suite == "车机-Standard"

    async def test_case_key_uses_case_id_path(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        await _seed(repo, trace_id="t-key", created_at=_BATCH_START)

        rec = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0]

        assert rec.case_key.normalized() == ("id", "车机-Standard", "standard-1", "")
        assert rec.case_key.tag == "navigation.direct"
        assert rec.case_key.query == "导航回家"

    async def test_vui_envelope_uses_user_query_for_case_key(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="vui-query",
                project_id="p-vui",
                agent_name="vui_agent",
                evaluator_name="vui_agent",
            ),
            status="scored",
            judge_json={
                "score": "1.0",
                "reason": "bad",
                "extra": {
                    "schema_version": "agent_eval_extra.v1",
                    "primary_score": 1.0,
                    "summary_reason": "bad",
                    "business_extension": {
                        "extension_type": "vui_eval",
                        "extension_version": "v2",
                        "payload": {"batch_tag": _BATCH, "user_query": "帮我导航去机场"},
                    },
                },
            },
        )
        await TraceRecord.filter(id=rec.id).update(created_at=_BATCH_START)

        got = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name="vui_agent",
            )
        )[0]

        assert got.case_key.query == "帮我导航去机场"
        assert got.case_key.normalized() == ("q", "", "", "帮我导航去机场", "")

    async def test_case_key_falls_back_without_case_id(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """No case_id (black-box/white-box/sampled case): fall back to the tag+query+turn_index path.

        Regression anchor: tag/query/turn_index must actually land in judge_json.extra,
        or the fallback path degenerates to (suite,"","","" ) — the dead-code issue found in review.
        """
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-nokey",
                project_id="p-1",
                agent_name="vehicle-agent",
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            ),
            status="scored",
            judge_json={
                "score": "0.0",
                "reason": "bad",
                "extra": {
                    "batch_tag": _BATCH,
                    "case_meta": {"source": "sampled", "suite": "pool-A"},
                    "tag": "multi-turn.s1",
                    "query": "取消",
                    "turn_index": "Q2",
                },
            },
        )
        await TraceRecord.filter(id=rec.id).update(created_at=_BATCH_START)

        got = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0]

        assert got.case_key.normalized() == ("q", "pool-A", "multi-turn.s1", "取消", "Q2")

    async def test_unparsable_batch_tag_returns_incomplete_recall_without_full_scan(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """An unparseable tag returns an empty result and must not scan the whole store by evaluator."""
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="t-manual",
                project_id="p-1",
                agent_name="vehicle-agent",
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            ),
            status="scored",
            judge_json={"score": "1.0", "extra": {"batch_tag": "manual-batch"}},
        )
        assert rec is not None

        records = await collector.fetch_batch_records(
            batch_tag="manual-batch",
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert records == []

    async def test_empty_batch_returns_empty_list(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )
        assert records == []


class TestSqliteAttributionThreeStates:
    async def test_none_state(self, repo: TraceRecordService, collector: PgCollector) -> None:
        """Attribution not triggered: no marker and no root_cause."""
        await _seed(repo, trace_id="t-none", created_at=_BATCH_START)

        rec = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0]

        assert rec.attribution.status == "none"
        assert not rec.attribution.is_pending
        assert not rec.attribution.is_completed

    async def test_pending_state(self, repo: TraceRecordService, collector: PgCollector) -> None:
        seeded = await _seed(repo, trace_id="t-pending", created_at=_BATCH_START)
        await repo.mark_attribution_pending(record_id=str(seeded.id))
        await TraceRecord.filter(id=seeded.id).update(created_at=_BATCH_START)

        rec = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0]

        assert rec.attribution.is_pending

    async def test_completed_state_with_all_fields(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """Completed: root_cause is present; every field filled in by T15 can be read back."""
        seeded = await _seed(repo, trace_id="t-done", created_at=_BATCH_START)
        await repo.mark_attribution_pending(record_id=str(seeded.id))
        await repo.update_attribution(
            record_id=str(seeded.id),
            update=AttributionUpdate(
                root_cause="CMD 缺失",
                owner_group="prompt-team",
                confidence=0.9,
                corrected_triage_level="watch",
                category="prompt",
                root_cause_code="cmd_tag_missing",
                all_root_cause_codes=["cmd_tag_missing", "tool_param_error"],
                conclusion_type="judge_false_positive",
                needs_human_review=True,
                evidence_details=[{"field": "cmd_list", "expected": "1", "actual": "0"}],
                action_hint="补 prompt 指令",
            ),
        )
        await TraceRecord.filter(id=seeded.id).update(created_at=_BATCH_START)

        attr = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0].attribution

        assert attr.is_completed
        assert attr.root_cause_code == "cmd_tag_missing"
        assert attr.all_root_cause_codes == ["cmd_tag_missing", "tool_param_error"]
        assert attr.category == "prompt"
        assert attr.owner_group == "prompt-team"
        assert attr.conclusion_type == "judge_false_positive"
        assert attr.is_overturn  # judge_false_positive → overturn the evaluation
        assert attr.corrected_triage_level == "watch"
        assert attr.needs_human_review is True
        assert attr.evidence_details[0]["field"] == "cmd_list"
        assert attr.action_hint == "补 prompt 指令"
        assert attr.confidence == 0.9

    async def test_category_falls_back_to_owner_domain(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """When old data has only owner_domain (historically it stored the category value), fall back to that read (F8)."""
        seeded = await _seed(repo, trace_id="t-legacy", created_at=_BATCH_START)
        await repo.update_attribution(
            record_id=str(seeded.id),
            update=AttributionUpdate(root_cause="旧归因", owner_domain="model"),
        )
        await TraceRecord.filter(id=seeded.id).update(created_at=_BATCH_START)

        attr = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0].attribution

        assert attr.category == "model"
        assert attr.root_cause_code == ""  # Old data has no such field, and that must not raise
        assert attr.all_root_cause_codes == []  # Old data falls back to an empty list


class TestSqliteCountProgress:
    async def test_three_state_counts(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        # 2 not triggered (distinct case_ids — multiple rows of the same case are deduped to the latest; see TestRetryDedupe)
        await _seed(repo, trace_id="t-a", case_id="c-a", created_at=_BATCH_START)
        await _seed(repo, trace_id="t-b", case_id="c-b", created_at=_BATCH_START)
        # 1 in progress
        pending = await _seed(repo, trace_id="t-c", case_id="c-c", created_at=_BATCH_START)
        await repo.mark_attribution_pending(record_id=str(pending.id))
        await TraceRecord.filter(id=pending.id).update(created_at=_BATCH_START)
        # 2 completed
        for tid in ("t-d", "t-e"):
            done = await _seed(repo, trace_id=tid, case_id=f"c-{tid}", created_at=_BATCH_START)
            await repo.update_attribution(
                record_id=str(done.id), update=AttributionUpdate(root_cause="x")
            )
            await TraceRecord.filter(id=done.id).update(created_at=_BATCH_START)

        progress = await collector.count_progress(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert progress == {
            "judged_count": 5,
            "should_attribute_count": 3,  # pending + completed
            "attributed_count": 2,
            "pending": 1,
        }

    async def test_empty_batch_zero_counts(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        progress = await collector.count_progress(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )
        assert progress["judged_count"] == 0
        assert progress["should_attribute_count"] == 0


class TestProgressNarrowColumns:
    """T6 narrow-column path: the watcher's high-frequency poll reads only the columns the counts need, and does not load large buckets.

    tortoise ``.only()`` raises AttributeError on an unselected column (fail-fast). This group both
    verifies the large buckets were not loaded and locks _PROGRESS_COLUMNS to every access of the
    count path (a future field access added to the count path will blow up here).
    """

    async def test_progress_path_skips_large_buckets(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        seeded = await _seed(repo, trace_id="t-narrow", created_at=_BATCH_START)
        # Write the large buckets directly, and verify the narrow-column path does not bring them back
        await TraceRecord.filter(id=seeded.id).update(
            trace_meta_json={"source": "http", "input": "x" * 1000},
            notification_job_json={"notification_job_payload": {"description": "y" * 1000}},
        )

        progress = await collector.count_progress(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )
        assert progress["judged_count"] == 1

        rows = await collector._query_rows(
            _BatchQuery(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
                narrow=True,
                options=BatchCollectionOptions(window_s=6 * 3600),
            )
        )
        for row in rows:
            with pytest.raises(AttributeError):
                _ = row.trace_meta_json
            with pytest.raises(AttributeError):
                _ = row.notification_job_json

    async def test_progress_columns_cover_count_access_surface(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """The narrow column set covers every field accessed by dedupe and the three-state counts (reading an unselected column fails fast)."""
        seeded = await _seed(repo, trace_id="t-covered", created_at=_BATCH_START)
        await repo.mark_attribution_pending(record_id=str(seeded.id))
        await TraceRecord.filter(id=seeded.id).update(created_at=_BATCH_START)

        progress = await collector.count_progress(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert progress == {
            "judged_count": 1,
            "should_attribute_count": 1,
            "attributed_count": 0,
            "pending": 1,
        }

    def test_progress_columns_exclude_large_buckets(self) -> None:
        """Guard: the narrow column set must not fall back to one that includes a large bucket (full evaluation IO / the full report payload)."""
        for large in ("trace_meta_json", "notification_job_json", "notification_dispatch_json"):
            assert large not in _PROGRESS_COLUMNS, large


class TestPgFetchBatchRecords:
    """Batch recall over a real PG connection (auto-skipped when there is no PG env).

    It takes the same "time window + Python-side filtering" path as SQLite (tortoise 1.1.7 has no
    QuerySet.extra(), so there is no JSONB pushdown). This group checks the real PG dialect behaves the same.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def _collector(self) -> PgCollector:
        return PgCollector()

    async def test_jsonb_filter_recalls_matching_batch(
        self, pg_trace_repo: TraceRecordService
    ) -> None:
        collector = PgCollector()
        await _seed(pg_trace_repo, trace_id="pg-1")
        await _seed(pg_trace_repo, trace_id="pg-2", batch_tag="run-20260728-100000")
        await _seed(pg_trace_repo, trace_id="pg-3", batch_tag=None)
        # Under PG, created_at is generated by the DB (current time); cover it with a wide window
        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
            options=BatchCollectionOptions(window_s=10 * 365 * 24 * 3600),
        )

        assert {r.trace_id for r in records} == {"pg-1"}

    async def test_jsonb_filter_narrows_by_evaluator(
        self, pg_trace_repo: TraceRecordService
    ) -> None:
        collector = PgCollector()
        await _seed(pg_trace_repo, trace_id="pg-same")
        await _seed(pg_trace_repo, trace_id="pg-same", evaluator_name="other_judge")

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
            options=BatchCollectionOptions(window_s=10 * 365 * 24 * 3600),
        )

        assert len(records) == 1
        assert records[0].evaluator_name == DEFAULT_EVALUATOR_NAME

    async def test_jsonb_progress_counts(self, pg_trace_repo: TraceRecordService) -> None:
        collector = PgCollector()
        await _seed(pg_trace_repo, trace_id="pg-x", case_id="c-x")
        done = await _seed(pg_trace_repo, trace_id="pg-y", case_id="c-y")
        await pg_trace_repo.update_attribution(
            record_id=str(done.id), update=AttributionUpdate(root_cause="x")
        )

        progress = await collector.count_progress(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
            options=BatchCollectionOptions(window_s=10 * 365 * 24 * 3600),
        )

        assert progress["judged_count"] == 2
        assert progress["attributed_count"] == 1


class TestRetryDedupe:
    """Multiple rows of the same case inside one batch (client retry / server redelivery) keep only the latest row.

    run-eval.py max_retries retries timeouts and business failures. Each retry produces a new trace_id →
    a different TraceRecord composite key → multiple rows for the same batch_tag, with identical case_meta.
    Without dedupe, total / pass_rate / dimension fail rates are all inflated.
    """

    async def test_same_case_keeps_latest_row(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """A successful retry's new row overrides the first failed row (latest by created_at)."""
        await _seed(
            repo,
            trace_id="try-1",
            case_id="standard-7",
            score="0.0",
            created_at=_BATCH_START,
        )
        await _seed(
            repo,
            trace_id="try-2",
            case_id="standard-7",
            score="2.0",
            created_at=_BATCH_START + timedelta(minutes=3),
        )

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert len(records) == 1
        assert records[0].trace_id == "try-2"
        assert records[0].score == 2.0

    async def test_row_order_independent(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """A row inserted later is still the one kept (does not depend on DB return order)."""
        await _seed(
            repo,
            trace_id="late",
            case_id="standard-8",
            score="2.0",
            created_at=_BATCH_START + timedelta(minutes=5),
        )
        await _seed(
            repo,
            trace_id="early",
            case_id="standard-8",
            score="0.0",
            created_at=_BATCH_START,
        )

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert [r.trace_id for r in records] == ["late"]

    async def test_multi_turn_not_collapsed(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """turn_index is part of the key: rounds of the same scenario must not be merged into one row by mistake."""
        await _seed(
            repo, trace_id="turn-1", case_id="loop-导航", turn_index="1", created_at=_BATCH_START
        )
        await _seed(
            repo, trace_id="turn-2", case_id="loop-导航", turn_index="2", created_at=_BATCH_START
        )

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert {r.trace_id for r in records} == {"turn-1", "turn-2"}

    async def test_distinct_cases_untouched(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """Different cases are all kept (dedupe must not damage a normal batch)."""
        for i in range(5):
            await _seed(repo, trace_id=f"c-{i}", case_id=f"standard-{i}", created_at=_BATCH_START)

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert len(records) == 5

    async def test_rows_without_identity_all_kept(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """Rows with no usable identity (case_id/tag/query all empty) are all kept.

        normalized() for these rows degenerates to the same empty key. If they joined dedupe they would collapse to one row →
        data loss. Old data missing case_meta / extra must take this fallback.
        """
        for tid in ("nokey-1", "nokey-2", "nokey-3"):
            await _seed(
                repo,
                trace_id=tid,
                case_id="",
                tag="",
                query="",
                created_at=_BATCH_START,
            )

        records = await collector.fetch_batch_records(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert {r.trace_id for r in records} == {"nokey-1", "nokey-2", "nokey-3"}

    async def test_progress_counts_deduped(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """progress converges the same way: judged_count is no longer inflated by retry rows (otherwise the watcher fills up early)."""
        await _seed(repo, trace_id="p-1", case_id="standard-1", created_at=_BATCH_START)
        await _seed(
            repo,
            trace_id="p-1-retry",
            case_id="standard-1",
            created_at=_BATCH_START + timedelta(minutes=1),
        )
        await _seed(repo, trace_id="p-2", case_id="standard-2", created_at=_BATCH_START)

        progress = await collector.count_progress(
            batch_tag=_BATCH,
            evaluator_name=DEFAULT_EVALUATOR_NAME,
        )

        assert progress["judged_count"] == 2


def _envelope_judge_json(*, score: float = 2.0) -> dict:
    """judge_json in the unified envelope shape (batch identity is carried on the Vehicle extension payload)."""

    legacy = {
        "schema_version": "vehicle_eval_extra.v3",
        "score": score,
        "reason": "envelope-baseline",
        "dimension_results": [{"dimension": "D4", "status": "pass"}],
        "batch_tag": _BATCH,
        "case_meta": {"source": "golden_testset", "suite": "车机-Standard", "case_id": "env-1"},
        "tag": "navigation.direct",
        "query": "导航去公司",
    }
    envelope = AgentEvalEnvelope(
        business_id="vehicle_agent",
        scenario="navigation",
        profile_id="vehicle_agent.standard",
        profile_version="v1",
        trace_id="env-1",
        primary_score=score,
        summary_reason="envelope-summary",
        dimension_results=[DimensionResult(dimension="D4", status="pass")],
        metric_observations=[
            MetricObservation(
                metric_key="first_text_ms",
                value=1800,
                unit="ms",
                status="exceeded",
                collector_name="perf",
            ),
        ],
        diagnostic_findings=[
            DiagnosticFinding(
                finding_key="llm_retry_recovered",
                severity="warning",
                status="recovered",
                source_name="runner",
            ),
        ],
        business_extension=VehicleEvalExtensionV3.model_validate(legacy).to_business_extension(),
    )
    return {
        "score": str(score),
        "reason": "envelope-summary",
        "extra": envelope.model_dump(mode="json"),
    }


class TestEnvelopeDirectRead:
    """Direct reads of the unified envelope's public core (``_envelope_sources``): envelope identity split keys and
    Metric/Diagnostic independent observations go straight onto the record. Historical non-envelope behavior stays an empty default, unchanged."""

    async def test_envelope_row_reads_public_core(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        rec = await repo.upsert(
            TraceRecordKey(
                trace_id="env-1",
                project_id="p-1",
                agent_name="vehicle-agent",
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            ),
            status="scored",
            judge_json=_envelope_judge_json(),
        )
        await TraceRecord.filter(id=rec.id).update(created_at=_BATCH_START)

        got = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0]

        assert got.business_id == "vehicle_agent"
        assert got.scenario == "navigation"
        assert got.profile_id == "vehicle_agent.standard"
        assert got.profile_version == "v1"
        assert got.score == 2.0
        assert got.reason == "envelope-summary"
        # An envelope DimensionResult serializes default fields; assert only the keys downstream actually consumes
        assert got.dimension_results[0]["dimension"] == "D4"
        assert got.dimension_results[0]["status"] == "pass"
        assert [item["metric_key"] for item in got.metric_observations] == ["first_text_ms"]
        assert [item["finding_key"] for item in got.diagnostic_findings] == [
            "llm_retry_recovered",
        ]

    async def test_legacy_row_envelope_fields_are_empty(
        self, repo: TraceRecordService, collector: PgCollector
    ) -> None:
        """A historical row (not an envelope): every direct-read envelope field is an empty default, and the legacy read view is unaffected."""
        await _seed(repo, trace_id="t-legacy", created_at=_BATCH_START)

        got = (
            await collector.fetch_batch_records(
                batch_tag=_BATCH,
                evaluator_name=DEFAULT_EVALUATOR_NAME,
            )
        )[0]

        assert got.business_id == ""
        assert got.scenario == ""
        assert got.profile_id == ""
        assert got.profile_version == ""
        assert got.metric_observations == []
        assert got.diagnostic_findings == []
        # The legacy read view keeps its old behavior
        assert got.score == 1.0
        assert got.dimension_results[0]["dimension"] == "D4"


async def test_online_window_preserves_independent_same_query_executions(
    repo: TraceRecordService, collector: PgCollector
) -> None:
    for tid, score in [("session-A", "0.0"), ("session-B", "2.0")]:
        await _seed(
            repo,
            trace_id=tid,
            batch_tag=None,
            evaluator_name="vui_agent",
            created_at=_BATCH_START,
            query="导航回家",
            tag="",
            score=score,
        )
    records = await collector.fetch_time_window_records(
        start_at=_BATCH_START - timedelta(minutes=1),
        end_at=_BATCH_START + timedelta(minutes=1),
        evaluator_name="vui_agent",
    )
    assert {record.trace_id: record.score for record in records} == {
        "session-A": 0.0,
        "session-B": 2.0,
    }
