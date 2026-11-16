"""Tests for BenchmarkService — CRUD + lifecycle (benchmark_runs summary + run_items cascade)."""

from datetime import UTC, date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.records.enums import BenchmarkStatus
from benchmark.runs.run_helpers import (
    looks_like_daily_run_id,
    parse_benchmark_run_data_date,
)
from benchmark.runs.run_lifecycle import ListRunsQuery
from benchmark.runs.run_service import (
    BenchmarkRunSpec,
    BenchmarkService,
    CompleteRunWrite,
    DailyTrendQuery,
    _extract_display_case_id,
    _extract_display_input,
    _extract_display_output,
    _resolve_report_case_id,
)

#: _rec kwarg field name → ORM row projection attribute name (judge / attribution column names carry a _json suffix).
_REC_FIELD_ALIASES = {"judge": "judge_json", "attribution": "attribution_detail_json"}


def _async_iterable(items):
    """Create an async-iterable mock that yields items for ``async for``."""

    async def _gen():
        for item in items:
            yield item

    obj = MagicMock()
    obj.__aiter__ = MagicMock(return_value=_gen())
    obj.order_by = MagicMock(return_value=obj)
    obj.exclude = MagicMock(return_value=obj)
    return obj


def _case_results_filter(records: list) -> MagicMock:
    """Mock ``BenchmarkRunItem.filter`` for the ``load_case_results`` paging / count chain."""

    class _CountQuery:
        def __init__(self, n: int) -> None:
            self._n = n

        async def count(self) -> int:
            return self._n

    class _Query:
        def __init__(self, rows: list) -> None:
            self._rows = rows
            self._offset = 0
            self._limit: int | None = None

        def exclude(self, **ex_kw: object) -> _CountQuery:
            ex_cid = ex_kw.get("criterion_id")
            filtered = [r for r in self._rows if r.criterion_id != ex_cid] if ex_cid else self._rows
            return _CountQuery(len(filtered))

        async def first(self) -> object | None:
            return self._rows[0] if self._rows else None

        def order_by(self, *args: object) -> "_Query":
            return self

        def offset(self, n: int) -> "_Query":
            self._offset = n
            return self

        def limit(self, n: int) -> "_Query":
            self._limit = n
            return self

        def __await__(self):
            async def _page() -> list:
                start = self._offset
                lim = self._limit
                if lim is None:
                    return self._rows[start:]
                return self._rows[start : start + lim]

            return _page().__await__()

    def _filter(**kwargs: object) -> _Query:
        rows = list(records)
        criterion_id = kwargs.get("criterion_id")
        if criterion_id is not None:
            rows = [r for r in rows if r.criterion_id == criterion_id]
        return _Query(rows)

    return MagicMock(side_effect=_filter)


@pytest.fixture
def service() -> BenchmarkService:
    return BenchmarkService()


class TestCreateRun:
    @pytest.mark.asyncio
    async def test_creates_run_with_initializing_status(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            MockRun.create = AsyncMock()
            persisted = MagicMock()
            persisted.status = BenchmarkStatus.INITIALIZING
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=persisted)
            MockRun.filter = MagicMock(return_value=mock_filter)
            run_id = await service.create_run(
                BenchmarkRunSpec(
                    benchmark_name="test_benchmark",
                    service_name="test_service",
                    dataset_name="test_dataset",
                    project_id="proj-123",
                    total_samples=10,
                    total_evaluators=3,
                ),
            )
            assert run_id
            assert run_id.startswith("bm_test_benchmark_")
            assert len(run_id) <= 128
            # Not a bare 32-char UUID hex
            assert len(run_id) != 32 or not all(c in "0123456789abcdef" for c in run_id)
            MockRun.create.assert_called_once()
            call_kwargs = MockRun.create.call_args.kwargs
            assert call_kwargs["run_id"] == run_id
            assert call_kwargs["benchmark_name"] == "test_benchmark"
            assert call_kwargs["status"] == BenchmarkStatus.INITIALIZING
            assert call_kwargs["total_samples"] == 10
            assert call_kwargs["total_evaluators"] == 3
            assert call_kwargs["completed_evaluators"] == 0
            assert call_kwargs["heartbeat_at"] is not None
            mock_filter.first.assert_awaited()

    @pytest.mark.asyncio
    async def test_create_run_job_id_matches_pattern(self, service: BenchmarkService):
        import re

        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            MockRun.create = AsyncMock()
            persisted = MagicMock()
            persisted.status = BenchmarkStatus.INITIALIZING
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=persisted)
            MockRun.filter = MagicMock(return_value=mock_filter)
            run_id = await service.create_run(
                BenchmarkRunSpec(
                    benchmark_name="example_benchmark",
                    service_name="amap_chat",
                    dataset_name="ds",
                    project_id="p1",
                ),
            )
            assert re.fullmatch(
                r"bm_[A-Za-z0-9_]+_\d{14}_[0-9a-f]{6}",
                run_id,
            )

    @pytest.mark.asyncio
    async def test_create_run_with_config_json(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            MockRun.create = AsyncMock()
            persisted = MagicMock()
            persisted.status = BenchmarkStatus.INITIALIZING
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=persisted)
            MockRun.filter = MagicMock(return_value=mock_filter)
            config = {"rubric": [{"id": "c1"}]}
            await service.create_run(
                BenchmarkRunSpec(
                    benchmark_name="test",
                    service_name="agent",
                    dataset_name="ds",
                    project_id="p1",
                    config_json=config,
                ),
            )
            call_kwargs = MockRun.create.call_args.kwargs
            assert call_kwargs["config_json"]["rubric"] == config["rubric"]
            assert "updated_at" in call_kwargs["config_json"]

    @pytest.mark.asyncio
    async def test_create_run_raises_when_row_missing_after_insert(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            MockRun.create = AsyncMock()
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=None)
            MockRun.filter = MagicMock(return_value=mock_filter)
            with pytest.raises(RuntimeError, match="row missing"):
                await service.create_run(
                    BenchmarkRunSpec(
                        benchmark_name="test",
                        service_name="agent",
                        dataset_name="ds",
                        project_id="p1",
                    ),
                )

    @pytest.mark.asyncio
    async def test_create_run_wraps_missing_table_error(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            MockRun.create = AsyncMock(
                side_effect=Exception('relation "benchmark_runs" does not exist'),
            )
            with pytest.raises(RuntimeError, match="sql/schema.sql"):
                await service.create_run(
                    BenchmarkRunSpec(
                        benchmark_name="test",
                        service_name="agent",
                        dataset_name="ds",
                        project_id="p1",
                    ),
                )


class TestGenerateBenchmarkJobId:
    def test_format_and_sanitize(self):
        import re

        from benchmark.runs.run_service import generate_benchmark_job_id

        job_id = generate_benchmark_job_id("amap-chat/benchmark!")
        assert job_id.startswith("bm_amap_chat_benchmark_")
        assert re.fullmatch(r"bm_[A-Za-z0-9_]+_\d{14}_[0-9a-f]{6}", job_id)
        assert len(job_id) <= 128

    def test_truncates_long_name(self):
        from benchmark.runs.run_service import generate_benchmark_job_id

        long_name = "x" * 200
        job_id = generate_benchmark_job_id(long_name)
        assert len(job_id) <= 128
        assert job_id.startswith("bm_")


class TestLoadCriterionEntriesFromPipeline:
    """load_criterion_entries_from_pipeline: ORM rows + Python aggregation (mock BenchmarkRunItem)."""

    @staticmethod
    def _rec(
        criterion_id: str,
        *,
        score: float | None,
        status: str = "success",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            criterion_id=criterion_id,
            score=score,
            status=status,
        )

    @staticmethod
    def _patch_run_config(config: dict[str, object] | None = None):
        """mock BenchmarkRun.filter().first() → config_json (stop latch)."""
        run = SimpleNamespace(config_json=config or {})
        return patch(
            "benchmark.runs.run_helpers.BenchmarkRun",
            filter=MagicMock(
                return_value=SimpleNamespace(first=AsyncMock(return_value=run)),
            ),
        )

    @pytest.mark.asyncio
    async def test_aggregates_rows_and_fills_missing(self, service: BenchmarkService):
        from benchmark.reporting.framework.schema import Criterion

        rubric = [
            Criterion(
                id="c1",
                name="C1",
                description="d",
                weight=0.5,
                evaluator_ref="e1",
            ),
            Criterion(
                id="c2",
                name="C2",
                description="d",
                weight=0.5,
                evaluator_ref="e2",
            ),
        ]
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            # c1：3 success + 1 failed
            self._rec("c1", score=0.0, status="failed"),
            self._rec("c1", score=0.6),
            self._rec("c1", score=0.9),
            self._rec("c1", score=1.0),
            # A row whose score is None is counted only in sample_count
            self._rec("c1", score=None),
        ]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
            self._patch_run_config(),
        ):
            MockItem.filter = MagicMock(return_value=_async_iterable(records))
            entries = await service.load_criterion_entries_from_pipeline(
                run_id=run_id,
                rubric=rubric,
                pass_threshold=0.6,
            )

        assert len(entries) == 2
        c1 = entries[0]
        assert c1.criterion_id == "c1"
        assert c1.sample_count == 5
        assert c1.failed_count == 1
        assert c1.success_count == 4
        stats = c1.criterion_stats_json
        assert stats is not None
        assert stats["mean"] == pytest.approx(0.625)
        assert stats["median"] == pytest.approx(0.75)
        assert stats["min"] == 0.0
        assert stats["max"] == 1.0
        assert stats["pass_rate"] == pytest.approx(0.6)
        # A criterion with no detail rows in the rubric is filled with zeros
        c2 = entries[1]
        assert c2.criterion_id == "c2"
        assert c2.sample_count == 0
        assert c2.criterion_stats_json is not None
        assert c2.criterion_stats_json["pass_rate"] == 0.0
        MockItem.filter.assert_called_once_with(run_id=run_id)

    @pytest.mark.asyncio
    async def test_external_error_callbacks_count_as_failed(
        self,
        service: BenchmarkService,
    ):
        """External trigger mode: error callbacks (empty criterion_scores)
        should count as failed entries for each criterion in the rubric."""
        from benchmark.reporting.framework.schema import Criterion

        rubric = [
            Criterion(
                id="c1",
                name="C1",
                description="d",
                weight=0.5,
                evaluator_ref="e1",
            ),
            Criterion(
                id="c2",
                name="C2",
                description="d",
                weight=0.5,
                evaluator_ref="e2",
            ),
        ]
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            SimpleNamespace(
                criterion_id="fused",
                score=0.0,
                status="failed",
                judge_json={"error": "timeout", "criterion_scores": {}},
            ),
            SimpleNamespace(
                criterion_id="fused",
                score=0.0,
                status="failed",
                judge_json={"error": "error", "criterion_scores": {}},
            ),
            # [stop-run only] a running prewrite must not be counted (otherwise an un-called-back row would be treated as score 0)
            SimpleNamespace(
                criterion_id="fused",
                score=None,
                status="running",
                judge_json=None,
            ),
        ]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
            self._patch_run_config({"stop_requested_at": "2026-11-04T01:00:00+00:00"}),
        ):
            MockItem.filter = MagicMock(return_value=_async_iterable(records))
            entries = await service.load_criterion_entries_from_pipeline(
                run_id=run_id,
                rubric=rubric,
                pass_threshold=0.8,
            )

        assert len(entries) == 2
        for entry in entries:
            assert entry.sample_count == 2
            assert entry.failed_count == 2
            assert entry.success_count == 0

    @pytest.mark.asyncio
    async def test_running_prewrite_rows_excluded_from_pipeline_stats(
        self,
        service: BenchmarkService,
    ) -> None:
        """stop-run (stop_requested_at) with only running prewrites aggregates to a sample of 0."""
        from benchmark.reporting.framework.schema import Criterion

        rubric = [
            Criterion(
                id="search_ai_card",
                name="综合",
                description="d",
                weight=1.0,
                evaluator_ref="search_ai_card",
            ),
        ]
        records = [
            SimpleNamespace(
                criterion_id="fused",
                score=None,
                status="running",
                judge_json=None,
            ),
            SimpleNamespace(
                criterion_id="fused",
                score=None,
                status="running",
                judge_json={},
            ),
        ]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
            self._patch_run_config({"stop_requested_at": "2026-11-04T01:00:00+00:00"}),
        ):
            MockItem.filter = MagicMock(return_value=_async_iterable(records))
            entries = await service.load_criterion_entries_from_pipeline(
                run_id="search_eval-20261104-m1",
                rubric=rubric,
                pass_threshold=0.8,
            )
        assert entries[0].sample_count == 0
        assert entries[0].failed_count == 0

    @pytest.mark.asyncio
    async def test_running_prewrite_kept_when_not_partial_stop(
        self,
        service: BenchmarkService,
    ) -> None:
        """Not a stop-run: a running row keeps the original semantics (an external error counts as failed)."""
        from benchmark.reporting.framework.schema import Criterion

        rubric = [
            Criterion(
                id="c1",
                name="C1",
                description="d",
                weight=1.0,
                evaluator_ref="e1",
            ),
        ]
        records = [
            SimpleNamespace(
                criterion_id="fused",
                score=None,
                status="running",
                judge_json=None,
            ),
        ]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
            self._patch_run_config(),
        ):
            MockItem.filter = MagicMock(return_value=_async_iterable(records))
            entries = await service.load_criterion_entries_from_pipeline(
                run_id="bm_normal_completed",
                rubric=rubric,
                pass_threshold=0.8,
            )
        assert entries[0].sample_count == 1
        assert entries[0].failed_count == 1


class TestUpdateRunStatus:
    @pytest.mark.asyncio
    async def test_updates_status_and_fields(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.update_run_status(
                "run-1",
                BenchmarkStatus.RUNNING,
                total_samples=50,
                total_evaluators=3,
            )
            mock_filter.update.assert_called_once()
            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["status"] == BenchmarkStatus.RUNNING
            assert call_kwargs["total_samples"] == 50
            assert call_kwargs["total_evaluators"] == 3


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_updates_heartbeat_at(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.heartbeat("run-1")
            mock_filter.update.assert_called_once()
            call_kwargs = mock_filter.update.call_args.kwargs
            assert "heartbeat_at" in call_kwargs
            assert isinstance(call_kwargs["heartbeat_at"], datetime)


class TestUpdateEvaluatorProgress:
    @pytest.mark.asyncio
    async def test_increments_completed_evaluators(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.update_evaluator_progress("run-1")
            mock_filter.update.assert_called_once()


class TestCompleteRun:
    @pytest.mark.asyncio
    async def test_completes_run_with_score_and_metrics(self, service: BenchmarkService):
        existing = MagicMock()
        existing.metrics_json = {
            "criterion_metrics": {"c0": {"mean": 0.5}},
        }
        existing.report_meta_json = None
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=existing)
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            metrics = {"total_samples": 50, "criterion_metrics": {"c1": {"mean": 0.9}}}
            report_meta = {"report_url": "http://oss/report.html"}
            await service.complete_run(
                CompleteRunWrite(
                    run_id="run-1",
                    final_score=0.85,
                    metrics_json=metrics,
                    report_meta_json=report_meta,
                    duration_ms=5000,
                ),
            )
            mock_filter.update.assert_called_once()
            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["status"] == BenchmarkStatus.COMPLETED
            assert call_kwargs["final_score"] == 0.85
            # merge keeps mid-run c0 and adds c1 + total_samples
            assert call_kwargs["metrics_json"]["criterion_metrics"]["c0"]["mean"] == 0.5
            assert call_kwargs["metrics_json"]["criterion_metrics"]["c1"]["mean"] == 0.9
            assert call_kwargs["metrics_json"]["total_samples"] == 50
            assert call_kwargs["report_meta_json"]["report_url"] == report_meta["report_url"]
            assert "updated_at" in call_kwargs["metrics_json"]
            assert call_kwargs["duration_ms"] == 5000
            assert "finished_at" in call_kwargs

    @pytest.mark.asyncio
    async def test_complete_preserves_report_meta_when_omitted(
        self,
        service: BenchmarkService,
    ) -> None:
        existing = MagicMock()
        existing.metrics_json = {}
        existing.report_meta_json = {"report_url": "keep-me"}
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=existing)
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.complete_run(
                CompleteRunWrite(
                    run_id="run-1",
                    final_score=1.0,
                    metrics_json={"total_samples": 1},
                    report_meta_json=None,
                ),
            )
            assert "report_meta_json" not in mock_filter.update.call_args.kwargs


class TestFailRun:
    @pytest.mark.asyncio
    async def test_fails_run_with_reason(self, service: BenchmarkService):
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            item_exclude = MagicMock()
            item_exclude.update = AsyncMock(return_value=0)
            item_filter = MagicMock()
            item_filter.exclude = MagicMock(return_value=item_exclude)
            MockItem.filter = MagicMock(return_value=item_filter)

            await service.fail_run(run_id="run-1", reason="test failure")
            mock_filter.update.assert_called_once()
            call_kwargs = mock_filter.update.call_args.kwargs
            assert call_kwargs["status"] == BenchmarkStatus.FAILED
            assert call_kwargs["error_message"] == "test failure"
            assert "finished_at" in call_kwargs

    @pytest.mark.asyncio
    async def test_fail_run_cascades_to_items(self, service: BenchmarkService):
        """fail_run must mark running items FAILED but preserve success items."""
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            run_filter = MagicMock()
            run_filter.update = AsyncMock(return_value=1)
            MockRun.filter = MagicMock(return_value=run_filter)

            item_exclude = MagicMock()
            item_exclude.update = AsyncMock(return_value=3)
            item_filter = MagicMock()
            item_filter.exclude = MagicMock(return_value=item_exclude)
            MockItem.filter = MagicMock(return_value=item_filter)

            await service.fail_run(run_id="run-1", reason="boom")

            # run marked FAILED
            assert run_filter.update.call_args.kwargs["status"] == BenchmarkStatus.FAILED
            # items cascaded: filter(run_id=...) → exclude(failed, success) → update(FAILED)
            MockItem.filter.assert_called_once_with(run_id="run-1")
            item_filter.exclude.assert_called_once_with(status__in=["failed", "success"])
            item_exclude.update.assert_awaited_once()
            item_update_kwargs = item_exclude.update.call_args.kwargs
            assert item_update_kwargs["status"] == "failed"


class TestMergeCriterionMetrics:
    @pytest.mark.asyncio
    async def test_merges_into_metrics_json(self, service: BenchmarkService):
        existing = MagicMock()
        existing.metrics_json = {
            "criterion_metrics": {"c0": {"mean": 0.5}},
        }
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=existing)
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.merge_criterion_metrics(
                "run-1",
                criterion_id="c1",
                criterion_payload={"mean": 0.9, "sample_count": 10},
            )
            mock_filter.update.assert_called_once()
            updated = mock_filter.update.call_args.kwargs["metrics_json"]
            assert updated["criterion_metrics"]["c0"]["mean"] == 0.5
            assert updated["criterion_metrics"]["c1"]["mean"] == 0.9
            assert "updated_at" in updated

    @pytest.mark.asyncio
    async def test_noop_when_run_missing(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=None)
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.merge_criterion_metrics(
                "missing",
                criterion_id="c1",
                criterion_payload={"mean": 0.1},
            )
            mock_filter.update.assert_not_called()


class TestMergeAttributionAndReportMeta:
    @pytest.mark.asyncio
    async def test_merge_attribution_keeps_criterion_metrics(
        self,
        service: BenchmarkService,
    ) -> None:
        existing = MagicMock()
        existing.metrics_json = {
            "criterion_metrics": {"c0": {"mean": 0.5}},
            "attribution_metrics": {"c0": {"triage_level_distribution": {"good": 1}}},
        }
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=existing)
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.merge_attribution_metrics(
                "run-1",
                attribution_by_id={
                    "fused": {
                        "triage_level_distribution": {"mild_bad": 2},
                        "root_cause_distribution": {},
                    },
                },
            )
            updated = mock_filter.update.call_args.kwargs["metrics_json"]
            assert updated["criterion_metrics"]["c0"]["mean"] == 0.5
            assert updated["attribution_metrics"]["c0"]["triage_level_distribution"] == {
                "good": 1,
            }
            assert updated["attribution_metrics"]["fused"]["triage_level_distribution"] == {
                "mild_bad": 2,
            }

    @pytest.mark.asyncio
    async def test_merge_report_meta_shallow(self, service: BenchmarkService) -> None:
        existing = MagicMock()
        existing.report_meta_json = {"report_url": "http://a", "size": 10}
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=existing)
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.merge_report_meta(
                "run-1",
                patch={"generated_at": "2026-01-01T00:00:00+00:00"},
            )
            updated = mock_filter.update.call_args.kwargs["report_meta_json"]
            assert updated["report_url"] == "http://a"
            assert updated["size"] == 10
            assert updated["generated_at"] == "2026-01-01T00:00:00+00:00"
            assert "updated_at" in updated


class TestMergeMetricsHelpers:
    def test_merge_metrics_json_nested_keys(self) -> None:
        from benchmark.runs.run_service import merge_metrics_json

        out = merge_metrics_json(
            {
                "criterion_metrics": {"a": {"mean": 1}},
                "aggregation_method": "old",
            },
            {
                "criterion_metrics": {"b": {"mean": 2}},
                "aggregation_method": "weighted_avg",
                "total_samples": 3,
            },
        )
        assert out["criterion_metrics"] == {"a": {"mean": 1}, "b": {"mean": 2}}
        assert out["aggregation_method"] == "weighted_avg"
        assert out["total_samples"] == 3
        assert "updated_at" in out

    @pytest.mark.asyncio
    async def test_update_run_status_merges_metrics(
        self,
        service: BenchmarkService,
    ) -> None:
        existing = MagicMock()
        existing.metrics_json = {"criterion_metrics": {"c0": {"mean": 0.1}}}
        existing.report_meta_json = None
        existing.config_json = None
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.first = AsyncMock(return_value=existing)
            mock_filter.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=mock_filter)

            await service.update_run_status(
                "run-1",
                BenchmarkStatus.AGGREGATING,
                metrics_json={"total_samples": 9},
            )
            updated = mock_filter.update.call_args.kwargs["metrics_json"]
            assert updated["criterion_metrics"]["c0"]["mean"] == 0.1
            assert updated["total_samples"] == 9


class TestGetRun:
    @pytest.mark.asyncio
    async def test_returns_run_when_found(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            expected_run = MagicMock()
            mock_first = AsyncMock(return_value=expected_run)
            mock_filter.first = mock_first
            MockRun.filter = MagicMock(return_value=mock_filter)

            result = await service.get_run("run-1")
            assert result is expected_run

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_first = AsyncMock(return_value=None)
            mock_filter.first = mock_first
            MockRun.filter = MagicMock(return_value=mock_filter)

            result = await service.get_run("nonexistent")
            assert result is None


class TestGetRunByIdOrPk:
    @pytest.mark.asyncio
    async def test_prefers_business_run_id(self, service: BenchmarkService):
        expected = MagicMock()
        with patch.object(service, "get_run", AsyncMock(return_value=expected)) as get_run:
            with patch.object(service, "get_run_by_pk", AsyncMock()) as get_pk:
                result = await service.get_run_by_id_or_pk("biz-run")
        assert result is expected
        get_run.assert_awaited_once_with("biz-run")
        get_pk.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_uuid_resolves_run_id_then_reloads(self, service: BenchmarkService):
        """UUID → look up pk to get run_id → then load data with get_run by run_id."""
        pk = "fbc90c19-5923-4de6-9437-78efe17b1d07"
        by_pk = MagicMock(run_id="biz-from-pk")
        by_biz = MagicMock(run_id="biz-from-pk")
        get_run = AsyncMock(side_effect=[None, by_biz])
        with patch.object(service, "get_run", get_run):
            with patch.object(service, "get_run_by_pk", AsyncMock(return_value=by_pk)) as get_pk:
                result = await service.get_run_by_id_or_pk(pk)
        assert result is by_biz
        get_pk.assert_awaited_once()
        assert str(get_pk.await_args.args[0]) == pk
        assert get_run.await_args_list[0].args == (pk,)
        assert get_run.await_args_list[1].args == ("biz-from-pk",)

    @pytest.mark.asyncio
    async def test_returns_none_for_non_uuid_miss(self, service: BenchmarkService):
        with patch.object(service, "get_run", AsyncMock(return_value=None)):
            with patch.object(service, "get_run_by_pk", AsyncMock()) as get_pk:
                result = await service.get_run_by_id_or_pk("not-a-uuid")
        assert result is None
        get_pk.assert_not_awaited()


class TestListRuns:
    @pytest.mark.asyncio
    async def test_lists_runs_by_benchmark_name(self, service: BenchmarkService):
        class _AwaitableList(list):
            def __await__(self):
                async def _coro():
                    return list(self)

                return _coro().__await__()

        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            alive_qs = MagicMock()
            matching = MagicMock(
                benchmark_name="test_bench",
                config_json={"benchmark_services": "test_bench"},
            )
            other = MagicMock(
                benchmark_name="other",
                config_json={"benchmark_services": "other"},
            )
            by_reg = MagicMock(
                benchmark_name="reg-1",
                config_json={
                    "benchmark_services": "test_bench",
                    "agent_registration": {"id": "reg-1", "name": "test_bench"},
                },
            )
            MockRun.filter = MagicMock(return_value=alive_qs)
            alive_qs.order_by = MagicMock(
                return_value=_AwaitableList([matching, other, by_reg]),
            )

            results = await service.list_runs(ListRunsQuery(benchmark_name="test_bench", limit=10))
            assert results == [matching, by_reg]
            # The list must skip soft-deleted rows; the logical name / registrationId is matched on the Python side
            MockRun.filter.assert_called_once_with(deleted_at__isnull=True)
            alive_qs.order_by.assert_called_once_with("-created_at")


class TestDeleteRun:
    @pytest.mark.asyncio
    async def test_soft_delete_marks_deleted_at_by_pk(
        self,
        service: BenchmarkService,
    ):
        """Delete only marks the row: set deleted_at by the row primary key id, do not delete the row, and do not touch details."""
        import uuid

        run_pk = uuid.uuid4()
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock(return_value=1)
            MockRun.filter = MagicMock(return_value=mock_filter)

            assert await service.delete_run(run_pk) is True

            MockRun.filter.assert_called_once_with(
                id=run_pk,
                deleted_at__isnull=True,
            )
            update_kwargs = mock_filter.update.call_args.kwargs
            assert "deleted_at" in update_kwargs
            mock_filter.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_soft_delete_missing_or_deleted_returns_false(
        self,
        service: BenchmarkService,
    ):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            mock_filter = MagicMock()
            mock_filter.update = AsyncMock(return_value=0)
            MockRun.filter = MagicMock(return_value=mock_filter)

            assert await service.delete_run("missing-pk") is False


class TestRecoverStaleRuns:
    @pytest.mark.asyncio
    async def test_no_stale_runs_returns_zero(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            empty_qs = _async_iterable([])
            empty_qs.filter = MagicMock(return_value=empty_qs)
            MockRun.filter = MagicMock(return_value=empty_qs)

            recovered = await service.recover_stale_runs(stale_timeout_s=120)
            assert recovered == 0

    @pytest.mark.asyncio
    async def test_stale_runs_marked_failed(self, service: BenchmarkService):
        stale_run = MagicMock()
        stale_run.run_id = "stale-1"
        stale_run.status = BenchmarkStatus.RUNNING
        stale_run.update_from_dict = MagicMock(return_value=stale_run)
        stale_run.save = AsyncMock()

        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            stale_qs = _async_iterable([stale_run])
            stale_qs.filter = MagicMock(return_value=stale_qs)
            MockRun.filter = MagicMock(return_value=stale_qs)

            item_exclude = MagicMock()
            item_exclude.update = AsyncMock(return_value=0)
            item_filter = MagicMock()
            item_filter.exclude = MagicMock(return_value=item_exclude)
            MockItem.filter = MagicMock(return_value=item_filter)

            recovered = await service.recover_stale_runs(stale_timeout_s=120)
            assert recovered == 1
            stale_run.update_from_dict.assert_called_once()
            stale_run.save.assert_called_once()
            # heartbeat_lt + never_heartbeat filters (the two queries are mutually exclusive on heartbeat_at)
            assert MockRun.filter.call_count == 2

    @pytest.mark.asyncio
    async def test_never_heartbeat_query_uses_created_at(self, service: BenchmarkService):
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            empty_qs = _async_iterable([])
            empty_qs.filter = MagicMock(return_value=empty_qs)
            MockRun.filter = MagicMock(return_value=empty_qs)
            await service.recover_stale_runs(stale_timeout_s=120)
            kwargs_list = [c.kwargs for c in MockRun.filter.call_args_list]
            assert any(
                "heartbeat_at__isnull" in kw and "created_at__lt" in kw for kw in kwargs_list
            )


class TestLoadCaseResults:
    """Tests for BenchmarkService.load_case_results (ORM row projection + Python assembly)."""

    @staticmethod
    def _rec(item_id: str, criterion_id: str, **overrides: object) -> SimpleNamespace:
        """Build an ORM row projection record; optional fields override defaults via kwargs.

        Supports: score / status / reason / sub_scores / judge / input /
        output / attribution / trace_id / trace_url (judge / attribution
        are mapped onto the judge_json / attribution_detail_json columns).
        """
        ns: dict[str, object] = {
            "item_id": item_id,
            "criterion_id": criterion_id,
            "score": None,
            "status": "success",
            "reason": None,
            "sub_scores": None,
            "judge_json": None,
            "input": None,
            "output": None,
            "attribution_detail_json": None,
            "trace_id": None,
            "trace_url": None,
        }
        for key, value in overrides.items():
            ns[_REC_FIELD_ALIASES.get(key, key)] = value
        return SimpleNamespace(**ns)

    @pytest.mark.asyncio
    async def test_groups_rows_by_case_id(self, service: BenchmarkService) -> None:
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "c1",
                score=8.0,
                reason="good",
                judge={"score": "8.0", "reason": "good", "triage_level": "good"},
                input="用户问：附近火锅店",
                output="附近有3家火锅店",
            ),
            self._rec(
                "item_1",
                "c2",
                score=6.0,
                reason="ok",
                judge={"score": "6.0", "reason": "ok"},
                input="用户问：附近火锅店",
                output="附近有3家火锅店",
            ),
            self._rec(
                "item_2",
                "c1",
                score=0.0,
                status="failed",
                reason="err",
                judge={"score": "0.0", "reason": "err", "eval_failed": True},
                input="用户问：天气",
                output="今天晴天",
            ),
            # fused row: case-level attribution
            self._rec(
                "item_1",
                "fused",
                attribution={
                    "triage_level": "good",
                    "root_cause": "hallucination",
                    "confidence": 0.9,
                },
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        assert len(results) == 2
        case1 = next(r for r in results if r["case_id"] == "item_1")
        assert case1["input"] == "用户问：附近火锅店"
        assert case1["output"] == "附近有3家火锅店"
        assert set(case1["scores"].keys()) == {"c1", "c2"}
        assert case1["scores"]["c1"]["score"] == 8.0
        assert case1["scores"]["c2"]["reason"] == "ok"
        case2 = next(r for r in results if r["case_id"] == "item_2")
        assert case1["scores"]["c1"]["judge"]["triage_level"] == "good"
        assert case1["attribution"]["root_cause"] == "hallucination"
        assert case2["input"] == "用户问：天气"
        assert case2["output"] == "今天晴天"
        assert case2["scores"]["c1"]["eval_failed"] is True
        assert "attribution" not in case2
        # inline mode: no fused composite row → do not stamp a grade
        assert "grade" not in case1
        assert "grade" not in case2
        assert MockItem.filter.called

    @pytest.mark.asyncio
    async def test_skips_fused_and_empty_case_id(
        self,
        service: BenchmarkService,
    ) -> None:
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            # item_id is empty: skip
            self._rec("", "c1", score=1.0),
            # fused row has no attribution → it does not enter case-level attribution
            self._rec("item_1", "fused"),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        assert results == []

    @pytest.mark.asyncio
    async def test_extracts_sub_scores_from_first_row(
        self,
        service: BenchmarkService,
    ) -> None:
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "c1",
                score=8.0,
                reason="",
                sub_scores={"axis_a": 7.0, "axis_b": 9.0},
                judge={"score": "8.0", "reason": ""},
                input="query",
                output="response",
            ),
            self._rec(
                "item_1",
                "c2",
                score=6.0,
                reason="",
                sub_scores={"axis_a": 7.0, "axis_b": 9.0},
                judge={"score": "6.0", "reason": ""},
                input="query",
                output="response",
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        assert len(results) == 1
        assert results[0]["input"] == "query"
        assert results[0]["output"] == "response"
        assert results[0]["sub_scores"] == {"axis_a": 7.0, "axis_b": 9.0}
        assert results[0]["scores"]["c1"]["judge"]["score"] == "8.0"

    @pytest.mark.asyncio
    async def test_external_error_callbacks_return_case_details(
        self,
        service: BenchmarkService,
    ) -> None:
        """External trigger mode: error callbacks (empty criterion_scores)
        should still return case results with input/output from fused row."""
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "fused",
                score=0.0,
                status="failed",
                judge={"error": "timeout", "criterion_scores": {}},
                input={"query": "\u9644\u8fd1\u7b2b\u706b\u5e97"},
                output={"response_text": "\u65e0\u54cd\u5e94"},
            ),
            self._rec(
                "item_2",
                "fused",
                score=0.0,
                status="failed",
                judge={"error": "error", "criterion_scores": {}},
                input={"query": "\u5929\u6c14"},
                output={"response_text": "\u65e0\u54cd\u5e94"},
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        assert len(results) == 2
        case1 = next(r for r in results if r["case_id"] == "item_1")
        assert case1["input"] == {"query": "\u9644\u8fd1\u7b2b\u706b\u5e97"}
        assert case1["output"] == {"response_text": "\u65e0\u54cd\u5e94"}
        assert case1["display_input"] == "\u9644\u8fd1\u7b2b\u706b\u5e97"
        assert case1["display_output"] == "\u65e0\u54cd\u5e94"
        # When criterion_scores is empty, the fused entry carries the error reason + judge_json
        assert "fused" in case1["scores"]
        assert case1["scores"]["fused"]["eval_failed"] is True
        assert case1["scores"]["fused"]["score"] == 0.0
        assert "timeout" in case1["scores"]["fused"]["reason"]
        assert case1["scores"]["fused"]["judge"] is not None
        case2 = next(r for r in results if r["case_id"] == "item_2")
        assert case2["input"] == {"query": "天气"}
        assert case2["display_input"] == "天气"
        assert "fused" in case2["scores"]
        assert case2["scores"]["fused"]["eval_failed"] is True
        # An evaluation failure (eval_failed) ≠ grade FAIL → do not stamp a grade
        assert "grade" not in case1
        assert "grade" not in case2

    @pytest.mark.asyncio
    async def test_external_fused_score_populated(
        self,
        service: BenchmarkService,
    ) -> None:
        """External trigger mode: fused row with non-empty criterion_scores
        should populate scores["fused"] with rec.score for the report's
        "composite score" column."""
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "fused",
                score=1.39,
                judge={
                    "score": "1.39",
                    "criterion_id": "fused",
                    "criterion_scores": {
                        "D0": {"score": 1.0, "reason": "Pass"},
                        "D1": {"score": 2.0, "reason": "correct"},
                    },
                },
                input={"query": "附近早餐店"},
                output={"response_text": "七姐早餐店"},
            ),
            # [stop-run only] running prewrite: input/output are present but there is no judge → it must not enter case detail
            self._rec(
                "item_running",
                "fused",
                status="running",
                score=None,
                input={"query": "未评测"},
                output={"response_text": "占位"},
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(
                run_id=run_id,
                config={"stop_requested_at": "2026-11-04T01:00:00+00:00"},
            )

        assert len(results) == 1
        case = results[0]
        assert case["case_id"] == "item_1"
        # per-criterion scores extracted from criterion_scores
        assert set(case["scores"].keys()) == {"D0", "D1", "fused"}
        assert case["scores"]["D0"]["score"] == 1.0
        assert case["scores"]["D1"]["score"] == 2.0
        # fused score from run_items.score (rec.score)
        assert "fused" in case["scores"]
        assert case["scores"]["fused"]["score"] == 1.39
        assert case["scores"]["fused"]["eval_failed"] is False
        assert case["scores"]["fused"]["judge"] is not None
        # 1.39 ∈ [1.0, 1.4) → ACCEPTABLE (rubric_v2 grade)
        assert case["grade"] == "ACCEPTABLE"

    @pytest.mark.asyncio
    async def test_running_prewrite_kept_in_case_results_without_stop(
        self,
        service: BenchmarkService,
    ) -> None:
        """Not a stop-run: a running prewrite row still enters case detail (the special semantics do not spread)."""
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_running",
                "fused",
                status="running",
                score=None,
                input={"query": "未评测"},
                output={"response_text": "占位"},
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        assert len(results) == 1
        assert results[0]["case_id"] == "item_running"

    @pytest.mark.asyncio
    async def test_grade_injected_from_fused_score(
        self,
        service: BenchmarkService,
    ) -> None:
        """A valid fused composite score → map the case grade through rubric_v2_grade."""
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "fused",
                score=1.85,
                judge={
                    "score": "1.85",
                    "criterion_scores": {"D0": {"score": 2.0}},
                },
                input="q1",
                output="r1",
            ),
            self._rec(
                "item_2",
                "fused",
                score=0.0,
                judge={
                    "score": "0.0",
                    "criterion_scores": {"D0": {"score": 0.0}},
                },
                input="q2",
                output="r2",
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        case1 = next(r for r in results if r["case_id"] == "item_1")
        case2 = next(r for r in results if r["case_id"] == "item_2")
        # 1.85 ≥ 1.8 → EXCELLENT
        assert case1["grade"] == "EXCELLENT"
        # 0.0 (the gate overwrites the composite score) → FAIL
        assert case2["grade"] == "FAIL"

    @pytest.mark.asyncio
    async def test_fused_trace_fields_injected_to_case(
        self,
        service: BenchmarkService,
    ) -> None:
        """fused-row trace_id / trace_url → the top level of the case (data source for the report
        "case id" / "trace link" columns, written by the judge callback)."""
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "fused",
                score=1.85,
                judge={
                    "score": "1.85",
                    "criterion_scores": {"D0": {"score": 2.0}},
                },
                input="q1",
                output="r1",
                trace_id="trace-abc-1",
                trace_url="https://trace.example.com/t/1",
            ),
            self._rec(
                "item_2",
                "fused",
                score=1.9,
                judge={
                    "score": "1.9",
                    "criterion_scores": {"D0": {"score": 2.0}},
                },
                input="q2",
                output="r2",
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        case1 = next(r for r in results if r["case_id"] == "item_1")
        case2 = next(r for r in results if r["case_id"] == "item_2")
        assert case1["trace_id"] == "trace-abc-1"
        assert case1["trace_url"] == "https://trace.example.com/t/1"
        assert "trace_id" not in case2
        assert "trace_url" not in case2


class TestBuildRunProgress:
    """build_run_progress: criterion_metrics extraction + run-level counters (mock ORM)."""

    @staticmethod
    def _run() -> MagicMock:
        run = MagicMock()
        run.run_id = "bm_1"
        run.total_samples = 100
        run.evaluated_items = 95
        run.failed_items = 5
        run.metrics_json = {
            "criterion_metrics": {
                "accuracy": {
                    "sample_count": 100,
                    "success_count": 95,
                    "failed_count": 5,
                },
                "safety": {
                    "sample_count": 100,
                    "success_count": 100,
                    "failed_count": 0,
                },
            },
        }
        run.config_json = {"sample_size": 50}
        return run

    @pytest.mark.asyncio
    async def test_aggregates_criterion_metrics_and_run_counters(
        self,
        service: BenchmarkService,
    ):
        progress = await service.build_run_progress(self._run())

        assert progress["total_items"] == 100
        assert progress["evaluated_items"] == 95
        assert progress["failed_items"] == 5
        assert progress["sample_size"] == 50
        assert progress["evaluators"] == [
            {"name": "accuracy", "total_cases": 100, "success": 95, "failed": 5},
            {"name": "safety", "total_cases": 100, "success": 100, "failed": 0},
        ]

    @pytest.mark.asyncio
    async def test_empty_progress_before_any_result(
        self,
        service: BenchmarkService,
    ):
        run = self._run()
        run.metrics_json = None
        run.config_json = None
        run.evaluated_items = 0
        run.failed_items = 0
        progress = await service.build_run_progress(run)

        assert progress["sample_size"] is None
        assert progress["evaluators"] == []
        assert progress["evaluated_items"] == 0
        assert progress["failed_items"] == 0


class TestExtractDisplayIO:
    """Unit tests for _extract_display_input / _extract_display_output."""

    def test_extract_display_input_from_dict(self) -> None:
        raw = {
            "request_body": {"query": "\u9644\u8fd1\u7684\u7b2b\u706b\u5e97", "user": {"uid": "u1"}}
        }
        assert _extract_display_input(raw) == "\u9644\u8fd1\u7684\u7b2b\u706b\u5e97"

    def test_extract_display_input_from_json_string(self) -> None:
        import json

        raw = json.dumps({"request_body": {"query": "\u5929\u6c14\u600e\u4e48\u6837"}})
        assert _extract_display_input(raw) == "\u5929\u6c14\u600e\u4e48\u6837"

    def test_extract_display_input_plain_string(self) -> None:
        assert (
            _extract_display_input("\u7eaf\u6587\u672c\u8f93\u5165")
            == "\u7eaf\u6587\u672c\u8f93\u5165"
        )

    def test_extract_display_input_missing_request_body(self) -> None:
        assert _extract_display_input({"other": "data"}) == ""

    def test_extract_display_input_missing_query(self) -> None:
        assert _extract_display_input({"request_body": {"user": "u1"}}) == ""

    def test_extract_display_input_none(self) -> None:
        assert _extract_display_input(None) == ""

    def test_extract_display_input_empty_string(self) -> None:
        assert _extract_display_input("") == ""

    def test_extract_display_input_external_oss_format(self) -> None:
        """External OSS trigger: query is top-level key."""
        raw = {"query": "\u9644\u8fd1\u5496\u5561\u5e97", "metadata": {"city": "\u5317\u4eac"}}
        assert _extract_display_input(raw) == "\u9644\u8fd1\u5496\u5561\u5e97"

    def test_extract_display_input_prefers_top_level_query(self) -> None:
        """When both top-level query and request_body.query exist, prefer top-level."""
        raw = {"query": "top_level", "request_body": {"query": "nested"}}
        assert _extract_display_input(raw) == "top_level"

    def test_extract_display_output_from_dict(self) -> None:
        raw = {
            "component": {
                "displayText": "\u9644\u8fd1\u67033\u5bb6\u7b2b\u706b\u5e97",
                "type": "container",
            }
        }
        assert _extract_display_output(raw) == "\u9644\u8fd1\u67033\u5bb6\u7b2b\u706b\u5e97"

    def test_extract_display_output_fallback_to_text(self) -> None:
        raw = {"component": {"text": "\u5907\u7528\u6587\u672c"}}
        assert _extract_display_output(raw) == "\u5907\u7528\u6587\u672c"

    def test_extract_display_output_from_json_string(self) -> None:
        import json

        raw = json.dumps({"component": {"displayText": "JSON\u5b57\u7b26\u4e32\u8f93\u5165"}})
        assert _extract_display_output(raw) == "JSON\u5b57\u7b26\u4e32\u8f93\u5165"

    def test_extract_display_output_plain_string(self) -> None:
        assert (
            _extract_display_output("\u7eaf\u6587\u672c\u8f93\u51fa")
            == "\u7eaf\u6587\u672c\u8f93\u51fa"
        )

    def test_extract_display_output_missing_component(self) -> None:
        assert _extract_display_output({"other": "data"}) == ""

    def test_extract_display_output_none(self) -> None:
        assert _extract_display_output(None) == ""

    def test_extract_display_output_empty_string(self) -> None:
        assert _extract_display_output("") == ""

    def test_extract_display_output_prefers_displaytext_over_text(self) -> None:
        raw = {"component": {"text": "\u5907\u7528", "displayText": "\u4f18\u5148"}}
        assert _extract_display_output(raw) == "\u4f18\u5148"

    def test_extract_display_output_result_field(self) -> None:
        """The ieval data-pipeline model output is in the result column (fall back when response_text is absent)."""
        assert (
            _extract_display_output({"result": "\u6a21\u578b\u56de\u7b54"})
            == "\u6a21\u578b\u56de\u7b54"
        )

    def test_extract_display_output_prefers_response_text_over_result(self) -> None:
        """response_text takes priority over result."""
        raw = {"response_text": "\u4f18\u5148", "result": "\u56de\u9000"}
        assert _extract_display_output(raw) == "\u4f18\u5148"


class TestExtractDisplayCaseId:
    """Unit tests for _extract_display_case_id (task_id → the table case-id column)."""

    def test_extract_task_id_from_dict(self) -> None:
        raw = {"task_id": "e81d955acaaf101d", "query": "去关岭县"}
        assert _extract_display_case_id(raw, "item_1") == "e81d955acaaf101d"

    def test_extract_task_id_from_json_string(self) -> None:
        import json

        raw = json.dumps({"task_id": "abc123", "query": "y"})
        assert _extract_display_case_id(raw, "item_1") == "abc123"

    def test_falls_back_when_task_id_missing(self) -> None:
        assert _extract_display_case_id({"query": "z"}, "item_1") == "item_1"

    def test_falls_back_when_task_id_empty(self) -> None:
        assert _extract_display_case_id({"task_id": ""}, "item_1") == "item_1"

    def test_falls_back_on_invalid_json_string(self) -> None:
        assert _extract_display_case_id("not json", "item_1") == "item_1"

    def test_falls_back_on_non_dict(self) -> None:
        assert _extract_display_case_id(None, "item_1") == "item_1"
        assert _extract_display_case_id(42, "item_1") == "item_1"

    def test_task_id_int_coerced_to_str(self) -> None:
        assert _extract_display_case_id({"task_id": 143}, "item_1") == "143"


class TestLoadCaseResultsDisplayIO:
    """Verify load_case_results injects display_input / display_output."""

    @staticmethod
    def _rec(item_id, criterion_id, **overrides):
        """Build an ORM row projection record; optional fields override defaults via kwargs.

        Supports: score / status / reason / sub_scores / judge / input /
        output / attribution (judge / attribution are mapped onto the _json columns).
        """
        ns = {
            "item_id": item_id,
            "criterion_id": criterion_id,
            "score": 8.0,
            "status": "completed",
            "reason": "ok",
            "sub_scores": None,
            "judge_json": {"score": "8.0", "reason": "ok"},
            "input": None,
            "output": None,
            "attribution_detail_json": None,
            "trace_id": None,
            "trace_url": None,
            "source_trace_id": None,
        }
        for key, value in overrides.items():
            ns[_REC_FIELD_ALIASES.get(key, key)] = value
        return SimpleNamespace(**ns)

    @pytest.mark.asyncio
    async def test_display_io_extracted_from_dict_input_output(
        self,
        service: BenchmarkService,
    ) -> None:
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "c1",
                input={"request_body": {"query": "\u9644\u8fd1\u5496\u5561"}},
                output={"component": {"displayText": "\u5496\u5561\u5e97A"}},
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        assert len(results) == 1
        case = results[0]
        assert case["display_input"] == "\u9644\u8fd1\u5496\u5561"
        assert case["display_output"] == "\u5496\u5561\u5e97A"
        # Original input/output preserved for detail modal
        assert isinstance(case["input"], dict)
        assert isinstance(case["output"], dict)

    @pytest.mark.asyncio
    async def test_display_io_empty_for_plain_string_io(
        self,
        service: BenchmarkService,
    ) -> None:
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "c1",
                input="\u7eaf\u6587\u672c\u8f93\u5165",
                output="\u7eaf\u6587\u672c\u8f93\u51fa",
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        assert len(results) == 1
        case = results[0]
        # Plain strings pass through as display values
        assert case["display_input"] == "\u7eaf\u6587\u672c\u8f93\u5165"
        assert case["display_output"] == "\u7eaf\u6587\u672c\u8f93\u51fa"

    def test_resolve_report_case_id_prefers_trace_id(self) -> None:
        assert _resolve_report_case_id({"trace_id": "  tr-1  ", "case_id": "item_1"}) == "tr-1"
        assert _resolve_report_case_id({"trace_id": "", "case_id": "item_2"}) == "item_2"
        assert _resolve_report_case_id({"case_id": "item_3"}) == "item_3"

    @pytest.mark.asyncio
    async def test_display_case_id_uses_trace_id_then_item_id(
        self,
        service: BenchmarkService,
    ) -> None:
        """Report case id: trace_id wins, and when it is missing fall back to case_id (item_id)."""
        run_id = "bm_test_20260101120000_abcdef"
        records = [
            self._rec(
                "item_1",
                "fused",
                trace_id="e81d955acaaf101d",
                score=1.5,
                judge={
                    "score": "1.5",
                    "criterion_scores": {"D0": {"score": 1.5, "reason": "ok"}},
                },
                input={"task_id": "task-should-not-win", "query": "去关岭县"},
                output={"response_text": "路线规划完成"},
            ),
            self._rec(
                "item_2",
                "fused",
                score=1.2,
                judge={
                    "score": "1.2",
                    "criterion_scores": {"D0": {"score": 1.2, "reason": "ok"}},
                },
                input={"request_body": {"query": "无 trace_id"}},
                output={"component": {"displayText": "ok"}},
            ),
        ]
        with patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem:
            MockItem.filter = _case_results_filter(records)
            results = await service.load_case_results(run_id=run_id)

        by_case = {c["case_id"]: c for c in results}
        assert by_case["item_1"]["display_case_id"] == "e81d955acaaf101d"
        assert by_case["item_1"]["trace_id"] == "e81d955acaaf101d"
        assert by_case["item_2"]["display_case_id"] == "item_2"
        assert by_case["item_1"]["case_id"] == "item_1"


class TestGetFusedScore:
    """Verify get_fused_score reads fused row score with None fallback."""

    @pytest.mark.asyncio
    async def test_returns_fused_score(
        self,
        service: BenchmarkService,
    ) -> None:
        row = MagicMock()
        row.score = 1.23
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(return_value=row)
            score = await service.get_fused_score("run1", "item1")
        assert score == 1.23
        _, kwargs = MockItem.get_or_none.call_args
        assert kwargs == {"run_id": "run1", "item_id": "item1", "criterion_id": "fused"}

    @pytest.mark.asyncio
    async def test_missing_row_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(return_value=None)
            assert await service.get_fused_score("run1", "item1") is None

    @pytest.mark.asyncio
    async def test_null_score_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        row = MagicMock()
        row.score = None
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(return_value=row)
            assert await service.get_fused_score("run1", "item1") is None

    @pytest.mark.asyncio
    async def test_query_error_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        """A query exception fails soft and returns None → the dispatch threshold lets it through, so a failed score lookup does not skip dispatch."""
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(side_effect=RuntimeError("db down"))
            assert await service.get_fused_score("run1", "item1") is None


class TestGetFusedQuery:
    """Verify get_fused_query reads fused row input.query with empty fallback."""

    @pytest.mark.asyncio
    async def test_returns_stripped_query(
        self,
        service: BenchmarkService,
    ) -> None:
        row = MagicMock()
        row.input = {"query": "  附近加油站 "}
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(return_value=row)
            assert await service.get_fused_query("run1", "item1") == "附近加油站"

    @pytest.mark.asyncio
    async def test_missing_row_returns_empty(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(return_value=None)
            assert await service.get_fused_query("run1", "item1") == ""

    @pytest.mark.asyncio
    async def test_query_error_returns_empty(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(side_effect=RuntimeError("db down"))
            assert await service.get_fused_query("run1", "item1") == ""


class TestGetServiceName:
    """Verify get_service_name reads run row with None fallback."""

    @pytest.mark.asyncio
    async def test_returns_service_name(
        self,
        service: BenchmarkService,
    ) -> None:
        row = MagicMock()
        row.service_name = "amap_chat"
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRun",
        ) as MockRun:
            MockRun.get_or_none = AsyncMock(return_value=row)
            name = await service.get_service_name("run1")
        assert name == "amap_chat"
        _, kwargs = MockRun.get_or_none.call_args
        assert kwargs == {"run_id": "run1"}

    @pytest.mark.asyncio
    async def test_missing_run_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRun",
        ) as MockRun:
            MockRun.get_or_none = AsyncMock(return_value=None)
            assert await service.get_service_name("run1") is None

    @pytest.mark.asyncio
    async def test_null_service_name_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        row = MagicMock()
        row.service_name = None
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRun",
        ) as MockRun:
            MockRun.get_or_none = AsyncMock(return_value=row)
            assert await service.get_service_name("run1") is None

    @pytest.mark.asyncio
    async def test_query_error_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        """A query exception fails soft and returns None → policy resolution falls back to the default implementation."""
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRun",
        ) as MockRun:
            MockRun.get_or_none = AsyncMock(side_effect=RuntimeError("db down"))
            assert await service.get_service_name("run1") is None


class TestGetTraceUrl:
    """Verify get_trace_url reads fused row trace_url with None fallback."""

    @pytest.mark.asyncio
    async def test_returns_trace_url(
        self,
        service: BenchmarkService,
    ) -> None:
        row = MagicMock()
        row.trace_url = "https://trace.example.com/t/1"
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(return_value=row)
            url = await service.get_trace_url("run1", "item1")
        assert url == "https://trace.example.com/t/1"
        _, kwargs = MockItem.get_or_none.call_args
        assert kwargs == {"run_id": "run1", "item_id": "item1", "criterion_id": "fused"}

    @pytest.mark.asyncio
    async def test_missing_row_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(return_value=None)
            assert await service.get_trace_url("run1", "item1") is None

    @pytest.mark.asyncio
    async def test_null_trace_url_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        row = MagicMock()
        row.trace_url = None
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(return_value=row)
            assert await service.get_trace_url("run1", "item1") is None

    @pytest.mark.asyncio
    async def test_query_error_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        """A query exception fails soft and returns None → the ticket Trace URL displays '-', and dispatch is not blocked."""
        with patch(
            "benchmark.runs.run_helpers.BenchmarkRunItem",
        ) as MockItem:
            MockItem.get_or_none = AsyncMock(side_effect=RuntimeError("db down"))
            assert await service.get_trace_url("run1", "item1") is None


_CN_TZ = timezone(timedelta(hours=8))


class TestAttachDailyReportContext:
    """Verify attach_daily_report_context (idempotent / degrade) and the trend-assembly methods."""

    @pytest.mark.asyncio
    async def test_attaches_trend_and_yesterday_means(
        self,
        service: BenchmarkService,
    ) -> None:
        """Happy path: the trend series + yesterday's dimension mean are injected into spec."""
        spec = {
            "benchmark_name": "bm1",
            "run_id": "bm1-20261102",
            "final_score": 1.5,
            "case_results": [],
            "total_samples": 10,
            "total_failed": 1,
        }
        trend = {"dates": ["08-24", "08-25"], "fused_avg": [1.4, 1.5]}
        with (
            patch.object(
                service,
                "load_daily_trend",
                AsyncMock(return_value=trend),
            ) as mock_trend,
            patch.object(
                service,
                "load_yesterday_criterion_means",
                AsyncMock(return_value={"d1": 1.2}),
            ) as mock_yday,
            patch.object(
                service,
                "load_yesterday_grade_distribution",
                AsyncMock(return_value={"EXCELLENT": 10, "GOOD": 8}),
            ) as mock_ygrades,
        ):
            await service.attach_daily_report_context(spec)
        assert spec["trend_7d"] == trend
        assert spec["yesterday_criterion_means"] == {"d1": 1.2}
        assert spec["yesterday_grade_distribution"] == {"EXCELLENT": 10, "GOOD": 8}
        # Today's point is built from the spec's current values; the anchor is the run_id data date
        query = mock_trend.await_args.args[0]
        assert query.today_point["fused_avg"] == 1.5
        assert query.anchor_date == date(2026, 11, 2)
        assert query.today_point["date"] == "11-02"
        assert mock_yday.await_args.kwargs["benchmark_name"] == "bm1"
        assert mock_yday.await_args.kwargs["anchor_date"] == query.anchor_date
        assert mock_ygrades.await_args.kwargs["benchmark_name"] == "bm1"
        assert mock_ygrades.await_args.kwargs["anchor_date"] == query.anchor_date

    @pytest.mark.asyncio
    async def test_idempotent_when_trend_present(
        self,
        service: BenchmarkService,
    ) -> None:
        """spec already contains a non-empty trend_7d → skip assembly."""
        spec = {
            "benchmark_name": "bm1",
            "trend_7d": {"dates": ["08-25"]},
        }
        with patch.object(
            service,
            "load_daily_trend",
            AsyncMock(),
        ) as mock_trend:
            await service.attach_daily_report_context(spec)
        mock_trend.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_without_benchmark_name(
        self,
        service: BenchmarkService,
    ) -> None:
        """benchmark_name is empty → skip (the same-series runs cannot be located)."""
        spec = {"benchmark_name": "", "final_score": 1.5}
        with patch.object(
            service,
            "load_daily_trend",
            AsyncMock(),
        ) as mock_trend:
            await service.attach_daily_report_context(spec)
        mock_trend.assert_not_called()
        assert "trend_7d" not in spec

    @pytest.mark.asyncio
    async def test_degrades_on_query_error(
        self,
        service: BenchmarkService,
    ) -> None:
        """A query exception degrades the whole step: do not raise, spec has no trend fields, and report generation is unaffected."""
        spec = {"benchmark_name": "bm1", "final_score": 1.5, "case_results": []}
        with patch.object(
            service,
            "load_daily_trend",
            AsyncMock(side_effect=RuntimeError("db down")),
        ):
            await service.attach_daily_report_context(spec)
        assert "trend_7d" not in spec
        assert "yesterday_criterion_means" not in spec

    @pytest.mark.asyncio
    async def test_trend_anchor_fallback_realtime_is_debug_not_warning(
        self,
        service: BenchmarkService,
    ) -> None:
        """A streaming run_id falling back to calendar today is expected → debug, and do not emit WARNING noise."""
        stream_run = "amap_agent_benchmark__reg__ba-abc__ts__prod-stream-20261031"
        spec = {
            "benchmark_name": "amap_agent_benchmark@ba-abc",
            "run_id": stream_run,
            "final_score": 1.2,
            "case_results": [],
        }
        with (
            patch.object(service, "load_daily_trend", AsyncMock(return_value=None)),
            patch.object(
                service,
                "load_yesterday_criterion_means",
                AsyncMock(return_value={}),
            ),
            patch.object(
                service,
                "load_yesterday_grade_distribution",
                AsyncMock(return_value={}),
            ),
            patch("benchmark.runs.run_helpers.logger") as mock_logger,
        ):
            await service.attach_daily_report_context(spec)
        debug_msgs = [
            c.args[0]
            for c in mock_logger.debug.call_args_list
            if c.args and "trend_anchor_fallback_now" in str(c.args[0])
        ]
        warn_msgs = [
            c.args[0]
            for c in mock_logger.warning.call_args_list
            if c.args and "trend_anchor_fallback_now" in str(c.args[0])
        ]
        assert debug_msgs
        assert not warn_msgs

    @pytest.mark.asyncio
    async def test_trend_anchor_fallback_malformed_daily_warns(
        self,
        service: BenchmarkService,
    ) -> None:
        """The daily-run prefix matches but the date is illegal → still WARNING, so a real exception can be investigated."""
        spec = {
            "benchmark_name": "ieval_external",
            "run_id": "ieval_external-20261399",
            "final_score": 1.2,
            "case_results": [],
        }
        with (
            patch.object(service, "load_daily_trend", AsyncMock(return_value=None)),
            patch.object(
                service,
                "load_yesterday_criterion_means",
                AsyncMock(return_value={}),
            ),
            patch.object(
                service,
                "load_yesterday_grade_distribution",
                AsyncMock(return_value={}),
            ),
            patch("benchmark.runs.run_helpers.logger") as mock_logger,
        ):
            await service.attach_daily_report_context(spec)
        warn_msgs = [
            c.args[0]
            for c in mock_logger.warning.call_args_list
            if c.args and "trend_anchor_fallback_now" in str(c.args[0])
        ]
        assert warn_msgs
        assert not any(
            c.args and "trend_anchor_fallback_now" in str(c.args[0])
            for c in mock_logger.debug.call_args_list
        )

    def test_build_today_trend_point(self) -> None:
        """Today's point is computed from the spec's current values: fused mean + grade distribution rates + evaluation success rate."""
        spec = {
            "final_score": 1.5,
            "case_results": [
                {"grade": "EXCELLENT"},
                {"grade": "GOOD"},
                {"grade": "ACCEPTABLE"},
                {"grade": "POOR"},
                {"grade": "FAIL"},
                {},
                "not-a-dict",
            ],
            "total_samples": 10,
            "total_failed": 2,
        }
        point = BenchmarkService._build_today_trend_point(spec)
        assert point is not None
        assert point["fused_avg"] == 1.5
        assert point["excellent_good_rate"] == 40.0  # (E+G)/5
        assert point["poor_fail_rate"] == 40.0  # (P+F)/5
        assert point["task_satisfaction_rate"] == 60.0  # (E+G+A)/5
        assert point["eval_success_rate"] == 80.0  # (10-2)/10
        assert point["date"] == datetime.now(_CN_TZ).strftime("%m-%d")

    def test_build_today_trend_point_uses_anchor_date(self) -> None:
        """The data-date label follows anchor_date, not the calendar today of report emission."""
        point = BenchmarkService._build_today_trend_point(
            {"final_score": 1.2},
            anchor_date=date(2026, 10, 31),
        )
        assert point is not None
        assert point["date"] == "10-31"

    def test_parse_benchmark_run_data_date(self) -> None:
        assert parse_benchmark_run_data_date("ieval_external-20261031", "ieval_external") == date(
            2026, 10, 31
        )
        assert parse_benchmark_run_data_date(
            "ieval_external-20261031-q2",
            "ieval_external@ba-111",
        ) == date(2026, 10, 31)
        assert parse_benchmark_run_data_date("bm_inline_abc", "bm_inline") is None

    def test_looks_like_daily_run_id(self) -> None:
        """A daily-run prefix is True (including an illegal date); streaming / inline are False."""
        assert looks_like_daily_run_id("ieval_external-20261031", "ieval_external")
        assert looks_like_daily_run_id("ieval_external-20261399", "ieval_external")
        assert not looks_like_daily_run_id("bm_inline_abc", "bm_inline")
        assert not looks_like_daily_run_id(
            "amap_agent_benchmark__reg__ba-abc__ts__prod-stream-20261031",
            "amap_agent_benchmark@ba-abc",
        )

    def test_build_today_trend_point_includes_performance(self) -> None:
        """Performance and efficiency metrics are projected into today's trend point (latency / TTFT / Token / ReAct)."""
        point = BenchmarkService._build_today_trend_point(
            {
                "final_score": 1.5,
                "performance_metrics": {
                    "latency": {"p50_ms": 2800.0, "sample_count": 3},
                    "ttft": {"p50_ms": 620.0, "sample_count": 3},
                    "tokens": {"avg_total": 16800.0, "sample_count": 3},
                    "react": {"avg_rounds": 2.3, "sample_count": 3},
                },
            },
        )
        assert point is not None
        assert point["latency_p50_ms"] == 2800.0
        assert point["ttft_p50_ms"] == 620.0
        assert point["token_avg"] == 16800.0
        assert point["react_avg_rounds"] == 2.3

    def test_build_today_trend_point_without_performance(self) -> None:
        """No performance data → the trend point carries no performance keys (the frontend uses that to skip the chart)."""
        point = BenchmarkService._build_today_trend_point({"final_score": 1.0})
        assert point is not None
        assert not set(point) & {"latency_p50_ms", "ttft_p50_ms", "token_avg"}

    def test_build_today_trend_point_invalid_score(self) -> None:
        """final_score is non-numeric (None / illegal str / bool) → no point for today."""
        for bad in (None, "abc", True, False):
            assert BenchmarkService._build_today_trend_point({"final_score": bad}) is None
        assert BenchmarkService._build_today_trend_point({}) is None

    def test_build_today_trend_point_no_grade_no_counts(self) -> None:
        """No grade / no sample count → emit only fused_avg + date."""
        point = BenchmarkService._build_today_trend_point({"final_score": 1.0})
        assert point is not None
        assert point["fused_avg"] == 1.0
        assert "excellent_good_rate" not in point
        assert "task_satisfaction_rate" not in point
        assert "eval_success_rate" not in point

    @pytest.mark.asyncio
    async def test_load_daily_trend_series(self, service: BenchmarkService) -> None:
        """The last-7-days series: the data-date point takes the spec's current values, the previous day comes from fused-row stats, and a missing day is null."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        run_y = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
        )
        today_point = {
            "date": now_cn.strftime("%m-%d"),
            "fused_avg": 1.5,
            "excellent_good_rate": 60.0,
            "poor_fail_rate": 20.0,
            "eval_success_rate": 100.0,
            "task_satisfaction_rate": 80.0,
        }
        # Yesterday's fused rows: 2 success (1.9 EXCELLENT + 1.5 GOOD) + 1 failed
        rows = [
            SimpleNamespace(status="success", score=1.9),
            SimpleNamespace(status="success", score=1.5),
            SimpleNamespace(status="failed", score=None),
        ]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[run_y])
            MockRun.filter = MagicMock(return_value=chain)
            MockItem.filter = AsyncMock(return_value=rows)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name="bm1",
                    today_point=today_point,
                ),
            )
        # Query rule: with no suffix, match only {benchmark_name}-{yyyymmdd} exactly, and do not fall back.
        exact_calls = [c for c in MockRun.filter.call_args_list if "run_id__in" in (c.kwargs or {})]
        assert exact_calls, f"expected run_id__in query, got {MockRun.filter.call_args_list!r}"
        filter_kwargs = exact_calls[0].kwargs
        expected_ids = [
            f"bm1-{(now_cn - timedelta(days=6 - i)).strftime('%Y%m%d')}" for i in range(7)
        ]
        assert filter_kwargs["run_id__in"] == expected_ids
        assert all("-retry" not in rid for rid in filter_kwargs["run_id__in"])
        # The no-suffix path must not issue a prefixed fallback query that carries a suffix
        assert len(MockRun.filter.call_args_list) == 1
        assert trend is not None
        assert len(trend["dates"]) == 7
        # Today's point (the last one) takes the spec's current values
        assert trend["dates"][-1] == now_cn.strftime("%m-%d")
        assert trend["fused_avg"][-1] == 1.5
        assert trend["excellent_good_rate"][-1] == 60.0
        assert trend["task_satisfaction_rate"][-1] == 80.0
        # Yesterday's point comes from fused-row stats: avg=(1.9+1.5)/2; E+G rate=100%;
        # task satisfaction=(E+G+A)/graded=100%; success rate=2/3
        assert trend["dates"][-2] == (now_cn - timedelta(days=1)).strftime("%m-%d")
        assert trend["fused_avg"][-2] == 1.7
        assert trend["excellent_good_rate"][-2] == 100.0
        assert trend["poor_fail_rate"][-2] == 0.0
        assert trend["task_satisfaction_rate"][-2] == 100.0
        assert trend["eval_success_rate"][-2] == 66.7
        # A date with no run is null
        assert trend["fused_avg"][0] is None
        # Yesterday's run has no metrics_json.performance_metrics → that point in the performance series stays empty
        assert trend["latency_p50_ms"][-2] is None

    @pytest.mark.asyncio
    async def test_load_daily_trend_anchors_on_data_date_not_publish_now(
        self,
        service: BenchmarkService,
    ) -> None:
        """A late report: the right edge of the window is the run_id data date, so the score lands on 10-31 rather than calendar today."""
        anchor = date(2026, 10, 31)
        prev = anchor - timedelta(days=1)
        run_prev = SimpleNamespace(
            run_id=f"ieval_external-{prev.strftime('%Y%m%d')}",
            created_at=datetime(2026, 10, 30, 12, 0, tzinfo=UTC),
        )
        today_point = {
            "date": "10-31",
            "fused_avg": 1.42,
            "excellent_good_rate": 50.0,
            "poor_fail_rate": 10.0,
            "eval_success_rate": 100.0,
            "task_satisfaction_rate": 70.0,
        }
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch.object(
                service,
                "_load_run_trend_point",
                AsyncMock(return_value={"fused_avg": 1.1}),
            ),
        ):
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[run_prev])
            MockRun.filter = MagicMock(return_value=chain)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name="ieval_external",
                    today_point=today_point,
                    anchor_date=anchor,
                ),
            )
        exact_calls = [c for c in MockRun.filter.call_args_list if "run_id__in" in (c.kwargs or {})]
        assert exact_calls
        expected_ids = [
            f"ieval_external-{(anchor - timedelta(days=6 - i)).strftime('%Y%m%d')}"
            for i in range(7)
        ]
        assert exact_calls[0].kwargs["run_id__in"] == expected_ids
        assert trend is not None
        assert trend["dates"][-1] == "10-31"
        assert trend["fused_avg"][-1] == 1.42
        assert trend["dates"][-2] == "10-30"
        assert trend["fused_avg"][-2] == 1.1
        # Do not pull the window into the report-emission month (November)
        assert "11-" not in "".join(trend["dates"])

    @pytest.mark.asyncio
    async def test_load_daily_trend_versioned_name_joins_legacy_stem_runs(
        self,
        service: BenchmarkService,
    ) -> None:
        """When reporting stem@reg, join run_id by the stem, and connect it to the old column's bare-stem completed run."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        versioned = "ieval_external@ba-58417e58-c9cb-429e-baba-5367aecaee60"
        run_y = SimpleNamespace(
            run_id=f"ieval_external-{yesterday_ymd}",
            benchmark_name="ieval_external",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
        )
        today_point = {
            "date": now_cn.strftime("%m-%d"),
            "fused_avg": 1.6,
            "excellent_good_rate": 55.0,
            "poor_fail_rate": 15.0,
            "eval_success_rate": 100.0,
            "task_satisfaction_rate": 80.0,
        }
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch.object(
                service,
                "_load_run_trend_point",
                AsyncMock(
                    return_value={
                        "fused_avg": 1.45,
                        "excellent_good_rate": 50.0,
                        "poor_fail_rate": 20.0,
                        "eval_success_rate": 99.0,
                        "task_satisfaction_rate": 75.0,
                    },
                ),
            ),
        ):
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[run_y])
            MockRun.filter = MagicMock(return_value=chain)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name=versioned,
                    today_point=today_point,
                ),
            )
        exact_calls = [c for c in MockRun.filter.call_args_list if "run_id__in" in (c.kwargs or {})]
        assert exact_calls
        filter_kwargs = exact_calls[0].kwargs
        expected_ids = [
            f"ieval_external-{(now_cn - timedelta(days=6 - i)).strftime('%Y%m%d')}"
            for i in range(7)
        ]
        assert filter_kwargs["run_id__in"] == expected_ids
        # Do not join run_id with the full stem@reg
        assert all("@" not in rid for rid in filter_kwargs["run_id__in"])
        # Compatibility Q: bare stem | stem@*
        assert exact_calls[0].args, "expected compatible benchmark_name Q"
        assert trend is not None
        assert trend["fused_avg"][-1] == 1.6
        assert trend["fused_avg"][-2] == 1.45

    @pytest.mark.asyncio
    async def test_load_daily_trend_reads_performance_from_metrics_json(
        self,
        service: BenchmarkService,
    ) -> None:
        """Historical-day performance trends are read straight from ``metrics_json.performance_metrics``, without rescanning details."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        run_y = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
            metrics_json={
                "performance_metrics": {
                    "latency": {"p50_ms": 3100.0, "sample_count": 2},
                    "tokens": {"avg_total": 17400.0, "sample_count": 2},
                },
            },
        )
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[run_y])
            MockRun.filter = MagicMock(return_value=chain)
            MockItem.filter = AsyncMock(return_value=[SimpleNamespace(status="success", score=1.5)])
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name="bm1",
                    today_point={
                        "date": now_cn.strftime("%m-%d"),
                        "fused_avg": 1.5,
                        "latency_p50_ms": 2800.0,
                        "token_avg": 16800.0,
                    },
                ),
            )
        assert trend is not None
        assert trend["latency_p50_ms"][-1] == 2800.0
        assert trend["latency_p50_ms"][-2] == 3100.0
        assert trend["token_avg"][-2] == 17400.0
        # Yesterday's run has no ttft / react → that series is entirely empty, and the frontend does not draw the corresponding line
        assert trend["ttft_p50_ms"][-2] is None
        assert trend["react_avg_rounds"][-2] is None

    @pytest.mark.asyncio
    async def test_merge_performance_metrics_patches_metrics_json(
        self,
        service: BenchmarkService,
    ) -> None:
        """Performance metrics are merged into metrics_json by sub-key, and existing aggregation results are not cleared."""
        run = SimpleNamespace(metrics_json={"criterion_metrics": {"D1": {"mean": 1.5}}})
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            chain = MagicMock()
            chain.first = AsyncMock(return_value=run)
            chain.update = AsyncMock()
            MockRun.filter = MagicMock(return_value=chain)
            await service.merge_performance_metrics(
                "bm1-20260922",
                performance_metrics={"latency": {"p50_ms": 2800.0, "sample_count": 3}},
            )
        merged = chain.update.await_args.kwargs["metrics_json"]
        assert merged["performance_metrics"]["latency"]["p50_ms"] == 2800.0
        assert merged["criterion_metrics"] == {"D1": {"mean": 1.5}}

    @pytest.mark.asyncio
    async def test_merge_performance_metrics_skips_empty(
        self,
        service: BenchmarkService,
    ) -> None:
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            MockRun.filter = MagicMock()
            await service.merge_performance_metrics("bm1-20260922", performance_metrics={})
        MockRun.filter.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_daily_trend_with_suffix_prefers_largest_samples(
        self,
        service: BenchmarkService,
    ) -> None:
        """With a suffix: every suffix on the same day is counted; when several exist on the same day, take the one with the largest total_samples."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        # The earliest created is smoke (very few samples); the later official full run has more samples
        run_smoke = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}-a",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
            total_samples=2,
        )
        run_main = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}-b",
            created_at=(now_cn - timedelta(hours=20)).astimezone(UTC),
            total_samples=200,
        )
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch.object(
                service,
                "_load_run_trend_point",
                AsyncMock(
                    return_value={
                        "fused_avg": 1.5,
                        "excellent_good_rate": 80.0,
                        "poor_fail_rate": 10.0,
                        "eval_success_rate": 90.0,
                        "task_satisfaction_rate": 85.0,
                    },
                ),
            ) as load_point,
        ):
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[run_smoke, run_main])
            MockRun.filter = MagicMock(return_value=chain)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name="bm1",
                    today_point=None,
                    run_id_suffix="q5",
                ),
            )
        # Query rule: the run_id prefix matches any suffix (Q object positional argument)
        args, filter_kwargs = MockRun.filter.call_args
        assert "run_id__in" not in filter_kwargs
        assert args, "带 suffix 时应使用 Q(run_id__startswith=...) 前缀查询"
        assert trend is not None
        assert trend["dates"][-2] == (now_cn - timedelta(days=1)).strftime("%m-%d")
        assert trend["fused_avg"][-2] == 1.5
        # Must take the large-sample main run, not the earliest smoke
        load_point.assert_awaited()
        chosen = load_point.await_args.args[0]
        assert chosen.run_id == run_main.run_id

    @pytest.mark.asyncio
    async def test_load_daily_trend_no_suffix_never_falls_back_to_suffixed(
        self,
        service: BenchmarkService,
    ) -> None:
        """No suffix: when the exact run is missing, stay blank, and never mix in a run that has a suffix."""

        def _filter_side_effect(*_args: object, **kwargs: object) -> MagicMock:
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[])
            return chain

        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch.object(
                service,
                "_load_run_trend_point",
                AsyncMock(
                    return_value={
                        "fused_avg": 1.25,
                        "excellent_good_rate": 80.0,
                        "poor_fail_rate": 10.0,
                        "eval_success_rate": 90.0,
                        "task_satisfaction_rate": 85.0,
                    },
                ),
            ) as load_point,
            patch(
                "benchmark.runs.run_metrics.BenchmarkRunMetrics"
                "._preferred_suffixed_run_by_day",
                new_callable=AsyncMock,
                return_value={},
            ) as preferred,
        ):
            MockRun.filter = MagicMock(side_effect=_filter_side_effect)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name="bm1",
                    today_point=None,
                    run_id_suffix=None,
                ),
            )
        assert trend is None
        load_point.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_load_daily_trend_excludes_mxxx_sxx_suffix_runs(
        self,
        service: BenchmarkService,
    ) -> None:
        """No-suffix main trend: runs with a suffix such as -mxxx / -sxx must not enter the stats."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        versioned = "ieval_external@ba-58417e58-c9cb-429e-baba-5367aecaee60"
        # Simulate the ORM mistakenly returning a suffixed row (defense: even if it is found, it does not enter the series key)
        leak_m = SimpleNamespace(
            run_id=f"ieval_external-{yesterday_ymd}-mxxx",
            benchmark_name=versioned,
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
        )
        leak_s = SimpleNamespace(
            run_id=f"ieval_external-{yesterday_ymd}-sxx",
            benchmark_name="ieval_external",
            created_at=(now_cn - timedelta(hours=20)).astimezone(UTC),
        )

        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch.object(
                service,
                "_load_run_trend_point",
                AsyncMock(
                    return_value={
                        "fused_avg": 1.99,
                        "excellent_good_rate": 100.0,
                        "poor_fail_rate": 0.0,
                        "eval_success_rate": 100.0,
                        "task_satisfaction_rate": 100.0,
                    },
                ),
            ) as load_point,
            patch(
                "benchmark.runs.run_metrics.BenchmarkRunMetrics"
                "._preferred_suffixed_run_by_day",
                new_callable=AsyncMock,
                return_value={},
            ) as preferred,
        ):
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[leak_m, leak_s])
            MockRun.filter = MagicMock(return_value=chain)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name=versioned,
                    today_point={
                        "date": now_cn.strftime("%m-%d"),
                        "fused_avg": 1.5,
                        "excellent_good_rate": 60.0,
                        "poor_fail_rate": 20.0,
                        "eval_success_rate": 100.0,
                        "task_satisfaction_rate": 80.0,
                    },
                    run_id_suffix=None,
                ),
            )
        exact_calls = [c for c in MockRun.filter.call_args_list if "run_id__in" in (c.kwargs or {})]
        assert exact_calls
        queried = exact_calls[0].kwargs["run_id__in"]
        # The query list must be the exact ``ieval_external-YYYYMMDD``, with no -mxxx/-sxx
        for rid in queried:
            assert "@" not in rid
            stem_part, ymd = rid.rsplit("-", 1)
            assert stem_part == "ieval_external"
            assert len(ymd) == 8 and ymd.isdigit()
        preferred.assert_not_awaited()
        load_point.assert_not_awaited()
        assert trend is not None
        # Yesterday has only a leaked suffixed row → that day stays null on the main series
        assert trend["fused_avg"][-2] is None
        assert trend["fused_avg"][-1] == 1.5
        preferred.assert_not_awaited()
        assert MockRun.filter.call_count == 1
        assert "run_id__in" in (MockRun.filter.call_args.kwargs or {})

    @pytest.mark.asyncio
    async def test_load_daily_trend_with_suffix_any_suffix_first_run(
        self,
        service: BenchmarkService,
    ) -> None:
        """With a suffix: when sample counts tie on the same day, keep the earliest created row."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        run_first = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}-a",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
            total_samples=10,
        )
        run_second = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}-b",
            created_at=(now_cn - timedelta(hours=20)).astimezone(UTC),
            total_samples=10,
        )
        rows = [SimpleNamespace(status="success", score=1.5)]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[run_first, run_second])
            MockRun.filter = MagicMock(return_value=chain)
            MockItem.filter = AsyncMock(return_value=rows)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name="bm1",
                    today_point=None,
                    run_id_suffix="q5",
                ),
            )
        # Query rule: the run_id prefix matches any suffix (Q object positional argument)
        args, filter_kwargs = MockRun.filter.call_args
        assert "run_id__in" not in filter_kwargs
        assert args, "带 suffix 时应使用 Q(run_id__startswith=...) 前缀查询"
        # Yesterday's point is taken from the earliest created run (-a) (tied sample counts)
        assert trend is not None
        assert trend["dates"][-2] == (now_cn - timedelta(days=1)).strftime("%m-%d")
        assert trend["fused_avg"][-2] == 1.5

    @pytest.mark.asyncio
    async def test_load_yesterday_criterion_means_with_suffix_any_suffix(
        self,
        service: BenchmarkService,
    ) -> None:
        """With a suffix: yesterday's completed run of any suffix (prefix match, take the main run)."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        run = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}-x9",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
            total_samples=10,
        )
        recs = [
            SimpleNamespace(criterion_id="d1", score=1.5, status="success"),
        ]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            # _completed_run_for_day → _preferred_suffixed_run_by_day：
            # filter().order_by() returns a list (no longer .first())
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[run])
            MockRun.filter = MagicMock(return_value=chain)
            MockItem.filter = AsyncMock(return_value=recs)
            means = await service.load_yesterday_criterion_means(
                benchmark_name="bm1",
                run_id_suffix="q5",
            )
        args, filter_kwargs = MockRun.filter.call_args
        assert args or "run_id__startswith" in filter_kwargs
        assert means == {"d1": 1.5}

    @pytest.mark.asyncio
    async def test_load_yesterday_means_from_stat_dimensions(
        self,
        service: BenchmarkService,
    ) -> None:
        """search_eval: yesterday's means are aggregated by sub_scores dimension key (aligned with the radar axes)."""
        from benchmark.runs.common.dimension_stats import (
            DimensionStatSpec,
            clear_stat_dimensions,
            register_stat_dimensions,
        )

        clear_stat_dimensions()
        register_stat_dimensions(
            "search_eval",
            (
                DimensionStatSpec(
                    key="intent_understanding",
                    name="意图理解",
                    full_score=2.0,
                ),
                DimensionStatSpec(
                    key="relevance",
                    name="内容切题",
                    full_score=2.0,
                ),
            ),
        )
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        run = SimpleNamespace(
            run_id=f"search_eval-{yesterday_ymd}",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
            config_json={},
        )
        recs = [
            SimpleNamespace(
                criterion_id="fused",
                score=1.5,
                status="success",
                judge_json={
                    "criterion_scores": {
                        "search_ai_card": {
                            "score": 1.5,
                            "sub_scores": {
                                "intent_understanding": 2.0,
                                "relevance": 1.0,
                            },
                        }
                    }
                },
            ),
        ]
        try:
            with (
                patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
                patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
            ):
                order_mock = MagicMock()
                order_mock.first = AsyncMock(return_value=run)
                chain = MagicMock()
                chain.order_by = MagicMock(return_value=order_mock)
                MockRun.filter = MagicMock(return_value=chain)
                MockItem.filter = AsyncMock(return_value=recs)
                means = await service.load_yesterday_criterion_means(
                    benchmark_name="search_eval",
                )
            assert means == {"intent_understanding": 2.0, "relevance": 1.0}
            assert "search_ai_card" not in means
        finally:
            clear_stat_dimensions()

    @pytest.mark.asyncio
    async def test_load_daily_trend_no_suffix_missing_day_is_blank(
        self,
        service: BenchmarkService,
    ) -> None:
        """No suffix: when the exact run is missing that day is null, and a same-day suffixed run is not mixed in."""
        now_cn = datetime.now(_CN_TZ)
        today_point = {
            "date": now_cn.strftime("%m-%d"),
            "fused_avg": 1.5,
            "excellent_good_rate": 60.0,
            "poor_fail_rate": 20.0,
            "eval_success_rate": 100.0,
            "task_satisfaction_rate": 80.0,
        }

        def _filter_side_effect(*_args: object, **kwargs: object) -> MagicMock:
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[])
            return chain

        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch.object(
                service,
                "_load_run_trend_point",
                AsyncMock(
                    return_value={
                        "fused_avg": 1.25,
                        "excellent_good_rate": 80.0,
                        "poor_fail_rate": 10.0,
                        "eval_success_rate": 90.0,
                        "task_satisfaction_rate": 85.0,
                    },
                ),
            ) as load_point,
            patch(
                "benchmark.runs.run_metrics.BenchmarkRunMetrics"
                "._preferred_suffixed_run_by_day",
                new_callable=AsyncMock,
                return_value={},
            ) as preferred,
        ):
            MockRun.filter = MagicMock(side_effect=_filter_side_effect)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name="bm1",
                    today_point=today_point,
                    run_id_suffix=None,
                ),
            )
        assert trend is not None
        assert trend["dates"][-2] == (now_cn - timedelta(days=1)).strftime("%m-%d")
        assert trend["fused_avg"][-2] is None
        assert trend["excellent_good_rate"][-2] is None
        assert trend["fused_avg"][-1] == 1.5
        load_point.assert_not_awaited()
        preferred.assert_not_awaited()
        assert MockRun.filter.call_count == 1

    @pytest.mark.asyncio
    async def test_load_yesterday_no_suffix_missing_is_blank(
        self,
        service: BenchmarkService,
    ) -> None:
        """No suffix: when yesterday's exact run is missing, return an empty dict, and do not fall back to a suffix."""
        calls: list[dict[str, object]] = []

        def _filter_side_effect(*_args: object, **kwargs: object) -> MagicMock:
            calls.append(dict(kwargs))
            chain = MagicMock()
            order_mock = MagicMock()
            order_mock.first = AsyncMock(return_value=None)
            chain.order_by = MagicMock(return_value=order_mock)
            return chain

        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
            patch(
                "benchmark.runs.run_metrics.BenchmarkRunMetrics"
                "._preferred_suffixed_run_by_day",
                new_callable=AsyncMock,
                return_value={},
            ) as preferred,
        ):
            MockRun.filter = MagicMock(side_effect=_filter_side_effect)
            MockItem.filter = AsyncMock(return_value=[])
            means = await service.load_yesterday_criterion_means(benchmark_name="bm1")
        assert means == {}
        preferred.assert_not_awaited()
        assert len(calls) == 1
        assert "run_id" in calls[0]
        assert "run_id__startswith" not in calls[0]

    def test_build_today_trend_point_coerces_string_score(
        self,
        service: BenchmarkService,
    ) -> None:
        """When final_score is a numeric string, today's trend point can still be built."""
        point = service._build_today_trend_point(
            {
                "final_score": "1.42",
                "case_results": [],
                "total_samples": 10,
                "total_failed": 1,
            },
        )
        assert point is not None
        assert point["fused_avg"] == 1.42
        assert point["eval_success_rate"] == 90.0

    @pytest.mark.asyncio
    async def test_load_daily_trend_no_data_returns_none(
        self,
        service: BenchmarkService,
    ) -> None:
        """No historical run and no point for today → None (the report omits the trend section)."""
        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            chain = MagicMock()
            chain.order_by = AsyncMock(return_value=[])
            MockRun.filter = MagicMock(return_value=chain)
            trend = await service.load_daily_trend(
                DailyTrendQuery(
                    benchmark_name="bm1",
                    today_point=None,
                ),
            )
        assert trend is None

    @pytest.mark.asyncio
    async def test_load_yesterday_criterion_means(
        self,
        service: BenchmarkService,
    ) -> None:
        """inline mode: read the mean from per-criterion rows, and skip the fused row."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        run = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
        )
        recs = [
            SimpleNamespace(criterion_id="d1", score=1.5, status="success"),
            SimpleNamespace(criterion_id="d1", score=1.7, status="success"),
            SimpleNamespace(criterion_id="d2", score=0.5, status="success"),
            # The fused row + non-success rows: neither is counted
            SimpleNamespace(criterion_id="fused", score=1.9, status="success"),
            SimpleNamespace(criterion_id="d3", score=2.0, status="failed"),
        ]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            first_mock = AsyncMock(return_value=run)
            order_mock = MagicMock()
            order_mock.first = first_mock
            chain = MagicMock()
            chain.order_by = MagicMock(return_value=order_mock)
            MockRun.filter = MagicMock(return_value=chain)
            MockItem.filter = AsyncMock(return_value=recs)
            means = await service.load_yesterday_criterion_means(benchmark_name="bm1")
        # Query rule: run_id matches {benchmark_name}-{yyyymmdd} exactly,
        # and a same-day rerun that carries run_id_suffix is not in the query set
        _, filter_kwargs = MockRun.filter.call_args
        assert filter_kwargs["run_id"] == f"bm1-{yesterday_ymd}"
        assert means == {"d1": 1.6, "d2": 0.5}

    @pytest.mark.asyncio
    async def test_load_yesterday_criterion_means_external(
        self,
        service: BenchmarkService,
    ) -> None:
        """external mode: there is no per-criterion row, so extract from the fused row's judge_json.criterion_scores."""
        now_cn = datetime.now(_CN_TZ)
        yesterday_ymd = (now_cn - timedelta(days=1)).strftime("%Y%m%d")
        run = SimpleNamespace(
            run_id=f"bm1-{yesterday_ymd}",
            created_at=(now_cn - timedelta(days=1)).astimezone(UTC),
        )
        recs = [
            SimpleNamespace(
                criterion_id="fused",
                score=1.9,
                status="success",
                judge_json={
                    "criterion_scores": {
                        "d1": {"score": 1.5},
                        "d2": {"score": 0.7},
                    },
                },
            ),
            SimpleNamespace(
                criterion_id="fused",
                score=1.7,
                status="success",
                judge_json={
                    "criterion_scores": {
                        "d1": {"score": 1.7},
                        "d2": {"score": 0.5},
                    },
                },
            ),
            # An error callback row: a 0 inside criterion_scores is not counted, and a valid score is counted as usual
            SimpleNamespace(
                criterion_id="fused",
                score=0.0,
                status="failed",
                judge_json={
                    "error": "judge timeout",
                    "criterion_scores": {
                        "d1": {"score": 0.0},
                        "d2": {"score": 0.3},
                    },
                },
            ),
            # A fused row with no criterion_scores (defense: skip it and do not error)
            SimpleNamespace(
                criterion_id="fused",
                score=None,
                status="failed",
                judge_json=None,
            ),
        ]
        with (
            patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun,
            patch("benchmark.runs.run_helpers.BenchmarkRunItem") as MockItem,
        ):
            first_mock = AsyncMock(return_value=run)
            order_mock = MagicMock()
            order_mock.first = first_mock
            chain = MagicMock()
            chain.order_by = MagicMock(return_value=order_mock)
            MockRun.filter = MagicMock(return_value=chain)
            MockItem.filter = AsyncMock(return_value=recs)
            means = await service.load_yesterday_criterion_means(benchmark_name="bm1")
        # d1: (1.5+1.7)/2=1.6; d2: (0.7+0.5+0.3)/3=0.5; the error row's d1 score of 0 is not counted
        assert means == {"d1": 1.6, "d2": 0.5}

    @pytest.mark.asyncio
    async def test_load_yesterday_criterion_means_no_run(
        self,
        service: BenchmarkService,
    ) -> None:
        """No completed run yesterday → an empty dict (the radar draws today only)."""

        def _filter_side_effect(*_args: object, **kwargs: object) -> MagicMock:
            chain = MagicMock()
            if "run_id" in kwargs:
                order_mock = MagicMock()
                order_mock.first = AsyncMock(return_value=None)
                chain.order_by = MagicMock(return_value=order_mock)
            else:
                chain.order_by = AsyncMock(return_value=[])
            return chain

        with patch("benchmark.runs.run_helpers.BenchmarkRun") as MockRun:
            MockRun.filter = MagicMock(side_effect=_filter_side_effect)
            means = await service.load_yesterday_criterion_means(benchmark_name="bm1")
        assert means == {}

    @pytest.mark.asyncio
    async def test_load_yesterday_grade_distribution(
        self,
        service: BenchmarkService,
    ) -> None:
        """Yesterday's successful fused scores are counted into the grade-band distribution by rubric_v2."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        run = SimpleNamespace(run_id="bm1-20260917")
        rows = [
            SimpleNamespace(status="success", score=2.0),  # EXCELLENT
            SimpleNamespace(status="success", score=1.6),  # GOOD
            SimpleNamespace(status="success", score=1.6),  # GOOD
            SimpleNamespace(status="failed", score=0.0),
        ]
        with (
            patch.object(service, "_completed_run_for_day", AsyncMock(return_value=run)),
            patch(
                "benchmark.runs.run_helpers.BenchmarkRunItem.filter",
                AsyncMock(return_value=rows),
            ),
        ):
            dist = await service.load_yesterday_grade_distribution(benchmark_name="bm1")
        assert dist == {"EXCELLENT": 1, "GOOD": 2}

    @pytest.mark.asyncio
    async def test_load_yesterday_grade_distribution_no_run(
        self,
        service: BenchmarkService,
    ) -> None:
        from unittest.mock import AsyncMock, patch

        with patch.object(service, "_completed_run_for_day", AsyncMock(return_value=None)):
            dist = await service.load_yesterday_grade_distribution(benchmark_name="bm1")
        assert dist == {}

    def test_normalize_grade_distribution_accepts_floats(self) -> None:
        """A float count, which is common after JSON deserialization, should be rounded to int."""
        from benchmark.runs.run_metrics import BenchmarkRunMetrics

        assert BenchmarkRunMetrics.normalize_grade_distribution(
            {"EXCELLENT": 10.0, "GOOD": 7.6, "POOR": "2", "bad": None},
        ) == {"EXCELLENT": 10, "GOOD": 8, "POOR": 2}


@pytest.mark.asyncio
async def test_attach_trend_propagates_run_id_suffix() -> None:
    """A run_id with a suffix → trend / yesterday stats look only at the series with the same suffix."""
    from unittest.mock import AsyncMock, patch

    from benchmark.runs.run_service import BenchmarkService

    service = BenchmarkService()
    spec = {
        "benchmark_name": "search_eval",
        "run_id": "search_eval-20260906-q5",
        "final_score": 1.5,
        "case_results": [],
    }
    with (
        patch.object(service, "load_daily_trend", AsyncMock(return_value=None)) as m_trend,
        patch.object(
            service,
            "load_yesterday_criterion_means",
            AsyncMock(return_value={}),
        ) as m_yday,
        patch.object(
            service,
            "load_yesterday_grade_distribution",
            AsyncMock(return_value={}),
        ),
    ):
        await service.attach_daily_report_context(spec)
    assert m_trend.await_args.args[0].run_id_suffix == "q5"
    assert m_yday.await_args.kwargs["run_id_suffix"] == "q5"


@pytest.mark.asyncio
async def test_attach_trend_no_suffix_keeps_none() -> None:
    """A run_id with no suffix → suffix=None (look only at unsuffixed scheduled runs; current behavior unchanged)."""
    from unittest.mock import AsyncMock, patch

    from benchmark.runs.run_service import BenchmarkService

    service = BenchmarkService()
    spec = {
        "benchmark_name": "ieval_external",
        "run_id": "ieval_external-20260906",
        "final_score": 1.5,
        "case_results": [],
    }
    with (
        patch.object(service, "load_daily_trend", AsyncMock(return_value=None)) as m_trend,
        patch.object(
            service,
            "load_yesterday_criterion_means",
            AsyncMock(return_value={}),
        ),
        patch.object(
            service,
            "load_yesterday_grade_distribution",
            AsyncMock(return_value={}),
        ),
    ):
        await service.attach_daily_report_context(spec)
    assert m_trend.await_args.args[0].run_id_suffix is None
