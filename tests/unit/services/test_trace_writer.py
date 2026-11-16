"""Unit tests for TraceRecordWriter abstraction — Protocol, Daily, and Benchmark implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.services.trace_writer import (
    ATTRIBUTION_LABEL_PREFIX_FIELDS,
    BenchmarkTraceRecordWriter,
    DailyTraceRecordWriter,
    RunMeta,
    ScoreWrite,
    TraceRecordWriter,
    TriageUpsert,
    TriageWrite,
    build_attribution_value_labels,
)

# ── 1. Protocol conformance ──────────────────────────────────────────────


class TestProtocolConformance:
    """Verify both writers conform to TraceRecordWriter Protocol."""

    def test_daily_writer_is_trace_record_writer(self) -> None:
        service = MagicMock()
        writer = DailyTraceRecordWriter(service)
        assert isinstance(writer, TraceRecordWriter)

    def test_benchmark_writer_is_trace_record_writer(self) -> None:
        writer = BenchmarkTraceRecordWriter(run_id="r1", criterion_id="c1")
        assert isinstance(writer, TraceRecordWriter)


# ── 2. DailyTraceRecordWriter ─────────────────────────────────────────────


class TestDailyTraceRecordWriter:
    """Verify DailyTraceRecordWriter delegates to TraceRecordService."""

    def test_init_stores_service(self) -> None:
        service = MagicMock()
        writer = DailyTraceRecordWriter(service)
        assert writer._service is service

    @pytest.mark.asyncio
    async def test_upsert_score_calls_service_upsert(self) -> None:
        service = MagicMock()
        service.upsert = AsyncMock(return_value=MagicMock(id="rec-123"))
        writer = DailyTraceRecordWriter(service)

        await writer.upsert_score(
            ScoreWrite(
                trace_id="t-1",
                evaluator_ref="judge_a",
                score=0.85,
                judge_json={"score": "0.85", "reason": "good"},
            ),
            project_id="proj-1",
            agent_name="agent-x",
            scene="non_voice",
        )

        service.upsert.assert_called_once()
        key = service.upsert.call_args.args[0]
        call_kwargs = service.upsert.call_args.kwargs
        assert key.trace_id == "t-1"
        assert key.evaluator_name == "judge_a"
        assert key.project_id == "proj-1"
        assert key.agent_name == "agent-x"
        assert call_kwargs["status"] == "scored"
        assert call_kwargs["scene"] == "non_voice"

    @pytest.mark.asyncio
    async def test_upsert_score_returns_record(self) -> None:
        service = MagicMock()
        mock_record = MagicMock(id="rec-456")
        service.upsert = AsyncMock(return_value=mock_record)
        writer = DailyTraceRecordWriter(service)

        record = await writer.upsert_score(
            ScoreWrite(
                trace_id="t-2",
                evaluator_ref="judge_b",
                score=0.9,
                judge_json={"score": "0.9"},
            ),
            project_id="p1",
            agent_name="a1",
        )
        assert record is mock_record

    @pytest.mark.asyncio
    async def test_upsert_triage_looks_up_record(self) -> None:
        service = MagicMock()
        mock_record = MagicMock(id="rec-789")
        service.get_by_trace_id = AsyncMock(return_value=mock_record)
        service.update_triage_level = AsyncMock()
        service.update_attribution = AsyncMock()
        writer = DailyTraceRecordWriter(service)

        await writer.upsert_triage(
            TriageUpsert(
                trace_id="t-3",
                evaluator_ref="judge_a",
                triage=TriageWrite(
                    triage_level="severe_bad",
                    root_cause="intent_error",
                    attribution_detail_json={
                        "owner_domain": "nlu",
                        "owner_group": "team-a",
                        "confidence": 0.9,
                    },
                ),
            ),
            project_id="p1",
            agent_name="a1",
        )

        service.get_by_trace_id.assert_called_once()
        service.update_triage_level.assert_called_once()
        assert service.update_triage_level.call_args.kwargs["record_id"] == "rec-789"
        service.update_attribution.assert_called_once()
        update = service.update_attribution.call_args.kwargs["update"]
        assert update.root_cause == "intent_error"

    @pytest.mark.asyncio
    async def test_upsert_triage_record_not_found(self) -> None:
        service = MagicMock()
        service.get_by_trace_id = AsyncMock(return_value=None)
        service.update_triage_level = AsyncMock()
        service.update_attribution = AsyncMock()
        writer = DailyTraceRecordWriter(service)

        await writer.upsert_triage(
            TriageUpsert(
                trace_id="missing",
                evaluator_ref="judge_a",
                triage=TriageWrite(
                    triage_level="good",
                    root_cause=None,
                    attribution_detail_json={},
                ),
            ),
            project_id="p1",
            agent_name="a1",
        )

        service.update_triage_level.assert_not_called()
        service.update_attribution.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_triage_empty_triage_level_skips_level_write(self) -> None:
        service = MagicMock()
        mock_record = MagicMock(id="rec-1")
        service.get_by_trace_id = AsyncMock(return_value=mock_record)
        service.update_triage_level = AsyncMock()
        service.update_attribution = AsyncMock()
        writer = DailyTraceRecordWriter(service)

        await writer.upsert_triage(
            TriageUpsert(
                trace_id="t-1",
                evaluator_ref="j1",
                triage=TriageWrite(
                    triage_level="",
                    root_cause="error",
                    attribution_detail_json={"owner_domain": "x"},
                ),
            ),
            project_id="p1",
            agent_name="a1",
        )

        service.update_triage_level.assert_not_called()
        service.update_attribution.assert_called_once()


# ── 3. BenchmarkTraceRecordWriter (benchmark_run_items) ──────────────────


class TestBenchmarkTraceRecordWriter:
    """Verify BenchmarkTraceRecordWriter persists to benchmark_run_items."""

    def test_init_stores_context(self) -> None:
        writer = BenchmarkTraceRecordWriter(
            run_id="run-1",
            criterion_id="crit-1",
            meta=RunMeta(
                benchmark_name="bench",
                service_name="agent",
                dataset_name="ds",
                project_id="proj",
            ),
        )
        assert writer._run_id == "run-1"
        assert writer._criterion_id == "crit-1"
        assert writer._benchmark_name == "bench"
        assert writer._service_name == "agent"
        assert writer._dataset_name == "ds"
        assert writer._project_id == "proj"

    @pytest.mark.asyncio
    @patch("benchmark.services.trace_writer.BenchmarkRunItem")
    async def test_upsert_score_writes_item_row(
        self,
        mock_model: MagicMock,
    ) -> None:
        mock_record = MagicMock()
        mock_model.update_or_create = AsyncMock(
            return_value=(mock_record, True),
        )
        writer = BenchmarkTraceRecordWriter(
            run_id="run-1",
            criterion_id="crit-1",
            meta=RunMeta(
                benchmark_name="bench",
                service_name="agent",
                dataset_name="ds",
                project_id="proj",
            ),
        )

        result = await writer.upsert_score(
            ScoreWrite(
                trace_id="source-trace-1",
                evaluator_ref="judge_a",
                score=0.75,
                judge_json={"score": "0.75", "reason": "ok"},
                dataset_item_id="item-1",
            ),
            project_id="proj-1",
        )

        mock_model.update_or_create.assert_called_once()
        call_args = mock_model.update_or_create.call_args
        assert call_args.kwargs["run_id"] == "run-1"
        assert call_args.kwargs["item_id"] == "item-1"
        assert call_args.kwargs["criterion_id"] == "crit-1"
        defaults = call_args.kwargs["defaults"]
        assert defaults["evaluator_ref"] == "judge_a"
        assert defaults["score"] == 0.75
        assert defaults["status"] == "success"
        assert defaults["judge_json"]["score"] == "0.75"
        assert defaults["judge_json"]["benchmark_run_id"] == "run-1"
        assert defaults["source_trace_id"] == "source-trace-1"
        assert result is mock_record

    @pytest.mark.asyncio
    @patch("benchmark.services.trace_writer.BenchmarkRunItem")
    async def test_upsert_score_extracts_io_from_trace_meta(
        self,
        mock_model: MagicMock,
    ) -> None:
        """Writer extracts input/output from trace_meta_json into structured columns."""
        mock_record = MagicMock()
        mock_model.update_or_create = AsyncMock(
            return_value=(mock_record, True),
        )
        writer = BenchmarkTraceRecordWriter(
            run_id="run-1",
            criterion_id="crit-1",
            meta=RunMeta(service_name="agent"),
        )
        await writer.upsert_score(
            ScoreWrite(
                trace_id="t1",
                evaluator_ref="judge_a",
                score=1.0,
                judge_json={"score": "1.0", "reason": "ok"},
                dataset_item_id="item-1",
            ),
            trace_meta_json={
                "source": "dataset",
                "dataset": {"name": "ds", "item_id": "item-1"},
                "input": "question",
                "output": "answer",
            },
        )
        defaults = mock_model.update_or_create.call_args.kwargs["defaults"]
        assert defaults["input"] == "question"
        assert defaults["output"] == "answer"

    @pytest.mark.asyncio
    @patch("benchmark.services.trace_writer.BenchmarkRunItem")
    async def test_upsert_triage_writes_attribution(
        self,
        mock_model: MagicMock,
    ) -> None:
        """upsert_triage with real attribution updates existing row."""
        mock_record = MagicMock()
        mock_model.update_or_create = AsyncMock(
            return_value=(mock_record, False),
        )
        # Simulate an existing row from a prior upsert_score call.
        mock_existing = MagicMock()
        mock_existing.input = "question"
        mock_existing.output = "answer"
        mock_existing.score = 0.5
        mock_existing.status = "success"
        mock_existing.source_trace_id = "trace-1"
        mock_filter = MagicMock()
        mock_filter.first = AsyncMock(return_value=mock_existing)
        mock_model.filter = MagicMock(return_value=mock_filter)

        writer = BenchmarkTraceRecordWriter(
            run_id="run-2",
            criterion_id="fused",
            meta=RunMeta(service_name="agent"),
        )

        await writer.upsert_triage(
            TriageUpsert(
                trace_id="source-2",
                evaluator_ref="benchmark_fused",
                triage=TriageWrite(
                    triage_level="mild_bad",
                    root_cause="tool_selection",
                    attribution_detail_json={
                        "category": "tools",
                        "agents": ["agent-a"],
                    },
                ),
                dataset_item_id="item-2",
            ),
        )

        mock_model.filter.assert_called_once_with(
            run_id="run-2",
            item_id="item-2",
            criterion_id="fused",
        )
        call_args = mock_model.update_or_create.call_args
        assert call_args.kwargs["run_id"] == "run-2"
        assert call_args.kwargs["item_id"] == "item-2"
        assert call_args.kwargs["criterion_id"] == "fused"
        defaults = call_args.kwargs["defaults"]
        assert defaults["evaluator_ref"] == "benchmark_fused"
        assert defaults["attribution_detail_json"]["triage_level"] == "mild_bad"
        assert defaults["attribution_detail_json"]["root_cause"] == "tool_selection"
        # IO preserved from existing row
        assert defaults["input"] == "question"
        assert defaults["output"] == "answer"

    @pytest.mark.asyncio
    @patch("benchmark.services.trace_writer.BenchmarkRunItem")
    async def test_upsert_triage_no_existing_row_skips(
        self,
        mock_model: MagicMock,
    ) -> None:
        """upsert_triage without a prior upsert_score row should skip."""
        mock_model.update_or_create = AsyncMock()
        mock_filter = MagicMock()
        mock_filter.first = AsyncMock(return_value=None)
        mock_model.filter = MagicMock(return_value=mock_filter)

        writer = BenchmarkTraceRecordWriter(
            run_id="run-x",
            criterion_id="c1",
        )
        result = await writer.upsert_triage(
            TriageUpsert(
                trace_id="t",
                evaluator_ref="j",
                triage=TriageWrite(
                    triage_level="mild_bad",
                    root_cause="err",
                    attribution_detail_json={"category": "x"},
                ),
                dataset_item_id="i",
            ),
        )
        assert result is None
        mock_model.update_or_create.assert_not_called()

    @pytest.mark.asyncio
    @patch("benchmark.services.trace_writer.BenchmarkRunItem")
    async def test_upsert_triage_empty_attribution_skips_write(
        self,
        mock_model: MagicMock,
    ) -> None:
        """Before-hook empty attribution must not write attribution_detail_json."""
        mock_record = MagicMock()
        mock_model.update_or_create = AsyncMock(
            return_value=(mock_record, False),
        )
        mock_existing = MagicMock()
        mock_existing.input = "q"
        mock_existing.output = "a"
        mock_existing.score = 0.8
        mock_existing.status = "success"
        mock_existing.source_trace_id = "t"
        mock_filter = MagicMock()
        mock_filter.first = AsyncMock(return_value=mock_existing)
        mock_model.filter = MagicMock(return_value=mock_filter)

        writer = BenchmarkTraceRecordWriter(
            run_id="run-e",
            criterion_id="c1",
        )
        await writer.upsert_triage(
            TriageUpsert(
                trace_id="t",
                evaluator_ref="j",
                triage=TriageWrite(
                    triage_level="good",
                    root_cause=None,
                    attribution_detail_json={},
                ),
                dataset_item_id="i",
            ),
        )
        defaults = mock_model.update_or_create.call_args.kwargs["defaults"]
        # attribution_detail_json must NOT be in defaults (left as None)
        assert "attribution_detail_json" not in defaults
        assert defaults["evaluator_ref"] == "j"

    @pytest.mark.asyncio
    @patch("benchmark.services.trace_writer.BenchmarkRunItem")
    async def test_aggregate_attribution_from_items(
        self,
        mock_model: MagicMock,
    ) -> None:
        fused = MagicMock()
        fused.criterion_id = "fused"
        fused.attribution_detail_json = {
            "primary_issue_category": "intent_error",
            "primary_issue_layer": "planning",
            "root_cause": "tool_selection",
        }

        mock_qs = MagicMock()
        mock_qs.exclude = AsyncMock(return_value=[fused])
        mock_model.filter = MagicMock(return_value=mock_qs)

        # Default field: only primary_issue_category
        attr = await BenchmarkTraceRecordWriter.aggregate_attribution("run-2")
        assert "fused" in attr
        assert attr["fused"]["field_distributions"]["primary_issue_category"] == {"intent_error": 1}
        assert "root_cause" not in attr["fused"]["field_distributions"]
        assert attr["fused"]["total_attributed"] == 1
        # value_labels: the layer/category display label, and the stats key is unchanged
        assert attr["fused"]["value_labels"]["primary_issue_category"] == {
            "intent_error": "planning/intent_error",
        }

        # Custom fields: the mechanism does not limit the field set (a field with no prefix mapping carries no value_labels)
        attr = await BenchmarkTraceRecordWriter.aggregate_attribution(
            "run-2",
            fields=("root_cause",),
        )
        assert attr["fused"]["field_distributions"]["root_cause"] == {
            "tool_selection": 1,
        }
        assert "root_cause" not in attr["fused"].get("value_labels", {})

    @pytest.mark.asyncio
    @patch("benchmark.services.trace_writer.BenchmarkRunItem")
    async def test_aggregate_attribution_empty_maps_unknown(
        self,
        mock_model: MagicMock,
    ) -> None:
        """An empty value / a missing field is mapped to unknown."""
        fused = MagicMock()
        fused.criterion_id = "fused"
        fused.item_id = "item-1"
        fused.attribution_detail_json = {
            "primary_issue_category": "",
        }
        fused2 = MagicMock()
        fused2.criterion_id = "fused"
        fused2.item_id = "item-2"
        fused2.attribution_detail_json = {
            # The primary_issue_category field is missing
            "root_cause": "tool_selection",
        }
        fused3 = MagicMock()
        fused3.criterion_id = "fused"
        fused3.item_id = "item-3"
        fused3.attribution_detail_json = {
            "primary_issue_category": "intent_error",
        }

        mock_qs = MagicMock()
        mock_qs.exclude = AsyncMock(return_value=[fused, fused2, fused3])
        mock_model.filter = MagicMock(return_value=mock_qs)

        attr = await BenchmarkTraceRecordWriter.aggregate_attribution("run-x")
        dist = attr["fused"]["field_distributions"]["primary_issue_category"]
        assert dist["unknown"] == 2
        assert dist["intent_error"] == 1
        assert attr["fused"]["total_attributed"] == 3

    @pytest.mark.asyncio
    @patch("benchmark.services.trace_writer.BenchmarkRunItem")
    async def test_delete_pipeline_records(
        self,
        mock_model: MagicMock,
    ) -> None:
        mock_qs = MagicMock()
        mock_qs.delete = AsyncMock(return_value=3)
        mock_model.filter = MagicMock(return_value=mock_qs)

        n = await BenchmarkTraceRecordWriter.delete_pipeline_records("run-9")
        assert n == 3
        mock_model.filter.assert_called_once_with(run_id="run-9")


class TestBuildAttributionValueLabels:
    """build_attribution_value_labels — the display label for a stats value (display layer only)."""

    def test_layer_prefix_label(self) -> None:
        """primary_issue_category → a "layer/category" display label."""
        attrs = [
            {
                "primary_issue_layer": "tool",
                "primary_issue_category": "tool_timeout",
            },
            {
                "primary_issue_layer": "planning",
                "primary_issue_category": "intent_error",
            },
        ]
        labels = build_attribution_value_labels(
            attrs,
            ("primary_issue_category",),
        )
        assert labels == {
            "primary_issue_category": {
                "tool_timeout": "tool/tool_timeout",
                "intent_error": "planning/intent_error",
            },
        }

    def test_same_value_multiple_prefixes_keeps_first(self) -> None:
        """When the same value has several prefixes, take the first non-empty prefix."""
        attrs = [
            {
                "primary_issue_layer": "tool",
                "primary_issue_category": "timeout",
            },
            {
                "primary_issue_layer": "generation",
                "primary_issue_category": "timeout",
            },
        ]
        labels = build_attribution_value_labels(
            attrs,
            ("primary_issue_category",),
        )
        assert labels["primary_issue_category"]["timeout"] == "tool/timeout"

    def test_missing_prefix_generates_no_label(self) -> None:
        """A missing prefix / an empty string / a non-string → do not generate a label (display falls back to the raw value)."""
        attrs = [
            {"primary_issue_category": "no_layer"},
            {
                "primary_issue_layer": "",
                "primary_issue_category": "empty_layer",
            },
            {
                "primary_issue_layer": 42,
                "primary_issue_category": "int_layer",
            },
            {"primary_issue_layer": "tool", "primary_issue_category": ""},
        ]
        labels = build_attribution_value_labels(
            attrs,
            ("primary_issue_category",),
        )
        assert labels == {}

    def test_unmapped_field_excluded(self) -> None:
        """A stats field with no prefix mapping configured does not appear in the result."""
        assert "root_cause" not in ATTRIBUTION_LABEL_PREFIX_FIELDS
        labels = build_attribution_value_labels(
            [{"root_cause": "tool_selection", "severity": "major"}],
            ("root_cause", "severity"),
        )
        assert labels == {}
