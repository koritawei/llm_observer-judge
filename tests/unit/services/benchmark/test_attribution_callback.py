"""Unit tests for the attribution callback flow — slot release, idempotent counts, and inflight is not preempted."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
parse_external_attribution_detail = None
from benchmark.runs.schema import AttributionCallback, AttributionResult
from benchmark.workers.external_runner import (
    ExternalBenchmarkRunner,
    ExternalRunnerDeps,
)


def _runner(*, queue_manager: MagicMock | None) -> ExternalBenchmarkRunner:
    svc = MagicMock()
    svc.get_run = AsyncMock(return_value=SimpleNamespace(status="running"))
    svc.resolve_item_id_by_job_id = AsyncMock(return_value="item1")
    svc.increment_attribution_received_once = AsyncMock(
        return_value=(1, {"attribution_expected_count": 1}, True),
    )
    svc.heartbeat = AsyncMock()
    return ExternalBenchmarkRunner(
        benchmark_service=svc,
        storage_client=None,
        deps=ExternalRunnerDeps(queue_manager=queue_manager),
    )


def _payload(**overrides: object) -> AttributionCallback:
    data = {
        "run_id": "run1",
        "job_id": "attr-task-1",
        "item_id": "item1",
        "ieval_job_id": "ieval-job-1",
        "attribution": AttributionResult(
            is_eval_misjudge=False,
            primary_issue_layer="tool",
            primary_issue_category="tool_unavailable",
            primary_issue_detail="工具未覆盖该场景",
            affected_dims=["D2", "D4"],
            severity="major",
            root_cause_chain="工具缺失导致无法获取 POI",
            improvement_priority="P1",
            improvement_suggestion="建设加油站搜索工具",
        ),
        "upstream": {
            "upstream_job_id": "846e26e3cf49841b",
            "upstream_pipeline": "rubric-scoring",
        },
    }
    data.update(overrides)
    return AttributionCallback.model_validate(data)


@pytest.mark.asyncio
async def test_callback_removes_pending_releases_attr_slot_and_promotes() -> None:
    pending = MagicMock()
    pending.id = "run1:item1"  # case_job_id, not Aone task_id
    pending.name = "pending_callback"
    pending.data = {
        "slot_key": "pre:bullmq:inflight:attr:n1:run1",
        "cluster_kind": "attr",
        "run_id": "run1",
        "item_id": "item1",
    }
    pending.remove = AsyncMock()
    running_q = MagicMock()
    wait_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(
        side_effect=lambda name: running_q if "running" in name else wait_q,
    )
    runner = _runner(queue_manager=qm)
    ri = MagicMock()
    ri.source_trace_id = "judge-job-1"

    with (
        patch(
            "benchmark.workers.external_runner.BenchmarkRunItem.filter",
        ) as filt,
        patch(
            "benchmark.workers.external_runner.get_queue_job",
            new_callable=AsyncMock,
            return_value=pending,
        ),
        patch(
            "benchmark.workers.external_runner.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.external_runner.release_cluster_slot",
            new_callable=AsyncMock,
        ) as release_cluster,
        patch(
            "benchmark.workers.external_runner.promote_delayed_for_run",
            new_callable=AsyncMock,
            return_value=1,
        ) as promote,
        patch(
            "benchmark.workers.external_runner.BenchmarkTraceRecordWriter",
        ) as writer_cls,
        patch(
            "benchmark.workers.external_runner.try_finalize_external",
            new_callable=AsyncMock,
        ) as finalize,
    ):
        qs = MagicMock()
        qs.first = AsyncMock(return_value=ri)
        filt.return_value = qs
        writer = MagicMock()
        writer.upsert_triage = AsyncMock()
        writer_cls.return_value = writer

        await runner._handle_attribution_callback(_payload())

    pending.remove.assert_awaited_once()
    release.assert_awaited_once_with("pre:bullmq:inflight:attr:n1:run1")
    release_cluster.assert_awaited_once_with("attr")
    promote.assert_awaited()
    writer.upsert_triage.assert_awaited_once()
    assert writer.upsert_triage.await_args.args[0].trace_id == "judge-job-1"
    # The raw callback body is appended to attribution_detail_json.raw_json (unfiltered and unparsed)
    triage_write = writer.upsert_triage.await_args.args[0].triage
    raw_json = triage_write.attribution_detail_json.get("raw_json")
    assert raw_json is not None
    assert raw_json.get("primary_issue_layer") == "tool"
    assert raw_json.get("root_cause_chain") == "工具缺失导致无法获取 POI"
    runner._benchmark_service.increment_attribution_received_once.assert_awaited_once_with(
        "run1",
        "item1",
    )
    finalize.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_leaves_inflight_for_worker() -> None:
    """A callback that lands on inflight (the worker is still submitting) → do not remove it and do not release the slot."""
    inflight = MagicMock()
    inflight.id = "run1:item1"
    inflight.name = "inflight"
    inflight.data = {"slot_key": "pre:bullmq:inflight:attr:n1:run1"}
    inflight.remove = AsyncMock()
    running_q = MagicMock()
    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(return_value=running_q)
    runner = _runner(queue_manager=qm)

    with (
        patch(
            "benchmark.workers.external_runner.BenchmarkRunItem.filter",
        ) as filt,
        patch(
            "benchmark.workers.external_runner.get_queue_job",
            new_callable=AsyncMock,
            return_value=inflight,
        ),
        patch(
            "benchmark.workers.external_runner.release_slot",
            new_callable=AsyncMock,
        ) as release,
        patch(
            "benchmark.workers.external_runner.BenchmarkTraceRecordWriter",
        ) as writer_cls,
        patch(
            "benchmark.workers.external_runner.try_finalize_external",
            new_callable=AsyncMock,
        ),
    ):
        qs = MagicMock()
        qs.first = AsyncMock(return_value=MagicMock(source_trace_id="j1"))
        filt.return_value = qs
        writer = MagicMock()
        writer.upsert_triage = AsyncMock()
        writer_cls.return_value = writer

        await runner._handle_attribution_callback(_payload())

    inflight.remove.assert_not_awaited()
    release.assert_not_awaited()
    runner._benchmark_service.increment_attribution_received_once.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_healer_race_second_increment_not_claimed() -> None:
    """The healer already counted it → the callback has claimed=False, still writes the result, and does not amplify the count again."""
    runner = _runner(queue_manager=None)
    runner._benchmark_service.increment_attribution_received_once = AsyncMock(
        return_value=(1, {"attribution_expected_count": 1}, False),
    )

    with (
        patch(
            "benchmark.workers.external_runner.BenchmarkRunItem.filter",
        ) as filt,
        patch(
            "benchmark.workers.external_runner.BenchmarkTraceRecordWriter",
        ) as writer_cls,
        patch(
            "benchmark.workers.external_runner.try_finalize_external",
            new_callable=AsyncMock,
        ) as finalize,
    ):
        qs = MagicMock()
        qs.first = AsyncMock(return_value=MagicMock(source_trace_id="j1"))
        filt.return_value = qs
        writer = MagicMock()
        writer.upsert_triage = AsyncMock()
        writer_cls.return_value = writer

        await runner._handle_attribution_callback(_payload())

    runner._benchmark_service.increment_attribution_received_once.assert_awaited_once()
    # CAS still uses the returned count (still 1 after the idempotent write)
    finalize.assert_awaited_once()


@pytest.mark.asyncio
async def test_callback_resolves_item_id_via_ieval_job_id_when_missing() -> None:
    """When item_id is missing, prefer ieval_job_id (the attribution task_id) to resolve it, and pass kind="attribution".

    Before the fix the old implementation resolved with payload.job_id (= params.job_id echoed back = the judge
    source_trace_id), so resolve_item_id_by_job_id always hit the
    source_trace_id column and the attribution_job_id column query was dead code.
    If Aone echoed a wrong params.job_id, the attribution result was written onto the wrong item's row.
    """
    runner = _runner(queue_manager=None)
    runner._benchmark_service.resolve_item_id_by_job_id = AsyncMock(
        return_value="resolved-item",
    )

    with (
        patch(
            "benchmark.workers.external_runner.BenchmarkRunItem.filter",
        ) as filt,
        patch(
            "benchmark.workers.external_runner.BenchmarkTraceRecordWriter",
        ) as writer_cls,
        patch(
            "benchmark.workers.external_runner.try_finalize_external",
            new_callable=AsyncMock,
        ),
    ):
        qs = MagicMock()
        qs.first = AsyncMock(
            return_value=MagicMock(source_trace_id="judge-src-1"),
        )
        filt.return_value = qs
        writer = MagicMock()
        writer.upsert_triage = AsyncMock()
        writer_cls.return_value = writer

        # item_id=None，ieval_job_id="ieval-job-1"，job_id="judge-src-1"
        payload = _payload(item_id=None)
        await runner._handle_attribution_callback(payload)

    resolve_mock = runner._benchmark_service.resolve_item_id_by_job_id
    resolve_mock.assert_awaited()
    # The first call must use ieval_job_id and kind="attribution"
    first_call = resolve_mock.await_args_list[0]
    assert first_call.args[1] == "ieval-job-1"
    assert first_call.kwargs == {"kind": "attribution"}
    # The attribution result must be written to the row for the source_trace_id looked up from the DB
    writer.upsert_triage.assert_awaited_once()
    assert writer.upsert_triage.await_args.args[0].trace_id == "judge-src-1"
    assert writer.upsert_triage.await_args.args[0].dataset_item_id == "resolved-item"


@pytest.mark.asyncio
async def test_callback_accepts_external_pipeline_schema() -> None:
    """When the Aone attribution pipeline returns a different field set (trace_id/issue/dimension and so on)
    it must not 422/500 — every AttributionResult field is optional, and extra="allow" keeps unknown fields.
    """
    notification_payload = AttributionCallback.model_validate(
        {
            "run_id": "run1",
            "job_id": "attr-task-1",
            "item_id": "item1",
            "attribution": {
                "trace_id": "abc123",
                "adiu": "u9cgc7cbbf336315e77da7f52d34dc",  # allow-secret
                "query": "查找加油站",
                "issue": {"verdict": "bad", "issue_hint": "brand mismatch"},
                "expected": "",
                "actual": "response text",
                "dimension": "D2",
            },
        }
    )
    assert notification_payload.attribution is not None
    # Default values for the design-doc fields
    assert notification_payload.attribution.primary_issue_layer == ""
    assert notification_payload.attribution.root_cause_chain == ""
    assert notification_payload.attribution.is_eval_misjudge is False
    # Default values compatible with the old fields
    assert notification_payload.attribution.triage_level == ""
    assert notification_payload.attribution.root_cause == ""
    # extra="allow" keeps Aone-specific fields
    dumped = notification_payload.attribution.model_dump()
    assert dumped.get("trace_id") == "abc123"
    assert dumped["issue"]["verdict"] == "bad"
    assert dumped["dimension"] == "D2"


@pytest.mark.asyncio
async def test_callback_design_doc_payload_structure() -> None:
    """Validate the success callback body from design doc §7.3.

    Covers: the top-level ieval_job_id field, the top-level upstream field,
    and primary_issue_layer / root_cause_chain / affected_dims inside attribution.
    """
    payload = AttributionCallback.model_validate(
        {
            "run_id": "ext-run-abc123",
            "type": "attribution",
            "task_id": "aone-attr-batch-456",
            "job_id": "846e26e3cf49841b",
            "item_id": None,
            "ieval_job_id": "b2c3d4e5f6789012",
            "attribution": {
                "is_eval_misjudge": False,
                "eval_misjudge_reason": "",
                "primary_issue_layer": "tool",
                "primary_issue_category": "tool_unavailable",
                "primary_issue_detail": "用户需要查加油站但工具未覆盖该场景",
                "affected_dims": ["D2", "D4"],
                "severity": "major",
                "root_cause_chain": "用户查询加油站→工具不存在→无法获取 POI→回复缺失关键信息",
                "improvement_priority": "P1",
                "improvement_suggestion": "建设加油站搜索工具",
            },
            "upstream": {
                "upstream_job_id": "846e26e3cf49841b",
                "upstream_celery_task_id": "099eaa9e-67e2-46de-a7da-e35755fdf5f4",
                "upstream_pipeline": "rubric-scoring",
            },
            "error": None,
        }
    )
    assert payload.ieval_job_id == "b2c3d4e5f6789012"
    assert payload.upstream is not None
    assert payload.upstream["upstream_pipeline"] == "rubric-scoring"
    assert payload.attribution is not None
    attr = payload.attribution
    assert attr.primary_issue_layer == "tool"
    assert attr.primary_issue_category == "tool_unavailable"
    assert attr.affected_dims == ["D2", "D4"]
    assert attr.severity == "major"
    assert "加油站" in attr.root_cause_chain
    assert attr.improvement_priority == "P1"
    # effective_* properties prefer the new fields
    assert attr.effective_triage_level == "tool"
    assert attr.effective_root_cause == attr.root_cause_chain


def test_strip_attribution_detail_removes_sensitive_fields() -> None:
    """parse_aone_attribution_detail should strip judge_result / query / adiu / trace_id
    / ieval_job_id / task_id / type / job_id / upstream and keep everything else.
    """
    raw = {
        "primary_issue_layer": "tool",
        "primary_issue_category": "tool_unavailable",
        "root_cause_chain": "工具缺失",
        "severity": "major",
        "affected_dims": ["D2"],
        "is_eval_misjudge": False,
        "improvement_priority": "P1",
        "triage_level": "minor_bad",
        "root_cause": "tool miss",
        "owner_group": "navi",
        "confidence": 0.8,
        "evidence": "",
        "trace_id": "abc123",
        "adiu": "u9cgc7cbbf336315e77da7f52d34dc",  # allow-secret
        "query": "查找加油站",
        "issue": {"verdict": "bad", "issue_hint": "brand mismatch"},
        "expected": "",
        "actual": "response text",
        "dimension": "D2",
        "judge_result": {"run_id": "run1", "type": "judge"},
        "ieval_job_id": "5eebc766e7261c3f",
        "task_id": "task-1",
        "type": "attribution",
        "job_id": "attr-task-1",
        "upstream": {"service": "amap-agent"},
    }
    stripped = parse_external_attribution_detail(raw)
    # Keep the attribution-conclusion fields from the design doc
    assert stripped["primary_issue_layer"] == "tool"
    assert stripped["root_cause_chain"] == "工具缺失"
    assert stripped["severity"] == "major"
    assert stripped["affected_dims"] == ["D2"]
    # Compatible old fields are kept too
    assert stripped["triage_level"] == "minor_bad"
    assert stripped["root_cause"] == "tool miss"
    assert stripped["issue"]["verdict"] == "bad"
    assert stripped["dimension"] == "D2"
    # Strip sensitive / redundant fields
    for key in (
        "judge_result",
        "query",
        "adiu",
        "trace_id",
        "ieval_job_id",
        "task_id",
        "type",
        "job_id",
        "upstream",
    ):
        assert key not in stripped, f"{key} should be stripped"


def test_parse_notification_full_callback_structure() -> None:
    """Validate parse_aone_attribution_detail with a complete Aone attribution callback payload.

    Covers issue / failure_modes / directed_conclusion / cross_verify and the other fields,
    and checks that attribution-analysis fields are kept while sensitive / redundant fields are stripped.
    """
    notification_attribution = {
        "trace_id": "2175365ab21c0ded8a388320b90c9c6d",
        "adiu": "u9cgc7cbbf336315e77da7f52d34dc",  # allow-secret
        "query": "查找珠海市香洲区港三路周边5公里内中石化或中石油加油站",
        "issue": {
            "verdict": "bad",
            "issue_hint": "D3: 推荐 POI 与工具返回不一致",
            "expected": "推荐3个附近加油站",
            "actual": "推荐了无关餐厅",
            "dimension": "D3",
        },
        "judge_result": {
            "type": "judge",
            "criterion_scores": {"D0": {"score": 1.0}, "D3": {"score": 1.0}},
            "error": None,
        },
        "directed_dimension": "D3",
        "directed_node_categories": ["工具", "基模"],
        "react_rounds": 3,
        "node_count": 12,
        "primary_fm": "FM-TOOL-4",
        "primary_node_category": "工具",
        "failure_modes": [
            {
                "code": "FM-TOOL-4",
                "category": "工具",
                "confidence": 0.9,
                "evidence": "工具调用返回正确但 LLM 忽略",
            },
        ],
        "success_patterns": [],
        "directed_conclusion": "定位到节点 #7 工具调用结果未正确采纳",
        "summary": "Agent 在工具调用后未正确使用返回数据",
        "cross_verify": {
            "overall": "confirmed",
            "confirmed_fms": ["FM-TOOL-4"],
        },
        "ieval_job_id": "b2c3d4e5f6789012",
        "task_id": "aone-attr-batch-456",
        "type": "attribution",
        "job_id": "846e26e3cf49841b",
        "upstream": {
            "upstream_job_id": "846e26e3cf49841b",
            "upstream_pipeline": "rubric-scoring",
        },
    }

    result = parse_external_attribution_detail(notification_attribution)

    # Attribution-analysis fields are kept
    assert result["issue"]["verdict"] == "bad"
    assert result["issue"]["dimension"] == "D3"
    assert result["directed_dimension"] == "D3"
    assert result["directed_node_categories"] == ["工具", "基模"]
    assert result["react_rounds"] == 3
    assert result["node_count"] == 12
    assert result["primary_fm"] == "FM-TOOL-4"
    assert result["primary_node_category"] == "工具"
    assert len(result["failure_modes"]) == 1
    assert result["failure_modes"][0]["code"] == "FM-TOOL-4"
    assert result["failure_modes"][0]["confidence"] == 0.9
    assert result["directed_conclusion"] == "定位到节点 #7 工具调用结果未正确采纳"
    assert result["summary"] == "Agent 在工具调用后未正确使用返回数据"
    assert result["cross_verify"]["overall"] == "confirmed"
    assert result["cross_verify"]["confirmed_fms"] == ["FM-TOOL-4"]

    # Sensitive / redundant fields have been stripped
    for key in (
        "judge_result",
        "query",
        "adiu",
        "trace_id",
        "ieval_job_id",
        "task_id",
        "type",
        "job_id",
        "upstream",
    ):
        assert key not in result, f"{key} should be stripped"

    # exclude_unset: fields that were not provided do not appear in the output
    assert "success_patterns" not in result or result["success_patterns"] == []


@pytest.mark.asyncio
async def test_callback_enqueues_code_attribution_without_advancing_received() -> None:
    """When the switch is on and the grade is FAIL, only enqueue; do not add received in the callback and do not dispatch a ticket."""
    from benchmark.settings.setting import settings

    pending = MagicMock()
    pending.id = "run1:item1"
    pending.name = "pending_callback"
    pending.data = {
        "slot_key": "pre:bullmq:inflight:attr:n1:run1",
        "cluster_kind": "attr",
        "run_id": "run1",
        "item_id": "item1",
    }
    pending.remove = AsyncMock()
    code_q = MagicMock()
    code_q.add = AsyncMock()
    running_q = MagicMock()
    wait_q = MagicMock()

    def _queue(name: str) -> MagicMock:
        if name == "code-attribution":
            return code_q
        if "running" in name:
            return running_q
        return wait_q

    qm = MagicMock()
    qm.prefix = "pre:bullmq"
    qm.get_queue = MagicMock(side_effect=_queue)
    runner = _runner(queue_manager=qm)
    ri = MagicMock()
    ri.source_trace_id = "judge-job-1"
    ri.score = 0.0
    previous = settings.code_attribution_enabled
    settings.code_attribution_enabled = True

    try:
        with (
            patch(
                "benchmark.workers.external_runner.BenchmarkRunItem.filter",
            ) as filt,
            patch(
                "benchmark.workers.external_runner.get_queue_job",
                new_callable=AsyncMock,
                return_value=pending,
            ),
            patch(
                "benchmark.workers.external_runner.release_slot",
                new_callable=AsyncMock,
            ),
            patch(
                "benchmark.workers.external_runner.release_cluster_slot",
                new_callable=AsyncMock,
            ),
            patch(
                "benchmark.workers.external_runner.promote_delayed_for_run",
                new_callable=AsyncMock,
            ),
            patch(
                "benchmark.workers.external_runner.BenchmarkTraceRecordWriter",
            ) as writer_cls,
            patch(
                "benchmark.workers.external_runner.try_finalize_external",
                new_callable=AsyncMock,
            ) as finalize,
            patch(
                "benchmark.workers.external_runner.schedule_notification_dispatch",
            ) as dispatch,
        ):
            qs = MagicMock()
            qs.first = AsyncMock(return_value=ri)
            filt.return_value = qs
            writer = MagicMock()
            writer.upsert_triage = AsyncMock()
            writer_cls.return_value = writer

            await runner._handle_attribution_callback(_payload())
    finally:
        settings.code_attribution_enabled = previous

    code_q.add.assert_awaited_once()
    assert code_q.add.await_args.args[1]["advance_received"] is True
    runner._benchmark_service.increment_attribution_received_once.assert_not_awaited()
    dispatch.assert_not_called()
    finalize.assert_not_awaited()
    writer.upsert_triage.assert_awaited_once()
