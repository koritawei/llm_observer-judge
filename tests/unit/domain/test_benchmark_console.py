"""Evaluation-console documents: page fields are persisted, and a status switch does not enter benchmark execution."""

import pytest
from benchmark.records.enums import BenchmarkStatus, TraceStatus
from benchmark.records.models.benchmark_run import BenchmarkRun
from benchmark.records.models.runtime_config_record import (
    KIND_BENCHMARK_AGENT,
    KIND_EVALUATOR,
    KIND_TRACE_REPORT,
    SCOPE_GLOBAL,
    RuntimeConfigRecord,
    benchmark_agent_record_key,
    compose_evaluator_key_tail,
    evaluator_record_key,
    project_scope_key,
    trace_report_record_key,
)
from benchmark.records.models.trace_record import TraceRecord
from benchmark.domain.agent_scope_ownership import (
    OwnershipPolicy,
    default_ownership_policy,
    set_ownership_policy,
)
from benchmark.domain.benchmark_console import (
    AgentDocument,
    AgentScopeDocument,
    ConsoleError,
    DatasetDocument,
    EvalTaskDocument,
    apply_remove_task,
    apply_rerun_task,
    apply_start_collection,
    apply_start_sampling,
    apply_stop_sampling,
    apply_stop_task,
    delete_agent,
    delete_dataset,
    delete_scope,
    delete_task,
    find_dataset_document,
    get_task,
    list_agent_documents,
    list_datasets,
    list_schedulable_tasks,
    list_scopes,
    list_tasks,
    page_rubrics,
    save_agent,
    save_dataset,
    save_scope,
    save_task,
    start_collection,
    start_sampling,
    stop_sampling,
    stop_task,
    strip_agent_execution_fields,
)
from benchmark.domain.eval_config import EvalConfigStore
from benchmark.domain.runtime_config_store import (
    refresh_eval_config_snapshot,
    replace_eval_config_file,
    retire_runtime_config,
    upsert_runtime_config,
)
from benchmark.domain.schemas import EvalConfigFile
from benchmark.runs.common.rubric_classes import RUBRIC_CLASS_REFS
from benchmark.runs.run_lifecycle import ListRunsQuery
from benchmark.runs.run_service import BenchmarkService
from pydantic import ValidationError


def _dataset(**overrides: object) -> DatasetDocument:
    body: dict[str, object] = {
        "id": "ds-1",
        "agentScope": "other",
        "name": "poi-search",
        "source": "static_csv",
        "itemsCount": 3,
        "status": "ready",
        "createdAt": "2026-10-12 10:00:00",
        "toolName": "poi_search",
        "sampleRows": [],
    }
    body.update(overrides)
    return DatasetDocument.model_validate(body)


def _task(**overrides: object) -> EvalTaskDocument:
    body: dict[str, object] = {
        "id": "task-1",
        "agentScope": "amap-agent",
        "name": "夜间回归",
        "agentId": "ba-1",
        "datasetId": "ds-1",
        "formula": "weighted-average",
        "trigger": "immediate",
        "status": "active",
        "processedCount": 0,
        "totalItems": 3,
        "startedAt": "2026-10-12 10:00:00",
    }
    body.update(overrides)
    return EvalTaskDocument.model_validate(body)


def test_async_local_binding_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentDocument.model_validate(
            {
                "id": "ba-1",
                "name": "judge",
                "owner": "eval-team",
                "registeredAt": "2026-10-12 10:00:00",
                "agentScope": "amap-agent",
                "judge": {"execution": "async", "placement": "local", "classRef": "pkg:Cls"},
            }
        )


def test_remote_binding_requires_class_and_url() -> None:
    agent = AgentDocument.model_validate(
        {
            "id": "ba-2",
            "name": "judge",
            "owner": "eval-team",
            "registeredAt": "2026-10-12 10:00:00",
            "judge": {
                "execution": "sync",
                "placement": "remote",
                "classRef": "pkg:Cls",
                "targetUrl": "https://example/trigger",
            },
            "attribution": {
                "execution": "async",
                "placement": "remote",
                "classRef": "pkg:Attr",
                "targetUrl": "http://example/attr",
            },
        }
    )
    assert agent.judge is not None
    assert agent.judge.class_ref == "pkg:Cls"
    assert agent.judge.target_url == "https://example/trigger"
    assert agent.attribution is not None
    assert agent.attribution.class_ref == "pkg:Attr"
    with pytest.raises(ValidationError):
        AgentDocument.model_validate(
            {
                "id": "ba-3",
                "name": "judge",
                "owner": "eval-team",
                "registeredAt": "2026-10-12 10:00:00",
                "agentScope": "amap-agent",
                "judge": {
                    "execution": "async",
                    "placement": "remote",
                    "targetUrl": "https://example/trigger",
                },
            }
        )


def test_dataset_rejects_access_key() -> None:
    with pytest.raises(ValidationError):
        DatasetDocument.model_validate(
            {
                "id": "ds-oss-1",
                "agentScope": "amap-agent",
                "name": "bucket/prefix",
                "source": "oss",
                "itemsCount": 1,
                "status": "parsing",
                "createdAt": "2026-10-12 10:00:00",
                "ossBucket": "bench",
                "ossAk": "secret-id",
                "ossSk": "secret-key",
            }
        )


def test_realtime_dataset_keeps_trace_sampling() -> None:
    saved = _dataset(
        source="realtime",
        status="sampling",
        samplingRatio=0.001,
        traceService="vui-agent",
        traceSampling={
            "environment": [" prod ", ""],
            "name": ["amap-vui-agent.voice_ws.vui_agent"],
            "tags": ["amap-vui-agent"],
            "variableMapping": [
                {
                    "templateVariable": "input",
                    "traceObject": "observation",
                    "selectedColumnId": "input",
                    "jsonSelector": "",
                }
            ],
        },
    )
    dumped = saved.model_dump(by_alias=True, exclude_none=True)
    assert dumped["traceSampling"]["environment"] == ["prod"]
    assert dumped["traceSampling"]["name"] == ["amap-vui-agent.voice_ws.vui_agent"]
    assert "jsonSelector" not in dumped["traceSampling"]["variableMapping"][0]


def test_realtime_dataset_accepts_ingest_and_defaults_kafka() -> None:
    """When the page carries ingest (kafka/sidecar) it is kept on write; the default falls back to kafka."""
    sidecar = _dataset(
        source="realtime",
        status="sampling",
        samplingRatio=0.01,
        traceService="vui-agent",
        traceSampling={
            "ingest": "sidecar",
            "environment": ["prod"],
            "name": ["amap-vui-agent.voice_ws.vui_agent"],
            "tags": [],
            "variableMapping": [],
        },
    )
    dumped = sidecar.model_dump(by_alias=True, exclude_none=True)
    assert dumped["traceSampling"]["ingest"] == "sidecar"

    default = _dataset(
        source="realtime",
        status="sampling",
        samplingRatio=0.01,
        traceService="vui-agent",
        traceSampling={
            "environment": ["prod"],
            "name": ["amap-vui-agent.voice_ws.vui_agent"],
            "tags": [],
            "variableMapping": [],
        },
    )
    assert default.trace_sampling is not None
    assert default.trace_sampling.ingest == "kafka"

    with pytest.raises(ValidationError):
        _dataset(
            source="realtime",
            status="sampling",
            samplingRatio=0.01,
            traceService="vui-agent",
            traceSampling={
                "ingest": "realtime",
                "environment": ["prod"],
                "name": ["amap-vui-agent.voice_ws.vui_agent"],
                "tags": [],
                "variableMapping": [],
            },
        )


def test_stop_sampling_only_from_sampling() -> None:
    ready = stop_sampling(_dataset(source="realtime", status="sampling", samplingRatio=0.1))
    assert ready.status == "ready"
    with pytest.raises(ConsoleError) as caught:
        stop_sampling(_dataset())
    assert caught.value.status_code == 409


def test_start_sampling_only_realtime_from_ready() -> None:
    """Start sampling: only a realtime dataset that is ready can enter sampling, and one already sampling is idempotent."""
    sampling = start_sampling(
        _dataset(source="realtime", status="ready", samplingRatio=0.1),
    )
    assert sampling.status == "sampling"
    # Already sampling: return the original document idempotently
    again = start_sampling(_dataset(source="realtime", status="sampling"))
    assert again.status == "sampling"
    # A non-realtime source is rejected
    with pytest.raises(ConsoleError) as caught:
        start_sampling(_dataset())
    assert caught.value.status_code == 409
    # realtime but not ready (parsing) is rejected
    with pytest.raises(ConsoleError) as caught:
        start_sampling(_dataset(source="realtime", status="parsing"))
    assert caught.value.status_code == 409


def test_collection_records_endpoint_without_calling_it() -> None:
    started = start_collection(_dataset())
    assert started.status == "collecting"
    assert started.collection is not None
    assert (
        started.collection.spec_agent_endpoint == "http://spec-agent-poi_search.vs.amap.com/invoke"
    )
    assert started.collection.collected_traces == 0
    with pytest.raises(ConsoleError) as caught:
        start_collection(_dataset(source="realtime", status="sampling"))
    assert caught.value.status_code == 409


def test_stop_task_idempotent_and_terminal_blocked() -> None:
    """active can be stopped; stopping again (already stopped) returns idempotently; an immediate-trigger terminal complete/failed is still 409."""
    active = _task()
    stopped = stop_task(active)
    assert stopped.status == "stopped"
    assert stopped.failure_reason == "手动停止"
    # The frontend list status may lag the first stop: stopping again succeeds idempotently, and is no longer a 409.
    again = stop_task(stopped)
    assert again is stopped
    for terminal in ("complete", "failed"):
        with pytest.raises(ConsoleError) as caught:
            stop_task(_task(status=terminal))
        assert caught.value.status_code == 409


def test_stop_task_allows_scheduled_terminal_to_unschedule() -> None:
    """A scheduled task mistakenly written as complete/failed can still be stopped, so the schedule can be turned off."""
    for terminal in ("complete", "failed"):
        stopped = stop_task(_task(trigger="daily", status=terminal))
        assert stopped.status == "stopped"
        assert stopped.failure_reason == "手动停止"


def test_cron_task_requires_five_fields() -> None:
    task = _task(trigger="cron", cronExpression="0 2 * * *")
    assert task.cron_expression == "0 2 * * *"
    with pytest.raises(ValidationError):
        _task(trigger="cron")


def test_page_rubrics_follow_code_registry() -> None:
    rubrics = page_rubrics()
    assert {item["rubricId"] for item in rubrics} == set(RUBRIC_CLASS_REFS)
    kafka = next(item for item in rubrics if item["rubricId"] == "amap_agent_kafka_rubric")
    assert kafka["formula"] == "weighted-average"
    assert kafka["criteria"][0]["evaluatorRef"] == "amap_agent"
    assert kafka["classRef"].endswith("AmapAgentKafkaRubric")


async def test_saved_dataset_roundtrip_keeps_page_fields() -> None:
    saved = await save_dataset(_dataset(sessionRef="oss://bench/session.json"))
    assert saved["agentScope"] == "other"
    assert saved["sessionRef"] == "oss://bench/session.json"
    assert saved["schema_version"] == 1
    assert "ossAk" not in saved
    listed = await list_datasets("other")
    assert [item["id"] for item in listed] == ["ds-1"]
    assert await list_datasets("skill") == []


async def test_realtime_dataset_items_count_sums_benchmark_processing() -> None:
    """Realtime-stream dataset count = the sum of processing counts of every benchmark using that stream (including NEW).

    The trace_pipeline_records composite unique key includes evaluator_name: the same trace processed once by each of two
    benchmarks is two rows, and COUNT(*) does not dedupe trace_id. Also verifies:
    ① evaluatorName of other kind=evaluator entries under the same project is included in the sum;
    ② a non-realtime dataset keeps its own itemsCount unchanged.
    """
    await save_dataset(
        _dataset(
            id="ds-rt-count",
            name="实时采样集",
            source="realtime",
            itemsCount=0,
            status="sampling",
            projectId="proj-count",
            evaluatorName="amap_agent",
            streamInstanceId="ds-rt-count",
        ),
    )
    await save_dataset(_dataset(id="ds-static", source="static_csv", itemsCount=7))
    # Another benchmark under the same project is also consuming this realtime stream
    await upsert_runtime_config(
        kind=KIND_EVALUATOR,
        record_key=evaluator_record_key("proj-count", "search_ai_card", ""),
        payload={"evaluatorName": "search_ai_card"},
        scope_key=project_scope_key("proj-count"),
    )
    rows = [
        ("amap_agent@ds-rt-count", TraceStatus.NEW),
        ("amap_agent@ds-rt-count", TraceStatus.SCORED),
        ("amap_agent@ds-rt-count", TraceStatus.TRIAGED),
        ("search_ai_card@ds-rt-count", TraceStatus.NEW),
        ("search_ai_card@ds-rt-count", TraceStatus.SCORED),
    ]
    for index, (evaluator_name, status) in enumerate(rows):
        await TraceRecord.create(
            trace_id=f"trace-rt-{index}",
            project_id="proj-count",
            evaluator_name=evaluator_name,
            agent_name="amap-agent",
            status=status,
        )
    # A row from an unrelated project must not be counted
    await TraceRecord.create(
        trace_id="trace-other",
        project_id="proj-other",
        evaluator_name="amap_agent",
        agent_name="amap-agent",
        status=TraceStatus.SCORED,
    )

    listed = {item["id"]: item for item in await list_datasets("other")}
    assert listed["ds-rt-count"]["itemsCount"] == 5
    assert listed["ds-static"]["itemsCount"] == 7


async def test_task_keeps_manual_run_id() -> None:
    saved = await save_task(_task(manualRunId="gpt-judge-pro-20261012-m1"))
    assert saved["manualRunId"] == "gpt-judge-pro-20261012-m1"


@pytest.mark.asyncio
async def test_list_tasks_heals_scheduled_failed_after_rerun() -> None:
    """After a CLI rerun of the same run_id succeeds: the list read path corrects the stored row back to active and refreshes the stale pk."""
    run_id = "search_eval-20261103"
    await save_task(
        _task(
            id="task-rerun-heal",
            trigger="daily",
            status="failed",
            failureReason="judge timeout",
            manualRunId=run_id,
            manualRunPk="pk-stale-deleted",
        ),
    )
    live = await BenchmarkRun.create(
        run_id=run_id,
        benchmark_name="search_eval",
        service_name="search_eval",
        dataset_name="ds",
        status=BenchmarkStatus.COMPLETED,
        total_samples=200,
        total_evaluators=1,
        evaluated_items=200,
        failed_items=0,
        config_json={"eval_task_id": "task-rerun-heal"},
    )

    listed = {item["id"]: item for item in await list_tasks()}
    healed = listed["task-rerun-heal"]
    assert healed["status"] == "active"
    assert healed.get("failureReason") in (None, "")
    assert healed["manualRunPk"] == str(live.id)

    # Read again after persisting and it is still active (not only an overlay on the response)
    again = await get_task("task-rerun-heal")
    assert again["status"] == "active"
    assert again["manualRunPk"] == str(live.id)


def test_dataset_oss_path_spliced_from_document() -> None:
    """Path assembly for an upload dataset: prefix + partition date are read from the dataset document and joined into the full string.

    The dataset document does not store a frozen full string. The trigger day (20261021) and the upload partition (20261018)
    differ, and it must still join onto the upload partition. The document wins: any oss_object_key on the request side is
    overwritten by the assembled result. A pipeline document (no partition date) is left alone.
    """
    from benchmark.routes.benchmark import _apply_dataset_oss_path
    from benchmark.runs.schema import ExternalBenchmarkRequest

    document = _dataset(
        id="ds-upload",
        name="search-ai-card",
        ossPrefix="oss://agent-eval-oss/search_eval_p",
        ossPartitionDate="20261018",
        ossUploadId="3",
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        dataset_name="search-ai-card",
        oss_object_key="other/20261020/benchmark_final_results.csv",
    )
    _apply_dataset_oss_path(payload, document)
    # Three segments: prefix + partition date + same-day sequence number; repeated uploads of the same prefix on the same day do not collide
    assert payload.oss_object_key == "search_eval_p/20261018/3/benchmark_final_results.csv"
    assert payload.oss_prefix == "oss://agent-eval-oss/search_eval_p"

    # An existing document has no batch segment: join the old layout {prefix}/{date}/{file} (the file is in the old layout)
    legacy_doc = _dataset(
        id="ds-legacy-upload",
        name="legacy-card",
        ossPrefix="search_eval_p",
        ossPartitionDate="20261018",
    )
    legacy_payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        dataset_name="legacy-card",
    )
    _apply_dataset_oss_path(legacy_payload, legacy_doc)
    assert legacy_payload.oss_object_key == "search_eval_p/20261018/benchmark_final_results.csv"

    pipeline_doc = _dataset(id="ds-pipe", name="pipe-ds", ossPrefix="pipe_p")
    pipe_payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        dataset_name="pipe-ds",
    )
    _apply_dataset_oss_path(pipe_payload, pipeline_doc)
    assert pipe_payload.oss_object_key is None


def test_dataset_oss_date_macro_expanded_from_document() -> None:
    """Date macros for an OSS-imported dataset: expand against this run's data date, and the expansion is the data-file directory.

    Only source=oss takes effect. CSV upload / realtime sampling and an upload snapshot (with a partition date)
    stay unchanged. An illegal macro is rejected with 422 at accept time.
    """
    from benchmark.routes.benchmark import _apply_dataset_oss_date_macro, _macro_oss_object_key
    from benchmark.runs.schema import ExternalBenchmarkRequest
    from fastapi import HTTPException

    document = _dataset(
        id="ds-oss-macro",
        name="daily-oss",
        source="oss",
        ossPrefix="agent-eval/daily/${yyyymmdd-1}",
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        dataset_name="daily-oss",
    )
    assert _apply_dataset_oss_date_macro(payload, document, base_date="20261021") is True
    assert payload.oss_prefix == "agent-eval/daily/20261020"
    # The macro is the date partition: the file lands directly under the expanded prefix, and the evaluation date is not appended
    assert (
        _macro_oss_object_key("oss://agent-eval-oss/agent-eval/daily/20261020")
        == "agent-eval/daily/20261020/benchmark_final_results.csv"
    )

    # No macro: the prefix is still backfilled from the document (a scheduled task that only carries dataset_id is located from the document)
    plain = _dataset(
        id="ds-oss-plain",
        name="plain-oss",
        source="oss",
        ossPrefix="agent-eval/plain",
    )
    plain_payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        dataset_name="plain-oss",
    )
    assert _apply_dataset_oss_date_macro(plain_payload, plain, base_date="20261021") is False
    assert plain_payload.oss_prefix == "agent-eval/plain"

    # Other sources (CSV upload) do not take part
    csv_doc = _dataset(id="ds-csv", source="static_csv", ossPrefix="a/${yyyymmdd}")
    csv_payload = ExternalBenchmarkRequest(benchmark_name="search_eval", date="20261021")
    assert _apply_dataset_oss_date_macro(csv_payload, csv_doc, base_date="20261021") is False
    assert csv_payload.oss_prefix is None

    # An upload snapshot (with a partition date) does not take part: the path is already fixed
    uploaded = _dataset(
        id="ds-up",
        source="oss",
        ossPrefix="p",
        ossPartitionDate="20261018",
    )
    up_payload = ExternalBenchmarkRequest(benchmark_name="search_eval", date="20261021")
    assert _apply_dataset_oss_date_macro(up_payload, uploaded, base_date="20261021") is False

    # An explicit prefix on the request wins: an illegal macro is a 422 at accept time
    bad_payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        oss_prefix="x/${yyyymmdd+abc}",
    )
    with pytest.raises(HTTPException) as exc:
        _apply_dataset_oss_date_macro(bad_payload, plain, base_date="20261021")
    assert exc.value.status_code == 422

    # After an immediate page trigger pre-expands and forwards a literal path: fall back to the document macro and expand it again with base_date
    preexpanded = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        oss_prefix="agent-eval/daily/20261020",
    )
    assert _apply_dataset_oss_date_macro(preexpanded, document, base_date="20261021") is True
    assert preexpanded.oss_prefix == "agent-eval/daily/20261020"

    # No dataset document: still expand macros on the request side (direct API / RemoteConfig calls)
    api_only = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        oss_prefix="agent-eval/daily/${yyyymmdd}",
    )
    assert _apply_dataset_oss_date_macro(api_only, None, base_date="20261021") is True
    assert api_only.oss_prefix == "agent-eval/daily/20261021"


def test_dataset_document_rejects_invalid_oss_date_macro() -> None:
    """An illegal macro is rejected when the dataset is stored, and must not be forwarded to OSS to become a file not found."""
    saved = _dataset(id="ds-oss-ok", source="oss", ossPrefix="a/${yyyymmdd+2}")
    assert saved.oss_prefix == "a/${yyyymmdd+2}"
    with pytest.raises(ValidationError, match="非法日期宏"):
        _dataset(id="ds-oss-bad", source="oss", ossPrefix="a/${yyyymmdd+abc}")


def test_legacy_dataset_oss_object_key_converted_to_partition_date() -> None:
    """Legacy conversion: when an old document's frozen full string is read, split out the partition date and drop the full string."""
    document = DatasetDocument.model_validate(
        {
            "id": "ds-legacy",
            "agentScope": "amap-agent",
            "name": "legacy-set",
            "source": "static_csv",
            "itemsCount": 3,
            "status": "ready",
            "createdAt": "2026-10-18 10:00:00",
            "sampleRows": [],
            "ossPrefix": "oss://agent-eval-oss/search_eval_p",
            "ossObjectKey": "search_eval_p/20261018/benchmark_final_results.csv",
        },
    )
    assert document.oss_partition_date == "20261018"
    dumped = document.model_dump(by_alias=True)
    assert dumped["ossPartitionDate"] == "20261018"
    assert "ossObjectKey" not in dumped


async def test_source_run_clone_reuses_snapshot_config() -> None:
    """Triggering a clone again: date / OSS path / execution class follow the source run's snapshot, and today is not recomputed."""
    from benchmark.routes.benchmark import _apply_source_run_clone
    from benchmark.runs.schema import ExternalBenchmarkRequest

    source = await BenchmarkRun.create(
        run_id="search_eval-20261018-m1",
        benchmark_name="search_eval",
        service_name="search_ai_card",
        dataset_name="search-ai-card",
        status=BenchmarkStatus.COMPLETED,
        total_samples=10,
        total_evaluators=1,
        final_score=0.8,
        config_json={
            "data_date": "20261018",
            "oss_prefix": "search_eval_p",
            "oss_object_key": "search_eval_p/20261018/benchmark_final_results.csv",
            "judge_class": "search_eval_judge",
            "generate_report": False,
        },
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        source_id=str(source.id),
    )
    cloned, date, snapshot = await _apply_source_run_clone(payload)
    assert date == "20261018"
    assert cloned.date == "20261018"
    assert cloned.oss_object_key == "search_eval_p/20261018/benchmark_final_results.csv"
    assert cloned.oss_prefix == "search_eval_p"
    assert cloned.judge_class == "search_eval_judge"
    assert cloned.generate_report is False
    # When the source run has no registration snapshot, do not inherit one (the new run resolves against the current registration)
    assert snapshot is None


async def test_soft_deleted_run_skipped_by_all_queries() -> None:
    """Soft-delete semantics: only stamp deleted_at, and keep the row; list/detail/count/clone all skip it."""
    from benchmark.routes.benchmark import _apply_source_run_clone
    from benchmark.runs.schema import ExternalBenchmarkRequest

    service = BenchmarkService()
    run = await BenchmarkRun.create(
        run_id="search_eval-20261018-m1",
        benchmark_name="search_eval",
        service_name="search_ai_card",
        dataset_name="search-ai-card",
        status=BenchmarkStatus.COMPLETED,
        total_samples=10,
        total_evaluators=1,
        final_score=0.8,
        config_json={"data_date": "20261018"},
    )

    assert await service.delete_run(run.id) is True
    # Deleting again: a row already marked cannot be deleted again
    assert await service.delete_run(run.id) is False

    # The row is still in the store; it only gained soft-delete markers (the two fields of BaseModelWithSoftDelete)
    row = await BenchmarkRun.get(id=run.id)
    assert row.is_deleted is True
    assert row.deleted_at is not None

    # Queries skip all of them
    assert await service.get_run("search_eval-20261018-m1") is None
    assert await service.get_run_by_pk(run.id) is None
    assert await service.list_runs(ListRunsQuery(benchmark_name="search_eval")) == []
    counts = await service.count_runs_by_benchmark()
    assert all(item["benchmarkName"] != "search_eval" for item in counts)

    # A deleted run cannot be cloned by another trigger
    payload = ExternalBenchmarkRequest(
        benchmark_name="search_eval",
        date="20261021",
        source_id=str(run.id),
    )
    with pytest.raises(Exception) as excinfo:
        await _apply_source_run_clone(payload)
    assert getattr(excinfo.value, "status_code", None) == 404


async def test_legacy_realtime_count_without_stream_uses_project_evaluators() -> None:
    """A legacy dataset with no stream still counts processing times of every evaluator under the project."""
    from benchmark.domain.benchmark_console import _fill_realtime_processed_counts

    await upsert_runtime_config(
        kind=KIND_EVALUATOR,
        record_key=evaluator_record_key("proj-legacy", "search_ai_card", ""),
        payload={"evaluatorName": "search_ai_card"},
        scope_key=project_scope_key("proj-legacy"),
    )
    for index, evaluator_name in enumerate(("amap_agent", "search_ai_card")):
        await TraceRecord.create(
            trace_id=f"trace-legacy-{index}",
            project_id="proj-legacy",
            evaluator_name=evaluator_name,
            agent_name="amap-agent",
            status=TraceStatus.SCORED,
        )
    items = [
        {
            "id": "ds-legacy",
            "source": "realtime",
            "projectId": "proj-legacy",
            "evaluatorName": "amap_agent",
            "itemsCount": 0,
        }
    ]
    await _fill_realtime_processed_counts(items)
    assert items[0]["itemsCount"] == 2


async def test_realtime_count_zero_falls_back_to_bound_agent_evaluation_sum() -> None:
    """When the trace count is 0, itemsCount = the sum of evaluationCount of agents bound to that stream."""
    from benchmark.domain.benchmark_console import (
        _fill_realtime_processed_counts,
        fill_realtime_bound_agent_counts,
    )

    await save_dataset(
        _dataset(
            id="ds-rt-zero",
            name="空采样流",
            source="realtime",
            itemsCount=0,
            status="sampling",
            projectId="proj-zero",
            evaluatorName="amap_agent",
            streamInstanceId="ds-rt-zero",
        ),
    )
    await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-bound-a",
                "name": "amap_agent_benchmark",
                "owner": "eval-team",
                "registeredAt": "2026-10-12 10:00:00",
                "agentScope": "amap-agent",
            }
        ),
    )
    await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-bound-b",
                "name": "search_eval",
                "owner": "eval-team",
                "registeredAt": "2026-10-12 10:00:00",
                "agentScope": "amap-agent",
            }
        ),
    )
    await save_task(
        _task(
            id="task-bound-a",
            agentId="ba-bound-a",
            datasetId="ds-rt-zero",
            agentScope="amap-agent",
        ),
    )
    await save_task(
        _task(
            id="task-bound-b",
            agentId="ba-bound-b",
            datasetId="ds-rt-zero",
            agentScope="amap-agent",
        ),
    )

    items = [
        {
            "id": "ds-rt-zero",
            "source": "realtime",
            "projectId": "proj-zero",
            "streamInstanceId": "ds-rt-zero",
            "itemsCount": 0,
        }
    ]
    await _fill_realtime_processed_counts(items)
    assert items[0]["itemsCount"] == 0
    await fill_realtime_bound_agent_counts(
        items,
        [
            {
                "benchmarkName": "amap_agent_benchmark",
                "registrationId": "ba-bound-a",
                "evaluationCount": 12,
                "batchCount": 4,
                "streamingCount": 8,
            },
            {
                "benchmarkName": "search_eval",
                "registrationId": "ba-bound-b",
                "evaluationCount": 5,
                "batchCount": 5,
                "streamingCount": 0,
            },
        ],
    )
    assert items[0]["itemsCount"] == 17


async def test_realtime_count_zero_binds_agents_via_stem_reg_stream_evaluator_name() -> None:
    """With no task, several agent bindings can still be recognized from ``stem@registrationId@streamInstanceId``."""
    from benchmark.domain.benchmark_console import (
        _fill_realtime_processed_counts,
        fill_realtime_bound_agent_counts,
    )

    stream = "ds-rt-1793370626586"
    reg_a = "ba-82a6e3f2-724c-4e82-8c4d-111935b81bc7"
    reg_b = "ba-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    await save_dataset(
        _dataset(
            id=stream,
            name="多 agent 实时流",
            source="realtime",
            itemsCount=0,
            status="sampling",
            projectId="proj-multi-agent",
            streamInstanceId=stream,
        ),
    )
    await upsert_runtime_config(
        kind=KIND_EVALUATOR,
        record_key=evaluator_record_key(
            "proj-multi-agent",
            f"amap_agent@{reg_a}@{stream}",
            "amap-agent",
            compose_evaluator_key_tail(reg_a, stream),
        ),
        payload={
            "evaluatorName": f"amap_agent@{reg_a}@{stream}",
            "agentRegistrationId": reg_a,
            "streamInstanceId": stream,
        },
        scope_key=project_scope_key("proj-multi-agent"),
    )
    await upsert_runtime_config(
        kind=KIND_EVALUATOR,
        record_key=evaluator_record_key(
            "proj-multi-agent",
            f"search_ai_card@{reg_b}@{stream}",
            "search-ai-card",
            compose_evaluator_key_tail(reg_b, stream),
        ),
        payload={
            "evaluatorName": f"search_ai_card@{reg_b}@{stream}",
            "agentRegistrationId": reg_b,
            "streamInstanceId": stream,
        },
        scope_key=project_scope_key("proj-multi-agent"),
    )

    items = [
        {
            "id": stream,
            "source": "realtime",
            "projectId": "proj-multi-agent",
            "streamInstanceId": stream,
            "itemsCount": 0,
        }
    ]
    await _fill_realtime_processed_counts(items)
    assert items[0]["itemsCount"] == 0
    await fill_realtime_bound_agent_counts(
        items,
        [
            {
                "benchmarkName": "amap_agent",
                "registrationId": reg_a,
                "evaluationCount": 9,
                "batchCount": 0,
                "streamingCount": 9,
            },
            {
                "benchmarkName": "search_ai_card",
                "registrationId": reg_b,
                "evaluationCount": 4,
                "batchCount": 0,
                "streamingCount": 4,
            },
        ],
    )
    assert items[0]["itemsCount"] == 13


def test_daily_report_matches_registration_isolation() -> None:
    """example dual chain: a version-isolated run belongs only to the task of the matching registration."""
    from types import SimpleNamespace

    from benchmark.domain.benchmark_console import (
        DailyReportMatchContext,
        _daily_report_matches_benchmark,
    )
    from benchmark.domain.trace_report_config import versioned_trace_report_name

    dataset = "example-ds"
    reg_a = "reg-v56"
    versioned = versioned_trace_report_name(dataset, reg_a)
    # Old semantics: the namespace is on benchmark_name, and dataset_name=trace_pipeline:*
    run = SimpleNamespace(
        benchmark_name=versioned,
        dataset_name="trace_pipeline:kafka",
    )
    config = {
        "trace_daily_report": True,
        "benchmark_component": "example_benchmark",
        "agent_registration_id": reg_a,
        "judge_content_version": "56",
    }
    assert _daily_report_matches_benchmark(
        run,  # type: ignore[arg-type]
        config,
        ctx=DailyReportMatchContext(
            agent_name="example_benchmark",
            dataset_name=dataset,
            registration_id=reg_a,
        ),
    )
    assert not _daily_report_matches_benchmark(
        run,  # type: ignore[arg-type]
        config,
        ctx=DailyReportMatchContext(
            agent_name="example_benchmark",
            dataset_name=dataset,
            registration_id="reg-v57",
        ),
    )
    # Customized-registration version isolation: an old report with no stamp is treated as shared history and still belongs (same shape as live amap_agent_daily)
    bare = SimpleNamespace(benchmark_name=dataset, dataset_name="trace_pipeline:kafka")
    assert _daily_report_matches_benchmark(
        bare,  # type: ignore[arg-type]
        {"trace_daily_report": True, "benchmark_component": "example_benchmark"},
        ctx=DailyReportMatchContext(
            agent_name="example_benchmark",
            dataset_name=dataset,
            registration_id=reg_a,
            require_version_isolation=True,
        ),
    )
    # Stamped but the registration does not match: still rejected
    assert not _daily_report_matches_benchmark(
        bare,  # type: ignore[arg-type]
        {
            "trace_daily_report": True,
            "benchmark_component": "example_benchmark",
            "agent_registration_id": "reg-other",
        },
        ctx=DailyReportMatchContext(
            agent_name="example_benchmark",
            dataset_name=dataset,
            registration_id=reg_a,
            require_version_isolation=True,
        ),
    )
    # New semantics: dataset_name is the page name, and benchmark_name is the internal agent
    new_run = SimpleNamespace(
        benchmark_name="example_benchmark",
        dataset_name=dataset,
    )
    new_cfg = {
        "trace_daily_report": True,
        "benchmark_component": "example_benchmark",
        "agent_registration_id": reg_a,
        "trace_report_namespace": versioned,
    }
    assert _daily_report_matches_benchmark(
        new_run,  # type: ignore[arg-type]
        new_cfg,
        ctx=DailyReportMatchContext(
            agent_name="example_benchmark",
            dataset_name=dataset,
            registration_id=reg_a,
            require_version_isolation=True,
        ),
    )
    assert not _daily_report_matches_benchmark(
        new_run,  # type: ignore[arg-type]
        {**new_cfg, "agent_registration_id": "reg-v57"},
        ctx=DailyReportMatchContext(
            agent_name="example_benchmark",
            dataset_name=dataset,
            registration_id=reg_a,
            require_version_isolation=True,
        ),
    )
    # Isolation not forced: when the task side still passes agent.id, align by the stamp; a stamp whose id matches should be accepted
    assert _daily_report_matches_benchmark(
        new_run,  # type: ignore[arg-type]
        new_cfg,
        ctx=DailyReportMatchContext(
            agent_name="example_benchmark",
            dataset_name=dataset,
            registration_id=reg_a,
            require_version_isolation=False,
        ),
    )
    # Isolation not forced and the task did not pass registration_id: do not reject just because the run has a stamp
    assert _daily_report_matches_benchmark(
        new_run,  # type: ignore[arg-type]
        new_cfg,
        ctx=DailyReportMatchContext(
            agent_name="example_benchmark",
            dataset_name=dataset,
            registration_id=None,
            require_version_isolation=False,
        ),
    )


async def test_realtime_task_daily_reports_filled_from_report_runs() -> None:
    """dailyReports for a realtime task: write an empty placeholder at create time, and the read path backfills from the dataset namespace plus the registration time."""
    await save_dataset(_dataset(source="realtime", status="sampling"))
    await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-1",
                "name": "example_benchmark",
                "owner": "eval-team",
                "registeredAt": "2026-10-12 09:00:00",
                "agentScope": "amap-agent",
                "capabilities": ["judge"],
            },
        ),
    )
    await save_task(_task(dailyReports=[], agentId="ba-1", startedAt="2026-10-12 10:00:00"))
    report_run = await BenchmarkRun.create(
        run_id="poi-search-20261020",
        benchmark_name="poi-search",
        service_name="amap_agent",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=12,
        total_evaluators=1,
        final_score=0.75,
        config_json={
            "trace_daily_report": True,
            "benchmark_component": "example_benchmark",
        },
        report_meta_json={
            "report_site_url": "https://pages.example/daily.html",
            "report_url": "https://reports.example/daily.html",
        },
    )
    # An unfinished run, or a run that is not a daily report, does not enter the daily report
    await BenchmarkRun.create(
        run_id="poi-search-20261021",
        benchmark_name="poi-search",
        service_name="amap_agent",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.RUNNING,
        total_samples=3,
        total_evaluators=1,
        config_json={
            "trace_daily_report": True,
            "benchmark_component": "example_benchmark",
        },
        report_meta_json={"report_url": "https://reports.example/running.html"},
    )

    task = await get_task("task-1")
    assert task["dailyReports"] == [
        {
            "date": "2026-10-20",
            "score": 0.75,
            "itemsCount": 12,
            "reportUrl": "https://pages.example/daily.html",
            "ossReportUrl": "https://reports.example/daily.html",
            "id": str(report_run.id),
        },
    ]
    listed = await list_tasks()
    assert listed[0]["dailyReports"] == task["dailyReports"]


async def test_realtime_daily_reports_use_dataset_namespace_not_agent_name() -> None:
    """A report run's benchmark_name is the dataset name (amap-agent-sampling-date) and must not be hard-matched against agent.name."""
    ds_name = "amap-agent-sampling-2026-10-20"
    await save_dataset(
        _dataset(id="ds-amap-rt", name=ds_name, source="realtime", status="sampling"),
    )
    await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-amap",
                "name": "amap_agent_benchmark",
                "owner": "eval-team",
                "registeredAt": "2026-10-01 09:00:00",
                "agentScope": "amap-agent",
                "capabilities": ["judge"],
            },
        ),
    )
    await save_task(
        _task(
            id="task-amap",
            agentId="ba-amap",
            datasetId="ds-amap-rt",
            dailyReports=[],
            startedAt="2026-10-01 10:00:00",
            name="amap 实时",
        ),
    )
    report_run = await BenchmarkRun.create(
        run_id=f"{ds_name}-20261019",
        benchmark_name=ds_name,
        service_name="amap_agent",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=8,
        total_evaluators=1,
        final_score=0.88,
        config_json={
            "trace_daily_report": True,
            "benchmark_component": "amap_agent_benchmark",
        },
        report_meta_json={"report_site_url": "https://pages.example/amap-daily.html"},
    )
    legacy_run = await BenchmarkRun.create(
        run_id=f"{ds_name}-20261018",
        benchmark_name=ds_name,
        service_name="amap_agent",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=5,
        total_evaluators=1,
        final_score=0.7,
        config_json={"trace_daily_report": True},
        report_meta_json={"report_site_url": "https://pages.example/amap-legacy.html"},
    )

    task = await get_task("task-amap")
    assert [item["id"] for item in task["dailyReports"]] == [
        str(report_run.id),
        str(legacy_run.id),
    ]


async def test_realtime_daily_reports_split_by_namespace_not_task_started_at() -> None:
    """A different benchmark is split apart by namespace; the time lower bound uses the registration date, and the task startedAt does not cut off history."""
    base = default_ownership_policy()
    set_ownership_policy(
        OwnershipPolicy(
            scopes=base.scopes,
            global_agents=base.global_agents | frozenset({"bench_a", "bench_b"}),
            name_aliases=base.name_aliases,
        ),
    )
    try:
        await _realtime_daily_reports_split_body()
    finally:
        set_ownership_policy(None)


async def _realtime_daily_reports_split_body() -> None:
    await save_dataset(_dataset(id="ds-rt", name="shared-ds", source="realtime", status="sampling"))
    await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-a",
                "name": "bench_a",
                "owner": "eval-team",
                "registeredAt": "2026-10-01 09:00:00",
                "agentScope": "amap-agent",
                "capabilities": ["judge"],
            },
        ),
    )
    await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-b",
                "name": "bench_b",
                "owner": "eval-team",
                "registeredAt": "2026-10-01 09:00:00",
                "agentScope": "amap-agent",
                "capabilities": ["judge"],
            },
        ),
    )
    # Stopping and starting again mid-task refreshes startedAt; a historical daily report must stay visible by its registration date.
    await save_task(
        _task(
            id="task-a",
            agentId="ba-a",
            datasetId="ds-rt",
            dailyReports=[],
            startedAt="2026-10-10 10:00:00",
            name="任务A",
        ),
    )
    await save_task(
        _task(
            id="task-b",
            agentId="ba-b",
            datasetId="ds-rt",
            dailyReports=[],
            startedAt="2026-10-18 10:00:00",
            name="任务B",
        ),
    )

    run_a_old = await BenchmarkRun.create(
        run_id="shared-ds-20261005",
        benchmark_name="shared-ds",
        service_name="svc",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=1,
        total_evaluators=1,
        final_score=0.1,
        config_json={"trace_daily_report": True, "benchmark_component": "bench_a"},
        report_meta_json={"report_site_url": "https://pages.example/a-old.html"},
    )
    run_a = await BenchmarkRun.create(
        run_id="shared-ds-20261012",
        benchmark_name="shared-ds",
        service_name="svc",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=2,
        total_evaluators=1,
        final_score=0.2,
        config_json={"trace_daily_report": True, "benchmark_component": "bench_a"},
        report_meta_json={"report_site_url": "https://pages.example/a.html"},
    )
    run_before_reg = await BenchmarkRun.create(
        run_id="shared-ds-20260920",
        benchmark_name="shared-ds",
        service_name="svc",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=1,
        total_evaluators=1,
        final_score=0.05,
        config_json={"trace_daily_report": True, "benchmark_component": "bench_a"},
        report_meta_json={"report_site_url": "https://pages.example/a-before-reg.html"},
    )
    run_b = await BenchmarkRun.create(
        run_id="bench_b-shared-ds-20261020",
        benchmark_name="bench_b-shared-ds",
        service_name="svc",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=3,
        total_evaluators=1,
        final_score=0.3,
        config_json={"trace_daily_report": True, "benchmark_component": "bench_b"},
        report_meta_json={"report_site_url": "https://pages.example/b.html"},
    )
    task_a = await get_task("task-a")
    task_b = await get_task("task-b")
    # Registration date 10-01: 10-05 / 10-12 are both visible; 09-20, before registration, is still filtered out
    assert str(run_before_reg.id) not in {item["id"] for item in task_a["dailyReports"]}
    assert [item["id"] for item in task_a["dailyReports"]] == [
        str(run_a.id),
        str(run_a_old.id),
    ]
    assert [item["date"] for item in task_a["dailyReports"]] == [
        "2026-10-12",
        "2026-10-05",
    ]
    assert [item["id"] for item in task_b["dailyReports"]] == [str(run_b.id)]
    assert [item["date"] for item in task_b["dailyReports"]] == ["2026-10-20"]


async def test_realtime_daily_reports_survive_task_restart_after_registration() -> None:
    """When the task startedAt is later than the daily-report data date, still backfill as long as it is ≥ agent.registeredAt.

    The dataset document name (example-v56) differs from both the task name and run.dataset_name (trace_pipeline:kafka).
    The backfill key is the dataset document name.
    """
    await save_dataset(
        _dataset(id="ds-ex", name="example-v56", source="realtime", status="sampling"),
    )
    await save_agent(
        AgentDocument.model_validate(
            {
                "id": "ba-ex-v56",
                "name": "example_benchmark",
                "owner": "eval-team",
                "registeredAt": "2026-10-20 09:00:00",
                "agentScope": "amap-agent",
                "capabilities": ["judge"],
            },
        ),
    )
    # Simulate stop then start: startedAt is refreshed to 10-26 (later than the 10-25 daily report)
    await save_task(
        _task(
            id="task-ex",
            agentId="ba-ex-v56",
            datasetId="ds-ex",
            dailyReports=[],
            startedAt="2026-10-26 09:16:11",
            name="example 评测任务名可不同于数据集",
        ),
    )
    report_run = await BenchmarkRun.create(
        run_id="example-v56-20261025",
        benchmark_name="example-v56",
        service_name="example",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=8,
        total_evaluators=2,
        final_score=6.25,
        config_json={
            "trace_daily_report": True,
            "benchmark_component": "example_benchmark",
        },
        report_meta_json={"report_site_url": "https://pages.example/ex-v56.html"},
    )
    task = await get_task("task-ex")
    assert [item["id"] for item in task["dailyReports"]] == [str(report_run.id)]
    assert task["dailyReports"][0]["date"] == "2026-10-25"


async def test_ensure_default_trace_report_registers_with_defaults() -> None:
    """Default report registration: write only required fields, the schedule point defaults to 09:00, and registering the same name again is idempotent."""
    from benchmark.domain.trace_report_config import (
        TraceReportSeed,
        ensure_default_trace_report,
        list_db_trace_report_items,
    )

    seed = TraceReportSeed(
        name="poi-search",
        benchmark_name="amap_agent_benchmark",
        source="kafka",
        agent_name="amap-agent",
        project_id="proj-1",
    )
    name = await ensure_default_trace_report(seed)
    assert name == "poi-search"
    again = await ensure_default_trace_report(seed)
    assert again == "poi-search"

    items = await list_db_trace_report_items()
    assert len(items) == 1
    item = items[0]
    assert item.benchmark_name == "amap_agent_benchmark"
    assert item.schedule_time == "09:00"
    assert item.generate_report is True
    assert item.sample_size is None


async def test_ensure_default_trace_report_versions_by_registration() -> None:
    """example dual chain: with a registrationId, the namespace is isolated from content_version."""
    from benchmark.domain.trace_report_config import (
        TraceReportSeed,
        ensure_default_trace_report,
        list_db_trace_report_items,
        versioned_trace_report_name,
    )

    seed = TraceReportSeed(
        name="example-ds",
        benchmark_name="example_benchmark",
        source="kafka",
        agent_name="amap-agent",
        project_id="proj-1",
        agent_registration_id="reg-v56",
        judge_content_version="56",
    )
    name = await ensure_default_trace_report(seed)
    assert name == versioned_trace_report_name("example-ds", "reg-v56")
    items = await list_db_trace_report_items()
    assert len(items) == 1
    assert items[0].agent_registration_id == "reg-v56"
    assert items[0].judge_content_version == "56"
    assert items[0].name == name


async def test_ensure_default_trace_report_never_overwrites_customized() -> None:
    """Default registration is only a fallback: an existing same-name item (including a schedule point or template the user edited) keeps its payload as-is."""
    from benchmark.records.models.runtime_config_record import (
        KIND_TRACE_REPORT,
        SCOPE_GLOBAL,
        RuntimeConfigRecord,
        trace_report_record_key,
    )
    from benchmark.domain.runtime_config_store import upsert_runtime_config
    from benchmark.domain.trace_report_config import (
        TraceReportSeed,
        ensure_default_trace_report,
    )

    await upsert_runtime_config(
        kind=KIND_TRACE_REPORT,
        record_key=trace_report_record_key("custom-ds"),
        payload={
            "name": "custom-ds",
            "benchmark_name": "amap_agent_benchmark",
            "source": "kafka",
            "schedule_time": "07:30",
            "report_html": "my_report.html",
            "sample_size": 500,
        },
        scope_key=SCOPE_GLOBAL,
    )
    name = await ensure_default_trace_report(
        TraceReportSeed(
            name="custom-ds",
            benchmark_name="amap_agent_benchmark",
            source="kafka",
            project_id="proj-1",
        )
    )
    assert name == "custom-ds"
    row = await RuntimeConfigRecord.filter(
        kind=KIND_TRACE_REPORT,
        record_key=trace_report_record_key("custom-ds"),
    ).first()
    assert row is not None and isinstance(row.payload, dict)
    assert row.payload["schedule_time"] == "07:30"
    assert row.payload["report_html"] == "my_report.html"
    assert row.payload["sample_size"] == 500


async def test_ensure_default_trace_report_disambiguates_name_collision() -> None:
    """When a same-name dataset is already taken by another benchmark, default registration adds a benchmark prefix to disambiguate."""
    from benchmark.domain.trace_report_config import (
        TraceReportSeed,
        ensure_default_trace_report,
        list_db_trace_report_items,
    )

    await ensure_default_trace_report(
        TraceReportSeed(
            name="shared-ds",
            benchmark_name="amap_agent_benchmark",
            source="kafka",
            project_id="proj-1",
        )
    )
    other = await ensure_default_trace_report(
        TraceReportSeed(
            name="shared-ds",
            benchmark_name="example_benchmark",
            source="kafka",
            project_id="proj-1",
        )
    )
    assert other == "example_benchmark-shared-ds"
    names = {item.name for item in await list_db_trace_report_items()}
    assert names == {"shared-ds", "example_benchmark-shared-ds"}


async def test_trace_report_defaults_and_evaluator_mapping() -> None:
    """The default schedule point is 09:00; the evaluator stem reverse-looks-up the benchmark name, and no matching component returns None."""
    from benchmark.domain.trace_report_config import (
        TraceReportConfig,
        benchmark_name_for_evaluator,
    )

    assert TraceReportConfig().schedule_time == "09:00"
    assert benchmark_name_for_evaluator("amap_agent") == "amap_agent_benchmark"
    assert benchmark_name_for_evaluator("no_such_evaluator") is None


async def test_run_counts_only_terminal_after_registration() -> None:
    """Count definition: terminal runs that carry a registration snapshot, deduped cases, and only counts after the registration time.

    An intermediate state (running) does not count; before the anchor does not count; a failed terminal state does count; several
    criteria of the same case count as 1. A batch with no snapshot does not count.
    """
    from datetime import UTC, datetime, timedelta

    from benchmark.records.models.benchmark_run_item import BenchmarkRunItem
    from benchmark.records.models.runtime_config_record import RuntimeConfigRecord

    anchor = datetime(2026, 10, 20, tzinfo=UTC)
    await RuntimeConfigRecord.create(
        kind="benchmark_agent",
        record_key="benchmark_agent:reg-se-1",
        payload={"id": "reg-se-1", "name": "search_eval"},
        scope_key="agent:reg-se-1",
    )
    await RuntimeConfigRecord.filter(record_key="benchmark_agent:reg-se-1").update(
        created_at=anchor,
    )

    async def _run(
        run_id: str,
        status: BenchmarkStatus,
        created: datetime,
        items: list[tuple[str, str]],
    ) -> None:
        run = await BenchmarkRun.create(
            run_id=run_id,
            benchmark_name="search_eval",
            service_name="search_ai_card",
            dataset_name="ds",
            status=status,
            total_samples=1,
            total_evaluators=1,
            config_json={"agent_registration": {"id": "reg-se-1"}},
        )
        await BenchmarkRun.filter(id=run.id).update(created_at=created)
        for item_id, criterion_id in items:
            await BenchmarkRunItem.create(
                run_id=run_id,
                item_id=item_id,
                criterion_id=criterion_id,
                evaluator_ref="judge",
                status="success",
            )

    await _run(
        "se-old-done",
        BenchmarkStatus.COMPLETED,
        anchor - timedelta(days=1),
        [("case-old", "c1")],
    )
    await _run(
        "se-new-done",
        BenchmarkStatus.COMPLETED,
        anchor + timedelta(hours=1),
        [("case-1", "c1"), ("case-1", "c2"), ("case-2", "c1")],
    )
    await _run(
        "se-new-fail",
        BenchmarkStatus.FAILED,
        anchor + timedelta(hours=2),
        [("case-3", "c1")],
    )
    await _run(
        "se-new-running",
        BenchmarkStatus.RUNNING,
        anchor + timedelta(hours=3),
        [("case-4", "c1")],
    )

    entries = {
        (item["benchmarkName"], item.get("registrationId")): item
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert entries[("search_eval", "reg-se-1")]["evaluationCount"] == 3
    assert entries[("search_eval", "reg-se-1")]["batchCount"] == 3
    assert entries[("search_eval", "reg-se-1")]["streamingCount"] == 0
    assert ("search_eval", None) not in entries


async def test_run_counts_split_by_registration_and_merge_runs() -> None:
    """Isolated instances are counted separately: one row per registration id, and several runs of the same registration are merged; a batch with no snapshot does not count."""
    from datetime import UTC, datetime, timedelta

    from benchmark.records.models.benchmark_run_item import BenchmarkRunItem
    from benchmark.records.models.runtime_config_record import RuntimeConfigRecord

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

    async def _run(
        run_id: str,
        created: datetime,
        items: list[str],
        registration_id: str | None,
    ) -> None:
        config = {"agent_registration": {"id": registration_id}} if registration_id else {}
        run = await BenchmarkRun.create(
            run_id=run_id,
            benchmark_name="example_benchmark",
            service_name="svc",
            dataset_name="ds",
            status=BenchmarkStatus.COMPLETED,
            total_samples=1,
            total_evaluators=1,
            config_json=config,
        )
        await BenchmarkRun.filter(id=run.id).update(created_at=created)
        for item_id in items:
            await BenchmarkRunItem.create(
                run_id=run_id,
                item_id=item_id,
                criterion_id="c1",
                evaluator_ref="judge",
                status="success",
            )

    await _run("reg-a-run-1", anchor + timedelta(hours=1), ["c1", "c2"], "reg-a")
    await _run("reg-a-run-2", anchor + timedelta(hours=2), ["c3"], "reg-a")
    await _run("reg-a-before-anchor", anchor - timedelta(days=1), ["c9"], "reg-a")
    await _run("reg-b-run-1", anchor + timedelta(hours=1), ["c1", "c2"], "reg-b")
    await _run("plain-run", anchor + timedelta(hours=1), ["c1", "c2", "c3"], None)

    entries = {
        (item["benchmarkName"], item.get("registrationId")): item
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert entries[("example_benchmark", "reg-a")]["evaluationCount"] == 3
    assert entries[("example_benchmark", "reg-a")]["batchCount"] == 3
    assert entries[("example_benchmark", "reg-a")]["streamingCount"] == 0
    assert entries[("example_benchmark", "reg-b")]["evaluationCount"] == 2
    assert entries[("example_benchmark", "reg-b")]["batchCount"] == 2
    # A batch with no registration snapshot does not enter the by-name total (so it does not pollute a newly registered card)
    assert ("example_benchmark", None) not in entries
    assert ("example", None) not in entries


async def test_list_runs_filters_by_task_and_page_summaries_carry_task_id() -> None:
    """The data source for reports isolated by task: list_runs filters by the snapshot eval_task_id, and the page row exposes taskId."""
    from benchmark.domain.benchmark_run_page import page_run_summary

    for run_id, task_id in (("t-a1", "task-a"), ("t-b1", "task-b"), ("t-none", None)):
        await BenchmarkRun.create(
            run_id=run_id,
            benchmark_name="search_eval",
            service_name="search_ai_card",
            dataset_name="ds",
            status=BenchmarkStatus.COMPLETED,
            total_samples=1,
            total_evaluators=1,
            config_json={"eval_task_id": task_id} if task_id else {},
        )
    service = BenchmarkService()
    mine = await service.list_runs(ListRunsQuery(task_id="task-a"))
    assert [run.run_id for run in mine] == ["t-a1"]
    summaries = {run.run_id: page_run_summary(run) for run in await service.list_runs()}
    assert summaries["t-a1"]["taskId"] == "task-a"
    assert summaries["t-none"]["taskId"] is None


async def test_run_counts_include_streaming_trace_calls() -> None:
    """A realtime stream is counted by the canonical evaluator_name, and also emits the execution-side alias row and the per-chain components."""
    for index in range(3):
        await TraceRecord.create(
            trace_id=f"trace-{index}",
            evaluator_name="amap_agent",
            agent_name="amap-agent",
            status=TraceStatus.SCORED,
        )
    counts = {
        item["benchmarkName"]: item for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert counts["amap_agent"]["evaluationCount"] == 3
    assert counts["amap_agent"]["batchCount"] == 0
    assert counts["amap_agent"]["streamingCount"] == 3
    assert counts["amap_agent_benchmark"]["evaluationCount"] == 3
    assert counts["amap_agent_benchmark"]["streamingCount"] == 3


async def test_run_counts_skip_non_canonical_evaluator_name() -> None:
    """A misspelled execution-side name and the example stem do not enter sharing-by-name (demo recognizes only registrationId)."""
    for index in range(2):
        await TraceRecord.create(
            trace_id=f"bad-{index}",
            evaluator_name="example_benchmark",
            agent_name="example",
            status=TraceStatus.SCORED,
        )
    await TraceRecord.create(
        trace_id="good-1",
        evaluator_name="example",
        agent_name="example",
        status=TraceStatus.SCORED,
    )
    counts = {
        item["benchmarkName"]: item["evaluationCount"]
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert "example" not in counts
    assert "example_benchmark" not in counts


async def test_run_counts_amap_agent_hyphen_and_underscore_aliases() -> None:
    """After amap-agent / amap_agent are normalized, both shape aliases are emitted so the card can match."""
    for index in range(3):
        await TraceRecord.create(
            trace_id=f"amap-hyphen-{index}",
            evaluator_name="amap-agent",
            agent_name="amap-agent",
            status=TraceStatus.SCORED,
        )
    counts = {
        item["benchmarkName"]: item["evaluationCount"]
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert counts["amap_agent"] == 3
    assert counts["amap-agent"] == 3
    assert counts["amap_agent_benchmark"] == 3


async def test_run_counts_skip_trace_daily_report_runs() -> None:
    """A T+1 daily report and a batch with no snapshot do not count; only a batch with a registration snapshot enters the registrationId row."""
    from benchmark.records.models.benchmark_run_item import BenchmarkRunItem

    daily = await BenchmarkRun.create(
        run_id="amap-agent-sampling-2026-10-20-20261019",
        benchmark_name="amap-agent-sampling-2026-10-20",
        service_name="amap_agent",
        dataset_name="trace_pipeline:kafka",
        status=BenchmarkStatus.COMPLETED,
        total_samples=1,
        total_evaluators=1,
        config_json={"trace_daily_report": True},
    )
    await BenchmarkRunItem.create(
        run_id=daily.run_id,
        item_id="case-1",
        criterion_id="c1",
        evaluator_ref="amap_agent",
        status="success",
    )
    orphan = await BenchmarkRun.create(
        run_id="amap-batch-orphan",
        benchmark_name="amap_agent_benchmark",
        service_name="amap_agent",
        dataset_name="ds",
        status=BenchmarkStatus.COMPLETED,
        total_samples=1,
        total_evaluators=1,
        config_json={},
    )
    await BenchmarkRunItem.create(
        run_id=orphan.run_id,
        item_id="case-2",
        criterion_id="c1",
        evaluator_ref="amap_agent",
        status="success",
    )
    registered = await BenchmarkRun.create(
        run_id="amap-batch-reg",
        # Newly stored: the column = registrationId; the card name comes from the snapshot name / benchmark_services
        benchmark_name="reg-amap-1",
        service_name="amap_agent",
        dataset_name="ds",
        status=BenchmarkStatus.COMPLETED,
        total_samples=1,
        total_evaluators=1,
        config_json={
            "agent_registration": {
                "id": "reg-amap-1",
                "name": "amap_agent_benchmark",
            },
            "benchmark_services": "amap_agent_benchmark",
        },
    )
    await BenchmarkRunItem.create(
        run_id=registered.run_id,
        item_id="case-3",
        criterion_id="c1",
        evaluator_ref="amap_agent",
        status="success",
    )
    counts = {
        (item["benchmarkName"], item.get("registrationId")): item["evaluationCount"]
        for item in await BenchmarkService().count_runs_by_benchmark()
    }
    assert ("amap-agent-sampling-2026-10-20", None) not in counts
    assert ("amap_agent", None) not in counts
    assert ("amap_agent_benchmark", None) not in counts
    assert counts[("amap_agent_benchmark", "reg-amap-1")] == 1


def _scope(scope_id: str = "webdev-agent") -> AgentScopeDocument:
    return AgentScopeDocument(
        id=scope_id,
        owner="qa",
        createdAt="2026-10-19 14:00:00",
    )


async def test_scope_roundtrip_and_builtin_guards() -> None:
    """A custom evaluation target: persist → list → query as an isolation key; a built-in item cannot be created again or deleted."""
    saved = await save_scope(_scope())
    assert saved["id"] == "webdev-agent"
    assert saved["schema_version"] == 1
    listed = await list_scopes()
    assert [item["id"] for item in listed] == ["webdev-agent"]
    # A custom scope is a legal isolation key: an empty list (no datasets under that scope) is not a 400
    assert await list_datasets("webdev-agent") == []
    # Saving the same id again revives/updates idempotently, and the list still has one row
    await save_scope(_scope())
    assert [item["id"] for item in await list_scopes()] == ["webdev-agent"]
    # A built-in item cannot be created again and cannot be deleted
    with pytest.raises(ConsoleError) as caught:
        await save_scope(_scope("amap-agent"))
    assert caught.value.status_code == 409
    with pytest.raises(ConsoleError) as caught:
        await delete_scope("skill")
    assert caught.value.status_code == 409


async def test_scope_delete_then_unregistered_scope_rejected() -> None:
    """After a soft delete the scope is no longer legal: querying datasets under it is a 400 (delete the data before the target)."""
    await save_scope(_scope("webdev-agent"))
    deleted = await delete_scope("webdev-agent")
    assert deleted == {"deleted": True}
    assert await list_scopes() == []
    with pytest.raises(ConsoleError) as caught:
        await list_datasets("webdev-agent")
    assert caught.value.status_code == 400
    with pytest.raises(ConsoleError) as caught:
        await delete_scope("webdev-agent")
    assert caught.value.status_code == 404


async def test_list_datasets_rejects_unknown_scope() -> None:
    """Querying an unregistered custom scope is a 400 (the message says to create it first)."""
    with pytest.raises(ConsoleError) as caught:
        await list_datasets("ghost-agent")
    assert caught.value.status_code == 400


async def test_dataset_project_id_roundtrip() -> None:
    """The dataset projectId (the routing key of the realtime evaluation chain) round-trips with the document."""
    saved = await save_dataset(
        _dataset(
            source="realtime",
            status="sampling",
            projectId="proj-webdev-3.0",
            samplingRatio=0.1,
        ),
    )
    assert saved["projectId"] == "proj-webdev-3.0"


async def test_delete_task_only_soft_deletes_document() -> None:
    """Deleting a task only soft-deletes the task document: after deletion the list no longer returns it, and no run side effects are attached.

    Evaluation results under that task name (runs / reports) are kept and left untouched. The delete API does not return
    a manualRunId, and the API layer must not terminate a run because of it.
    """
    await save_task(_task(manualRunId="bm-42", status="active"))
    assert await delete_task("task-1") == {"deleted": True}
    assert await list_tasks() == []


async def test_list_schedulable_heals_terminal_scheduled_task() -> None:
    """When a scheduled task was mistakenly written complete/failed, the schedule list corrects it back to active and includes it in the schedule."""
    await save_task(_task(trigger="daily", status="complete"))
    tasks = await list_schedulable_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == "active"
    viewed = await get_task("task-1")
    assert viewed["status"] == "active"


async def test_list_schedulable_skips_stopped() -> None:
    await save_task(_task(trigger="weekly", status="stopped"))
    assert await list_schedulable_tasks() == []


async def _seed_realtime_eval_entry(project: str, evaluator: str, agent: str) -> str:
    """Seed one active kind=evaluator entry (the driver of the realtime-stream execution surface)."""
    key = evaluator_record_key(project, evaluator, agent)
    await upsert_runtime_config(
        kind=KIND_EVALUATOR,
        record_key=key,
        payload={
            "evaluatorName": evaluator,
            "sampling": 0.1,
            **({"agentName": agent} if agent else {}),
        },
        scope_key=project_scope_key(project),
    )
    return key


async def _seed_trace_report_entry(name: str, benchmark: str = "amap_agent_benchmark") -> str:
    """Seed one active kind=trace_report entry (a T+1 report item)."""
    key = trace_report_record_key(name)
    await upsert_runtime_config(
        kind=KIND_TRACE_REPORT,
        record_key=key,
        payload={
            "name": name,
            "benchmark_name": benchmark,
            "source": "kafka",
            "project_id": "proj-r",
        },
        scope_key=project_scope_key("proj-r"),
    )
    return key


async def _entry_alive(key: str) -> bool:
    row = await RuntimeConfigRecord.filter(record_key=key, deleted_at__isnull=True).first()
    return row is not None and not row.is_deleted


async def test_stop_task_cascades_realtime_eval_entry() -> None:
    """Stopping a realtime-stream task also disables the evaluator entry and the T+1 report item; the dataset state is not touched."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        report_key = await _seed_trace_report_entry("poi-search")
        await save_task(_task())

        stopped = await apply_stop_task("task-1")

        assert stopped["status"] == "stopped"
        assert not await _entry_alive(key)
        assert not await _entry_alive(report_key)
        # Stopping a task only converges the execution surface: dataset sampling state is an explicit page intent and stays sampling
        dataset = await find_dataset_document("ds-1")
        assert dataset is not None
        assert dataset.status == "sampling"
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_stop_task_keeps_sampling_state_and_stop_sampling_works() -> None:
    """After the last task is stopped the dataset stays sampling, and the page can still finish converging by stopping sampling."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await save_task(_task())

        await apply_stop_task("task-1")
        dataset = await find_dataset_document("ds-1")
        assert dataset is not None
        assert dataset.status == "sampling"

        stopped = await apply_stop_sampling("ds-1")
        assert stopped["status"] == "ready"
        assert not await _entry_alive(key)
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_stop_task_non_realtime_dataset_keeps_entry() -> None:
    """Stopping a task for a non-realtime dataset does not touch the evaluator entry."""
    try:
        await save_dataset(_dataset())
        key = await _seed_realtime_eval_entry("proj-x", "amap_agent", "amap-agent")
        await save_task(_task())

        await apply_stop_task("task-1")

        assert await _entry_alive(key)
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_stop_sampling_retires_and_start_sampling_revives_entry() -> None:
    """Stop sampling: disable the entry and stop related tasks; the entry is restored only when sampling starts while a task is running."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await save_task(_task())

        stopped = await apply_stop_sampling("ds-1")
        assert stopped["status"] == "ready"
        assert not await _entry_alive(key)
        # Sampling stopped → running tasks that use that sampling are stopped together
        task = await get_task("task-1")
        assert task["status"] == "stopped"

        # Starting sampling when no task is running: do not restore the entry (both switches must be on to evaluate)
        started = await apply_start_sampling("ds-1")
        assert started["status"] == "sampling"
        assert not await _entry_alive(key)

        # After the task runs again (active), starting sampling → the entry is restored
        await save_task(_task(status="active"))
        await apply_stop_sampling("ds-1")
        await save_task(_task(status="active"))
        started = await apply_start_sampling("ds-1")
        assert started["status"] == "sampling"
        assert await _entry_alive(key)
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_stop_sampling_stops_all_active_tasks_of_dataset() -> None:
    """Sampling stopped: every running task under that dataset stops, and tasks of other datasets are unaffected."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        await save_dataset(
            _dataset(
                id="ds-other",
                name="其他实时集",
                source="realtime",
                status="sampling",
                projectId="proj-other",
                evaluatorName="amap_agent",
            ),
        )
        await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await save_task(_task())
        await save_task(_task(id="task-2", name="第二个任务"))
        await save_task(_task(id="task-other", datasetId="ds-other", name="其他任务"))

        await apply_stop_sampling("ds-1")

        assert (await get_task("task-1"))["status"] == "stopped"
        assert (await get_task("task-2"))["status"] == "stopped"
        assert (await get_task("task-other"))["status"] == "active"
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_stop_task_keeps_entry_when_other_task_active() -> None:
    """While another task sharing the same sampling is still running, stopping one task does not touch sampling or the entry."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await save_task(_task())
        await save_task(_task(id="task-2", name="第二个任务"))

        await apply_stop_task("task-1")

        assert await _entry_alive(key)
        dataset = await find_dataset_document("ds-1")
        assert dataset is not None
        assert dataset.status == "sampling"

        # The last running task stops → the entry is disabled (evaluation no longer proceeds), and the state stays sampling
        await apply_stop_task("task-2")
        assert not await _entry_alive(key)
        dataset = await find_dataset_document("ds-1")
        assert dataset is not None
        assert dataset.status == "sampling"
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_remove_last_active_task_stops_sampling_and_entry() -> None:
    """Deleting the last running realtime-stream task: the entry is disabled, and the dataset state stays sampling."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await save_task(_task())

        result = await apply_remove_task("task-1")

        assert result["deleted"] is True
        assert not await _entry_alive(key)
        # Deleting a task only converges the execution surface: the dataset sampling state stays unchanged
        dataset = await find_dataset_document("ds-1")
        assert dataset is not None
        assert dataset.status == "sampling"
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_remove_task_retires_entry_when_dataset_already_deleted() -> None:
    """Delete the dataset first, then the task: still converge orphan evaluators from the soft-deleted dataset."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await save_task(_task())
        await delete_dataset("ds-1")
        assert not await _entry_alive(key)

        # Seed another leftover that simulates "the old logic did not converge when the dataset was deleted"; deleting the task should clear it
        key2 = await _seed_realtime_eval_entry("proj-r", "example", "amap-agent")
        result = await apply_remove_task("task-1")

        assert result["deleted"] is True
        assert not await _entry_alive(key2)
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_delete_agent_cascades_evaluator_by_registration() -> None:
    """Deleting an evaluation agent: converge execution-surface entries by registrationId and the task's datasets."""
    try:
        await save_agent(
            AgentDocument.model_validate(
                {
                    "id": "ba-reg-1",
                    "name": "example",
                    "owner": "eval-team",
                    "registeredAt": "2026-10-12 10:00:00",
                    "agentScope": "amap-agent",
                },
            ),
        )
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="example",
            ),
        )
        reg_key = evaluator_record_key("proj-r", "example", "amap-agent", "ba-reg-1")
        await upsert_runtime_config(
            kind=KIND_EVALUATOR,
            record_key=reg_key,
            payload={
                "evaluatorName": "example",
                "agentName": "amap-agent",
                "agent_registration_id": "ba-reg-1",
                "sampling": 0.1,
            },
            scope_key=project_scope_key("proj-r"),
        )
        default_key = await _seed_realtime_eval_entry("proj-r", "example", "amap-agent")
        await save_task(_task(agentId="ba-reg-1"))

        result = await delete_agent("ba-reg-1")

        assert result["deleted"] is True
        assert not await _entry_alive(reg_key)
        assert not await _entry_alive(default_key)
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_delete_dataset_cascades_realtime_evaluator() -> None:
    """Deleting a realtime dataset also disables the evaluator entry."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")

        result = await delete_dataset("ds-1")

        assert result["deleted"] is True
        assert not await _entry_alive(key)
        assert await find_dataset_document("ds-1") is None
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_remove_task_keeps_entry_when_other_task_active() -> None:
    """While other tasks are still running, deleting one task does not touch sampling or the entry."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await save_task(_task())
        await save_task(_task(id="task-2", name="第二个任务"))

        await apply_remove_task("task-1")

        assert await _entry_alive(key)
        dataset = await find_dataset_document("ds-1")
        assert dataset is not None
        assert dataset.status == "sampling"
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_start_sampling_without_active_task_keeps_entry_retired() -> None:
    """Starting sampling when no task is running does not restore the entry (evaluation runs only when both the realtime stream and the task are enabled)."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="ready",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await retire_runtime_config(KIND_EVALUATOR, key)

        started = await apply_start_sampling("ds-1")

        assert started["status"] == "sampling"
        assert not await _entry_alive(key)
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_rerun_task_revives_realtime_entry() -> None:
    """Running a realtime-stream task again also restores sampling and the entry."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="ready",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="amap_agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        await retire_runtime_config(KIND_EVALUATOR, key)
        await save_task(_task(status="stopped"))

        rerun = await apply_rerun_task("task-1")

        assert rerun["status"] == "active"
        assert await _entry_alive(key)
        dataset = await find_dataset_document("ds-1")
        assert dataset is not None
        assert dataset.status == "sampling"
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_stop_sampling_fallback_matches_agent_name() -> None:
    """An old dataset with no evaluatorName: locate what to disable by scanning (project, agentName)."""
    try:
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
            ),
        )
        key = await _seed_realtime_eval_entry("proj-r", "amap_agent", "amap-agent")
        other = await _seed_realtime_eval_entry("proj-r", "amap_agent", "vui_agent")

        await apply_stop_sampling("ds-1")

        assert not await _entry_alive(key)
        assert await _entry_alive(other)  # Entries of other agents are left alone
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_collection_start_persists_ready_document() -> None:
    await save_dataset(_dataset())
    saved = await apply_start_collection("ds-1")
    assert saved["status"] == "collecting"
    assert saved["collection"]["done"] is False


def test_agent_report_binding_accepts_class_names() -> None:
    agent = AgentDocument.model_validate(
        {
            "id": "ba-report",
            "name": "judge",
            "owner": "eval-team",
            "registeredAt": "2026-10-12 10:00:00",
            "agentScope": "amap-agent",
            "rubricClass": "business.runs.search_eval.rubric:SearchEvalRubric",
            "rubricDocUrl": "  ",
            "generateReport": True,
            "reportParser": (
                "benchmark.reporting.framework.unified_report.report_parser:SearchEvalReportParser"
            ),
            "reportHtml": "amap_agent_daily.html",
        }
    )
    assert agent.report_html == "amap_agent_daily.html"
    assert agent.rubric_doc_url is None
    assert agent.generate_report is True
    assert agent.report_parser.endswith("SearchEvalReportParser")


def test_agent_without_report_omits_parser() -> None:
    agent = AgentDocument.model_validate(
        {
            "id": "ba-plain",
            "name": "judge",
            "owner": "eval-team",
            "registeredAt": "2026-10-12 10:00:00",
            "agentScope": "amap-agent",
            "rubricClass": "business.runs.search_eval.rubric:SearchEvalRubric",
        }
    )
    assert agent.generate_report is False
    assert agent.report_parser is None


def test_agent_keeps_parallel_evaluators_and_two_reports() -> None:
    judge = {
        "execution": "sync",
        "placement": "local",
        "classRef": "business.runs.search_eval.local_judge:SearchEvalLocalJudge",
    }
    second_judge = {
        "execution": "async",
        "placement": "remote",
        "classRef": "benchmark.runs.common.execution_classes:RemoteJudge",
        "targetUrl": "https://aone.example/judge",
    }
    attribution = {
        "execution": "sync",
        "placement": "local",
        "classRef": "benchmark.runs.common.execution_classes:LocalAttribution",
    }
    agent = AgentDocument.model_validate(
        {
            "id": "ba-parallel",
            "name": "judge",
            "owner": "eval-team",
            "registeredAt": "2026-10-12 10:00:00",
            "agentScope": "amap-agent",
            "rubricClass": "business.runs.search_eval.rubric:SearchEvalRubric",
            "judges": [judge, second_judge],
            "attributions": [attribution],
            "generateReport": True,
            "reportParser": (
                "benchmark.reporting.framework.unified_report.report_parser:SearchEvalReportParser"
            ),
            "generateAttributionReport": True,
            "attributionReportParser": (
                "benchmark.reporting.framework.unified_report.report_parser:IEvalReportParser"
            ),
            "attributionReportHtml": "benchmark_report.html",
        }
    )
    assert agent.judge is not None
    assert agent.judge.class_ref == judge["classRef"]
    assert len(agent.judges) == 2
    assert agent.attribution is not None
    assert len(agent.attributions) == 1
    assert agent.attribution_report_parser is not None
    assert agent.attribution_report_parser.endswith("IEvalReportParser")
    assert agent.attribution_report_html == "benchmark_report.html"


def test_legacy_single_binding_expands_into_lists() -> None:
    agent = AgentDocument.model_validate(
        {
            "id": "ba-legacy",
            "name": "judge",
            "owner": "eval-team",
            "registeredAt": "2026-10-12 10:00:00",
            "agentScope": "amap-agent",
            "judge": {
                "execution": "sync",
                "placement": "local",
                "classRef": "business.runs.search_eval.local_judge:SearchEvalLocalJudge",
            },
        }
    )
    assert agent.judge is not None
    assert len(agent.judges) == 1
    assert agent.judges[0].class_ref == agent.judge.class_ref


def test_duplicate_evaluators_are_rejected() -> None:
    binding = {
        "execution": "sync",
        "placement": "local",
        "classRef": "business.runs.search_eval.local_judge:SearchEvalLocalJudge",
    }
    with pytest.raises(ValidationError):
        AgentDocument.model_validate(
            {
                "id": "ba-dup",
                "name": "judge",
                "owner": "eval-team",
                "registeredAt": "2026-10-12 10:00:00",
                "agentScope": "amap-agent",
                "judges": [binding, binding],
            }
        )


def test_agent_report_binding_rejects_alias_and_unchecked_parser() -> None:
    with pytest.raises(ValidationError):
        AgentDocument.model_validate(
            {
                "id": "ba-bad",
                "name": "judge",
                "owner": "eval-team",
                "registeredAt": "2026-10-12 10:00:00",
                "agentScope": "amap-agent",
                "rubricClass": "search_eval_rubric",
            }
        )
    with pytest.raises(ValidationError):
        AgentDocument.model_validate(
            {
                "id": "ba-bad",
                "name": "judge",
                "owner": "eval-team",
                "registeredAt": "2026-10-12 10:00:00",
                "agentScope": "amap-agent",
                "generateReport": False,
                "reportParser": (
                    "benchmark.reporting.framework.unified_report.report_parser:DefaultReportParser"
                ),
            }
        )


def _legacy_agent_document() -> AgentDocument:
    """An agent document that still carries execution-chain fields left over from the page (written before capabilities were managed by the server)."""
    return AgentDocument.model_validate(
        {
            "id": "ba-legacy",
            "name": "amap_agent_benchmark",
            "owner": "eval-team",
            "registeredAt": "2026-10-12 10:00:00",
            "agentScope": "amap-agent",
            "generateReport": True,
            "reportParser": (
                "benchmark.reporting.framework.unified_report.report_parser:IEvalReportParser"
            ),
            "rubricDocUrl": "https://docs.example/rubric",
            "judge": {
                "execution": "sync",
                "placement": "local",
                "classRef": ("business.runs.search_eval.local_judge:SearchEvalLocalJudge"),
            },
        }
    )


def test_strip_agent_execution_fields_clears_legacy_bindings() -> None:
    """Strip capability execution-chain fields before writing: judge / report / doc links are all cleared."""
    stripped = strip_agent_execution_fields(_legacy_agent_document())
    assert stripped.judge is None
    assert stripped.judges == []
    assert stripped.attribution is None
    assert stripped.attributions == []
    assert stripped.report_parser is None
    assert stripped.report_html is None
    assert stripped.generate_report is False
    assert stripped.attribution_report_parser is None
    assert stripped.attribution_report_html is None
    assert stripped.rubric_doc_url is None
    # Agent-shaped fields such as name / rubricClass / owner are kept
    assert stripped.name == "amap_agent_benchmark"
    assert stripped.owner == "eval-team"


async def test_save_agent_strips_execution_fields() -> None:
    """save_agent strips execution-chain fields before persisting (optional fields disappear, and list/bool return to their defaults)."""
    saved = await save_agent(_legacy_agent_document())
    assert saved["name"] == "amap_agent_benchmark"
    assert "judge" not in saved
    assert saved["judges"] == []
    assert "reportParser" not in saved
    assert "rubricDocUrl" not in saved
    assert saved["generateReport"] is False


def _plain_agent(
    *,
    doc_id: str,
    name: str = "amap_agent_benchmark",
    registered_at: str = "2026-10-21 10:00:00",
    is_default: bool = False,
    agent_scope: str = "amap-agent",
) -> AgentDocument:
    return AgentDocument.model_validate(
        {
            "id": doc_id,
            "name": name,
            "owner": "eval-team",
            "registeredAt": registered_at,
            "isDefault": is_default,
            "agentScope": agent_scope,
        }
    )


async def test_save_agent_plain_same_name_reuses_existing_id() -> None:
    """Same name, no customization: a new ba-* request idempotently reuses the existing document id and does not open another registration."""
    first = await save_agent(_plain_agent(doc_id="ba-111", registered_at="2026-10-21 10:00:00"))
    assert first["id"] == "ba-111"
    again = await save_agent(
        _plain_agent(
            doc_id="ba-999",
            registered_at="2026-10-27 10:00:00",
        )
    )
    assert again["id"] == "ba-111"
    assert again["registeredAt"] == "2026-10-21 10:00:00"
    docs = await list_agent_documents()
    plain = [item for item in docs if item.name == "amap_agent_benchmark"]
    assert len(plain) == 1
    assert plain[0].id == "ba-111"


async def test_save_agent_plain_reuse_prefers_default() -> None:
    """When several uncustomized same-name rows already coexist, reuse prefers isDefault."""
    # A direct upsert bypasses save_agent idempotency, simulating duplicate ba-* rows from before the migration.
    for doc in (
        _plain_agent(doc_id="ba-older"),
        _plain_agent(doc_id="ba-default", is_default=True),
    ):
        await upsert_runtime_config(
            kind=KIND_BENCHMARK_AGENT,
            record_key=benchmark_agent_record_key(doc.id),
            payload=doc.model_dump(by_alias=True, exclude_none=True),
            scope_key=SCOPE_GLOBAL,
        )
    reused = await save_agent(_plain_agent(doc_id="ba-fresh"))
    assert reused["id"] == "ba-default"


@pytest.mark.asyncio
async def test_stop_task_with_stream_tailed_entry() -> None:
    await save_dataset(
        _dataset(
            source="realtime",
            status="sampling",
            projectId="proj-r",
            agentName="amap-agent",
            evaluatorName="amap_agent",
            streamInstanceId="ds-rt-123",
        )
    )
    tail = compose_evaluator_key_tail("reg-1", "ds-rt-123")
    key = evaluator_record_key("proj-r", "amap_agent", "amap-agent", tail)
    await upsert_runtime_config(
        kind="evaluator",
        record_key=key,
        payload={
            "evaluatorName": "amap_agent",
            "sampling": 0.1,
            "agentName": "amap-agent",
            "ingest": "kafka",
        },
        scope_key="project:proj-r",
    )
    await save_task(_task())
    await refresh_eval_config_snapshot()
    store = EvalConfigStore()
    assert store.get_configs("proj-r")

    await apply_stop_task("task-1")

    row = await RuntimeConfigRecord.filter(record_key=key).first()
    assert row is not None and row.is_deleted
    assert store.get_configs("proj-r") == []


async def test_save_agent_offline_cascades_evaluator_by_registration() -> None:
    """Registration taken offline (status=offline): cascade-disable realtime entries bound to that registration and refresh the snapshot."""
    try:
        await save_agent(
            AgentDocument.model_validate(
                {
                    "id": "ba-off-1",
                    "name": "example",
                    "owner": "eval-team",
                    "registeredAt": "2026-10-12 10:00:00",
                    "agentScope": "amap-agent",
                },
            ),
        )
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="example",
            ),
        )
        reg_key = evaluator_record_key("proj-r", "example", "amap-agent", "ba-off-1")
        await upsert_runtime_config(
            kind=KIND_EVALUATOR,
            record_key=reg_key,
            payload={
                "evaluatorName": "example",
                "agentName": "amap-agent",
                "agent_registration_id": "ba-off-1",
                "sampling": 0.1,
                "ingest": "kafka",
            },
            scope_key=project_scope_key("proj-r"),
        )
        await save_task(_task(agentId="ba-off-1"))
        await refresh_eval_config_snapshot()
        store = EvalConfigStore()
        assert store.get_configs("proj-r")

        await save_agent(
            AgentDocument.model_validate(
                {
                    "id": "ba-off-1",
                    "name": "example",
                    "owner": "eval-team",
                    "registeredAt": "2026-10-12 10:00:00",
                    "agentScope": "amap-agent",
                    "status": "offline",
                },
            ),
        )

        row = await RuntimeConfigRecord.filter(record_key=reg_key).first()
        assert row is not None and row.is_deleted
        assert store.get_configs("proj-r") == []
    finally:
        replace_eval_config_file(EvalConfigFile())


async def test_save_agent_online_keeps_evaluator_entry() -> None:
    """An online save does not cascade: realtime entries stay valid (the symmetric case of offline not auto-reviving them)."""
    try:
        await save_agent(
            AgentDocument.model_validate(
                {
                    "id": "ba-on-1",
                    "name": "example",
                    "owner": "eval-team",
                    "registeredAt": "2026-10-12 10:00:00",
                    "agentScope": "amap-agent",
                },
            ),
        )
        await save_dataset(
            _dataset(
                source="realtime",
                status="sampling",
                projectId="proj-r",
                agentName="amap-agent",
                evaluatorName="example",
            ),
        )
        reg_key = evaluator_record_key("proj-r", "example", "amap-agent", "ba-on-1")
        await upsert_runtime_config(
            kind=KIND_EVALUATOR,
            record_key=reg_key,
            payload={
                "evaluatorName": "example",
                "agentName": "amap-agent",
                "agent_registration_id": "ba-on-1",
                "sampling": 0.1,
                "ingest": "kafka",
            },
            scope_key=project_scope_key("proj-r"),
        )
        await refresh_eval_config_snapshot()

        await save_agent(
            AgentDocument.model_validate(
                {
                    "id": "ba-on-1",
                    "name": "example",
                    "owner": "eval-team",
                    "registeredAt": "2026-10-12 10:00:00",
                    "agentScope": "amap-agent",
                    "status": "online",
                },
            ),
        )
        row = await RuntimeConfigRecord.filter(record_key=reg_key).first()
        assert row is not None and not row.is_deleted
        assert EvalConfigStore().get_configs("proj-r")
    finally:
        replace_eval_config_file(EvalConfigFile())
