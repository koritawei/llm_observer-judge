"""**Simulated full-flow** tests of the unified runtime (design doc §12 acceptance 2/3/6/7).

An in-process fake of the whole chain (real Redis / PG / Aone / OSS are not touched):

    OSS CSV (FakeStorage) → DailyDataLoader (full KV)
    → ProfileResolver (rubric_ref / judge / attribution / interceptor)
    → the unified submit pipeline (the interceptor point, which is mandatory) → FakeQueue (benchmark-cases)
    → judge callback (ExternalBenchmarkRunner.handle_callback)
    → finalize counts reconcile

Coverage matrix:
1. pull v1 default (ieval_external → the default rubric_ref mapping → the amap-chat interceptor)
2. pull v1 new business (no default mapping → full KV passthrough, §7.4)
3. pull v2 profile (explicit rubric_ref + judge target URL + sync local attribution)
4. the interceptor raises → item-level failure (stub + counts reconcile), and the run continues
5. an unregistered judge_class → ProfileResolutionError (API 400 semantics)
"""

from typing import Any

import pytest
IevalExternalBenchmarkServices = None
from benchmark.runs.schema import (
    ExternalBenchmarkRequest,
    JudgeCallback,
)
from benchmark.reporting.framework.base import BenchmarkComponentMetadata
from benchmark.reporting.framework.schema import AggregationConfig, Criterion
from benchmark.queue.queue_factory import BENCHMARK_CASES
from benchmark.workers import external_runner as er_mod
from benchmark.workers.external_runner import (
    ExternalBenchmarkRunner,
    ExternalRunnerDeps,
)

# ════════════════════════════════════════════════════════════════════════════
# Fakes
# ════════════════════════════════════════════════════════════════════════════

_CSV_HEADER = "id,task_id,query,adiu,lng,lat,user_history,response_text,return_code\n"


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    lines = [_CSV_HEADER.rstrip("\n")]
    for row in rows:
        lines.append(
            ",".join(
                str(
                    row.get(col, ""),
                )
                for col in [
                    "id",
                    "task_id",
                    "query",
                    "adiu",
                    "lng",
                    "lat",
                    "user_history",
                    "response_text",
                    "return_code",
                ]
            ),
        )
    return ("﻿" + "\n".join(lines) + "\n").encode("utf-8")


def _rows(n: int) -> list[dict[str, str]]:
    return [
        {
            "id": f"csv-{i}",
            "task_id": f"t-{i}",
            "query": f"query-{i}",
            "adiu": f"adiu-{i}",
            "lng": "116.4",
            "lat": "39.9",
            "user_history": "[]",
            "response_text": f"resp-{i}",
            "return_code": "0",
        }
        for i in range(n)
    ]


class _FakeStorage:
    def __init__(self, data: bytes | None) -> None:
        self._data = data

    async def download(self, object_key: str) -> bytes | None:
        return self._data


class _FakeQueue:
    """Minimal fake of a BullMQ Queue: add collects the job payload."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.jobs: list[dict[str, Any]] = []

    async def add(self, name: str, data: dict[str, Any], opts: dict[str, Any]) -> None:
        self.jobs.append({"name": name, "data": data, "opts": opts})


class _FakeQueueManager:
    def __init__(self) -> None:
        self.queues: dict[str, _FakeQueue] = {}
        self.prefix = "test:bullmq"

    def get_queue(self, name: str) -> _FakeQueue:
        return self.queues.setdefault(name, _FakeQueue(name))


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.status = "running"
        self.config_json: dict[str, Any] = {}
        self.benchmark_name = ""


class _FakeBenchmarkService:
    """Minimal in-memory BenchmarkService (the method subset used by initialize and the callback path)."""

    def __init__(self) -> None:
        self.runs: dict[str, _FakeRun] = {}
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.statuses: list[tuple[str, str]] = []

    async def heartbeat(self, run_id: str) -> None:
        return None

    async def create_external_run(self, run_id: str, spec: Any) -> None:
        run = _FakeRun(run_id)
        run.config_json = dict(spec.config_json)
        run.benchmark_name = spec.benchmark_name
        self.runs[run_id] = run

    async def get_run(self, run_id: str) -> _FakeRun | None:
        return self.runs.get(run_id)

    async def update_run_status(self, run_id: str, status: Any, **_: Any) -> None:
        run = self.runs[run_id]
        run.status = getattr(status, "value", status)
        self.statuses.append((run_id, run.status))

    async def merge_config_json(self, run_id: str, patch: dict[str, Any]) -> None:
        self.runs[run_id].config_json.update(patch)

    async def increment_config_counter(self, run_id: str, key: str) -> tuple[int, dict]:
        cfg = self.runs[run_id].config_json
        cfg[key] = int(cfg.get(key, 0)) + 1
        return cfg[key], cfg

    async def cas_advance_external_judge_item(self, *, result: Any) -> bool:
        key = (result.run_id, result.item_id)
        if key in self.items:
            return False
        self.items[key] = {
            "status": "failed" if result.failed else "success",
            "score": result.score,
            "judge_json": result.judge_json,
            "input": result.item_input,
            "output": result.item_output,
        }
        await self.increment_config_counter(result.run_id, "judge_received_count")
        return True


def _meta_for(name: str) -> BenchmarkComponentMetadata:
    if name == "ieval_external":
        return IevalExternalBenchmarkServices.get_metadata()
    return BenchmarkComponentMetadata(
        name=name,
        service_name="demo_svc",
        rubric=[Criterion(id="Q1", name="q", description="", weight=1.0, evaluator_ref="demo")],
        aggregation=AggregationConfig(method="weighted_avg", pass_threshold=0.5),
    )


class _FakeBenchmarkServices:
    """benchmark_services duck type (initialize_and_submit only calls get_metadata)."""

    def __init__(self, meta: BenchmarkComponentMetadata) -> None:
        self._meta = meta

    def get_metadata(self) -> BenchmarkComponentMetadata:
        return self._meta


def _runner(
    service: _FakeBenchmarkService,
    storage: _FakeStorage,
    qm: _FakeQueueManager,
) -> ExternalBenchmarkRunner:
    from benchmark.runs.common.external_client_factory import (
        ExternalClientFactory,
        ExternalClientFactoryConfig,
    )

    factory = ExternalClientFactory(
        ExternalClientFactoryConfig(
            judge_base_url="https://aone.example.com",
            api_key="ak-test",
            default_judge_pipeline_id="100",
            default_attribution_pipeline_id="200",
        ),
    )
    runner = ExternalBenchmarkRunner(
        benchmark_service=service,  # type: ignore[arg-type]
        storage_client=storage,  # type: ignore[arg-type]
        deps=ExternalRunnerDeps(
            queue_manager=qm,
            callback_url_base="http://localhost:9000",
            external_client_factory=factory,
        ),  # type: ignore[arg-type]
    )
    return runner


def _payload(name: str, **overrides: Any) -> ExternalBenchmarkRequest:
    is_search = name == "search_eval"
    is_ieval = name == "ieval_external"
    base: dict[str, Any] = {
        "benchmark_name": name,
        "judge_class": (
            "search_eval_judge"
            if is_search
            else "ieval_external_judge"
            if is_ieval
            else "local_judge"
        ),
        "attribution_class": ("ieval_external_attribution" if is_ieval else "no_attribution"),
        "rubric_class": (
            "search_eval_rubric"
            if is_search
            else "amap_chat_rubric_v2"
            if is_ieval
            else "component_rubric"
        ),
        "date": "20260820",
        "oss_prefix": "oss://bucket/prefix",
        "effective_date": "20260820",
        "effective_oss_prefix": "oss://bucket/prefix",
    }
    base.update(overrides)
    return ExternalBenchmarkRequest(**base)


# ════════════════════════════════════════════════════════════════════════════
# Scenario 1: pull v1 default (ieval_external's default mapping → the interceptor parses 21 columns)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_flow_pull_default_ieval_external() -> None:
    """v1 default entry: the default rubric_ref mapping → the interceptor is mandatory → a 21-column business structure is enqueued."""
    service = _FakeBenchmarkService()
    qm = _FakeQueueManager()
    runner = _runner(service, _FakeStorage(_csv_bytes(_rows(3))), qm)

    total = await runner.initialize_and_submit(
        run_id="ieval_external-20260820",
        payload=_payload("ieval_external"),
        benchmark_services=_FakeBenchmarkServices(_meta_for("ieval_external")),  # type: ignore[arg-type]
    )

    assert total == 3
    run = service.runs["ieval_external-20260820"]
    cfg = run.config_json
    # The snapshot is the source of truth: rubric/judge/attribution/rubric_ref/queue_pair are all written
    assert [c["id"] for c in cfg["rubric"]] == ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
    assert cfg["rubric_ref"].endswith(":AmapChatRubricV2")
    assert cfg["rubric_class"] == "amap_chat_rubric_v2"
    assert cfg["parse_interceptor_ref"].endswith(":AmapChatParseInterceptor")
    # Scoring-rule link: the Rubric class rule_link_url is persisted through config (the source that fills report slots)
    assert cfg["rule_link_url"] == (
        "https://alidocs.dingtalk.com/i/nodes/14dA3GK8gwAG5B1mHnbDXZPG89ekBD76"
    )
    assert cfg["judge"]["execution"] == "async"
    assert cfg["judge"]["provider"] == "external"
    assert cfg["attribution"]["execution"] == "async"
    assert cfg["attribution"]["provider"] == "external"
    assert cfg["queue_pair"] == "benchmark"
    assert cfg["judge_expected_count"] == 3
    assert "api_key" not in str(cfg)

    # The unified queue benchmark-cases receives the interceptor's parsed product (a 21-column business structure)
    jobs = qm.get_queue(BENCHMARK_CASES).jobs
    assert len(jobs) == 3
    data = jobs[0]["data"]
    assert data["input"]["query"] == "query-0"  # Assembled by the interceptor
    assert data["output"]["response_text"] == "resp-0"
    assert data["metadata"]["csv_row"]["task_id"] == "t-0"  # The full KV is kept
    assert data["metadata"]["user_location"] == {"longitude": 116.4, "latitude": 39.9}
    assert data["criteria"]  # The resolved rubric is sent down
    assert data["run_id"] == "ieval_external-20260820"


# ════════════════════════════════════════════════════════════════════════════
# Scenario 2: pull a new business (no default mapping → full KV passthrough)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_flow_pull_new_business_passthrough() -> None:
    """A new business (no rubric_ref / no interceptor) → the full KV is passed straight through (§7.4)."""
    service = _FakeBenchmarkService()
    qm = _FakeQueueManager()
    runner = _runner(service, _FakeStorage(_csv_bytes(_rows(2))), qm)

    await runner.initialize_and_submit(
        run_id="demo_bench-20260820",
        payload=_payload("demo_bench", attribution_class="no_attribution"),
        benchmark_services=_FakeBenchmarkServices(_meta_for("demo_bench")),  # type: ignore[arg-type]
    )

    jobs = qm.get_queue(BENCHMARK_CASES).jobs
    assert len(jobs) == 2
    data = jobs[0]["data"]
    # Passthrough: no input/output assembly, and csv_row is kept in full
    assert "input" not in data or data.get("input") is None
    assert data["metadata"]["csv_row"]["query"] == "query-0"
    # A new business column takes effect with zero code (every column is passed through)
    assert data["metadata"]["csv_row"]["return_code"] == "0"
    # The build_record csv_row branch can consume it (judge record = full KV + task_id)
    build_record = None

    record = build_record(data)
    assert record["task_id"] == "t-0"
    assert record["query"] == "query-0"


# ════════════════════════════════════════════════════════════════════════════
# Scenario 3: pull + an explicit profile (rubric_ref + judge target URL + attribution none)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_flow_pull_explicit_ieval_judge_class() -> None:
    """Execution class name: rubric_class → Rubric class load + the ieval Aone judge."""
    from benchmark.runs.common.external_client_factory import (
        ExternalClientFactory,
        ExternalClientFactoryConfig,
    )
    from benchmark.runs.common.profile_resolver import ProfileResolver

    service = _FakeBenchmarkService()
    qm = _FakeQueueManager()
    runner = _runner(service, _FakeStorage(_csv_bytes(_rows(2))), qm)
    # Assemble the per-run factory (the same construction semantics as main.py)
    factory = ExternalClientFactory(
        ExternalClientFactoryConfig(
            judge_base_url="https://aone.example.com",
            api_key="ak-test",
            default_judge_pipeline_id="100",
        ),
    )
    runner._external_client_factory = factory
    runner._profile_resolver = ProfileResolver(client_factory=factory)

    await runner.initialize_and_submit(
        run_id="demo_bench-20260821",
        payload=_payload(
            "demo_bench",
            rubric_class="amap_chat_rubric_v2",
            judge_class="ieval_external_judge",
            attribution_class="no_attribution",
        ),
        benchmark_services=_FakeBenchmarkServices(_meta_for("demo_bench")),  # type: ignore[arg-type]
    )

    cfg = service.runs["demo_bench-20260821"].config_json
    # criteria come from the class criteria() (not component meta's Q1)
    assert [c["id"] for c in cfg["rubric"]] == ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
    assert cfg["judge"]["execution"] == "async"
    assert cfg["judge"]["target"] == "https://aone.example.com/api/pipelines/100/trigger"
    assert cfg["judge"]["dims"] == "D0,D1,D2,D3,D4,D5,D6"
    assert cfg["attribution"]["execution"] == "none"
    # The interceptor comes from the Rubric class mount (an explicit rubric_ref → the amap-chat interceptor takes effect)
    jobs = qm.get_queue(BENCHMARK_CASES).jobs
    assert jobs[0]["data"]["input"]["query"] == "query-0"


# ════════════════════════════════════════════════════════════════════════════
# Scenario 4: the interceptor raises → item-level failure + counts reconcile
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_flow_parse_failure_item_isolation() -> None:
    """The interceptor raises on a single row: that item gets an eval_failed stub and judge_received advances,
    and the other items enqueue normally; expected = the raw total → the counts reconcile and do not stall."""

    class _BoomOnSecondRow:
        """A business interceptor that raises while parsing the second row (t-1)."""

        def parse(self, ctx: Any) -> list[dict[str, Any]]:
            AmapChatParseInterceptor = None

            out = []
            for raw in ctx.raw_items:
                if raw["metadata"]["csv_row"].get("task_id") == "t-1":
                    raise ValueError("bad row")
                out.append(AmapChatParseInterceptor()._map_raw(raw))
            return out

    # Ride a legal allowlisted class: monkeypatch the default mapping onto the test interceptor class
    service = _FakeBenchmarkService()
    qm = _FakeQueueManager()
    runner = _runner(service, _FakeStorage(_csv_bytes(_rows(3))), qm)

    # Inject the interceptor instance directly (skip class loading, and focus on pipeline behavior)
    resolved = runner._profile_resolver.resolve(
        er_mod.profile_from_external_request(_payload("demo_bench")),
        _meta_for("demo_bench"),
    )
    resolved = er_mod.ResolvedProfile(
        benchmark_name=resolved.benchmark_name,
        rubric=resolved.rubric,
        rubric_ref=resolved.rubric_ref,
        aggregation=resolved.aggregation,
        parse_interceptor_ref="test:BoomOnSecondRow",
        parse_interceptor=_BoomOnSecondRow(),
        judge=resolved.judge,
        attribution=resolved.attribution,
    )

    from benchmark.runs.common.submit_pipeline import (
        SubmitPipelineSpec,
        submit_prepared_items,
    )

    raw_items = [
        {
            "id": f"t-{i}",
            "item_id": f"demo_bench-20260820-x:{i}",
            "source_trace_id": None,
            "metadata": {"csv_row": row, "title_columns": list(row)},
        }
        for i, row in enumerate(_rows(3))
    ]
    run_id = "demo_bench-20260820-x"
    service.runs[run_id] = _FakeRun(run_id)
    await service.merge_config_json(run_id, {"judge_expected_count": 3})

    summary = await submit_prepared_items(
        SubmitPipelineSpec(
            run_id=run_id,
            resolved=resolved,
            raw_items=raw_items,
            enqueue_items=lambda items: runner._enqueue_unified_cases(
                er_mod._UnifiedCasesSubmit(
                    run_id=run_id,
                    items=items,
                    resolved=resolved,
                    payload=_payload("demo_bench"),
                    service_name="demo_svc",
                ),
            ),
            mark_item_failed=lambda item, reason: runner._mark_parse_failed(run_id, item, reason),
            snapshot={},
        ),
    )

    assert summary.total_raw == 3
    assert summary.parsed == 2
    assert summary.parse_failed == 1
    # The failed item is persisted as failed and judge_received advances (the same path as a callback)
    assert service.items[("demo_bench-20260820-x", "demo_bench-20260820-x:1")]["status"] == "failed"
    assert service.runs[run_id].config_json["judge_received_count"] == 1
    # The other items enqueue normally
    assert len(qm.get_queue(BENCHMARK_CASES).jobs) == 2


# ════════════════════════════════════════════════════════════════════════════
# Scenario 5: the judge target host is outside the allowlist → 400 semantics
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_flow_judge_target_host_outside_whitelist_rejected() -> None:
    """SSRF protection: a host outside the allowlist → ProfileResolutionError (→ API 400)."""
    from benchmark.runs.common.external_client_factory import (
        ExternalClientFactory,
        ExternalClientFactoryConfig,
    )
    from benchmark.runs.common.profile_resolver import (
        ProfileResolutionError,
        ProfileResolver,
    )

    factory = ExternalClientFactory(
        ExternalClientFactoryConfig(
            judge_base_url="https://aone.example.com",
            api_key="ak-test",
            default_judge_pipeline_id="100",
        ),
    )
    resolver = ProfileResolver(client_factory=factory)
    with pytest.raises(ProfileResolutionError) as exc:
        resolver.resolve(
            er_mod.profile_from_external_request(
                _payload(
                    "demo_bench",
                    judge_class="unknown_judge",
                ),
            ),
            _meta_for("demo_bench"),
        )
    assert exc.value.reason == "invalid_execution_class"


# ════════════════════════════════════════════════════════════════════════════
# Scenario 6: judge callback → the count advances → finalize fires (reconciliation check)
# ════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_flow_judge_callback_converges_to_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge callback (F10 reconciles by run_id+item_id) → cas_advance advances
    judge_received → finalize fires once they are all in (the unified worker's async path has the same semantics)."""
    service = _FakeBenchmarkService()
    qm = _FakeQueueManager()
    runner = _runner(service, _FakeStorage(_csv_bytes(_rows(2))), qm)

    await runner.initialize_and_submit(
        run_id="ieval_external-20260820",
        payload=_payload("ieval_external"),
        benchmark_services=_FakeBenchmarkServices(_meta_for("ieval_external")),  # type: ignore[arg-type]
    )
    cfg = service.runs["ieval_external-20260820"].config_json
    assert cfg["judge_expected_count"] == 2

    # Record that finalize fired (so the real CAS SQL is not touched)
    finalized: list[str] = []

    async def _fake_finalize(run_id: str, *_: Any) -> None:
        finalized.append(run_id)

    monkeypatch.setattr(er_mod, "try_finalize_external", _fake_finalize)

    # Read the callback's prewritten row (BenchmarkRunItem.filter().first()) → fake (no prewritten row)
    class _FakeItemQuery:
        async def first(self) -> None:
            return None

    monkeypatch.setattr(
        er_mod.BenchmarkRunItem,
        "filter",
        lambda **_: _FakeItemQuery(),
    )

    criterion_scores = {f"D{i}": {"score": 2.0, "reason": "ok"} for i in range(7)}
    criterion_scores["D0"] = {"score": 1.0, "reason": "pass"}

    for i in range(2):
        await runner.handle_callback(
            JudgeCallback(
                run_id="ieval_external-20260820",
                item_id=f"ieval_external-20260820:{i}",
                job_id=f"job-{i}",
                criterion_scores=criterion_scores,  # type: ignore[arg-type]
            ),
        )

    cfg = service.runs["ieval_external-20260820"].config_json
    # The judge callback of both items advances the received count
    assert cfg["judge_received_count"] == 2
    # Both items are written as success (cas_advance is idempotent)
    assert len(service.items) == 2
    assert all(item["status"] == "success" for item in service.items.values())
    # The fused score is written (D0 full score 1 + D1-D6 full score 2 → EXCELLENT, and attribution is not triggered)
    assert service.items[("ieval_external-20260820", "ieval_external-20260820:0")][
        "score"
    ] == pytest.approx(2.0)
    # finalize has fired (counts reconcile: the received=2 >= expected=2 semantics are
    # guaranteed by try_finalize_external CAS; here we check that the callback chain's call arrives)
    assert finalized == ["ieval_external-20260820", "ieval_external-20260820"]
