"""STEP-10: Benchmark sampling, batch completion (incl. DEAD), T+1 cron."""

from __future__ import annotations

from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.benchmark.api import create_benchmark_app
from llmobserverjudge.benchmark.sampling import stratified_sample, target_sample_size
from llmobserverjudge.benchmark.service import BenchmarkService
from llmobserverjudge.benchmark.signal import InMemoryBatchSignal
from llmobserverjudge.contracts.models import BatchJobStatus


def test_target_sample_size_semantics() -> None:
    assert target_sample_size(100, rate=0.1, min_size=5000) == 100  # < min → all
    assert target_sample_size(100_000, rate=0.1, min_size=5000) == 10000
    # count×rate < min_size → use min_size
    assert target_sample_size(20_000, rate=0.1, min_size=5000) == 5000


def test_stratified_sample_deterministic() -> None:
    rows = []
    for i in range(200):
        rows.append(
            {
                "trace_id": f"{i:032x}"[:32],
                "eval_agent_id": "judge_safety" if i % 2 == 0 else "judge_quality",
                "score_status": "scored",
                "normalized_score": 0.5 + (i % 10) * 0.05,
                "tenant_id": "t1",
                "date": "2026-07-16",
            }
        )
    s1, m1 = stratified_sample(rows, rate=0.1, min_size=5000)
    s2, _ = stratified_sample(rows, rate=0.1, min_size=5000)
    assert len(s1) == 200  # count < min_size
    assert [r["trace_id"] for r in s1] == [r["trace_id"] for r in s2]
    assert m1["total"] == 200

    big = rows * 40  # 8000
    s3, m3 = stratified_sample(big, rate=0.1, min_size=5000)
    assert len(s3) == 5000  # max(800, 5000)=5000
    assert m3["sampled"] == 5000


@pytest.mark.asyncio
async def test_batch_signal_dead_counts() -> None:
    sig = InMemoryBatchSignal()
    await sig.init_batch("b1", total=3)
    assert await sig.incr("b1", "completed") == "PENDING"
    assert await sig.incr("b1", "failed") == "PENDING"
    assert await sig.incr("b1", "dead") == "COMPLETE"
    assert await sig.incr("b1", "completed") == "ALREADY_SIGNALED"
    assert await sig.try_claim_benchmark("b1") is True
    assert await sig.try_claim_benchmark("b1") is False


@pytest.mark.asyncio
async def test_batch_eval_generates_report() -> None:
    svc = BenchmarkService()
    job = await svc.trigger_batch_eval(
        tenant_id="t1",
        name="demo",
        traces=[
            {
                "trace_id": "0123456789abcdef0123456789abcdef",
                "trace_summary": {
                    "status": "OK",
                    "response": "a sufficiently long answer",
                },
            },
            {
                "trace_id": "1123456789abcdef0123456789abcdef",
                "trace_summary": {
                    "status": "OK",
                    "response": "another long enough answer",
                },
            },
        ],
        eval_configs=[{"agent_id": "judge_quality", "model": "default-judge"}],
    )
    assert job.status == BatchJobStatus.COMPLETED
    report = await svc.store.get_report_by_batch(job.batch_job_id)
    assert report is not None
    assert report.summary["sample_size"] >= 1
    assert report.sampling_meta.get("dead_count") == 0

    # idempotent: second claim skipped
    again = await svc._on_batch_completed(job.batch_job_id)
    assert again is not None
    assert again.report_id == report.report_id


@pytest.mark.asyncio
async def test_dead_tasks_still_complete_batch() -> None:
    svc = BenchmarkService()
    job = await svc.trigger_batch_eval(
        tenant_id="t1",
        name="dead-demo",
        traces=[
            {
                "trace_id": "0123456789abcdef0123456789abcdef",
                "trace_summary": {"status": "OK", "response": "hello world answer"},
            },
            {
                "trace_id": "2123456789abcdef0123456789abcdef",
                "trace_summary": {"status": "OK", "response": "hello world answer"},
            },
        ],
        eval_configs=[{"agent_id": "judge_quality", "model": "default-judge"}],
        execute=False,
    )
    tasks = await svc.store.list_tasks(job.batch_job_id)
    assert len(tasks) == 2
    # complete one via run, kill the other
    await svc._run_task(job, tasks[0])
    sig = await svc.mark_task_dead(job.batch_job_id, tasks[1].task_id)
    assert sig == "COMPLETE"
    report = await svc.store.get_report_by_batch(job.batch_job_id)
    assert report is not None
    assert report.sampling_meta["dead_count"] == 1
    assert any("dead_count" in i for i in report.insights)


@pytest.mark.asyncio
async def test_daily_benchmark_cron() -> None:
    svc = BenchmarkService()
    day = date(2026, 7, 16)
    rows = [
        {
            "tenant_id": "t1",
            "date": day.isoformat(),
            "trace_id": f"{i:032x}"[:32],
            "eval_agent_id": "judge_quality",
            "score_status": "scored",
            "normalized_score": 0.7,
            "is_bad_case": False,
        }
        for i in range(100)
    ]
    await svc.scores.add_many(rows)
    report = await svc.trigger_daily_benchmark("t1", day, rate=0.1, min_size=5000)
    assert report.date == day.isoformat()
    assert report.sampling_meta["total"] == 100
    assert report.sampling_meta["sampled"] == 100
    assert report.summary["mean_score"] == 0.7


@pytest.mark.asyncio
async def test_benchmark_api() -> None:
    svc = BenchmarkService()
    app = create_benchmark_app(svc)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/api/v1/batch-eval/trigger",
            json={
                "tenant_id": "t1",
                "name": "api-batch",
                "traces": [
                    {
                        "trace_id": "0123456789abcdef0123456789abcdef",
                        "trace_summary": {
                            "status": "OK",
                            "response": "long enough for quality heuristic",
                        },
                    }
                ],
                "eval_configs": [
                    {"agent_id": "judge_quality", "model": "default-judge"}
                ],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["report_id"]
        gr = await client.get(f"/api/v1/benchmarks/{body['report_id']}")
        assert gr.status_code == 200
        assert gr.json()["summary"]["sample_size"] >= 1
