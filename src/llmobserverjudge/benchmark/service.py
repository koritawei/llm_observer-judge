"""Benchmark batch + T+1 cron orchestration (design §11)."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any
from uuid import uuid4

from llmobserverjudge.benchmark.agent import BenchmarkAgent
from llmobserverjudge.benchmark.sampling import stratified_sample
from llmobserverjudge.benchmark.signal import BatchSignal, InMemoryBatchSignal
from llmobserverjudge.benchmark.store import (
    InMemoryBenchmarkStore,
    InMemoryEvalScoreStore,
)
from llmobserverjudge.contracts.models import (
    BatchEvalTask,
    BatchJob,
    BatchJobStatus,
    BatchTaskStatus,
    BenchmarkReport,
    EvalConfig,
    EvalJob,
    Priority,
)
from llmobserverjudge.eval_worker.graph import run_eval_graph
from llmobserverjudge.eval_worker.judges import JudgeModelRouter, default_judge_router

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Independent of Trace Graph — batch eval + daily benchmark."""

    def __init__(
        self,
        *,
        store: InMemoryBenchmarkStore | None = None,
        scores: InMemoryEvalScoreStore | None = None,
        signal: BatchSignal | None = None,
        agent: BenchmarkAgent | None = None,
        judge_router: JudgeModelRouter | None = None,
    ) -> None:
        self.store = store or InMemoryBenchmarkStore()
        self.scores = scores or InMemoryEvalScoreStore()
        self.signal = signal or InMemoryBatchSignal()
        self.agent = agent or BenchmarkAgent()
        self.judge_router = judge_router or default_judge_router()

    async def trigger_batch_eval(
        self,
        *,
        tenant_id: str,
        name: str,
        traces: list[dict[str, Any]],
        eval_configs: list[dict[str, Any]],
        priority: Priority = Priority.P2,
        execute: bool = True,
    ) -> BatchJob:
        """Create BatchJob + tasks; run in-process (prod: MQ eval_jobs)."""
        job = BatchJob(
            batch_job_id=str(uuid4()),
            tenant_id=tenant_id,
            name=name,
            source_config={"trace_count": len(traces)},
            eval_configs=eval_configs,
            priority=priority,
            status=BatchJobStatus.RUNNING if execute else BatchJobStatus.QUEUED,
            total_tasks=0,
            created_at=datetime.now(UTC),
        )
        tasks: list[BatchEvalTask] = []
        for tr in traces:
            tid = str(tr["trace_id"])
            for cfg in eval_configs:
                tasks.append(
                    BatchEvalTask(
                        task_id=str(uuid4()),
                        batch_job_id=job.batch_job_id,
                        trace_id=tid,
                        tenant_id=tenant_id,
                        eval_config_id=str(cfg.get("agent_id") or "eval"),
                        status=BatchTaskStatus.PENDING,
                        trace_summary=dict(tr.get("trace_summary") or tr),
                    )
                )
        job.total_tasks = len(tasks)
        await self.store.save_job(job)
        for t in tasks:
            await self.store.save_task(t)
        await self.signal.init_batch(job.batch_job_id, job.total_tasks)

        if execute:
            for t in tasks:
                await self._run_task(job, t)

        if job.total_tasks == 0:
            job = job.model_copy(
                update={
                    "status": BatchJobStatus.COMPLETED,
                    "completed_at": datetime.now(UTC),
                }
            )
            await self.store.save_job(job)
        return (await self.store.get_job(job.batch_job_id)) or job

    async def _run_task(self, job: BatchJob, task: BatchEvalTask) -> None:
        task = task.model_copy(
            update={
                "status": BatchTaskStatus.RUNNING,
                "dispatched_at": datetime.now(UTC),
                "worker_id": "inline-batch",
            }
        )
        await self.store.save_task(task)
        try:
            cfgs = [
                c
                for c in job.eval_configs
                if c.get("agent_id") == task.eval_config_id
            ] or job.eval_configs[:1]
            eval_configs = [EvalConfig.model_validate(c) for c in cfgs]
            ej = EvalJob(
                trace_id=task.trace_id,
                tenant_id=job.tenant_id,
                trace_summary=task.trace_summary,
                eval_configs=eval_configs,
                priority=job.priority,
                idempotency_key=f"{job.batch_job_id}:{task.task_id}",
                created_at=datetime.now(UTC),
            )
            state = await run_eval_graph(
                ej,
                judge_router=self.judge_router,
                dispatch_client=None,
                seed_prompts=False,
            )
            result = {
                "final_score": state.final_score,
                "bad_case_detected": state.bad_case_detected,
                "eval_scores": state.eval_scores,
            }
            task = task.model_copy(
                update={
                    "status": BatchTaskStatus.COMPLETED,
                    "result": result,
                    "completed_at": datetime.now(UTC),
                }
            )
            await self.store.save_task(task)
            # feed daily corpus
            for es in state.eval_scores:
                await self.scores.add(
                    {
                        "tenant_id": job.tenant_id,
                        "date": datetime.now(UTC).date().isoformat(),
                        "trace_id": task.trace_id,
                        "eval_agent_id": es.get("agent_id"),
                        "score_status": es.get("score_status"),
                        "normalized_score": es.get("normalized_score"),
                        "is_bad_case": es.get("is_bad_case"),
                        "label": es.get("label"),
                    }
                )
            sig = await self.signal.incr(job.batch_job_id, "completed")
        except Exception as e:
            logger.exception("batch_task_failed task=%s", task.task_id)
            attempt = task.attempt + 1
            terminal = (
                BatchTaskStatus.DEAD
                if attempt >= task.max_retries
                else BatchTaskStatus.FAILED
            )
            task = task.model_copy(
                update={
                    "status": terminal,
                    "attempt": attempt,
                    "error": str(e),
                    "completed_at": datetime.now(UTC),
                }
            )
            await self.store.save_task(task)
            field = "dead" if terminal == BatchTaskStatus.DEAD else "failed"
            sig = await self.signal.incr(job.batch_job_id, field)

        if sig == "COMPLETE":
            await self._on_batch_completed(job.batch_job_id)

    async def mark_task_dead(self, batch_job_id: str, task_id: str) -> str:
        """Test/ops helper: force DEAD and advance completion signal."""
        task = self.store.tasks.get(task_id)
        if task is None or task.batch_job_id != batch_job_id:
            raise KeyError(task_id)
        task = task.model_copy(
            update={
                "status": BatchTaskStatus.DEAD,
                "completed_at": datetime.now(UTC),
            }
        )
        await self.store.save_task(task)
        sig = await self.signal.incr(batch_job_id, "dead")
        if sig == "COMPLETE":
            await self._on_batch_completed(batch_job_id)
        return sig

    async def _on_batch_completed(self, batch_job_id: str) -> BenchmarkReport | None:
        if not await self.signal.try_claim_benchmark(batch_job_id):
            logger.info("benchmark_already_processed batch=%s", batch_job_id)
            return await self.store.get_report_by_batch(batch_job_id)

        job = await self.store.get_job(batch_job_id)
        if job is None:
            return None
        snap = await self.signal.snapshot(batch_job_id)
        tasks = await self.store.list_tasks(batch_job_id)
        sample: list[dict[str, Any]] = []
        for t in tasks:
            if not t.result:
                continue
            for es in t.result.get("eval_scores") or []:
                sample.append(
                    {
                        "tenant_id": job.tenant_id,
                        "trace_id": t.trace_id,
                        "eval_agent_id": es.get("agent_id"),
                        "score_status": es.get("score_status"),
                        "normalized_score": es.get("normalized_score"),
                        "is_bad_case": es.get("is_bad_case"),
                        "label": es.get("label"),
                    }
                )
        meta = {
            "total": snap.get("total", 0),
            "sampled": len(sample),
            "actual_sample_rate": 1.0,
            "dead_count": snap.get("dead", 0),
            "failed_count": snap.get("failed", 0),
            "completed_count": snap.get("completed", 0),
            "source": "batch_eval",
        }
        report = await self.agent.generate(
            tenant_id=job.tenant_id,
            sample=sample,
            meta=meta,
            batch_job_id=batch_job_id,
        )
        await self.store.save_report(report)
        job = job.model_copy(
            update={
                "status": BatchJobStatus.COMPLETED,
                "completed_at": datetime.now(UTC),
            }
        )
        await self.store.save_job(job)
        return report

    async def trigger_daily_benchmark(
        self,
        tenant_id: str,
        day: date,
        *,
        rate: float = 0.1,
        min_size: int = 5000,
    ) -> BenchmarkReport:
        """T+1 Cron entry — independent of Trace Graph."""
        rows = await self.scores.query(tenant_id, day)
        sample, meta = stratified_sample(rows, rate=rate, min_size=min_size)
        historical = None
        # pick latest prior report for tenant
        priors = [
            r
            for r in self.store.reports.values()
            if r.tenant_id == tenant_id and r.date and r.date < day.isoformat()
        ]
        if priors:
            historical = max(priors, key=lambda r: r.date or "")

        report = await self.agent.generate(
            tenant_id=tenant_id,
            sample=sample,
            meta=meta,
            day=day.isoformat(),
            historical=historical,
        )
        await self.store.save_report(report)
        return report


def main() -> None:
    print("llmj-benchmark — T+1 cron / batch-eval; not part of Trace Graph")
