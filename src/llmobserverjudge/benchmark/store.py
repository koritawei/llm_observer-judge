"""In-memory stores for batch jobs, eval score corpus, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from llmobserverjudge.contracts.models import (
    BatchEvalTask,
    BatchJob,
    BenchmarkReport,
)


@dataclass
class InMemoryEvalScoreStore:
    """Stand-in for CH eval_scores — rows used by T+1 sampling."""

    _rows: list[dict[str, Any]] = field(default_factory=list)

    async def add(self, row: dict[str, Any]) -> None:
        self._rows.append(dict(row))

    async def add_many(self, rows: list[dict[str, Any]]) -> None:
        self._rows.extend(dict(r) for r in rows)

    async def query(
        self, tenant_id: str, day: date
    ) -> list[dict[str, Any]]:
        day_s = day.isoformat()
        return [
            r
            for r in self._rows
            if r.get("tenant_id") == tenant_id and str(r.get("date") or "") == day_s
        ]


@dataclass
class InMemoryBenchmarkStore:
    jobs: dict[str, BatchJob] = field(default_factory=dict)
    tasks: dict[str, BatchEvalTask] = field(default_factory=dict)
    reports: dict[str, BenchmarkReport] = field(default_factory=dict)
    _by_batch_tasks: dict[str, list[str]] = field(default_factory=dict)

    async def save_job(self, job: BatchJob) -> None:
        self.jobs[job.batch_job_id] = job

    async def get_job(self, batch_job_id: str) -> BatchJob | None:
        return self.jobs.get(batch_job_id)

    async def save_task(self, task: BatchEvalTask) -> None:
        self.tasks[task.task_id] = task
        self._by_batch_tasks.setdefault(task.batch_job_id, [])
        if task.task_id not in self._by_batch_tasks[task.batch_job_id]:
            self._by_batch_tasks[task.batch_job_id].append(task.task_id)

    async def list_tasks(self, batch_job_id: str) -> list[BatchEvalTask]:
        ids = self._by_batch_tasks.get(batch_job_id, [])
        return [self.tasks[i] for i in ids if i in self.tasks]

    async def save_report(self, report: BenchmarkReport) -> None:
        self.reports[report.report_id] = report

    async def get_report(self, report_id: str) -> BenchmarkReport | None:
        return self.reports.get(report_id)

    async def get_report_by_batch(self, batch_job_id: str) -> BenchmarkReport | None:
        for r in self.reports.values():
            if r.batch_job_id == batch_job_id:
                return r
        return None
