"""SQL-backed Benchmark report store (subset)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.contracts.models import BenchmarkReport
from llmobserverjudge.shared.tables import BenchmarkReportRow


class SqlBenchmarkReportStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    def _to_model(self, row: BenchmarkReportRow) -> BenchmarkReport:
        payload = dict(row.payload or {})
        return BenchmarkReport(
            report_id=row.report_id,
            tenant_id=row.tenant_id,
            batch_job_id=row.batch_job_id,
            date=row.date,
            summary=payload.get("summary") or {},
            dimensions=payload.get("dimensions") or {},
            comparison=payload.get("comparison") or {},
            bad_case_analysis=payload.get("bad_case_analysis") or {},
            human_annotation_summary=payload.get("human_annotation_summary") or {},
            insights=list(payload.get("insights") or []),
            sampling_meta=payload.get("sampling_meta") or {},
            created_at=row.created_at,
        )

    async def save_report(self, report: BenchmarkReport) -> None:
        async with self._sf() as session:
            session.add(
                BenchmarkReportRow(
                    report_id=report.report_id,
                    tenant_id=report.tenant_id,
                    batch_job_id=report.batch_job_id,
                    date=report.date,
                    payload={
                        "summary": report.summary,
                        "dimensions": report.dimensions,
                        "comparison": report.comparison,
                        "bad_case_analysis": report.bad_case_analysis,
                        "human_annotation_summary": report.human_annotation_summary,
                        "insights": report.insights,
                        "sampling_meta": report.sampling_meta,
                    },
                    created_at=report.created_at,
                )
            )
            await session.commit()

    async def get_report(self, report_id: str) -> BenchmarkReport | None:
        async with self._sf() as session:
            q = await session.execute(
                select(BenchmarkReportRow).where(BenchmarkReportRow.report_id == report_id)
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None

    async def get_report_by_batch(self, batch_job_id: str) -> BenchmarkReport | None:
        async with self._sf() as session:
            q = await session.execute(
                select(BenchmarkReportRow).where(
                    BenchmarkReportRow.batch_job_id == batch_job_id
                )
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None
