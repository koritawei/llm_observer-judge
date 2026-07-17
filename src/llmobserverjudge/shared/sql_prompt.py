"""SQL-backed PromptService — versions + A/B experiments durable."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.eval_worker.prompt import (
    PromptExperiment,
    PromptMeta,
    PromptVersion,
)
from llmobserverjudge.shared.tables import (
    PromptExperimentRow,
    PromptMetaRow,
    PromptVersionRow,
)


def _bucket(trace_id: str) -> int:
    return int(hashlib.md5(trace_id.encode(), usedforsecurity=False).hexdigest(), 16) % 100


class SqlPromptService:
    """Durable prompt versions, meta, and A/B experiments."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    @staticmethod
    def render_template(content: str, **variables: object) -> str:
        from llmobserverjudge.eval_worker.prompt import InMemoryPromptService

        return InMemoryPromptService.render_template(content, **variables)

    def _to_model(self, row: PromptVersionRow) -> PromptVersion:
        return PromptVersion(
            version_id=row.version_id,
            prompt_id=row.prompt_id,
            version=row.version,
            content=row.content,
            status=row.status,
            variables=list(row.variables or []),
            published_at=row.published_at,
        )

    def _exp_to_model(self, row: PromptExperimentRow) -> PromptExperiment:
        return PromptExperiment(
            experiment_id=row.experiment_id,
            prompt_id=row.prompt_id,
            control_version_id=row.control_version_id,
            treatment_version_id=row.treatment_version_id,
            traffic_pct=row.traffic_pct,
            status=row.status,
            metric_primary=row.metric_primary,
            started_at=row.started_at or datetime.now(UTC),
            ended_at=row.ended_at,
            conclusion=row.conclusion,
            winner_version_id=row.winner_version_id,
        )

    async def _upsert_meta(
        self,
        session: AsyncSession,
        *,
        prompt_id: str,
        name: str = "",
        category: str = "JUDGE",
        tenant_id: str | None = None,
    ) -> None:
        q = await session.execute(
            select(PromptMetaRow).where(PromptMetaRow.prompt_id == prompt_id)
        )
        row = q.scalar_one_or_none()
        if row is None:
            session.add(
                PromptMetaRow(
                    prompt_id=prompt_id,
                    name=name or prompt_id,
                    category=category,
                    tenant_id=tenant_id,
                )
            )
        else:
            if name:
                row.name = name
            if category:
                row.category = category
            if tenant_id is not None:
                row.tenant_id = tenant_id

    async def create_prompt(
        self,
        *,
        prompt_id: str | None = None,
        content: str,
        variables: list[str] | None = None,
        name: str = "",
        category: str = "JUDGE",
        tenant_id: str | None = None,
    ) -> PromptVersion:
        pid = prompt_id or str(uuid4())
        async with self._sf() as session:
            await self._upsert_meta(
                session, prompt_id=pid, name=name or pid, category=category, tenant_id=tenant_id
            )
            q = await session.execute(
                select(PromptVersionRow)
                .where(PromptVersionRow.prompt_id == pid)
                .order_by(PromptVersionRow.version.desc())
                .limit(1)
            )
            last = q.scalar_one_or_none()
            ver_num = (last.version + 1) if last else 1
            row = PromptVersionRow(
                version_id=str(uuid4()),
                prompt_id=pid,
                version=ver_num,
                content=content,
                status="DRAFT",
                variables=list(variables or []),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return self._to_model(row)

    async def create_version(
        self,
        prompt_id: str,
        *,
        content: str,
        variables: list[str] | None = None,
        changelog: str | None = None,
    ) -> PromptVersion:
        _ = changelog
        return await self.create_prompt(
            prompt_id=prompt_id, content=content, variables=variables, name=prompt_id
        )

    async def publish(self, prompt_id: str, version_id: str) -> PromptVersion:
        async with self._sf() as session:
            q = await session.execute(
                select(PromptVersionRow).where(PromptVersionRow.prompt_id == prompt_id)
            )
            rows = list(q.scalars().all())
            target = None
            for r in rows:
                if r.version_id == version_id:
                    target = r
                elif r.status == "ACTIVE":
                    r.status = "DEPRECATED"
            if target is None:
                raise KeyError(f"version not found: {version_id}")
            target.status = "ACTIVE"
            target.published_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(target)
            return self._to_model(target)

    async def publish_experimental(self, prompt_id: str, version_id: str) -> PromptVersion:
        async with self._sf() as session:
            q = await session.execute(
                select(PromptVersionRow).where(
                    PromptVersionRow.prompt_id == prompt_id,
                    PromptVersionRow.version_id == version_id,
                )
            )
            row = q.scalar_one_or_none()
            if row is None:
                raise KeyError(f"version not found: {version_id}")
            if row.status == "ACTIVE":
                raise ValueError("cannot mark ACTIVE as EXPERIMENTAL")
            row.status = "EXPERIMENTAL"
            await session.commit()
            await session.refresh(row)
            return self._to_model(row)

    async def get_active_version(self, prompt_id: str) -> PromptVersion | None:
        async with self._sf() as session:
            q = await session.execute(
                select(PromptVersionRow).where(
                    PromptVersionRow.prompt_id == prompt_id,
                    PromptVersionRow.status == "ACTIVE",
                )
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None

    async def get_version(self, version_id: str) -> PromptVersion | None:
        async with self._sf() as session:
            q = await session.execute(
                select(PromptVersionRow).where(PromptVersionRow.version_id == version_id)
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None

    async def list_versions(self, prompt_id: str) -> list[PromptVersion]:
        async with self._sf() as session:
            q = await session.execute(
                select(PromptVersionRow)
                .where(PromptVersionRow.prompt_id == prompt_id)
                .order_by(PromptVersionRow.version.asc())
            )
            return [self._to_model(r) for r in q.scalars().all()]

    async def list_prompts(self, *, tenant_id: str | None = None) -> list[dict[str, Any]]:
        async with self._sf() as session:
            q = await session.execute(select(PromptVersionRow))
            rows = list(q.scalars().all())
            mq = await session.execute(select(PromptMetaRow))
            meta_map = {m.prompt_id: m for m in mq.scalars().all()}
        by_pid: dict[str, list[PromptVersionRow]] = {}
        for r in rows:
            by_pid.setdefault(r.prompt_id, []).append(r)
        out: list[dict[str, Any]] = []
        for pid, vers in by_pid.items():
            meta_row = meta_map.get(pid)
            meta = (
                PromptMeta(
                    prompt_id=pid,
                    name=meta_row.name if meta_row else pid,
                    category=meta_row.category if meta_row else "JUDGE",
                    tenant_id=meta_row.tenant_id if meta_row else None,
                )
            )
            if tenant_id and meta.tenant_id and meta.tenant_id != tenant_id:
                continue
            active = next((v for v in vers if v.status == "ACTIVE"), None)
            out.append(
                {
                    "prompt_id": pid,
                    "name": meta.name,
                    "category": meta.category,
                    "tenant_id": meta.tenant_id,
                    "version_count": len(vers),
                    "active_version_id": active.version_id if active else None,
                }
            )
        out.sort(key=lambda x: x["prompt_id"])
        return out

    async def create_experiment(
        self,
        prompt_id: str,
        *,
        control_version_id: str,
        treatment_version_id: str,
        traffic_pct: int = 20,
        metric_primary: str = "mean_score",
    ) -> PromptExperiment:
        if not 0 <= traffic_pct <= 100:
            raise ValueError("traffic_pct must be 0..100")
        control = await self.get_version(control_version_id)
        treatment = await self.get_version(treatment_version_id)
        if control is None or control.prompt_id != prompt_id:
            raise KeyError("control version not found")
        if treatment is None or treatment.prompt_id != prompt_id:
            raise KeyError("treatment version not found")
        async with self._sf() as session:
            q = await session.execute(
                select(PromptExperimentRow).where(
                    PromptExperimentRow.prompt_id == prompt_id,
                    PromptExperimentRow.status == "RUNNING",
                )
            )
            if q.scalar_one_or_none() is not None:
                raise ValueError("prompt already has a RUNNING experiment")
        if treatment.status == "DRAFT":
            await self.publish_experimental(prompt_id, treatment_version_id)
        exp_id = str(uuid4())
        async with self._sf() as session:
            row = PromptExperimentRow(
                experiment_id=exp_id,
                prompt_id=prompt_id,
                control_version_id=control_version_id,
                treatment_version_id=treatment_version_id,
                traffic_pct=traffic_pct,
                status="RUNNING",
                metric_primary=metric_primary,
                started_at=datetime.now(UTC),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return self._exp_to_model(row)

    async def list_experiments(
        self, *, prompt_id: str | None = None, status: str | None = None
    ) -> list[PromptExperiment]:
        async with self._sf() as session:
            stmt = select(PromptExperimentRow).order_by(PromptExperimentRow.started_at.desc())
            if prompt_id:
                stmt = stmt.where(PromptExperimentRow.prompt_id == prompt_id)
            if status:
                stmt = stmt.where(PromptExperimentRow.status == status)
            q = await session.execute(stmt)
            return [self._exp_to_model(r) for r in q.scalars().all()]

    async def get_experiment(self, experiment_id: str) -> PromptExperiment | None:
        async with self._sf() as session:
            q = await session.execute(
                select(PromptExperimentRow).where(
                    PromptExperimentRow.experiment_id == experiment_id
                )
            )
            row = q.scalar_one_or_none()
            return self._exp_to_model(row) if row else None

    async def conclude_experiment(
        self,
        experiment_id: str,
        *,
        winner: str = "control",
        conclusion: str | None = None,
    ) -> PromptExperiment:
        exp = await self.get_experiment(experiment_id)
        if exp is None:
            raise KeyError("experiment not found")
        if exp.status != "RUNNING":
            raise ValueError(f"experiment status is {exp.status}")
        winner = winner.lower().strip()
        if winner in ("control", "a"):
            win_id, lose_id = exp.control_version_id, exp.treatment_version_id
        elif winner in ("treatment", "b"):
            win_id, lose_id = exp.treatment_version_id, exp.control_version_id
        else:
            raise ValueError("winner must be control|treatment")
        await self.publish(exp.prompt_id, win_id)
        async with self._sf() as session:
            q = await session.execute(
                select(PromptVersionRow).where(PromptVersionRow.version_id == lose_id)
            )
            lose = q.scalar_one_or_none()
            if lose and lose.status != "DEPRECATED":
                lose.status = "DEPRECATED"
            qe = await session.execute(
                select(PromptExperimentRow).where(
                    PromptExperimentRow.experiment_id == experiment_id
                )
            )
            row = qe.scalar_one()
            row.status = "CONCLUDED"
            row.ended_at = datetime.now(UTC)
            row.winner_version_id = win_id
            row.conclusion = conclusion or f"winner={winner}"
            await session.commit()
            await session.refresh(row)
            return self._exp_to_model(row)

    async def resolve_for_trace(
        self, prompt_id: str, trace_id: str
    ) -> PromptVersion | None:
        running = await self.list_experiments(prompt_id=prompt_id, status="RUNNING")
        if running:
            exp = running[0]
            if _bucket(trace_id) < exp.traffic_pct:
                treatment = await self.get_version(exp.treatment_version_id)
                if treatment is not None:
                    return treatment
            control = await self.get_version(exp.control_version_id)
            if control is not None:
                return control
        return await self.get_active_version(prompt_id)
