"""SQL-backed PromptService (subset of design §5.3) + in-memory A/B experiments."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.eval_worker.prompt import (
    PromptExperiment,
    PromptMeta,
    PromptVersion,
)
from llmobserverjudge.shared.tables import PromptVersionRow


def _bucket(trace_id: str) -> int:
    return int(hashlib.md5(trace_id.encode(), usedforsecurity=False).hexdigest(), 16) % 100


class SqlPromptService:
    """Durable prompt versions; A/B experiments held in-process for now."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory
        self._meta: dict[str, PromptMeta] = {}
        self._experiments: dict[str, PromptExperiment] = {}

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
        self._meta[pid] = PromptMeta(
            prompt_id=pid, name=name or pid, category=category, tenant_id=tenant_id
        )
        async with self._sf() as session:
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
                select(PromptVersionRow).where(PromptVersionRow.version_id == version_id)
            )
            target = q.scalar_one_or_none()
            if target is None or target.prompt_id != prompt_id:
                raise KeyError(f"version not found: {version_id}")
            if target.status not in ("DRAFT", "EXPERIMENTAL"):
                raise ValueError(f"Cannot publish version in status {target.status}")
            await session.execute(
                update(PromptVersionRow)
                .where(
                    PromptVersionRow.prompt_id == prompt_id,
                    PromptVersionRow.status == "ACTIVE",
                )
                .values(status="DEPRECATED")
            )
            target.status = "ACTIVE"
            target.published_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(target)
            return self._to_model(target)

    async def publish_experimental(self, prompt_id: str, version_id: str) -> PromptVersion:
        async with self._sf() as session:
            q = await session.execute(
                select(PromptVersionRow).where(PromptVersionRow.version_id == version_id)
            )
            target = q.scalar_one_or_none()
            if target is None or target.prompt_id != prompt_id:
                raise KeyError(f"version not found: {version_id}")
            if target.status != "DRAFT":
                raise ValueError(f"Cannot mark experimental from status {target.status}")
            target.status = "EXPERIMENTAL"
            target.published_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(target)
            return self._to_model(target)

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
        by_pid: dict[str, list[PromptVersionRow]] = {}
        for r in rows:
            by_pid.setdefault(r.prompt_id, []).append(r)
        out: list[dict[str, Any]] = []
        for pid, vers in by_pid.items():
            meta = self._meta.get(pid) or PromptMeta(prompt_id=pid, name=pid)
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
        for exp in self._experiments.values():
            if exp.prompt_id == prompt_id and exp.status == "RUNNING":
                raise ValueError("prompt already has a RUNNING experiment")
        if treatment.status == "DRAFT":
            await self.publish_experimental(prompt_id, treatment_version_id)
        exp = PromptExperiment(
            experiment_id=str(uuid4()),
            prompt_id=prompt_id,
            control_version_id=control_version_id,
            treatment_version_id=treatment_version_id,
            traffic_pct=traffic_pct,
            metric_primary=metric_primary,
        )
        self._experiments[exp.experiment_id] = exp
        return exp

    async def list_experiments(
        self, *, prompt_id: str | None = None, status: str | None = None
    ) -> list[PromptExperiment]:
        rows = list(self._experiments.values())
        if prompt_id:
            rows = [e for e in rows if e.prompt_id == prompt_id]
        if status:
            rows = [e for e in rows if e.status == status]
        rows.sort(key=lambda e: e.started_at, reverse=True)
        return rows

    async def get_experiment(self, experiment_id: str) -> PromptExperiment | None:
        return self._experiments.get(experiment_id)

    async def conclude_experiment(
        self,
        experiment_id: str,
        *,
        winner: str = "control",
        conclusion: str | None = None,
    ) -> PromptExperiment:
        exp = self._experiments.get(experiment_id)
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
                await session.commit()
        exp.status = "CONCLUDED"
        exp.ended_at = datetime.now(UTC)
        exp.winner_version_id = win_id
        exp.conclusion = conclusion or f"winner={winner}"
        return exp

    async def resolve_for_trace(
        self, prompt_id: str, trace_id: str
    ) -> PromptVersion | None:
        running = next(
            (
                e
                for e in self._experiments.values()
                if e.prompt_id == prompt_id and e.status == "RUNNING"
            ),
            None,
        )
        if running is not None:
            if _bucket(trace_id) < running.traffic_pct:
                treatment = await self.get_version(running.treatment_version_id)
                if treatment is not None:
                    return treatment
            control = await self.get_version(running.control_version_id)
            if control is not None:
                return control
        return await self.get_active_version(prompt_id)
