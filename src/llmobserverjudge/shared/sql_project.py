"""Project registry — one tenant owns many projects; names unique per tenant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llmobserverjudge.shared.project import (
    DEFAULT_PROJECT_ID,
    DEFAULT_PROJECT_NAME,
    Project,
    new_project_id,
    normalize_project_name,
)
from llmobserverjudge.shared.tables import ProjectRow


@runtime_checkable
class ProjectStore(Protocol):
    async def ensure_default(self, tenant_id: str) -> Project: ...

    async def get(self, tenant_id: str, project_id: str) -> Project | None: ...

    async def get_by_name(self, tenant_id: str, name: str) -> Project | None: ...

    async def list_for_tenant(self, tenant_id: str) -> list[Project]: ...

    async def create(
        self, tenant_id: str, name: str, *, project_id: str | None = None
    ) -> Project: ...

    async def resolve(
        self, tenant_id: str, *, project_id: str | None = None, name: str | None = None
    ) -> Project: ...


@dataclass
class InMemoryProjectStore:
    # key: tenant_id\\0project_id -> Project
    _by_id: dict[str, Project] = field(default_factory=dict)
    # key: tenant_id:lower(name) -> project_id
    _by_name: dict[str, str] = field(default_factory=dict)

    def _id_key(self, tenant_id: str, project_id: str) -> str:
        return f"{tenant_id}\0{project_id}"

    def _name_key(self, tenant_id: str, name: str) -> str:
        return f"{tenant_id}:{normalize_project_name(name).lower()}"

    async def ensure_default(self, tenant_id: str) -> Project:
        existing = await self.get(tenant_id, DEFAULT_PROJECT_ID)
        if existing is not None:
            return existing
        return await self.create(
            tenant_id, DEFAULT_PROJECT_NAME, project_id=DEFAULT_PROJECT_ID
        )

    async def get(self, tenant_id: str, project_id: str) -> Project | None:
        return self._by_id.get(self._id_key(tenant_id, project_id))

    async def get_by_name(self, tenant_id: str, name: str) -> Project | None:
        pid = self._by_name.get(self._name_key(tenant_id, name))
        if not pid:
            return None
        return await self.get(tenant_id, pid)

    async def list_for_tenant(self, tenant_id: str) -> list[Project]:
        prefix = f"{tenant_id}\0"
        rows = [p for k, p in self._by_id.items() if k.startswith(prefix)]
        rows.sort(key=lambda p: (p.name != DEFAULT_PROJECT_NAME, p.name.lower()))
        return rows

    async def create(
        self, tenant_id: str, name: str, *, project_id: str | None = None
    ) -> Project:
        n = normalize_project_name(name)
        if await self.get_by_name(tenant_id, n) is not None:
            raise ValueError(f"project name already exists in tenant: {n}")
        pid = project_id or new_project_id(n)
        key = self._id_key(tenant_id, pid)
        if key in self._by_id:
            raise ValueError(f"project_id already exists in tenant: {pid}")
        proj = Project(project_id=pid, tenant_id=tenant_id, name=n)
        self._by_id[key] = proj
        self._by_name[self._name_key(tenant_id, n)] = pid
        return proj

    async def resolve(
        self, tenant_id: str, *, project_id: str | None = None, name: str | None = None
    ) -> Project:
        if project_id:
            row = await self.get(tenant_id, project_id)
            if row is None:
                if project_id == DEFAULT_PROJECT_ID:
                    return await self.ensure_default(tenant_id)
                raise KeyError(f"project not found: {project_id}")
            return row
        if name:
            row = await self.get_by_name(tenant_id, name)
            if row is None:
                raise KeyError(f"project not found by name: {name}")
            return row
        return await self.ensure_default(tenant_id)


class SqlProjectStore:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    def _to_model(self, row: ProjectRow) -> Project:
        return Project(
            project_id=row.project_id, tenant_id=row.tenant_id, name=row.name
        )

    async def ensure_default(self, tenant_id: str) -> Project:
        existing = await self.get(tenant_id, DEFAULT_PROJECT_ID)
        if existing is not None:
            return existing
        try:
            return await self.create(
                tenant_id, DEFAULT_PROJECT_NAME, project_id=DEFAULT_PROJECT_ID
            )
        except ValueError:
            row = await self.get(tenant_id, DEFAULT_PROJECT_ID)
            if row is None:
                raise
            return row

    async def get(self, tenant_id: str, project_id: str) -> Project | None:
        async with self._sf() as session:
            q = await session.execute(
                select(ProjectRow).where(
                    ProjectRow.tenant_id == tenant_id,
                    ProjectRow.project_id == project_id,
                )
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None

    async def get_by_name(self, tenant_id: str, name: str) -> Project | None:
        n = normalize_project_name(name)
        async with self._sf() as session:
            q = await session.execute(
                select(ProjectRow).where(
                    ProjectRow.tenant_id == tenant_id,
                    ProjectRow.name == n,
                )
            )
            row = q.scalar_one_or_none()
            return self._to_model(row) if row else None

    async def list_for_tenant(self, tenant_id: str) -> list[Project]:
        async with self._sf() as session:
            q = await session.execute(
                select(ProjectRow)
                .where(ProjectRow.tenant_id == tenant_id)
                .order_by(ProjectRow.name.asc())
            )
            rows = [self._to_model(r) for r in q.scalars().all()]
        rows.sort(key=lambda p: (p.name != DEFAULT_PROJECT_NAME, p.name.lower()))
        return rows

    async def create(
        self, tenant_id: str, name: str, *, project_id: str | None = None
    ) -> Project:
        n = normalize_project_name(name)
        if await self.get_by_name(tenant_id, n) is not None:
            raise ValueError(f"project name already exists in tenant: {n}")
        pid = project_id or new_project_id(n)
        if await self.get(tenant_id, pid) is not None:
            raise ValueError(f"project_id already exists in tenant: {pid}")
        async with self._sf() as session:
            session.add(ProjectRow(project_id=pid, tenant_id=tenant_id, name=n))
            try:
                await session.commit()
            except Exception as exc:
                await session.rollback()
                raise ValueError(f"project create failed: {exc}") from exc
        row = await self.get(tenant_id, pid)
        assert row is not None
        return row

    async def resolve(
        self, tenant_id: str, *, project_id: str | None = None, name: str | None = None
    ) -> Project:
        if project_id:
            row = await self.get(tenant_id, project_id)
            if row is None:
                if project_id == DEFAULT_PROJECT_ID:
                    return await self.ensure_default(tenant_id)
                raise KeyError(f"project not found: {project_id}")
            return row
        if name:
            row = await self.get_by_name(tenant_id, name)
            if row is None:
                raise KeyError(f"project not found by name: {name}")
            return row
        return await self.ensure_default(tenant_id)
