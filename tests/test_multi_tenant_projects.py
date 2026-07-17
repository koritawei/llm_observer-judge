"""Multi-tenant / multi-project isolation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from llmobserverjudge.runtime.stack import DevStack
from llmobserverjudge.shared.persistence import create_persistence
from llmobserverjudge.shared.project import DEFAULT_PROJECT_ID
from llmobserverjudge.shared.sql_project import InMemoryProjectStore


@pytest.mark.asyncio
async def test_same_project_name_allowed_across_tenants() -> None:
    store = InMemoryProjectStore()
    a = await store.create("tenant-a", "demo")
    b = await store.create("tenant-b", "demo")
    assert a.name == b.name == "demo"
    assert a.tenant_id != b.tenant_id
    da = await store.ensure_default("tenant-a")
    db = await store.ensure_default("tenant-b")
    assert da.project_id == db.project_id == DEFAULT_PROJECT_ID
    assert da.tenant_id == "tenant-a"
    assert db.tenant_id == "tenant-b"


@pytest.mark.asyncio
async def test_project_name_unique_within_tenant() -> None:
    store = InMemoryProjectStore()
    await store.create("t1", "alpha")
    with pytest.raises(ValueError, match="already exists"):
        await store.create("t1", "alpha")


@pytest.mark.asyncio
async def test_devstack_project_isolation_same_trace_id() -> None:
    """Same trace_id under different tenants/projects must not collide."""
    stack = await DevStack.create_from_env(
        api_keys={"ka": "tenant-a", "kb": "tenant-b"},
        force_sample=True,
        persistence=await create_persistence("memory"),
    )
    assert stack.app is not None
    transport = ASGITransport(app=stack.app)
    trace = "0123456789abcdef0123456789abcdef"
    span = "0123456789abcdef"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        p1 = await client.post(
            "/projects/api/v1/projects?tenant_id=tenant-a",
            headers={"X-API-Key": "ka"},
            json={"name": "alpha"},
        )
        assert p1.status_code == 200, p1.text
        pid_a = p1.json()["project_id"]

        p2 = await client.post(
            "/projects/api/v1/projects?tenant_id=tenant-b",
            headers={"X-API-Key": "kb"},
            json={"name": "alpha"},
        )
        assert p2.status_code == 200, p2.text
        pid_b = p2.json()["project_id"]
        assert p1.json()["name"] == p2.json()["name"] == "alpha"

        for key, pid in (("ka", pid_a), ("kb", pid_b)):
            r = await client.post(
                "/api/v1/ingest",
                headers={"X-API-Key": key, "X-Project-Id": pid},
                json={
                    "spans": [
                        {
                            "tenant_id": "ignored",
                            "trace_id": trace,
                            "span_id": span,
                            "name": "chat",
                            "timestamp": datetime.now(UTC).isoformat(),
                            "status": "OK",
                            "kind": "LLM",
                            "response": "a sufficiently long answer for quality judge",
                        }
                    ]
                },
            )
            assert r.status_code == 200, r.text

        await stack.pump(max_rounds=50)

        la = await client.get(
            f"/dashboard/api/v1/traces?tenant_id=tenant-a&project_id={pid_a}"
        )
        lb = await client.get(
            f"/dashboard/api/v1/traces?tenant_id=tenant-b&project_id={pid_b}"
        )
        assert la.status_code == 200
        assert lb.status_code == 200
        assert la.json()["count"] >= 1
        assert lb.json()["count"] >= 1
        assert all(i.get("project_id") == pid_a for i in la.json()["items"])
        assert all(i.get("project_id") == pid_b for i in lb.json()["items"])

        leak = await client.get(
            f"/dashboard/api/v1/traces?tenant_id=tenant-a&project_id={pid_b}"
        )
        assert leak.json()["count"] == 0

        dup = await client.post(
            "/projects/api/v1/projects?tenant_id=tenant-a",
            headers={"X-API-Key": "ka"},
            json={"name": "alpha"},
        )
        assert dup.status_code == 409
