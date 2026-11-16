import asyncio
import hashlib
import hmac
import json

import pytest
from httpx import ASGITransport, AsyncClient

from benchmark.kernel.api import create_app
from tests.conftest import CONF, case, wait_terminal


@pytest.mark.integration
async def test_lease_never_exceeds_limit(kernel):
    acquired = []

    async def take():
        lease = await kernel.leases.acquire_pair(
            kind="judge",
            node_id="n1",
            run_id="run-limit",
            node_limit=5,
            cluster_limit=5,
        )
        return lease

    results = await asyncio.gather(*[take() for _ in range(12)])
    acquired = [item for item in results if item is not None]
    assert len(acquired) == 5
    assert await kernel.leases.redis.zcard(kernel.leases.cluster_key("judge")) == 5
    for lease in acquired:
        await kernel.leases.release(lease)
    assert await kernel.leases.redis.zcard(kernel.leases.cluster_key("judge")) == 0


@pytest.mark.integration
async def test_sync_run_completes(live):
    created = await live.publish_run({
        "benchmark_name": "demo",
        "profile_id": "benchmark_sync",
        "cases": [case("1"), case("2", output="no", expected="yes")],
    })
    finished = await wait_terminal(live, created["run_id"])
    assert finished["status"] == "completed"
    assert finished["total_samples"] == 2
    assert finished["evaluated_items"] == 2
    assert finished["final_score"] == 0.5
    items = await live.store.list_items(created["run_id"])
    assert {row["score"] for row in items} == {1.0, 0.0}


@pytest.mark.integration
async def test_duplicate_callback_counts_once(live):
    created = await live.publish_run({
        "benchmark_name": "async-demo",
        "profile_id": "benchmark_async",
        "cases": [case("1")],
    })
    external_id = await _wait_external(live, created["run_id"])
    body = json.dumps({
        "rows": [{"criterion_id": "fused", "score": 1, "reason": "ok", "status": "success"}]
    }).encode()
    signature = hmac.new(b"test-callback", body, hashlib.sha256).hexdigest()
    headers = {"x-signature": signature}

    async def post():
        return await live.callback(external_id, body, headers)

    results = await asyncio.gather(post(), post())
    assert any(item["applied"] for item in results)
    finished = await wait_terminal(live, created["run_id"])
    assert finished["evaluated_items"] == 1
    assert finished["judge_received_count"] == 1
    assert finished["status"] == "completed"


@pytest.mark.integration
async def test_stop_keeps_total_samples(kernel):
    created = await kernel.publish_run({
        "benchmark_name": "stop-demo",
        "profile_id": "benchmark_sync",
        "cases": [case(str(i)) for i in range(4)],
    })
    stopped = await kernel.stop(created["run_id"])
    run = await kernel.store.get_run(created["run_id"])
    assert run["total_samples"] == 4
    assert run["evaluated_items"] == 0
    assert run["status"] == "stopped"
    assert stopped["total_samples"] == 4
    waiting = await kernel.dispatch.waiting_and_delayed("benchmark-cases")
    assert [job for job in waiting if (job.data or {}).get("run_id") == created["run_id"]] == []


@pytest.mark.integration
async def test_replay_does_not_overwrite_parent(live):
    parent = await live.publish_run({
        "benchmark_name": "replay-demo",
        "profile_id": "benchmark_sync",
        "cases": [case("c1", output="no", expected="yes")],
    })
    parent_done = await wait_terminal(live, parent["run_id"])
    assert parent_done["final_score"] == 0
    child = await live.playground_replay(
        parent["run_id"],
        {"case_ids": ["c1"], "agent_id": "agent-1", "triggered_by": "user-1"},
        "Bearer secret-token",
    )
    child_done = await wait_terminal(live, child["run_id"])
    parent_after = await live.store.get_run(parent["run_id"])
    assert parent_after["final_score"] == 0
    assert parent_after["status"] == "completed"
    assert child_done["status"] == "completed"
    assert child_done["final_score"] == 1
    assert child["dispatch_domain_id"] == "local-eval"
    items = await live.store.list_items(child["run_id"])
    assert len(items) == 1
    stored = await live.leases.redis.get(live._cred_key(child["run_id"]))
    assert stored in {None, ""}


@pytest.mark.integration
async def test_attribution_failure_keeps_judge_status(live):
    created = await live.publish_run({
        "benchmark_name": "attr-demo",
        "profile_id": "benchmark_attributed",
        "cases": [{
            "item_id": "1",
            "input": {"text": "q"},
            "output": {"text": "wrong"},
            "expected": {"text": "right", "attribution": "fail"},
        }],
    })
    finished = await wait_terminal(live, created["run_id"])
    assert finished["status"] == "completed"
    item = (await live.store.list_items(created["run_id"]))[0]
    assert item["status"] == "success"
    assert item["score"] == 0
    assert item["attribution_detail_json"]["status"] == "failed"


@pytest.mark.integration
async def test_dataset_domain_and_unknown_domain(live):
    with pytest.raises(Exception) as caught:
        await live.publish_dataset({"eval_domain_id": "local-eval", "cases": [case()]})
    assert caught.value.status == 400
    created = await live.publish_dataset({
        "eval_domain_id": "answer_quality",
        "cases": [case("1", output="hello world", expected="hello")],
    })
    finished = await wait_terminal(live, created["run_id"])
    assert finished["status"] == "completed"
    assert finished["total_evaluators"] == 2
    assert finished["evaluated_items"] == 2


@pytest.mark.integration
async def test_l2_queue_is_isolated(kernel):
    kernel.conf.app["worker_profile"] = "default"
    kernel.start_workers()
    created = await kernel.publish_run({
        "benchmark_name": "heavy",
        "profile_id": "benchmark_sync",
        "business": "heavy_business",
        "cases": [case("1")],
    })
    await asyncio.sleep(0.4)
    assert (await kernel.store.get_run(created["run_id"]))["status"] == "running"
    waiting = await kernel.dispatch.waiting_and_delayed("benchmark-cases-l2")
    assert any((job.data or {}).get("run_id") == created["run_id"] for job in waiting)
    other = type(kernel)(kernel.conf, node_id="l2-node")
    other.conf.app["worker_profile"] = "l2"
    other.start_workers()
    try:
        finished = await wait_terminal(other, created["run_id"])
    finally:
        await other.close()
    assert finished["status"] == "completed"


@pytest.mark.integration
async def test_skill_and_tool_profiles(live):
    skill = await live.publish_run({
        "benchmark_name": "skill",
        "profile_id": "skill_eval",
        "cases": [{
            "item_id": "1",
            "input": {"task": "search"},
            "output": {"skills_used": ["search"]},
            "expected": {"skill_id": "search"},
        }],
    })
    tool = await live.publish_run({
        "benchmark_name": "tool",
        "profile_id": "tool_contract_eval",
        "cases": [{
            "item_id": "1",
            "input": {},
            "output": {"tool_calls": [{"name": "lookup", "arguments": {"q": "a"}}]},
            "expected": {"required_fields": ["q"]},
        }],
    })
    assert (await wait_terminal(live, skill["run_id"]))["final_score"] == 1
    assert (await wait_terminal(live, tool["run_id"]))["final_score"] == 1


@pytest.mark.integration
async def test_message_does_not_create_run(kernel):
    result = await kernel.ingest_message({
        "trace_id": "t-1",
        "project_id": "p",
        "agent_name": "demo",
        "evaluator_name": "exact",
        "output": {"text": "hello"},
        "expected": {"text": "hello"},
    })
    assert result["run_created"] is False
    assert result["status"] == "recorded"
    assert await kernel.store.list_runs() == []


@pytest.mark.integration
async def test_healer_reaps_orphan_lease_only(kernel):
    kernel.conf.app["healer"]["orphan_grace_ms"] = 0
    lease = await kernel.leases.acquire("orphan-key", 2)
    assert lease != "busy"
    await asyncio.sleep(0.05)
    await kernel.healer_tick()
    assert await kernel.leases.redis.zscore("orphan-key", lease) is None
    kept = await kernel.leases.acquire_pair(
        kind="sync", node_id="n", run_id="kept", node_limit=2, cluster_limit=0
    )
    async with await kernel.store.connect() as conn:
        await conn.execute(
            """
            INSERT INTO lease_ledger (lease_id, capacity_key, run_id, item_id, abandoned)
            VALUES (%s, %s, 'kept', '1', false)
            """,
            (kept.node_id, kept.node_key),
        )
        await conn.commit()
    await asyncio.sleep(0.05)
    await kernel.healer_tick()
    assert await kernel.leases.redis.zscore(kept.node_key, kept.node_id) is not None


@pytest.mark.integration
async def test_http_replay_and_stop(live):
    app = create_app(CONF)
    app.state.kernel = live
    app.state.owns_kernel = False
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://benchmark") as client:
        created = await client.post("/v1/runs", json={
            "benchmark_name": "http-demo",
            "profile_id": "benchmark_sync",
            "cases": [case("1")],
        })
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        await wait_terminal(live, run_id)
        denied = await client.post(f"/v1/benchmarks/runs/{run_id}/playground-replay", json={
            "case_ids": ["1"],
            "agent_id": "agent-1",
            "triggered_by": "user",
            "callback_url": "http://evil.example",
        })
        assert denied.status_code == 400
        replayed = await client.post(
            f"/v1/benchmarks/runs/{run_id}/playground-replay",
            headers={"authorization": "Bearer token"},
            json={"case_ids": ["1"], "agent_id": "agent-1", "triggered_by": "user"},
        )
        assert replayed.status_code == 201
        page = await client.get("/")
        assert page.status_code == 200
        assert "Eval Benchmark" in page.text


async def _wait_external(kernel, run_id: str) -> str:
    deadline = asyncio.get_event_loop().time() + 15
    while asyncio.get_event_loop().time() < deadline:
        jobs = await kernel.dispatch.waiting_and_delayed("benchmark-running")
        for job in jobs:
            if (job.data or {}).get("run_id") == run_id and job.name == "pending_callback":
                return job.id
        await asyncio.sleep(0.05)
    raise AssertionError("external job was not registered")
