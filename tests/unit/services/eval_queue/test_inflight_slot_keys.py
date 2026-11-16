"""inflight slot-key construction — judge/attr are isolated, aligned with design D7/D13."""

from unittest.mock import AsyncMock, patch

import pytest
from benchmark.queue.inflight_slot import (
    _CLAMP_LUA,
    SlotKeySpec,
    build_probe_key,
    build_slot_key,
    clamp_slot_count,
)


def test_async_slot_keys_omit_eval_type() -> None:
    judge = build_slot_key(
        "pre:bullmq",
        SlotKeySpec(
            kind="judge",
            node="n1",
            run_id="run1",
            eval_type="external",
        ),
    )
    attr = build_slot_key(
        "pre:bullmq",
        SlotKeySpec(
            kind="attr",
            node="n1",
            run_id="run1",
            eval_type="attribution",
        ),
    )
    assert judge == "pre:bullmq:inflight:judge:n1:run1"
    assert attr == "pre:bullmq:inflight:attr:n1:run1"
    assert judge != attr


def test_sync_slot_key_keeps_eval_type() -> None:
    bench = build_slot_key(
        "pre:bullmq",
        SlotKeySpec(
            kind="sync",
            node="n1",
            run_id="run1",
            eval_type="benchmark",
        ),
    )
    dataset = build_slot_key(
        "pre:bullmq",
        SlotKeySpec(
            kind="sync",
            node="n1",
            run_id="run1",
            eval_type="dataset",
        ),
    )
    assert bench == "pre:bullmq:inflight:sync:benchmark:n1:run1"
    assert dataset == "pre:bullmq:inflight:sync:dataset:n1:run1"
    assert bench != dataset


def test_multi_node_same_run_uses_distinct_keys() -> None:
    a = build_slot_key(
        "pre:bullmq",
        SlotKeySpec(
            kind="attr",
            node="node-a",
            run_id="run1",
        ),
    )
    b = build_slot_key(
        "pre:bullmq",
        SlotKeySpec(
            kind="attr",
            node="node-b",
            run_id="run1",
        ),
    )
    assert a != b
    assert a.endswith(":node-a:run1")
    assert b.endswith(":node-b:run1")


def test_probe_keys_isolated_by_kind() -> None:
    judge = build_probe_key("pre:bullmq", kind="judge", node="n1", run_id="run1")
    attr = build_probe_key("pre:bullmq", kind="attr", node="n1", run_id="run1")
    assert judge == "pre:bullmq:inflight:probe:judge:n1:run1"
    assert attr == "pre:bullmq:inflight:probe:attr:n1:run1"
    assert judge != attr


def test_clamp_lua_restores_and_renews_ttl() -> None:
    """When the TTL runs out during hold, it must be SET back, and when it is equal EXPIRE renews it."""
    assert "if current < expected then" in _CLAMP_LUA
    assert "redis.call('SET', KEYS[1], expected, 'EX', ttl)" in _CLAMP_LUA
    assert "redis.call('EXPIRE', KEYS[1], ttl)" in _CLAMP_LUA


@pytest.mark.asyncio
async def test_clamp_slot_count_forwards_expected_and_ttl() -> None:
    script = AsyncMock(return_value=0)
    with (
        patch("benchmark.queue.inflight_slot.get_redis"),
        patch(
            "benchmark.queue.inflight_slot._runtime",
        ) as runtime,
    ):
        runtime.clamp_script = script
        n = await clamp_slot_count("k1", expected=2, ttl_s=600)
    assert n == 0
    script.assert_awaited_once_with(keys=["k1"], args=[2, 600])
