"""SchedulerHub: APScheduler job registration + redis-py Lock leader election + exception isolation."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from benchmark.workers.scheduler_hub import (
    SchedulerHub,
    SchedulerHubConfig,
    TickJobOptions,
    build_scheduler_leader_lock_key,
)
from redis.exceptions import LockError


def test_build_scheduler_leader_lock_key_isolates_lanes() -> None:
    assert build_scheduler_leader_lock_key("pre") == "pre:scheduler:leader"
    assert build_scheduler_leader_lock_key("prod") == "prod:scheduler:leader"
    assert build_scheduler_leader_lock_key("daily") == "daily:scheduler:leader"
    assert build_scheduler_leader_lock_key("pre_agent-eval") == "pre_agent-eval:scheduler:leader"
    # The same rule as normalize_queue_prefix: only strip whitespace, and do not fold case or colons
    assert build_scheduler_leader_lock_key("  prod  ") == "prod:scheduler:leader"


def test_build_scheduler_leader_lock_key_rejects_empty() -> None:
    with pytest.raises(ValueError, match="queue_prefix"):
        build_scheduler_leader_lock_key("")
    with pytest.raises(ValueError, match="queue_prefix"):
        build_scheduler_leader_lock_key("   ")


def test_bind_queue_prefix_before_start() -> None:
    hub = SchedulerHub()
    assert hub.leader_lock_key == "eval:scheduler:leader"
    hub.bind_queue_prefix("pre")
    assert hub.leader_lock_key == "pre:scheduler:leader"
    hub.bind_queue_prefix("prod")
    assert hub.leader_lock_key == "prod:scheduler:leader"


@pytest.mark.asyncio
async def test_bind_queue_prefix_same_key_noop_after_start() -> None:
    """After ODPS has called hub.start, lifespan binding the same lane again must be idempotent, and must not take down daily/webui."""
    hub = SchedulerHub(SchedulerHubConfig(prefix="pre"))
    await hub.start()
    hub.bind_queue_prefix("pre")
    assert hub.leader_lock_key == "pre:scheduler:leader"
    await hub.shutdown()


@pytest.mark.asyncio
async def test_bind_queue_prefix_rejects_rebind_after_start() -> None:
    hub = SchedulerHub(SchedulerHubConfig(prefix="pre"))
    await hub.start()
    with pytest.raises(RuntimeError, match="after start"):
        hub.bind_queue_prefix("prod")
    await hub.shutdown()


@pytest.mark.asyncio
async def test_lane_leader_locks_do_not_contend() -> None:
    """pre / prod each hold their own lock and do not affect each other — fixes both environments grabbing the same global lock."""
    redis = _FakeRedis()
    pre = SchedulerHub(SchedulerHubConfig(redis=redis, prefix="pre", node_name="pre"))
    prod = SchedulerHub(SchedulerHubConfig(redis=redis, prefix="prod", node_name="prod"))
    await pre.start()
    await prod.start()
    assert await pre.ensure_leader() is True
    assert await prod.ensure_leader() is True
    assert redis.store["pre:scheduler:leader"]
    assert redis.store["prod:scheduler:leader"]
    await pre.shutdown()
    await prod.shutdown()


class _FakeLock:
    """Simulate ``redis.asyncio.lock.Lock``: token-safe acquire / extend / release."""

    def __init__(self, redis: "_FakeRedis", name: str, timeout: float) -> None:
        self._redis = redis
        self.name = name
        self.timeout = timeout
        self.local: str | None = None

    async def acquire(self, blocking: bool = False, **_: object) -> bool:
        # Simulate a sticky local.token: the token is still there but the key no longer belongs to this node,
        # and some redis-py paths raise LockError directly (rather than SET NX returning False).
        if self.local is not None:
            raise LockError("Cannot acquire a lock that's already owned")
        self._redis.set_calls += 1
        if self.name in self._redis.store:
            return False
        if not self._redis.allow_acquire:
            return False
        token = f"tok-{self._redis.set_calls}"
        self.local = token
        self._redis.store[self.name] = token
        return True

    async def owned(self) -> bool:
        return self.local is not None and self._redis.store.get(self.name) == self.local

    async def extend(self, additional_time: float, replace_ttl: bool = False) -> bool:
        del additional_time, replace_ttl
        if not await self.owned():
            raise LockError("Cannot extend an unlocked lock")
        if not self._redis.allow_renew:
            # Simulate the lease being taken / the TTL having changed owners
            if self._redis.store.get(self.name) == self.local:
                del self._redis.store[self.name]
            return False
        return True

    async def release(self) -> None:
        if not await self.owned():
            raise LockError("Cannot release a lock that's no longer owned")
        del self._redis.store[self.name]
        self.local = None


class _FakeRedis:
    """Minimal Redis stand-in: it only provides ``lock()`` (consistent with the production path)."""

    def __init__(self, *, allow_acquire: bool = True, allow_renew: bool = True) -> None:
        self.store: dict[str, str] = {}
        self.allow_acquire = allow_acquire
        self.allow_renew = allow_renew
        self.set_calls = 0

    def lock(
        self,
        name: str,
        timeout: float | None = None,
        **_: object,
    ) -> _FakeLock:
        # Consistent with redis-py: every lock() is a new instance (with its own token); the hub reuses it through self._lock
        return _FakeLock(self, name, float(timeout or 90))


@pytest.mark.asyncio
async def test_without_redis_every_node_is_leader() -> None:
    hub = SchedulerHub()
    await hub.start()
    assert await hub.ensure_leader() is True
    await hub.shutdown()


@pytest.mark.asyncio
async def test_leader_acquire_then_renew() -> None:
    redis = _FakeRedis()
    hub = SchedulerHub(SchedulerHubConfig(redis=redis, lock_key="k"))
    await hub.start()
    assert await hub.ensure_leader() is True
    # Already holding the lock takes extend, and does not SET NX again
    assert await hub.ensure_leader() is True
    assert redis.set_calls == 1
    await hub.shutdown()
    # graceful shutdown → a token-safe release
    assert "k" not in redis.store


@pytest.mark.asyncio
async def test_leadership_lost_falls_back_to_acquire() -> None:
    redis = _FakeRedis(allow_acquire=False, allow_renew=False)
    hub = SchedulerHub(SchedulerHubConfig(redis=redis, lock_key="k"))
    await hub.start()
    redis.allow_acquire = True
    assert await hub.ensure_leader() is True
    redis.allow_acquire = False
    redis.allow_renew = False
    # Renewing the lease fails → grabbing it again is also rejected → not the leader
    assert await hub.ensure_leader() is False
    await hub.shutdown()


@pytest.mark.asyncio
async def test_reacquire_after_leadership_loss_resets_lock() -> None:
    """After losing the lock the sticky Lock must be dropped; otherwise acquire raises LockError, which is swallowed, and it stays non-leader forever."""
    redis = _FakeRedis()
    hub = SchedulerHub(SchedulerHubConfig(redis=redis, lock_key="k"))
    await hub.start()
    assert await hub.ensure_leader() is True
    sticky = hub._lock
    assert sticky is not None

    # Simulate the lease being taken / the TTL changing owners: the Redis key is gone, but the in-process Lock.local is still left behind
    redis.store.clear()
    redis.allow_renew = False
    assert await hub.ensure_leader() is True
    assert hub._lock is not sticky
    assert redis.store.get("k") is not None
    await hub.shutdown()


@pytest.mark.asyncio
async def test_ttl_expiry_allows_new_leader() -> None:
    """The old leader crashed without release: after the TTL expires another node can grab it — there is not permanently no leader."""
    redis = _FakeRedis()
    old = SchedulerHub(SchedulerHubConfig(redis=redis, lock_key="k", node_name="old"))
    await old.start()
    assert await old.ensure_leader() is True
    # Simulate kill -9: do not go through shutdown, and only drop the in-process state; the Redis key is still there
    old._is_leader = False
    old._lock = None

    new = SchedulerHub(SchedulerHubConfig(redis=redis, lock_key="k", node_name="new"))
    await new.start()
    assert await new.ensure_leader() is False  # The TTL has not expired, so it cannot be grabbed

    # The TTL has expired
    redis.store.clear()
    assert await new.ensure_leader() is True
    await new.shutdown()


@pytest.mark.asyncio
async def test_shutdown_release_does_not_delete_other_leader() -> None:
    """A late shutdown from the old process must not delete the new leader's lease by mistake (contrast with a bare DELETE)."""
    redis = _FakeRedis()
    old = SchedulerHub(SchedulerHubConfig(redis=redis, lock_key="k", node_name="old"))
    await old.start()
    assert await old.ensure_leader() is True
    # The TTL expired + a new node grabbed the lock
    redis.store.clear()
    new = SchedulerHub(SchedulerHubConfig(redis=redis, lock_key="k", node_name="new"))
    await new.start()
    assert await new.ensure_leader() is True
    new_token = redis.store["k"]

    # The old process shuts down only now: owned()=False, and it must not delete the new leader
    await old.shutdown()
    assert redis.store.get("k") == new_token
    await new.shutdown()


@pytest.mark.asyncio
async def test_job_skipped_when_not_leader_and_errors_isolated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    redis = _FakeRedis(allow_acquire=False)
    hub = SchedulerHub(SchedulerHubConfig(redis=redis, lock_key="k"))
    await hub.start()
    ran = AsyncMock()
    hub.add_tick("tick", ran, TickJobOptions(seconds=3600))
    job = hub._scheduler.get_job("tick")
    assert job is not None
    with caplog.at_level("WARNING", logger="agent_eval.eval_workers.scheduler_hub"):
        await job.func()
    ran.assert_not_called()
    assert any("job_skipped_not_leader" in r.message for r in caplog.records)

    redis.allow_acquire = True
    boom = AsyncMock(side_effect=RuntimeError("boom"))
    await hub._wrap("boom", boom)()
    boom.assert_called_once()
    await hub.shutdown()


@pytest.mark.asyncio
async def test_add_tick_never_pauses_job() -> None:
    """run_now=False must also produce a concrete next_run_time (None means APScheduler is paused)."""
    hub = SchedulerHub()
    await hub.start()
    hub.add_tick("tick", AsyncMock(), TickJobOptions(seconds=60, run_now=False))
    job = hub._scheduler.get_job("tick")
    assert job is not None
    assert job.next_run_time is not None
    assert job.next_run_time.tzinfo is not None
    delta = (job.next_run_time - datetime.now(UTC)).total_seconds()
    assert 50 <= delta <= 70
    await hub.shutdown()


@pytest.mark.asyncio
async def test_add_cron_sets_misfire_grace_for_on_time_fire() -> None:
    """cron must carry misfire_grace: a short restart around the fixed time still goes through cron, and does not rely on catch_up."""
    from benchmark.workers.scheduler_hub import CRON_MISFIRE_GRACE_S
    from apscheduler.triggers.cron import CronTrigger  # noqa: TID251

    hub = SchedulerHub()
    await hub.start()
    hub.add_cron("cron-1", AsyncMock(), CronTrigger(hour=2, minute=0, timezone="Asia/Shanghai"))
    job = hub._scheduler.get_job("cron-1")
    assert job is not None
    assert job.misfire_grace_time == CRON_MISFIRE_GRACE_S
    assert job.next_run_time is not None
    await hub.shutdown()


@pytest.mark.asyncio
async def test_job_registration_and_removal() -> None:
    hub = SchedulerHub()
    await hub.start()
    hub.add_tick("tick", AsyncMock(), TickJobOptions(seconds=3600))
    assert hub.has_job("tick")
    hub.remove("tick")
    assert not hub.has_job("tick")
    hub.remove("missing")
    await hub.shutdown()
    assert not hub.started
