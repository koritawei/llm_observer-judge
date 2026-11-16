"""APScheduler hub: scheduled tasks inside agent-eval all go through ``AsyncIOScheduler``.

Design points:
1. **Single scheduler**: the process holds one ``AsyncIOScheduler``; each business scheduler
   only registers jobs (60s tick / cron / one-shot compensation) and does not run its own
   long-lived ``asyncio.sleep`` loop.
2. **Multi-instance leader election**: before each job, take a leader lease with the **official
   redis-py** ``redis.asyncio.lock.Lock`` (``SET NX`` + token-safe renew / release); only the
   leader fires, so replicas do not create duplicate runs. If Redis is unavailable, fall back
   to "this host is the leader". The lock key follows the BullMQ lane as
   ``{queue_prefix}:scheduler:leader`` (``daily`` / ``pre`` / ``prod`` / isolated prefixes stay
   independent); do not hard-code a cross-env shared ``agent-eval:scheduler:leader``.
3. **Failure isolation**: one job raising only logs; it does not take down the scheduler or other jobs.

Leader failover (inherent to the lease model, not an implementation defect):
- Holder crash without graceful release → other nodes wait at most ``LEADER_TTL_S`` (90s)
  then take the lock; there is **no** "TTL expired and no live leader forever" state.
- A rolling restart that reaches ``shutdown`` → token-safe ``release``, and the gap is near 0.
"""

import os
import socket
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

# APScheduler cron: in-repo ruff TID251 defaults to recommending Celery, but this service's
# scheduled tasks are dynamic page CRUD (runtime add/update/delete of cron). Celery Beat's
# in-process beat_schedule needs a reload to take effect (design doc §2.1 rejected it), so TID251 is waived.
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # noqa: TID251
from apscheduler.triggers.cron import CronTrigger  # noqa: TID251
from apscheduler.triggers.date import DateTrigger  # noqa: TID251
from apscheduler.triggers.interval import IntervalTrigger  # noqa: TID251
from redis.exceptions import LockError

from benchmark.queue.queue_prefix_policy import normalize_queue_prefix

if TYPE_CHECKING:
    from redis.asyncio import Redis


class _LeaderLock(Protocol):
    """Leader-lease surface shared by redis-py ``Lock`` and the test FakeLock."""

    async def acquire(self, blocking: bool = False) -> bool: ...

    async def owned(self) -> bool: ...

    async def extend(self, additional_time: float, replace_ttl: bool = False) -> bool: ...

    async def release(self) -> None: ...


logger = get_logger("agent_eval.eval_workers.scheduler_hub")

#: Shared tick interval for cron schedulers: each scheduler keeps its existing due check (2-minute
#: window). A 60s poll will not miss a slot, and config changes take effect within one minute.
TICK_SECONDS = 60

#: Leader lock TTL: slightly longer than the tick; after the holder dies, takeover is within about 90s.
LEADER_TTL_S = 90

#: Cron misfire grace (seconds). APScheduler defaults to 1s: if leader election or process
#: startup lands on the slot, an on-time cron is dropped as a misfire and only catch_up can
#: fill it — that misses "fire accurately the next day". 5 minutes covers election and short restarts.
CRON_MISFIRE_GRACE_S = 300

#: Warning throttle (seconds) when a non-leader skips work. Do not return silently — otherwise a
#: paused tick / lost lock leaves no trace_report logs for days and is very hard to debug.
_NON_LEADER_WARN_INTERVAL_S = 300.0

JobFunc = Callable[[], Awaitable[None]]

#: Provisional lock key before the BullMQ prefix is bound (must be overwritten before ``start``
#: by :meth:`SchedulerHub.bind_queue_prefix` with the lane lock).
_PROVISIONAL_LEADER_LOCK_KEY = "eval:scheduler:leader"


def build_scheduler_leader_lock_key(queue_prefix: str) -> str:
    """Build the leader lock from the BullMQ ``queue_prefix``, isolating prod / pre / isolated lanes.

    Prefix normalization reuses :func:`normalize_queue_prefix` (same rule as ``_build_queue_prefix``),
    then appends ``:scheduler:leader``, so it does not drift from the lane-prefix rules.

    Examples:
        ``pre`` → ``pre:scheduler:leader``;
        ``prod`` → ``prod:scheduler:leader``;
        ``pre_agent-eval`` / ``dev-<id>`` → ``{prefix}:scheduler:leader``.
    """
    normalized = normalize_queue_prefix(queue_prefix)
    if not normalized:
        raise ValueError("queue_prefix is required for scheduler leader lock key")
    return f"{normalized}:scheduler:leader"


@dataclass(frozen=True)
class SchedulerHubConfig:
    """Constructor arguments for ``SchedulerHub`` (FUNC-PARAMS-001).

    ``prefix``: bare BullMQ lane prefix (aligned with ``settings.eval_queue_prefix``).
    ``lock_key``: explicit lock key (tests / ops override); when omitted, derived from ``prefix``.
    If both are empty, use the provisional ``eval:scheduler:leader`` and call
    ``bind_queue_prefix`` before ``start``.
    """

    redis: "Redis | None" = None
    redis_getter: "Callable[[], Redis | None] | None" = None
    prefix: str = ""
    node_name: str = ""
    lock_key: str | None = None
    ttl_s: int = LEADER_TTL_S


@dataclass(frozen=True)
class TickJobOptions:
    """Optional arguments for ``add_tick`` (FUNC-PARAMS-001)."""

    seconds: int = TICK_SECONDS
    run_now: bool = False


class SchedulerHub:
    """APScheduler wrapper: job registration + Redis leader election + exception isolation."""

    def __init__(self, config: SchedulerHubConfig | None = None) -> None:
        cfg = config or SchedulerHubConfig()
        self._scheduler = AsyncIOScheduler(timezone=UTC)
        self._redis = cfg.redis
        #: Lazy Redis: schedulers init before the BullMQ stack; Redis exists only on the first job run
        self._redis_getter = cfg.redis_getter
        self._node = cfg.node_name or f"{socket.gethostname()}:{os.getpid()}"
        if cfg.lock_key:
            self._lock_key = cfg.lock_key
        elif str(cfg.prefix or "").strip():
            self._lock_key = build_scheduler_leader_lock_key(cfg.prefix)
        else:
            self._lock_key = _PROVISIONAL_LEADER_LOCK_KEY
        self._ttl_s = cfg.ttl_s
        self._is_leader = False
        #: redis-py Lock: reuse the same instance while held (token is fixed);
        #: **must set None after losing the lock**, or a leftover local.token makes some redis-py
        #: versions raise ``LockError("Cannot acquire a lock that's already owned")`` from
        #: ``acquire()``, swallowed so this node never retakes leader (only a restart recovers).
        self._lock: _LeaderLock | None = None
        self._started = False
        self._warned_no_redis = False
        #: Last non-leader warning timestamp (monotonic), throttled per job_id
        self._non_leader_warned_at: dict[str, float] = {}

    @property
    def leader_lock_key(self) -> str:
        """Current leader lock key (``{queue_prefix}:scheduler:leader`` or provisional)."""
        return self._lock_key

    def bind_queue_prefix(self, queue_prefix: str) -> None:
        """Bind the lane leader lock after the BullMQ prefix is resolved.

        Idempotent for the same key (including after hub ``start`` / while holding the lease):
        lifespan calls this again from BullMQ resolve, ODPS start, and daily/trace/webui start.
        A second bind must not take down the whole scheduler set. A key change still must happen
        before ``start`` / lock acquisition.
        """
        new_key = build_scheduler_leader_lock_key(queue_prefix)
        if new_key == self._lock_key:
            logger.debug(
                "scheduler.leader_lock_bind_noop lock_key=%s started=%s node=%s",
                self._lock_key,
                self._started,
                self._node,
            )
            return
        if self._started:
            raise RuntimeError(
                "cannot rebind scheduler leader lock after start "
                f"(current={self._lock_key} requested={new_key})"
            )
        if self._is_leader or self._lock is not None:
            raise RuntimeError(
                "cannot rebind scheduler leader lock while holding lease "
                f"(current={self._lock_key} requested={new_key})"
            )
        logger.info(
            "scheduler.leader_lock_bound old=%s new=%s node=%s",
            self._lock_key,
            new_key,
            self._node,
        )
        self._lock_key = new_key

    # ── lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Start APScheduler (idempotent). Redis for the leader lock resolves on the first job run."""
        if self._started:
            return
        self._scheduler.start()
        self._started = True
        logger.info(
            "scheduler.started jobs=%d leader_lock=%s node=%s ttl_s=%d backend=redis-py.Lock",
            len(self._scheduler.get_jobs()),
            self._lock_key,
            self._node,
            self._ttl_s,
        )

    async def shutdown(self) -> None:
        """Stop the scheduler; if this node still holds the leader lease, release it token-safely."""
        if not self._started:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            logger.warning("scheduler.shutdown_failed", exc_info=True)
        self._started = False
        await self._release_leader()
        logger.info("scheduler.stopped")

    @property
    def started(self) -> bool:
        return self._started

    # ── job registration ────────────────────────────────────────────

    def add_tick(
        self,
        job_id: str,
        func: JobFunc,
        options: TickJobOptions | None = None,
    ) -> None:
        """Register an interval job (default 60s tick; the caller keeps its own due check).

        **Do not** pass ``next_run_time=None``: under APScheduler ``None`` means the job is
        **paused** and will never fire. That once left jobs such as ``trace-daily-report-tick``
        with ``run_now=False`` reporting only via startup compensation, with the slot tick silent
        all day. ``run_now=True`` → first fire immediately; ``False`` → first fire after one interval.
        """
        opts = options or TickJobOptions()
        delay_s = 0 if opts.run_now else max(int(opts.seconds), 1)
        first_run = datetime.now(UTC) + timedelta(seconds=delay_s)
        self._scheduler.add_job(
            self._wrap(job_id, func),
            trigger=IntervalTrigger(seconds=opts.seconds),
            id=job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            next_run_time=first_run,
        )
        logger.info(
            "scheduler.job_added job_id=%s interval_s=%d run_now=%s next_run=%s",
            job_id,
            opts.seconds,
            opts.run_now,
            first_run.isoformat(),
        )

    def add_cron(self, job_id: str, func: JobFunc, trigger: CronTrigger) -> None:
        """Register a cron job (webUI tasks fire by expression; next-day slots use this, not catch_up).

        ``misfire_grace_time``: a short restart or lock race around the slot still fires on time;
        same-day misses past the grace window are filled by ``WebuiTaskScheduler.catch_up``.
        """
        self._scheduler.add_job(
            self._wrap(job_id, func),
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=CRON_MISFIRE_GRACE_S,
        )

    def add_once(self, job_id: str, func: JobFunc) -> None:
        """Register a one-shot compensation job (startup compensation: queue it to run immediately)."""
        self._scheduler.add_job(
            self._wrap(job_id, func),
            trigger=DateTrigger(run_date=datetime.now(UTC)),
            id=job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

    def remove(self, job_id: str) -> None:
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            # An unregistered or already-finished one-shot job is not an error
            return

    def has_job(self, job_id: str) -> bool:
        return self._scheduler.get_job(job_id) is not None

    def next_run_time(self, job_id: str) -> datetime | None:
        job = self._scheduler.get_job(job_id)
        return getattr(job, "next_run_time", None) if job is not None else None

    # ── leader election (redis-py Lock) ─────────────────────────────

    async def ensure_leader(self) -> bool:
        """Whether this instance is the leader (renew if held; non-blocking acquire if not).

        Lease gap: if the old leader is kill -9 without release, other nodes wait at most one TTL
        then acquire; after TTL expiry the next job can always acquire, so a leader is never gone
        forever. After this node loses the lock it drops the ``Lock`` instance and retries next tick
        with a new token, so a sticky ``local.token`` cannot block reacquire forever.
        """
        self._resolve_redis()
        if self._redis is None:
            return True
        if self._is_leader:
            lock = self._leader_lock()
            if await self._renew_leader(lock):
                return True
            logger.warning(
                "scheduler.leadership_lost lock_key=%s node=%s",
                self._lock_key,
                self._node,
            )
            self._is_leader = False
            self._discard_leader_lock()
        return await self._acquire_leader()

    def _resolve_redis(self) -> None:
        """Resolve shared Redis on the first job run (schedulers start before the BullMQ stack)."""
        if self._redis is None and self._redis_getter is not None:
            try:
                self._redis = self._redis_getter()
            except Exception:
                self._redis = None
            if self._redis is None and not self._warned_no_redis:
                self._warned_no_redis = True
                logger.warning(
                    "scheduler.leader_election_disabled (redis unavailable; every node ticks) "
                    "lock_key=%s",
                    self._lock_key,
                )

    def _leader_lock(self) -> _LeaderLock:
        """Reuse the same ``Lock`` while held; after losing it, call ``_discard_leader_lock`` first."""
        if self._lock is None:
            assert self._redis is not None
            self._lock = self._redis.lock(
                self._lock_key,
                timeout=float(self._ttl_s),
                blocking=False,
                thread_local=False,
            )
        return self._lock

    def _discard_leader_lock(self) -> None:
        """Drop the sticky Lock so the next ``_leader_lock`` takes a new token."""
        self._lock = None

    async def _try_acquire(self, lock: _LeaderLock) -> bool:
        """Non-blocking acquire; on ``LockError`` (sticky token) drop the instance and retry once."""
        try:
            return bool(await lock.acquire(blocking=False))
        except LockError:
            self._discard_leader_lock()
            lock = self._leader_lock()
            return bool(await lock.acquire(blocking=False))

    async def _acquire_leader(self) -> bool:
        lock = self._leader_lock()
        try:
            acquired = await self._try_acquire(lock)
        except Exception:
            logger.warning("scheduler.leader_acquire_failed", exc_info=True)
            self._discard_leader_lock()
            return False
        self._is_leader = bool(acquired)
        if self._is_leader:
            logger.warning(
                "scheduler.leader_acquired lock_key=%s node=%s ttl_s=%d backend=redis-py.Lock",
                self._lock_key,
                self._node,
                self._ttl_s,
            )
        return self._is_leader

    async def _renew_leader(self, lock: _LeaderLock) -> bool:
        try:
            if not await lock.owned():
                return False
            return bool(await lock.extend(self._ttl_s, replace_ttl=True))
        except LockError:
            return False
        except Exception:
            logger.warning("scheduler.leader_renew_failed", exc_info=True)
            return False

    async def _release_leader(self) -> None:
        """Release the lease on graceful shutdown: release only while still owned, so the new leader is kept."""
        lock = self._lock
        self._is_leader = False
        if lock is None:
            return
        try:
            if await lock.owned():
                await lock.release()
                logger.warning(
                    "scheduler.leader_released lock_key=%s node=%s",
                    self._lock_key,
                    self._node,
                )
        except LockError:
            # Lease already expired or held by someone else — ignore
            pass
        except Exception:
            logger.warning("scheduler.leader_release_failed", exc_info=True)
        finally:
            self._discard_leader_lock()

    # ── internals ───────────────────────────────────────────────────

    def _warn_non_leader(self, job_id: str) -> None:
        """Non-leader skip: throttled WARNING, so it does not spam every 60s and is not fully silent."""
        now = time.monotonic()
        last = self._non_leader_warned_at.get(job_id, 0.0)
        if now - last < _NON_LEADER_WARN_INTERVAL_S:
            return
        self._non_leader_warned_at[job_id] = now
        logger.warning(
            "scheduler.job_skipped_not_leader job_id=%s lock_key=%s node=%s "
            "(will retry within ttl_s=%d; warn at most every %ss)",
            job_id,
            self._lock_key,
            self._node,
            self._ttl_s,
            int(_NON_LEADER_WARN_INTERVAL_S),
        )

    def _wrap(self, job_id: str, func: JobFunc) -> JobFunc:
        async def _runner() -> None:
            if not await self.ensure_leader():
                self._warn_non_leader(job_id)
                return
            try:
                await func()
            except Exception:
                logger.exception("scheduler.job_failed job_id=%s", job_id)

        return _runner


__all__ = [
    "CRON_MISFIRE_GRACE_S",
    "LEADER_TTL_S",
    "TICK_SECONDS",
    "SchedulerHub",
    "SchedulerHubConfig",
    "TickJobOptions",
    "build_scheduler_leader_lock_key",
]
