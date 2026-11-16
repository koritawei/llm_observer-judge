"""running_healer — crash/restart recovery (design doc §9.2 D16).

Responsibilities:
1. **Heartbeat refresh**: refresh the PG heartbeat while wait or running still has jobs for the run (avoid a false stale)
2. **Sync orphan inflight**: Worker crashed after the running registration → clean up / requeue to wait
3. **Async callback timeout**: ``pending_callback`` past ``callback_timeout_ms``
   (judge and attribution both default to 10min; attribution follows the Aone config) → check job status:
   wait on ``pending`` / ``running``; for ``completed`` / ``failed``, if a GET body can
   restore the callback JSON → go through ``handle_callback`` (same path as the HTTP callback); otherwise
   ``completed`` must distinguish a lost callback from one in flight / already landed (see
   ``_completed_callback_verdict``); other non-in-flight states release the slot and requeue to wait (keep
   ``first_enqueued_at``). Past that case's own window (default 2h, from the first entry into running)
   or TTL → the case fails terminally (skip if PG is already terminal; do not steal callback cleanup).
4. **Startup D16 guard**: collect still-in-flight run_ids before ``recover_stale``, and skip false kills
5. **Slot-delta sweep**: align the Redis inflight count with running jobs that hold a slot (including
   async judge on ``benchmark-running``); refill when low / decrease when high / renew when equal

Invariants: do not invent a wait entry with no running job; requeue goes through ``requeue_or_fail_from_running``
and must not rewrite ``first_enqueued_at``; Aone/provider ``pending|running`` waits for the callback;
if GET can restore the callback body, go through ``handle_callback``; otherwise ``completed`` requeues to
wait only after a lost callback is confirmed.
"""

import asyncio
import socket
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import redis.asyncio as aioredis
from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Job, Queue
from tortoise import Tortoise

from benchmark.settings.setting import Settings
from benchmark.records.enums import BenchmarkStatus
from benchmark.records.models.benchmark_run import BenchmarkRun
from benchmark.libs.jobqueue.job_listing import get_queue_job, list_queue_jobs
from benchmark.libs.queue_runtime import get_redis
from benchmark.runs.common.external_client import NOTIFICATION_IN_FLIGHT_JOB_STATUSES
class AttributionProviderRegistry:
    pass
JobStatusMapSpec = None
map_external_job_status_to_callback = None
from benchmark.runs.run_service import BenchmarkService
from benchmark.runs.run_stop import (
    count_running_benchmark_items,
    is_stop_requested,
    reconcile_stop_target_to_processed,
    stop_attribution_pending,
    stop_reconcile_in_grace,
)
from benchmark.runs.stop_inflight import run_has_stop_inflight
from benchmark.queue.inflight_slot import (
    SlotKeySpec,
    build_cluster_slot_key,
    build_slot_key,
    clamp_slot_count,
    release_cluster_slot,
    release_slot,
    scan_slot_keys,
)
from benchmark.queue.queue_factory import (
    ATTRIBUTION_RUNNING,
    ATTRIBUTION_WAIT,
    BENCHMARK_CASES,
    BENCHMARK_RUNNING,
    EXTERNAL_RUNNING,
    EXTERNAL_WAIT,
    EvalQueueRegistry,
)
from benchmark.queue.retry_requeue import (
    RequeueDeps,
    RequeueOpts,
    force_fail_from_running,
    requeue_or_fail_from_running,
    retry_policy_from_settings,
    stamp_first_running_at,
)
from benchmark.workers.finalize_trigger import (
    try_finalize_benchmark,
    try_finalize_external,
)

if TYPE_CHECKING:
    from benchmark.runs.common.external_client import ExternalJudgeClient

logger = get_logger("agent_eval.eval_workers.running_healer")

#: Node identity (leader-election value)
_WORKER_NAME = socket.gethostname()

#: D19 leader-election Lua script — safe renew (only the holder may renew)
_RENEW_LEADER_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
end
return 0
"""

#: The running queue has no Worker; marker jobs land in waiting
_RUNNING_JOB_TYPES = ["waiting", "delayed", "paused", "prioritized"]
#: In-flight states on the wait queue (used by the D16 heartbeat / stale guard)
_WAIT_JOB_TYPES = ["waiting", "delayed", "active", "paused", "prioritized"]
#: Short-lock TTL for the attribution sweep when D19 is off, so two nodes do not heal at once
_ATTR_HEAL_LOCK_TTL_S = 30

#: When provider ``completed`` is first seen and PG has not landed yet, write it into job.data,
#: then wait one more ``callback_timeout_ms`` before calling it a lost callback (avoid requeueing one still in flight).
_COMPLETED_SEEN_AT_KEY = "callback_completed_seen_at"

_SLOT_KINDS = frozenset({"judge", "attr", "sync"})


def _is_provider_job_in_flight(status: object) -> bool:
    """The Aone/provider job is still queued or running (callback not terminal)."""
    return str(status or "").strip().lower() in NOTIFICATION_IN_FLIGHT_JOB_STATUSES


def _running_job_slot_kind(
    job: Job,
    *,
    queue_name: str,
    default_kind: str,
) -> str:
    """Resolve the slot-holding kind from a running job (judge/sync share the unified queue)."""
    data = dict(job.data or {})
    kind = str(data.get("cluster_kind") or "").strip()
    if kind in _SLOT_KINDS:
        return kind
    if (job.name or "") == "pending_callback":
        return "attr" if queue_name == ATTRIBUTION_RUNNING else "judge"
    if queue_name == BENCHMARK_RUNNING:
        return "sync"
    return default_kind


HeartbeatFn = Callable[[str], Awaitable[None]]
MarkFailedFn = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class HealerCallbacks:
    """Optional callbacks / dependency injection for RunningHealer (constructor parameter object).

    Heartbeat / terminal-fail markers / AGGREGATING recovery callbacks are provided in pairs per queue
    family; external clients, the provider registry, and the run service are injected here too. All may
    be omitted (the matching healer capability is enabled only when present)."""

    benchmark_heartbeat: HeartbeatFn | None = None
    benchmark_mark_failed: MarkFailedFn | None = None
    external_mark_failed: MarkFailedFn | None = None
    external_aggregating_finalize: HeartbeatFn | None = None
    benchmark_aggregating_finalize: HeartbeatFn | None = None
    external_judge_client: "AoneJudgeClient | None" = None
    attribution_provider_registry: AttributionProviderRegistry | None = None
    benchmark_run_service: BenchmarkService | None = None
    external_callback_handler: Callable[[Any], Awaitable[None]] | None = None


@dataclass
class SyncHealContext:
    """One-pass scan context for ``_heal_one_sync_inflight`` (queues + grace / terminal-fail policy)."""

    wait_q: Queue
    running_q: Queue
    grace_ms: int
    now_ms: int
    mark_failed: MarkFailedFn | None


@dataclass
class CallbackVerdictSpec:
    """Verdict arguments for ``_completed_callback_verdict`` (callback landed + grace window + log prefix)."""

    applied: bool
    grace_ms: int
    now_ms: int
    log_prefix: str


@dataclass
class CallbackReplayContext:
    """Replay context for ``_replay_callback_from_job_status`` (job / queue / data / platform id)."""

    job: Job
    running_q: Queue
    data: dict[str, Any]
    platform_job_id: str


@dataclass
class CallbackRequeueContext:
    """Requeue context for ``_requeue_callback_timeout_case`` (queue / data / platform status)."""

    running_q: Queue
    wait_q: Queue
    data: dict[str, Any]
    notification_job_id: str
    notification_status: str


@dataclass
class HealJobContext:
    """Shared arguments for ``_heal_*`` callback-timeout / inflight cleanup (FUNC-PARAMS-001)."""

    job: Job
    running_q: Queue
    wait_q: Queue
    data: dict[str, Any]


@dataclass(frozen=True)
class SlotClampSpec:
    """Clamp spec for ``_clamp_slot_counters_for_kind`` (FUNC-PARAMS-001)."""

    queue_names: list[str]
    slot_kind: str
    slot_pattern: str
    ttl_ms: int


@dataclass(frozen=True)
class InflightCleanGuard:
    """Race-cleanup arguments for ``_clean_inflight_guarded`` (FUNC-PARAMS-001)."""

    job: Job
    data: dict[str, Any]
    race_tag: str
    case_job_id: str
    wait_state: str


async def collect_active_run_ids(queue_manager: EvalQueueRegistry) -> set[str]:
    """Collect ``run_id``s that still have a job on the wait/running queues (D16)."""
    run_ids: set[str] = set()
    pairs = (
        (BENCHMARK_CASES, _WAIT_JOB_TYPES),
        (BENCHMARK_RUNNING, _RUNNING_JOB_TYPES),
        (EXTERNAL_WAIT, _WAIT_JOB_TYPES),
        (EXTERNAL_RUNNING, _RUNNING_JOB_TYPES),
        (ATTRIBUTION_WAIT, _WAIT_JOB_TYPES),
        (ATTRIBUTION_RUNNING, _RUNNING_JOB_TYPES),
    )
    for queue_name, types in pairs:
        queue = queue_manager.get_queue(queue_name)
        try:
            jobs = await list_queue_jobs(queue, types)
        except Exception:
            logger.warning(
                "healer.collect_jobs_failed queue=%s",
                queue_name,
                exc_info=True,
            )
            continue
        for job in jobs:
            if job is None:
                continue
            rid = (job.data or {}).get("run_id")
            if rid:
                run_ids.add(str(rid))
    return run_ids


async def collect_running_queue_run_ids(queue_manager: EvalQueueRegistry) -> set[str]:
    """Collect ``run_id``s that still have a job on the running queues.

    Unlike ``collect_active_run_ids``, this scans only the running queues
    (BENCHMARK_RUNNING / EXTERNAL_RUNNING /
    ATTRIBUTION_RUNNING) and does not include the wait queues. Used as the reverse guard for
    ``_heal_stuck_external_from_pg``, so a stuck job in
    ATTRIBUTION_WAIT does not block PG recovery.
    """
    run_ids: set[str] = set()
    pairs = (
        (BENCHMARK_RUNNING, _RUNNING_JOB_TYPES),
        (EXTERNAL_RUNNING, _RUNNING_JOB_TYPES),
        (ATTRIBUTION_RUNNING, _RUNNING_JOB_TYPES),
    )
    for queue_name, types in pairs:
        queue = queue_manager.get_queue(queue_name)
        try:
            jobs = await list_queue_jobs(queue, types)
        except Exception:
            logger.warning(
                "healer.collect_running_jobs_failed queue=%s",
                queue_name,
                exc_info=True,
            )
            continue
        for job in jobs:
            if job is None:
                continue
            rid = (job.data or {}).get("run_id")
            if rid:
                run_ids.add(str(rid))
    return run_ids


async def recover_stale_with_queue_guard(
    *,
    queue_manager: EvalQueueRegistry,
    recover_fn: Callable[..., Awaitable[int]],
    heartbeat_fn: HeartbeatFn | None = None,
    stale_timeout_s: int | None = None,
) -> int:
    """D16: refresh the heartbeat for in-flight runs (wait/running non-empty) before a stale kill."""
    active = await collect_active_run_ids(queue_manager)
    if active and heartbeat_fn is not None:
        for run_id in active:
            try:
                await heartbeat_fn(run_id)
            except Exception:
                logger.warning(
                    "healer.guard_heartbeat_failed run_id=%s",
                    run_id,
                    exc_info=True,
                )
        logger.info(
            "healer.d16_guard protected_runs=%d (heartbeat refreshed before stale recover)",
            len(active),
        )

    if stale_timeout_s is not None:
        return await recover_fn(stale_timeout_s=stale_timeout_s)
    return await recover_fn()


class RunningHealer:
    """Background sweep: heartbeat + orphan running recovery + external TTL."""

    def __init__(
        self,
        *,
        queue_manager: EvalQueueRegistry,
        settings: Settings,
        callbacks: HealerCallbacks | None = None,
    ) -> None:
        cb = callbacks or HealerCallbacks()
        self._qm = queue_manager
        self._settings = settings
        self._benchmark_heartbeat = cb.benchmark_heartbeat
        self._benchmark_mark_failed = cb.benchmark_mark_failed
        self._external_mark_failed = cb.external_mark_failed
        self._external_aggregating_finalize = cb.external_aggregating_finalize
        self._benchmark_aggregating_finalize = cb.benchmark_aggregating_finalize
        self._external_judge_client = cb.external_judge_client
        self._attribution_provider_registry = cb.attribution_provider_registry
        self._benchmark_run_service = cb.benchmark_run_service
        self._external_callback_handler = cb.external_callback_handler
        self._task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        # D19: leader-election state
        self._leader_redis: aioredis.Redis | None = None
        self._is_leader = False
        self._leader_lock_key = f"{queue_manager.prefix}:healer:leader"
        self._leader_renew_script: object | None = None
        # D20: tick cache
        self._tick_active_ids: set[str] | None = None
        # D20: running-queue tick cache (wait queues excluded)
        self._tick_running_ids: set[str] | None = None

    @property
    def interval_s(self) -> int:
        """Sweep interval: the smaller of the sync and external heal intervals."""
        return min(
            self._settings.sync_slot_heal_interval_s,
            self._settings.external_slot_heal_interval_s,
        )

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopped.clear()
        # D19: open the leader-election Redis connection
        if not self._settings.eval_queue_healer_leader_disabled:
            redis_client = aioredis.from_url(
                self._qm.redis_url,
                decode_responses=True,
            )
            self._leader_redis = redis_client
            self._leader_renew_script = redis_client.register_script(
                _RENEW_LEADER_LUA,
            )  # type: ignore[union-attr]
            logger.info(
                "healer.leader_election_enabled lock_key=%s ttl=%ds renew=%ds",
                self._leader_lock_key,
                self._settings.healer_leader_ttl_s,
                self._settings.healer_leader_renew_s,
            )
        else:
            logger.warning("healer.leader_election_disabled (all nodes will tick)")
        self._task = asyncio.create_task(
            self._loop(),
            name="eval-queue-running-healer",
        )
        logger.info("healer.started interval_s=%d", self.interval_s)

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        # D19: release the leader-election lock + close Redis
        if self._is_leader and self._leader_redis is not None:
            try:
                await self._leader_redis.delete(self._leader_lock_key)
            except Exception:
                logger.warning("healer.leader_lock_release_failed", exc_info=True)
        self._is_leader = False
        if self._leader_redis is not None:
            await self._leader_redis.aclose()
            self._leader_redis = None
        logger.info("healer.stopped")

    # ── D19 leader election ─────────────────────────────────────

    async def _try_acquire_leader_lock(self) -> bool:
        """Try to acquire the leader-election lock (SET NX EX)."""
        if self._leader_redis is None:
            return True  # leader election disabled → always the leader
        try:
            acquired = await self._leader_redis.set(
                self._leader_lock_key,
                _WORKER_NAME,
                nx=True,
                ex=self._settings.healer_leader_ttl_s,
            )
            return bool(acquired)
        except Exception:
            logger.warning("healer.leader_acquire_failed", exc_info=True)
            return False

    async def _renew_leader_lock(self) -> bool:
        """Renew the leader-election lock (only the holder may renew)."""
        if self._leader_redis is None:
            return True
        assert self._leader_renew_script is not None
        try:
            result = await self._leader_renew_script(  # type: ignore[operator]
                keys=[self._leader_lock_key],
                args=[_WORKER_NAME, self._settings.healer_leader_ttl_s],
            )
            return bool(int(result))  # type: ignore[arg-type]
        except Exception:
            logger.warning("healer.leader_renew_failed", exc_info=True)
            return False

    async def _ensure_leader(self) -> bool:
        """Check / acquire leader identity."""
        if self._leader_redis is None:
            return True  # leader election disabled
        if self._is_leader:
            self._is_leader = await self._renew_leader_lock()
            if not self._is_leader:
                logger.warning("healer.leadership_lost (renew failed)")
        else:
            self._is_leader = await self._try_acquire_leader_lock()
            if self._is_leader:
                logger.info("healer.acquired_leadership node=%s", _WORKER_NAME)
        return self._is_leader

    async def _loop(self) -> None:
        renew_at = 0.0
        while not self._stopped.is_set():
            now = time.time()
            # D19: leader election / renew
            if now >= renew_at:
                is_leader = await self._ensure_leader()
                renew_at = now + self._settings.healer_leader_renew_s
                if not is_leader:
                    try:
                        await asyncio.wait_for(
                            self._stopped.wait(),
                            timeout=float(self.interval_s),
                        )
                        break
                    except TimeoutError:
                        continue
            try:
                await self.tick()
            except Exception:
                logger.exception("healer.tick_failed")
            try:
                await asyncio.wait_for(
                    self._stopped.wait(),
                    timeout=float(self.interval_s),
                )
                break
            except TimeoutError:
                continue

    async def tick(self) -> dict[str, int]:
        """Run one sweep (unit-testable). Return per-action counts."""
        stats = {
            "heartbeat_runs": 0,
            "sync_cleaned": 0,
            "sync_requeued": 0,
            "external_callback_timeout": 0,
            "external_inflight_cleaned": 0,
            "slot_excess_released": 0,
            "external_aggregating_recovered": 0,
            "external_stuck_running_recovered": 0,
            "benchmark_aggregating_recovered": 0,
            "attribution_callback_timeout": 0,
            "attribution_inflight_cleaned": 0,
            "attribution_slot_clamped": 0,
            "attribution_heal_skipped": 0,
            "attribution_wait_stuck": 0,
            "external_wait_stuck": 0,
            "external_run_deadline_failed": 0,
            "stop_target_reconciled": 0,
        }
        # D20: tick cache — collect_active_run_ids is called only once
        self._tick_active_ids = await collect_active_run_ids(self._qm)
        # D20: running-queue cache — reverse guard for _heal_stuck_external_from_pg
        self._tick_running_ids = await collect_running_queue_run_ids(self._qm)
        try:
            stats["heartbeat_runs"] = await self._refresh_heartbeats()
            s_clean, s_req = await self._heal_sync_pair(
                wait_name=BENCHMARK_CASES,
                running_name=BENCHMARK_RUNNING,
                mark_failed=self._benchmark_mark_failed,
            )
            stats["sync_cleaned"] += s_clean
            stats["sync_requeued"] += s_req
            ext_timeout, ext_clean = await self._heal_external()
            stats["external_callback_timeout"] = ext_timeout
            stats["external_inflight_cleaned"] = ext_clean
            stats["slot_excess_released"] = await self._heal_slot_counters()
            stats["external_aggregating_recovered"] = await self._heal_aggregating_external()
            stats["external_stuck_running_recovered"] = await self._heal_stuck_external_from_pg()
            stats["external_run_deadline_failed"] = await self._heal_external_run_deadline()
            # EXTERNAL_WAIT stuck detection (fallback for the BullMQ stalled check)
            stats["external_wait_stuck"] = await self._heal_external_wait_stuck()
            stats["benchmark_aggregating_recovered"] = await self._heal_aggregating_sync(
                self._benchmark_aggregating_finalize,
            )
            stats["generating_report_recovered"] = await self._heal_stuck_generating_report()
            # stop-run: queue empty but stop_target unreachable → realign to processed, then finalize
            stats["stop_target_reconciled"] = await self._heal_stuck_stop_target()

            # Attribution-queue heal
            # D19 already guarantees only the leader enters tick; the short lock is only for when election is disabled
            use_attr_lock = bool(
                getattr(self._settings, "eval_queue_healer_leader_disabled", False),
            )
            attr_lock_key = f"{self._qm.prefix}:heal:lock:attr"
            attr_lock_held = False
            if use_attr_lock:
                attr_lock_held = await self._try_acquire_heal_lock(
                    attr_lock_key,
                    ttl_s=_ATTR_HEAL_LOCK_TTL_S,
                )
                if not attr_lock_held:
                    stats["attribution_heal_skipped"] = 1
                    logger.debug("healer.attr_skipped_locked")
            if not use_attr_lock or attr_lock_held:
                try:
                    attr_timeout, attr_clean = await self._heal_attribution()
                    stats["attribution_callback_timeout"] = attr_timeout
                    stats["attribution_inflight_cleaned"] = attr_clean
                    attr_slot_clamped = await self._heal_attr_slot_counters()
                    stats["attribution_slot_clamped"] = attr_slot_clamped
                finally:
                    if attr_lock_held:
                        await self._release_heal_lock(attr_lock_key)
                # ATTRIBUTION_WAIT stuck detection (fallback for the BullMQ stalled check)
                stats["attribution_wait_stuck"] = await self._heal_attribution_wait_stuck()
        finally:
            self._tick_active_ids = None
            self._tick_running_ids = None
        if any(stats.values()):
            logger.info("healer.tick %s", stats)
        return stats

    def _get_tick_active_ids(self) -> set[str]:
        """D20: return the tick-cached active run_ids."""
        return self._tick_active_ids or set()

    def _get_tick_running_ids(self) -> set[str]:
        """D20: return the tick-cached running-queue run_ids."""
        return self._tick_running_ids or set()

    async def _refresh_heartbeats(self) -> int:
        active = self._tick_active_ids or set()
        if not active:
            return 0
        for run_id in active:
            if self._benchmark_heartbeat is not None:
                try:
                    await self._benchmark_heartbeat(run_id)
                except Exception:
                    logger.warning(
                        "healer.heartbeat_benchmark_failed run_id=%s",
                        run_id,
                        exc_info=True,
                    )
        return len(active)

    async def _heal_sync_pair(
        self,
        *,
        wait_name: str,
        running_name: str,
        mark_failed: MarkFailedFn | None,
    ) -> tuple[int, int]:
        """Return ``(cleaned_orphan_markers, requeued_or_failed)``."""
        wait_q = self._qm.get_queue(wait_name)
        running_q = self._qm.get_queue(running_name)
        grace_ms = self._settings.eval_queue_lock_duration_ms
        cleaned = 0
        requeued = 0
        try:
            jobs = await list_queue_jobs(running_q, _RUNNING_JOB_TYPES)
        except Exception:
            logger.warning(
                "healer.list_running_failed queue=%s",
                running_name,
                exc_info=True,
            )
            return 0, 0

        now_ms = int(time.time() * 1000)
        ctx = SyncHealContext(
            wait_q=wait_q,
            running_q=running_q,
            grace_ms=grace_ms,
            now_ms=now_ms,
            mark_failed=mark_failed,
        )
        for job in jobs:
            if job is None or (job.name and job.name != "inflight"):
                continue
            try:
                action = await self._heal_one_sync_inflight(job, ctx)
            except Exception:
                logger.warning(
                    "healer.sync_inflight_failed job_id=%s",
                    job.id,
                    exc_info=True,
                )
                continue
            if action == "cleaned":
                cleaned += 1
            elif action in ("requeued", "failed"):
                requeued += 1
        return cleaned, requeued

    async def _heal_one_sync_inflight(
        self,
        job: Job,
        ctx: SyncHealContext,
    ) -> str:
        wait_q = ctx.wait_q
        running_q = ctx.running_q
        grace_ms = ctx.grace_ms
        now_ms = ctx.now_ms
        mark_failed = ctx.mark_failed
        data = dict(job.data or {})
        run_id = data.get("run_id")
        item_id = data.get("item_id")
        if not run_id or not item_id:
            return "skip"
        age_ms = now_ms - int(job.timestamp or 0)
        if age_ms < grace_ms:
            return "skip"

        case_job_id = f"{run_id}:{item_id}"
        slot_key = data.get("slot_key")
        wait_job = await get_queue_job(wait_q, case_job_id)

        if wait_job is not None:
            try:
                state = await wait_job.getState()
            except Exception:
                state = "unknown"
            if state == "active":
                return "skip"
            # wait is already back on the queue from stalled: only clear the orphan running + slot, do not bump retry
            await job.remove()
            if slot_key:
                await release_slot(str(slot_key))
            logger.info(
                "healer.sync_orphan_cleaned job_id=%s wait_state=%s age_ms=%d",
                case_job_id,
                state,
                age_ms,
            )
            return "cleaned"

        # wait is already gone: must requeue running→wait or fail terminally
        if slot_key:
            await release_slot(str(slot_key))
        if mark_failed is None:
            await job.remove()
            logger.warning(
                "healer.sync_orphan_no_mark_failed job_id=%s → removed only",
                case_job_id,
            )
            return "cleaned"

        outcome = await requeue_or_fail_from_running(
            RequeueDeps(
                running_queue=running_q,
                wait_queue=wait_q,
                policy=retry_policy_from_settings(
                    self._settings,
                    retry_max_duration_ms=self._settings.sync_retry_max_duration_ms,
                ),
                mark_failed=mark_failed,
            ),
            case_job_id,
        )
        logger.info(
            "healer.sync_orphan_requeue job_id=%s outcome=%s age_ms=%d",
            case_job_id,
            outcome,
            age_ms,
        )
        return outcome

    async def _heal_external(self) -> tuple[int, int]:
        """Return ``(callback_timeout_handled, inflight_cleaned)`` (sum of both queue pairs).

        §8.3 queue merge: a unified run's pending_callback lives on
        ``benchmark-running``/``benchmark-cases``, scanned separately from the legacy
        ``external-running``/``external-wait`` pair.
        """
        timeout_n = 0
        clean_n = 0
        for running_name, wait_name in (
            (EXTERNAL_RUNNING, EXTERNAL_WAIT),
            (BENCHMARK_RUNNING, BENCHMARK_CASES),
        ):
            handled, cleaned = await self._heal_external_pair(
                running_name=running_name,
                wait_name=wait_name,
            )
            timeout_n += handled
            clean_n += cleaned
        return timeout_n, clean_n

    async def _heal_external_pair(
        self,
        *,
        running_name: str,
        wait_name: str,
    ) -> tuple[int, int]:
        """Callback timeout + inflight cleanup for one wait/running queue pair (shared by the §8.3 merge).

        ``pending_callback`` past ``external_callback_timeout_ms`` →
        query Aone job status → branch (skip ``running``; ``completed`` requeues to wait only after
        the PG/in-flight observation calls it a lost callback; other non-running states requeue to wait;
        past TTL / the per-case window and PG not terminal → ``force_fail``).

        ``inflight`` past ``external_wait_lock_duration_ms`` →
        check the wait queue → clean up or ``requeue_or_fail``.
        """
        running_q = self._qm.get_queue(running_name)
        wait_q = self._qm.get_queue(wait_name)
        callback_timeout_ms = self._settings.external_callback_timeout_ms
        grace_ms = self._settings.external_wait_lock_duration_ms
        timeout_n = 0
        clean_n = 0
        try:
            jobs = await list_queue_jobs(running_q, _RUNNING_JOB_TYPES)
        except Exception:
            logger.warning("healer.list_external_running_failed", exc_info=True)
            return 0, 0

        now_ms = int(time.time() * 1000)
        for job in jobs:
            if job is None:
                continue
            data = dict(job.data or {})
            age_ms = now_ms - int(job.timestamp or 0)
            name = job.name or ""
            try:
                if name == "pending_callback":
                    overdue = self._is_case_window_exhausted(
                        data,
                        now_ms=now_ms,
                    )
                    if age_ms < callback_timeout_ms and not overdue:
                        continue
                    if await self._heal_external_callback_timeout(
                        HealJobContext(
                            job=job,
                            running_q=running_q,
                            wait_q=wait_q,
                            data=data,
                        ),
                    ):
                        timeout_n += 1
                elif name == "inflight":
                    overdue = self._is_case_window_exhausted(
                        data,
                        now_ms=now_ms,
                    )
                    if age_ms < grace_ms and not overdue:
                        continue
                    if await self._heal_external_inflight(
                        HealJobContext(
                            job=job,
                            running_q=running_q,
                            wait_q=wait_q,
                            data=data,
                        ),
                    ):
                        clean_n += 1
            except Exception:
                logger.warning(
                    "healer.external_job_failed job_id=%s name=%s",
                    job.id,
                    name,
                    exc_info=True,
                )
        return timeout_n, clean_n

    async def _run_has_stop_inflight(self, run_id: str) -> bool:
        """Whether a truly in-flight job remains (same ``stop_inflight`` rule as the console)."""
        return await run_has_stop_inflight(self._qm, run_id, fail_closed=True)

    async def _list_stop_latched_running_ids(self) -> list[str]:
        """Find run_ids still running that already have the stop latch (at most 50)."""
        try:
            conn = Tortoise.get_connection("default")
            rows = await conn.execute_query(
                """
                SELECT run_id FROM benchmark_runs
                WHERE status = 'running'
                  AND deleted_at IS NULL
                  AND (config_json->>'stop_requested_at') IS NOT NULL
                  AND run_id NOT LIKE 'de_%'
                LIMIT 50
                """,
                [],
            )
        except Exception:
            logger.warning("healer.stuck_stop_query_failed", exc_info=True)
            return []
        if not (rows and rows[1]):
            return []
        out: list[str] = []
        for row in rows[1]:
            run_id = str(row[0]) if isinstance(row, (list, tuple)) else str(row.get("run_id", ""))
            if run_id:
                out.append(run_id)
        return out

    async def _finalize_after_stop_reconcile(
        self,
        run_id: str,
        svc: BenchmarkService,
    ) -> None:
        """After pressing stop_target down, best-effort fire the external / sync finalize CAS."""
        try:
            await try_finalize_external(run_id, svc)
        except Exception:
            logger.warning(
                "healer.stop_finalize_external_failed run_id=%s",
                run_id,
                exc_info=True,
            )
        try:
            await try_finalize_benchmark(run_id, svc)
        except Exception:
            logger.warning(
                "healer.stop_finalize_benchmark_failed run_id=%s",
                run_id,
                exc_info=True,
            )

    async def _stop_reconcile_defer_reason(
        self,
        run_id: str,
        config: dict[str, Any],
    ) -> str | None:
        """Return why stop reconcile should wait; ``None`` means the target may be pressed / finalized."""
        if stop_reconcile_in_grace(config):
            return "within mark→align grace"
        if await self._run_has_stop_inflight(run_id):
            return "queue inflight"
        running_items = await count_running_benchmark_items(run_id)
        if running_items > 0:
            return f"running_items={running_items}"
        # Registered attribution is not complete: do not press expected / finalize early (attribution submit is unaffected)
        attr_pending = stop_attribution_pending(config)
        if attr_pending > 0:
            return f"attribution_pending={attr_pending}"
        return None

    async def _reconcile_one_stuck_stop(
        self,
        run_id: str,
        svc: BenchmarkService,
    ) -> bool:
        """One stuck stop run: if nothing is truly in flight, align the expectation and finalize. Return True on success.

        Call ``reconcile_stop_target_to_processed`` only with nothing in flight and past ``STOP_RECONCILE_GRACE_S``:
        covers a missing stop_target, a provisional value that is too high, an interrupted align that never pressed
        expected, and similar cases, so external ``received < expected`` does not stick on running forever. Skip
        inside the grace window so it does not overlap the console mark→drain→align window and CAS early.
        """
        try:
            run = await svc.get_run(run_id)
        except Exception:
            logger.warning(
                "healer.stuck_stop_get_run_failed run_id=%s",
                run_id,
                exc_info=True,
            )
            return False
        if run is None or not isinstance(run.config_json, dict):
            return False
        config = run.config_json
        if not is_stop_requested(config):
            return False
        defer = await self._stop_reconcile_defer_reason(run_id, config)
        if defer is not None:
            logger.info(
                "healer.stop_reconcile_deferred run_id=%s reason=%s",
                run_id,
                defer,
            )
            return False
        processed = int(run.evaluated_items or 0) + int(run.failed_items or 0)
        raw_target = config.get("stop_target_count")
        aligned = await reconcile_stop_target_to_processed(run_id)
        logger.warning(
            "healer.stop_target_reconciled run_id=%s old_target=%s new_target=%s processed=%d",
            run_id,
            raw_target,
            aligned.get("stop_target_count"),
            processed,
        )
        await self._finalize_after_stop_reconcile(run_id, svc)
        return True

    async def _heal_stuck_stop_target(self) -> int:
        """Stop fallback: once nothing is truly in flight and the grace window has passed, align the expectation to processed and finalize.

        Covers: double-count/miscount making stop_target unreachable, align never landing, external
        expected never compressed, and similar; automatic cleanup after the API already returned waiting_inflight.
        """
        svc = self._benchmark_run_service
        if svc is None:
            return 0
        recovered = 0
        for run_id in await self._list_stop_latched_running_ids():
            if await self._reconcile_one_stuck_stop(run_id, svc):
                recovered += 1
        return recovered

    async def _heal_stuck_external_from_pg(self) -> int:
        """Recover, via a PG query, an external run whose BullMQ queues are empty but that is stuck in running/aggregating.

        Case: every judge callback has arrived (``judge_received_count >= judge_expected_count``),
        but the ``try_finalize_external`` CAS failed because ``attribution_received_count``
        is not satisfied. The BullMQ queues are empty → ``collect_active_run_ids`` cannot find
        the run → the healer cannot recover it.

        This method queries PG directly for such stuck runs and calls ``_finalize_run`` to advance them.
        Phase 1 of ``_finalize_run`` is an unconditional RUNNING → AGGREGATING,
        Phase 2 is CAS AGGREGATING → GENERATING_REPORT; both are idempotent and safe.
        """
        if self._external_aggregating_finalize is None:
            return 0
        try:
            conn = Tortoise.get_connection("default")
            rows = await conn.execute_query(
                """
                SELECT run_id FROM benchmark_runs
                WHERE status IN ('running', 'aggregating')
                  AND run_id NOT LIKE 'de_%'
                  AND (config_json->>'judge_received_count')::int
                       >= COALESCE((config_json->>'judge_expected_count')::int, 0)
                  AND updated_at < NOW() - INTERVAL '5 minutes'
                """,
                [],
            )
        except Exception:
            logger.warning("healer.stuck_external_query_failed", exc_info=True)
            return 0
        if not (rows and rows[1]):
            return 0
        # Exclude runs already found in BullMQ running queues
        # (handled by _heal_aggregating_external)
        # Use _tick_running_ids (not _tick_active_ids) to avoid ATTRIBUTION_WAIT
        # jobs blocking recovery of runs stuck due to lost attribution jobs.
        active = self._tick_running_ids or set()
        count = 0
        for row in rows[1]:
            run_id = str(row[0]) if isinstance(row, (list, tuple)) else str(row.get("run_id", ""))
            if not run_id or run_id in active:
                continue
            try:
                await self._external_aggregating_finalize(run_id)
                count += 1
                logger.info(
                    "healer.stuck_external_recovered run_id=%s "
                    "(judge complete but attribution/CAS stuck)",
                    run_id,
                )
            except Exception:
                logger.warning(
                    "healer.stuck_external_finalize_failed run_id=%s",
                    run_id,
                    exc_info=True,
                )
        return count

    async def _heal_external_run_deadline(self) -> int:
        """If the whole run exceeds ``external_run_max_duration_ms`` (default 24h), finish it as failed.

        Independent of the per-case window: a case timeout fails only that case; this method uses the
        ``run.created_at`` wall clock, clears the remaining wait/running jobs, and ``fail_run``.
        Only RUNNING runs whose run_id contains ``external``, so a long sync eval is not killed by mistake.
        """
        max_ms = int(
            getattr(self._settings, "external_run_max_duration_ms", 0) or 0,
        )
        if max_ms <= 0 or self._benchmark_run_service is None:
            return 0
        cutoff = datetime.now(UTC) - timedelta(milliseconds=max_ms)
        try:
            overdue = await BenchmarkRun.filter(
                status=BenchmarkStatus.RUNNING,
                created_at__lt=cutoff,
                run_id__contains="external",
            ).only("run_id")
        except Exception:
            logger.warning("healer.run_deadline_query_failed", exc_info=True)
            return 0
        count = 0
        for run in overdue:
            run_id = str(run.run_id)
            try:
                await self._fail_run_for_deadline(run_id)
                count += 1
            except Exception:
                logger.warning(
                    "healer.run_deadline_fail_failed run_id=%s",
                    run_id,
                    exc_info=True,
                )
        return count

    async def _fail_run_for_deadline(self, run_id: str) -> None:
        """Release slots, remove queue jobs, and mark the run failed."""
        drained = await self._drain_queue_jobs_for_run(run_id)
        svc = self._benchmark_run_service
        if svc is None:
            return
        await svc.fail_run(run_id=run_id, reason="run_max_duration")
        logger.warning(
            "healer.run_deadline_failed run_id=%s drained=%d",
            run_id,
            drained,
        )

    async def _drain_queue_jobs_for_run(self, run_id: str) -> int:
        """Remove this run's jobs on the wait+running queues and release slots (including the unified queues, §8.3)."""
        pairs: tuple[tuple[str, list[str], str], ...] = (
            (EXTERNAL_RUNNING, _RUNNING_JOB_TYPES, "judge"),
            (BENCHMARK_RUNNING, _RUNNING_JOB_TYPES, "judge"),
            (ATTRIBUTION_RUNNING, _RUNNING_JOB_TYPES, "attr"),
            (EXTERNAL_WAIT, _WAIT_JOB_TYPES, ""),
            (BENCHMARK_CASES, _WAIT_JOB_TYPES, ""),
            (ATTRIBUTION_WAIT, _WAIT_JOB_TYPES, ""),
        )
        removed = 0
        for queue_name, types, slot_kind in pairs:
            queue = self._qm.get_queue(queue_name)
            try:
                jobs = await list_queue_jobs(queue, types)
            except Exception:
                logger.warning(
                    "healer.run_deadline_list_failed queue=%s run_id=%s",
                    queue_name,
                    run_id,
                    exc_info=True,
                )
                continue
            for job in jobs:
                if job is None:
                    continue
                data = dict(job.data or {})
                if str(data.get("run_id") or "") != run_id:
                    continue
                if slot_kind == "judge":
                    await self._release_external_slots(data)
                elif slot_kind == "attr":
                    await self._release_attribution_slots(data)
                try:
                    await job.remove()
                except Exception:
                    logger.warning(
                        "healer.run_deadline_remove_failed job_id=%s run_id=%s",
                        job.id,
                        run_id,
                        exc_info=True,
                    )
                    continue
                removed += 1
        return removed

    async def _heal_stuck_generating_report(self) -> int:
        """Recover a run stuck in GENERATING_REPORT (process crashed during aggregate/report).

        Case: the Phase 2 CAS of ``_finalize_run`` / ``_finalize_benchmark``
        already moved status to ``generating_report``, then the process crashed during
        aggregate → report → complete. There is no BullMQ job to recover, ``_tick_active_ids``
        does not contain the run, and the healer's aggregating recovery path cannot reach it.

        This method queries PG for such runs and dispatches the recovery callback by type:
        - external → ``_external_aggregating_finalize`` (reset → aggregating → _finalize_run)
        - benchmark → ``_benchmark_aggregating_finalize`` (call _finalize_benchmark directly; idempotent)
        """
        try:
            conn = Tortoise.get_connection("default")
            rows = await conn.execute_query(
                """
                SELECT run_id FROM benchmark_runs
                WHERE status = 'generating_report'
                  AND run_id NOT LIKE 'de_%'
                  AND updated_at < NOW() - INTERVAL '5 minutes'
                """,
                [],
            )
        except Exception:
            logger.warning("healer.generating_report_query_failed", exc_info=True)
            return 0
        if not (rows and rows[1]):
            return 0
        active = self._tick_active_ids or set()
        count = 0
        for row in rows[1]:
            run_id = str(row[0]) if isinstance(row, (list, tuple)) else str(row.get("run_id", ""))
            if not run_id or run_id in active:
                continue
            try:
                if "external" in run_id:
                    if self._external_aggregating_finalize is not None:
                        await self._external_aggregating_finalize(run_id)
                    else:
                        continue
                elif self._benchmark_aggregating_finalize is not None:
                    await self._benchmark_aggregating_finalize(run_id)
                else:
                    continue
                count += 1
                logger.info(
                    "healer.generating_report_recovered run_id=%s",
                    run_id,
                )
            except Exception:
                logger.warning(
                    "healer.generating_report_finalize_failed run_id=%s",
                    run_id,
                    exc_info=True,
                )
        return count

    async def _heal_aggregating_external(self) -> int:
        """Recover an external run stuck in AGGREGATING.

        Phase 1 CAS of ``try_finalize_external`` checks ``status='running'``,
        so once a run enters ``aggregating`` the callback cannot fire again.
        Each sweep this method calls the finalize callback for active external runs;
        the callback checks whether status is ``aggregating`` and, if so, runs the Phase 2 CAS.
        """
        if self._external_aggregating_finalize is None:
            return 0
        active = self._tick_active_ids or set()
        if not active:
            return 0
        count = 0
        for run_id in active:
            try:
                await self._external_aggregating_finalize(run_id)
                count += 1
            except Exception:
                logger.warning(
                    "healer.external_aggregating_finalize_failed run_id=%s",
                    run_id,
                    exc_info=True,
                )
        return count

    async def _heal_aggregating_sync(
        self,
        callback: HeartbeatFn | None,
        *,
        prefix: str | None = None,
        exclude_prefix: str | None = None,
    ) -> int:
        """Recover a benchmark batch run stuck in AGGREGATING.

        Same shape as ``_heal_aggregating_external``:
        Phase 1 CAS of ``try_finalize_benchmark``
        checks ``status='running'``, so once a run enters ``aggregating`` it cannot fire again.
        Each sweep this method calls the finalize callback for active runs;
        the callback checks whether status is ``aggregating`` and, if so, finishes the run.
        """
        if callback is None:
            return 0
        active = self._tick_active_ids or set()
        if not active:
            return 0
        count = 0
        for run_id in active:
            if prefix is not None and not run_id.startswith(prefix):
                continue
            if exclude_prefix is not None and run_id.startswith(exclude_prefix):
                continue
            try:
                await callback(run_id)
                count += 1
            except Exception:
                logger.warning(
                    "healer.sync_aggregating_finalize_failed run_id=%s",
                    run_id,
                    exc_info=True,
                )
        return count

    async def _heal_slot_counters(self) -> int:
        """Delta sweep: align the Redis slot count with the real number of running slot holders.

        ``cluster_kind=judge`` / ``pending_callback`` on ``external-running`` + ``benchmark-running``
        count as judge slots; sync inflight of a benchmark batch counts as a sync slot.
        ``pending_callback`` / ``inflight`` still hold a slot.
        Refill when the count is low (TTL expired during a hold); decrease when it is high.
        """
        released = 0
        released += await self._clamp_slot_counters_for_kind(
            SlotClampSpec(
                queue_names=[EXTERNAL_RUNNING, BENCHMARK_RUNNING],
                slot_kind="judge",
                slot_pattern=f"{self._qm.prefix}:inflight:judge:*",
                ttl_ms=self._settings.external_running_ttl_ms,
            ),
        )
        released += await self._clamp_slot_counters_for_kind(
            SlotClampSpec(
                queue_names=[BENCHMARK_RUNNING],
                slot_kind="sync",
                slot_pattern=f"{self._qm.prefix}:inflight:sync:*",
                ttl_ms=self._settings.eval_queue_lock_duration_ms,
            ),
        )
        return released

    async def _clamp_slot_counters_for_kind(
        self,
        spec: SlotClampSpec,
    ) -> int:
        """Clamp the Redis slot count for one kind to the real number of running slot holders.

        Refill when low, decrease when high, renew when equal. If ``slot_key`` is missing, try to rebuild it
        (judge kind only; sync kind cannot be rebuilt because it includes an eval_type component, so SCAN is the fallback).
        On the unified queue, filter by ``cluster_kind`` / job name so judge and sync do not cross.
        """
        queue_names = spec.queue_names
        slot_kind = spec.slot_kind
        slot_pattern = spec.slot_pattern
        ttl_ms = spec.ttl_ms
        expected_by_key: Counter[str] = Counter()
        cluster_expected = 0

        for qn in queue_names:
            running_q = self._qm.get_queue(qn)
            try:
                jobs = await list_queue_jobs(running_q, _RUNNING_JOB_TYPES)
            except Exception:
                logger.warning(
                    "healer.list_running_for_slots_failed queue=%s",
                    qn,
                    exc_info=True,
                )
                continue

            for job in jobs:
                if job is None or (job.name or "") not in ("pending_callback", "inflight"):
                    continue
                if (
                    _running_job_slot_kind(
                        job,
                        queue_name=qn,
                        default_kind=slot_kind,
                    )
                    != slot_kind
                ):
                    continue
                data = dict(job.data or {})
                cluster_expected += 1
                slot_key = data.get("slot_key")
                if slot_key:
                    expected_by_key[str(slot_key)] += 1
                    continue
                # Rebuild a missing slot_key (judge kind only)
                if slot_kind == "judge":
                    node = data.get("submitter_node_id")
                    run_id = data.get("run_id")
                    if node and run_id:
                        reconstructed = build_slot_key(
                            self._qm.prefix,
                            SlotKeySpec(
                                kind="judge",
                                node=str(node),
                                run_id=str(run_id),
                            ),
                        )
                        expected_by_key[reconstructed] += 1

        ttl_s = max(1, int(ttl_ms) // 1000)
        released = 0
        try:
            cluster_key = build_cluster_slot_key(slot_kind)
            released += await clamp_slot_count(
                cluster_key,
                expected=cluster_expected,
                ttl_s=ttl_s,
            )
            scanned = set(await scan_slot_keys(slot_pattern))
            for key in scanned | set(expected_by_key):
                released += await clamp_slot_count(
                    key,
                    expected=expected_by_key.get(key, 0),
                    ttl_s=ttl_s,
                )
        except Exception:
            logger.warning(
                "healer.slot_counter_heal_failed kind=%s",
                slot_kind,
                exc_info=True,
            )
            return released

        if released > 0:
            logger.warning(
                "healer.slot_excess_released kind=%s n=%d cluster_expected=%d node_keys=%d",
                slot_kind,
                released,
                cluster_expected,
                len(expected_by_key),
            )
        return released

    def _is_case_window_exhausted(
        self,
        data: dict[str, Any],
        *,
        now_ms: int,
    ) -> bool:
        """Per-case window: starts at that case's own ``first_enqueued_at`` (first entry into running).

        Missing means it has not entered running yet, so the clock has not started.
        """
        first = data.get("first_enqueued_at")
        max_ms = int(self._settings.external_retry_max_duration_ms)
        if first is None:
            return False
        return now_ms >= int(first) + max_ms

    async def _release_external_slots(self, data: dict[str, Any]) -> None:
        slot_key = data.get("slot_key")
        if slot_key:
            await release_slot(str(slot_key))
        kind = data.get("cluster_kind") or "judge"
        await release_cluster_slot(str(kind))

    async def _is_judge_callback_applied(self, data: dict[str, Any]) -> bool:
        """PG fused item is already success/failed → the judge callback has landed."""
        svc = self._benchmark_run_service
        run_id = data.get("run_id")
        item_id = data.get("item_id")
        if svc is None or not run_id or not item_id:
            return False
        try:
            return await svc.is_external_judge_item_terminal(
                str(run_id),
                str(item_id),
            )
        except Exception:
            logger.warning(
                "healer.judge_callback_applied_query_failed run_id=%s item=%s",
                run_id,
                item_id,
                exc_info=True,
            )
            return False

    async def _is_attr_callback_applied(self, data: dict[str, Any]) -> bool:
        """attribution_received_item_ids already contains the item → the attribution callback has landed."""
        svc = self._benchmark_run_service
        run_id = data.get("run_id")
        item_id = data.get("item_id")
        if svc is None or not run_id or not item_id:
            return False
        try:
            return await svc.is_attribution_received(str(run_id), str(item_id))
        except Exception:
            logger.warning(
                "healer.attr_callback_applied_query_failed run_id=%s item=%s",
                run_id,
                item_id,
                exc_info=True,
            )
            return False

    async def _completed_callback_verdict(
        self,
        job: Job,
        data: dict[str, Any],
        spec: CallbackVerdictSpec,
    ) -> Literal["skip", "lost"]:
        """completed branch: already landed / callback in flight → skip; lost callback confirmed → lost.

        The first time completed is seen and PG is unwritten, stamp ``callback_completed_seen_at``,
        then wait ``grace_ms`` (same as callback_timeout) before requeueing to wait, so an in-flight
        callback and the healer do not double-write.
        """
        applied = spec.applied
        grace_ms = spec.grace_ms
        now_ms = spec.now_ms
        log_prefix = spec.log_prefix
        running_job_id = str(job.id)
        if applied:
            logger.info(
                "%s_completed_already_applied job_id=%s run_id=%s item_id=%s "
                "— skip (normal callback path)",
                log_prefix,
                running_job_id,
                data.get("run_id"),
                data.get("item_id"),
            )
            return "skip"

        raw_seen = data.get(_COMPLETED_SEEN_AT_KEY)
        try:
            seen_at = int(raw_seen) if raw_seen is not None else None
        except (TypeError, ValueError):
            seen_at = None

        if seen_at is None:
            data[_COMPLETED_SEEN_AT_KEY] = now_ms
            try:
                await job.updateData(data)
            except Exception:
                logger.warning(
                    "%s_completed_seen_stamp_failed job_id=%s",
                    log_prefix,
                    running_job_id,
                    exc_info=True,
                )
            logger.info(
                "%s_completed_in_flight job_id=%s run_id=%s item_id=%s "
                "grace_ms=%d — skip (callback may still be arriving)",
                log_prefix,
                running_job_id,
                data.get("run_id"),
                data.get("item_id"),
                grace_ms,
            )
            return "skip"

        if now_ms < seen_at + max(grace_ms, 0):
            logger.info(
                "%s_completed_in_flight job_id=%s seen_at=%s age_since_seen=%d grace_ms=%d — skip",
                log_prefix,
                running_job_id,
                seen_at,
                now_ms - seen_at,
                grace_ms,
            )
            return "skip"

        logger.info(
            "%s_completed_callback_lost job_id=%s run_id=%s item_id=%s "
            "seen_at=%s grace_ms=%d — requeue wait",
            log_prefix,
            running_job_id,
            data.get("run_id"),
            data.get("item_id"),
            seen_at,
            grace_ms,
        )
        return "lost"

    async def _replay_callback_from_job_status(
        self,
        resp: object,
        *,
        kind: Literal["judge", "attribution"],
        ctx: CallbackReplayContext,
    ) -> bool:
        """If the GET body can restore the callback JSON, go through ``handle_callback``; return True when cleanup succeeds."""
        job = ctx.job
        running_q = ctx.running_q
        data = ctx.data
        platform_job_id = ctx.platform_job_id
        handler = self._external_callback_handler
        if handler is None or resp is None:
            return False
        payload = map_external_job_status_to_callback(
            resp,
            JobStatusMapSpec(
                kind=kind,
                fallback_run_id=str(data.get("run_id") or ""),
                fallback_item_id=str(data.get("item_id") or "") or None,
                fallback_job_id=platform_job_id,
            ),
        )
        if payload is None:
            return False
        logger.info(
            "healer.callback_replay_from_get kind=%s job_id=%s "
            "run_id=%s item_id=%s platform_job_id=%s error=%s",
            kind,
            job.id,
            getattr(payload, "run_id", ""),
            getattr(payload, "item_id", ""),
            platform_job_id,
            getattr(payload, "error", None),
        )
        try:
            await handler(payload)
        except Exception:
            logger.exception(
                "healer.callback_replay_failed kind=%s job_id=%s platform_job_id=%s",
                kind,
                job.id,
                platform_job_id,
            )
            return False

        still = await get_queue_job(running_q, str(job.id))
        if still is None:
            logger.info(
                "healer.callback_replay_consumed kind=%s job_id=%s platform_job_id=%s",
                kind,
                job.id,
                platform_job_id,
            )
            return True

        applied = (
            await self._is_judge_callback_applied(data)
            if kind == "judge"
            else await self._is_attr_callback_applied(data)
        )
        if not applied:
            logger.warning(
                "healer.callback_replay_not_applied kind=%s job_id=%s "
                "platform_job_id=%s — keep running for grace/requeue",
                kind,
                job.id,
                platform_job_id,
            )
            return False

        try:
            await still.remove()
        except Exception:
            logger.warning(
                "healer.callback_replay_remove_failed kind=%s job_id=%s",
                kind,
                job.id,
                exc_info=True,
            )
        if kind == "judge":
            await self._release_external_slots(data)
        else:
            await self._release_attribution_slots(data)
        logger.info(
            "healer.callback_replay_applied_cleanup kind=%s job_id=%s platform_job_id=%s",
            kind,
            job.id,
            platform_job_id,
        )
        return True

    async def _force_fail_callback_timeout_case(
        self,
        job: Job,
        *,
        running_q: Queue,
        data: dict[str, Any],
    ) -> bool:
        """TTL / per-case window exceeded → release the slot + purge + force_fail (terminal-fail path)."""
        await self._release_external_slots(data)
        await self._purge_case_from_other_running(
            str(data.get("run_id") or ""),
            str(data.get("item_id") or ""),
            exclude_kind="judge",
        )
        if self._external_mark_failed is None:
            await job.remove()
            return True
        await force_fail_from_running(
            running_queue=running_q,
            running_job_id=str(job.id),
            mark_failed=self._external_mark_failed,
        )
        return True

    async def _requeue_callback_timeout_case(
        self,
        job: Job,
        ctx: CallbackRequeueContext,
    ) -> bool:
        """Lost callback / failed / error → release the slot + requeue to wait (keep first_enqueued_at)."""
        running_q = ctx.running_q
        wait_q = ctx.wait_q
        data = ctx.data
        notification_job_id = ctx.notification_job_id
        notification_status = ctx.notification_status
        running_job_id = str(job.id)
        await self._release_external_slots(data)
        if self._external_mark_failed is None:
            await job.remove()
            logger.info(
                "healer.callback_requeue_removed job_id=%s notification_job_id=%s "
                "aone_status=%s (no mark_failed)",
                running_job_id,
                notification_job_id,
                notification_status,
            )
            return True
        outcome = await requeue_or_fail_from_running(
            RequeueDeps(
                running_queue=running_q,
                wait_queue=wait_q,
                policy=retry_policy_from_settings(
                    self._settings,
                    retry_max_duration_ms=self._settings.external_retry_max_duration_ms,
                ),
                mark_failed=self._external_mark_failed,
            ),
            running_job_id,
        )
        if outcome == "failed":
            await self._purge_case_from_other_running(
                str(data.get("run_id") or ""),
                str(data.get("item_id") or ""),
                exclude_kind="judge",
            )
        logger.info(
            "healer.callback_requeue job_id=%s notification_job_id=%s "
            "run_id=%s item_id=%s aone_status=%s outcome=%s",
            running_job_id,
            notification_job_id,
            data.get("run_id"),
            data.get("item_id"),
            notification_status,
            outcome,
        )
        return True

    async def _heal_external_callback_timeout(
        self,
        heal_ctx: HealJobContext,
    ) -> bool:
        """pending_callback past callback_timeout → query Aone status → branch.

        - PG already terminal: skip and let the callback finish normally (no requeue, no force_fail).
        - Aone ``pending`` / ``running``: skip and wait for the callback (per-case window / TTL not exceeded).
        - Aone ``completed`` / ``failed`` and the GET body can restore the callback JSON:
          go through ``handle_callback`` (same path as the HTTP callback).
        - Aone ``completed`` but GET has no usable result: observe one round if PG is unwritten;
          requeue to wait only after a lost callback is confirmed.
        - Other states (failed with no usable error / error / no client): release the slot and requeue
          to wait, **keeping** ``first_enqueued_at``, so a held slot does not block later cases.
        - Count ≥ 3, or past the per-case window / running TTL since first entry into running: the case fails terminally.
        """
        job = heal_ctx.job
        running_q = heal_ctx.running_q
        wait_q = heal_ctx.wait_q
        data = heal_ctx.data
        running_job_id = str(job.id)
        notification_job_id = data.get("notification_job_id") or running_job_id
        now_ms = int(time.time() * 1000)
        age_ms = now_ms - int(job.timestamp or 0)
        ttl_ms = self._settings.external_running_ttl_ms
        case_overdue = self._is_case_window_exhausted(data, now_ms=now_ms)

        # 0. Callback already landed → do not steal the state transition (including TTL/1h, so a successful item is not marked failed)
        if await self._is_judge_callback_applied(data):
            logger.info(
                "healer.callback_already_applied job_id=%s notification_job_id=%s "
                "run_id=%s item_id=%s — skip",
                running_job_id,
                notification_job_id,
                data.get("run_id"),
                data.get("item_id"),
            )
            return False

        # 1. Absolute TTL / per-case window guard — past it, force_fail regardless of Aone status
        if age_ms >= ttl_ms or case_overdue:
            reason = "ttl_exceeded" if age_ms >= ttl_ms else "case_window_exceeded"
            logger.warning(
                "healer.callback_timeout_%s job_id=%s "
                "notification_job_id=%s age_ms=%d ttl_ms=%d — force_fail",
                reason,
                running_job_id,
                notification_job_id,
                age_ms,
                ttl_ms,
            )
            return await self._force_fail_callback_timeout_case(
                job,
                running_q=running_q,
                data=data,
            )

        # 2. Query Aone job status
        notification_status = "unknown"
        external_resp: object | None = None
        if self._external_judge_client is not None:
            try:
                external_resp = await self._external_judge_client.query_job_status(
                    notification_job_id,
                )
                notification_status = (
                    getattr(external_resp, "status", None)
                    or getattr(external_resp, "error", None)
                    or "unknown"
                )
                logger.info(
                    "healer.callback_timeout_aone_query job_id=%s notification_job_id=%s aone_status=%s",
                    running_job_id,
                    notification_job_id,
                    notification_status,
                )
            except Exception:
                logger.warning(
                    "healer.callback_timeout_aone_query_failed job_id=%s notification_job_id=%s",
                    running_job_id,
                    notification_job_id,
                    exc_info=True,
                )
                notification_status = "error"
        else:
            logger.info(
                "healer.callback_timeout_no_client job_id=%s "
                "notification_job_id=%s (no client, force_fail directly)",
                running_job_id,
                notification_job_id,
            )
            notification_status = "no_client"

        # 3. Aone pending/running — the task is not finished; keep waiting for the callback
        if _is_provider_job_in_flight(notification_status):
            logger.info(
                "healer.callback_timeout_skip job_id=%s "
                "aone_status=%s age_ms=%d ttl_ms=%d — "
                "waiting for delayed callback",
                running_job_id,
                notification_status,
                age_ms,
                ttl_ms,
            )
            return False

        # 3.5 GET body can restore the callback → same path as the HTTP callback (CAS is idempotent)
        if await self._replay_callback_from_job_status(
            external_resp,
            kind="judge",
            ctx=CallbackReplayContext(
                job=job,
                running_q=running_q,
                data=data,
                platform_job_id=str(notification_job_id),
            ),
        ):
            return True

        # 4. completed: requeue only for a lost callback; skip if in flight / already landed
        if str(notification_status).lower() == "completed":
            applied = await self._is_judge_callback_applied(data)
            verdict = await self._completed_callback_verdict(
                job,
                data,
                CallbackVerdictSpec(
                    applied=applied,
                    grace_ms=self._settings.external_callback_timeout_ms,
                    now_ms=now_ms,
                    log_prefix="healer.callback",
                ),
            )
            if verdict == "skip":
                return False

        # 5. Lost callback / failed / error / … → release the slot and requeue to wait
        return await self._requeue_callback_timeout_case(
            job,
            ctx=CallbackRequeueContext(
                running_q=running_q,
                wait_q=wait_q,
                data=data,
                notification_job_id=str(notification_job_id),
                notification_status=str(notification_status),
            ),
        )

    async def _heal_external_inflight(
        self,
        heal_ctx: HealJobContext,
    ) -> bool:
        job = heal_ctx.job
        running_q = heal_ctx.running_q
        wait_q = heal_ctx.wait_q
        data = heal_ctx.data
        run_id = data.get("run_id")
        item_id = data.get("item_id")
        if not run_id or not item_id:
            return False
        case_job_id = f"{run_id}:{item_id}"
        wait_job = await get_queue_job(wait_q, case_job_id)
        if wait_job is not None:
            try:
                state = await wait_job.getState()
            except Exception:
                state = "unknown"
            if state == "active":
                return False
            return await self._clean_inflight_guarded(
                InflightCleanGuard(
                    job=job,
                    data=data,
                    race_tag="external",
                    case_job_id=case_job_id,
                    wait_state=state,
                ),
            )

        if self._external_mark_failed is None:
            # The pure-cleanup path uses the same guard; the loser returns False (does not release the slot).
            return await self._clean_inflight_guarded(
                InflightCleanGuard(
                    job=job,
                    data=data,
                    race_tag="external",
                    case_job_id=case_job_id,
                    wait_state="absent",
                ),
            )
        await self._release_external_slots(data)
        outcome = await requeue_or_fail_from_running(
            RequeueDeps(
                running_queue=running_q,
                wait_queue=wait_q,
                policy=retry_policy_from_settings(
                    self._settings,
                    retry_max_duration_ms=self._settings.external_retry_max_duration_ms,
                ),
                mark_failed=self._external_mark_failed,
            ),
            case_job_id,
        )
        if outcome == "failed":
            await self._purge_case_from_other_running(
                str(run_id),
                str(item_id),
                exclude_kind="judge",
            )
        logger.info(
            "healer.external_inflight_requeue job_id=%s outcome=%s",
            case_job_id,
            outcome,
        )
        return True

    # ── attribution-queue heal (§7.5 / §7.7) ─────────────────────

    async def _clean_inflight_guarded(
        self,
        guard: InflightCleanGuard,
    ) -> bool:
        """Inflight remove-first race cleanup (same guard pattern as the external_runner callback).

        remove is atomic (Lua): when this races the worker callback cleaning the same job, only one delete wins;
        release is not idempotent (DECR), so the loser returns False and does not release the slot, avoiding a
        double release that would decrement another run's slot count. ``race_tag`` selects the judge (``external``)
        or attribution (``attr``) slot release and log name.
        """
        job = guard.job
        data = guard.data
        race_tag = guard.race_tag
        case_job_id = guard.case_job_id
        wait_state = guard.wait_state
        try:
            await job.remove()
        except Exception:
            logger.info(
                "healer.%s_inflight_remove_race job_id=%s wait_state=%s",
                race_tag,
                case_job_id,
                wait_state,
            )
            return False
        if race_tag == "attr":
            await self._release_attribution_slots(data)
            event = "healer.attribution_inflight_cleaned"
        else:
            await self._release_external_slots(data)
            event = "healer.external_inflight_cleaned"
        logger.info("%s job_id=%s wait_state=%s", event, case_job_id, wait_state)
        return True

    async def _heal_attribution(self) -> tuple[int, int]:
        """Return ``(callback_timeout_handled, inflight_cleaned)``.

        Same pattern as ``_heal_external``, but it operates on the ``ATTRIBUTION_RUNNING`` queue.
        """
        running_q = self._qm.get_queue(ATTRIBUTION_RUNNING)
        wait_q = self._qm.get_queue(ATTRIBUTION_WAIT)
        callback_timeout_ms = self._settings.attr_callback_timeout_ms
        grace_ms = self._settings.external_wait_lock_duration_ms
        timeout_n = 0
        clean_n = 0

        try:
            jobs = await list_queue_jobs(running_q, _RUNNING_JOB_TYPES)
        except Exception:
            logger.warning(
                "healer.list_attribution_running_failed",
                exc_info=True,
            )
            return 0, 0

        now_ms = int(time.time() * 1000)
        for job in jobs:
            if job is None:
                continue
            data = dict(job.data or {})
            age_ms = now_ms - int(job.timestamp or 0)
            name = job.name or ""
            try:
                if name == "pending_callback":
                    overdue = self._is_case_window_exhausted(
                        data,
                        now_ms=now_ms,
                    )
                    if age_ms < callback_timeout_ms and not overdue:
                        continue
                    if await self._heal_attribution_callback_timeout(
                        HealJobContext(
                            job=job,
                            running_q=running_q,
                            wait_q=wait_q,
                            data=data,
                        ),
                    ):
                        timeout_n += 1
                elif name == "inflight":
                    overdue = self._is_case_window_exhausted(
                        data,
                        now_ms=now_ms,
                    )
                    if age_ms < grace_ms and not overdue:
                        continue
                    if await self._heal_attribution_inflight(
                        HealJobContext(
                            job=job,
                            running_q=running_q,
                            wait_q=wait_q,
                            data=data,
                        ),
                    ):
                        clean_n += 1
            except Exception:
                logger.warning(
                    "healer.attribution_job_failed job_id=%s name=%s",
                    job.id,
                    name,
                    exc_info=True,
                )
        return timeout_n, clean_n

    async def _release_attribution_slots(self, data: dict[str, Any]) -> None:
        """Release the attribution slot (kind is fixed to ``attr``).

        Multi-node safe: call only after the caller of this method has removed the job,
        so nodes do not release twice.
        """
        slot_key = data.get("slot_key")
        if slot_key:
            await release_slot(str(slot_key))
        await release_cluster_slot("attr")

    async def _purge_case_from_other_running(
        self,
        run_id: str,
        item_id: str,
        *,
        exclude_kind: str,
    ) -> int:
        """Defensive cross-queue cleanup: on terminal fail, clear the same case's leftover job + slot on the other running queue.

        The normal flow is two serial phases (attribution starts only after judge finishes), with no overlap.
        An edge case (lost callback + healer force-fail) can leave a pending_callback on the other queue,
        leaking a cluster slot → cluster_slot_busy blocks later work.

        Idempotent: skip if the job is gone; DECR of an already-released slot down to 0 has no extra effect.
        """
        pairs = (
            (EXTERNAL_RUNNING, "judge"),
            (ATTRIBUTION_RUNNING, "attr"),
        )
        removed = 0
        for queue_name, default_kind in pairs:
            if default_kind == exclude_kind:
                continue
            q = self._qm.get_queue(queue_name)
            try:
                jobs = await list_queue_jobs(q, _RUNNING_JOB_TYPES)
            except Exception:
                logger.warning(
                    "healer.purge_case_list_failed queue=%s run_id=%s",
                    queue_name,
                    run_id,
                    exc_info=True,
                )
                continue
            for j in jobs:
                if j is None:
                    continue
                jdata = dict(j.data or {})
                if str(jdata.get("run_id") or "") != run_id:
                    continue
                if str(jdata.get("item_id") or "") != item_id:
                    continue
                if default_kind == "judge":
                    await self._release_external_slots(jdata)
                else:
                    await self._release_attribution_slots(jdata)
                try:
                    await j.remove()
                    removed += 1
                    logger.info(
                        "healer.purge_case_cleaned queue=%s job_id=%s run_id=%s item_id=%s kind=%s",
                        queue_name,
                        j.id,
                        run_id,
                        item_id,
                        default_kind,
                    )
                except Exception:
                    logger.warning(
                        "healer.purge_case_remove_failed job_id=%s",
                        j.id,
                        exc_info=True,
                    )
        return removed

    async def _attribution_inflight_terminal_fail(
        self,
        run_id: str,
        item_id: str,
        *,
        reason: str,
    ) -> None:
        """Inflight terminal fail: idempotent count + finalize, so the run does not stick on running."""
        if self._benchmark_run_service is None or not run_id or not item_id:
            return
        try:
            await self._benchmark_run_service.increment_attribution_received_once(
                run_id,
                item_id,
            )
        except Exception:
            logger.warning(
                "healer.attribution_inflight_increment_failed run_id=%s item=%s reason=%s",
                run_id,
                item_id,
                reason,
                exc_info=True,
            )
        try:
            await try_finalize_external(run_id, self._benchmark_run_service)
        except Exception:
            logger.warning(
                "healer.attribution_inflight_finalize_failed run_id=%s item=%s reason=%s",
                run_id,
                item_id,
                reason,
                exc_info=True,
            )

    async def _heal_attribution_callback_timeout(
        self,
        heal_ctx: HealJobContext,
    ) -> bool:
        """pending_callback past callback_timeout → query provider status.

        Same as judge/Aone: skip if PG has landed; wait on ``running``;
        ``completed`` requeues to wait only after a lost callback is confirmed (keep ``first_enqueued_at``).
        Past the per-case window / TTL and not yet landed → fail terminally and increment received.
        """
        job = heal_ctx.job
        running_q = heal_ctx.running_q
        wait_q = heal_ctx.wait_q
        data = heal_ctx.data
        running_job_id = str(job.id)
        attribution_job_id = data.get("attribution_job_id") or running_job_id
        now_ms = int(time.time() * 1000)
        age_ms = now_ms - int(job.timestamp or 0)
        ttl_ms = self._settings.external_running_ttl_ms
        case_overdue = self._is_case_window_exhausted(data, now_ms=now_ms)

        if await self._is_attr_callback_applied(data):
            logger.info(
                "healer.attr_callback_already_applied job_id=%s run_id=%s item_id=%s — skip",
                running_job_id,
                data.get("run_id"),
                data.get("item_id"),
            )
            return False

        # 0. Absolute TTL / per-case window guard
        if age_ms >= ttl_ms or case_overdue:
            reason = "ttl_exceeded" if age_ms >= ttl_ms else "case_window_exceeded"
            logger.warning(
                "healer.attr_callback_timeout_%s job_id=%s age_ms=%d ttl_ms=%d — force_fail",
                reason,
                running_job_id,
                age_ms,
                ttl_ms,
            )
            return await self._fail_attribution_pending(
                job,
                running_q=running_q,
                data=data,
            )

        # 1. Query provider job status
        provider_status = "unknown"
        provider_resp: object | None = None
        if self._attribution_provider_registry is not None:
            try:
                provider = self._attribution_provider_registry.resolve_for_job(data)
                if provider is not None:
                    provider_resp = await provider.query_job_status(
                        attribution_job_id,
                    )
                    provider_status = (
                        getattr(provider_resp, "status", None)
                        or getattr(provider_resp, "error", None)
                        or "unknown"
                    )
            except Exception:
                logger.warning(
                    "healer.attr_callback_timeout_provider_query_failed job_id=%s",
                    attribution_job_id,
                    exc_info=True,
                )
                provider_status = "error"

        logger.info(
            "healer.attr_callback_timeout job_id=%s provider_job_id=%s "
            "provider_status=%s age_ms=%d",
            running_job_id,
            attribution_job_id,
            provider_status,
            age_ms,
        )

        if _is_provider_job_in_flight(provider_status):
            logger.info(
                "healer.attr_callback_timeout_skip job_id=%s "
                "provider_status=%s — waiting for delayed callback",
                running_job_id,
                provider_status,
            )
            return False

        if await self._replay_callback_from_job_status(
            provider_resp,
            kind="attribution",
            ctx=CallbackReplayContext(
                job=job,
                running_q=running_q,
                data=data,
                platform_job_id=str(attribution_job_id),
            ),
        ):
            return True

        if str(provider_status).lower() == "completed":
            applied = await self._is_attr_callback_applied(data)
            verdict = await self._completed_callback_verdict(
                job,
                data,
                CallbackVerdictSpec(
                    applied=applied,
                    grace_ms=self._settings.attr_callback_timeout_ms,
                    now_ms=now_ms,
                    log_prefix="healer.attr_callback",
                ),
            )
            if verdict == "skip":
                return False

        # Lost callback / failed / error → release the slot and requeue to wait
        await self._release_attribution_slots(data)

        async def _mark_failed(
            d: dict[str, Any],
            *,
            retry_count: int = 0,
            reason: str = "",
            **_kw: object,
        ) -> None:
            await self._attribution_inflight_terminal_fail(
                str(d.get("run_id") or ""),
                str(d.get("item_id") or ""),
                reason=reason or "retry_deadline",
            )

        outcome = await requeue_or_fail_from_running(
            RequeueDeps(
                running_queue=running_q,
                wait_queue=wait_q,
                policy=retry_policy_from_settings(
                    self._settings,
                    retry_max_duration_ms=self._settings.external_retry_max_duration_ms,
                ),
                mark_failed=_mark_failed,
            ),
            running_job_id,
            RequeueOpts(wait_job_name="attr_case"),
        )
        if outcome == "failed":
            await self._purge_case_from_other_running(
                str(data.get("run_id") or ""),
                str(data.get("item_id") or ""),
                exclude_kind="attr",
            )
        logger.info(
            "healer.attr_callback_requeue job_id=%s provider_status=%s outcome=%s",
            running_job_id,
            provider_status,
            outcome,
        )
        return True

    async def _fail_attribution_pending(
        self,
        job: Job,
        *,
        running_q: Queue,
        data: dict[str, Any],
    ) -> bool:
        """TTL / per-case window terminal fail: release slot → purge → force_fail (aligned with judge).

        Aligned with the TTL path of ``_heal_external_callback_timeout``:
        release the slot + purge first, then ``force_fail_from_running``
        (internally get_job → remove → mark_failed).
        """
        running_job_id = str(job.id)
        run_id = str(data.get("run_id") or "")
        item_id = str(data.get("item_id") or "")

        # 1. Release the slot (before job.remove, so a release exception does not drop the job)
        await self._release_attribution_slots(data)
        await self._purge_case_from_other_running(
            run_id,
            item_id,
            exclude_kind="attr",
        )

        # 2. force_fail_from_running (internally get_job → remove → mark_failed)
        async def _mark_failed(
            d: dict[str, Any],
            *,
            retry_count: int = 0,
            reason: str = "",
            **_kw: object,
        ) -> None:
            await self._attribution_inflight_terminal_fail(
                str(d.get("run_id") or run_id),
                str(d.get("item_id") or item_id),
                reason=reason or "ttl_exceeded",
            )

        await force_fail_from_running(
            running_queue=running_q,
            running_job_id=running_job_id,
            mark_failed=_mark_failed,
        )
        return True

    async def _heal_attribution_inflight(
        self,
        heal_ctx: HealJobContext,
    ) -> bool:
        """Attribution inflight orphan cleanup.

        Multi-node safe (v2.1 fix, same race guard as the external_runner callback):
        - wait has the same job id and it is not active → **remove inflight first (atomic), and release
          the slot only on success**: racing the worker callback on the same job, only one delete wins; the
          loser does not release, so a double release does not decrement another run's slot count
        - wait has no job → release the slot then ``requeue_or_fail_from_running`` (it checks that the
          running job exists and removes/requeues it; with no running registration it fails terminally,
          so there is no invented retry)

        Relies on the ``tick()`` distributed lock (§7.7) so only one healer handles it at a time.
        """

        job = heal_ctx.job
        running_q = heal_ctx.running_q
        wait_q = heal_ctx.wait_q
        data = heal_ctx.data
        run_id = data.get("run_id")
        item_id = data.get("item_id")
        if not run_id or not item_id:
            return False
        case_job_id = f"{run_id}:{item_id}"
        wait_job = await get_queue_job(wait_q, case_job_id)

        if wait_job is not None:
            try:
                state = await wait_job.getState()
            except Exception:
                state = "unknown"
            if state == "active":
                return False
            return await self._clean_inflight_guarded(
                InflightCleanGuard(
                    job=job,
                    data=data,
                    race_tag="attr",
                    case_job_id=case_job_id,
                    wait_state=state,
                ),
            )

        # wait has no job → release the slot → requeue_or_fail (aligned with judge
        # _heal_external_inflight)
        await self._release_attribution_slots(data)

        async def _mark_failed(
            d: dict[str, Any],
            *,
            retry_count: int = 0,
            reason: str = "",
            **_kw: object,
        ) -> None:
            await self._attribution_inflight_terminal_fail(
                str(d.get("run_id") or run_id),
                str(d.get("item_id") or item_id),
                reason=reason or "retry_deadline",
            )

        outcome = await requeue_or_fail_from_running(
            RequeueDeps(
                running_queue=running_q,
                wait_queue=wait_q,
                policy=retry_policy_from_settings(
                    self._settings,
                    retry_max_duration_ms=self._settings.external_retry_max_duration_ms,
                ),
                mark_failed=_mark_failed,
            ),
            case_job_id,
            RequeueOpts(wait_job_name="attr_case"),
        )
        if outcome == "failed":
            await self._purge_case_from_other_running(
                str(run_id),
                str(item_id),
                exclude_kind="attr",
            )
        logger.info(
            "healer.attribution_inflight_requeue job_id=%s outcome=%s",
            case_job_id,
            outcome,
        )
        return True

    async def _heal_attr_slot_counters(self) -> int:
        """Attribution slot-delta sweep (aligned with §7.5 / the §16 risk table).

        v2.1 fix (full logic aligned with the Judge healer ``_heal_slot_counters``):
        1. Count real slot holders among pending_callback/inflight on ``ATTRIBUTION_RUNNING``
        2. **SCAN Redis orphan keys**
        3. **Clamp the cluster key** ``{prefix}:inflight:cluster:attr``
        Refill when low / decrease when high / renew when equal.
        """
        running_q = self._qm.get_queue(ATTRIBUTION_RUNNING)
        clamped = 0

        try:
            jobs = await list_queue_jobs(running_q, _RUNNING_JOB_TYPES)
        except Exception:
            logger.warning(
                "healer.list_attr_running_failed",
                exc_info=True,
            )
            return 0

        expected_by_key: Counter[str] = Counter()
        cluster_expected = 0
        for job in jobs:
            if job is None or (job.name or "") not in (
                "pending_callback",
                "inflight",
            ):
                continue
            data = dict(job.data or {})
            cluster_expected += 1
            slot_key = data.get("slot_key")
            if slot_key:
                expected_by_key[str(slot_key)] += 1
                continue
            # fallback: rebuild slot_key from submitter_node_id + run_id
            node = data.get("submitter_node_id", "")
            run_id = data.get("run_id", "")
            if node and run_id:
                reconstructed = build_slot_key(
                    self._qm.prefix,
                    SlotKeySpec(
                        kind="attr",
                        node=str(node),
                        run_id=str(run_id),
                    ),
                )
                expected_by_key[reconstructed] += 1

        ttl_s = max(1, int(self._settings.external_running_ttl_ms) // 1000)
        try:
            cluster_key = build_cluster_slot_key("attr")
            clamped += await clamp_slot_count(
                cluster_key,
                expected=cluster_expected,
                ttl_s=ttl_s,
            )
            pattern = f"{self._qm.prefix}:inflight:attr:*"
            scanned = set(await scan_slot_keys(pattern))
            for key in scanned | set(expected_by_key):
                clamped += await clamp_slot_count(
                    key,
                    expected=expected_by_key.get(key, 0),
                    ttl_s=ttl_s,
                )
        except Exception:
            logger.warning(
                "healer.attr_slot_counter_heal_failed",
                exc_info=True,
            )
            return clamped

        if clamped > 0:
            logger.warning(
                "healer.attr_slot_excess_released n=%d cluster_expected=%d node_keys=%d",
                clamped,
                cluster_expected,
                len(expected_by_key),
            )
        return clamped

    async def _try_acquire_heal_lock(
        self,
        key: str,
        *,
        ttl_s: int,
    ) -> bool:
        """Redis distributed lock: only one healer instance handles the attribution queue at a time.

        Uses the atomic SET NX EX; failure to acquire means another node's healer is already handling it.
        """
        r = get_redis()
        result = await r.set(key, "1", nx=True, ex=ttl_s)
        return bool(result)

    async def _release_heal_lock(self, key: str) -> None:
        """Release the healer distributed lock."""
        r = get_redis()
        await r.delete(key)

    async def _heal_external_wait_stuck(self) -> int:
        """Detect jobs stuck active too long on EXTERNAL_WAIT.

        The BullMQ stalled check moves an active job whose lock expired back to wait every 30s,
        but if the Worker process is alive and the event loop is blocked (for example a hung HTTP request),
        lock renewal may still succeed, so the job stays active for a long time.
        The per-node slot it holds cannot be released, and later jobs are all slot_busy.

        This method is the fallback: a job still active past ``external_wait_lock_duration_ms`` × 2
        is treated as stuck; the healer removes it and enqueues it again.
        """
        wait_q = self._qm.get_queue(EXTERNAL_WAIT)
        lock_duration_ms = int(
            getattr(self._settings, "external_wait_lock_duration_ms", 300_000),
        )
        stuck_threshold_ms = lock_duration_ms * 2
        stuck_n = 0

        try:
            jobs = await list_queue_jobs(wait_q, ["active"])
        except Exception:
            logger.warning(
                "healer.list_external_wait_active_failed",
                exc_info=True,
            )
            return 0

        now_ms = int(time.time() * 1000)
        for job in jobs:
            if job is None:
                continue
            age_ms = now_ms - int(job.timestamp or 0)
            if age_ms < stuck_threshold_ms:
                continue
            data = dict(job.data or {})
            run_id = data.get("run_id", "")
            item_id = data.get("item_id", "")
            case_job_id = f"{run_id}:{item_id}" if run_id and item_id else str(job.id)
            try:
                await job.remove()
                logger.warning(
                    "healer.external_wait_stuck_removed job_id=%s run_id=%s item_id=%s age_ms=%d",
                    case_job_id,
                    run_id,
                    item_id,
                    age_ms,
                )
                # Enqueue again (keep retry_count / first_enqueued_at)
                stamp_first_running_at(data, now_ms=now_ms)
                await wait_q.add(
                    "case",
                    data,
                    {
                        "jobId": case_job_id,
                        "attempts": 1,
                        "delay": 0,
                        "removeOnComplete": True,
                        "removeOnFail": True,
                    },
                )
                logger.info(
                    "healer.external_wait_stuck_requeued job_id=%s",
                    case_job_id,
                )
                stuck_n += 1
            except Exception:
                logger.warning(
                    "healer.external_wait_stuck_failed job_id=%s",
                    case_job_id,
                    exc_info=True,
                )
        return stuck_n

    async def _heal_attribution_wait_stuck(self) -> int:
        """Detect jobs stuck active too long on ATTRIBUTION_WAIT.

        The BullMQ stalled check moves an active job whose lock expired back to wait every 30s,
        but if the Worker process is alive and the event loop is blocked, lock renewal may still succeed,
        so the job stays active for a long time. This method is the fallback:
        a job still active past ``external_wait_lock_duration_ms`` × 2
        is treated as stuck; the healer removes it and enqueues it again.

        Complements the BullMQ stalled check; it does not replace that behavior.
        """
        wait_q = self._qm.get_queue(ATTRIBUTION_WAIT)
        lock_duration_ms = int(
            getattr(self._settings, "external_wait_lock_duration_ms", 300_000),
        )
        stuck_threshold_ms = lock_duration_ms * 2
        stuck_n = 0

        try:
            jobs = await list_queue_jobs(wait_q, ["active"])
        except Exception:
            logger.warning(
                "healer.list_attribution_wait_active_failed",
                exc_info=True,
            )
            return 0

        now_ms = int(time.time() * 1000)
        for job in jobs:
            if job is None:
                continue
            age_ms = now_ms - int(job.timestamp or 0)
            if age_ms < stuck_threshold_ms:
                continue
            data = dict(job.data or {})
            run_id = data.get("run_id", "")
            item_id = data.get("item_id", "")
            case_job_id = f"{run_id}:{item_id}" if run_id and item_id else str(job.id)
            try:
                await job.remove()
                logger.warning(
                    "healer.attribution_wait_stuck_removed job_id=%s "
                    "run_id=%s item_id=%s age_ms=%d",
                    case_job_id,
                    run_id,
                    item_id,
                    age_ms,
                )
                # Enqueue again (keep retry_count / first_enqueued_at)
                stamp_first_running_at(data, now_ms=now_ms)
                await wait_q.add(
                    "attr_case",
                    data,
                    {
                        "jobId": case_job_id,
                        "attempts": 3,
                        "delay": 0,
                        "removeOnComplete": True,
                        "removeOnFail": False,
                    },
                )
                logger.info(
                    "healer.attribution_wait_stuck_requeued job_id=%s",
                    case_job_id,
                )
                stuck_n += 1
            except Exception:
                logger.warning(
                    "healer.attribution_wait_stuck_failed job_id=%s",
                    case_job_id,
                    exc_info=True,
                )
        return stuck_n


__all__ = [
    "RunningHealer",
    "collect_active_run_ids",
    "collect_running_queue_run_ids",
    "recover_stale_with_queue_guard",
]
