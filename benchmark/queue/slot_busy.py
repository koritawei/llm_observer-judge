"""slot_busy — delay + jitter + probe rate limit when the slot is busy (design doc §6.5 D15).

Called when ``acquire_slot`` returns False (the slot is full), via ``delay_for_slot_busy``:
- compute ``delay = base + random(0, base)`` (jitter)
- move the job into the BullMQ delayed state
- **not a failure**: do not move running→wait, and **do not increment retry_count**

Probe rate limit: at most ``max_slot_probes_per_run`` probes per run at once,
to avoid a thundering herd (many workers probing together and cascading).

bullmq-py (≥2.25) notes:
- the Worker calls ``processor(job, token)``; ``moveToDelayed`` must carry the lock token
- ``Job`` has no ``moveToDelayed``; use ``job.scripts.moveToDelayed``
- after a successful delay, ``raise WaitingChildrenError``, or the Worker will
  ``moveToCompleted`` and treat the job as a success (with removeOnComplete it is deleted)
"""

import random
import time
from dataclasses import dataclass

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Job, WaitingChildrenError

from benchmark.queue.inflight_slot import acquire_slot, release_slot

logger = get_logger("agent_eval.eval_queue.slot_busy")


@dataclass(frozen=True)
class SlotProbePolicy:
    """Probe rate-limit policy (the parameter object for ``delay_for_slot_busy``).

    Pass ``probe=None`` to disable the limit.
    """

    probe_key: str
    max_probes: int = 2


@dataclass(frozen=True)
class SlotBusyDelay:
    """Arguments for ``delay_for_slot_busy`` (FUNC-PARAMS-001 parameter folding).

    ``token`` is the Worker job lock token (required); ``probe=None`` disables the limit.
    """

    run_id: str
    base_delay_ms: int
    token: str
    probe: SlotProbePolicy | None = None


async def delay_for_slot_busy(job: Job, delay: SlotBusyDelay) -> None:
    """Slot busy: delay the job and retry without incrementing ``retry_count`` (D15).

    On success raise ``WaitingChildrenError`` (the Worker swallows it; no complete / fail).

    Args:
        job: the BullMQ wait-job instance.
        delay: delay parameter bundle (run_id / base_delay_ms / token / probe).
    """
    run_id = delay.run_id
    base_delay_ms = delay.base_delay_ms
    token = delay.token
    probe = delay.probe
    jitter = random.randint(0, base_delay_ms)
    delay_ms = base_delay_ms + jitter

    # Probe rate limit: past the cap, increase the delay (exponential backoff)
    held_probe = False
    if probe is not None:
        held_probe = await acquire_slot(
            probe.probe_key,
            limit=probe.max_probes,
            ttl_s=30,
        )
        if not held_probe:
            # Probe count is full: delay longer (×3) and wait for an existing probe to release
            delay_ms = delay_ms * 3
            logger.debug(
                "slot_busy.probe_limited run_id=%s delay=%dms",
                run_id,
                delay_ms,
            )

    timestamp = int(time.time() * 1000) + delay_ms
    try:
        await job.scripts.moveToDelayed(job.id, timestamp, delay_ms, token)
        logger.info(
            "slot_busy.delayed run_id=%s job=%s delay=%dms",
            run_id,
            job.id,
            delay_ms,
        )
    except Exception:
        logger.warning(
            "slot_busy.move_failed run_id=%s job=%s delay=%dms",
            run_id,
            job.id,
            delay_ms,
            exc_info=True,
        )
        raise
    finally:
        if held_probe and probe is not None:
            await release_slot(probe.probe_key)

    # Stop the Worker from moveToCompleted after the processor returns (Node DelayedError semantics)
    raise WaitingChildrenError()


__all__ = ["SlotBusyDelay", "SlotProbePolicy", "delay_for_slot_busy"]
