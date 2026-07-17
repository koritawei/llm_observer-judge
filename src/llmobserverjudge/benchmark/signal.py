"""Batch completion signal protocol + in-memory impl (design §11.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class BatchSignal(Protocol):
    """completed + failed + dead >= total → COMPLETE once (Redis Lua equiv)."""

    async def init_batch(self, batch_job_id: str, total: int) -> None: ...

    async def incr(self, batch_job_id: str, field: str) -> str: ...

    async def try_claim_benchmark(self, batch_job_id: str) -> bool: ...

    async def snapshot(self, batch_job_id: str) -> dict[str, int]: ...


@dataclass
class InMemoryBatchSignal:
    """Process-local counters for tests / Redis-unavailable fallback."""

    _batches: dict[str, dict[str, int]] = field(default_factory=dict)
    _signals: set[str] = field(default_factory=set)
    _benchmark_done: set[str] = field(default_factory=set)

    async def init_batch(self, batch_job_id: str, total: int) -> None:
        self._batches[batch_job_id] = {
            "total": total,
            "completed": 0,
            "failed": 0,
            "dead": 0,
        }
        self._signals.discard(batch_job_id)

    async def incr(self, batch_job_id: str, field: str) -> str:
        """field: completed | failed | dead. Returns COMPLETE | ALREADY_SIGNALED | PENDING."""
        if field not in ("completed", "failed", "dead"):
            raise ValueError(f"invalid field: {field}")
        batch = self._batches.setdefault(
            batch_job_id, {"total": 0, "completed": 0, "failed": 0, "dead": 0}
        )
        batch[field] = int(batch.get(field, 0)) + 1
        total = int(batch.get("total", 0))
        done = (
            int(batch.get("completed", 0))
            + int(batch.get("failed", 0))
            + int(batch.get("dead", 0))
        )
        if total > 0 and done >= total:
            if batch_job_id in self._signals:
                return "ALREADY_SIGNALED"
            self._signals.add(batch_job_id)
            return "COMPLETE"
        return "PENDING"

    async def try_claim_benchmark(self, batch_job_id: str) -> bool:
        """Idempotent benchmark processing (SET NX)."""
        if batch_job_id in self._benchmark_done:
            return False
        self._benchmark_done.add(batch_job_id)
        return True

    async def snapshot(self, batch_job_id: str) -> dict[str, int]:
        return dict(self._batches.get(batch_job_id, {}))
