"""In-memory token bucket (Redis-backed in Phase 3 ops)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    rate_per_sec: float
    burst: float
    tokens: float = field(init=False)
    updated_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.burst
        self.updated_at = time.monotonic()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.updated_at
        self.updated_at = now
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate_per_sec)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class RateLimiter:
    """Per-tenant limiter. fail-closed when over quota."""

    def __init__(
        self,
        *,
        default_rate_per_sec: float = 100.0,
        default_burst: float = 200.0,
        telemetry_rate_per_sec: float = 1 / 30,
    ) -> None:
        self.default_rate = default_rate_per_sec
        self.default_burst = default_burst
        self.telemetry_rate = telemetry_rate_per_sec
        self._buckets: dict[str, TokenBucket] = {}
        self._telemetry: dict[str, TokenBucket] = {}

    def _bucket(self, store: dict[str, TokenBucket], key: str, rate: float, burst: float) -> TokenBucket:
        if key not in store:
            store[key] = TokenBucket(rate_per_sec=rate, burst=burst)
        return store[key]

    def allow_ingest(self, tenant_id: str, cost: float = 1.0) -> bool:
        b = self._bucket(
            self._buckets, tenant_id, self.default_rate, self.default_burst
        )
        return b.allow(cost)

    def allow_telemetry(self, tenant_id: str) -> bool:
        # ~1 req / 30s
        b = self._bucket(
            self._telemetry, f"tel:{tenant_id}", self.telemetry_rate, 1.0
        )
        return b.allow(1.0)
