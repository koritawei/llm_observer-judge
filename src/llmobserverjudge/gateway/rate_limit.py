"""Rate limiter backed by the `limits` library (token-window / moving window)."""

from __future__ import annotations

from limits import RateLimitItemPerSecond
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter


class RateLimiter:
    """Per-tenant limiter using limits.MovingWindowRateLimiter. fail-closed when over quota."""

    def __init__(
        self,
        *,
        default_rate_per_sec: float = 100.0,
        default_burst: float = 200.0,
        telemetry_rate_per_sec: float = 1 / 30,
        storage: MemoryStorage | None = None,
    ) -> None:
        self.default_rate = default_rate_per_sec
        self.default_burst = default_burst
        self.telemetry_rate = telemetry_rate_per_sec
        self._storage = storage or MemoryStorage()
        self._limiter = MovingWindowRateLimiter(self._storage)
        # Approximate former token-bucket burst by allowing burst hits in a 1s window
        # when rate is high; otherwise use ceil(rate) per second.
        ingest_per_sec = max(1, int(default_burst if default_burst > 0 else default_rate_per_sec))
        self._ingest_item = RateLimitItemPerSecond(ingest_per_sec, 1)
        # telemetry: 1 request per ~30s → 1 per 30-second window
        tel_window = max(1, int(round(1.0 / telemetry_rate_per_sec))) if telemetry_rate_per_sec > 0 else 30
        self._telemetry_item = RateLimitItemPerSecond(1, tel_window)

    def allow_ingest(self, tenant_id: str, cost: float = 1.0) -> bool:
        hits = max(1, int(cost))
        # rate==0 / burst==0 → always deny (tests set 0.0)
        if self.default_rate <= 0 or self.default_burst <= 0:
            return False
        return self._limiter.hit(self._ingest_item, "ingest", tenant_id, cost=hits)

    def allow_telemetry(self, tenant_id: str) -> bool:
        if self.telemetry_rate <= 0:
            return False
        return self._limiter.hit(self._telemetry_item, "tel", tenant_id)


# Back-compat alias used by redis_rate_limit imports
class TokenBucket:
    """Deprecated stand-in — prefer RateLimiter."""

    def __init__(self, rate_per_sec: float, burst: float) -> None:
        self._inner = RateLimiter(
            default_rate_per_sec=rate_per_sec,
            default_burst=burst,
            telemetry_rate_per_sec=rate_per_sec,
        )

    def allow(self, cost: float = 1.0) -> bool:
        return self._inner.allow_ingest("local", cost=cost)
