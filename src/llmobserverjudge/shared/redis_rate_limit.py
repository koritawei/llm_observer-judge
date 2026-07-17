"""Redis-backed rate limiter via `limits` RedisStorage (shared across Gateway replicas)."""

from __future__ import annotations

import logging
from typing import Any

from limits import RateLimitItemPerSecond
from limits.strategies import MovingWindowRateLimiter

from llmobserverjudge.gateway.rate_limit import RateLimiter

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """Per-tenant limiter using limits + Redis storage; fail-closed when redis mode."""

    def __init__(
        self,
        redis_url: str,
        *,
        default_rate_per_sec: float = 100.0,
        default_burst: float = 200.0,
        telemetry_rate_per_sec: float = 1 / 30,
        fail_closed: bool | None = None,
    ) -> None:
        import os

        from limits.storage import storage_from_string

        from llmobserverjudge.shared.deployment import resolve_region

        self.default_rate = default_rate_per_sec
        self.default_burst = default_burst
        self.telemetry_rate = telemetry_rate_per_sec
        self._scope = resolve_region() or "global"
        prefer = (os.environ.get("LLMJUDGE_RATE_LIMIT") or "").lower().strip()
        env_fc = (os.environ.get("LLMJUDGE_RATE_LIMIT_FAIL_CLOSED") or "").lower()
        if fail_closed is None:
            self._fail_closed = prefer == "redis" or env_fc in ("1", "true", "yes")
        else:
            self._fail_closed = fail_closed
        self._local = RateLimiter(
            default_rate_per_sec=default_rate_per_sec,
            default_burst=default_burst,
            telemetry_rate_per_sec=telemetry_rate_per_sec,
        )
        try:
            self._storage = storage_from_string(redis_url)
            self._limiter = MovingWindowRateLimiter(self._storage)
            ingest_per_sec = max(
                1, int(default_burst if default_burst > 0 else default_rate_per_sec)
            )
            self._ingest_item = RateLimitItemPerSecond(ingest_per_sec, 1)
            tel_window = (
                max(1, int(round(1.0 / telemetry_rate_per_sec)))
                if telemetry_rate_per_sec > 0
                else 30
            )
            self._telemetry_item = RateLimitItemPerSecond(1, tel_window)
            self._ok = True
        except Exception:
            logger.warning("limits_redis_storage_unavailable", exc_info=True)
            self._ok = False

    def allow_ingest(self, tenant_id: str, cost: float = 1.0) -> bool:
        if self.default_rate <= 0 or self.default_burst <= 0:
            return False
        if not self._ok:
            if self._fail_closed:
                logger.warning("rate_limit_denied redis_unavailable tenant=%s", tenant_id)
                return False
            return self._local.allow_ingest(tenant_id, cost=cost)
        try:
            hits = max(1, int(cost))
            return bool(
                self._limiter.hit(
                    self._ingest_item, f"{self._scope}:ingest", tenant_id, cost=hits
                )
            )
        except Exception:
            logger.warning("redis_rate_limit_error tenant=%s", tenant_id, exc_info=True)
            if self._fail_closed:
                return False
            return self._local.allow_ingest(tenant_id, cost=cost)

    def allow_telemetry(self, tenant_id: str) -> bool:
        if self.telemetry_rate <= 0:
            return False
        if not self._ok:
            if self._fail_closed:
                return False
            return self._local.allow_telemetry(tenant_id)
        try:
            return bool(self._limiter.hit(self._telemetry_item, f"{self._scope}:tel", tenant_id))
        except Exception:
            logger.warning("redis_rate_limit_error tenant=%s", tenant_id, exc_info=True)
            if self._fail_closed:
                return False
            return self._local.allow_telemetry(tenant_id)

    async def allow_ingest_async(self, tenant_id: str, cost: float = 1.0) -> bool:
        return self.allow_ingest(tenant_id, cost=cost)

    async def allow_telemetry_async(self, tenant_id: str) -> bool:
        return self.allow_telemetry(tenant_id)


def create_rate_limiter_from_env() -> RateLimiter | RedisRateLimiter:
    import os

    url = (os.environ.get("LLMJUDGE_REDIS_URL") or "").strip()
    prefer = (os.environ.get("LLMJUDGE_RATE_LIMIT") or "").lower().strip()
    if prefer in ("memory", "local"):
        return RateLimiter()
    if not url and prefer not in ("redis",):
        return RateLimiter()
    if not url:
        return RateLimiter()
    try:
        return RedisRateLimiter(url)
    except Exception:
        logger.warning("rate_limiter_redis_unavailable; using memory", exc_info=True)
        return RateLimiter()


# silence unused
_ = Any
