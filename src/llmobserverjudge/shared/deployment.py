"""Deployment metadata: region prefix, TTLs, and multi-cluster tuning."""

from __future__ import annotations

import os
import re


def resolve_region() -> str | None:
    """Optional region / AZ label (e.g. ``cn-east-1``, ``us-west-2``)."""
    region = (os.environ.get("LLMJUDGE_REGION") or "").strip()
    return region or None


def redis_key_prefix() -> str:
    """Prefix for Redis keys/channels when sharing a cluster across deployments."""
    region = resolve_region()
    if not region:
        return "llmj:"
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", region)
    return f"llmj:{safe}:"


def resolve_idempotency_ttl() -> int:
    return max(60, int(os.environ.get("LLMJUDGE_IDEMPOTENCY_TTL_SECONDS", "86400")))


def resolve_processing_lock_ttl() -> int:
    return max(30, int(os.environ.get("LLMJUDGE_PROCESSING_LOCK_TTL_SECONDS", "900")))


def resolve_outbox_claim_lease_seconds() -> int:
    return max(30, int(os.environ.get("LLMJUDGE_OUTBOX_CLAIM_LEASE_SECONDS", "300")))


def resolve_mq_claim_min_idle_ms() -> int:
    return max(1000, int(os.environ.get("LLMJUDGE_MQ_CLAIM_MIN_IDLE_MS", "60000")))


def deployment_metadata(*, dispatch_version: int | None = None) -> dict[str, object]:
    meta: dict[str, object] = {"region": resolve_region()}
    if dispatch_version is not None:
        meta["dispatch_config_version"] = dispatch_version
    return meta


def uses_redis_backend() -> bool:
    for key in ("LLMJUDGE_MQ", "LLMJUDGE_EVENT_BUS", "LLMJUDGE_RATE_LIMIT"):
        if (os.environ.get(key) or "").lower().strip() == "redis":
            return True
    return bool((os.environ.get("LLMJUDGE_REDIS_URL") or "").strip())
