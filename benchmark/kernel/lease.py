"""Single-key lease sets. Capacity is membership, not a TTL counter."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import redis.asyncio as aioredis

ACQUIRE_LUA = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
for i = 1, 3 do
  if redis.call('ZCARD', key) >= limit then
    return 'busy'
  end
  local lease_id = ARGV[i + 1]
  if not lease_id or lease_id == '' then
    return 'busy'
  end
  if not redis.call('ZSCORE', key, lease_id) then
    local t = redis.call('TIME')
    local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
    redis.call('ZADD', key, now_ms, lease_id)
    return lease_id
  end
end
return 'busy'
"""

RENEW_LEADER_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  return 1
end
return 0
"""


@dataclass
class LeasePair:
    node_key: str
    node_id: str
    cluster_key: str | None = None
    cluster_id: str | None = None
    kind: str = "sync"

    def ids(self) -> list[tuple[str, str]]:
        pairs = [(self.node_key, self.node_id)]
        if self.cluster_key and self.cluster_id:
            pairs.append((self.cluster_key, self.cluster_id))
        return pairs


class LeaseManager:
    def __init__(self, redis_url: str, namespace: str):
        self.redis = aioredis.from_url(redis_url, decode_responses=True)
        self.namespace = namespace
        self._leader_token = uuid.uuid4().hex

    def node_key(self, kind: str, node_id: str, run_id: str) -> str:
        if kind == "sync":
            return f"{self.namespace}:inflight:sync:batch:{node_id}:{run_id}"
        return f"{self.namespace}:inflight:{kind}:{node_id}:{run_id}"

    def cluster_key(self, kind: str) -> str:
        return f"{self.namespace}:inflight:cluster:{kind}"

    def probe_key(self, kind: str, node_id: str, run_id: str) -> str:
        return f"{self.namespace}:inflight:probe:{kind}:{node_id}:{run_id}"

    def leader_key(self) -> str:
        return f"{self.namespace}:healer:leader"

    async def acquire(self, key: str, limit: int) -> str:
        candidates = [uuid.uuid4().hex for _ in range(3)]
        result = await self.redis.eval(ACQUIRE_LUA, 1, key, str(limit), *candidates)
        lease_id = result if isinstance(result, str) else result.decode()
        if lease_id != "busy":
            await self.redis.sadd(f"{self.namespace}:lease-index", key)
        return lease_id

    async def acquire_pair(
        self,
        *,
        kind: str,
        node_id: str,
        run_id: str,
        node_limit: int,
        cluster_limit: int,
    ) -> LeasePair | None:
        cluster_id = None
        cluster_key = None
        if cluster_limit > 0:
            cluster_key = self.cluster_key(kind)
            cluster_id = await self.acquire(cluster_key, cluster_limit)
            if cluster_id == "busy":
                return None
        node_key = self.node_key(kind, node_id, run_id)
        try:
            node_id_lease = await self.acquire(node_key, node_limit)
        except Exception:
            if cluster_key and cluster_id:
                await self.release_one(cluster_key, cluster_id)
            raise
        if node_id_lease == "busy":
            if cluster_key and cluster_id:
                await self.release_one(cluster_key, cluster_id)
            return None
        return LeasePair(
            node_key=node_key,
            node_id=node_id_lease,
            cluster_key=cluster_key,
            cluster_id=cluster_id,
            kind=kind,
        )

    async def still_held(self, lease: LeasePair) -> bool:
        for key, lease_id in lease.ids():
            if await self.redis.zscore(key, lease_id) is None:
                return False
        return True

    async def release_one(self, key: str, lease_id: str) -> bool:
        removed = await self.redis.zrem(key, lease_id)
        return int(removed) > 0

    async def release(self, lease: LeasePair) -> None:
        await self.release_one(lease.node_key, lease.node_id)
        if lease.cluster_key and lease.cluster_id:
            await self.release_one(lease.cluster_key, lease.cluster_id)

    async def note_probe(self, kind: str, node_id: str, run_id: str, ttl_s: int) -> int:
        key = self.probe_key(kind, node_id, run_id)
        count = int(await self.redis.incr(key))
        await self.redis.expire(key, ttl_s)
        return count

    async def try_lead(self, lease_ms: int) -> bool:
        ok = await self.redis.set(self.leader_key(), self._leader_token, nx=True, px=lease_ms)
        if ok:
            return True
        renewed = await self.redis.eval(
            RENEW_LEADER_LUA, 1, self.leader_key(), self._leader_token, str(lease_ms)
        )
        return int(renewed) == 1

    async def redis_now_ms(self) -> int:
        seconds, micros = await self.redis.time()
        return int(seconds) * 1000 + int(micros) // 1000

    async def members(self, key: str) -> list[tuple[str, float]]:
        rows = await self.redis.zrange(key, 0, -1, withscores=True)
        return [(member, float(score)) for member, score in rows]

    async def indexed_keys(self) -> list[str]:
        return list(await self.redis.smembers(f"{self.namespace}:lease-index"))

    async def close(self) -> None:
        await self.redis.aclose()


def now_ms() -> int:
    return int(time.time() * 1000)
