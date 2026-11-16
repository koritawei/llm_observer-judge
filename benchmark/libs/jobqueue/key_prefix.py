"""BullMQ Redis key prefix（含 Cluster hash tag）。"""


def queue_key_prefix(env_prefix: str, queue_name: str) -> str:
    """构建 BullMQ ``prefix`` 选项（含 Redis Cluster hash tag）。

    Redis Cluster 下同一 Lua 脚本（如 ``getRanges`` / ``moveToActive``）会同时
    访问 ``wait`` / ``meta`` / job hash 等多个 key；无 hash tag 时后缀不同会
    落在不同 slot，触发 ``Script attempted to access a non local key``。

    返回 ``{pre:bullmq:dataset-cases}``，实际 Redis key 形如::

        {pre:bullmq:dataset-cases}:dataset-cases:wait

    单机 Redis 上 hash tag 无副作用；预发/生产阿里云 KvStore 常按 cluster
    语义校验 slot，故**始终**启用。
    """
    return f"{{{env_prefix}:{queue_name}}}"


__all__ = ["queue_key_prefix"]
