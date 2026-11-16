"""BullMQ Worker 续锁补丁 — 禁止把 ``extendLock`` 包进 Redis MULTI/EXEC。

bullmq 2.25 的 ``Worker.extendLocks`` 用 ``pipeline()``（默认事务）批量
``EVALSHA``。事务内任一条被拒 → ``EXECABORT``，整批锁都不续。
``extendLock`` Lua 在锁不匹配时只返回 ``0``，不会自己抛错；出现
``ResponseError`` 表示命令被拒。续锁周期为 ``lockDuration / 2``
（同步 20min 锁 → 约每 10min 一次）。redis-py 6.4.0 还会把 pipeline
错误印成字面量 ``{exception.args}``，故用 ``raise_on_error=False``
保留 Redis 原始错误。
"""

from typing import override

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Worker as BullmqWorker

logger = get_logger("agent_eval.job_queue.worker")


class Worker(BullmqWorker):
    """与 ``bullmq.Worker`` 同用法；续锁走非事务 pipeline。"""

    @override
    async def extendLocks(self) -> None:  # noqa: N802 — 覆盖上游 camelCase 方法名
        active = list(self.jobs)
        if not active:
            return
        try:
            pipe = self.client.pipeline(transaction=False)
            for job, token in active:
                await self.scripts.extendLock(
                    job.id,
                    token,
                    self.opts.get("lockDuration"),
                    pipe,
                )
            results = await pipe.execute(raise_on_error=False)
        except Exception:
            # 定时器回调：任何异常都必须吞掉，否则后续续锁停摆（对齐上游）。
            logger.exception("job_queue.lock_renew_failed")
            return

        if len(results) != len(active):
            logger.error(
                "job_queue.lock_renew_failed expected=%d got=%d",
                len(active),
                len(results),
            )
        for (job, _token), result in zip(active, results, strict=False):
            if isinstance(result, Exception):
                logger.error(
                    "job_queue.lock_renew_failed job_id=%s err=%s",
                    job.id,
                    result,
                )


__all__ = ["Worker"]
