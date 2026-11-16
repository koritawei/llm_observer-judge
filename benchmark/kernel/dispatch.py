"""BullMQ dispatch. Running queues have no worker; they are the in-flight ledger."""

from __future__ import annotations

from typing import Any

from bullmq import Job, Queue

JUDGE_QUEUES = ("benchmark-cases", "benchmark-cases-l2", "dataset-cases")
RUNNING_QUEUE = {
    "benchmark-cases": "benchmark-running",
    "benchmark-cases-l2": "benchmark-running",
    "dataset-cases": "dataset-running",
    "attribution-wait": "attribution-running",
}


class Dispatch:
    def __init__(self, redis_url: str, env_prefix: str, generation: str):
        self.redis_url = redis_url
        self.env_prefix = env_prefix
        self.generation = generation
        self._queues: dict[str, Queue] = {}

    def prefix_for(self, queue_name: str) -> str:
        return "{" + f"{self.env_prefix}:bullmq:{self.generation}:{queue_name}" + "}"

    def options(self, queue_name: str) -> dict[str, Any]:
        return {
            "connection": self.redis_url,
            "prefix": self.prefix_for(queue_name),
            "skipVersionCheck": True,
        }

    def queue(self, name: str) -> Queue:
        if name not in self._queues:
            self._queues[name] = Queue(name, self.options(name))
        return self._queues[name]

    async def enqueue(self, queue_name: str, job_id: str, name: str, data: dict[str, Any], *, keep: bool) -> bool:
        existing = await Job.fromId(self.queue(queue_name), job_id)
        if existing is not None:
            return False
        await self.queue(queue_name).add(
            name,
            data,
            {
                "jobId": job_id,
                "attempts": 1,
                "removeOnComplete": not keep,
                "removeOnFail": False,
            },
        )
        return True

    async def get(self, queue_name: str, job_id: str) -> Job | None:
        return await Job.fromId(self.queue(queue_name), job_id)

    async def waiting_and_delayed(self, queue_name: str) -> list[Job]:
        return await self.queue(queue_name).getJobs(["waiting", "delayed"], 0, -1, True)

    async def close(self) -> None:
        for queue in self._queues.values():
            await queue.close()
        self._queues.clear()
