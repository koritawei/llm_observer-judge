"""StreamingTaskRegistry — in-process task registry (name → RunningTask, §6.1).

Declarative RemoteConfig is the only source of truth; the registry holds **runtime state** (consumer instance groups /
run tasks / stop_event / runtime-state flags) for reconciler diffing and the management API's dual view.
A process restart clears it; cold start reconverges from RemoteConfig (or the env fallback).
"""

import asyncio
from dataclasses import dataclass, field

from benchmark.libs.mq import MQConsumer

from benchmark.streaming.consumer import BenchmarkStreamingEvaluator
from benchmark.streaming.tasks import (
    StreamingTaskSpec,
    StreamingTaskState,
)


@dataclass
class RunningTask:
    """In-process runtime instance group for one streaming task.

    ``consumers`` / ``run_tasks`` / ``stop_events`` are organized at **topic granularity**
    (§6.3): when the topics set changes, add or remove by element, and do not rebuild instances for unchanged topics, to avoid
    interrupting in-flight work. Draining a topic instance group sets the matching ``stop_events[topic]``,
    which interrupts that topic instance's holdback wait (§3.2).
    """

    spec: StreamingTaskSpec
    evaluator: BenchmarkStreamingEvaluator
    consumers: dict[str, list[MQConsumer]] = field(default_factory=dict)
    run_tasks: dict[str, list[asyncio.Task[None]]] = field(default_factory=dict)
    stop_events: dict[str, asyncio.Event] = field(default_factory=dict)
    runtime_state: StreamingTaskState = "running"

    @property
    def is_running(self) -> bool:
        """Whether any consumer instance of this task is running.

        ``MQConsumer`` only exposes ``_running``. Reconciliation only reads this flag to see whether the instance is still in the
        ``run()`` loop; it does not start / stop here.
        """
        return any(c._running for instances in self.consumers.values() for c in instances)

    @property
    def instance_count(self) -> int:
        """Total running consumer instances (topic × concurrency)."""
        return sum(len(instances) for instances in self.consumers.values())

    @property
    def subscribed_topics(self) -> list[str]:
        """Topics actually subscribed right now (runtime view)."""
        return sorted(self.consumers)


class StreamingTaskRegistry:
    """In-process task registry (name → RunningTask).

    Where the reconciler applies its diff, and the data source for the management API's runtime view;
    every operation is idempotent (repeated upsert/remove is safe).
    """

    def __init__(self) -> None:
        self._tasks: dict[str, RunningTask] = {}

    def get(self, name: str) -> RunningTask | None:
        """Get a task's runtime state by name; return ``None`` if it is absent."""
        return self._tasks.get(name)

    def names(self) -> list[str]:
        """Task names in the registry (sorted, stable output)."""
        return sorted(self._tasks)

    def list(self) -> list[RunningTask]:
        """Runtime state of every task (sorted by name)."""
        return [self._tasks[name] for name in self.names()]

    def upsert(self, task: RunningTask) -> RunningTask | None:
        """Register or replace a task's runtime state; return the replaced task (``None`` if there was none).

        The caller stops the old task's instances (this registry does not manage lifecycle).
        """
        previous = self._tasks.get(task.spec.name)
        self._tasks[task.spec.name] = task
        return previous

    def remove(self, name: str) -> RunningTask | None:
        """Remove a task's runtime state; return the removed task (``None`` if there was none)."""
        return self._tasks.pop(name, None)

    def mark_state(self, name: str, state: StreamingTaskState) -> None:
        """Mark a task's runtime state (paused / stopped intent stays registered, §6.2)."""
        task = self._tasks.get(name)
        if task is not None:
            task.runtime_state = state


__all__ = [
    "RunningTask",
    "StreamingTaskRegistry",
]
