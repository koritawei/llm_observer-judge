"""Streaming task model — declarative spec and state machine (§5.2 / §6.2).

State machine (aligned with reference design v4 terms)::

    config adds an entry → running ⇄ paused (pause window accumulates lag, catch up after resume)
                           running ⇄ stopped (stop intent; reconciliation does not auto-start)
    config removes an entry = delete (terminal; instance stops + registry removal)
"""

from dataclasses import dataclass
from typing import Literal

from benchmark.runs.common.execution_classes import Execution

#: Task state-machine values (§6.2): running / paused / stopped; delete = remove the entry.
StreamingTaskState = Literal["running", "paused", "stopped"]


@dataclass(frozen=True)
class StreamingTaskSpec:
    """Declarative spec for one streaming task.

    RemoteConfig ``agent-eval.benchmark.streaming`` ``tasks[]`` entries and the env
    static fallback (P1 shape) share this model.

    Attributes:
        name: task name (registry primary key; same name as the trace-daily-report item ``name``).
        topics: subscribed topic set (adding or removing an element adds or removes that task's subscription, §6.3).
        group_id: consumer group (resolved): a ``null`` migration task takes over the same value as ``JUDGE_KAFKA_GROUP_ID``
            (cutover continues the offset); every other task must set an explicit independent value.
        state: declared state (running / paused / stopped).
        judge_class: streaming judge alias (``STREAMING_JUDGE_CLASS_REFS``);
            ``None`` = current zero-code routing (scoreName → ``judge_<scoreName>`` agent).
        judge_execution: effective judge execution mode (``sync`` / ``async``);
            composed from the class declaration and the ``tasks[].judge_execution`` override.
        triage_class: streaming triage alias; ``None`` = look up by evaluator_name.
        attribution_class: ``None`` = current synchronous triage attribution chain;
            ``no_attribution`` = turn off attribution and dispatch; an explicit alias overrides the agent.
        attribution_execution: effective attribution execution mode (``sync`` / ``async`` /
            ``none``); may be combined freely with judge_execution.
        rubric_class: ``RUBRIC_CLASS_REFS`` allowlist alias (optional; association-check key,
            not read at runtime — report wording follows the trace-daily-report item, §5.1).
        sampling: sampling-rate override ``(0, 1]`` (optional; default follows the RemoteConfig eval config,
            and an override departs from the online config, for emergency mitigation only).
        daily_report_enabled: corresponds to v4 ``t1_enabled`` (association check only, §7.2;
            does not drive scheduling).
        shadow: shadow reconciliation mode (short-circuits from S8; no execute / no persist / no attribution).
    """

    name: str
    topics: list[str]
    group_id: str
    state: StreamingTaskState = "running"
    judge_class: str | None = None
    judge_execution: Execution = "sync"
    triage_class: str | None = None
    attribution_class: str | None = None
    attribution_execution: Execution = "sync"
    rubric_class: str | None = None
    sampling: float | None = None
    daily_report_enabled: bool = False
    shadow: bool = False

    @property
    def no_attribution(self) -> bool:
        """True when attribution and dispatch are turned off at the task level."""
        return self.attribution_class == "no_attribution" or self.attribution_execution == "none"


#: Static env fallback task name (P1 shape / cold-start fallback when RemoteConfig is unavailable, §8).
ENV_FALLBACK_TASK_NAME = "judge_kafka_env_fallback"


__all__ = [
    "ENV_FALLBACK_TASK_NAME",
    "StreamingTaskSpec",
    "StreamingTaskState",
]
