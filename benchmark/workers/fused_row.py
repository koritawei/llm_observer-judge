"""Prewrite fused rows — persist running before judge submit, and never overwrite a terminal state.

``update_or_create(status=running)`` would flip an already success/failed row back to
running, so ``evaluated_items`` / ``judge_received_count`` are already incremented while
aggregation ``exclude(status=running)`` drops the case. This module writes lookup fields
(source_trace_id / input / output) only on create or while the row is still running.
"""

from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.models import BenchmarkRunItem
from benchmark.runs.schema import FUSED_CRITERION_ID

logger = get_logger("agent_eval.eval_workers.fused_row")

TERMINAL_ITEM_STATUSES = frozenset({"success", "failed"})


@dataclass(frozen=True)
class FusedRunningPrewrite:
    """Arguments for ``prewrite_running_fused_item`` (FUNC-PARAMS-001)."""

    run_id: str
    item_id: str
    source_trace_id: str | None = None
    item_input: object = None
    item_output: object = None
    evaluator_ref: str = "fused"
    create_score: float | None = None
    create_judge_json: dict[str, Any] | None = None


def is_fused_item_terminal(status: str | None) -> bool:
    """Whether the fused row is already success/failed (do not prewrite running again)."""
    return str(status or "") in TERMINAL_ITEM_STATUSES


async def prewrite_running_fused_item(write: FusedRunningPrewrite) -> bool:
    """Insert or refresh a fused running row; return unchanged if it is already terminal.

    Returns:
        True: the row is still in flight (newly created or still running); the caller may
            continue submit / write job_id.
        False: already success/failed; the caller must not submit or change status.
    """
    defaults: dict[str, Any] = {
        "evaluator_ref": write.evaluator_ref,
        "status": "running",
        "input": write.item_input,
        "output": write.item_output,
        "source_trace_id": write.source_trace_id,
    }
    if write.create_score is not None:
        defaults["score"] = write.create_score
    if write.create_judge_json is not None:
        defaults["judge_json"] = write.create_judge_json

    item, created = await BenchmarkRunItem.get_or_create(
        run_id=write.run_id,
        item_id=write.item_id,
        criterion_id=FUSED_CRITERION_ID,
        defaults=defaults,
    )
    if is_fused_item_terminal(item.status):
        logger.info(
            "fused.prewrite_skip_terminal run_id=%s item_id=%s status=%s",
            write.run_id,
            write.item_id,
            item.status,
        )
        return False
    if created:
        return True

    update_fields: list[str] = []
    if write.source_trace_id:
        item.source_trace_id = write.source_trace_id
        update_fields.append("source_trace_id")
    if write.item_input is not None:
        item.input = write.item_input
        update_fields.append("input")
    if write.item_output is not None:
        item.output = write.item_output
        update_fields.append("output")
    if update_fields:
        await item.save(update_fields=update_fields)
    return True
