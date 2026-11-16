"""finalize_trigger — atomic CAS completion detection + report trigger (design doc §10).

Two completion-detection modes:
- **Sync model** (benchmark batch): atomic CAS on progress columns
  ``(evaluated_items + failed_items) >= total_items * evaluator_count``
- **Async model** (external): atomic CAS on config_json counters
  ``judge_received_count >= judge_expected_count``

The CAS winner runs the report flow (aggregate + generate report → completed).

The ``on_finalize`` callback is registered by ``main.py`` and wraps aggregation +
report generation (it needs ``BenchmarkRunner`` and other heavy dependencies).
"""

from collections.abc import Awaitable, Callable
from typing import Literal

from benchmark.libs.agenthost.observability.logger import get_logger
from tortoise import Tortoise

from benchmark.runs.run_service import BenchmarkService, CompleteRunWrite
from benchmark.runs.run_stop import (
    is_stop_requested,
    stop_items_ready_for_report,
)

logger = get_logger("agent_eval.eval_queue.finalize")


async def _stop_latch_blocks_report(run_id: str, config: object) -> bool:
    """Under the stop latch, refuse the report while any item is still running (wait for success/failed)."""
    if not is_stop_requested(config):
        return False
    ready = await stop_items_ready_for_report(run_id)
    if ready:
        return False
    logger.info(
        "finalize.stop_waiting_running_items run_id=%s — defer report until items terminal",
        run_id,
    )
    return True


#: Finalize callback type: ``(run_id) -> None``, registered by main.py.
FinalizeCallback = Callable[[str], Awaitable[None]]

#: eval_type routing key (aligned with worker-side ``SyncEvalHandlers.eval_type``).
EVAL_TYPE_BENCHMARK = "benchmark"
EVAL_TYPE_EXTERNAL = "external"

#: Legal eval_type set (shared by register / tick routing; typos are statically visible).
EvalType = Literal["benchmark", "external"]

#: Global callback registry (set by main.py at startup).
_finalize_callbacks: dict[str, FinalizeCallback] = {}

#: [stop only] latch + stop_target: enter aggregating only when processed count >= stop_target.
#: stop_target_count **must exist and be numeric** — missing means not ready (do not proceed).
#: Otherwise the window between mark_stop and align treats "latch present" as immediate aggregating,
#: emitting a partial report before in-flight results have landed.
#: Also require no ``benchmark_run_items.status=running`` (same rule as the Python latch;
#: the write CAS prevents TOCTOU: do not report if running rows remain when the count is met).
_STOP_NO_RUNNING_ITEMS_SQL = """
NOT EXISTS (
    SELECT 1 FROM benchmark_run_items bri
    WHERE bri.run_id = benchmark_runs.run_id
      AND bri.status = 'running'
)
"""

_STOP_TARGET_READY_SQL = (
    """
(
    (config_json->>'stop_requested_at') IS NOT NULL
    AND (config_json->>'stop_target_count') ~ '^[0-9]+$'
    AND (evaluated_items + failed_items)
        >= (config_json->>'stop_target_count')::int
    AND """
    + _STOP_NO_RUNNING_ITEMS_SQL
    + """
)
"""
)

#: external CAS: with a stop latch, also refuse to advance while any item is still running.
_STOP_LATCH_ALLOWS_EXTERNAL_CAS_SQL = (
    """
(
    (config_json->>'stop_requested_at') IS NULL
    OR """
    + _STOP_NO_RUNNING_ITEMS_SQL
    + """
)
"""
)


def register_finalize_callback(eval_type: EvalType, callback: FinalizeCallback) -> None:
    """Register the finalize callback (called by main.py during lifespan startup).

    Args:
        eval_type: benchmark / external (see :data:`EvalType`).
        callback: ``(run_id) -> None``; runs aggregation + report + completed.
    """
    _finalize_callbacks[eval_type] = callback
    logger.info("finalize.callback_registered eval_type=%s", eval_type)


def _cas_won(rows: tuple) -> bool:
    """``execute_query`` returns an ``(affected, rows)`` pair; treat a malformed shape as a CAS loss."""
    return bool(rows and len(rows) > 1 and rows[1])


async def try_finalize_benchmark(
    run_id: str,
    run_service: BenchmarkService,
) -> bool:
    """Sync-model benchmark CAS completion detection (§10.1).

    The completion threshold depends on triage_mode:
    - **fused**: ``(evaluated + failed) >= total_samples``
      (each item writes 1 row with criterion_id='fused')
    - **per_eval / none**: ``(evaluated + failed) >= total_samples * total_evaluators``
      (each item writes N rows, one per criterion)
    - **stop only**: also allow aggregating before the run finishes when
      ``stop_requested_at`` is set; a normal completed must not forge that latch.

    The winner runs aggregation + FUSED triage + report generation → completed/stopped
    (via ``on_finalize``).

    Returns:
        True if this call won the CAS.
    """
    # Read triage_mode to choose the completion threshold
    run = await run_service.get_run(run_id)
    if run is None:
        logger.error("benchmark.finalize_run_not_found run_id=%s", run_id)
        return False
    config = run.config_json if isinstance(run.config_json, dict) else {}
    if await _stop_latch_blocks_report(run_id, config):
        return False
    triage_mode = config.get("triage_mode", "fused")
    is_fused = triage_mode == "fused"

    # fused: 1 row per item → threshold = total_samples
    # per_eval/none: N rows per item → threshold = total_samples * total_evaluators
    # [stop only] see ``_STOP_TARGET_READY_SQL`` (wait for in-flight running results);
    # normal completion still requires processed >= total; do not forge the latch without stop.
    # Threshold is fixed branches only (no external input): splice constants into the full SQL; never f-string user input.
    if is_fused:
        cas_sql = (
            """
        UPDATE benchmark_runs
        SET    status = 'aggregating'
        WHERE  run_id = $1
          AND  status = 'running'
          AND  (
                   """
            + _STOP_TARGET_READY_SQL
            + """
                OR (evaluated_items + failed_items) >= total_samples
               )
        RETURNING run_id
        """
        )
    else:
        cas_sql = (
            """
        UPDATE benchmark_runs
        SET    status = 'aggregating'
        WHERE  run_id = $1
          AND  status = 'running'
          AND  (
                   """
            + _STOP_TARGET_READY_SQL
            + """
                OR (evaluated_items + failed_items)
                   >= total_samples * total_evaluators
               )
        RETURNING run_id
        """
        )

    conn = Tortoise.get_connection("default")
    rows = await conn.execute_query(cas_sql, [run_id])
    won = _cas_won(rows)

    if not won:
        # Phase 1 CAS lost — check whether status is already AGGREGATING (stuck recovery)
        rows = await conn.execute_query(
            "SELECT 1 FROM benchmark_runs WHERE run_id = $1 AND status = 'aggregating'",
            [run_id],
        )
        if not _cas_won(rows):
            return False
        # Stop latch with non-terminal items: pull back to running so stuck recovery cannot report early
        if await _stop_latch_blocks_report(run_id, config):
            await conn.execute_query(
                """
                UPDATE benchmark_runs
                SET    status = 'running'
                WHERE  run_id = $1
                  AND  status = 'aggregating'
                """,
                [run_id],
            )
            return False
        logger.info("benchmark.stuck_aggregating_recovery run_id=%s", run_id)
    else:
        logger.info("benchmark.cas_won run_id=%s → aggregating", run_id)

    # Callback invocation (shared by Phase 1 win / stuck recovery)
    callback = _finalize_callbacks.get(EVAL_TYPE_BENCHMARK)
    try:
        if callback is not None:
            await callback(run_id)
        else:
            fused_avg = await run_service.load_fused_score_average(run_id)
            await run_service.complete_run(
                CompleteRunWrite(
                    run_id=run_id,
                    final_score=float(fused_avg) if fused_avg is not None else 0.0,
                    metrics_json={"finalize_source": "benchmark_cas_no_callback"},
                ),
            )
    except Exception:
        logger.exception("benchmark.finalize_failed run_id=%s", run_id)
        try:
            await run_service.fail_run(run_id=run_id, reason="finalize failed")
        except Exception:
            logger.exception(
                "benchmark.finalize_fail_run_also_failed run_id=%s (status may remain aggregating)",
                run_id,
            )
    return True


async def try_finalize_external(
    run_id: str,
    run_service: BenchmarkService,
) -> bool:
    """Async-model external CAS completion detection (§10.2 counter semantics).

    Atomic SQL: ``status = 'running' AND judge_received >= judge_expected``.
    The winner runs attribution routing → aggregation → report → completed.

    Returns:
        True if this call won the CAS.
    """
    run = await run_service.get_run(run_id)
    if run is None:
        logger.error("external.finalize_run_not_found run_id=%s", run_id)
        return False
    config = run.config_json if isinstance(run.config_json, dict) else {}
    if await _stop_latch_blocks_report(run_id, config):
        return False

    conn = Tortoise.get_connection("default")
    # Phase 1: RUNNING → AGGREGATING (CAS + counter condition).
    # CASE regex guards ::int: an empty or non-numeric counter becomes NULL (the comparison
    # fails and the CAS does not win) instead of raising and aborting the whole finalize flow.
    rows = await conn.execute_query(
        """
        UPDATE benchmark_runs
        SET    status = 'aggregating'
        WHERE  run_id = $1
          AND  status = 'running'
          AND  (CASE WHEN config_json->>'judge_received_count' ~ '^[0-9]+$'
                     THEN (config_json->>'judge_received_count')::int END)
               >= COALESCE(
                   CASE WHEN config_json->>'judge_expected_count' ~ '^[0-9]+$'
                        THEN (config_json->>'judge_expected_count')::int END, 0)
          AND  COALESCE(
                   CASE WHEN config_json->>'attribution_received_count' ~ '^[0-9]+$'
                        THEN (config_json->>'attribution_received_count')::int END, 0)
               >= COALESCE(
                   CASE WHEN config_json->>'attribution_expected_count' ~ '^[0-9]+$'
                        THEN (config_json->>'attribution_expected_count')::int END, 0)
          AND  """
        + _STOP_LATCH_ALLOWS_EXTERNAL_CAS_SQL
        + """
        RETURNING run_id
        """,
        [run_id],
    )
    won = _cas_won(rows)

    if not won:
        # Phase 1 CAS lost — check whether status is already AGGREGATING (stuck recovery)
        rows = await conn.execute_query(
            "SELECT 1 FROM benchmark_runs WHERE run_id = $1 AND status = 'aggregating'",
            [run_id],
        )
        if not _cas_won(rows):
            logger.info(
                "external.cas_lost run_id=%s conditions_not_met_or_already_finalized",
                run_id,
            )
            return False
        # Stop latch with non-terminal items: pull back to running so stuck recovery cannot report early
        if await _stop_latch_blocks_report(run_id, config):
            await conn.execute_query(
                """
                UPDATE benchmark_runs
                SET    status = 'running'
                WHERE  run_id = $1
                  AND  status = 'aggregating'
                """,
                [run_id],
            )
            return False
        logger.info("external.stuck_aggregating_recovery run_id=%s", run_id)
    else:
        logger.info("external.cas_won run_id=%s → aggregating", run_id)

    # Callback invocation (shared by Phase 1 win / stuck recovery)
    callback = _finalize_callbacks.get(EVAL_TYPE_EXTERNAL)
    try:
        if callback is not None:
            logger.info(
                "external.finalize_callback_start run_id=%s",
                run_id,
            )
            await callback(run_id)
            logger.info(
                "external.finalize_callback_done run_id=%s",
                run_id,
            )
        else:
            fused_avg = await run_service.load_fused_score_average(run_id)
            await run_service.complete_run(
                CompleteRunWrite(
                    run_id=run_id,
                    final_score=float(fused_avg) if fused_avg is not None else 0.0,
                    metrics_json={"finalize_source": "external_cas_no_callback"},
                ),
            )
    except Exception:
        logger.exception("external.finalize_failed run_id=%s", run_id)
        try:
            await run_service.fail_run(run_id=run_id, reason="finalize failed")
        except Exception:
            logger.exception(
                "external.finalize_fail_run_also_failed run_id=%s (status may remain aggregating)",
                run_id,
            )
    return True


async def tick_and_maybe_finalize(
    run_id: str,
    eval_type: EvalType,
    *,
    benchmark_run_service: BenchmarkService | None = None,
) -> None:
    """Called from processor finally: attempt CAS completion detection.

    Route to the matching finalize function by eval_type.
    Non-terminal runs do not fire (CAS WHERE status='running' filters them naturally).
    """
    try:
        if eval_type == EVAL_TYPE_BENCHMARK and benchmark_run_service is not None:
            await try_finalize_benchmark(run_id, benchmark_run_service)
        elif eval_type == EVAL_TYPE_EXTERNAL and benchmark_run_service is not None:
            await try_finalize_external(run_id, benchmark_run_service)
    except Exception:
        logger.warning(
            "finalize.tick_failed run_id=%s eval_type=%s",
            run_id,
            eval_type,
            exc_info=True,
        )


__all__ = [
    "EVAL_TYPE_BENCHMARK",
    "EVAL_TYPE_EXTERNAL",
    "EvalType",
    "register_finalize_callback",
    "tick_and_maybe_finalize",
    "try_finalize_benchmark",
    "try_finalize_external",
]
