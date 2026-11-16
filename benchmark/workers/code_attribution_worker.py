"""Code-attribution queue. The callback persists and returns first; this worker advances received and dispatch after it finishes."""

import asyncio
from concurrent.futures import Executor, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from bullmq import Job, Worker

from benchmark.settings.setting import Settings
from benchmark.records.models import BenchmarkRunItem
DispatchRequest = None
class schedule_notification_dispatch:
    pass
from benchmark.runs.run_service import BenchmarkService
from benchmark.runs.schema import FUSED_CRITERION_ID, AttributionResult
from benchmark.attribution.code.document import evidence_text
from benchmark.attribution.code.pipeline import AnalyzeAndApply, CodeAttributionCase
from benchmark.queue.queue_factory import CODE_ATTRIBUTION, EvalQueueRegistry
from benchmark.workers.finalize_trigger import try_finalize_external

logger = get_logger("agent_eval.eval_workers.code_attribution")


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _case_from_row(run_id: str, item_id: str, row: BenchmarkRunItem) -> CodeAttributionCase:
    stored = row.attribution_detail_json if isinstance(row.attribution_detail_json, dict) else {}
    raw = stored.get("raw_json") if isinstance(stored.get("raw_json"), dict) else {}
    item_input = row.input if isinstance(row.input, dict) else {}
    item_output = row.output if isinstance(row.output, dict) else {}
    sources = [raw, stored, item_output]
    response = _text(item_output.get("response_text") or item_output.get("result"))
    return CodeAttributionCase(
        case_id=f"{run_id}:{item_id}",
        query=_text(item_input.get("query")),
        tool_calls_json=evidence_text(sources, "tool_calls_json", "full_tool_calls_json"),
        llm_calls_json=evidence_text(sources, "llm_calls_json", "full_llm_calls_json"),
        last_generation_input=evidence_text(
            sources,
            "last_generation_input",
            "full_last_generation_input",
        ),
        response_text=response,
        document=stored,
    )


@dataclass(frozen=True)
class CodeAttributionJobDeps:
    """Host objects one worker processing step depends on."""

    run_service: BenchmarkService
    executor: Executor
    step: AnalyzeAndApply | None = None


async def process_code_attribution_job(
    job: Job,
    _token: str,
    deps: CodeAttributionJobDeps,
) -> None:
    """Run code attribution. Advance received even on failure so the run does not stay unfinished."""
    data = dict(job.data or {})
    run_id = str(data.get("run_id") or "")
    item_id = str(data.get("item_id") or "")
    advance_received = bool(data.get("advance_received"))
    source_trace_id = _text(data.get("source_trace_id")) or None
    run_service = deps.run_service
    runner = deps.step if deps.step is not None else AnalyzeAndApply()
    dispatch_doc: dict[str, Any] | None = None
    try:
        row = await BenchmarkRunItem.filter(
            run_id=run_id,
            item_id=item_id,
            criterion_id=FUSED_CRITERION_ID,
        ).first()
        if row is None:
            logger.warning(
                "code_attribution.missing_row run_id=%s item_id=%s",
                run_id,
                item_id,
            )
        else:
            if not source_trace_id:
                source_trace_id = _text(row.source_trace_id) or None
            stored = (
                row.attribution_detail_json if isinstance(row.attribution_detail_json, dict) else {}
            )
            dispatch_doc = stored
            context = copy_context()
            outcome = await asyncio.get_running_loop().run_in_executor(
                deps.executor,
                context.run,
                runner.run,
                _case_from_row(run_id, item_id, row),
            )
            if outcome.applied:
                await BenchmarkRunItem.filter(
                    run_id=run_id,
                    item_id=item_id,
                    criterion_id=FUSED_CRITERION_ID,
                ).update(attribution_detail_json=outcome.document)
                dispatch_doc = outcome.document
                logger.info(
                    "code_attribution.applied run_id=%s item_id=%s",
                    run_id,
                    item_id,
                )
            else:
                logger.info(
                    "code_attribution.kept run_id=%s item_id=%s",
                    run_id,
                    item_id,
                )
    except Exception:
        logger.exception(
            "code_attribution.failed run_id=%s item_id=%s",
            run_id,
            item_id,
        )
    if not advance_received:
        return
    new_count, config, claimed = await run_service.increment_attribution_received_once(
        run_id,
        item_id,
    )
    if claimed and dispatch_doc is not None:
        raw = dispatch_doc.get("raw_json")
        payload = raw if isinstance(raw, dict) else dispatch_doc
        try:
            schedule_notification_dispatch(
                DispatchRequest(
                    store=run_service,
                    run_id=run_id,
                    item_id=item_id,
                    source_trace_id=source_trace_id,
                    config=config,
                ),
                data=AttributionResult.model_validate(payload),
            )
        except Exception:
            logger.exception(
                "code_attribution.dispatch_failed run_id=%s item_id=%s",
                run_id,
                item_id,
            )
    await run_service.heartbeat(run_id)
    expected = config.get("attribution_expected_count", 0)
    if new_count >= expected:
        await try_finalize_external(run_id, run_service)


def _bind_executor_close(worker: Worker, executor: ThreadPoolExecutor) -> Worker:
    """Shut down the analysis thread pool after ``worker.close`` returns."""
    close_worker = worker.close

    async def close(force: bool = False) -> None:
        try:
            await close_worker(force=force)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    worker.close = close  # type: ignore[method-assign]
    return worker


def create_code_attribution_worker(
    *,
    queue_manager: EvalQueueRegistry,
    run_service: BenchmarkService,
    settings: Settings,
) -> Worker:
    """Create the code-attribution queue Worker. Concurrency is 1, matching the in-module file lock."""
    queue = queue_manager.get_queue(CODE_ATTRIBUTION)
    opts = queue_manager.build_worker_opts(
        queue_name=CODE_ATTRIBUTION,
        lock_duration_ms=settings.code_attribution_lock_duration_ms,
        stalled_interval_ms=settings.eval_queue_stalled_interval_ms,
    )
    opts["concurrency"] = 1
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="code-attribution")

    async def processor(job: Job, token: str) -> None:
        await process_code_attribution_job(
            job,
            token,
            CodeAttributionJobDeps(run_service=run_service, executor=executor),
        )

    try:
        worker = Worker(queue.name, processor, opts)
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    logger.info("code_attribution_worker.started queue=%s concurrency=1", queue.name)
    return _bind_executor_close(worker, executor)
