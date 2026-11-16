"""Dataset / agent read path: services inject call counts to avoid a domain↔BenchmarkService import cycle."""

from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.benchmark_console import (
    fill_agent_evaluation_counts,
    fill_realtime_bound_agent_counts,
)
from benchmark.runs.run_service import BenchmarkService

logger = get_logger(__name__)


async def enrich_realtime_bound_agent_counts(items: list[dict[str, Any]]) -> None:
    """For streaming datasets whose trace count is still 0, backfill with the sum of bound agents' evaluationCount."""
    zero_targets = [
        item
        for item in items
        if item.get("source") == "realtime" and int(item.get("itemsCount") or 0) == 0
    ]
    if not zero_targets:
        return
    try:
        counts = await BenchmarkService().count_runs_by_benchmark()
    except Exception:
        logger.warning("realtime_dataset_count.bound_agent_fallback_failed", exc_info=True)
        return
    await fill_realtime_bound_agent_counts(zero_targets, counts)


async def enrich_agent_evaluation_counts(items: list[dict[str, Any]]) -> None:
    """Registration-page evaluationCount: fetch run-counts, then write the public field."""
    if not items:
        return
    try:
        counts = await BenchmarkService().count_runs_by_benchmark()
    except Exception:
        logger.warning("agent_eval_counts.run_counts_failed", exc_info=True)
        for item in items:
            item["evaluationCount"] = 0
            item["batchCount"] = 0
            item["streamingCount"] = 0
        return
    await fill_agent_evaluation_counts(items, counts)
