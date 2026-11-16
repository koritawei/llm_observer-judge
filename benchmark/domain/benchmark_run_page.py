"""Fold a benchmark_runs row into fields the eval-page task row can use directly.

Does not start an eval; only maps fields at read time.
"""

import re
from datetime import datetime
from typing import Protocol

from benchmark.domain.eval_config import (
    logical_benchmark_run_name,
    parse_registration_eval_key,
)

_RUN_ID_SUFFIX_RE = re.compile(r"-\d{8}(?:-(?P<suffix>.+))?$")


def _run_id_suffix(run_id: str) -> str | None:
    """Expose only the suffix the page uses to classify, not a second page id copied from run_id."""
    matched = _RUN_ID_SUFFIX_RE.search(run_id)
    return matched.group("suffix") if matched is not None else None


def page_task_status(status: object) -> str:
    """Map a run status onto the page's four states.

    ``stopped`` may come from the task document (schedule on/off) or from a run terminal
    state (a manual stop of this run that emits a partial report; see §6.8).
    Anything other than completed / failed / stopped **is shown as active**. That fallback
    is intentional: an unrecognized or new intermediate state should look in progress
    rather than be mislabeled terminal.
    """
    value = getattr(status, "value", status)
    text = str(value)
    if text == "completed":
        return "complete"
    if text == "failed":
        return "failed"
    if text == "stopped":
        return "stopped"
    return "active"


def _iso(value: object) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    return str(value)


class RunPageSource(Protocol):
    """Run fields used when reading progress."""

    id: object
    run_id: str
    benchmark_name: str
    dataset_name: str
    config_json: object
    status: object
    evaluated_items: int | None
    failed_items: int | None
    total_samples: int | None
    final_score: float | None
    error_message: str | None
    report_meta_json: object
    started_at: datetime | None
    finished_at: datetime | None


def page_run_summary(run: RunPageSource) -> dict[str, object]:
    """One run mapped onto the eval-task list: progress, score, report, failure reason."""
    report_meta = run.report_meta_json if isinstance(run.report_meta_json, dict) else {}
    # Pages is the official report entry; OSS report_url is the independent fallback when Pages is missing.
    page_url = report_meta.get("report_site_url")
    oss_report_url = report_meta.get("report_url")
    raw_config = getattr(run, "config_json", None)
    config = raw_config if isinstance(raw_config, dict) else {}
    task_id = config.get("eval_task_id")
    started = run.started_at or getattr(run, "created_at", None)
    display_name = logical_benchmark_run_name(
        column_name=run.benchmark_name,
        config=config,
    )
    registration_id = ""
    snap = config.get("agent_registration")
    if isinstance(snap, dict):
        registration_id = str(snap.get("id") or "").strip()
    if not registration_id:
        registration_id = str(config.get("agent_registration_id") or "").strip()
    if not registration_id:
        _stem, col_reg = parse_registration_eval_key(run.benchmark_name)
        if col_reg:
            registration_id = col_reg
    summary: dict[str, object] = {
        # id: row primary key (delete / task link); runId: business run_id (preferred for case fetches).
        "id": str(run.id),
        "runId": run.run_id,
        # Display logical name; the benchmark_name column may already be a registrationId.
        "benchmarkName": display_name or run.benchmark_name,
        "datasetName": run.dataset_name,
        # Owning task id (snapshot at creation): page reports isolate by task using only this field.
        # A run with no owner (direct API call and the like) is null and enters no task detail.
        "taskId": task_id if isinstance(task_id, str) and task_id else None,
        # Immediate m / scheduled manual s / scheduled on-time has no suffix; the page filters by it.
        "runIdSuffix": _run_id_suffix(run.run_id),
        "status": page_task_status(run.status),
        "runStatus": getattr(run.status, "value", run.status),
        # Settled rows = success + failure. Counting only success shows 0/N on an all-failed run.
        "processedCount": int(run.evaluated_items or 0) + int(run.failed_items or 0),
        "totalItems": int(run.total_samples or 0),
        "score": run.final_score,
        "reportUrl": page_url if isinstance(page_url, str) and page_url else None,
        "ossReportUrl": (
            oss_report_url if isinstance(oss_report_url, str) and oss_report_url else None
        ),
        "failureReason": run.error_message,
        "startedAt": _iso(started),
        "finishedAt": _iso(run.finished_at),
    }
    if registration_id:
        summary["registrationId"] = registration_id
    return summary
