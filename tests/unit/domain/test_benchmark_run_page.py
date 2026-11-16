"""Run rows mapped onto evaluation-page fields."""

from datetime import UTC, datetime
from types import SimpleNamespace

from benchmark.domain.benchmark_run_page import page_run_summary, page_task_status


def test_active_statuses_stay_active() -> None:
    assert page_task_status("running") == "active"
    assert page_task_status("generating_report") == "active"
    assert page_task_status("completed") == "complete"
    assert page_task_status("failed") == "failed"
    assert page_task_status("stopped") == "stopped"


def test_page_summary_uses_task_columns() -> None:
    started = datetime(2026, 10, 12, 2, 0, tzinfo=UTC)
    run = SimpleNamespace(
        id="0f8fad5b-d9cb-469f-a165-70867728950e",
        run_id="gpt-judge-pro-20261012-m1",
        benchmark_name="gpt-judge-pro",
        dataset_name="default-evaluation-set-v1",
        status="completed",
        evaluated_items=120,
        failed_items=0,
        total_samples=120,
        final_score=0.83,
        error_message=None,
        report_meta_json={
            "report_site_url": "https://pages.example/report.html",
            "report_url": "https://oss.example/report.html",
        },
        started_at=started,
        finished_at=started,
        created_at=started,
    )
    summary = page_run_summary(run)
    # id is for delete/association; runId is for loading cases; runIdSuffix classifies the report namespace.
    assert summary["id"] == "0f8fad5b-d9cb-469f-a165-70867728950e"
    assert summary["runId"] == "gpt-judge-pro-20261012-m1"
    assert summary["runIdSuffix"] == "m1"
    assert summary["status"] == "complete"
    assert summary["processedCount"] == 120
    assert summary["totalItems"] == 120
    assert summary["score"] == 0.83
    # reportUrl prefers Aone Pages; the OSS report_url is also exposed as a fallback under ossReportUrl
    assert summary["reportUrl"] == "https://pages.example/report.html"
    assert summary["ossReportUrl"] == "https://oss.example/report.html"
    assert summary["failureReason"] is None
    assert summary["startedAt"] == started.isoformat()


def test_page_summary_exposes_oss_report_when_pages_missing() -> None:
    started = datetime(2026, 10, 12, 3, 0, tzinfo=UTC)
    run = SimpleNamespace(
        id="2a8fad5b-d9cb-469f-a165-708677289510",
        run_id="gpt-judge-pro-20261012-s2",
        benchmark_name="gpt-judge-pro",
        dataset_name="default-evaluation-set-v1",
        status="completed",
        evaluated_items=10,
        failed_items=0,
        total_samples=10,
        final_score=0.5,
        error_message=None,
        report_meta_json={"report_url": "https://oss.example/only-oss.html"},
        started_at=started,
        finished_at=started,
        created_at=started,
    )
    summary = page_run_summary(run)
    assert summary["reportUrl"] is None
    assert summary["ossReportUrl"] == "https://oss.example/only-oss.html"
    assert summary["runIdSuffix"] == "s2"


def test_page_summary_counts_failed_items_as_processed() -> None:
    """Progress of an all-failed run = processed (success + failed), not the success count.

    Live incident: every judge llm_auth_failed → evaluated=0 failed=10, and the page
    showed 0/10 while the status was "all complete" — progress contradicted the terminal state. Aligned with the submit-pipeline invariant
    ``(evaluated + failed) >= total``, it should show 10/10.
    """
    started = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)
    run = SimpleNamespace(
        id="1c9fad5b-d9cb-469f-a165-70867728950f",
        run_id="search-eval-20261020",
        benchmark_name="search_eval",
        dataset_name="ai-card-sample",
        status="completed",
        evaluated_items=0,
        failed_items=10,
        total_samples=10,
        final_score=None,
        error_message="llm call failed: BadRequestError: 无效的api key (CE-003)",
        report_meta_json={},
        started_at=started,
        finished_at=started,
        created_at=started,
    )
    summary = page_run_summary(run)
    assert summary["processedCount"] == 10
    assert summary["totalItems"] == 10
    assert summary["status"] == "complete"
    assert summary["runIdSuffix"] is None
