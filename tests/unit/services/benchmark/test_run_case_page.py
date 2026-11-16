"""Case paging: the display plugin normalizes judge/attribution, plus the items and Kafka load paths."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.report_summary_html import strip_case_details_from_report_html
from benchmark.runs.run_case_page import (
    CaseDetailQuery,
    CaseListItemSpec,
    CasePageQuery,
    build_case_detail,
    build_case_page,
    case_detail_item,
    case_list_item,
    detail_expected_text,
    detail_input_text,
    extract_attribution_category,
    find_report_case,
    load_item_notification_row,
    load_report_cases,
    report_url_of,
    supports_attribution_category,
)
from benchmark.runs.schema import FUSED_CRITERION_ID
from benchmark.reporting.framework.schema import Criterion

_SH = timezone(timedelta(hours=8))


def _fused_case(case_id: str, *, query: str) -> dict:
    judge = {
        "score": 1.7,
        "reason": "ok",
        "criterion_scores": {"D0": {"score": 2.0, "reason": "pass"}},
    }
    attribution = {
        "attribution_md": "## 根因\n\ntool_timeout",
        "root_cause": "tool_timeout",
        "confidence": 0.8,
    }
    return {
        "case_id": case_id,
        "input": {"query": query, "request_body": {"query": query}},
        "output": {"result": "沿机场高速"},
        "display_input": query,
        "trace_url": f"https://trace.example/{case_id}",
        "scores": {
            "D0": {"score": 2.0, "judge": judge, "attribution": None},
            "fused": {"score": 1.7, "judge": judge, "attribution": None},
        },
        "attribution": attribution,
    }


def test_judge_and_attribution_match_cases_data_objects() -> None:
    case = _fused_case("item-1", query="从机场怎么走")
    case["grade"] = "GOOD"
    case["attribution"] = {
        **case["attribution"],
        "raw_json": {
            "primary_issue_layer": "tool",
            "primary_issue_category": "tool_unavailable",
        },
    }
    item = case_list_item(
        case,
        spec=CaseListItemSpec(
            notification_dispatch={
                "notification_url": "https://aone.example/1",
                "notification_ref": "1",
                "notification_assigned_to": "八鹤",
            },
            notification_source="items",
            benchmark_name="ieval_external",
            include_attribution_category=True,
        ),
    )
    assert item["id"] == "item-1"
    assert item["query"] == "从机场怎么走"
    assert item["score"] == 1.7
    assert item["status"] == "GOOD"
    assert item["attribution_category"] == "tool/tool_unavailable"
    assert "input" not in item
    assert item["output"] == "沿机场高速"
    assert item["judge"]["score"] == 1.7
    assert item["judge"]["reason"] == "ok"
    assert item["judge"]["dimensions"][0]["id"] == "D0"
    assert item["attribution"]["format"] == "markdown"
    assert "tool_timeout" in (item["attribution"]["markdown"] or "")
    assert item["trace_url"] == "https://trace.example/item-1"
    assert item["notification_url"] == "https://aone.example/1"
    assert item["notification_ref"] == "1"
    assert item["notification_assigned_to"] == "八鹤"


def test_case_detail_item_adds_input_and_expected() -> None:
    case = _fused_case("item-1", query="从机场怎么走")
    case["expected_output"] = {"reference": "应推荐机场大巴"}
    item = case_detail_item(case, spec=CaseListItemSpec(notification_source="items"))
    assert "input" in item
    assert "query" in str(item["input"])
    assert "应推荐机场大巴" in str(item["expected_output"])
    assert detail_input_text(case)
    assert detail_expected_text(case)


def test_find_report_case_matches_id() -> None:
    cases = [_fused_case("a", query="q1"), _fused_case("b", query="q2")]
    assert find_report_case(cases, "b")["case_id"] == "b"
    assert find_report_case(cases, "missing") is None


@pytest.mark.asyncio
async def test_build_case_detail_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    run = SimpleNamespace(run_id="run-9", benchmark_name="ieval_external", config_json={})
    case = _fused_case("item-1", query="从机场怎么走")
    case["expected_output"] = "应推荐机场大巴"

    async def _load(_service: object, _run: object) -> tuple[list, str]:
        return [case], "items"

    monkeypatch.setattr(
        "benchmark.runs.run_case_page.load_report_cases",
        _load,
    )
    monkeypatch.setattr(
        "benchmark.runs.run_case_page.lookup_item_notification_dispatch",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "benchmark.runs.run_case_page.owners_for_run",
        AsyncMock(return_value=["八鹤"]),
    )
    payload = await build_case_detail(
        MagicMock(),
        run,
        detail_query=CaseDetailQuery(case_id="item-1"),
    )
    assert payload is not None
    assert payload["data"]["id"] == "item-1"
    assert payload["data"]["expected_output"] == "应推荐机场大巴"
    assert payload["meta"]["owners"] == ["八鹤"]


def test_report_url_of_uses_report_site_url_not_oss() -> None:
    assert (
        report_url_of(
            SimpleNamespace(
                report_meta_json={
                    "report_site_url": "https://pages.example/r.html",
                    "report_url": "https://oss.example/r.html",
                }
            )
        )
        == "https://pages.example/r.html"
    )
    assert (
        report_url_of(
            SimpleNamespace(report_meta_json={"report_url": "https://oss.example/r.html"})
        )
        is None
    )
    assert report_url_of(SimpleNamespace(report_meta_json={})) is None


def test_attribution_category_trace_and_search_eval() -> None:
    assert supports_attribution_category("ieval") is True
    assert supports_attribution_category("search_eval") is False
    assert supports_attribution_category("search_eval_v2") is False
    trace_case = {"attribution": {"category": "prompt"}}
    assert (
        extract_attribution_category(
            trace_case,
            notification_source="trace",
            benchmark_name="amap-agent",
        )
        == "prompt"
    )
    assert (
        extract_attribution_category(
            trace_case,
            notification_source="items",
            benchmark_name="search_eval_v2",
        )
        is None
    )


def test_strip_case_details_from_report_html() -> None:
    html = (
        "<html><head></head><body>"
        '<div id="kpi-grid">kpi</div>'
        '<div id="case-section">cases</div>'
        '<script type="application/json" id="cases-data">[{"id":1}]</script>'
        '<script type="application/json" id="case-columns-data">[1]</script>'
        "</body></html>"
    )
    out = strip_case_details_from_report_html(html)
    assert 'id="cases-data">[]</script>' in out
    assert 'id="case-columns-data">[]</script>' in out
    assert "agent-eval-case-summary-hide" in out
    assert "#case-section" in out


def test_assert_http_report_url_rejects_non_http() -> None:
    from benchmark.runs.report_summary_html import _assert_http_report_url

    with pytest.raises(ValueError, match="unsupported report url"):
        _assert_http_report_url("file:///etc/passwd")
    assert _assert_http_report_url("https://oss.example/r.html") == ("https://oss.example/r.html")


def test_query_falls_back_to_display_extractor() -> None:
    case = {
        "case_id": "c2",
        "input": {"request_body": {"query": "去西湖"}},
        "output": "plain",
        "scores": {
            "D1": {
                "judge": {"score": 1, "reason": "ok"},
                "attribution": {"root_cause": "x", "confidence": 0.5},
            }
        },
    }
    item = case_list_item(case, spec=CaseListItemSpec(notification_source="items"))
    assert item["query"] == "去西湖"
    assert item["output"] == "plain"
    assert item["judge"]["score"] == 1.0
    assert item["attribution"]["format"] == "table"
    assert item["attribution"]["reasoning"] == "x"
    assert item["trace_url"] is None
    assert item["notification_url"] is None


@pytest.mark.asyncio
async def test_build_page_slices_items_and_joins_notification(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [_fused_case(f"item-{i}", query=f"q{i}") for i in range(3)]
    service = SimpleNamespace(
        load_case_results=AsyncMock(return_value=cases),
        load_kafka_case_results=AsyncMock(),
        load_trace_daily_case_results=AsyncMock(),
    )
    run = SimpleNamespace(run_id="run-1", benchmark_name="ieval_external", config_json={})

    async def _notification(run_id: str, case_ids: list[str]) -> dict[str, dict[str, str]]:
        assert run_id == "run-1"
        assert case_ids == ["item-1"]
        return {
            "item-1": {
                "notification_url": "https://aone.example/item-1",
                "notification_ref": "W-1",
                "notification_assigned_to": "八鹤",
            }
        }

    async def _owners(_config: dict[str, object]) -> list[str]:
        return ["八鹤", "青冬"]

    monkeypatch.setattr(
        "benchmark.runs.run_case_page.lookup_item_notification_dispatch",
        _notification,
    )
    monkeypatch.setattr(
        "benchmark.runs.run_case_page.owners_for_run",
        _owners,
    )
    page = await build_case_page(
        service, run, page_query=CasePageQuery(page=2, page_size=1, status="ALL")
    )
    assert page["meta"]["page"] == 2
    assert page["meta"]["pageSize"] == 1
    assert page["meta"]["total"] == 3
    assert page["meta"]["owners"] == ["八鹤", "青冬"]
    data = page["data"]
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "item-1"
    assert data[0]["notification_url"] == "https://aone.example/item-1"
    assert data[0]["notification_assigned_to"] == "八鹤"
    assert data[0]["judge"]["score"] == 1.7
    service.load_case_results.assert_awaited_once()
    service.load_kafka_case_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_page_past_end_is_empty() -> None:
    service = SimpleNamespace(load_case_results=AsyncMock(return_value=[]))
    run = SimpleNamespace(run_id="run-empty", benchmark_name="ieval", config_json={})
    page = await build_case_page(
        service, run, page_query=CasePageQuery(page=3, page_size=50, status="ALL")
    )
    assert page["data"] == []
    assert page["meta"]["total"] == 0


@pytest.mark.asyncio
async def test_build_page_filters_query_and_default_fail_poor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        {
            **_fused_case("good-1", query="去机场"),
            "grade": "GOOD",
            "scores": {"fused": {"score": 1.7}},
        },
        {
            **_fused_case("fail-1", query="去西湖"),
            "grade": "FAIL",
            "scores": {"fused": {"score": 0.0}},
        },
        {
            **_fused_case("poor-1", query="去西湖夜景"),
            "grade": "POOR",
            "scores": {"fused": {"score": 0.5}},
        },
    ]
    service = SimpleNamespace(load_case_results=AsyncMock(return_value=cases))
    run = SimpleNamespace(run_id="run-filter", benchmark_name="ieval", config_json={})
    monkeypatch.setattr(
        "benchmark.runs.run_case_page.lookup_item_notification_dispatch",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "benchmark.runs.run_case_page.owners_for_run",
        AsyncMock(return_value=[]),
    )
    default_page = await build_case_page(
        service, run, page_query=CasePageQuery(page=1, page_size=10)
    )
    assert {row["id"] for row in default_page["data"]} == {"fail-1", "poor-1"}
    assert default_page["meta"]["total"] == 2

    queried = await build_case_page(
        service,
        run,
        page_query=CasePageQuery(page=1, page_size=10, query="夜景", status="FAIL,POOR"),
    )
    assert [row["id"] for row in queried["data"]] == ["poor-1"]
    assert queried["meta"]["total"] == 1


@pytest.mark.asyncio
async def test_trace_daily_uses_kafka_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = {
        "score": 0.8,
        "reason": "daily",
        "extra": {"expression": {"score": 1, "reason": "ok"}},
    }
    kafka_case = {
        "case_id": "trace-9",
        "input": "你好",
        "output": {"component": {"displayText": "你好呀"}},
        "display_input": "你好",
        "trace_url": "https://langfuse.example/trace-9",
        "scores": {"amap_agent_overall": {"judge": judge}},
        "attribution": {
            "category": "none",
            "confidence": 0.1,
            "evidence_raw": "e",
            "reasoning_raw": "r",
        },
    }
    service = SimpleNamespace(
        load_case_results=AsyncMock(return_value=[{"case_id": "should-not-use"}]),
        sample_trace_pipeline_records=AsyncMock(return_value=["trace-9"]),
        load_trace_daily_case_results=AsyncMock(return_value=[kafka_case]),
    )
    finished = datetime(2026, 11, 3, 8, 0, tzinfo=_SH)
    run = SimpleNamespace(
        run_id="amap-agent-sampling-20261102",
        benchmark_name="ba-1",
        project_id="p1",
        finished_at=finished,
        config_json={
            "trace_daily_report": True,
            "trace_report_namespace": "amap-agent-sampling",
            "source": "kafka",
            "trace_daily_sample": {
                "source": "kafka",
                "timezone": "Asia/Shanghai",
                "pass_threshold": 0.5,
                "agent_name": "amap_agent",
                "project_id": "p1",
                "criteria": [
                    {
                        "id": "amap_agent_overall",
                        "name": "综合质量",
                        "weight": 1.0,
                        "evaluator_ref": "amap_agent",
                        "description": "",
                    }
                ],
            },
        },
    )

    async def _trace_notification(case_ids: list[str]) -> dict[str, dict[str, str]]:
        assert case_ids == ["trace-9"]
        return {"trace-9": {"notification_url": "https://aone.example/trace-9", "notification_ref": "W-9"}}

    monkeypatch.setattr(
        "benchmark.runs.run_case_page.lookup_trace_notification_dispatch",
        _trace_notification,
    )
    monkeypatch.setattr(
        "benchmark.runs.run_case_page.owners_for_run",
        AsyncMock(return_value=[]),
    )
    page = await build_case_page(service, run, page_query=CasePageQuery(page=1, page_size=50))
    service.load_case_results.assert_not_awaited()
    service.sample_trace_pipeline_records.assert_awaited_once()
    service.load_trace_daily_case_results.assert_awaited_once()
    loaded = service.load_trace_daily_case_results.await_args.kwargs
    assert loaded["trace_ids"] == ["trace-9"]
    assert isinstance(loaded["rubric"][0], Criterion)
    assert loaded["rubric"][0].id == "amap_agent_overall"
    data = page["data"]
    assert isinstance(data, list)
    assert data[0]["id"] == "trace-9"
    assert data[0]["output"] == "你好呀"
    assert data[0]["judge"]["score"] == 0.8
    assert data[0]["judge"]["dimensions"][0]["id"] == "expression"
    assert data[0]["attribution"]["format"] == "table"
    assert data[0]["notification_url"] == "https://aone.example/trace-9"
    assert data[0]["trace_url"] == "https://langfuse.example/trace-9"


@pytest.mark.asyncio
async def test_pipeline_kafka_uses_kafka_case_results() -> None:
    service = SimpleNamespace(
        load_case_results=AsyncMock(),
        sample_kafka_pipeline_records=AsyncMock(return_value=["t-1"]),
        load_kafka_case_results=AsyncMock(
            return_value=[
                {
                    "case_id": "t-1",
                    "input": {"query": "q"},
                    "output": "a",
                    "display_input": "q",
                    "scores": {"fused": {"judge": {"score": 1}}},
                }
            ]
        ),
    )
    run = SimpleNamespace(
        run_id="pipe-1",
        benchmark_name="example",
        project_id="proj",
        config_json={
            "source": "pipeline-kafka",
            "date_start": "2026-11-02T00:00:00+08:00",
            "date_end": "2026-11-03T00:00:00+08:00",
            "sample_size": 10,
            "rubric": [
                {
                    "id": "D0",
                    "name": "安全与合规",
                    "weight": 0.0,
                    "evaluator_ref": "example",
                }
            ],
        },
    )
    cases, source = await load_report_cases(service, run)
    assert source == "trace"
    assert cases[0]["case_id"] == "t-1"
    service.load_case_results.assert_not_awaited()
    query = service.sample_kafka_pipeline_records.await_args.args[0]
    assert query.evaluator_names
    assert query.sample_size == 10
    assert service.load_kafka_case_results.await_args.kwargs["trace_ids"] == ["t-1"]


@pytest.mark.asyncio
async def test_load_item_notification_row_falls_back_when_fused_bucket_empty() -> None:
    """An empty fused bucket and a non-fused link → fall back on the same rule as lookup_item_notification_dispatch."""
    fused = SimpleNamespace(
        item_id="case-1",
        criterion_id=FUSED_CRITERION_ID,
        notification_dispatch_json={},
    )
    dim = SimpleNamespace(
        item_id="case-1",
        criterion_id="D0",
        notification_dispatch_json={
            "notification_ref": "W-dim",
            "notification_url": "https://aone.example/W-dim",
        },
    )
    with patch(
        "benchmark.runs.run_case_page.BenchmarkRunItem.filter",
        AsyncMock(return_value=[fused, dim]),
    ):
        row = await load_item_notification_row("run-1", "case-1")
    assert row is dim


@pytest.mark.asyncio
async def test_load_item_notification_row_prefers_fused_with_link() -> None:
    fused = SimpleNamespace(
        item_id="case-1",
        criterion_id=FUSED_CRITERION_ID,
        notification_dispatch_json={
            "notification_ref": "W-fused",
            "notification_url": "https://aone.example/W-fused",
        },
    )
    dim = SimpleNamespace(
        item_id="case-1",
        criterion_id="D0",
        notification_dispatch_json={
            "notification_ref": "W-dim",
            "notification_url": "https://aone.example/W-dim",
        },
    )
    with patch(
        "benchmark.runs.run_case_page.BenchmarkRunItem.filter",
        AsyncMock(return_value=[fused, dim]),
    ):
        row = await load_item_notification_row("run-1", "case-1")
    assert row is fused
