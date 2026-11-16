"""GET /v1/benchmarks/runs/{run_id}/cases"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.routes import benchmark_run_cases as api
from benchmark.routes.benchmark_run_cases import (
    BenchmarkRunCasesQuery,
    BenchmarkRunCasesResponse,
)
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_list_run_cases_404_when_missing() -> None:
    svc = MagicMock()
    svc.get_run_by_id_or_pk = AsyncMock(return_value=None)
    with pytest.raises(HTTPException) as exc_info:
        await api.list_run_cases(
            run_id="missing",
            benchmark_service=svc,
            params=BenchmarkRunCasesQuery(),
        )
    assert exc_info.value.status_code == 404
    assert "missing" in exc_info.value.detail


@pytest.mark.asyncio
async def test_list_run_cases_accepts_row_pk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The WebUI "view all cases" action passes the row primary-key UUID, which must resolve to a run."""
    run = SimpleNamespace(run_id="biz-run-9")
    svc = MagicMock()
    svc.get_run_by_id_or_pk = AsyncMock(return_value=run)
    payload = {
        "data": [],
        "meta": {
            "page": 1,
            "pageSize": 50,
            "total": 0,
            "owners": [],
            "notificationSource": None,
            "supportsAttributionCategory": True,
            "attributionCategories": [],
            "reportUrl": None,
        },
    }

    async def _page(*_args: object, **kwargs: object) -> dict:
        page_query = kwargs["page_query"]
        assert page_query.page == 1
        assert page_query.page_size == 50
        return payload

    monkeypatch.setattr(api, "build_case_page", _page)
    pk = "fbc90c19-5923-4de6-9437-78efe17b1d07"
    resp = await api.list_run_cases(
        run_id=pk,
        benchmark_service=svc,
        params=BenchmarkRunCasesQuery(page=1, page_size=50),
    )
    assert isinstance(resp, BenchmarkRunCasesResponse)
    svc.get_run_by_id_or_pk.assert_awaited_once_with(pk)


@pytest.mark.asyncio
async def test_list_run_cases_returns_page_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    run = SimpleNamespace(run_id="run-9")
    svc = MagicMock()
    svc.get_run_by_id_or_pk = AsyncMock(return_value=run)
    payload = {
        "data": [
            {
                "id": "item-1",
                "query": "从机场怎么走",
                "attribution_category": "tool/tool_unavailable",
                "output": "沿机场高速",
                "judge": {
                    "score": 1.72,
                    "reason": "ok",
                    "dimensions": [{"id": "D0", "score": 2.0, "reason": "pass"}],
                },
                "attribution": {
                    "format": "markdown",
                    "markdown": "## 根因\n\ntool_timeout",
                },
                "trace_url": "https://trace.example/1",
                "notification_url": "https://aone.example/1",
            }
        ],
        "meta": {
            "page": 1,
            "pageSize": 50,
            "total": 1,
            "owners": [],
            "notificationSource": "items",
            "supportsAttributionCategory": True,
            "attributionCategories": ["tool/tool_unavailable"],
            "reportUrl": "https://pages.example/report.html",
        },
    }

    async def _page(*_args: object, **kwargs: object) -> dict:
        page_query = kwargs["page_query"]
        assert page_query.page == 1
        assert page_query.page_size == 20
        assert page_query.attribution_category == "tool/tool_unavailable"
        return payload

    monkeypatch.setattr(api, "build_case_page", _page)
    resp = await api.list_run_cases(
        run_id="run-9",
        benchmark_service=svc,
        params=BenchmarkRunCasesQuery(
            page=1,
            page_size=20,
            attribution_category="tool/tool_unavailable",
        ),
    )
    assert isinstance(resp, BenchmarkRunCasesResponse)
    body = resp.model_dump(by_alias=True)
    assert body["meta"]["supportsAttributionCategory"] is True
    assert body["meta"]["attributionCategories"] == ["tool/tool_unavailable"]
    assert body["meta"]["reportUrl"] == "https://pages.example/report.html"
    assert body["data"][0]["attribution_category"] == "tool/tool_unavailable"
    assert body["data"][0]["judge"]["score"] == 1.72
    assert body["data"][0]["notification_url"] == "https://aone.example/1"
    svc.get_run_by_id_or_pk.assert_awaited_once_with("run-9")


@pytest.mark.asyncio
async def test_get_run_case_404_when_missing_run() -> None:
    svc = MagicMock()
    svc.get_run_by_id_or_pk = AsyncMock(return_value=None)
    with pytest.raises(HTTPException) as exc_info:
        await api.get_run_case(
            run_id="missing",
            case_id="c1",
            benchmark_service=svc,
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_run_case_404_when_missing_case(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = MagicMock()
    svc.get_run_by_id_or_pk = AsyncMock(return_value=SimpleNamespace(run_id="run-9"))

    async def _detail(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(api, "build_case_detail", _detail)
    with pytest.raises(HTTPException) as exc_info:
        await api.get_run_case(
            run_id="run-9",
            case_id="gone",
            benchmark_service=svc,
        )
    assert exc_info.value.status_code == 404
    assert "gone" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_run_case_returns_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = MagicMock()
    svc.get_run_by_id_or_pk = AsyncMock(return_value=SimpleNamespace(run_id="run-9"))
    payload = {
        "data": {
            "id": "item-1",
            "query": "从机场怎么走",
            "output": "沿机场高速",
            "input": '{\n  "query": "从机场怎么走"\n}',
            "expected_output": "应推荐机场大巴",
            "judge": None,
            "attribution": None,
            "trace_url": "https://trace.example/1",
            "notification_url": None,
            "notification_ref": None,
            "notification_assigned_to": None,
        },
        "meta": {
            "owners": ["八鹤"],
            "notificationSource": "items",
            "supportsAttributionCategory": True,
            "reportUrl": "https://pages.example/report.html",
        },
    }

    async def _detail(*_args: object, **kwargs: object) -> dict:
        assert kwargs["detail_query"].case_id == "item-1"
        return payload

    monkeypatch.setattr(api, "build_case_detail", _detail)
    resp = await api.get_run_case(
        run_id="run-9",
        case_id="item-1",
        benchmark_service=svc,
    )
    body = resp.model_dump(by_alias=True)
    assert body["data"]["id"] == "item-1"
    assert body["data"]["expectedOutput"] == "应推荐机场大巴"
    assert "query" in body["data"]["input"]
    assert body["meta"]["owners"] == ["八鹤"]


@pytest.mark.asyncio
async def test_report_summary_404_without_url() -> None:
    svc = MagicMock()
    svc.get_run_by_id_or_pk = AsyncMock(
        return_value=SimpleNamespace(
            run_id="run-1",
            report_meta_json={"report_url": "https://oss.example/only.html"},
        ),
    )
    with pytest.raises(HTTPException) as exc_info:
        await api.get_run_report_summary(run_id="run-1", benchmark_service=svc)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_report_summary_fetches_report_site_url(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = MagicMock()
    svc.get_run_by_id_or_pk = AsyncMock(
        return_value=SimpleNamespace(
            run_id="run-1",
            report_meta_json={
                "report_site_url": "https://pages.example/r.html",
                "report_url": "https://oss.example/r.html",
            },
        ),
    )
    fetch = AsyncMock(return_value="<html>summary</html>")
    monkeypatch.setattr(api, "fetch_report_summary_html", fetch)
    resp = await api.get_run_report_summary(run_id="run-1", benchmark_service=svc)
    assert resp.body == b"<html>summary</html>"
    fetch.assert_awaited_once_with("https://pages.example/r.html")
