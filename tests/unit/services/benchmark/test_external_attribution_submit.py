"""build_record / AoneAttributionProvider.submit — params.issue + judge_result。"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
ExternalAttributionProvider = None
AttributionSubmitRequest = None
AttributionSubmitResult = None
ExternalAttributionTriggerRequest = None
build_record = None
split_attribution_issue_payload = None


def test_build_record_string_output_does_not_raise() -> None:
    """When output is a str, build_record must not call .get on the str."""
    record = build_record(
        {
            "id": "item-1",
            "item_id": "item-1",
            "input": {"query": "附近早餐"},
            "output": "agent reply text",
        },
    )
    assert record["task_id"] == "item-1"
    assert record["query"] == "附近早餐"
    assert record["response_text"] == ""


def test_build_record_string_input_and_metadata() -> None:
    """When input / metadata is not a dict, fall back to an empty object the same way."""
    record = build_record(
        {
            "item_id": "x",
            "input": "bad",
            "output": "also-bad",
            "metadata": "nope",
        },
    )
    assert record["task_id"] == "x"
    assert record["query"] == ""


def test_split_attribution_issue_payload_slim_issue() -> None:
    """A complete judge → issue has only five fields; judge_result keeps the full scores."""
    judge: dict[str, Any] = {
        "run_id": "run-1",
        "type": "judge",
        "query": "附近加油站",
        "expected": "推荐3个附近加油站",
        "actual": {"response_text": "推荐了无关餐厅"},
        "criterion_scores": {
            "D0": {"score": 1.0, "reason": "Pass", "sub_scores": None},
            "D1": {"score": 2.0, "reason": "意图准", "sub_scores": None},
            "D3": {
                "score": 0.0,
                "reason": "推荐 POI 与工具返回不一致，疑似幻觉",
                "sub_scores": None,
            },
        },
        "error": None,
    }
    issue, judge_result = split_attribution_issue_payload(judge)
    assert set(issue.keys()) == {
        "verdict",
        "dimension",
        "issue_hint",
        "expected",
        "actual",
    }
    assert issue["verdict"] == "bad"
    assert issue["dimension"] in {"D0", "D3"}
    assert "D3" in issue["issue_hint"] or "D0" in issue["issue_hint"]
    assert "criterion_scores" not in issue
    assert "D1" in judge_result["criterion_scores"]
    assert judge_result["query"] == "附近加油站"
    assert isinstance(judge_result["actual"], dict)


@pytest.mark.asyncio
async def test_attribution_submit_native_params_issue_and_judge_result() -> None:
    """params are native issue/judge_result objects, with no Rubric records."""
    mock_client = MagicMock()
    mock_client.submit_attribution = AsyncMock(
        return_value=MagicMock(
            job_id="attr-job-1",
            celery_task_id="celery-1",
            item_id="ieval_external-20260812-27:ieval_external-20260812-27:3",
            cached=False,
            deduplicated=False,
        ),
    )
    provider = ExternalAttributionProvider(external_client=mock_client)

    slim_issue = {
        "verdict": "bad",
        "dimension": "D3",
        "issue_hint": "D3: 推荐 POI 与工具返回不一致，疑似幻觉",
        "expected": "推荐3个附近加油站",
        "actual": "推荐了无关餐厅",
    }
    judge_result = {
        "run_id": "ieval_external-20260812-27",
        "type": "judge",
        "job_id": "d0566f2ef89f2010",
        "query": "附近加油站",
        "criterion_scores": {
            "D0": {"score": 1.0, "reason": "Pass"},
            "D3": {"score": 0.0, "reason": "幻觉"},
        },
        "actual": {"response_text": "推荐了无关餐厅"},
    }
    result = await provider.submit(
        AttributionSubmitRequest(
            run_id="ieval_external-20260812-27",
            item_id="ieval_external-20260812-27:3",
            job_id="d0566f2ef89f2010",
            adiu="adiu-1",
            ds_start="20260812",
            ds_end="20260812",
            trace_id="trace-abc",
            do_cross_verify=True,
            issue=slim_issue,
            judge_result=judge_result,
            callback_url="",
        )
    )
    assert isinstance(result, AttributionSubmitResult)
    assert result.job_id == "attr-job-1"
    assert result.response_item_id == "ieval_external-20260812-27:ieval_external-20260812-27:3"
    assert result.cached is False
    assert result.deduplicated is False
    request = mock_client.submit_attribution.await_args.args[0]
    assert isinstance(request, ExternalAttributionTriggerRequest)
    params = request.params
    assert not hasattr(params, "records") or getattr(params, "records", None) is None
    assert params.adiu == "adiu-1"
    assert params.ds_start == "20260812"
    assert params.ds_end == "20260812"
    assert params.trace_id == "trace-abc"
    assert params.do_cross_verify is True
    assert params.issue == slim_issue
    assert params.judge_result == judge_result
    assert isinstance(params.issue, dict)
    assert isinstance(params.judge_result, dict)
    assert params.run_id == "ieval_external-20260812-27"
    assert params.job_id == "d0566f2ef89f2010"
    assert params.item_id == "ieval_external-20260812-27:3"
    assert params.upstream_job_id == "d0566f2ef89f2010"
    assert params.upstream_pipeline == "rubric-scoring"
    assert "criterion_scores" not in params.issue


@pytest.mark.asyncio
async def test_attribution_submit_cached_response() -> None:
    """An Aone idempotent hit (completed) → cached=True, and the existing result is reused."""
    mock_client = MagicMock()
    mock_client.submit_attribution = AsyncMock(
        return_value=MagicMock(
            job_id="8454a537c8566c4e",
            celery_task_id="f28db98e-abc",
            item_id="ieval_external-20260819:116",
            cached=True,
            deduplicated=False,
        ),
    )
    provider = ExternalAttributionProvider(external_client=mock_client)

    result = await provider.submit(
        AttributionSubmitRequest(
            run_id="ieval_external-20260819",
            item_id="116",
            job_id="",
            adiu="",
            ds_start="20260819",
            issue={
                "verdict": "bad",
                "dimension": "",
                "issue_hint": "",
                "expected": "",
                "actual": "",
            },
            callback_url="",
        )
    )

    assert isinstance(result, AttributionSubmitResult)
    assert result.job_id == "8454a537c8566c4e"
    assert result.response_item_id == "ieval_external-20260819:116"
    assert result.cached is True
    assert result.deduplicated is False


@pytest.mark.asyncio
async def test_attribution_submit_deduplicated_response() -> None:
    """An Aone dedup hit (pending/running) → deduplicated=True, and the existing task is reused."""
    mock_client = MagicMock()
    mock_client.submit_attribution = AsyncMock(
        return_value=MagicMock(
            job_id="8454a537c8566c4e",
            celery_task_id="f28db98e-abc",
            item_id="ieval_external-20260819:116",
            cached=False,
            deduplicated=True,
        ),
    )
    provider = ExternalAttributionProvider(external_client=mock_client)

    result = await provider.submit(
        AttributionSubmitRequest(
            run_id="ieval_external-20260819",
            item_id="116",
            job_id="",
            adiu="",
            ds_start="20260819",
            issue={
                "verdict": "bad",
                "dimension": "",
                "issue_hint": "",
                "expected": "",
                "actual": "",
            },
            callback_url="",
        )
    )

    assert isinstance(result, AttributionSubmitResult)
    assert result.job_id == "8454a537c8566c4e"
    assert result.response_item_id == "ieval_external-20260819:116"
    assert result.cached is False
    assert result.deduplicated is True
