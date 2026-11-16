"""aone_client HTTP behavior — log redaction / job status query / URL normalization.

query / response / records / judge_result / issue / full_tool_calls_json
may contain user input and model output, and must be replaced with a length-bearing placeholder when logged;
non-sensitive reconciliation fields (ds / run_id / item_id and so on) stay in clear text for debugging.

query_job_status (404 → not_found terminal state / normal parse / HTTPError re-raised)
and base_url normalization (strip an ``/api/pipelines`` suffix) live on the shared base class, so
the judge and attribution clients must behave the same.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.runs.common.external_client import (
    _REDACT_MAX_DEPTH,
    ExternalAttributionClient,
    ExternalAttributionParams,
    ExternalAttributionTriggerRequest,
    ExternalJobStatusResponse,
    ExternalJudgeClient,
    ExternalPipelineConfig,
    ExternalRubricParams,
    ExternalRubricTriggerRequest,
    _redact_for_log,
)
from benchmark.utils.http_transport import HTTPError


def test_redact_sensitive_fields_keep_structure() -> None:
    body = {
        "params": {
            "query": "secret-user-query",
            "response": "secret-model-response",
            "records": [{"query": "q1"}, {"query": "q2"}],
            "judge_result": {"score": 1, "actual": "secret-actual"},
            "issue": {"verdict": "bad"},
            "full_tool_calls_json": '[{"secret": true}]',
            "ds": "20260812",
            "run_id": "ieval_external-20260812-01",
        }
    }
    out = _redact_for_log(body)
    text = json.dumps(out, ensure_ascii=False)
    # A sensitive value never appears in clear text
    assert "secret" not in text
    # The placeholder keeps the length
    assert "<redacted chars=17>" in text  # query
    assert "<redacted items=2>" in text  # records
    assert "<redacted keys=2>" in text  # judge_result
    # Non-sensitive reconciliation fields stay in clear text
    assert "20260812" in text
    assert "ieval_external-20260812-01" in text


def test_redact_nested_and_passthrough() -> None:
    """Nested dict/list are handled recursively; a scalar is returned as-is."""
    body = {"params": {"nested": {"query": "hidden"}, "limit": 3}}
    out = _redact_for_log(body)
    assert out == {"params": {"nested": {"query": "<redacted chars=6>"}, "limit": 3}}
    assert _redact_for_log(42) == 42
    assert _redact_for_log([{"response": "x"}]) == [{"response": "<redacted chars=1>"}]


def test_redact_deep_nesting_placeholder() -> None:
    """Nesting deeper than _REDACT_MAX_DEPTH is not descended into; the whole value is a placeholder so the stack cannot overflow."""
    body: dict[str, object] = {"ds": "20260812"}
    for _ in range(_REDACT_MAX_DEPTH + 10):
        body = {"nested": body}
    out = _redact_for_log(body)
    text = json.dumps(out, ensure_ascii=False)
    assert "<redacted keys=1>" in text  # The cut-off layer is one placeholder
    assert "20260812" not in text  # Content below the cut-off layer is not expanded
    # Depth cap: the outer nested wrapping is exactly _REDACT_MAX_DEPTH layers
    depth = 0
    node: object = out
    while isinstance(node, dict):
        node = node["nested"]
        depth += 1
    assert depth == _REDACT_MAX_DEPTH


def test_redact_self_referencing_placeholder() -> None:
    """A self-reference cycle does not recurse forever; past the limit the whole value is a placeholder."""
    loop: dict[str, object] = {"ds": "20260812"}
    loop["self"] = loop
    out = _redact_for_log(loop)
    text = json.dumps(out, ensure_ascii=False)
    assert "20260812" in text  # A top-level non-sensitive field is kept
    assert "<redacted keys=2>" in text  # The cycle is cut to a placeholder at the depth cap


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.text = json.dumps(payload)
    resp.headers = {}
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock()
    return resp


def _patch_http(resp: MagicMock):
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)
    http.get = AsyncMock(return_value=resp)
    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(return_value=http)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return cls


def _both_clients() -> tuple[ExternalJudgeClient, ExternalAttributionClient]:
    """Both the judge and attribution clients share the base-class query_job_status, and their behavior must match."""
    return (
        ExternalJudgeClient(
            ExternalPipelineConfig(
                base_url="https://aone.test",
                api_key="k",
                pipeline_id="p1",
            ),
        ),
        ExternalAttributionClient(
            ExternalPipelineConfig(
                base_url="https://aone.test",
                api_key="k",
                pipeline_id="p1",
            ),
        ),
    )


@pytest.mark.parametrize(
    "client_cls",
    [ExternalJudgeClient, ExternalAttributionClient],
)
def test_base_url_strips_pipelines_suffix(
    client_cls: type[ExternalJudgeClient] | type[ExternalAttributionClient],
) -> None:
    """When base_url already has an /api/pipelines suffix it is trimmed, and both clients use the same rule."""
    client = client_cls(
        ExternalPipelineConfig(
            base_url="https://aone.test/api/pipelines",
            api_key="k",
            pipeline_id="p1",
        ),
    )
    assert client._base_url == "https://aone.test"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_idx", [0, 1])
async def test_query_job_status_not_found_returns_terminal(client_idx: int) -> None:
    """404 → the not_found terminal state (raise_for_status is not called), and both clients behave the same."""
    client = _both_clients()[client_idx]
    resp = MagicMock()
    resp.status_code = 404
    resp.text = "{}"
    resp.headers = {}
    with patch("benchmark.runs.common.external_client.AsyncClient", _patch_http(resp)):
        out = await client.query_job_status("missing-job")
    assert out.status == "not_found"
    assert out.job_id == "missing-job"
    assert "404" in (out.error or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("client_idx", [0, 1])
async def test_query_job_status_ok_parses_row(client_idx: int) -> None:
    """A normal response parses the whole jobs-table row (status/progress/result)."""
    client = _both_clients()[client_idx]
    payload = {
        "job_id": "j1",
        "status": "completed",
        "progress": {"current": 3, "total": 3, "stage": "done"},
        "result": {"score": 1.5},
    }
    resp = _mock_response(payload)
    with patch("benchmark.runs.common.external_client.AsyncClient", _patch_http(resp)):
        out = await client.query_job_status("j1")
    assert out == ExternalJobStatusResponse.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_idx", [0, 1])
async def test_query_job_status_http_error_reraised(client_idx: int) -> None:
    """A network error is logged as warning and then re-raised as-is (not swallowed and not translated)."""
    client = _both_clients()[client_idx]
    http = MagicMock()
    http.get = AsyncMock(side_effect=HTTPError("boom"))
    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(return_value=http)
    cls.return_value.__aexit__ = AsyncMock(return_value=False)
    with (
        patch("benchmark.runs.common.external_client.AsyncClient", cls),
        pytest.raises(HTTPError),
    ):
        await client.query_job_status("j1")


@pytest.mark.asyncio
async def test_submit_attribution_log_redacts_query_response() -> None:
    client = ExternalAttributionClient(
        ExternalPipelineConfig(
            base_url="https://aone.test",
            api_key="k",
            pipeline_id="p1",
        ),
    )
    request = ExternalAttributionTriggerRequest(
        params=ExternalAttributionParams(
            query="secret-user-query",
            response="secret-model-response",
            adiu="d1",
            ds_start="20260812",
            judge_result={"score": 0},
            issue={"verdict": "bad"},
        )
    )
    cls = _patch_http(_mock_response({"job_id": "j1"}))
    with (
        patch("benchmark.runs.common.external_client.AsyncClient", cls),
        patch("benchmark.runs.common.external_client.logger") as log,
    ):
        await client.submit_attribution(request)
    logged = " ".join(str(c) for c in log.info.call_args_list)
    assert "secret-user-query" not in logged
    assert "secret-model-response" not in logged
    assert "<redacted chars=17>" in logged
    assert "20260812" in logged  # ds_start is kept


@pytest.mark.asyncio
async def test_submit_evaluation_log_redacts_records() -> None:
    client = ExternalJudgeClient(
        ExternalPipelineConfig(
            base_url="https://aone.test",
            api_key="k",
            pipeline_id="p1",
        ),
    )
    request = ExternalRubricTriggerRequest(
        params=ExternalRubricParams(
            records=[{"task_id": "t1", "query": "secret-in-record"}],
            ds="20260812",
            run_id="ieval_external-20260812-01",
        )
    )
    cls = _patch_http(_mock_response({"job_id": "j1"}))
    with (
        patch("benchmark.runs.common.external_client.AsyncClient", cls),
        patch("benchmark.runs.common.external_client.logger") as log,
    ):
        await client.submit_evaluation(request)
    logged = " ".join(str(c) for c in log.info.call_args_list)
    assert "secret-in-record" not in logged
    assert "<redacted items=1>" in logged
    assert "ieval_external-20260812-01" in logged  # run_id is kept


@pytest.mark.asyncio
async def test_submit_batch_attribution_log_keeps_record_count() -> None:
    """The batch-submit log keeps records=N (for reconciliation), and the records contents stay redacted."""
    client = ExternalAttributionClient(
        ExternalPipelineConfig(
            base_url="https://aone.test",
            api_key="k",
            pipeline_id="p1",
        ),
    )
    request = ExternalRubricTriggerRequest(
        params=ExternalRubricParams(
            records=[
                {"task_id": "t1", "query": "secret-in-record"},
                {"task_id": "t2", "query": "secret-2"},
            ],
            ds="20260812",
        )
    )
    cls = _patch_http(_mock_response({"job_id": "j1"}))
    with (
        patch("benchmark.runs.common.external_client.AsyncClient", cls),
        patch("benchmark.runs.common.external_client.logger") as log,
    ):
        await client.submit_batch_attribution(request)
    # The logger is mocked, so %d is not formatted — lock it from the call arguments: fmt contains records=%d and the argument is the row count
    req_calls = [c for c in log.info.call_args_list if "http_request" in str(c.args[0])]
    assert len(req_calls) == 1
    fmt, *fmt_args = req_calls[0].args
    assert "records=%d" in fmt
    assert 2 in fmt_args
    logged = " ".join(str(c) for c in log.info.call_args_list)
    assert "secret-in-record" not in logged
    assert "secret-2" not in logged
    assert "<redacted items=2>" in logged
    assert "20260812" in logged  # ds is kept
