"""Use the real v2 client and field-projected HTTP responses for collection."""

import httpx  # noqa: TID251 - test transport for the real client contract
from benchmark.settings.trace_endpoints import TraceEndpoint
from benchmark.reporting.cluster_service.collector import TracePerfCollector
from benchmark.trace.http_backend import TraceHttpBackend


def _projected_response(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/public/v2/observations"
    groups = set(request.url.params["fields"].split(","))
    generation = request.url.params.get("type") == "GENERATION"
    row = {
        "id": "gen" if generation else "root",
        "traceId": "trace-1",
        "type": "GENERATION" if generation else "SPAN",
        "name": "AmapChatModel.chat",
        "startTime": "2026-10-30T01:00:00Z",
        "tags": ["standard"],
    }
    if "io" in groups:
        row["input"] = {"query": "导航回家"}
    if "metadata" in groups:
        row["metadata"] = {"first_text_response_time_ms": 123.0}
    if "metrics" in groups:
        row["latency"] = 0.5 if generation else 1.5
        row["calculatedTotalCost"] = 0.1
    if "basic" in groups:
        row["model"] = "test-model"
    if "usage" in groups:
        row["usageDetails"] = {"input": 10, "output": 3, "total": 13}
    return httpx.Response(200, json={"data": [row]})


async def test_real_client_requests_metric_and_usage_fields() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _projected_response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = TraceHttpBackend(
            TraceEndpoint(
                base_url="https://lf.example.com", public_key="test-pk", secret_key="test-sk"
            ),
            http,
            max_retries=0,
        )
        perf = (
            await TracePerfCollector(client=client, min_interval_s=0).fetch_perf(["trace-1"])
        )["trace-1"]
    assert perf.error == ""
    assert perf.e2e_ms == 1500.0
    assert perf.first_text_ms == 123.0
    assert perf.buckets["react"].total_tokens == 13
    assert perf.models["test-model"].total_tokens == 13
    root, generation = requests
    assert root.url.params.get("expandMetadata") == "first_text_response_time_ms"
    assert set(generation.url.params["fields"].split(",")) >= {
        "core",
        "basic",
        "io",
        "usage",
        "metrics",
    }


async def test_real_client_input_reads_request_io_group() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_projected_response)) as http:
        client = TraceHttpBackend(
            TraceEndpoint(
                base_url="https://lf.example.com", public_key="test-pk", secret_key="test-sk"
            ),
            http,
            max_retries=0,
        )
        inputs = await TracePerfCollector(client=client, min_interval_s=0).fetch_inputs(
            ["trace-1"]
        )
    assert inputs == {"trace-1": {"query": "导航回家"}}
