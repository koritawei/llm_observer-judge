from benchmark.pipeline.profiles import MetricBinding
from benchmark.services.metrics import TraceMetricCollector


class _Resolver:
    async def fetch_metric_inputs(self, *, project_id: str, primary_id: str):
        assert project_id == "project"
        assert primary_id == "trace"
        return (
            {
                "latency": 12.2,
                "metadata": {"first_text_response_time_ms": 3200},
            },
            [
                {
                    "name": "AmapChatModel.chat",
                    "promptTokens": 100,
                    "completionTokens": 30,
                    "latency": 2.5,
                }
            ],
        )


async def test_trace_metric_collector_normalizes_latency_and_react_scope() -> None:
    observations = await TraceMetricCollector().collect(
        resolver=_Resolver(),  # type: ignore[arg-type]
        project_id="project",
        trace_id="trace",
        bindings={
            "first_packet_latency_ms": MetricBinding(
                collector="vehicle_trace",
                source_field="trace.metadata.first_text_response_time_ms",
                threshold=3000,
            ),
            "total_latency_ms": MetricBinding(
                collector="vehicle_trace",
                source_field="trace.latency",
                threshold=10000,
            ),
            "llm_token_count": MetricBinding(
                collector="vehicle_trace",
                source_field="observations[type=GENERATION]",
                threshold=100,
            ),
        },
    )

    by_key_scope = {(item.metric_key, item.sample_scope): item for item in observations}
    assert by_key_scope[("first_packet_latency_ms", None)].status == "exceeded"
    assert by_key_scope[("total_latency_ms", None)].value == 12200
    react_tokens = by_key_scope[("llm_token_count", "react")]
    assert react_tokens.value == 130
    assert react_tokens.status == "exceeded"
