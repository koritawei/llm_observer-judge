"""Local sidecar observation tree must not call Langfuse."""

from unittest.mock import AsyncMock, MagicMock

import pytest
PreprocessNode = None
EVAL_CASE = None
REQUEST = None
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.trace.tools import (
    _SIDECAR_OBSERVATIONS_KEY,
    TraceToolProvider,
    bind_sidecar_observations,
)


@pytest.mark.asyncio
async def test_preprocess_uses_sidecar_observations_without_trace() -> None:
    provider = MagicMock(spec=TraceToolProvider)
    provider.call_tool = AsyncMock()
    node = PreprocessNode(trace_provider=provider)
    patch = await node(
        {
            REQUEST: EvaluatorRequest(
                input={"query": "附近咖啡"},
                output={"component": {"text": "去星巴克"}},
                metadata={
                    "pipeline_source": "sidecar",
                    _SIDECAR_OBSERVATIONS_KEY: [
                        {
                            "span_id": "tool-1",
                            "name": "search_poi",
                            "type": "TOOL",
                            "input": {"keywords": "咖啡"},
                            "output": {"name": "星巴克"},
                        }
                    ],
                },
                trace_id="trace-1",
                project_id="project-1",
                skip_trace_resolve=True,
            )
        }
    )
    provider.call_tool.assert_not_awaited()
    case = patch[EVAL_CASE]
    assert case.query == "附近咖啡"
    assert "search_poi" in case.trace_messages_json
    assert "星巴克" in case.trace_messages_json


@pytest.mark.asyncio
async def test_preprocess_uses_contextvar_without_metadata_tree() -> None:
    provider = MagicMock(spec=TraceToolProvider)
    provider.call_tool = AsyncMock()
    node = PreprocessNode(trace_provider=provider)
    observations = [
        {
            "span_id": "tool-1",
            "name": "search_poi",
            "type": "TOOL",
            "input": {"keywords": "咖啡"},
            "output": {"name": "星巴克"},
        }
    ]
    with bind_sidecar_observations(observations):
        patch = await node(
            {
                REQUEST: EvaluatorRequest(
                    input={"query": "附近咖啡"},
                    output={"component": {"text": "去星巴克"}},
                    metadata={"pipeline_source": "sidecar"},
                    trace_id="trace-1",
                    project_id="project-1",
                    skip_trace_resolve=True,
                )
            }
        )
    provider.call_tool.assert_not_awaited()
    case = patch[EVAL_CASE]
    assert "search_poi" in case.trace_messages_json
