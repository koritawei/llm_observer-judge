"""Unit tests for the SubmitPipeline unified submit pipeline (design doc §7.2).

Locked acceptance points (§12):
- no interceptor = pass the full KV through (§7.4 framework default)
- an interceptor that raises or emits illegal output → item-level failure (the failure callback is persisted), the run is not blocked,
  and the item does not enter the queue
- the interceptor product = the enqueued content (the only channel through which downstream consumes the parsed product)
"""

from typing import Any

import pytest
from benchmark.runs.common.parse_interceptor import (
    DataParseContext,
    validate_business_item,
)
from benchmark.runs.common.profile_resolver import (
    ResolvedAttribution,
    ResolvedJudge,
    ResolvedProfile,
)
from benchmark.runs.common.submit_pipeline import (
    SubmitPipelineSpec,
    submit_prepared_items,
)
from benchmark.reporting.framework.schema import AggregationConfig, Criterion


def _resolved(interceptor: object | None) -> ResolvedProfile:
    return ResolvedProfile(
        benchmark_name="demo",
        rubric=[Criterion(id="Q1", name="q", description="", weight=1.0, evaluator_ref="e")],
        rubric_ref=None,
        aggregation=AggregationConfig(method="weighted_avg", pass_threshold=0.5),
        parse_interceptor_ref="x:Y" if interceptor is not None else None,
        parse_interceptor=interceptor,  # type: ignore[arg-type]
        judge=ResolvedJudge(
            execution="sync",
            target=None,
            provider="local",
            dims="Q1",
            prompt_version=None,
            local_eval_class=None,
        ),
        attribution=ResolvedAttribution(
            execution="none",
            provider="none",
            target=None,
            do_cross_verify=True,
            local_attr_class=None,
        ),
    )


def _raw(i: int) -> dict[str, Any]:
    return {"id": f"t{i}", "item_id": f"run:{i}", "metadata": {"csv_row": {"task_id": f"t{i}"}}}


class _Collector:
    def __init__(self) -> None:
        self.enqueued: list[dict[str, Any]] = []
        self.failed: list[tuple[dict[str, Any], str]] = []

    async def enqueue(self, items: list[dict[str, Any]]) -> int:
        self.enqueued.extend(items)
        return len(items)

    async def mark_failed(self, item: dict[str, Any], reason: str) -> None:
        self.failed.append((item, reason))


class _IdentityInterceptor:
    def parse(self, ctx: DataParseContext) -> list[dict[str, Any]]:
        out = []
        for raw in ctx.raw_items:
            out.append(
                {
                    "id": raw["id"],
                    "item_id": raw["item_id"],
                    "input": {"query": raw["metadata"]["csv_row"].get("task_id")},
                    "output": {"response_text": "r"},
                    "metadata": dict(raw["metadata"]),
                },
            )
        return out


class _BoomOnSecondInterceptor:
    def parse(self, ctx: DataParseContext) -> list[dict[str, Any]]:
        raw = ctx.raw_items[0]
        if raw["id"] == "t1":
            raise ValueError("boom")
        return _IdentityInterceptor().parse(ctx)


class _BadOutputInterceptor:
    def parse(self, ctx: DataParseContext) -> list[dict[str, Any]]:
        return [{"id": "x"}]  # Missing item_id/input/output/metadata → illegal contract


@pytest.mark.asyncio
async def test_passthrough_without_interceptor() -> None:
    """No interceptor → pass the full KV through (§7.4): when csv_row has no generic columns, pass it through as-is."""
    collector = _Collector()
    raws = [_raw(0), _raw(1)]
    summary = await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(None),
            raw_items=raws,
            enqueue_items=collector.enqueue,
            mark_item_failed=collector.mark_failed,
        ),
    )
    assert summary.total_raw == 2
    assert summary.parsed == 2
    assert summary.parse_failed == 0
    assert summary.enqueued == 2
    # csv_row has no input/output columns → content stays as-is (copy only; do not append missing keys)
    assert [i.get("id") for i in collector.enqueued] == ["t0", "t1"]
    assert all("input" not in item for item in collector.enqueued)
    assert collector.failed == []


@pytest.mark.asyncio
async def test_passthrough_non_dict_item_is_marked_failed() -> None:
    """A non-dict passthrough matches the interceptor branch: the failure stub advances the count and is not counted as parsed."""
    collector = _Collector()
    summary = await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(None),
            raw_items=[_raw(0), ["not-a-dict"], _raw(2)],
            enqueue_items=collector.enqueue,
            mark_item_failed=collector.mark_failed,
        ),
    )

    assert summary.total_raw == 3
    assert summary.parsed == 2
    assert summary.parse_failed == 1
    assert summary.enqueued == 2
    assert [item["id"] for item in collector.enqueued] == ["t0", "t2"]
    assert collector.failed == [
        (
            {"id": "run:1", "item_id": "run:1"},
            "raw item must be dict, got list",
        )
    ]


@pytest.mark.asyncio
async def test_passthrough_lifts_common_csv_columns() -> None:
    """No interceptor + csv_row contains input/output/expected_output (the webUI upload

    contract) → generic columns are promoted to the top level, so prewritten run_items are no longer input/output=None.
    """
    collector = _Collector()
    raws = [
        {
            "id": "t0",
            "item_id": "run:0",
            "metadata": {
                "csv_row": {
                    "input": "附近咖啡",
                    "output": "去星巴克",
                    "expected_output": "星巴克",
                    "traceid": "tr-1",
                },
            },
        },
    ]
    await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(None),
            raw_items=raws,
            enqueue_items=collector.enqueue,
            mark_item_failed=collector.mark_failed,
        ),
    )
    item = collector.enqueued[0]
    assert item["input"] == "附近咖啡"
    assert item["output"] == "去星巴克"
    assert item["expected_output"] == "星巴克"
    # metadata.csv_row is kept as-is (the full KV remains the only source of truth)
    assert item["metadata"]["csv_row"]["traceid"] == "tr-1"
    # An existing top-level key is not overwritten by csv_row (priority: business top-level > promoted column)
    raws[0]["output"] = {"preset": True}
    collector2 = _Collector()
    await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(None),
            raw_items=raws,
            enqueue_items=collector2.enqueue,
            mark_item_failed=collector2.mark_failed,
        ),
    )
    assert collector2.enqueued[0]["output"] == {"preset": True}


@pytest.mark.asyncio
async def test_passthrough_lifts_project_and_trace_id() -> None:
    """With no interceptor, promote csv project/trace_id to metadata.project_id / source_trace_id.

    Aligned with search_eval parse, so passthrough chains such as example_benchmark can pull the trace.
    """
    collector = _Collector()
    raws = [
        {
            "id": "c1",
            "item_id": "run:0",
            "metadata": {
                "csv_row": {
                    "input": '{"request_body":{"query":"q"}}',
                    "output": '{"component":{"text":"a"}}',
                    "project": "cmpwln0qs04fo6j8eryfdgznb",
                    "trace_id": "tr-abc",
                },
            },
        },
    ]
    await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(None),
            raw_items=raws,
            enqueue_items=collector.enqueue,
            mark_item_failed=collector.mark_failed,
        ),
    )
    item = collector.enqueued[0]
    assert item["source_trace_id"] == "tr-abc"
    assert item["metadata"]["project_id"] == "cmpwln0qs04fo6j8eryfdgznb"
    assert item["metadata"]["csv_row"]["project"] == "cmpwln0qs04fo6j8eryfdgznb"
    # An existing identity is not overwritten
    raws[0]["source_trace_id"] = "preset-tr"
    raws[0]["metadata"]["project_id"] = "preset-proj"
    collector2 = _Collector()
    await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(None),
            raw_items=raws,
            enqueue_items=collector2.enqueue,
            mark_item_failed=collector2.mark_failed,
        ),
    )
    assert collector2.enqueued[0]["source_trace_id"] == "preset-tr"
    assert collector2.enqueued[0]["metadata"]["project_id"] == "preset-proj"


@pytest.mark.asyncio
async def test_interceptor_output_enqueued() -> None:
    """The interceptor's parsed product = the enqueued content (the business structure includes input/output)."""
    collector = _Collector()
    summary = await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(_IdentityInterceptor()),
            raw_items=[_raw(0)],
            enqueue_items=collector.enqueue,
            mark_item_failed=collector.mark_failed,
        ),
    )
    assert summary.enqueued == 1
    assert collector.enqueued[0]["input"] == {"query": "t0"}
    assert collector.enqueued[0]["output"] == {"response_text": "r"}


@pytest.mark.asyncio
async def test_parse_failure_isolated_per_item() -> None:
    """An interceptor exception fails only that one item: the failure callback fires, it is not enqueued, and the run continues (§7.6)."""
    collector = _Collector()
    summary = await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(_BoomOnSecondInterceptor()),
            raw_items=[_raw(0), _raw(1), _raw(2)],
            enqueue_items=collector.enqueue,
            mark_item_failed=collector.mark_failed,
        ),
    )
    assert summary.total_raw == 3
    assert summary.parsed == 2
    assert summary.parse_failed == 1
    assert summary.enqueued == 2
    # The failed item hits only t1; the rest enqueue normally
    assert [item["id"] for item, _ in collector.failed] == ["t1"]
    assert "boom" in collector.failed[0][1]
    assert [item["id"] for item in collector.enqueued] == ["t0", "t2"]


@pytest.mark.asyncio
async def test_invalid_output_contract_item_failed() -> None:
    """Interceptor output missing a required field → item-level failure (§7.6 contract check)."""
    collector = _Collector()
    summary = await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(_BadOutputInterceptor()),
            raw_items=[_raw(0)],
            enqueue_items=collector.enqueue,
            mark_item_failed=collector.mark_failed,
        ),
    )
    assert summary.parse_failed == 1
    assert summary.enqueued == 0
    assert "missing required field" in collector.failed[0][1]


@pytest.mark.asyncio
async def test_empty_input_no_enqueue_call() -> None:
    collector = _Collector()
    summary = await submit_prepared_items(
        SubmitPipelineSpec(
            run_id="run",
            resolved=_resolved(_IdentityInterceptor()),
            raw_items=[],
            enqueue_items=collector.enqueue,
            mark_item_failed=collector.mark_failed,
        ),
    )
    assert summary.enqueued == 0


def test_validate_business_item_contract() -> None:
    """Business item output contract: id/item_id/input/output/metadata are required."""
    assert (
        validate_business_item(
            {
                "id": "a",
                "item_id": "b",
                "input": {},
                "output": {},
                "metadata": {},
            },
        )
        is None
    )
    assert validate_business_item({"id": "a"}) is not None
    assert validate_business_item("not-a-dict") is not None
    assert (
        validate_business_item(
            {"id": "", "item_id": "b", "input": {}, "output": {}, "metadata": {}},
        )
        is not None
    )
