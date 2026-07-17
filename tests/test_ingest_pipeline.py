"""STEP-04: Sampling / Index / Write / Outbox."""

from __future__ import annotations

import pytest

from llmobserverjudge.contracts.idempotency import span_key
from llmobserverjudge.contracts.topics import Component, Topic
from llmobserverjudge.ingest.idempotency import InMemoryIdempotencyStore
from llmobserverjudge.ingest.nodes import IndexNode, IngestState, SamplingConfig, SamplingNode
from llmobserverjudge.ingest.outbox import InMemoryOutbox
from llmobserverjudge.ingest.pipeline import IngestPipeline
from llmobserverjudge.ingest.worker import IngestWorker
from llmobserverjudge.shared.mq import InMemoryMQ, create_mq
from llmobserverjudge.shared.storage import InMemorySpanStore


def _span(**kwargs) -> dict:
    base = {
        "tenant_id": "t1",
        "trace_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "span_id": "bbbbbbbbbbbbbbbb",
        "name": "chat",
        "status": "OK",
        "tags": [],
    }
    base.update(kwargs)
    return base


@pytest.mark.asyncio
async def test_error_first_sampling() -> None:
    mq = InMemoryMQ()
    node = SamplingNode(mq, config=SamplingConfig(rate=0.0))
    state = await node.run(IngestState(span=_span(status="ERROR")))
    assert state.sampled is True
    assert state.sample_reason == "error_first"


@pytest.mark.asyncio
async def test_event_tag_sampling() -> None:
    mq = InMemoryMQ()
    node = SamplingNode(mq, config=SamplingConfig(rate=0.0))
    state = await node.run(IngestState(span=_span(tags=["safety"])))
    assert state.sampled is True
    assert state.sample_reason == "event_triggered"


@pytest.mark.asyncio
async def test_lag_hysteresis_downsamples() -> None:
    mq = InMemoryMQ(simulated_lag={Topic.SPANS.value: 20_000})
    node = SamplingNode(mq, config=SamplingConfig(rate=1.0))
    # With rate capped to 0.1, most traces skipped — pick a known hash
    # Force check effective rate path
    state = await node.run(IngestState(span=_span(trace_id="0" * 32)))
    assert state.effective_sample_rate == 0.1
    assert state.consumer_lag == 20_000


@pytest.mark.asyncio
async def test_lag_low_restores_base_rate() -> None:
    mq = InMemoryMQ(simulated_lag={Topic.SPANS.value: 100})
    node = SamplingNode(mq, config=SamplingConfig(rate=0.8))
    node._last_rate = 0.1
    state = await node.run(IngestState(span=_span(trace_id="1" * 32)))
    assert state.effective_sample_rate == 0.8


@pytest.mark.asyncio
async def test_index_offloads_large_payload() -> None:
    big = "x" * 600_000
    state = IngestState(span=_span(prompt=big, response=big))
    state = await IndexNode(inline_max_bytes=1_000_000).run(state)
    assert state.storage_location is not None
    assert state.span["prompt"] == ""
    assert state.span["attributes"]["offloaded_bytes"] > 1_000_000


@pytest.mark.asyncio
async def test_write_and_outbox_only_when_sampled() -> None:
    mq = InMemoryMQ()
    store = InMemorySpanStore()
    outbox = InMemoryOutbox()
    worker = IngestWorker(
        mq,
        outbox=outbox,
        store=store,
        force_sample=False,
    )
    await mq.publish(Component.GATEWAY, Topic.SPANS, _span())
    n = await worker.process_once()
    assert n == 1
    assert await store.get_span("t1", "bbbbbbbbbbbbbbbb") is not None
    assert await outbox.fetch_unpublished() == []


@pytest.mark.asyncio
async def test_outbox_inserted_when_sampled() -> None:
    mq = InMemoryMQ()
    outbox = InMemoryOutbox()
    worker = IngestWorker(mq, outbox=outbox, force_sample=True)
    await mq.publish(
        Component.GATEWAY,
        Topic.SPANS,
        _span(tags=["quality"]),
    )
    await worker.process_once()
    rows = await outbox.fetch_unpublished()
    assert len(rows) == 1
    assert rows[0]["trace_summary"]["tags"] == ["quality"]


@pytest.mark.asyncio
async def test_idempotent_reprocess() -> None:
    mq = InMemoryMQ()
    idem = InMemoryIdempotencyStore()
    worker = IngestWorker(mq, idempotency=idem, force_sample=True)
    msg = _span()
    await mq.publish(Component.GATEWAY, Topic.SPANS, msg)
    assert await worker.process_once() == 1
    await mq.publish(Component.GATEWAY, Topic.SPANS, msg)
    assert await worker.process_once() == 0
    assert await idem.exists(span_key("t1", "bbbbbbbbbbbbbbbb"))


@pytest.mark.asyncio
async def test_outbox_merges_same_trace() -> None:
    outbox = InMemoryOutbox()
    id1 = await outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="tr",
        trace_summary={"tags": ["a"], "span_id": "1"},
    )
    id2 = await outbox.insert_trace_closed(
        tenant_id="t1",
        trace_id="tr",
        trace_summary={"tags": ["b"], "span_id": "2"},
    )
    assert id1 == id2
    rows = await outbox.fetch_unpublished()
    assert set(rows[0]["trace_summary"]["tags"]) == {"a", "b"}


def test_create_mq_factory() -> None:
    mq = create_mq("memory")
    assert isinstance(mq, InMemoryMQ)
    with pytest.raises(NotImplementedError, match="Redis Streams"):
        create_mq("rocketmq")
    with pytest.raises(ValueError):
        create_mq("kafka-unknown")


@pytest.mark.asyncio
async def test_pipeline_force_sample_compat() -> None:
    mq = InMemoryMQ()
    store = InMemorySpanStore()
    pipe = IngestPipeline(mq=mq, store=store, force_sample=True)
    state = await pipe.run(IngestState(span=_span()))
    assert state.sampled and state.written
