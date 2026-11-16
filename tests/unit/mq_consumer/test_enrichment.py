"""enrichment-layer unit tests — generic enrich dispatch plus luoshu Langfuse enricher degradation/merge.

Generic dispatch claims an enricher by :attr:`BaseTaskMessage.source` (driven by StubTaskMessage's metadata.source).
The luoshu enricher handles only :class:`LuoshuTaskMessage`. It pulls a Langfuse skeleton by project_id + trace_id and
hangs it on the ``langfuse`` field (exposed via metadata). Every degradation path returns the message as-is.
"""

import benchmark.ingest.mq.consumers.luoshu.enrichment as luoshu_enrichment
from benchmark.ingest.mq.consumers.luoshu.enrichment import LuoshuTraceEnricher
from benchmark.ingest.mq.consumers.luoshu.schema import (
    STATUS_NO_IDS,
    STATUS_QUERY_ERROR,
    STATUS_TRACE_MISS,
    STATUS_UNKNOWN_PROJECT,
    LuoshuTaskMessage,
)
from benchmark.ingest.mq.enrichment import _ENRICHERS, enrich, register_enricher
from benchmark.ingest.mq.protocol import BaseTaskMessage
from tests.unit.mq_consumer.stubs import StubTaskMessage


class _FakeEnricher:
    source = "fake_src"

    def __init__(self) -> None:
        self.calls: list[BaseTaskMessage] = []

    async def enrich(self, message: StubTaskMessage) -> StubTaskMessage:
        self.calls.append(message)
        return message.model_copy(
            update={"metadata": {**message.metadata, "enriched": True}}
        )


# ── generic dispatch ──────────────────────────────────────────────────────────────


async def test_enrich_dispatches_by_source() -> None:
    fake = _FakeEnricher()
    register_enricher(fake)
    try:
        msg = StubTaskMessage(metadata={"source": "fake_src"}, trace_id="t")
        out = await enrich(msg)
        assert out.metadata.get("enriched") is True
        assert len(fake.calls) == 1
    finally:
        _ENRICHERS.pop("fake_src", None)


async def test_enrich_passthrough_when_no_source() -> None:
    msg = StubTaskMessage(metadata={}, trace_id="t")
    out = await enrich(msg)
    assert out is msg


async def test_enrich_passthrough_when_unknown_source() -> None:
    msg = StubTaskMessage(metadata={"source": "nope"}, trace_id="t")
    out = await enrich(msg)
    assert out is msg


async def test_enrich_swallows_enricher_exception() -> None:
    class _Boom:
        source = "boom_src"

        async def enrich(self, message: BaseTaskMessage) -> BaseTaskMessage:
            raise RuntimeError("boom")

    register_enricher(_Boom())
    try:
        msg = StubTaskMessage(metadata={"source": "boom_src"}, trace_id="t")
        out = await enrich(msg)
        assert out is msg  # An exception → return as-is, and do not block evaluation
    finally:
        _ENRICHERS.pop("boom_src", None)


# ── luoshu Langfuse enricher ──────────────────────────────────────────────────


async def test_luoshu_enricher_skips_without_ids() -> None:
    """Missing ids do not enrich, but the cause must be written into ``enrich_status`` (the old implementation could not tell the four causes apart)."""
    enricher = LuoshuTraceEnricher()
    msg = LuoshuTaskMessage()  # no project_id / trace_id
    out = await enricher.enrich(msg)
    assert out.langfuse is None
    assert out.enrich_status == STATUS_NO_IDS
    # A permanent cause → skip rather than retry, and do not report "trace does not exist" again
    assert out.enrich_retry_reason() is None
    assert "trace miss" not in (out.unevaluatable_reason() or "")


async def test_luoshu_enricher_skips_unknown_project(monkeypatch) -> None:
    """An empty instance map (credentials not configured): mark status=unknown_project, and do not retry it as a transient failure (it is cached
    for the process, so a retry can never succeed). unevaluatable_reason is now tightened to block only no_ids; this cause is allowed
    through to the evaluator, which records UNEVALUATED_SCORE (it is no longer skipped silently here).
    """
    monkeypatch.setattr(
        luoshu_enrichment, "load_luoshu_trace_instances", lambda _path: {}
    )
    enricher = LuoshuTraceEnricher()
    msg = LuoshuTaskMessage(project_id="luoshu", trace_id="s-1")
    out = await enricher.enrich(msg)
    assert out.langfuse is None
    assert out.enrich_status == STATUS_UNKNOWN_PROJECT
    assert "langfuse" not in out.metadata
    assert out.enrich_retry_reason() is None
    # With trace_id + project_id → no longer blocked on unevaluatable_reason; hand it to the evaluator.
    assert out.unevaluatable_reason() is None


async def test_luoshu_enricher_marks_query_error_as_retryable(monkeypatch) -> None:
    """A query error (timeout/5xx/network) is a transient failure → it should be retried, not permanently discarded.

    This is the class of dropped message in the old implementation: Langfuse blips, and the batch is silently skipped as "trace does not exist".
    """

    class _BoomResolver:
        async def fetch_trace(self, *_args, **_kwargs):
            raise TimeoutError("read timeout")

    monkeypatch.setattr(
        luoshu_enrichment,
        "load_luoshu_trace_instances",
        lambda _path: {"luoshu": object()},
    )
    enricher = LuoshuTraceEnricher()
    enricher._resolver = _BoomResolver()
    msg = LuoshuTaskMessage(project_id="luoshu", trace_id="s-1")

    out = await enricher.enrich(msg)

    assert out.langfuse is None
    assert out.enrich_status == STATUS_QUERY_ERROR
    # The point: take the retry path, not the skip path
    assert out.enrich_retry_reason() is not None
    assert out.unevaluatable_reason() is None


async def test_luoshu_enricher_marks_trace_miss(monkeypatch) -> None:
    """A 404 (the trace really does not exist): mark status=trace_miss and do not retry; unevaluatable_reason stays None,
    and the evaluator stores UNEVALUATED_SCORE as a record (so "however many were sent, that many are stored").
    """

    class _MissResolver:
        async def fetch_trace(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        luoshu_enrichment,
        "load_luoshu_trace_instances",
        lambda _path: {"luoshu": object()},
    )
    enricher = LuoshuTraceEnricher()
    enricher._resolver = _MissResolver()
    msg = LuoshuTaskMessage(project_id="luoshu", trace_id="s-1")

    out = await enricher.enrich(msg)

    assert out.enrich_status == STATUS_TRACE_MISS
    assert out.enrich_retry_reason() is None
    assert out.unevaluatable_reason() is None


async def test_luoshu_enricher_attaches_trace_on_hit(monkeypatch) -> None:
    class _FakeTrace:
        input = {"query": "hi"}
        output = {"text": "answer"}
        metadata = {"env": "prod"}

    class _FakeResolver:
        def __init__(self, instances) -> None:  # noqa: ANN001
            self.closed = False
            self.calls: list[dict] = []

        async def fetch_trace(self, trace_id, project_id, *, resolve_primary=True):  # noqa: ANN001
            self.calls.append(
                {"trace_id": trace_id, "project_id": project_id, "resolve_primary": resolve_primary}
            )
            return _FakeTrace()

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        luoshu_enrichment,
        "load_luoshu_trace_instances",
        lambda _path: {"proj": object()},
    )
    monkeypatch.setattr(luoshu_enrichment, "LuoshuTraceResolver", _FakeResolver)

    enricher = LuoshuTraceEnricher()
    msg = LuoshuTaskMessage(project_id="proj", trace_id="s-1")
    out = await enricher.enrich(msg)
    # Enrichment data is hung on the langfuse field and exposed via metadata.langfuse: the trace's top-level input/output/metadata.
    langfuse = out.metadata["langfuse"]
    assert langfuse["input"] == {"query": "hi"}
    assert langfuse["output"] == {"text": "answer"}
    assert langfuse["metadata"] == {"env": "prod"}
