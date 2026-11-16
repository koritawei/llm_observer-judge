"""Unit tests for LocalAttributionRunner — semaphore limiting is wired, plus the count/finalize contract.

Regression background: ``_attribute_one`` used to declare a ``semaphore`` parameter but never entered ``async
with`` (the limit passed by the caller was silently ignored); the batch method ``run_attributions``
had never been called in history and was deleted. This locks the real execution model: per-item inline await, one item
holds one concurrency slot for its whole run, and an unfinished item is not counted.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
class lar_mod:
    pass
class LocalAttributionRunner:
    pass
LocalAttributionRunnerDeps = None
LocalAttributionTask = None
_attribution_detail_bucket = None
_item_project_id = None
_resolve_attribution_project_id = None
_resolve_triage_evaluator_name = None
_write_benchmark_attribution = None


class _FakeTriageService:
    """A triage-service stub that records the number of concurrent active calls (arguments match the TriageRunInput call contract)."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.evaluator_names: list[str] = []

    async def run(
        self,
        input: Any,
    ) -> SimpleNamespace:
        self.calls += 1
        self.evaluator_names.append(getattr(input, "evaluator_name", ""))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.02)  # Give concurrent tasks a chance to overlap
        self.active -= 1
        return SimpleNamespace(status="done", reason="ok")


class _FakeBenchmarkService:
    """Stubs for get_run_item / get_run / increment_config_counter."""

    def __init__(self, expected: int = 99, *, service_name: str = "svc") -> None:
        self._expected = expected
        self._service_name = service_name
        self.count = 0

    async def get_run_item(self, run_id: str, item_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            input={"q": "x"},
            output={"a": "y"},
            source_trace_id="trace-1",
            judge_json={},
            item_id=item_id,
        )

    async def get_run(self, run_id: str) -> SimpleNamespace:
        return SimpleNamespace(service_name=self._service_name, project_id="proj")

    async def increment_config_counter(self, run_id: str, key: str) -> tuple[int, dict]:
        self.count += 1
        return self.count, {"attribution_expected_count": self._expected}


@pytest.fixture(autouse=True)
def _known_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip langfuse instance-config loading and force the attribution main path."""
    monkeypatch.setattr(lar_mod, "_is_known_trace_project", lambda pid: True)


def test_item_project_id_reads_metadata_then_csv_row() -> None:
    assert (
        _item_project_id(
            SimpleNamespace(metadata={"project_id": "meta-p", "csv_row": {"project": "row-p"}})
        )
        == "meta-p"
    )
    assert _item_project_id(SimpleNamespace(metadata={"csv_row": {"project": "row-p"}})) == "row-p"
    assert _item_project_id(SimpleNamespace(metadata={})) is None


def test_item_project_id_reads_input_and_judge_json_when_no_metadata() -> None:
    """BenchmarkRunItem has no metadata column: fall back from input / judge_json."""
    assert (
        _item_project_id(SimpleNamespace(input={"project_id": "from-input"}, judge_json={}))
        == "from-input"
    )
    assert (
        _item_project_id(SimpleNamespace(input={}, judge_json={"project_id": "from-judge-json"}))
        == "from-judge-json"
    )


def test_resolve_attribution_project_prefers_known_run_then_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = {"real-proj", "judge-proj"}
    monkeypatch.setattr(
        lar_mod,
        "_is_known_trace_project",
        lambda pid: bool(pid) and pid in known,
    )
    item = SimpleNamespace(metadata={"csv_row": {"project": "real-proj"}})
    # run is known → use run
    assert _resolve_attribution_project_id("real-proj", item) == "real-proj"
    # run is an unknown placeholder → fall back to item/csv
    assert _resolve_attribution_project_id("20261026_benchmark_final_results", item) == "real-proj"
    # Already resolved on the judge side (same source as EvaluatorRequest.project_id) and preferred over an empty run
    assert (
        _resolve_attribution_project_id(
            None,
            SimpleNamespace(metadata={}),
            judge_project_id="judge-proj",
        )
        == "judge-proj"
    )
    # Both unknown → keep run (so the skip log stays useful)
    assert _resolve_attribution_project_id("fake", SimpleNamespace(metadata={})) == "fake"


async def _run_concurrent(
    runner: LocalAttributionRunner, n: int
) -> tuple[list[asyncio.Task[None]], _FakeBenchmarkService]:
    tasks = [
        asyncio.create_task(
            runner._attribute_one(
                LocalAttributionTask(
                    run_id="r1",
                    item_id=f"item-{i}",
                    fused_score=0.3,
                    provider="example",
                ),
            )
        )
        for i in range(n)
    ]
    await asyncio.gather(*tasks)
    return tasks, runner._benchmark_service  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_attribute_one_semaphore_limits_concurrency() -> None:
    """Under Semaphore(1), concurrent items enter triage run serially (the limit really takes effect)."""
    triage = _FakeTriageService()
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=triage,
            benchmark_service=_FakeBenchmarkService(),
            semaphore=asyncio.Semaphore(1),
        ),
    )
    await _run_concurrent(runner, 3)

    assert triage.calls == 3
    assert triage.max_active == 1


@pytest.mark.asyncio
async def test_attribute_one_default_max_concurrency_serializes() -> None:
    """When semaphore is omitted, the default max_concurrency=1 is serial too."""
    triage = _FakeTriageService()
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=triage,
            benchmark_service=_FakeBenchmarkService(),
        ),
    )
    await _run_concurrent(runner, 2)

    assert triage.calls == 2
    assert triage.max_active == 1


@pytest.mark.asyncio
async def test_attribute_one_semaphore_two_allows_overlap() -> None:
    """Semaphore(2) allows exactly 2 overlapping calls — so wiring it does not force permanent serial execution."""
    triage = _FakeTriageService()
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=triage,
            benchmark_service=_FakeBenchmarkService(),
            semaphore=asyncio.Semaphore(2),
        ),
    )
    await _run_concurrent(runner, 4)

    assert triage.calls == 4
    assert triage.max_active == 2


@pytest.mark.asyncio
async def test_attribute_one_increments_count_and_finalizes() -> None:
    """When attribution_expected_count is reached, finalize_callback fires (synchronously)."""
    triage = _FakeTriageService()
    benchmark = _FakeBenchmarkService(expected=1)
    finalized: list[str] = []
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=triage,
            benchmark_service=benchmark,
        ),
    )
    await runner._attribute_one(
        LocalAttributionTask(
            run_id="r1",
            item_id="item-0",
            fused_score=0.3,
            provider="example",
            finalize_callback=finalized.append,
        ),
    )

    assert triage.calls == 1
    assert benchmark.count == 1
    assert finalized == ["r1"]


@pytest.mark.asyncio
async def test_attribute_one_finalize_waits_when_below_expected() -> None:
    """When the count has not reached expected, finalize does not fire."""
    triage = _FakeTriageService()
    benchmark = _FakeBenchmarkService(expected=5)
    finalized: list[str] = []
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=triage,
            benchmark_service=benchmark,
        ),
    )
    await runner._attribute_one(
        LocalAttributionTask(
            run_id="r1",
            item_id="item-0",
            fused_score=0.3,
            provider="example",
            finalize_callback=finalized.append,
        ),
    )

    assert benchmark.count == 1
    assert finalized == []


@pytest.mark.asyncio
async def test_attribute_one_increment_failure_does_not_leak() -> None:
    """A failed count increment does not leak out (it does not interrupt the caller's batch loop), and finalize does not fire."""
    triage = _FakeTriageService()
    benchmark = _FakeBenchmarkService(expected=1)

    async def _broken_increment(run_id: str, key: str) -> tuple[int, dict]:
        raise RuntimeError("db down")

    benchmark.increment_config_counter = _broken_increment  # type: ignore[method-assign]
    finalized: list[str] = []
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=triage,
            benchmark_service=benchmark,
        ),
    )

    # Must not raise (leaking the exception would blow up the caller's per-item inline loop)
    await runner._attribute_one(
        LocalAttributionTask(
            run_id="r1",
            item_id="item-0",
            fused_score=0.3,
            provider="example",
            finalize_callback=finalized.append,
        ),
    )
    assert triage.calls == 1  # The attribution main path is unaffected
    assert finalized == []


@pytest.mark.asyncio
async def test_attribute_one_finalize_callback_failure_does_not_leak() -> None:
    """A failed finalize_callback does not leak out, and the count has already been incremented (later items still re-check)."""
    triage = _FakeTriageService()
    benchmark = _FakeBenchmarkService(expected=1)
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=triage,
            benchmark_service=benchmark,
        ),
    )

    async def _bad_finalize(run_id: str) -> None:
        raise RuntimeError("finalize boom")

    await runner._attribute_one(
        LocalAttributionTask(
            run_id="r1",
            item_id="item-0",
            fused_score=0.3,
            provider="example",
            finalize_callback=_bad_finalize,
        ),
    )
    assert benchmark.count == 1  # The count has been persisted


def test_resolve_triage_evaluator_name_maps_placement_to_service() -> None:
    """provider=local/aone/none falls back to service_name; a business name is kept as-is."""
    assert _resolve_triage_evaluator_name("local", "example") == "example"
    assert _resolve_triage_evaluator_name("external", "example") == "example"
    assert _resolve_triage_evaluator_name("example", "other") == "example"
    # When there is no service_name, keep local (the failure reason stays observable when the caller/run lacks the field)
    assert _resolve_triage_evaluator_name("local", None) == "local"
    assert _resolve_triage_evaluator_name("local", "") == "local"


def test_attribution_detail_bucket_maps_report_fields() -> None:
    """AttributionResult → attribution_detail_json contains the fields the report needs."""
    attr = SimpleNamespace(
        model_dump=lambda mode="json": {
            "root_cause": "tool_error",
            "category": "planning",
            "reasoning": "因为工具失败",
            "confidence": 0.8,
        },
        root_cause="tool_error",
        category="planning",
        reasoning="因为工具失败",
        confidence=0.8,
    )
    bucket = _attribution_detail_bucket(attr)
    assert bucket["root_cause"] == "tool_error"
    assert bucket["primary_issue_category"] == "planning"
    assert bucket["attribution_md"] == "因为工具失败"
    assert bucket["confidence"] == 0.8


@pytest.mark.asyncio
async def test_write_benchmark_attribution_upserts_fused_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After local attribution succeeds, benchmark_run_items.attribution_detail_json must be written."""
    captured: dict[str, Any] = {}

    class _Writer:
        def __init__(self, *, run_id: str, criterion_id: str) -> None:
            captured["run_id"] = run_id
            captured["criterion_id"] = criterion_id

        async def upsert_triage(self, write: Any, **kwargs: object) -> object:
            captured["write"] = write
            return object()

    monkeypatch.setattr(lar_mod, "BenchmarkTraceRecordWriter", _Writer)
    attr = SimpleNamespace(
        model_dump=lambda mode="json": {
            "root_cause": "intent_miss",
            "category": "understanding",
            "reasoning": "未回答能否出去",
            "confidence": 0.9,
        },
        root_cause="intent_miss",
        category="understanding",
        reasoning="未回答能否出去",
        confidence=0.9,
        corrected_triage_level=None,
    )
    await _write_benchmark_attribution(
        run_id="run-1",
        item_id="item-0",
        item=SimpleNamespace(source_trace_id="tr-1"),
        provider="local",
        evaluator_name="example",
        outcome=SimpleNamespace(status="created", reason=None, attribution=attr),
    )
    assert captured["run_id"] == "run-1"
    assert captured["criterion_id"] == "fused"
    triage = captured["write"].triage
    assert triage.root_cause == "intent_miss"
    assert triage.attribution_detail_json["root_cause"] == "intent_miss"
    assert triage.attribution_detail_json["attribution_md"] == "未回答能否出去"


@pytest.mark.asyncio
async def test_attribute_one_provider_local_uses_service_name_as_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: provider=local must not pass evaluator_name as local (there is no classifier)."""
    writes: list[str] = []

    async def _fake_write(**kwargs: Any) -> None:
        writes.append(str(kwargs.get("evaluator_name")))

    monkeypatch.setattr(lar_mod, "_write_benchmark_attribution", _fake_write)
    triage = _FakeTriageService()
    runner = LocalAttributionRunner(
        LocalAttributionRunnerDeps(
            triage_service=triage,
            benchmark_service=_FakeBenchmarkService(service_name="example"),
        ),
    )
    await runner._attribute_one(
        LocalAttributionTask(
            run_id="r1",
            item_id="item-0",
            fused_score=1.3,
            provider="local",
        ),
    )
    assert triage.calls == 1
    assert triage.evaluator_names == ["example"]
    assert writes == ["example"]


def test_known_trace_project_ids_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """The project_id set is loaded only once (base64/json parsing is not repeated per item)."""
    # Note: the autouse fixture already mocks the _is_known_langfuse_project entry,
    # and this tests the cache-function contract inside it directly.
    calls = 0

    def _fake_load(encoded: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"proj-a": object()}

    monkeypatch.setattr(lar_mod, "load_trace_instances", _fake_load)
    lar_mod._known_trace_project_ids.cache_clear()
    try:
        ids_first = lar_mod._known_trace_project_ids()
        ids_second = lar_mod._known_trace_project_ids()
        assert calls == 1  # Two calls load the config only once
        assert "proj-a" in ids_first
        assert "proj-b" not in ids_first
        assert ids_first is ids_second  # lru_cache returns the same instance
    finally:
        lar_mod._known_trace_project_ids.cache_clear()
