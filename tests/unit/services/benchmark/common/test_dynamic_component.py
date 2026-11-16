"""Dynamic benchmark components: static registration wins; on a miss, build dynamically from the page registration document."""

import pytest
from benchmark.domain.benchmark_console import AgentDocument
from benchmark.runs.common.dynamic_component import (
    AppRuntimeLike,
    DynamicBenchmarkServices,
    ensure_dynamic_benchmark_component,
)
from benchmark.reporting.framework.base import (
    BenchmarkComponentMetadata,
)
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    Criterion,
    DatasetRef,
)

SEARCH_EVAL_RUBRIC = "business.runs.search_eval.rubric:SearchEvalRubric"


class _AgentManager:
    """Minimal AgentManager test double (register fail-fast / get KeyError semantics aligned)."""

    def __init__(self) -> None:
        self._agents: dict = {}

    def register(self, name: str, agent) -> None:
        if name in self._agents:
            raise ValueError(f"Agent {name!r} 已注册")
        self._agents[name] = agent

    def get(self, name: str):
        if name not in self._agents:
            raise KeyError(name)
        return self._agents[name]


class _Runtime(AppRuntimeLike):
    def __init__(self) -> None:
        self.agent_manager = _AgentManager()


def _document(name: str, rubric_class: str | None = SEARCH_EVAL_RUBRIC) -> AgentDocument:
    return AgentDocument(
        id="ba-1",
        name=name,
        owner="eval-team",
        registeredAt="2026-10-16 12:00:00",
        rubricIds=[rubric_class] if rubric_class else [],
        rubricClass=rubric_class,
    )


def _patch_find_document(monkeypatch, document: AgentDocument | None) -> None:
    """Replace the domain query with a test double that returns a fixed document (the component layer does not touch the ORM)."""

    async def _find(name: str):
        return document

    monkeypatch.setattr(
        "benchmark.runs.common.dynamic_component.find_agent_document_by_name",
        _find,
    )


@pytest.mark.asyncio
async def test_static_registration_takes_priority(monkeypatch) -> None:
    """On a static-registration hit, return directly and do not query documents."""
    runtime = _Runtime()
    sentinel = object()
    runtime.agent_manager.register("example_benchmark", sentinel)

    async def _fail_find(name: str):
        raise AssertionError("static hit must not query documents")

    monkeypatch.setattr(
        "benchmark.runs.common.dynamic_component.find_agent_document_by_name",
        _fail_find,
    )
    assert await ensure_dynamic_benchmark_component("example_benchmark", runtime) is sentinel


@pytest.mark.asyncio
async def test_static_registry_takes_priority_over_same_name_document(
    monkeypatch,
) -> None:
    """Static class registry as the fallback: when agent_manager is not assembled (the YAML omitted it) but the component class exists,
    the static instance takes priority over a same-name page registration document (otherwise an empty-rubric document would seize it, and the streaming
    split would report that the rubric did not declare an evaluator)."""
    runtime = _Runtime()
    # A same-name page document exists and rubricClass is empty (the real shape after the page was simplified)
    _patch_find_document(monkeypatch, _document("amap_agent_benchmark", rubric_class=None))

    services = await ensure_dynamic_benchmark_component("amap_agent_benchmark", runtime)
    # Hits the static class AmapAgentBenchmarkServices, not DynamicBenchmarkServices
    assert type(services).__name__ == "AmapAgentBenchmarkServices"
    meta = services.get_metadata()
    assert meta.service_name == "amap_agent"
    # The static rubric carries evaluator_ref (the streaming routing key)
    assert any(getattr(c, "evaluator_ref", "") == "amap_agent" for c in meta.rubric)
    # Already registered in the manager: a second call hits directly
    assert await ensure_dynamic_benchmark_component("amap_agent_benchmark", runtime) is services


@pytest.mark.asyncio
async def test_dynamic_registration_from_document(monkeypatch) -> None:
    """On a static-registration miss, hit the document by name, build it dynamically, and register it."""
    runtime = _Runtime()
    _patch_find_document(monkeypatch, _document("amap agent"))

    services = await ensure_dynamic_benchmark_component("amap agent", runtime)
    assert isinstance(services, DynamicBenchmarkServices)
    # Already registered in the manager: a second call hits directly (static path)
    assert await ensure_dynamic_benchmark_component("amap agent", runtime) is services

    meta = services.get_metadata()
    assert meta.name == "amap agent"
    assert meta.service_name == "amap agent"
    # The rubric comes from the document rubricClass (criteria of SearchEvalRubric)
    assert len(meta.rubric) > 0
    assert any(c.id == "search_ai_card" for c in meta.rubric)


@pytest.mark.asyncio
async def test_missing_document_raises_keyerror(monkeypatch) -> None:
    """Neither the static registry nor a document hits → KeyError (the caller returns 404)."""
    runtime = _Runtime()
    _patch_find_document(monkeypatch, None)
    with pytest.raises(KeyError):
        await ensure_dynamic_benchmark_component("ghost", runtime)


@pytest.mark.asyncio
async def test_document_without_rubric_class_registers_empty_rubric(monkeypatch) -> None:
    """A document with no rubricClass bound → an empty-rubric component registers successfully (default semantics after the page was simplified)."""
    runtime = _Runtime()
    _patch_find_document(monkeypatch, _document("no-rubric", rubric_class=None))
    services = await ensure_dynamic_benchmark_component("no-rubric", runtime)
    meta = services.get_metadata()
    assert meta.rubric == []
    assert meta.aggregation.method == "weighted_avg"


def test_load_rubric_rejects_unknown_class_ref() -> None:
    """Shared load defensive branch: the class name does not exist in the module → ValueError.

    The AgentDocument schema already rejects an illegal rubricClass by allowlist (a full
    module+Class match). This case covers a defensive load failure (for example a leftover document after the class was removed in a refactor).
    """
    from benchmark.runs.common.rubric_base import load_rubric_class_ref

    with pytest.raises(ValueError):
        load_rubric_class_ref("business.runs.search_eval.rubric:NoSuchRubric")


def test_load_rubric_rejects_non_rubric_class() -> None:
    """Shared load defensive branch: the target class is not a BaseRubric subclass → ValueError."""
    from benchmark.runs.common.rubric_base import load_rubric_class_ref

    with pytest.raises(ValueError, match="BaseRubric"):
        load_rubric_class_ref(
            "benchmark.runs.common.dynamic_component:DynamicBenchmarkServices"
        )


@pytest.mark.asyncio
async def test_concurrent_registration_reuses_existing(monkeypatch) -> None:
    """Concurrent triggers of the same custom name: on a register conflict, reuse the already registered instance."""
    runtime = _Runtime()
    _patch_find_document(monkeypatch, _document("amap agent"))
    first = await ensure_dynamic_benchmark_component("amap agent", runtime)
    # Simulate another request that already finished registering: running the full flow again does not error and reuses the same instance
    runtime.agent_manager._agents.clear()  # noqa: SLF001 -- test only simulates the race window
    second = await ensure_dynamic_benchmark_component("amap agent", runtime)
    assert isinstance(second, DynamicBenchmarkServices)
    assert second.get_metadata().name == "amap agent"


def test_metadata_is_per_subclass() -> None:
    """Each dynamic name generates an independent subclass, and metadata does not leak across them."""
    from benchmark.runs.common.dynamic_component import _spawn_services_class

    def meta_of(name: str) -> BenchmarkComponentMetadata:
        return BenchmarkComponentMetadata(
            name=name,
            service_name=name,
            rubric=[
                Criterion(id="c1", name="维度", description="", weight=1.0, evaluator_ref=name),
            ],
            aggregation=AggregationConfig(method="weighted_avg"),
            attribution_services=[],
            dataset=DatasetRef(dataset_name="", project_id=""),
        )

    cls_a = _spawn_services_class("alpha", meta_of("alpha"))
    cls_b = _spawn_services_class("beta", meta_of("beta"))
    assert cls_a is not cls_b
    assert cls_a.get_metadata().name == "alpha"
    assert cls_b.get_metadata().name == "beta"


def test_dynamic_services_builds_evaluator_request() -> None:
    """_build_evaluator_request uses the generic mapping (duck typing)."""
    from types import SimpleNamespace

    from benchmark.runs.common.dynamic_component import _spawn_services_class

    cls = _spawn_services_class(
        "amap agent",
        BenchmarkComponentMetadata(
            name="amap agent",
            service_name="amap agent",
            rubric=[
                Criterion(id="c1", name="维度", description="", weight=1.0, evaluator_ref="x"),
            ],
            aggregation=AggregationConfig(method="weighted_avg"),
            attribution_services=[],
            dataset=DatasetRef(dataset_name="ds", project_id="p1"),
        ),
    )
    services = cls(name="amap agent")
    sample = SimpleNamespace(
        id="item-1",
        input="问一下",
        output="答一句",
        expected_output="期望答案",
        source_trace_id="t-1",
    )
    request = SimpleNamespace(project_id=None)
    evaluator_request = services._build_evaluator_request(sample, "run-1", request)
    assert evaluator_request.input == "问一下"
    assert evaluator_request.output == "答一句"
    assert evaluator_request.expected_output == "期望答案"
    assert evaluator_request.agent_name == "amap agent"
    assert evaluator_request.project_id == "p1"


__all__ = []
