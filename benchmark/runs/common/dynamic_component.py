"""Runtime dynamic benchmark components.

A benchmark-agent name registered on the page may be chosen by the business (for example
``amap agent``) and does not have to equal a component name statically registered with
``@benchmark_component``. Trigger entries (manual-runs / run-pipeline) look up the
``agent_manager`` static registration first. On a miss, they look up a ``benchmark_agent``
document by name; on a hit they build a component from the document's rubric class, register
it, and then run normally. Only when the document is also missing do they 404.

A dynamic component's rubric / aggregation comes from the :class:`BaseRubric` subclass
pointed at by the document ``rubricClass``. The dataset reference defaults to empty (the
trigger request's ``dataset_name`` / ``project_id`` overrides win, same as a static
component falling back to ``meta.dataset``).
"""

from benchmark.libs.agenthost.agents.manager import AgentManager
from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.benchmark_console import find_agent_document_by_name
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.runs.common.rubric_base import load_rubric_class_ref
from benchmark.runs.common.sample_mapping import (
    RunSample,
    build_evaluator_request,
)
from benchmark.runs.registry import BENCHMARK_SERVICES_CLASSES
from benchmark.runs.schema import BenchmarkRequest
from benchmark.reporting.framework.base import (
    BENCHMARK_METADATA_ATTR,
    BaseBenchmarkServices,
    BenchmarkComponentMetadata,
)
from benchmark.reporting.framework.schema import (
    AggregationConfig,
    DatasetRef,
)

logger = get_logger(__name__)


class DynamicBenchmarkServices(BaseBenchmarkServices):
    """Benchmark coordinator driven by a page-registration document.

    Metadata is attached to the **subclass** by :func:`ensure_dynamic_benchmark_component`
    (``get_metadata`` is a classmethod and reads the class attribute ``__benchmark_component__``).
    Each custom name gets its own subclass, so instances do not interfere. There is no YAML
    assembly; ``from_config`` only satisfies the abstract contract. Sample conversion reuses the
    generic mapping (same shape as the example component).
    """

    @classmethod
    def from_config(cls, cfg: object, ctx: object) -> "DynamicBenchmarkServices":
        """A dynamic component is not assembled from YAML; this only implements the abstract contract."""
        return cls(name=getattr(ctx, "name", ""))

    def _build_evaluator_request(
        self,
        sample: object,
        run_id: str,
        request: BenchmarkRequest,
    ) -> EvaluatorRequest:
        """Dataset item → EvaluatorRequest (generic mapping; see sample_mapping)."""
        return build_evaluator_request(self, RunSample(sample, run_id), request)


async def ensure_dynamic_benchmark_component(
    name: str,
    runtime: "AppRuntimeLike",
) -> BaseBenchmarkServices:
    """Make sure a benchmark component is available: agent_manager → static class registry → page document.

    Resolution order (static first, so a same-named page document cannot take over):

    1. An instance already registered on ``agent_manager`` (YAML assembly / an earlier dynamic registration);
    2. The ``BENCHMARK_SERVICES_CLASSES`` static class registry: the component is registered with
       ``@benchmark_component`` but not declared in the agents section of ``conf/app.<env>.yaml``
       (a new component was left out of config). Instantiate it from metadata.name and register it,
       so a same-named page document (whose rubric defaults to empty) cannot take over;
    3. Build dynamically from the ``benchmark_agent`` page-registration document.

    Args:
        name: benchmark_name from the trigger request (the business's custom name).
        runtime: ``AppRuntime``, which provides the ``agent_manager`` registry.

    Returns:
        An executable benchmark services instance.

    Raises:
        KeyError: none of the three layers hit (the caller returns 404).
        ValueError: the document's rubricClass is missing or cannot be loaded (the caller returns 400).
    """
    try:
        return runtime.agent_manager.get(name)
    except KeyError:
        pass

    normalized_name = name.strip()
    static_cls = next(
        (cls for cls in BENCHMARK_SERVICES_CLASSES if cls.get_metadata().name == normalized_name),
        None,
    )
    if static_cls is not None:
        services = static_cls(name=normalized_name)
        try:
            runtime.agent_manager.register(normalized_name, services)
        except ValueError:
            return runtime.agent_manager.get(normalized_name)  # type: ignore[return-value]
        logger.info(
            "static benchmark component registered from class registry",
            extra={"benchmark_name": normalized_name},
        )
        return services

    document = await find_agent_document_by_name(name)
    if document is None:
        raise KeyError(
            f"Benchmark services not found: {name} (静态组件与页面注册文档均未命中)",
        )

    rubric_class = (document.rubric_class or "").strip()
    if rubric_class:
        criteria, aggregation, _ = load_rubric_class_ref(rubric_class)
    else:
        # After the page was simplified it no longer collects rubricClass: aligned with the static component's
        # component_rubric default — an empty rubric (the trigger request's rubric_class override wins),
        # while judge / attribution / report are managed by the agent-eval server default chain (profile defaults,
        # RemoteConfig agent-eval.service.evaluators / external-daily).
        criteria, aggregation = [], AggregationConfig(method="weighted_avg")

    normalized_name = name.strip()
    metadata = BenchmarkComponentMetadata(
        name=normalized_name,
        service_name=normalized_name,
        rubric=criteria,
        aggregation=aggregation,
        attribution_services=[],
        dataset=DatasetRef(dataset_name="", project_id=""),
    )
    services = _spawn_services_class(normalized_name, metadata)(name=normalized_name)
    try:
        runtime.agent_manager.register(normalized_name, services)
    except ValueError:
        # Concurrent triggers of the same custom name: another request already registered it, so reuse that one.
        existing = runtime.agent_manager.get(normalized_name)
        return existing  # type: ignore[return-value]
    logger.info(
        "dynamic benchmark component registered",
        extra={"benchmark_name": normalized_name, "rubric_class": rubric_class},
    )
    return services


def _spawn_services_class(
    name: str,
    metadata: BenchmarkComponentMetadata,
) -> "type[DynamicBenchmarkServices]":
    """Generate a subclass that carries its own metadata for one custom name.

    ``get_metadata`` is a classmethod that reads a class attribute. One subclass per name, so
    instances and metadata do not interfere. The class-name suffix is limited to alphanumerics
    (anything else is normalized to ``_``).
    """
    safe_suffix = "".join(ch if ch.isalnum() else "_" for ch in name)[:40] or "custom"
    cls = type(
        f"DynamicBenchmark_{safe_suffix}",
        (DynamicBenchmarkServices,),
        {},
    )
    setattr(cls, BENCHMARK_METADATA_ATTR, metadata)
    return cls


#: Structural stand-in for ``AppRuntime``: only the ``agent_manager`` field is required (avoid a cross-layer
#: import of the concrete runtime type; the caller in the api layer holds the real AppRuntime).
class AppRuntimeLike:  # noqa: B024
    """Structural protocol: holds ``agent_manager`` (the ``AgentManager`` registry)."""

    agent_manager: AgentManager


__all__ = [
    "AppRuntimeLike",
    "DynamicBenchmarkServices",
    "ensure_dynamic_benchmark_component",
]
