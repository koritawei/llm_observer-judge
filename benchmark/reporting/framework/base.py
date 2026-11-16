"""@benchmark_component decorator + BaseBenchmarkServices.

A *benchmark coordinator* is a :class:`BaseAgent` subclass decorated with
``@benchmark_component`` (agentgraph registration only). Metadata carries
``service_name`` (evaluated **business service**), rubric, aggregation, and
dataset so :class:`BenchmarkRunner` can orchestrate multi-criterion offline
scoring without hard-coding logic.

Usage::

    @benchmark_component(
        BenchmarkComponentMetadata(
            name="example_benchmark",
            service_name="example",
            rubric=[
                Criterion(id="factual_accuracy", name="factual accuracy", ...),
                Criterion(id="format_correctness", name="format correctness", ...),
            ],
            aggregation=AggregationConfig(method="weighted_avg"),
            dataset=DatasetRef(dataset_name="amap_chat_eval", project_id="..."),
            # optional: declare L1-catalog-visible evaluation targets (multi-select); empty relies on the ownership seed only
            agent_scopes=("amap-agent",),
        ),
        ext_schema=ExampleConfigSchema,
    )
    class ExampleBenchmarkServices(BaseBenchmarkServices):
        def _build_evaluator_request(self, sample, run_id, request):
            ...

The decorator internally calls ``@agent_component(name=metadata.name,
ext_schema=ext_schema)`` so the class is also discoverable by ``AppRuntime``
bootstrap.
"""

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from benchmark.libs.agenthost.agents.base import BaseAgent
from benchmark.libs.agenthost.agents.card import AgentCard
from benchmark.libs.agenthost.agents.component_metadata import agent_component
from benchmark.libs.agenthost.agents.config.context import AgentProviderContext
from pydantic import BaseModel

from benchmark.domain.eval_config import logical_benchmark_run_name
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.reporting.framework.schema import (
    AggregatedMetrics,
    AggregationConfig,
    Criterion,
    DatasetRef,
)

#: Attribute name on the decorated class holding :class:`BenchmarkComponentMetadata`.
BENCHMARK_METADATA_ATTR = "__benchmark_component__"


class TriggerRequestView(Protocol):
    """Minimal duck view of a benchmark trigger request (dependency inversion; decoupled from the domain schema).

    ``BaseBenchmarkServices`` reads only ``sample_size`` / ``max_concurrency`` /
    ``triage_mode``. The domain
    :class:`~benchmark.runs.schema.BenchmarkRequest` (a Pydantic model)
    structurally matches this protocol, and the framework does not import the domain package.
    """

    sample_size: int | None
    max_concurrency: int
    triage_mode: str


@dataclass(frozen=True)
class PromptSlot:
    """One prompt-template slot a business may override (the whitelist unit of upload validation)."""

    key: str
    """Slot name (business filename / slot name, for example ``core_completion``)."""

    variables: tuple[str, ...] = ()
    """Set of ``${var}`` names allowed in the template; an unknown variable is rejected on upload.

    A literal ``*`` allows any identifier (used when example and similar cases expand macros from input fields).
    """

    required: bool = False
    """Whether this slot is required when a prompt is uploaded."""

    max_bytes: int = 32 * 1024


@dataclass(frozen=True)
class StageCustomizationPolicy:
    """Business-customization capability declared for one stage (judge / attribution); all off by default."""

    allow_prompt: bool = False
    allow_skill: bool = False
    prompt_slots: tuple[PromptSlot, ...] = ()
    skill_entry: str = "SKILL.md"
    max_prompt_bytes: int = 64 * 1024
    max_skill_bytes: int = 256 * 1024
    max_files: int = 32
    note: str = ""


@dataclass(frozen=True)
class BenchmarkCustomizationPolicy:
    """Agent-level business customization policy: the stage × form grid plus permission to toggle attribution."""

    judge: StageCustomizationPolicy = StageCustomizationPolicy()
    attribution: StageCustomizationPolicy = StageCustomizationPolicy()
    #: Whether a business may turn attribution off at registration (allowed by default: attribution is optional,
    #: and once off, execution is skipped even if attribution config/customization exists; an agent may set False to force it on)
    allow_attribution_toggle: bool = True
    note: str = ""


@dataclass(frozen=True)
class BenchmarkComponentMetadata:
    """Metadata attached by :func:`benchmark_component`.

    Read by ``BenchmarkRunner`` / ``BenchmarkService`` at run time to
    determine the rubric, aggregation strategy, and data source.
    """

    name: str
    """Component registration name (matches ``@agent_component`` name)."""

    service_name: str
    """Name of the evaluated business **service** (e.g. ``"amap_chat"``).

    Not an agentgraph agent id — the service under test. Pipeline rows still
    store this value in ``TraceRecord.agent_name`` (shared daily-table column).
    """

    rubric: list[Criterion]
    """Ordered list of scoring dimensions."""

    aggregation: AggregationConfig
    """How per-criterion scores are combined into a final score."""

    rubric_class: str | None = None
    """Optional rubric class reference ``module:Class`` (aligned with document.rubric_class on the page)."""

    attribution_services: list[str] = field(default_factory=list)
    """Attribution logic names (optional).

    Prefer bare names such as ``amap_agent`` / ``example`` — triage resolves
    ``attribution_<name>`` via ``stage_agent_key``. Full keys with an
    ``attribution_`` prefix are also accepted and stripped.
    Empty = derive from criterion ``evaluator_ref`` / ``service_name``.
    """

    dataset: DatasetRef = field(
        default_factory=lambda: DatasetRef(dataset_name="", project_id=""),
    )
    """Langfuse dataset reference (data source for the benchmark)."""

    customization: BenchmarkCustomizationPolicy = field(
        default_factory=BenchmarkCustomizationPolicy,
    )
    """Business prompt/skill customization policy (all off by default; the page shows the entry only after it is declared)."""

    agent_scopes: tuple[str, ...] = ()
    """Ids of L1-catalog-visible evaluation targets (multi-select, for example ``("amap-agent", "vehicle-agent")``).

    An empty tuple contributes nothing to the ownership overlay (the repo seed / ``global_agents``
    can still apply). When non-empty, startup and catalog sync add ``name`` to each scope
    whitelist and do not change L2 registration ownership.
    """

    def __post_init__(self) -> None:
        # Frozen dataclass: must use object.__setattr__ to copy mutable fields.
        # Without this, a caller mutating the original list would corrupt metadata.
        object.__setattr__(self, "rubric", list(self.rubric))
        object.__setattr__(self, "attribution_services", list(self.attribution_services))
        cleaned = tuple(
            scope.strip() for scope in self.agent_scopes if scope and str(scope).strip()
        )
        object.__setattr__(self, "agent_scopes", cleaned)


def benchmark_component(
    metadata: BenchmarkComponentMetadata,
    *,
    ext_schema: type | None = None,
) -> Callable[[type[BaseAgent]], type[BaseAgent]]:
    """Decorator that registers a class as a benchmark services.

    Internally calls ``@agent_component(name=metadata.name,
    ext_schema=ext_schema)`` so the class is discoverable by
    ``AppRuntime`` bootstrap, then attaches ``__benchmark_component__``
    metadata.

    Args:
        metadata: Benchmark component metadata — component registration name
            (must match YAML ``type``), evaluated business **service** name,
            ordered rubric, aggregation config, attribution logic names and
            dataset reference. The frozen dataclass copies mutable fields in
            ``__post_init__``, so the object can be attached as-is.
        ext_schema: Optional Pydantic schema for YAML ``agents.<name>`` block.

    Returns:
        Class decorator that returns the class with metadata attached.
    """

    def _decorate(cls: type[BaseAgent]) -> type[BaseAgent]:
        # 1. Apply @agent_component for runtime discovery
        decorated = agent_component(name=metadata.name, ext_schema=ext_schema)(  # type: ignore[arg-type]
            cls
        )
        # 2. Attach benchmark metadata (frozen + list-copied in __post_init__)
        setattr(decorated, BENCHMARK_METADATA_ATTR, metadata)
        return decorated

    return _decorate


def get_benchmark_metadata(cls: type) -> BenchmarkComponentMetadata | None:
    """Retrieve ``__benchmark_component__`` metadata from a class.

    Returns ``None`` if the class was not decorated with
    :func:`benchmark_component`.
    """
    return getattr(cls, BENCHMARK_METADATA_ATTR, None)


# ════════════════════════════════════════════════════════════════════════════
# BaseBenchmarkServices — coordinator base class
# ════════════════════════════════════════════════════════════════════════════


class BaseBenchmarkServices(BaseAgent):
    """Base class for benchmark **coordinators** (agentgraph registration).

    Does **not** evaluate the service itself. Carries
    ``service_name`` / rubric / aggregation / dataset via
    ``@benchmark_component``; ``BenchmarkRunner`` drives offline scoring.

    Subclasses implement:
        - :meth:`from_config` — construct from YAML config
        - :meth:`_build_evaluator_request` — convert a dataset item to
          :class:`EvaluatorRequest`
        - :meth:`_extract_react_rounds` — (optional) extract ReAct rounds
    """

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        cfg: BaseModel,
        ctx: AgentProviderContext,
    ) -> "BaseBenchmarkServices":
        """Construct the benchmark services from YAML config + runtime context.

        Subclasses typically just return ``cls(name=ctx.name)`` — the heavy
        lifting (rubric, dataset) is already baked into the class by the
        decorator. Override only when YAML-level config is needed.
        """

    @classmethod
    def get_metadata(cls) -> BenchmarkComponentMetadata:
        """Return the ``__benchmark_component__`` metadata.

        Raises:
            RuntimeError: if the class was not decorated with
                ``@benchmark_component``.
        """
        meta = getattr(cls, BENCHMARK_METADATA_ATTR, None)
        if meta is None:
            raise RuntimeError(f"{cls.__name__} is not decorated with @benchmark_component")
        return meta

    def get_agent_card(self) -> AgentCard | None:
        """A2A self-description card of the benchmark coordinator.

        Built from ``@benchmark_component`` metadata (name / service under test /
        rubric dimension count). An undecorated subclass returns ``None`` (aligned with
        the default ``BaseAgent`` semantics). The coordinator itself exposes no service
        entry; the card is only for ``AgentExplorer`` metadata discovery.
        """
        meta = getattr(self.__class__, BENCHMARK_METADATA_ATTR, None)
        if meta is None:
            return None
        return AgentCard(
            name=meta.name,
            description=(
                f"Benchmark coordinator for service '{meta.service_name}' "
                f"({len(meta.rubric)} rubric criteria); offline scoring only."
            ),
        )

    async def run(self, request: object, **kwargs: object) -> object:
        """Satisfy ``BaseAgent`` ABC contract.

        Benchmark services are orchestrated by ``BenchmarkService``, not called
        via the standard ``agent.run()`` interface. Calling this directly
        indicates a programming error.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a benchmark services component; use "
            f"BenchmarkService to create runs, not agent.run()"
        )

    # ── Hooks called by BenchmarkRunner ──────────────────────────────────

    @abstractmethod
    def _build_evaluator_request(
        self,
        sample: object,
        run_id: str,
        request: TriggerRequestView,
    ) -> EvaluatorRequest:
        """Convert a Langfuse dataset item into an :class:`EvaluatorRequest`.

        Args:
            sample: A Langfuse SDK ``DatasetItem`` (``.input``,
                ``.expected_output``, ``.id``, ``.source_trace_id``).
            run_id: Current benchmark run ID.
            request: The original trigger request (carries
                ``sample_size`` / ``max_concurrency`` / ``triage_mode``;
                structurally matches :class:`TriggerRequestView`).

        Returns:
            :class:`EvaluatorRequest` with ``project_id`` set to the
            Langfuse project ID, ``trace_id`` from the dataset item (or
            synthetic ``dataset:<itemId>``), Langfuse ``expected_output``
            mapped to ``output`` (not ``expected_output``; dataset-eval
            uses the opposite mapping), and
            ``metadata.pipeline_source=benchmark`` (runner also re-stamps).
        """

    def _extract_react_rounds(self, sample: object) -> int | None:
        """Extract ReAct rounds from a dataset item (optional hook).

        Returns:
            Number of ReAct reasoning rounds, or ``None`` if not applicable.
            Default implementation returns ``None``; subclasses override
            when the evaluated service uses ReAct-style prompting.
        """
        return None

    def _build_report_spec(
        self,
        run: object,
        aggregated: AggregatedMetrics,
        final_score: float,
    ) -> dict[str, Any]:
        """Business extension hook: may return extra ReportSpec fields.

        Standard fields (criteria / total_failed / rule_link_url /
        criterion_metrics with failed_count removed, and so on) are assembled by the
        unified kernel ``eval_framework.report_spec``. Keys in this method's return value
        that **share a name with a standard field are overwritten by the kernel**; only
        extension keys the kernel does not declare are kept.

        Name fields (benchmark_name / service_name / dataset_name) prefer the DB run
        record. Under multi-node deployment this process's registry may disagree with the
        snapshot from run creation (the component may not even be registered), so the DB
        is the only source of truth. When ``run=None`` or a field is missing, fall back to
        registry metadata. The ``benchmark_name`` column may already be a registrationId;
        the report header shows the logical name (config / component meta).
        """
        meta = self.get_metadata()
        cfg = getattr(run, "config_json", None) if run is not None else None
        display_name = logical_benchmark_run_name(
            column_name=getattr(run, "benchmark_name", None) if run is not None else None,
            config=cfg,
        )
        return {
            "benchmark_name": display_name or meta.name,
            "service_name": getattr(run, "service_name", None) or meta.service_name,
            "dataset_name": getattr(run, "dataset_name", None) or meta.dataset.dataset_name,
            "final_score": final_score,
            "aggregation_method": aggregated.aggregation_method,
            "total_samples": aggregated.total_samples,
            "total_duration_ms": aggregated.total_duration_ms,
            "criteria": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "weight": c.weight,
                    "evaluator_ref": c.evaluator_ref,
                }
                for c in meta.rubric
            ],
            "criterion_metrics": {
                cid: stats.model_dump() for cid, stats in aggregated.criterion_metrics.items()
            },
            "attribution_metrics": {
                cid: stats.model_dump() for cid, stats in aggregated.attribution_metrics.items()
            },
            "react_rounds_summary": dict(aggregated.react_rounds_summary),
        }


__all__ = [
    "BENCHMARK_METADATA_ATTR",
    "BenchmarkComponentMetadata",
    "BenchmarkCustomizationPolicy",
    "PromptSlot",
    "StageCustomizationPolicy",
    "benchmark_component",
    "get_benchmark_metadata",
    "BaseBenchmarkServices",
]
