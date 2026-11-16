"""ExampleBenchmarkServices — reference benchmark coordinator.

Demonstrates the ``@benchmark_component`` decorator usage:
- Holistic rubric (one criterion) + three-axis ``stat_dimensions`` (accuracy / relevance /
  completeness); one ``judge_example`` call produces the total score and ``sub_scores``
- Uses ``weighted_avg`` aggregation
- Downstream judge / attribution both use the ``example`` logic name
  (``judge_example`` / ``attribution_example``)
- Converts Langfuse dataset items to ``EvaluatorRequest`` via
  ``_build_evaluator_request``

The class is a *coordinator* registered via agentgraph — it does NOT perform
evaluation itself. ``BenchmarkRunner`` calls ``get_metadata()`` for
rubric/aggregation/dataset and ``_build_evaluator_request()`` per item.
``service_name`` identifies the evaluated business **service**.
"""

from benchmark.libs.agenthost.agents.config.context import AgentProviderContext
from pydantic import BaseModel

from benchmark.runs.common.agent_capabilities import (
    CapabilityBindingView,
    streaming_capabilities,
)
from benchmark.runs.common.dimension_stats import register_stat_dimensions
from benchmark.runs.common.sample_mapping import (
    RunSample,
    build_evaluator_request,
)
from benchmark.runs.example.rubric import ExampleRubric
from benchmark.runs.schema import BenchmarkRequest
from benchmark.reporting.framework.base import (
    BaseBenchmarkServices,
    BenchmarkComponentMetadata,
    benchmark_component,
)
from benchmark.reporting.framework.schema import DatasetRef

#: Importing the component registers stat dimensions (same shape as SearchEvalRubric; idempotent overwrite).
register_stat_dimensions("example_benchmark", ExampleRubric.stat_dimensions)

#: The attribution trigger (fused < 3) is built into
#: ``attribution_trigger_policy._register_builtin_trigger_policies``
#: (``EXAMPLE_TRIGGER_MAX_EXCLUSIVE``, aligned with ``ExampleRubric.pass_threshold``).

#: Batch chain: shares ``evaluator_service_*`` with streaming (same EvaluatorService route).
_EXAMPLE_BATCH_CAPABILITIES: tuple[CapabilityBindingView, ...] = (
    CapabilityBindingView(
        pipeline="batch",
        kind="judge",
        execution="sync",
        provider="local",
        description="对数据集样本一次本地打分（总分 + 准确度/相关性/完整性分项）",
        alias="evaluator_service_judge",
        class_ref=("benchmark.runs.common.execution_classes:EvaluatorServiceJudge"),
        note="EvaluatorService 路由 judge_example（与实时流同别名；judge_execution_policy=local）",
    ),
    CapabilityBindingView(
        pipeline="batch",
        kind="attribution",
        execution="sync",
        # provider=local is the execution route (same as the evaluator_service_attribution marker);
        # the triage classifier/agent key uses run.service_name=example; see LocalAttributionRunner.
        provider="local",
        description="对 fused < 3 的样本走本地 attribution_example 取证归因",
        alias="evaluator_service_attribution",
        class_ref=(
            "benchmark.runs.common.execution_classes:EvaluatorServiceAttributionMarker"
        ),
        note="EvaluatorService 路由 attribution_example（与实时流同别名，注册 customization 共用）",
    ),
    CapabilityBindingView(
        pipeline="batch",
        kind="report",
        execution="sync",
        provider="webdev",
        description="评测完成后生成 HTML 报表（维度得分 / 归因分布 / 样例明细）",
        note="按 benchmark 名缺省解析报表插件与模板",
    ),
)


class ExampleBenchmarkConfigSchema(BaseModel, extra="forbid"):
    """ext_schema for ``agents.example_benchmark`` YAML block."""

    webdev_agent_url: str = "http://webdev-agent:9880/"


@benchmark_component(
    BenchmarkComponentMetadata(
        name="example_benchmark",
        service_name="example",
        rubric=ExampleRubric.criteria(),
        aggregation=ExampleRubric.default_aggregation(),
        rubric_class="benchmark.runs.example.rubric:ExampleRubric",
        # Logic name → stage_agent_key(attribution, …) = attribution_example
        attribution_services=["example"],
        dataset=DatasetRef(
            dataset_name="amap_chat_eval",
            project_id="cmpwln0qs04fo6j8eryfdgznb",
            sample_size=50,
            sample_seed=42,
        ),
        # Left empty on purpose: L1 uses ownership.global_agents and does not write an overlay
        agent_scopes=(),
    ),
    ext_schema=ExampleBenchmarkConfigSchema,
)
class ExampleBenchmarkServices(BaseBenchmarkServices):
    """Reference coordinator — demo agent for both chains of the example evaluator.

    Declares both batch and streaming. Both chains use the unified aliases
    ``evaluator_service_judge`` / ``evaluator_service_attribution``, which land on
    ``judge_example`` / ``attribution_example``. The registration-level prompt/skill/attribution
    switches are shared. Rubric ``judge_execution_policy=local`` forbids a silent fallback to Aone async.
    """

    benchmark_description: str = (
        "example 评测器双链演示：批量与实时流统一 evaluator_service_judge / "
        "evaluator_service_attribution（落地 judge_example / attribution_example）"
        "与注册 customization。"
    )

    benchmark_rubric_description: str = (
        "holistic 单维综合质量（权重 1.0）+ 准确度/相关性/完整性分项统计，"
        "一次 judge_example 调用产出；通过阈值 3.0（0–10 分制，"
        "fused < 3 触发 attribution_example）；"
        "批量仅允许本地 sync 评测（judge_execution_policy=local）。"
    )

    # batch comes first: a batch trigger takes evaluator_service_*; the streaming chain appends the same aliases after
    # it.
    benchmark_capabilities: tuple[CapabilityBindingView, ...] = (
        *_EXAMPLE_BATCH_CAPABILITIES,
        *tuple(streaming_capabilities(["example"])),
    )

    @classmethod
    def from_config(
        cls,
        cfg: ExampleBenchmarkConfigSchema,
        ctx: AgentProviderContext,
    ) -> "ExampleBenchmarkServices":
        """Construct from YAML config + runtime context.

        The rubric/aggregation/dataset are baked into the class by the
        decorator; ``from_config`` only needs the runtime name.
        """
        return cls(name=ctx.name)

    def _build_evaluator_request(
        self,
        sample: object,
        run_id: str,
        request: BenchmarkRequest,
    ) -> "object":
        """Dataset item → EvaluatorRequest (generic mapping, with the example scene marker attached)."""
        return build_evaluator_request(
            self,
            RunSample(sample, run_id),
            request,
            extra_metadata={"scene": "non_voice"},
        )

    def _extract_react_rounds(self, sample: object) -> int | None:
        """Extract ReAct rounds from a dataset item's metadata.

        Looks for ``turns`` or ``react_rounds`` in the item's metadata.
        Returns ``None`` if not available.

        Langfuse ``DatasetItem.metadata`` is typed ``Any`` (a value stored in the dataset
        may be a str / list / int or another non-dict shape). A non-dict is treated as having
        no extractable fields and is not passed to ``.get``.
        """
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        turns = metadata.get("turns")
        if isinstance(turns, list):
            return len(turns)
        react_rounds = metadata.get("react_rounds")
        if isinstance(react_rounds, int):
            return react_rounds
        return None


# Importing the component attaches the unified report (default payload; idempotent overwrite).
from benchmark.reporting.framework.unified_report.plugin import (  # noqa: E402
    register_report_payload_plugin,
)
from benchmark.reporting.framework.unified_report.plugins.example import (  # noqa: E402
    ExampleBenchmarkReportPayloadPlugin,
)

register_report_payload_plugin("example_benchmark", ExampleBenchmarkReportPayloadPlugin())


__all__ = ["ExampleBenchmarkServices", "ExampleBenchmarkConfigSchema"]
