"""ExampleRubric — batch local-evaluation rubric for example_benchmark.

One holistic dimension (``evaluator_ref=example``) makes one local ``judge_example`` call
and produces a total score plus per-axis ``sub_scores`` (accuracy / relevance / completeness,
aligned with online ``agents.judge_example.sub_score_axes``). Axes are expanded for stats via
``stat_dimensions`` and are **not** split into extra criteria, so the worker does not call
the LLM once per axis.

``judge_execution_policy=local`` forbids falling through to the process Aone async pipeline
by default (that would drop the customization prompt and mismatch the dimension set).
"""

from benchmark.runs.common.dimension_stats import DimensionStatSpec
from benchmark.runs.common.rubric_base import BaseRubric
from benchmark.reporting.framework.schema import Criterion

#: Aligned position-by-position with ``conf/app.*.yaml`` ``agents.judge_example.sub_score_axes``.
_STAT_DIMENSIONS: tuple[DimensionStatSpec, ...] = (
    DimensionStatSpec(
        key="准确度",
        name="准确度",
        full_score=10.0,
        description="事实与表述是否准确，有无与证据矛盾或明显错误。",
    ),
    DimensionStatSpec(
        key="相关性",
        name="相关性",
        full_score=10.0,
        description="回答是否切中用户意图与 expected_output 核心要求。",
    ),
    DimensionStatSpec(
        key="完整性",
        name="完整性",
        full_score=10.0,
        description="核心检查点与关键约束是否覆盖充分，结构是否完整可用。",
    ),
)

#: Pass bar on a 0-10 scale; fused < this value → fail, and attribution is triggered (same rule as the prompt
#: "when score < 3, write a failure summary").
EXAMPLE_PASS_THRESHOLD = 3.0

_EXAMPLE = Criterion(
    id="example",
    name="综合质量",
    description=(
        "一次 judge 调用产出 0–10 总分，并按准确度 / 相关性 / 完整性输出分项；"
        "分项仅作统计与报告展开，不经平台多 criterion 拆分重复评测。"
    ),
    weight=1.0,
    evaluator_ref="example",
)


class ExampleRubric(BaseRubric):
    """example batch local evaluation: one holistic dimension plus three axes; sync local judge only.

    ``pass_threshold`` is for reports / the default. The runtime attribution gate uses the judge
    ``extra.attribution_threshold`` (declared in the prompt/YAML), and falls back to
    ``EXAMPLE_TRIGGER_MAX_EXCLUSIVE`` when that is missing. The dynamic composite score uses
    judge ``extra.weights`` × ``sub_scores``, and falls back to this Rubric's static weights
    (single-dimension weight=1.0) when those are missing.
    """

    aggregation_method = "weighted_avg"
    pass_threshold = EXAMPLE_PASS_THRESHOLD
    judge_execution_policy = "local"
    parse_interceptor_ref = "benchmark.runs.example.parse:ExampleParseInterceptor"
    stat_dimensions = _STAT_DIMENSIONS

    @classmethod
    def criteria(cls) -> list[Criterion]:
        """One holistic dimension (the judge emits three-axis sub_scores internally)."""
        return [_EXAMPLE]


__all__ = ["EXAMPLE_PASS_THRESHOLD", "ExampleRubric"]
