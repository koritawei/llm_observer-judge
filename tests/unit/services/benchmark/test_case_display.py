"""case_display plugins: built-in businesses plus registrable extensions."""

from collections.abc import Mapping
from typing import Any

from benchmark.runs.case_display import (
    CaseDisplayContext,
    CaseDisplayRequest,
    CaseDisplayResult,
    clear_case_display_plugins,
    normalize_case_display,
    register_case_display_plugin,
    resolve_case_display_plugin,
)
from benchmark.runs.case_display.helpers import (
    as_finite,
    dimension_of,
    first_present_score,
    judge_from_extra_bucket,
    judge_from_sub_scores,
)
from benchmark.runs.case_display.plugins import (
    ensure_builtin_plugins,
    reset_builtin_plugins,
)
from benchmark.runs.case_display.views import CaseJudgeView


def setup_function() -> None:
    reset_builtin_plugins()
    ensure_builtin_plugins()


def test_trace_judge_extra_dims_and_display_text() -> None:
    case = {
        "output": {
            "component": {"displayText": "先别急，深呼吸一下。"},
        },
        "scores": {
            "fused": {
                "judge": {
                    "score": "0.5",
                    "reason": "综合判定为 0.5。",
                    "extra": {
                        "expression": {"score": 1, "reason": "位置暴露"},
                        "core_completion": {"score": 1, "reason": "可接受"},
                        "weights": {"expression": 0.5},
                    },
                },
                "attribution": {
                    "category": "privacy",
                    "confidence": 0.9,
                    "evidence_raw": ["证据A"],
                    "reasoning_raw": "主动暴露地址",
                },
            }
        },
    }
    output, judge, attr = normalize_case_display(
        case,
        CaseDisplayContext(notification_source="trace"),
    )
    assert output == "先别急，深呼吸一下。"
    assert judge is not None
    assert judge.score == 0.5
    assert "综合判定" in judge.reason
    assert {d.id for d in judge.dimensions} == {"expression", "core_completion"}
    assert attr is not None
    assert attr.format == "table"
    assert attr.category == "privacy"
    assert attr.confidence == 0.9
    assert attr.evidence == "证据A"
    assert attr.reasoning == "主动暴露地址"


def test_batch_ieval_criterion_scores_and_attribution_md() -> None:
    case = {
        "output": {"result": "可通过小德上报助手反馈"},
        "scores": {
            "fused": {
                "judge": {
                    "score": "1.2",
                    "reason": None,
                    "raw_json": {
                        "criterion_scores": {
                            "D0": {"score": 1.0, "reason": "红线未触发"},
                            "D4": {"score": 0.0, "reason": "未输出上报卡"},
                        }
                    },
                    "criterion_scores": {
                        "D0": {"score": 1.0, "reason": "红线未触发"},
                        "D4": {"score": 0.0, "reason": "未输出上报卡"},
                    },
                }
            }
        },
        "attribution": {
            "raw_json": {
                "attribution_md": "## 问题说明\n\n缺 Native 上报卡",
            },
            "severity": "major",
        },
    }
    output, judge, attr = normalize_case_display(
        case,
        CaseDisplayContext(notification_source="items", benchmark_name="ieval_external"),
    )
    assert output == "可通过小德上报助手反馈"
    assert judge is not None
    assert judge.score == 1.2
    assert [d.id for d in judge.dimensions] == ["D0", "D4"]
    assert attr is not None
    assert attr.format == "markdown"
    assert "Native" in (attr.markdown or "")


def test_batch_search_eval_v1_sub_scores() -> None:
    case = {
        "output": "附近网点清晰",
        "scores": {
            "fused": {
                "judge": {
                    "grade": "EXCELLENT",
                    "fused_score": 2.0,
                    "raw_json": {
                        "search_ai_card": {
                            "score": 2.0,
                            "reason": "附近网点完整。",
                            "sub_scores": {
                                "relevance": 2.0,
                                "factual_accuracy": 2.0,
                            },
                        }
                    },
                }
            }
        },
    }
    output, judge, _attr = normalize_case_display(
        case,
        CaseDisplayContext(notification_source="items", benchmark_name="search_eval"),
    )
    assert output == "附近网点清晰"
    assert judge is not None
    assert judge.score == 2.0
    assert judge.reason == "附近网点完整。"
    assert {d.id for d in judge.dimensions} == {"relevance", "factual_accuracy"}


def test_zero_score_not_swallowed_by_fused_fallback() -> None:
    assert first_present_score(0, 1.5) == 0.0
    extra_view = judge_from_extra_bucket(
        {
            "fused_score": 1.5,
            "raw_json": {
                "search_ai_card_v2": {
                    "score": 0,
                    "reason": "全扣",
                    "extra": {"style": {"score": 0, "reason": "禁用"}},
                }
            },
        }
    )
    assert extra_view.score == 0.0
    sub_view = judge_from_sub_scores(
        {
            "fused_score": 2.0,
            "raw_json": {
                "search_ai_card": {
                    "score": 0,
                    "reason": "零分",
                    "sub_scores": {"relevance": 0},
                }
            },
        }
    )
    assert sub_view.score == 0.0


def test_custom_plugin_registers_by_exact_key() -> None:
    """A business only has to register a plugin; it does not have to change normalize / run_case_page."""

    class DemoPlugin:
        name = "demo_biz"
        priority = 90

        def matches(self, request: CaseDisplayRequest) -> bool:
            del request
            return False

        def normalize(self, request: CaseDisplayRequest) -> CaseDisplayResult:
            del request
            return CaseDisplayResult(
                output="demo-output",
                judge=CaseJudgeView(score=9, reason="demo", dimensions=[]),
                attribution=None,
            )

    register_case_display_plugin(DemoPlugin(), keys=("demo_benchmark",))
    case: Mapping[str, Any] = {"output": "ignored"}
    ctx = CaseDisplayContext(notification_source="items", benchmark_name="demo_benchmark")
    plugin = resolve_case_display_plugin(CaseDisplayRequest(case=case, context=ctx))
    assert plugin.name == "demo_biz"
    output, judge, _attr = normalize_case_display(case, ctx)
    assert output == "demo-output"
    assert judge is not None
    assert judge.score == 9


def test_clear_and_reload_builtins() -> None:
    clear_case_display_plugins()
    reset_builtin_plugins()
    ensure_builtin_plugins()
    plugin = resolve_case_display_plugin(
        CaseDisplayRequest(
            case={},
            context=CaseDisplayContext(notification_source="trace"),
        )
    )
    assert plugin.name == "trace"


def test_registered_benchmarks_use_own_plugins() -> None:
    """An unregistered business uses default; trace applies only when aone_source=trace."""
    for benchmark_name in ("ieval_external", "search_eval"):
        plugin = resolve_case_display_plugin(
            CaseDisplayRequest(
                case={},
                context=CaseDisplayContext(notification_source="items", benchmark_name=benchmark_name),
            )
        )
        assert plugin.name == "default", benchmark_name
    trace_plugin = resolve_case_display_plugin(
        CaseDisplayRequest(
            case={},
            context=CaseDisplayContext(notification_source="trace", benchmark_name="amap_agent_benchmark"),
        )
    )
    assert trace_plugin.name == "trace"


def test_amap_agent_items_does_not_use_trace_plugin() -> None:
    """On the items side, amap_agent* does not attach the trace exact key, so a batch shape is not emptied by displayText parsing."""
    for benchmark_name in ("amap_agent", "amap_agent_benchmark"):
        plugin = resolve_case_display_plugin(
            CaseDisplayRequest(
                case={},
                context=CaseDisplayContext(notification_source="items", benchmark_name=benchmark_name),
            )
        )
        assert plugin.name == "default", benchmark_name


def test_trace_does_not_claim_amap_agent_prefix_siblings() -> None:
    """Under items, a non-exact name that contains the amap_agent substring must not be taken over by trace."""
    for benchmark_name, expected in (
        ("amap_agent_ieval", "default"),
        ("amap_agent_search_eval", "default"),
        ("amap_agent_search_eval_v2", "default"),
        ("foo_amap_agent_bar", "default"),
    ):
        plugin = resolve_case_display_plugin(
            CaseDisplayRequest(
                case={},
                context=CaseDisplayContext(notification_source="items", benchmark_name=benchmark_name),
            )
        )
        assert plugin.name == expected, benchmark_name


def test_unregistered_example_benchmark_uses_default_sub_scores() -> None:
    """An unregistered business (such as example) uses default, and sub_scores are parsed in the example shape."""
    case = {
        "output": {"result": "示例回答"},
        "judge": {
            "fused_score": 7.5,
            "raw_json": {
                "example": {
                    "score": 7.5,
                    "reason": "整体可用",
                    "sub_scores": {
                        "accuracy": 8.0,
                        "relevance": 7.0,
                        "completeness": 7.5,
                    },
                }
            },
        },
        "attribution": {
            "category": "tool/timeout",
            "confidence": 0.8,
            "evidence": "工具超时",
            "reasoning": "下游超时导致答非所问",
        },
    }
    plugin = resolve_case_display_plugin(
        CaseDisplayRequest(
            case=case,
            context=CaseDisplayContext(notification_source="items", benchmark_name="example_benchmark"),
        )
    )
    assert plugin.name == "default"
    output, judge, attr = normalize_case_display(
        case,
        CaseDisplayContext(notification_source="items", benchmark_name="example_benchmark"),
    )
    assert output == "示例回答"
    assert judge is not None
    assert judge.score == 7.5
    assert {d.id: d.score for d in judge.dimensions} == {
        "accuracy": 8.0,
        "relevance": 7.0,
        "completeness": 7.5,
    }
    assert attr is not None
    assert attr.category == "tool/timeout"


def test_ieval_does_not_claim_unknown_benchmark_with_criterion_scores() -> None:
    """When there is no ieval name, criterion_scores must not take over default (extra dimensions are kept)."""
    case = {
        "output": {"result": "unknown biz reply"},
        "scores": {
            "fused": {
                "judge": {
                    "score": 1.0,
                    "reason": "ok",
                    "extra": {
                        "style": {"score": 1, "reason": "语气正常"},
                    },
                    "raw_json": {
                        "criterion_scores": {
                            "D0": {"score": 1.0, "reason": "红线未触发"},
                        }
                    },
                    "criterion_scores": {
                        "D0": {"score": 1.0, "reason": "红线未触发"},
                    },
                }
            }
        },
    }
    plugin = resolve_case_display_plugin(
        CaseDisplayRequest(
            case=case,
            context=CaseDisplayContext(notification_source="items", benchmark_name=""),
        )
    )
    assert plugin.name == "default"
    _output, judge, _attr = normalize_case_display(
        case,
        CaseDisplayContext(notification_source="items", benchmark_name="unknown_biz"),
    )
    assert judge is not None
    assert {d.id for d in judge.dimensions} == {"style"}


def test_as_finite_rejects_infinity() -> None:
    assert as_finite(float("inf")) is None
    assert as_finite(float("-inf")) is None
    assert as_finite(1.5) == 1.5


def test_dimension_of_parses_numeric_string_score() -> None:
    dim = dimension_of("D0", {"score": "1.2", "reason": "pass"})
    assert dim is not None
    assert dim.score == 1.2
    assert dim.reason == "pass"
    bare = dimension_of("D1", "0")
    assert bare is not None
    assert bare.score == 0.0


def test_early_custom_register_still_loads_builtins() -> None:
    """A business registers first and then resolves for the first time; the built-in trace plugin is still available."""
    reset_builtin_plugins()

    class EarlyPlugin:
        name = "early"
        priority = 10

        def matches(self, request: CaseDisplayRequest) -> bool:
            return request.context.benchmark_name == "early_bm"

        def normalize(self, request: CaseDisplayRequest) -> CaseDisplayResult:
            del request
            return CaseDisplayResult(output="early", judge=None, attribution=None)

    register_case_display_plugin(EarlyPlugin(), keys=("early_bm",))
    # Do not call ensure by hand: rely on resolve to lazy-load the built-ins
    trace_plugin = resolve_case_display_plugin(
        CaseDisplayRequest(case={}, context=CaseDisplayContext(notification_source="trace"))
    )
    assert trace_plugin.name == "trace"
    early = resolve_case_display_plugin(
        CaseDisplayRequest(
            case={},
            context=CaseDisplayContext(notification_source="items", benchmark_name="early_bm"),
        )
    )
    assert early.name == "early"
