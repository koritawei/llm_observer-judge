"""Unit tests for the dimension_stats registry and sub_scores dimension expansion.

Locked contract:
- register / resolve / isolation (an unregistered business → an empty dimension set, and aggregation does not change);
- ``collect_dimension_scores``: fused-row sub_scores (normalized to 0~1) × full_score
  restores the raw score; the union across criteria does not overwrite; dirty values are tolerated; a missing dimension is recorded as None;
- ``expand_dimension_metrics``: registered dimensions are merged into criterion_metrics (full score 2
  scale, pass_rate = full-score rate); an unregistered business does not touch the DB.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from benchmark.runs.common.dimension_stats import (
    DimensionStatSpec,
    clear_stat_dimensions,
    collect_dimension_scores,
    register_stat_dimensions,
    resolve_stat_dimensions,
)
from benchmark.runs.run_service import BenchmarkService, ExpandDimensionMetricsSpec

_DIMS = (
    DimensionStatSpec(key="relevance", name="内容切题", full_score=2.0),
    DimensionStatSpec(key="decision_completeness", name="决策信息完整度", full_score=2.0),
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    clear_stat_dimensions()
    yield
    clear_stat_dimensions()


def _fused_row(sub_scores: dict[str, float] | None, status: str = "success") -> Any:
    criterion_scores = (
        {"search_ai_card": {"score": 0.5, "reason": "", "sub_scores": sub_scores}}
        if sub_scores is not None
        else {}
    )
    return SimpleNamespace(
        status=status,
        judge_json={"criterion_scores": criterion_scores},
    )


# ════════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════════


def test_register_and_resolve_isolated() -> None:
    register_stat_dimensions("search_eval", _DIMS)
    assert resolve_stat_dimensions("search_eval") == _DIMS
    # Unregistered business → empty (legacy businesses such as ieval are not expanded)
    assert resolve_stat_dimensions("ieval_external") == ()
    assert resolve_stat_dimensions(None) == ()
    # Same-name override is idempotent
    register_stat_dimensions("search_eval", _DIMS)
    assert resolve_stat_dimensions("search_eval") == _DIMS


def test_resolve_run_stat_dimensions_prefers_config_snapshot() -> None:
    """The config snapshot takes priority over the registry (the snapshot is used even when the registry is empty)."""
    from benchmark.runs.common.dimension_stats import resolve_run_stat_dimensions

    config = {
        "stat_dimensions": [
            {
                "key": "relevance",
                "name": "内容切题",
                "full_score": 2.0,
                "description": "切题",
            },
        ],
    }
    dims = resolve_run_stat_dimensions(config, "search_eval")
    assert len(dims) == 1
    assert dims[0].key == "relevance"


def test_resolve_run_stat_dimensions_rubric_ref_fallback() -> None:
    """When both the snapshot and the registry are empty, load SearchEvalRubric.stat_dimensions via rubric_ref."""
    from benchmark.runs.common.dimension_stats import resolve_run_stat_dimensions
    SearchEvalRubric = None

    config = {
        "rubric_ref": "business.runs.search_eval.rubric:SearchEvalRubric",
    }
    dims = resolve_run_stat_dimensions(config, "search_eval")
    assert [d.key for d in dims] == [d.key for d in SearchEvalRubric.stat_dimensions]


def test_resolve_run_stat_dimensions_default_rubric_map() -> None:
    """With no rubric_ref, fall back to the benchmark default mapping."""
    from benchmark.runs.common.dimension_stats import resolve_run_stat_dimensions
    SearchEvalRubric = None

    dims = resolve_run_stat_dimensions({}, "search_eval")
    assert len(dims) == len(SearchEvalRubric.stat_dimensions)


# ════════════════════════════════════════════════════════════════════════════
# collect_dimension_scores
# ════════════════════════════════════════════════════════════════════════════


def test_collect_restores_raw_score_scale() -> None:
    """sub_scores are read as raw scores (the v3 protocol persists 0/1/2 and does not normalize)."""
    rows = [
        _fused_row({"relevance": 1.0, "decision_completeness": 2.0}),
        _fused_row({"relevance": 0.0, "decision_completeness": 1.0}),
    ]
    by_key = collect_dimension_scores(rows, _DIMS)
    assert by_key["relevance"] == [(1.0, False), (0.0, False)]
    assert by_key["decision_completeness"] == [(2.0, False), (1.0, False)]


def test_collect_missing_dimension_marks_none() -> None:
    """A row missing a dimension key → (None, failed_flag) (counted in the sample size, excluded from the mean)."""
    rows = [
        _fused_row({"relevance": 2.0}),  # Missing decision_completeness
        _fused_row(None, status="failed"),  # Failed row with no sub_scores
    ]
    by_key = collect_dimension_scores(rows, _DIMS)
    assert by_key["relevance"] == [(2.0, False), (None, True)]
    assert by_key["decision_completeness"] == [(None, False), (None, True)]


def test_collect_empty_dimensions_or_rows_returns_empty() -> None:
    assert collect_dimension_scores([_fused_row({"relevance": 1.0})], ()) == {}
    assert collect_dimension_scores([], _DIMS) == {}
    # Every row lacks sub_scores → empty (do not manufacture all-None stats)
    assert collect_dimension_scores([_fused_row(None)], _DIMS) == {}


# ════════════════════════════════════════════════════════════════════════════
# expand_dimension_metrics (BenchmarkService method, mock ORM)
# ════════════════════════════════════════════════════════════════════════════


class _FakeQuery:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _self() -> list[Any]:
            return self._rows

        return _self().__await__()


async def test_expand_merges_dimension_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registered dimensions → merged into criterion_metrics as six-dimension stats (mean is on the raw-score scale)."""
    register_stat_dimensions("search_eval", _DIMS)
    rows = [
        _fused_row({"relevance": 2.0, "decision_completeness": 1.0}),
        _fused_row({"relevance": 1.0, "decision_completeness": 1.0}),
    ]
    monkeypatch.setattr(
        "benchmark.runs.run_helpers.BenchmarkRunItem.filter",
        lambda **kw: _FakeQuery(rows),
    )
    metrics = SimpleNamespace(criterion_metrics={})

    expanded = await BenchmarkService().expand_dimension_metrics(
        ExpandDimensionMetricsSpec(
            run_id="run-1",
            benchmark_name="search_eval",
            metrics=metrics,
        ),
    )

    assert expanded is True
    rel = metrics.criterion_metrics["relevance"]
    # (2.0 + 1.0) / 2 = 1.5 (v3 raw scores read directly)
    assert rel.mean == 1.5
    assert rel.evaluator_ref == "sub_scores"
    assert rel.sample_count == 2
    # pass_threshold = full_score 2.0 → full-score rate (first row 2.0 is full score, second row 1.0 is not)
    assert rel.pass_rate == 0.5
    dec = metrics.criterion_metrics["decision_completeness"]
    assert dec.mean == 1.0


async def test_expand_noop_for_unregistered_business(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unregistered business (ieval and others): do not query the DB, do not change metrics, and return False."""
    called = []

    def _spy_filter(**kw: Any) -> _FakeQuery:
        called.append(kw)
        return _FakeQuery([])

    monkeypatch.setattr(
        "benchmark.runs.run_helpers.BenchmarkRunItem.filter",
        _spy_filter,
    )
    metrics = SimpleNamespace(criterion_metrics={"D1": "untouched"})

    expanded = await BenchmarkService().expand_dimension_metrics(
        ExpandDimensionMetricsSpec(
            run_id="run-1",
            benchmark_name="ieval_external",
            metrics=metrics,
        ),
    )

    assert expanded is False
    assert called == []
    assert metrics.criterion_metrics == {"D1": "untouched"}


# ════════════════════════════════════════════════════════════════════════════
# Snapshot serialization round-trip + automatic description generation (config_json persistence contract)
# ════════════════════════════════════════════════════════════════════════════


def test_dimension_spec_description_defaults_to_name() -> None:
    """When there is no Chinese description, backfill name automatically (reports and snapshots do not lack a description column)."""
    from benchmark.runs.common.dimension_stats import DimensionStatSpec

    dim = DimensionStatSpec(key="relevance", name="内容切题")
    assert dim.description == "内容切题"
    dim2 = DimensionStatSpec(key="k", name="n", description="自定义描述")
    assert dim2.description == "自定义描述"


def test_stat_dimensions_snapshot_roundtrip() -> None:
    """search_eval: Rubric class → Resolver snapshot → deserialize, six dimensions including Chinese descriptions."""
    from benchmark.runs.common.dimension_stats import (
        deserialize_stat_dimensions,
    )
    from benchmark.runs.common.profile_resolver import ProfileResolver
    from benchmark.runs.schema import BenchmarkRuntimeProfile
    SearchEvalBenchmarkServices = None

    resolved = ProfileResolver().resolve(
        BenchmarkRuntimeProfile(benchmark_name="search_eval"),
        SearchEvalBenchmarkServices.get_metadata(),
    )
    snapshot = resolved.config_snapshot()
    dims_raw = snapshot["stat_dimensions"]
    # The snapshot is JSON-serializable (config_json write contract)
    import json

    json.dumps(dims_raw, ensure_ascii=False)
    assert len(dims_raw) == 6
    assert all(d["description"] for d in dims_raw)  # Chinese names are backfilled automatically
    assert dims_raw[0]["full_score"] == 2.0

    # Deserialization round-trip
    dims = deserialize_stat_dimensions(dims_raw)
    assert [d.key for d in dims] == [d["key"] for d in dims_raw]
    assert dims[0].description == dims_raw[0]["description"]


def test_collect_reads_raw_json_sub_scores_first() -> None:
    """raw_json shape takes priority: read judge_json.raw_json[cid].sub_scores (the judge's original values).

    When criterion_scores is also present, raw_json wins (one source per row, not counted twice).
    """
    row = SimpleNamespace(
        status="success",
        judge_json={
            "criterion_scores": {
                "search_ai_card": {"score": 1.0, "sub_scores": {"relevance": 0.0}},
            },
            "raw_json": {
                "search_ai_card": {
                    "score": 1.0,
                    "sub_scores": {"relevance": 1.0, "decision_completeness": 2.0},
                },
            },
        },
    )
    by_key = collect_dimension_scores([row], _DIMS)
    # Values in the raw_json shape (not the 0.0 from criterion_scores)
    assert by_key["relevance"] == [(1.0, False)]
    assert by_key["decision_completeness"] == [(2.0, False)]


def test_collect_fallback_to_criterion_scores() -> None:
    """When raw_json is absent, fall back to criterion_scores[*].sub_scores (legacy-shape compatibility)."""
    row = SimpleNamespace(
        status="success",
        judge_json={
            "criterion_scores": {
                "search_ai_card": {"score": 1.0, "sub_scores": {"relevance": 1.0}},
            },
        },
    )
    by_key = collect_dimension_scores([row], _DIMS)
    assert by_key["relevance"] == [(1.0, False)]


def test_expand_case_dimension_scores_uses_sub_scores() -> None:
    """Case detail: a business with registered dimensions expands a score row per dimension from sub_scores (including the dimension reason)."""
    from benchmark.runs.common.dimension_stats import (
        register_stat_dimensions,
    )
    from benchmark.runs.run_service import _expand_case_dimension_scores

    register_stat_dimensions("search_eval", _DIMS)
    case = {
        "scores": {
            "search_ai_card": {
                "score": 1.0,
                "judge": {
                    "criterion_scores": {
                        "search_ai_card": {
                            "score": 1.0,
                            "sub_scores": {"relevance": 1.0, "decision_completeness": 2.0},
                            "extra": {
                                "dimensions": {
                                    "relevance": {"reason": "部分切题"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    _expand_case_dimension_scores(case, "search_eval")
    # Two of the six registered dimensions expand into independent rows
    assert case["scores"]["relevance"] == {
        "score": 1.0,
        "reason": "部分切题",
        "eval_failed": False,
        "sub_scores": None,
        "judge": None,
        "attribution": None,
    }
    assert case["scores"]["decision_completeness"]["score"] == 2.0
    # The primary criterion row is kept
    assert case["scores"]["search_ai_card"]["score"] == 1.0


def test_expand_case_dimension_scores_unregistered_noop() -> None:
    """Unregistered business (ieval and others): do not expand; case.scores stays as-is."""
    from benchmark.runs.run_service import _expand_case_dimension_scores

    case = {"scores": {"D1": {"score": 1.5, "judge": {"criterion_scores": {}}}}}
    _expand_case_dimension_scores(case, "ieval_external")
    assert list(case["scores"].keys()) == ["D1"]


def test_expand_case_dimension_scores_parses_example_dim_reason() -> None:
    """example: when extra.dimensions is absent, slice each dimension's reason out of the holistic reason."""
    from benchmark.runs.common.dimension_stats import register_stat_dimensions
    from benchmark.runs.example.rubric import ExampleRubric
    from benchmark.runs.run_service import _expand_case_dimension_scores

    register_stat_dimensions("example_benchmark", ExampleRubric.stat_dimensions)
    reason = (
        "准确度8：地点均来自真实搜索。"
        "相关性7：间接回答能出去。"
        "完整性4：未直接说能出去。"
        "总评：完整度欠缺。"
    )
    case = {
        "scores": {
            "example": {
                "score": 6.4,
                "judge": {
                    "criterion_scores": {
                        "example": {
                            "score": 6.4,
                            "reason": reason,
                            "sub_scores": {"准确度": 8.0, "相关性": 7.0, "完整性": 4.0},
                        },
                    },
                },
            },
        },
    }
    _expand_case_dimension_scores(case, "example_benchmark")
    assert "地点均来自真实搜索" in str(case["scores"]["准确度"]["reason"])
    assert "间接回答能出去" in str(case["scores"]["相关性"]["reason"])
    assert "未直接说能出去" in str(case["scores"]["完整性"]["reason"])


def test_expand_case_dimension_scores_uses_config_when_registry_empty() -> None:
    """When the registry is cleared, config.stat_dimensions can still expand (the same source as the report)."""
    from benchmark.runs.run_service import _expand_case_dimension_scores

    clear_stat_dimensions()
    case: dict[str, Any] = {
        "scores": {
            "search_ai_card": {
                "score": 1.0,
                "judge": {
                    "criterion_scores": {
                        "search_ai_card": {
                            "score": 1.0,
                            "sub_scores": {"relevance": 1.0},
                            "extra": {
                                "dimensions": {
                                    "relevance": {"reason": "切题不足"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    config = {
        "stat_dimensions": [
            {
                "key": "relevance",
                "name": "内容切题",
                "full_score": 2.0,
                "description": "切题",
            },
        ],
    }
    _expand_case_dimension_scores(case, "search_eval", config=config)
    assert case["scores"]["relevance"]["score"] == 1.0
    assert case["scores"]["relevance"]["reason"] == "切题不足"
