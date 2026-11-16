"""Unit tests for the BaseRubric protocol (design doc §6.1)."""

import pytest
from benchmark.runs.common.rubric_base import BaseRubric
AmapChatRubricV2 = None
from benchmark.reporting.framework.schema import Criterion


def test_base_criteria_not_implemented() -> None:
    """The base-class criteria() must be implemented by a subclass."""
    with pytest.raises(NotImplementedError):
        BaseRubric.criteria()


def test_base_default_aggregation() -> None:
    """Default aggregation: weighted_avg / pass_threshold=1.0 / no fusion."""
    agg = BaseRubric.default_aggregation()
    assert agg.method == "weighted_avg"
    assert agg.pass_threshold == 1.0
    assert agg.fusion_fn_ref is None


def test_amap_chat_rubric_v2_contract() -> None:
    """amap-chat migration example: seven dimensions D0-D6 + weights + interceptor/fusion mounts."""
    criteria = AmapChatRubricV2.criteria()
    assert [c.id for c in criteria] == ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
    assert all(isinstance(c, Criterion) for c in criteria)
    weights = {c.id: c.weight for c in criteria}
    assert weights["D0"] == 0.0  # Gate dimensions do not participate in the weighted sum
    assert weights["D4"] == 0.25
    # Weighted-dimension weights sum to 1.0 (D1-D6)
    assert abs(sum(w for cid, w in weights.items() if cid != "D0") - 1.0) < 1e-9

    # Mounted as class attributes (§7.5: changing the rubric changes parse)
    assert AmapChatRubricV2.parse_interceptor_ref == (
        "business.runs.ieval.amap_chat.parse:AmapChatParseInterceptor"
    )
    assert AmapChatRubricV2.fusion_fn_ref == (
        "business.runs.ieval.amap_chat.services:rubric_v2_fuse"
    )
    agg = AmapChatRubricV2.default_aggregation()
    assert agg.method == "weighted_avg"
    assert agg.fusion_fn_ref == AmapChatRubricV2.fusion_fn_ref


def test_amap_chat_component_meta_shares_rubric_class() -> None:
    """The component metadata rubric and the Rubric class share one source (§9: the push default fallback matches)."""
    IevalExternalBenchmarkServices = None

    meta = IevalExternalBenchmarkServices.get_metadata()
    assert [c.id for c in meta.rubric] == [c.id for c in AmapChatRubricV2.criteria()]
    assert [c.weight for c in meta.rubric] == [c.weight for c in AmapChatRubricV2.criteria()]
