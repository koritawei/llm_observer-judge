"""Unit tests for the shared attribution-trigger policy — default implementation, per-dimension full-score scale, and business registry.

Covers:
- ``RubricV2TriggerPolicy``: trigger when fused < 1.4 (FAIL/POOR/ACCEPTABLE),
  do not trigger when equal to the threshold, trigger when fused=None, and
  criterion_scores does not participate in the default decision.
- ``DimensionFullScoreTriggerPolicy`` (ieval_external): trigger when any of
  D0–D4 is short of full score (D0 full score is 1, D1–D4 full score is 2),
  trigger conservatively on a None score, tolerate case differences, a missing
  dimension key does not participate, fall back when all are missing, D5/D6
  do not participate, and the dimension decision takes priority over fused.
- Default policy singleton ``get_default_trigger_policy``.
- Registry: register/resolve hit, unregistered falls back to the default,
  same-name override, and built-in registration (ieval_external) is restored
  after clear.
- Protocol duck typing: a business-custom policy satisfies ``AttributionTriggerPolicy``.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from benchmark.attribution.badcase.attribution_trigger_policy import (
    DIMENSION_FULL_SCORES,
    EXAMPLE_TRIGGER_MAX_EXCLUSIVE,
    RUBRIC_V2_TRIGGER_MAX_EXCLUSIVE,
    AttributionTriggerPolicy,
    DimensionFullScoreTriggerPolicy,
    RubricV2TriggerPolicy,
    clear_trigger_policies,
    get_default_trigger_policy,
    register_trigger_policy,
    resolve_trigger_policy,
)

#: D0–D4 all-full-score sample (decision baseline)
_FULL: dict[str, float] = {
    "D0": 1.0,
    "D1": 2.0,
    "D2": 2.0,
    "D3": 2.0,
    "D4": 2.0,
}


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Isolate the business registry — restore built-in registrations around each test to avoid registry leaks."""
    clear_trigger_policies()
    yield
    clear_trigger_policies()


class TestRubricV2TriggerPolicy:
    """Default policy — trigger when fused < 1.4 (grade in FAIL/POOR/ACCEPTABLE)."""

    @pytest.mark.parametrize("fused", [0.0, 0.79, 1.0, 1.39])
    def test_below_threshold_triggers(self, fused: float) -> None:
        assert RubricV2TriggerPolicy().should_trigger(fused_score=fused) is True

    @pytest.mark.parametrize("fused", [1.4, 1.79, 1.8, 2.0])
    def test_at_or_above_threshold_skips(self, fused: float) -> None:
        assert RubricV2TriggerPolicy().should_trigger(fused_score=fused) is False

    def test_none_fused_triggers(self) -> None:
        """Trigger when the fused row is missing or score lookup fails — do not skip attribution for a missing score."""
        assert RubricV2TriggerPolicy().should_trigger(fused_score=None) is True

    def test_criterion_scores_ignored_by_default_policy(self) -> None:
        """criterion_scores is consumed only by dimension-aware policies: the default implementation looks only at fused."""
        policy = RubricV2TriggerPolicy()
        assert (
            policy.should_trigger(
                fused_score=1.9,
                criterion_scores=dict(_FULL),
            )
            is False
        )
        assert (
            policy.should_trigger(
                fused_score=0.5,
                criterion_scores={"D0": 0.0},
            )
            is True
        )

    def test_max_exclusive_property(self) -> None:
        policy = RubricV2TriggerPolicy()
        assert policy.max_exclusive == RUBRIC_V2_TRIGGER_MAX_EXCLUSIVE
        assert RubricV2TriggerPolicy(max_exclusive=1.0).max_exclusive == 1.0


class TestDimensionFullScoreTriggerPolicy:
    """Dimension full-score policy (ieval_external) — trigger when any of D0–D4 is short of full score."""

    @pytest.mark.parametrize(
        ("dim", "below_full"),
        [
            ("D0", 0.5),
            ("D1", 1.5),
            ("D2", 1.9),
            ("D3", 0.0),
            ("D4", 1.0),
        ],
    )
    def test_any_dimension_below_full_triggers(
        self,
        dim: str,
        below_full: float,
    ) -> None:
        scores = dict(_FULL)
        scores[dim] = below_full
        assert (
            DimensionFullScoreTriggerPolicy().should_trigger(
                fused_score=1.95,
                criterion_scores=scores,
            )
            is True
        )

    def test_all_full_scores_skip(self) -> None:
        """D0–D4 all at full score → do not trigger (dimension scores win even if the fused score is low)."""
        assert (
            DimensionFullScoreTriggerPolicy().should_trigger(
                fused_score=1.2,
                criterion_scores=dict(_FULL),
            )
            is False
        )

    def test_none_score_treated_as_below_full(self) -> None:
        """Dimension present but score=None → trigger conservatively (do not skip attribution for a missing score)."""
        scores: dict[str, float | None] = dict(_FULL)  # type: ignore[assignment]
        scores["D2"] = None
        assert (
            DimensionFullScoreTriggerPolicy().should_trigger(
                fused_score=1.95,
                criterion_scores=scores,
            )
            is True
        )

    @pytest.mark.parametrize(
        ("scores", "expected"),
        [
            ({"d0": 0.5, **{k: v for k, v in _FULL.items() if k != "D0"}}, True),
            ({"d0": 1.0, "d1": 2.0, "d2": 2.0, "d3": 2.0, "d4": 2.0}, False),
        ],
    )
    def test_case_insensitive_keys(
        self,
        scores: dict[str, float | None],
        expected: bool,
    ) -> None:
        """Dimension id is case-insensitive — "d0" ≡ "D0"."""
        assert (
            DimensionFullScoreTriggerPolicy().should_trigger(
                fused_score=1.95,
                criterion_scores=scores,
            )
            is expected
        )

    def test_absent_dimensions_not_judged(self) -> None:
        """A missing dimension key does not participate — a single dimension at full score means do not trigger."""
        assert (
            DimensionFullScoreTriggerPolicy().should_trigger(
                fused_score=0.2,
                criterion_scores={"D0": 1.0},
            )
            is False
        )

    def test_all_absent_falls_back_to_fused(self) -> None:
        """D0–D4 all missing (non-dimensional callback) → fall back to the default fused < 1.4."""
        policy = DimensionFullScoreTriggerPolicy()
        assert (
            policy.should_trigger(
                fused_score=1.9,
                criterion_scores={"D5": 0.0, "D6": 0.0},
            )
            is False
        )
        assert (
            policy.should_trigger(
                fused_score=1.2,
                criterion_scores={"D5": 0.0, "D6": 0.0},
            )
            is True
        )

    @pytest.mark.parametrize("empty", [None, {}])
    def test_empty_scores_fall_back(
        self,
        empty: dict[str, float | None] | None,
    ) -> None:
        policy = DimensionFullScoreTriggerPolicy()
        assert (
            policy.should_trigger(
                fused_score=1.9,
                criterion_scores=empty,
            )
            is False
        )
        assert (
            policy.should_trigger(
                fused_score=None,
                criterion_scores=empty,
            )
            is True
        )

    def test_d5_d6_not_judged(self) -> None:
        """A low score on a dimension outside the full-score table (D5/D6) does not trigger — only D0–D4 participate."""
        scores: dict[str, float | None] = {**_FULL, "D5": 0.0, "D6": 1.0}
        assert (
            DimensionFullScoreTriggerPolicy().should_trigger(
                fused_score=1.95,
                criterion_scores=scores,
            )
            is False
        )

    def test_custom_full_scores(self) -> None:
        """A business may customize the full-score table — after D0's full score changes to 2, a score of 1 is short of full score."""
        policy = DimensionFullScoreTriggerPolicy(
            full_scores={"D0": 2.0},
        )
        assert (
            policy.should_trigger(
                fused_score=1.95,
                criterion_scores={"D0": 1.0},
            )
            is True
        )

    def test_custom_fallback_used_when_all_absent(self) -> None:
        """When all dimensions are missing, use the injected fallback rather than the module default singleton."""
        policy = DimensionFullScoreTriggerPolicy(
            fallback=RubricV2TriggerPolicy(max_exclusive=2.0),
        )
        assert (
            policy.should_trigger(
                fused_score=1.9,
                criterion_scores={"D9": 1.0},
            )
            is True
        )

    def test_full_scores_property(self) -> None:
        policy = DimensionFullScoreTriggerPolicy()
        assert policy.full_scores == DIMENSION_FULL_SCORES
        # Return a copy — external mutation must not pollute the internal decision table
        policy.full_scores["D0"] = 99.0
        assert policy.full_scores["D0"] == 1.0


class TestDefaultPolicy:
    """Default policy singleton."""

    def test_default_is_rubric_v2(self) -> None:
        assert isinstance(get_default_trigger_policy(), RubricV2TriggerPolicy)

    def test_default_singleton_stable(self) -> None:
        assert get_default_trigger_policy() is get_default_trigger_policy()

    def test_default_not_affected_by_registry_ops(self) -> None:
        """Register / clear does not affect the default singleton."""
        default = get_default_trigger_policy()
        register_trigger_policy("svc", RubricV2TriggerPolicy())
        clear_trigger_policies()
        assert get_default_trigger_policy() is default


class TestPolicyRegistry:
    """Business policy registry — benchmark component name → policy; unregistered falls back to the default."""

    def test_register_and_resolve(self) -> None:
        policy = RubricV2TriggerPolicy(max_exclusive=0.5)
        register_trigger_policy("webdev", policy)
        assert resolve_trigger_policy("webdev") is policy

    @pytest.mark.parametrize("benchmark_name", [None, "", "unregistered"])
    def test_unregistered_falls_back_to_default(
        self,
        benchmark_name: str | None,
    ) -> None:
        assert resolve_trigger_policy(benchmark_name) is get_default_trigger_policy()

    def test_same_name_overrides(self) -> None:
        """Same-name override, idempotent — the later registration wins."""
        first = RubricV2TriggerPolicy(max_exclusive=0.5)
        second = RubricV2TriggerPolicy(max_exclusive=1.8)
        register_trigger_policy("svc", first)
        register_trigger_policy("svc", second)
        assert resolve_trigger_policy("svc") is second

    def test_ieval_external_builtin_registration(self) -> None:
        """ieval_external registers the dimension full-score policy when the module is imported."""
        assert isinstance(
            resolve_trigger_policy("ieval_external"),
            DimensionFullScoreTriggerPolicy,
        )

    def test_clear_restores_builtin_registrations(self) -> None:
        """clear drops custom registrations and restores built-in ones (test isolation)."""
        register_trigger_policy("webdev", RubricV2TriggerPolicy())
        clear_trigger_policies()
        assert resolve_trigger_policy("webdev") is get_default_trigger_policy()
        assert isinstance(
            resolve_trigger_policy("ieval_external"),
            DimensionFullScoreTriggerPolicy,
        )
        example_policy = resolve_trigger_policy("example_benchmark")
        assert isinstance(example_policy, RubricV2TriggerPolicy)
        assert example_policy.max_exclusive == EXAMPLE_TRIGGER_MAX_EXCLUSIVE

    def test_ieval_external_policy_behavior(self) -> None:
        """End-to-end sample — the policy resolved from the registry decides on the dimension full-score scale."""
        policy = resolve_trigger_policy("ieval_external")
        assert (
            policy.should_trigger(
                fused_score=1.95,
                criterion_scores={**_FULL, "D4": 1.0},
            )
            is True
        )
        assert (
            policy.should_trigger(
                fused_score=1.2,
                criterion_scores=dict(_FULL),
            )
            is False
        )

    def test_example_benchmark_triggers_below_three(self) -> None:
        """example_benchmark: on a 0–10 scale, fused < 3 triggers attribution and ≥3 skips it."""
        policy = resolve_trigger_policy("example_benchmark")
        assert isinstance(policy, RubricV2TriggerPolicy)
        assert policy.max_exclusive == EXAMPLE_TRIGGER_MAX_EXCLUSIVE == 3.0
        assert policy.should_trigger(fused_score=2.99) is True
        assert policy.should_trigger(fused_score=0.0) is True
        assert policy.should_trigger(fused_score=None) is True
        assert policy.should_trigger(fused_score=3.0) is False
        assert policy.should_trigger(fused_score=9.5) is False


class TestPolicyProtocol:
    """Protocol structural compatibility — a business-custom implementation plugs in via duck typing."""

    def test_custom_policy_satisfies_protocol(self) -> None:
        class _FusedOnlyPolicy:
            def should_trigger(
                self,
                *,
                fused_score: float | None,
                criterion_scores: Any = None,
            ) -> bool:
                return fused_score == 0.0

        policy: Any = _FusedOnlyPolicy()
        assert isinstance(policy, AttributionTriggerPolicy)
        register_trigger_policy("custom", policy)
        assert resolve_trigger_policy("custom") is policy
