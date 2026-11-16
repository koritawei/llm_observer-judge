"""Unit tests for the shared Aone dispatch policy — default implementation plus business registry.

Covers:
- ``RubricV2DispatchPolicy``: let POOR/FAIL through, block ACCEPTABLE and above,
  let grade=None through, the spare score input does not participate in the
  default decision, and a custom grade set.
- Default policy singleton ``get_default_dispatch_policy``.
- Registry: register/resolve hit, unregistered falls back to the default,
  same-name override, and clear.
- Protocol duck typing: a business-custom policy satisfies ``AoneDispatchPolicy``.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from benchmark.attribution.badcase.dispatch_policy import (
    DISPATCH_GRADES_RUBRIC_V2,
    NotificationDispatchPolicy,
    RubricV2DispatchPolicy,
    clear_dispatch_policies,
    get_default_dispatch_policy,
    register_dispatch_policy,
    resolve_dispatch_policy,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Isolate the business registry — clear it around each test so registrations do not leak into other tests."""
    clear_dispatch_policies()
    yield
    clear_dispatch_policies()


class TestRubricV2DispatchPolicy:
    """Default policy — rubric_v2 grade decision (only POOR / FAIL are dispatched)."""

    @pytest.mark.parametrize("grade", ["POOR", "FAIL"])
    def test_poor_and_fail_dispatch(self, grade: str) -> None:
        assert RubricV2DispatchPolicy().should_dispatch(grade=grade) is True

    @pytest.mark.parametrize("grade", ["ACCEPTABLE", "GOOD", "EXCELLENT"])
    def test_acceptable_and_above_skip(self, grade: str) -> None:
        assert RubricV2DispatchPolicy().should_dispatch(grade=grade) is False

    def test_none_grade_passes(self) -> None:
        """A missing grade (score lookup failed) is let through — fail-open, so missing data does not skip dispatch.

        Severity tiering is deferred and does not block (attribution_dispatch); once past the threshold, push.
        This test locks the fail-open behavior; after severity gating returns, the None case is covered by that fallback.
        """
        assert RubricV2DispatchPolicy().should_dispatch(grade=None) is True

    def test_score_ignored_by_default_policy(self) -> None:
        """score is only a spare input: the default implementation looks only at grade, and the score does not override the decision."""
        policy = RubricV2DispatchPolicy()
        assert policy.should_dispatch(grade="ACCEPTABLE", score=0.5) is False
        assert policy.should_dispatch(grade="FAIL", score=1.9) is True

    def test_custom_dispatch_grades(self) -> None:
        """A business may customize the grade set — only FAIL is dispatched (POOR is blocked)."""
        policy = RubricV2DispatchPolicy(dispatch_grades=frozenset({"FAIL"}))
        assert policy.should_dispatch(grade="FAIL") is True
        assert policy.should_dispatch(grade="POOR") is False

    def test_dispatch_grades_property(self) -> None:
        policy = RubricV2DispatchPolicy()
        assert policy.dispatch_grades == DISPATCH_GRADES_RUBRIC_V2


class TestDefaultPolicy:
    """Default policy singleton."""

    def test_default_is_rubric_v2(self) -> None:
        assert isinstance(get_default_dispatch_policy(), RubricV2DispatchPolicy)

    def test_default_singleton_stable(self) -> None:
        assert get_default_dispatch_policy() is get_default_dispatch_policy()

    def test_default_not_affected_by_registry_ops(self) -> None:
        """Register / clear does not affect the default singleton."""
        default = get_default_dispatch_policy()
        register_dispatch_policy("svc", RubricV2DispatchPolicy())
        clear_dispatch_policies()
        assert get_default_dispatch_policy() is default


class TestPolicyRegistry:
    """Business policy registry — service_name → policy; unregistered falls back to the default."""

    def test_register_and_resolve(self) -> None:
        policy = RubricV2DispatchPolicy(dispatch_grades=frozenset({"FAIL"}))
        register_dispatch_policy("webdev", policy)
        assert resolve_dispatch_policy("webdev") is policy

    @pytest.mark.parametrize("service_name", [None, "", "unregistered"])
    def test_unregistered_falls_back_to_default(
        self,
        service_name: str | None,
    ) -> None:
        assert resolve_dispatch_policy(service_name) is get_default_dispatch_policy()

    def test_same_name_overrides(self) -> None:
        """Same-name override, idempotent — the later registration wins."""
        first = RubricV2DispatchPolicy(dispatch_grades=frozenset({"FAIL"}))
        second = RubricV2DispatchPolicy(dispatch_grades=frozenset({"POOR"}))
        register_dispatch_policy("svc", first)
        register_dispatch_policy("svc", second)
        assert resolve_dispatch_policy("svc") is second

    def test_clear_falls_back_to_default(self) -> None:
        register_dispatch_policy("webdev", RubricV2DispatchPolicy())
        clear_dispatch_policies()
        assert resolve_dispatch_policy("webdev") is get_default_dispatch_policy()


class TestPolicyProtocol:
    """Protocol structural compatibility — a business-custom implementation plugs in via duck typing."""

    def test_custom_policy_satisfies_protocol(self) -> None:
        class _FailOnlyPolicy:
            def should_dispatch(
                self,
                *,
                grade: str | None,
                score: float | None = None,
            ) -> bool:
                return grade == "FAIL"

        policy: Any = _FailOnlyPolicy()
        assert isinstance(policy, NotificationDispatchPolicy)
        register_dispatch_policy("custom", policy)
        assert resolve_dispatch_policy("custom") is policy
