"""Checker registry and degraded-execution behavior tests."""

import pytest
from benchmark.pipeline.checker import make_checker_result
from benchmark.pipeline.checker_registry import (
    CheckerDescriptor,
    CheckerRegistry,
    CheckerSpec,
    LegacyCheckerSetAdapter,
)
from benchmark.pipeline.signals import CheckerResult


class _Checker:
    def __init__(self, name: str, dimension: str, *, fails: bool = False) -> None:
        self.name = name
        self.dimensions = (dimension,)
        self._fails = fails

    def check(self, context: object) -> CheckerResult:
        del context
        if self._fails:
            raise RuntimeError("boom")
        return make_checker_result(
            name=self.name, dimensions=self.dimensions, status="pass", reason="ok"
        )


def test_registry_resolves_by_priority_without_reordering_ties() -> None:
    registry = CheckerRegistry()
    registry.register(
        "late",
        CheckerDescriptor(name="late", dimensions=("D1",), priority=200),
        lambda spec: _Checker(spec.name or "late", "D1"),
    )
    registry.register(
        "early",
        CheckerDescriptor(name="early", dimensions=("D1",), priority=100),
        lambda spec: _Checker(spec.name or "early", "D1"),
    )

    resolved = registry.resolve(
        [CheckerSpec(type="late"), CheckerSpec(type="early")], allowed_dimensions={"D1"}
    )

    assert [checker.name for checker in resolved] == ["early", "late"]


def test_registry_rejects_unknown_dimensions_and_duplicate_identity() -> None:
    registry = CheckerRegistry()
    registry.register(
        "shape",
        CheckerDescriptor(name="shape", dimensions=("D4",)),
        lambda spec: _Checker(spec.name or "shape", "D4"),
    )

    with pytest.raises(ValueError, match="unknown dimensions"):
        registry.resolve([CheckerSpec(type="shape")], allowed_dimensions={"V1"})
    with pytest.raises(ValueError, match="duplicate checker identity"):
        registry.resolve([CheckerSpec(type="shape"), CheckerSpec(type="shape")])


def test_legacy_adapter_preserves_order_and_failure_fallback() -> None:
    adapter = LegacyCheckerSetAdapter(
        (_Checker("first", "D1"), _Checker("second", "D2", fails=True))
    )

    results = adapter.run(object())

    assert [result.source_name for result in results] == ["first", "second"]
    assert results[1].status == "unknown"
    assert results[1].decision == "run_judge"
