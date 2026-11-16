"""Triage classifier contract — a pure-logic SPI each business implements per evaluator_name.

A single abstract method ``classify``: synchronous, pure CPU, framework-free. How triage is
decided (ranges / labels / any logic) is entirely the business's call; the platform only ships
the contract — no default implementation, no policy/loader.
"""

from abc import ABC, abstractmethod

from benchmark.attribution.service.schema import TriageInput, TriageVerdict


class TriageClassifier(ABC):
    """Each business implements one triage classifier — only :meth:`classify`."""

    @abstractmethod
    def classify(self, request: TriageInput) -> TriageVerdict:
        """Classify one evaluation into a verdict (synchronous, pure logic)."""
