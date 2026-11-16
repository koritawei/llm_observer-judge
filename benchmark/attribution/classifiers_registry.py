"""Triage classifiers. Only the example classifier is registered."""

from benchmark.attribution.service import TriageClassifier
from business.agents.example.classifier import ExampleTriageClassifier


def triage_classifiers() -> dict[str, TriageClassifier]:
    return {"example": ExampleTriageClassifier()}


__all__ = ["triage_classifiers"]
