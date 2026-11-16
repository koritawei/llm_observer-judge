"""Example business triage classifier — the business implements ``classify`` itself.

This shows a simple score-threshold classifier; a real business writes its own logic here
(ranges / labels / anything). The registry entry is :mod:`benchmark.attribution.classifiers_registry`,
symmetric with :mod:`business.evaluators.registry`.
"""

from benchmark.attribution.service import (
    TriageClassifier,
    TriageInput,
    TriageLevel,
    TriageVerdict,
)

#: Example thresholds: a raw (un-normalized) score falling in a range matches that level.
GOOD_MIN_SCORE = 8.0
WATCH_MIN_SCORE = 6.0
MILD_BAD_MIN_SCORE = 3.0

#: Levels worth downstream attribution.
_ATTRIBUTE_LEVELS = (TriageLevel.MILD_BAD, TriageLevel.SEVERE_BAD)


class ExampleTriageClassifier(TriageClassifier):
    """Example classifier: thresholds >=8 good / >=6 watch / >=3 mild_bad / else severe_bad."""

    def classify(self, request: TriageInput) -> TriageVerdict:
        if request.score is None:
            return TriageVerdict(level=TriageLevel.NO_DECISION, should_attribute=False)
        level = self._level_for(request.score)
        return TriageVerdict(level=level, should_attribute=level in _ATTRIBUTE_LEVELS)

    @staticmethod
    def _level_for(score: float) -> TriageLevel:
        if score >= GOOD_MIN_SCORE:
            return TriageLevel.GOOD
        if score >= WATCH_MIN_SCORE:
            return TriageLevel.WATCH
        if score >= MILD_BAD_MIN_SCORE:
            return TriageLevel.MILD_BAD
        return TriageLevel.SEVERE_BAD
