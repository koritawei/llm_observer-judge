"""The evaluators package — the evaluator agents and their shared contract.

Holds :mod:`benchmark.evaluators.base` (the development standard: ``BaseEvaluatorAgent``
+ ``EvaluatorRequest`` / ``EvaluatorResult``) and one package per concrete evaluator
(only ``example`` this round). ``EvaluatorError`` + the error codes live in
:mod:`benchmark.utils.exceptions`. The surrounding service layer — config / registry /
response / orchestration — lives in :mod:`benchmark.services.evaluation`.

This ``__init__`` re-exports only the base contract; callers reach concrete
evaluators via their own packages (e.g. ``business.evaluators.example``).
"""

from benchmark.evaluators.base import (
    BaseEvaluatorAgent,
    EvaluatorRequest,
    EvaluatorResult,
)

__all__ = [
    "BaseEvaluatorAgent",
    "EvaluatorRequest",
    "EvaluatorResult",
]
