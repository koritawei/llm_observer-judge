"""Evaluator registration. Only the example benchmark is loaded."""

from benchmark.evaluators.base import BaseEvaluatorAgent
from business.evaluators.example.agent import ExampleEvaluatorAgent  # noqa: F401

__all__ = ["ExampleEvaluatorAgent"]

_EVALUATOR_AGENTS = {
    "judge_example": ExampleEvaluatorAgent,
}


def evaluator_agents() -> dict[str, type[BaseEvaluatorAgent]]:
    return dict(_EVALUATOR_AGENTS)
