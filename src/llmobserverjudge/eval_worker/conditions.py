"""Safe AST condition evaluator for EvalDAG (SDD-EV-21 / ADR-0010)."""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Attribute,
    ast.Constant,
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def evaluate_eval_condition(expr: str, upstream_results: dict[str, Any]) -> bool:
    """Evaluate condition against upstream agent results. No eval()/exec()."""
    if not expr or not expr.strip():
        return True
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(f"disallowed AST node: {type(node).__name__}")
        return bool(_resolve(tree.body, upstream_results))
    except Exception as e:
        logger.warning("condition_eval_failed expr=%s err=%s", expr, e)
        return False


def _resolve(node: ast.AST, upstream_results: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        raise ValueError(f"bare name not allowed: {node.id}")
    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name):
            raise ValueError("only one-level attribute access allowed")
        agent_result = upstream_results.get(node.value.id) or {}
        if not agent_result:
            return None
        return agent_result.get(node.attr)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _resolve(node.operand, upstream_results)
    if isinstance(node, ast.BoolOp):
        vals = [_resolve(v, upstream_results) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Compare):
        left = _resolve(node.left, upstream_results)
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = _resolve(comparator, upstream_results)
            if not _compare(op, left, right):
                return False
            left = right
        return True
    raise ValueError(f"unhandled node: {type(node).__name__}")


def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
    if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) and (left is None or right is None):
        return False
    mapping: dict[type, bool] = {
        ast.Eq: left == right,
        ast.NotEq: left != right,
        ast.Lt: left < right,
        ast.LtE: left <= right,
        ast.Gt: left > right,
        ast.GtE: left >= right,
    }
    return bool(mapping.get(type(op), False))
