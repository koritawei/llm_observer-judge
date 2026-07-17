"""EvalDAGExecutor — sole production multi-eval path (SDD-EV-21 / ADR-0006)."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

from llmobserverjudge.eval_worker.conditions import evaluate_eval_condition

EvalFn = Callable[
    [str, dict[str, Any], dict[str, Any]],
    Awaitable[dict[str, Any]],
]


class EvalDAGExecutor:
    def __init__(
        self, eval_configs: list[dict[str, Any]], max_concurrent_per_layer: int = 10
    ) -> None:
        self.configs = {c["agent_id"]: c for c in eval_configs}
        self.max_concurrent_per_layer = max_concurrent_per_layer
        self.dag: dict[str, list[str]] = {}
        self.successors: dict[str, list[str]] = defaultdict(list)
        self.conditions: dict[str, str] = {}
        for aid, cfg in self.configs.items():
            deps = list(cfg.get("depends_on") or [])
            for dep in deps:
                if dep not in self.configs:
                    raise ValueError(
                        f"EvalDAG: agent '{aid}' depends on unknown agent '{dep}'"
                    )
            self.dag[aid] = deps
            if cfg.get("condition"):
                self.conditions[aid] = str(cfg["condition"])
            for dep in deps:
                self.successors[dep].append(aid)
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> None:
            if node in in_stack:
                raise ValueError(f"Cycle detected at {node}")
            if node in visited:
                return
            visited.add(node)
            in_stack.add(node)
            for dep in self.dag.get(node, []):
                dfs(dep)
            in_stack.discard(node)

        for node in self.dag:
            dfs(node)

    def topological_layers(self) -> list[list[str]]:
        in_degree = {n: len(self.dag.get(n, [])) for n in self.dag}
        queue = deque([n for n, d in in_degree.items() if d == 0])
        layers: list[list[str]] = []
        while queue:
            layer = list(queue)
            layers.append(layer)
            next_q: deque[str] = deque()
            for node in layer:
                for succ in self.successors.get(node, []):
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        next_q.append(succ)
            queue = next_q
        if sum(len(x) for x in layers) != len(self.dag):
            raise ValueError("Cycle or disconnected dependency residual")
        return layers

    def evaluate_condition(self, expr: str, upstream_results: dict[str, Any]) -> bool:
        return evaluate_eval_condition(expr, upstream_results)

    async def execute(self, trace: dict[str, Any], eval_fn: EvalFn) -> list[dict[str, Any]]:
        """Layer-parallel execution with semaphore, skip, and condition gates.

        - ``infra_error``: hard fail — dependents + subsequent layers are skipped
          (fail-stop).
        - ``model_error`` / scored bad-case: soft — does **not** abort the pipeline
          or skip unrelated peers (bad case must not block overall flow).
        """
        layers = self.topological_layers()
        all_results: dict[str, dict[str, Any]] = {}
        sem = asyncio.Semaphore(self.max_concurrent_per_layer)
        pipeline_aborted = False
        abort_reason = ""

        async def run_with_sem(
            aid: str, cfg: dict[str, Any], upstream: dict[str, Any]
        ) -> dict[str, Any]:
            async with sem:
                return await eval_fn(aid, cfg, upstream)

        for layer in layers:
            if pipeline_aborted:
                for aid in layer:
                    deps = list(self.configs[aid].get("depends_on") or [])
                    upstream_failed = any(
                        all_results.get(d, {}).get("score_status") == "infra_error"
                        for d in deps
                    )
                    reason = (
                        "upstream_failed"
                        if upstream_failed
                        else (abort_reason or "pipeline_aborted")
                    )
                    all_results[aid] = {
                        "agent_id": aid,
                        "score_status": "skipped",
                        "skip_reason": reason,
                    }
                continue

            to_run: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
            for aid in layer:
                cfg = self.configs[aid]
                deps = list(cfg.get("depends_on") or [])
                upstream = {d: all_results.get(d, {}) for d in deps}
                # Only hard infra_error from dependencies blocks a node.
                # model_error / bad_case scores do not affect overall flow.
                upstream_failed = any(
                    upstream.get(d, {}).get("score_status") == "infra_error"
                    for d in deps
                )
                if upstream_failed:
                    all_results[aid] = {
                        "agent_id": aid,
                        "score_status": "skipped",
                        "skip_reason": "upstream_failed",
                    }
                    continue
                condition = cfg.get("condition")
                if condition and not self.evaluate_condition(str(condition), upstream):
                    all_results[aid] = {
                        "agent_id": aid,
                        "score_status": "skipped",
                        "skip_reason": f"condition not met: {condition}",
                    }
                    continue
                to_run.append((aid, cfg, upstream))

            tasks = [run_with_sem(aid, cfg, up) for aid, cfg, up in to_run]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            layer_hard_fail = False
            for (aid, _, _), r in zip(to_run, results, strict=True):
                if isinstance(r, Exception):
                    all_results[aid] = {
                        "agent_id": aid,
                        "score_status": "infra_error",
                        "error": str(r),
                        "error_type": type(r).__name__,
                    }
                    layer_hard_fail = True
                    abort_reason = f"aborted_after:{aid}"
                else:
                    all_results[aid] = r
                    if r.get("score_status") == "infra_error":
                        layer_hard_fail = True
                        abort_reason = f"aborted_after:{aid}"

            if layer_hard_fail:
                pipeline_aborted = True

        # Stable order: layer order then agent_id within layer
        ordered: list[dict[str, Any]] = []
        for layer in layers:
            for aid in layer:
                if aid in all_results:
                    ordered.append(all_results[aid])
        return ordered
