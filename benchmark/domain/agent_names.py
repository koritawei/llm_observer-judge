"""Canonical agent name (domain pure function, no service dependency).

Normalizes a benchmark name / realtime evaluator_name to the public agent name:
``example_benchmark`` → ``example``, ``search_eval`` → ``search_ai_card``.
"""

#: Explicit benchmark name → agent name map (historical names the suffix rule cannot cover).
_AGENT_NAME_OVERRIDES: dict[str, str] = {
    "search_eval": "search_ai_card",
}


_BENCHMARK_SUFFIX = "_benchmark"


def canonical_agent_name(benchmark_name: str, service_name: str | None = None) -> str:
    """benchmark / registration name → canonical agent name (public view key).

    ``service_name`` is kept for compatibility (legacy callers pass it); normalization
    currently looks only at ``benchmark_name`` and the explicit override.
    """
    _ = service_name
    name = benchmark_name.strip()
    if name in _AGENT_NAME_OVERRIDES:
        return _AGENT_NAME_OVERRIDES[name]
    if name.endswith(_BENCHMARK_SUFFIX):
        return name[: -len(_BENCHMARK_SUFFIX)]
    return name


__all__ = ["canonical_agent_name"]
