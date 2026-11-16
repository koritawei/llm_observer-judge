"""Eval call-count key normalization and aliases (domain pure function, no service dependency).

Aligned with the frontend ``normalizeRunCountKey`` and run-counts response keys: hyphens and
underscores are interchangeable, and the execution-side ``*_benchmark`` name matches the evaluator stem.
"""

from benchmark.domain.agent_names import canonical_agent_name


def normalize_eval_count_key(name: str) -> str:
    """Normalize amap-agent / amap_agent to lowercase underscores."""
    return name.strip().lower().replace("-", "_")


def eval_count_name_aliases(name: str) -> list[str]:
    """Count-key aliases: evaluator stem ↔ execution-side benchmark name."""
    cleaned = name.strip()
    if not cleaned:
        return []
    stem = canonical_agent_name(cleaned)
    stem_norm = normalize_eval_count_key(stem) if stem else ""
    aliases = [cleaned, stem, stem_norm]
    if stem_norm and not stem_norm.endswith("_benchmark"):
        aliases.append(f"{stem_norm}_benchmark")
        hyphen = stem_norm.replace("_", "-")
        aliases.extend([hyphen, f"{hyphen}_benchmark"])
    if cleaned == "search_eval" or stem_norm == "search_ai_card" or stem == "search_ai_card":
        aliases.extend(["search_eval", "search_ai_card"])
    seen: set[str] = set()
    ordered: list[str] = []
    for item in aliases:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
