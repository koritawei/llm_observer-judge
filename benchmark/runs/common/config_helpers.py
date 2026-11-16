"""config_json rebuild helpers — rebuild rubric/aggregation from serialized config during callback handling.

Generic config rebuild functions extracted from external_runner.py.
Any callback-driven evaluation system must rebuild rubric/aggregation from ``config_json``,
because the original ``BaseBenchmarkServices`` metadata is no longer in memory when the callback arrives.

Also includes :func:`resolve_oss_prefix` (D4): normalize an OSS prefix that is a relative segment or a full path.
"""

from typing import Any

from benchmark.runs.common.utils import to_dict
from benchmark.reporting.framework.schema import AggregationConfig, Criterion


def load_rubric_from_config(config: dict[str, Any]) -> list[Criterion]:
    """Rebuild the rubric from config_json (for callback handling).

    Prefer ``config["rubric"]``, and fall back to ``config["criteria"]``.
    Each criterion dict maps to a :class:`Criterion`.
    Missing fields are filled with defaults (weight=1.0, evaluator_ref="", description="").
    """
    rubric_data = config.get("rubric", [])
    if not rubric_data:
        rubric_data = config.get("criteria", [])
    return [
        Criterion(
            id=c.get("id", ""),
            name=c.get("name", ""),
            weight=c.get("weight", 1.0),
            evaluator_ref=c.get("evaluator_ref", ""),
            description=c.get("description", ""),
        )
        for c in rubric_data
    ]


def load_aggregation_from_config(config: dict[str, Any]) -> AggregationConfig:
    """Rebuild aggregation from config_json (for callback handling).

    Read method / pass_threshold / custom_fn_ref / fusion_fn_ref from ``config["aggregation"]``.
    Missing fields use defaults (method="weighted_avg", pass_threshold=0.8).
    """
    agg = config.get("aggregation", {})
    return AggregationConfig(
        method=agg.get("method", "weighted_avg"),
        pass_threshold=agg.get("pass_threshold", 0.8),
        custom_fn_ref=agg.get("custom_fn_ref"),
        fusion_fn_ref=agg.get("fusion_fn_ref"),
    )


def strip_per_criterion_failed_count(stats: object) -> dict[str, Any]:
    """Strip ``failed_count`` from per-criterion stats before sending to webdev-agent.

    The webdev-agent generates JS that may sum ``criterion_metrics[cid].failed_count``
    across N criteria, producing N × actual_failed.  Removing the field forces
    the generated JS to use the top-level ``total_failed`` (item-level distinct
    count) instead.

    The type of ``stats`` is not stable on the callback serialization path: a Pydantic model
    uses ``model_dump()``, and everything else is funneled through :func:`to_dict` (a dict /
    an object with ``__dict__`` / ``None`` and so on all return safely and do not raise
    ``TypeError``). Always shallow-copy one layer so a later ``pop`` does not mutate the
    caller's dict or the instance ``__dict__``.
    """
    d = stats.model_dump() if hasattr(stats, "model_dump") else dict(to_dict(stats))
    d.pop("failed_count", None)
    return d


def resolve_oss_prefix(raw: str | None, *, default_prefix: str) -> str:
    """Normalize an OSS prefix (the prefix is the full path; **do not concatenate anything**).

    - empty ``raw`` → ``default_prefix`` (RemoteConfig ``agent-eval.benchmark.oss``);
    - non-empty ``raw`` → **return it unchanged** (anything that reached the business side is
      already a full path, for example ``search-eval/benchmark_result/20261023``); do not prepend
      the shared prefix ``ieval-service/benchmark/benchmark_result``, or the path doubles and
      becomes ``NoSuchKey``;
    - a value that contains the ``oss://`` scheme is also kept as-is; the loader strips the scheme;
    - every shape rejects a ``..`` segment and an empty segment.

    Raises:
        ValueError: a ``..`` segment or an empty segment is present (the API maps this to 422; the scheduler skips with an error).
    """
    text = (raw or "").strip()
    if not text:
        return default_prefix
    _validate_oss_prefix_segments(text)
    return text


def _validate_oss_prefix_segments(prefix: str) -> None:
    """Reject a ``..`` segment and an empty segment (v4 §1.4 validation)."""
    path = prefix.split("://", 1)[1] if "://" in prefix else prefix
    stripped = path.strip("/")
    segments = stripped.split("/") if stripped else []
    if any(not seg or seg == ".." for seg in segments):
        raise ValueError(f"oss_prefix contains empty or '..' segments: {prefix!r}")


__all__ = [
    "load_rubric_from_config",
    "load_aggregation_from_config",
    "resolve_oss_prefix",
    "strip_per_criterion_failed_count",
]
