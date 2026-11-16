"""In-process notify filter using existing EvalConfigEntry dimensions."""

from benchmark.domain.agent_config import get_agent_names_by_project_id
from benchmark.domain.eval_config import (
    EvalConfigEntry,
    EvalConfigStore,
    evaluator_sample_key,
    normalize_agent_name,
)
from benchmark.domain.trace_filters import (
    matches_trace_filters,
    matches_trace_name,
    should_sample_deterministic,
    should_skip_test_adiu,
)
from benchmark.streaming.metadata import is_discard_agent_name
from benchmark.ingest.sidecar.notify import SidecarTraceNotify


def declared_name_hit(notify: SidecarTraceNotify, config: EvalConfigEntry) -> bool:
    """A registered ``agentName`` applies only when the name filter matches the whole string (same as kafka S3')."""
    declared = (config.agent_name or "").strip()
    return bool(declared and config.name and notify.name and notify.name in config.name)


def resolved_agent_name(raw: str | None, project_id: str) -> str | None:
    """Return a usable agent_name, matching streaming evaluator S3/S3b.

    Prefer *raw* when it is not blank/``unknown``. Otherwise look up the RemoteConfig
    agent registry by *project_id* (a multi-value candidate set: one project may register several
    agents, and a non-empty candidate set is admitted; stream-level ownership is decided by the eval config entry's ``agentName``).
    ``None`` means skip (do not evaluate).
    """
    if not is_discard_agent_name(raw):
        return str(raw).strip()
    pid = project_id.strip()
    if not pid:
        return None
    candidates = get_agent_names_by_project_id(pid)
    if not candidates:
        return None
    return candidates[0].strip()


def notify_match_reason(  # noqa: PLR0911 — filter reasons return in short-circuit order, matching notify_matches_config
    notify: SidecarTraceNotify,
    config: EvalConfigEntry,
    *,
    project_id: str,
) -> str:
    """Why *notify* missed this config, or ``hit``.

    Order matches :func:`notify_matches_config`. Does not read Lindorm/OSS.
    """
    if not config.is_sidecar_ingest:
        return "filter_no_config"
    notify_project = (notify.project_id or "").strip()
    if not notify_project or notify_project != project_id.strip():
        return "filter_no_config"
    raw_agent = (notify.agent_name or "").strip() if notify.agent_name else ""
    declared = (config.agent_name or "").strip()
    name_hit = declared_name_hit(notify, config)
    if is_discard_agent_name(raw_agent):
# Missing name: a name hit and an entry that declares agentName → use the declared value; otherwise fall back to the registry,
# and block when there is no candidate.
        if not name_hit and resolved_agent_name(raw_agent, notify_project) is None:
            return "filter_agent_name"
    elif name_hit and normalize_agent_name(raw_agent) != normalize_agent_name(declared):
        # Same as kafka ``agent_declared_in_evaluators``: hyphen and underscore are the same agent
        return "filter_agent_name"
    if should_skip_test_adiu(user_id=notify.user_id):
        return "filter_test_adiu"
    if config.environment and (notify.environment or "default") not in config.environment:
        return "filter_env"
    attr_reason = _attr_reason(config, notify)
    if attr_reason is not None:
        return attr_reason

    sample_key = evaluator_sample_key(notify.trace_id, config)
    if not should_sample_deterministic(config.sampling, sample_key):
        return "sample_miss"
    return "hit"


def _attr_reason(config: EvalConfigEntry, notify: SidecarTraceNotify) -> str | None:
    """Name / tags / time miss, in the same order as ``matches_trace_filters``."""
    if not matches_trace_name(config, notify.name):
        return "filter_name"
    if config.tags:
        trace_tags = set(notify.tags) if notify.tags else set()
        if not (trace_tags & set(config.tags)):
            return "filter_tags"
    if (config.start_time or config.end_time) and not matches_trace_filters(config, notify):
        return "filter_time"
    return None


def notify_matches_config(
    notify: SidecarTraceNotify,
    config: EvalConfigEntry,
    *,
    project_id: str,
) -> bool:
    """Whether *notify* passes this bound eval config. Does not read Lindorm/OSS."""
    return notify_match_reason(notify, config, project_id=project_id) == "hit"


def select_sampled_configs(
    notify: SidecarTraceNotify,
    config_store: EvalConfigStore,
) -> list[EvalConfigEntry]:
    """Return sidecar-ingest configs that pass thin filters + sampling.

    Missing ``project_id``, discardable ``agent_name``, or kafka ingest
    yields an empty list (caller acks without readback).
    """
    project_id = (notify.project_id or "").strip()
    if not project_id:
        return []
    return [
        cfg
        for cfg in config_store.get_configs(project_id)
        if notify_matches_config(notify, cfg, project_id=project_id)
    ]


__all__ = [
    "declared_name_hit",
    "notify_match_reason",
    "notify_matches_config",
    "resolved_agent_name",
    "select_sampled_configs",
]
