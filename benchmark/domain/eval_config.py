"""Eval config store — read eval config from the DB snapshot.

This module re-exports the Pydantic schema types from :mod:`benchmark.domain.schemas`
(for backward compatibility) and provides :class:`EvalConfigStore`.
The page writes config into ``runtime_config_records`` (kind=evaluator), and
environment-level fields are stamped at write time.
This store no longer reads RemoteConfig or a fallback file.

Schema types (re-exported from :mod:`benchmark.domain.schemas`):
    - :class:`VariableMapping` — observation field → evaluator prompt variable
    - :class:`RemoteConfigModelConfig` — model configuration embedded in RemoteConfig eval config
    - :class:`EvalConfigEntry` — single eval config entry (one per evaluator per projectId)
    - :class:`EvalConfigFile` — RemoteConfig config file root (version + projectId → config array)

Store:
    - :class:`EvalConfigStore` — reads the DB snapshot, not RemoteConfig

Read order: the in-process snapshot (refreshed from the DB). An empty snapshot skips that project's traces.
"""

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.runtime_snapshot import current_eval_config_file
from benchmark.domain.schemas import (
    RemoteConfigModelConfig,
    EvalConfigEntry,
    EvalConfigFile,
    VariableMapping,
)

logger = get_logger(__name__)


def sanitize_eval_config_raw(raw: dict[str, object]) -> dict[str, object]:
    """Drop non-list values under ``configs`` before Pydantic validation.

    Official RemoteConfig examples embed ``_comment_*`` string keys inside
    ``configs``.  Those values are not ``list[EvalConfigEntry]`` and cause
    ``EvalConfigFile.model_validate`` to fail for the **entire** file, which
    the store then caches as empty — every ``projectId`` then appears
    unconfigured and no ``name`` match (``projectId not configured and no name match``).
    """
    configs = raw.get("configs")
    if not isinstance(configs, dict):
        return raw
    cleaned: dict[str, object] = {}
    dropped: list[str] = []
    for key, value in configs.items():
        if isinstance(value, list):
            cleaned[str(key)] = value
        else:
            dropped.append(str(key))
    if dropped:
        logger.warning(
            "Ignoring non-list keys in evaluators.configs (comment/noise): %s",
            dropped,
        )
    return {**raw, "configs": cleaned}


def normalize_agent_name(name: object) -> str:
    """Canonical form for agent-name comparison.

    The page scope id uses hyphens (``amap-agent``); the SDK reports
    ``metadata.agent_name`` with underscores (``amap_agent``). They name the same agent.
    Compare them as lowercase underscores so a separator difference is not treated as a mismatch.
    """
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace("-", "_")


def declared_agent_names(configs: list[EvalConfigEntry]) -> set[str]:
    """Return non-empty ``agentName`` values declared on *configs* (the registered original).

    Keep the form written at registration (for example ``amap-agent``). The S3' missing-name
    injection writes that declared value into the trace's ``metadata.agent_name``. By
    convention that is the agent name stored when the benchmark agent was registered, used
    verbatim with no normalization. Normalization happens only at compare time (see
    :func:`agent_declared_in_evaluators` and consumer S4e) and does not rewrite the injected
    value or the stored owner.
    """
    return {
        cfg.agent_name.strip()
        for cfg in configs
        if isinstance(cfg.agent_name, str) and cfg.agent_name.strip()
    }


def agent_declared_in_evaluators(
    agent_name: object,
    configs: list[EvalConfigEntry],
) -> bool:
    """Whether *agent_name* is allowed by evaluators ``agentName`` declarations.

    - If no entry declares ``agentName``, return ``True`` (gate inactive;
      name/tags/env remain the only filters).
    - Otherwise *agent_name* must match one declared value after
      :func:`normalize_agent_name` canonicalization (``amap-agent`` from the
      web UI scope and ``amap_agent`` reported by the SDK are the same agent).
    """
    allowed = {normalize_agent_name(name) for name in declared_agent_names(configs)}
    if not allowed:
        return True
    normalized = normalize_agent_name(agent_name)
    if not normalized:
        return False
    return normalized in allowed


def evaluator_registration_id(config: EvalConfigEntry) -> str:
    """Registration id for logs. Empty becomes ``-`` (do not fall back to internal / agentName)."""
    return (config.agent_registration_id or "").strip() or "-"


_STREAM_INSTANCE_KEY_PREFIX = "stream:"


def registration_eval_key(
    evaluator_name: str,
    registration_id: str | None = None,
    stream_instance_id: str | None = None,
) -> str:
    """Storage / sampling / lookup key: ``evaluator[@registration][@stream]``.

    Same shape as the log slot :func:`evaluator_slot_label`. Omit registration / stream
    when they are absent. A stream with no registration uses ``@stream:<id>``, distinct
    from the old two-part ``stem@reg``.
    """
    stem = (evaluator_name or "").strip()
    reg = (registration_id or "").strip()
    stream = (stream_instance_id or "").strip()
    parts: list[str] = []
    if stem:
        parts.append(stem)
    if reg:
        parts.append(reg)
    if stream:
        parts.append(stream if reg else f"{_STREAM_INSTANCE_KEY_PREFIX}{stream}")
    return "@".join(parts)


# Split ``stem@reg@stream``: 3 or more parts include stream; exactly 2 parts accept an explicit stream prefix or a legacy reg.
_IDENTITY_PARTS_WITH_STREAM = 3
_IDENTITY_PARTS_TWO = 2


def parse_eval_identity_key(key: object) -> tuple[str, str | None, str | None]:
    """Split a storage / slot key ``stem[@registrationId][@streamInstanceId]``.

    Returns ``(stem_or_raw, registration_id | None, stream_instance_id | None)``.

    - Three or more parts: ``stem@reg@stream``. Stream and reg are not allowed to
      contain ``@``; with extra parts, part 2 is reg and the rest are joined as stream.
    - Two parts: a second part with an explicit ``stream:`` prefix → ``(stem, None, stream)``.
      Otherwise parse as the legacy ``stem@registrationId``. Do not guess the type from a dataset-id prefix.
    - No ``@``: the whole string is a stem, a bare registrationId, or a bare stream (the caller decides).
    """
    text = str(key or "").strip()
    if not text:
        return "", None, None
    parts = [part.strip() for part in text.split("@")]
    parts = [part for part in parts if part]
    if len(parts) >= _IDENTITY_PARTS_WITH_STREAM:
        stem, reg, stream = parts[0], parts[1], "@".join(parts[2:])
        return stem, reg or None, stream or None
    if len(parts) == _IDENTITY_PARTS_TWO:
        stem, second = parts[0], parts[1]
        if second.startswith(_STREAM_INSTANCE_KEY_PREFIX):
            stream = second[len(_STREAM_INSTANCE_KEY_PREFIX) :].strip()
            if stream:
                return stem, None, stream
        return stem, second or None, None
    return text, None, None


def parse_registration_eval_key(key: object) -> tuple[str, str | None]:
    """Split ``stem[@registrationId][@stream]`` → ``(stem, registration_id)``.

    The stream segment is stripped and does not enter registration_id (so ``reg@stream``
    is not mistaken for a registration id). A bare key has registration_id None (the
    caller checks the registry).
    """
    stem, reg, _stream = parse_eval_identity_key(key)
    return stem, reg


def benchmark_run_id_stem(name: object) -> str:
    """Stem used as the ``benchmark_runs.run_id`` prefix (aligned with daily ``{stem}-{yyyymmdd}``).

    A newly stored column may be ``stem@registrationId``, but run_id still uses the bare
    stem (for example ``ieval_external-20261101``). Trend / yesterday comparisons must
    build run_id from this stem, and recognize both the old column ``benchmark_name=stem``
    and the new column ``stem@reg``.
    """
    text = str(name or "").strip()
    if not text:
        return ""
    stem, _reg, _stream = parse_eval_identity_key(text)
    return (stem or text).strip()


def persist_evaluator_name(evaluator_name: str, metadata: object = None) -> str:
    """Storage key for ``trace_pipeline_records.evaluator_name``.

    When ``agent_registration_id`` / ``stream_instance_id`` are set, write the composite
    key so several versions of one stem, and several realtime datasets under one filter,
    stay distinguishable in the stats.
    """
    reg: str | None = None
    stream: str | None = None
    if isinstance(metadata, dict):
        reg = str(metadata.get("agent_registration_id") or "").strip() or None
        stream = str(metadata.get("stream_instance_id") or "").strip() or None
    return registration_eval_key(evaluator_name, reg, stream)


def stored_benchmark_run_name(
    *,
    logical_name: str = "",
    registration_id: str | None = None,
    evaluator_stem: str = "",
) -> str:
    """Storage key for the ``benchmark_runs.benchmark_name`` column.

    When a registrationId is present, write ``evaluator stem@registrationId`` (same shape
    as the realtime ``evaluator_name`` / slot label, so lookup / report / log share it).
    Otherwise write the logical name. The stem is normalized in the services layer and
    passed in as ``evaluator_stem``. Domain does not import services in reverse and does
    not copy the canonical rule.
    """
    logical = (logical_name or "").strip()
    reg = (registration_id or "").strip() or None
    if not reg:
        return logical
    stem = (evaluator_stem or "").strip() or logical
    return registration_eval_key(stem, reg)


def logical_benchmark_run_name(
    *,
    column_name: object = None,
    config: object = None,
) -> str:
    """Recover the logical / display name (component or agent name) from the run column and ``config_json``.

    Prefer ``benchmark_services`` / ``benchmark_component`` / ``agent_registration.name``.
    A column that is already ``stem@reg`` or a bare registrationId is not used as the display name.
    """
    cfg = config if isinstance(config, dict) else {}
    for key in ("benchmark_services", "benchmark_component"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    snap = cfg.get("agent_registration")
    if isinstance(snap, dict):
        name = snap.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        reg = str(snap.get("id") or "").strip()
    else:
        reg = ""
    if not reg:
        reg = str(cfg.get("agent_registration_id") or "").strip()
    col = str(column_name or "").strip()
    stem, col_reg = parse_registration_eval_key(col)
    # Composite key / bare registrationId → display stem (no config fallback).
    if col_reg or (reg and col == reg):
        return stem if (col_reg and stem) else (stem or col)
    if col and col != reg:
        return col
    return col


def run_matches_benchmark_name(  # noqa: PLR0911, PLR0912 — matching logical name / bare id / composite key / stream shapes.
    *,
    column_name: object,
    config: object,
    query_name: str,
) -> bool:
    """list / clone: ``query_name`` may be a logical name, a bare registrationId, ``stem@reg``,
    or ``stem@reg@stream`` (align on registration and ignore stream)."""
    query = (query_name or "").strip()
    if not query:
        return True
    col = str(column_name or "").strip()
    if col == query:
        return True
    _q_stem, q_reg, q_stream = parse_eval_identity_key(query)
    _c_stem, c_reg, c_stream = parse_eval_identity_key(col)
    if q_reg and c_reg and q_reg == c_reg:
        # Same registration: the query omitted stream, or both streams match.
        if not q_stream or not c_stream or q_stream == c_stream:
            return True
    if q_stream and c_stream and q_stream == c_stream and not q_reg and not c_reg:
        return True
    if q_reg and col == q_reg:
        return True
    if c_reg and query == c_reg:
        return True
    cfg = config if isinstance(config, dict) else {}
    for key in ("benchmark_services", "benchmark_component", "agent_registration_id"):
        if str(cfg.get(key) or "").strip() == query:
            return True
    cfg_stream = str(cfg.get("stream_instance_id") or "").strip()
    if q_stream and cfg_stream and q_stream == cfg_stream:
        return True
    snap = cfg.get("agent_registration")
    if isinstance(snap, dict):
        snap_id = str(snap.get("id") or "").strip()
        if snap_id and snap_id in {query, q_reg, c_reg}:
            return True
        if str(snap.get("name") or "").strip() == query:
            return True
    return False


def sample_evaluator_names_for_report(
    *,
    rubric_refs: list[str],
    registration_id: str | None = None,
    stream_instance_id: str | None = None,
) -> list[str]:
    """``evaluator_name__in`` list used when sampling T+1 / kafka reports.

    When a stream is set, sample only the exact ``stem[@reg]@stream`` key. Legacy rows
    with no stream may be read only by a report that also has no stream, so several
    realtime streams do not consume the same historical rows twice.
    """
    stems = [ref.strip() for ref in rubric_refs if ref and str(ref).strip()]
    reg = (registration_id or "").strip() or None
    stream = (stream_instance_id or "").strip() or None
    if stream:
        return list(
            dict.fromkeys(registration_eval_key(stem, reg, stream) for stem in stems if stem)
        )
    if not reg:
        return list(dict.fromkeys(stems))
    keys: list[str] = []
    for stem in stems:
        keys.append(registration_eval_key(stem, reg, None))
    if reg:
        keys.append(reg)
    keys.extend(stems)
    return list(dict.fromkeys(key for key in keys if key))


def eval_name_to_criterion_ids(
    *,
    rubric_refs_to_ids: dict[str, list[str]],
    registration_id: str | None = None,
    stream_instance_id: str | None = None,
) -> dict[str, list[str]]:
    """DB ``evaluator_name`` → criterion id (including aliases such as ``stem@reg@stream``)."""
    mapping: dict[str, list[str]] = {
        ref: list(ids) for ref, ids in rubric_refs_to_ids.items() if ref
    }
    reg = (registration_id or "").strip() or None
    stream = (stream_instance_id or "").strip() or None
    if not reg and not stream:
        return mapping
    # A bare registration_id key still uses the full rubric union (the caller knows reg, not stem).
    all_ids: list[str] = []
    for ids in mapping.values():
        for cid in ids:
            if cid and cid not in all_ids:
                all_ids.append(cid)
    if not all_ids:
        return mapping
# A composite key maps only that evaluator_ref's own criterion ids by stem,
# so a shared-row score is not written into an unrelated criterion under a multi-ref rubric.
    for stem in list(mapping):
        stem_ids = [cid for cid in mapping[stem] if cid]
        if not stem_ids:
            continue
        mapping[registration_eval_key(stem, reg, stream)] = list(stem_ids)
        if reg and stream:
            mapping[registration_eval_key(stem, reg, None)] = list(stem_ids)
    if reg:
        mapping[reg] = all_ids
    return mapping


def evaluator_slot_label(config: EvalConfigEntry) -> str:
    """Slot label for logs and sampling display: stem + registration + stream.

    Several content versions of one ``(name, business, variant)`` **share** one
    ``internal_agent_name`` (design §12.2). Logs must **not** use the internal name to
    tell slots apart. Always use ``evaluator_name@registrationId[@streamInstanceId]``.

    Example: ``example@example--biz--default--v56@ds-rt-1730``. With no suffix, fall back to the stem.
    """
    return (
        registration_eval_key(
            (config.evaluator_name or "").strip() or "<unknown>",
            (config.agent_registration_id or "").strip() or None,
            (config.stream_instance_id or "").strip() or None,
        )
        or "<unknown>"
    )


def evaluator_slot_from_request(evaluator_name: str, metadata: object) -> str:
    """Recover the slot label from an evaluate request (metadata must already carry registration / stream)."""
    stem = (evaluator_name or "").strip() or "<unknown>"
    if not isinstance(metadata, dict):
        return stem
    reg = str(metadata.get("agent_registration_id") or "").strip() or None
    stream = str(metadata.get("stream_instance_id") or "").strip() or None
    return registration_eval_key(stem, reg, stream) or stem


def evaluator_sample_key(trace_id: str, config: EvalConfigEntry) -> str:
    """Deterministic sampling key: several registrations or realtime streams of one stem must stay separate."""
    reg = (config.agent_registration_id or "").strip()
    stream = (config.stream_instance_id or "").strip()
    return f"{trace_id}:{config.evaluator_name}:{reg or '_'}:{stream or '_'}"


def retain_configs_requiring_name_when_multi(
    configs: list[EvalConfigEntry],
) -> tuple[list[EvalConfigEntry], list[EvalConfigEntry]]:
    """When one project has several realtime entries, drop entries that do not filter on ``name``.

    A single entry keeps the old behavior (an empty ``name`` means unrestricted). If several
    entries were allowed to have an empty ``name``, they would also evaluate other agents'
    traffic (for example ``example`` would consume ``amap-agent`` traces).

    Returns ``(kept, dropped)``.
    """
    if len(configs) <= 1:
        return list(configs), []
    kept = [cfg for cfg in configs if cfg.name]
    dropped = [cfg for cfg in configs if not cfg.name]
    return kept, dropped


def ingest_conflict_message(
    existing: list[EvalConfigEntry],
    *,
    evaluator_name: str,
    ingest: str,
) -> str | None:
    """Return an error string when the same project already has this ``evaluator_name`` with a different ``ingest``."""
    want = (ingest or "kafka").strip().lower() or "kafka"
    for cfg in existing:
        if cfg.evaluator_name != evaluator_name:
            continue
        other = (cfg.ingest or "kafka").strip().lower() or "kafka"
        if other != want:
            return (
                f"evaluator {evaluator_name!r} already registered with ingest={other!r}; "
                f"cannot also use ingest={want!r} on the same project "
                "(kafka and sidecar must not share one evaluator)"
            )
    return None


class EvalConfigStore:
    """Read eval config from the DB snapshot.

    Usage::

        store = EvalConfigStore()
        configs = store.get_configs("proj-vehicle-3.0")
        if not configs:
            # projectId is not configured → ack SKIPPED
    """

    def __init__(
        self,
        *,
        data_id: str | None = None,
        group: str | None = None,
        fallback_file: str | None = None,
    ) -> None:
        # Keep the constructor parameters so old callers do not break. These values are no longer read.
        del data_id, group, fallback_file

    def get_config_file(self) -> EvalConfigFile:
        """Return the current snapshot. Empty when no task has been written."""
        snapshot = current_eval_config_file()
        return snapshot if isinstance(snapshot, EvalConfigFile) else EvalConfigFile()

    def get_configs(self, project_id: str) -> list[EvalConfigEntry]:
        """Return eval entries for ``project_id``. An empty list means skip."""
        return self.get_config_file().configs.get(project_id, [])

    def reload(self) -> None:
        """Compatible with old callers. The snapshot is updated by the DB refresh task; this does not read RemoteConfig."""
        return None


__all__ = [
    "RemoteConfigModelConfig",
    "EvalConfigEntry",
    "EvalConfigFile",
    "EvalConfigStore",
    "VariableMapping",
    "agent_declared_in_evaluators",
    "benchmark_run_id_stem",
    "declared_agent_names",
    "eval_name_to_criterion_ids",
    "evaluator_registration_id",
    "evaluator_sample_key",
    "evaluator_slot_from_request",
    "evaluator_slot_label",
    "ingest_conflict_message",
    "logical_benchmark_run_name",
    "normalize_agent_name",
    "parse_eval_identity_key",
    "parse_registration_eval_key",
    "persist_evaluator_name",
    "registration_eval_key",
    "retain_configs_requiring_name_when_multi",
    "run_matches_benchmark_name",
    "sample_evaluator_names_for_report",
    "sanitize_eval_config_raw",
    "stored_benchmark_run_name",
]
