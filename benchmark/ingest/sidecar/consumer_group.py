"""Per-eval-config MetaQ consumer group names and sidecar entry iteration."""

import hashlib
import re
from dataclasses import dataclass

from benchmark.domain.schemas import EvalConfigEntry, EvalConfigFile

_GROUP_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")
_GROUP_MAX_LENGTH = 200


def iter_sidecar_eval_entries(
    config_file: EvalConfigFile,
) -> list[tuple[str, EvalConfigEntry]]:
    """Return sidecar-ingest ``(project_id, entry)`` pairs in stable order."""
    items: list[tuple[str, EvalConfigEntry]] = []
    for project_id in sorted(pid for pid in config_file.configs if pid):
        items.extend(
            (project_id, entry)
            for entry in config_file.configs[project_id]
            if entry.is_sidecar_ingest
        )
    return items


@dataclass(frozen=True, slots=True)
class SidecarConsumerIdentity:
    """MetaQ subscription identity: only ``project_id`` + ``evaluator_name``.

    The ACL allowlist CID is fixed as ``{prefix}_{project}_{evaluator}`` and must not append a
    registration / stream suffix. Multiple stream datasets for the same project+evaluator
    share one consumer, then fan out to each entry on the message side.
    """

    project_id: str
    evaluator_name: str

    @classmethod
    def from_entry(
        cls,
        project_id: str,
        entry: EvalConfigEntry,
    ) -> "SidecarConsumerIdentity":
        return cls(
            project_id=project_id,
            evaluator_name=entry.evaluator_name,
        )


def live_sidecar_entries(
    configs: list[EvalConfigEntry],
    identity: SidecarConsumerIdentity,
) -> list[EvalConfigEntry]:
    """Every ``ingest=sidecar`` entry under the same project that matches this evaluator."""
    return [
        cfg
        for cfg in configs
        if cfg.evaluator_name == identity.evaluator_name and cfg.is_sidecar_ingest
    ]


def consumer_group_for(
    prefix: str,
    identity: SidecarConsumerIdentity,
) -> str:
    """Fixed ACL CID: ``{prefix}_{project_id}_{evaluator_name}``."""
    group = f"{prefix}_{_group_part(identity.project_id)}_{_group_part(identity.evaluator_name)}"
    if len(group) <= _GROUP_MAX_LENGTH:
        return group
    digest_src = f"{identity.project_id}:{identity.evaluator_name}"
    digest = hashlib.sha256(digest_src.encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _group_part(value: str) -> str:
    cleaned = _GROUP_UNSAFE.sub("_", value.strip())
    return cleaned or "x"


__all__ = [
    "SidecarConsumerIdentity",
    "consumer_group_for",
    "iter_sidecar_eval_entries",
    "live_sidecar_entries",
]
