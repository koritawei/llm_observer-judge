"""Load trace instance credentials.

:func:`load_trace_instances` decodes ``AGENT_EVAL_TRACE_INSTANCES`` (base64 JSON
array). Each element is one :class:`TraceEndpoint`, keyed by ``projectId`` /
``project_id`` (``name`` is the key only when those are absent).

:func:`select_trace_instances_by_name` keeps the entries whose ``name`` matches.
The wanted name comes from ``custom.trace.instance_name``.

Keys must match ``EvaluatorRequest.project_id``.
"""

import base64
import binascii
import json
from pathlib import Path
from typing import Any

import yaml
from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError
from pydantic.alias_generators import to_camel

logger = get_logger("agent_eval.trace_instances")


class TraceEndpoint(BaseModel):
    """One Langfuse deployment, addressed by ``project_id``.

    Credentials are per project. ``metadata_key`` is used **only at eval
    entry** to filter business id → Langfuse primary id.

    The operator config is supplied as camelCase JSON (``baseUrl`` /
    ``publicKey`` / ``secretKey``), so we accept that via an alias generator
    while still allowing snake_case construction (``populate_by_name``).

    The credential model lives in the config layer (beside this loader):
    services.trace_resolver imports the model from this module at top level. Defining it
    on the services side would make this module import services in reverse and create a
    config → services → config cycle.
    """

    model_config = ConfigDict(
        frozen=True,
        alias_generator=to_camel,
        populate_by_name=True,
    )

    base_url: str
    public_key: str
    secret_key: SecretStr
    metadata_key: str = "trace_id"
    #: Optional operator-facing label, echoed in logs.
    name: str | None = None


def _instance_key(item: dict[str, Any]) -> str | None:
    for field in ("projectId", "project_id"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    name = item.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def load_trace_instances(encoded: str) -> dict[str, TraceEndpoint]:
    """Decode env payload into ``project_id`` → :class:`LangfuseInstance`."""

    if not encoded.strip():
        return {}

    try:
        raw = base64.b64decode(encoded, validate=True)
        items = json.loads(raw)
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("trace_instances.decode_failed error=%s", exc)
        return {}

    if not isinstance(items, list):
        logger.warning("trace_instances.invalid_shape expected=array")
        return {}

    result: dict[str, TraceEndpoint] = {}
    for item in items:
        if not isinstance(item, dict):
            logger.warning("trace_instances.skip_non_object item=%r", item)
            continue
        key = _instance_key(item)
        if key is None:
            logger.warning("trace_instances.missing_project_key item=%s", item)
            continue
        try:
            result[key] = TraceEndpoint.model_validate(item)
        except ValidationError as exc:
            logger.warning(
                "trace_instances.invalid_item project_key=%s error=%s",
                key,
                exc,
            )
    return result


__all__ = [
    "TraceEndpoint",
    "load_trace_instances",
    "read_trace_instance_name",
    "select_trace_instances_by_name",
]


def select_trace_instances_by_name(
    encoded: str,
    name: str,
) -> dict[str, TraceEndpoint]:
    """Instances from ``encoded`` whose ``name`` equals ``name``.

    The dict key stays the project id from :func:`load_trace_instances`.
    """
    wanted = name.strip()
    if not wanted:
        return {}
    return {
        key: inst
        for key, inst in load_trace_instances(encoded).items()
        if inst.name == wanted
    }


def read_trace_instance_name(yaml_path: Path) -> str | None:
    """``custom.trace.instance_name`` from ``conf/app.<env>.yaml``.

    Missing file, section, or blank value → ``None``.
    """
    path = Path(yaml_path)
    if not path.exists():
        return None
    try:
        with path.open() as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("trace_instances.yaml_read_failed path=%s error=%s", path, exc)
        return None
    custom = data.get("custom") or {}
    section = custom.get("trace") if isinstance(custom, dict) else None
    if not isinstance(section, dict):
        return None
    name = section.get("instance_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None
