"""Stamp environment-level fixed fields onto eval / ODPS config submitted by the page.

The page submits only business-variable fields. Before the save to DB, this module fills
in the fields the business does not specify, using process env vars. If the request
already carries those fields, the environment values still overwrite them.
"""

import json
import re
from collections.abc import Mapping

from benchmark.settings.setting import Settings, settings
from benchmark.domain.odps_export_config import (
    DEFAULT_OBJECT_NAME,
    DEFAULT_SCHEDULE_TIMEZONE,
    DEFAULT_SQL_TEMPLATE,
    OdpsExportJob,
)
from benchmark.domain.schemas import EvalConfigEntry

_SCHEDULE_TIME = re.compile(r"^\d{2}:\d{2}$")

#: Default mapping written to the DB when AGENT_EVAL_EVALUATOR_VARIABLE_MAPPING is unset.
DEFAULT_VARIABLE_MAPPING: list[dict[str, object]] = [
    {
        "templateVariable": "input",
        "traceObject": "observation",
        "selectedColumnId": "input",
        "jsonSelector": None,
    },
    {
        "templateVariable": "output",
        "traceObject": "observation",
        "selectedColumnId": "output",
        "jsonSelector": None,
    },
    {
        "templateVariable": "metadata",
        "traceObject": "observation",
        "selectedColumnId": "metadata",
        "jsonSelector": None,
    },
]


def _active_settings(app_settings: Settings | None) -> Settings:
    return app_settings or settings


def evaluator_environment(app_settings: Settings | None = None) -> list[str] | None:
    """Parse ``AGENT_EVAL_EVALUATOR_ENVIRONMENT``. An empty string means any environment."""
    raw = _active_settings(app_settings).evaluator_environment.strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parts or None


def evaluator_variable_mapping(app_settings: Settings | None = None) -> list[dict[str, object]]:
    """Parse the variable mapping. An empty env var uses the default three-part mapping.

    **fail-fast** (intentional): illegal JSON or a non-array raises ``ValueError`` and
    does not fall back to the default mapping. stamp_eval_entry → this function is the
    save path; a silent fallback would write the default mapping into the DB from a bad
    env var, drifting config in a way that is hard to trace. A failed save exposes the
    deploy error immediately. Only "unset" (empty string) takes the default.
    """
    raw = _active_settings(app_settings).evaluator_variable_mapping.strip()
    if not raw:
        return [dict(item) for item in DEFAULT_VARIABLE_MAPPING]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = (
            f"AGENT_EVAL_EVALUATOR_VARIABLE_MAPPING 非法 JSON"
            f"（fail-fast，不回落默认映射）: {exc.msg}"
        )
        raise ValueError(msg) from exc
    if not isinstance(parsed, list):
        msg = "AGENT_EVAL_EVALUATOR_VARIABLE_MAPPING 必须是 JSON 数组（fail-fast，不回落默认映射）"
        raise ValueError(msg)
    return parsed


def evaluator_model_override(app_settings: Settings | None = None) -> dict[str, object] | None:
    """Build the model override. None when the key fields are all empty; execution falls back to YAML."""
    cfg = _active_settings(app_settings)
    model: dict[str, object] = {"temperature": cfg.evaluator_model_temperature}
    if cfg.evaluator_model_provider.strip():
        model["provider"] = cfg.evaluator_model_provider.strip()
    if cfg.evaluator_model_name.strip():
        model["modelName"] = cfg.evaluator_model_name.strip()
    if cfg.evaluator_model_base_url.strip():
        model["baseUrl"] = cfg.evaluator_model_base_url.strip()
    if cfg.evaluator_model_api_key_env.strip():
        model["apiKeyEnv"] = cfg.evaluator_model_api_key_env.strip()
    if cfg.evaluator_model_max_tokens is not None:
        model["maxTokens"] = cfg.evaluator_model_max_tokens
    if cfg.evaluator_model_timeout is not None:
        model["timeout"] = cfg.evaluator_model_timeout
    if not any(model.get(key) for key in ("modelName", "baseUrl", "apiKeyEnv")):
        return None
    return model


def resolved_odps_sql_template(app_settings: Settings | None = None) -> str:
    """SQL template: the env var wins, otherwise the in-process built-in template."""
    raw = _active_settings(app_settings).odps_sql_template.strip()
    return raw or DEFAULT_SQL_TEMPLATE


def stamp_eval_entry(
    raw: Mapping[str, object],
    app_settings: Settings | None = None,
) -> dict[str, object]:
    """Fill one eval config from env vars and validate it as ``EvalConfigEntry``."""
    # Variable mapping: an explicit raw value (for example a realtime dataset mapping) wins.
    raw_mapping = raw.get("variable_mapping") or raw.get("variableMapping")
    payload: dict[str, object] = {
        "evaluatorName": raw.get("evaluator_name") or raw.get("evaluatorName"),
        "sampling": raw.get("sampling"),
        "ingest": raw.get("ingest") or "kafka",
        "variableMapping": (
            raw_mapping
            if isinstance(raw_mapping, list) and raw_mapping
            else evaluator_variable_mapping(app_settings)
        ),
    }
    agent_name = raw.get("agent_name", raw.get("agentName"))
    if isinstance(agent_name, str) and agent_name.strip():
        payload["agentName"] = agent_name.strip()
    for source, target in (
        ("name", "name"),
        ("tags", "tags"),
        ("start_time", "startTime"),
        ("end_time", "endTime"),
        ("prompt", "prompt"),
        # Business customization (written from the registration at realtime enroll; omitted by default).
        ("agent_registration_id", "agentRegistrationId"),
        ("stream_instance_id", "streamInstanceId"),
        ("internal_agent_name", "internalAgentName"),
        ("prompt_bundle_ref", "promptBundleRef"),
        ("skill_bundle_ref", "skillBundleRef"),
        ("attribution_execution", "attributionExecution"),
    ):
        value = raw.get(source, raw.get(target))
        if value not in (None, "", []):
            payload[target] = value
    # environment / model: an explicit raw value (sampling conditions, page-level model
    # override) wins; env vars are stamped only when the request did not carry them.
    raw_environment = raw.get("environment")
    if isinstance(raw_environment, list) and raw_environment:
        payload["environment"] = raw_environment
    else:
        environment = evaluator_environment(app_settings)
        if environment is not None:
            payload["environment"] = environment
    raw_model = raw.get("model")
    if isinstance(raw_model, dict) and raw_model:
        payload["model"] = raw_model
    else:
        model = evaluator_model_override(app_settings)
        if model is not None:
            payload["model"] = model
    entry = EvalConfigEntry.model_validate(payload)
    return entry.model_dump(by_alias=True, exclude_none=True)


def stamp_odps_job(
    raw: Mapping[str, object],
    app_settings: Settings | None = None,
) -> dict[str, object]:
    """Fill one ODPS job from env vars and validate it as ``OdpsExportJob``."""
    cfg = _active_settings(app_settings)
    limit = raw.get("limit")
    if limit is None or limit == "":
        limit = cfg.odps_default_limit
    job = OdpsExportJob.model_validate(
        {
            "id": raw.get("id") or "default",
            "enabled": True if raw.get("enabled") is None else raw.get("enabled"),
            "schedule_time": str(raw.get("schedule_time") or raw.get("scheduleTime") or "").strip(),
            "schedule_timezone": cfg.odps_schedule_timezone.strip() or DEFAULT_SCHEDULE_TIMEZONE,
            "limit": limit,
            "oss_prefix": str(raw.get("oss_prefix") or raw.get("ossPrefix") or "").strip(),
            "object_name": cfg.odps_object_name.strip() or DEFAULT_OBJECT_NAME,
            "project_id": str(raw.get("project_id") or raw.get("projectId") or "").strip(),
            "sql_template": resolved_odps_sql_template(cfg),
        },
    )
    if not _SCHEDULE_TIME.match(job.schedule_time):
        msg = "schedule_time 必须是 HH:MM"
        raise ValueError(msg)
    if not job.oss_prefix:
        msg = "oss_prefix 不能为空"
        raise ValueError(msg)
    if "{project_id}" in job.sql_template and not job.project_id:
        msg = "sql_template 含 {project_id}，必须填写 project_id"
        raise ValueError(msg)
    return job.model_dump()
