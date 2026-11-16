"""Environment-variable stamp: fixed fields are written into the result, and business fields are kept."""

import json

import pytest
from benchmark.routes.runtime_config import EvalEntryWrite, OdpsJobWrite
from benchmark.settings.setting import settings
from benchmark.domain import odps_export_config as odps_export_config_module
from benchmark.domain.odps_export_config import OdpsExportConfig, OdpsExportJob, fetch_odps_export_config
from benchmark.domain.runtime_config_defaults import stamp_eval_entry, stamp_odps_job
from benchmark.domain.runtime_config_store import replace_odps_export_config
from benchmark.reporting.framework.odps_export import DEFAULT_SQL_TEMPLATE
from pydantic import ValidationError


def test_stamp_eval_entry_writes_env_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "evaluator_environment", "prod, pre")
    monkeypatch.setattr(settings, "evaluator_variable_mapping", "")
    monkeypatch.setattr(settings, "evaluator_model_name", "deepseek-v4-flash")
    monkeypatch.setattr(settings, "evaluator_model_provider", "openai_chat")
    monkeypatch.setattr(settings, "evaluator_model_api_key_env", "AGENT_EVAL_MODEL_API_KEY")
    monkeypatch.setattr(settings, "evaluator_model_base_url", "")
    monkeypatch.setattr(settings, "evaluator_model_temperature", 0.0)
    monkeypatch.setattr(settings, "evaluator_model_max_tokens", None)
    monkeypatch.setattr(settings, "evaluator_model_timeout", None)

    stamped = stamp_eval_entry(
        {
            "evaluator_name": "example",
            "sampling": 0.2,
            "agent_name": "amap-chat",
            "variableMapping": [],
        },
    )

    assert stamped["evaluatorName"] == "example"
    assert stamped["sampling"] == 0.2
    assert stamped["agentName"] == "amap-chat"
    # raw does not carry environment / model: stamp from environment variables (existing page-write semantics stay)
    assert stamped["environment"] == ["prod", "pre"]
    assert stamped["model"]["modelName"] == "deepseek-v4-flash"
    assert stamped["model"]["apiKeyEnv"] == "AGENT_EVAL_MODEL_API_KEY"
    assert stamped["variableMapping"][0]["templateVariable"] == "input"


def test_stamp_eval_entry_omits_empty_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "evaluator_environment", "")
    monkeypatch.setattr(settings, "evaluator_variable_mapping", "")
    monkeypatch.setattr(settings, "evaluator_model_name", "")
    monkeypatch.setattr(settings, "evaluator_model_provider", "")
    monkeypatch.setattr(settings, "evaluator_model_api_key_env", "")
    monkeypatch.setattr(settings, "evaluator_model_base_url", "")
    monkeypatch.setattr(settings, "evaluator_model_max_tokens", None)
    monkeypatch.setattr(settings, "evaluator_model_timeout", None)

    stamped = stamp_eval_entry({"evaluator_name": "amap_agent", "sampling": 1})
    assert "environment" not in stamped
    assert "model" not in stamped
    assert stamped["ingest"] == "kafka"


def test_stamp_eval_entry_raw_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When raw explicitly carries environment / model / prompt, that wins over the environment-variable stamp."""
    monkeypatch.setattr(settings, "evaluator_environment", "prod")
    monkeypatch.setattr(settings, "evaluator_variable_mapping", "")
    monkeypatch.setattr(settings, "evaluator_model_name", "env-model")
    monkeypatch.setattr(settings, "evaluator_model_provider", "openai")
    monkeypatch.setattr(settings, "evaluator_model_api_key_env", "ENV_KEY")
    monkeypatch.setattr(settings, "evaluator_model_base_url", "")
    monkeypatch.setattr(settings, "evaluator_model_temperature", 0.0)
    monkeypatch.setattr(settings, "evaluator_model_max_tokens", None)
    monkeypatch.setattr(settings, "evaluator_model_timeout", None)

    stamped = stamp_eval_entry(
        {
            "evaluator_name": "amap_agent",
            "sampling": 0.5,
            "environment": ["pre"],
            "prompt": "打分时注意安全性 ${input}",
            "model": {"modelName": "page-model", "apiKeyEnv": "PAGE_KEY"},
        },
    )
    assert stamped["environment"] == ["pre"]
    assert stamped["prompt"] == "打分时注意安全性 ${input}"
    assert stamped["model"]["modelName"] == "page-model"
    assert stamped["model"]["apiKeyEnv"] == "PAGE_KEY"


def test_stamp_eval_entry_raw_empty_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When raw does not carry them (or they are empty), keep the existing behavior of stamping from environment variables."""
    monkeypatch.setattr(settings, "evaluator_environment", "prod, daily")
    monkeypatch.setattr(settings, "evaluator_variable_mapping", "")
    monkeypatch.setattr(settings, "evaluator_model_name", "env-model")
    monkeypatch.setattr(settings, "evaluator_model_provider", "openai")
    monkeypatch.setattr(settings, "evaluator_model_api_key_env", "ENV_KEY")
    monkeypatch.setattr(settings, "evaluator_model_base_url", "")
    monkeypatch.setattr(settings, "evaluator_model_temperature", 0.0)
    monkeypatch.setattr(settings, "evaluator_model_max_tokens", None)
    monkeypatch.setattr(settings, "evaluator_model_timeout", None)

    stamped = stamp_eval_entry(
        {
            "evaluator_name": "amap_agent",
            "sampling": 1,
            "environment": [],
            "model": {},
        },
    )
    assert stamped["environment"] == ["prod", "daily"]
    assert stamped["model"]["modelName"] == "env-model"


def test_variable_mapping_rejects_non_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "evaluator_variable_mapping", json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="JSON 数组"):
        stamp_eval_entry({"evaluator_name": "example", "sampling": 0.1})


def test_stamp_odps_job_writes_env_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "odps_schedule_timezone", "Asia/Shanghai")
    monkeypatch.setattr(settings, "odps_object_name", "benchmark_final_results.csv")
    monkeypatch.setattr(settings, "odps_sql_template", "")
    monkeypatch.setattr(settings, "odps_default_limit", 200)

    stamped = stamp_odps_job(
        {
            "id": "search",
            "schedule_time": "10:30",
            "oss_prefix": "search-eval/benchmark_result/",
            "project_id": "proj-1",
            "sql_template": "SELECT 1",
            "schedule_timezone": "UTC",
            "object_name": "other.csv",
        },
    )

    assert stamped["schedule_timezone"] == "Asia/Shanghai"
    assert stamped["object_name"] == "benchmark_final_results.csv"
    assert stamped["sql_template"] == DEFAULT_SQL_TEMPLATE
    assert stamped["limit"] == 200
    assert stamped["project_id"] == "proj-1"


def test_page_models_reject_env_fields() -> None:
    with pytest.raises(ValidationError):
        EvalEntryWrite.model_validate(
            {
                "project_id": "p",
                "evaluator_name": "example",
                "sampling": 0.1,
                "environment": ["prod"],
            },
        )
    with pytest.raises(ValidationError):
        OdpsJobWrite.model_validate(
            {
                "schedule_time": "10:30",
                "oss_prefix": "a/",
                "sql_template": "SELECT 1",
            },
        )


def test_fetch_odps_export_config_merges_remote_config_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ODPS is not masked: RemoteConfig is the base, a DB job with the same id overrides it, and a job that exists only on RemoteConfig is kept."""
    replace_odps_export_config(
        OdpsExportConfig(
            jobs=[
                OdpsExportJob(id="biz-a", sql_template="SELECT 'db'"),
            ],
        ),
    )
    remote_config = OdpsExportConfig(
        jobs=[
            OdpsExportJob(id="default", sql_template="SELECT 'remote_config-default'"),
            OdpsExportJob(id="biz-a", sql_template="SELECT 'remote_config-a'"),
        ],
    )
    monkeypatch.setattr(
        odps_export_config_module,
        "_fetch_remote_config_odps_config",
        lambda **_: remote_config,
    )

    jobs = {job.id: job for job in fetch_odps_export_config().resolved_jobs()}
    assert jobs["default"].sql_template == "SELECT 'remote_config-default'"
    # Same id: the page job overrides RemoteConfig
    assert jobs["biz-a"].sql_template == "SELECT 'db'"

    # RemoteConfig unavailable or unconfigured → fall back to the DB snapshot
    monkeypatch.setattr(
        odps_export_config_module,
        "_fetch_remote_config_odps_config",
        lambda **_: OdpsExportConfig(),
    )
    assert [job.id for job in fetch_odps_export_config().resolved_jobs()] == ["biz-a"]
    replace_odps_export_config(OdpsExportConfig())
