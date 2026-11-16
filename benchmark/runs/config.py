"""Benchmark runtime config from ``custom.benchmark`` in ``conf/app.<env>.yaml``.

Storage settings (bucket / endpoint / cdn / credential_mode) are **not** in
YAML — they are read from RemoteConfig in ``benchmark.domain.report_storage_config``,
following the agent-eval RemoteConfig pattern (data_id: ``agent-eval.benchmark.oss``).

The env var ``BENCHMARK_WEBDEV_AGENT_URL`` is read in this module (the config layer);
it takes priority over YAML. Business code does not read the env var directly.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmark.constant import RUN_STALE_TIMEOUT_S

#: Env var that overrides ``webdev_agent_url`` (read here, in the config layer).
WEBDEV_AGENT_URL_ENV = "BENCHMARK_WEBDEV_AGENT_URL"

#: External trigger — callback URL base for Aone → agent-eval callbacks.
EXTERNAL_CALLBACK_URL_BASE_ENV = "EXTERNAL_CALLBACK_URL_BASE"
#: External trigger — Aone judge service base URL.
EXTERNAL_JUDGE_BASE_URL_ENV = "EXTERNAL_JUDGE_BASE_URL"
#: External trigger — Aone attribution service base URL.
EXTERNAL_ATTRIBUTION_BASE_URL_ENV = "EXTERNAL_ATTRIBUTION_BASE_URL"
#: External trigger — Aone X-API-Key (pipeline trigger auth).
EXTERNAL_API_KEY_ENV = "EXTERNAL_API_KEY"
#: External trigger — Aone rubric-scoring pipeline ID (judge).
EXTERNAL_RUBRIC_PIPELINE_ID_ENV = "EXTERNAL_RUBRIC_PIPELINE_ID"
#: External trigger — Aone attribution pipeline ID (separate from judge).
EXTERNAL_ATTRIBUTION_PIPELINE_ID_ENV = "EXTERNAL_ATTRIBUTION_PIPELINE_ID"
#: Badcase dispatch — local ``a1`` CLI executable override (setenv.sh injects
#: the in-container path; debug envs take precedence over yaml ``a1_bin``).
A1_CLI_PATH_ENV = "AGENT_EVAL_NOTIFICATION_MCP_CLI_PATH"
#: Badcase dispatch — fallback ``a1`` CLI executable override.
A1_BIN_ENV = "A1_BIN"
#: Badcase dispatch — ``a1 link`` working directory (unset → parent cwd).
A1_PROJECT_DIR_ENV = "A1_PROJECT_DIR"
#: Badcase dispatch — optional workitem URL template and fallback project.
NOTIFICATION_WORKITEM_URL_TEMPLATE_ENV = "NOTIFICATION_WORKITEM_URL_TEMPLATE"
NOTIFICATION_PROJECT_ID_ENV = "NOTIFICATION_PROJECT_ID"


class FusedClassifierConfig(BaseModel):
    """Optional fused-triage score thresholds (reserved for future wiring)."""

    model_config = {"extra": "ignore"}

    badcase_threshold: float = 0.5
    watch_threshold: float = 0.8


class BenchmarkConfig(BaseModel):
    """``custom.benchmark`` — agent-eval Benchmark Runner / API defaults.

    OSS-related settings (``report_upload_to_oss``, bucket, endpoint, etc.)
    are intentionally absent — read from RemoteConfig in ``benchmark.domain.report_storage_config``.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    default_max_concurrency: int = Field(default=10, ge=1)
    default_parallel_evaluators: bool = True
    evaluate_timeout_s: int = Field(default=300, ge=1)
    #: Heartbeat / created_at cutoff for recover_stale_runs after crash/deploy.
    #: Defaults to ``RUN_STALE_TIMEOUT_S`` (constant.py), and must be >
    #: ``evaluate_timeout_s``, or a run is marked dead while its evaluator call is still in progress.
    stale_heartbeat_timeout_s: int = Field(default=RUN_STALE_TIMEOUT_S, ge=1)

    @model_validator(mode="after")
    def _stale_timeout_above_evaluate(self) -> "BenchmarkConfig":
        """``stale_heartbeat_timeout_s`` must be > ``evaluate_timeout_s``.

        Otherwise a run whose long evaluator call is still in progress is marked dead too early
        by recover_stale_runs and executed again (historical incident: stale 120s < evaluate 540s).
        """
        if self.stale_heartbeat_timeout_s <= self.evaluate_timeout_s:
            raise ValueError(
                f"stale_heartbeat_timeout_s ({self.stale_heartbeat_timeout_s}) "
                f"必须 > evaluate_timeout_s ({self.evaluate_timeout_s})"
            )
        return self

    fused_classifier: FusedClassifierConfig = Field(
        default_factory=FusedClassifierConfig,
    )
    #: webdev-agent A2A endpoint; env ``BENCHMARK_WEBDEV_AGENT_URL`` wins
    #: over this YAML value (applied in :func:`load_benchmark_config`).
    webdev_agent_url: str = "http://webdev-agent:9880/"
    #: HTTP timeout for webdev-agent report generation (LLM call can be slow).
    webdev_report_timeout_s: float = Field(default=3600.0, ge=1)
    #: External trigger — callback URL base for Aone callbacks.
    #: env ``EXTERNAL_CALLBACK_URL_BASE`` wins over this default.
    external_callback_url_base: str = "http://agent-eval"
    #: External trigger — judge service base URL (empty → LocalEvalService).
    #: env ``EXTERNAL_JUDGE_BASE_URL`` wins over this default.
    external_judge_base_url: str = ""
    #: External trigger — attribution service base URL (empty → skip attribution).
    #: env ``EXTERNAL_ATTRIBUTION_BASE_URL`` wins over this default.
    external_attribution_base_url: str = ""
    #: External trigger — API key for pipeline trigger auth.
    #: env ``EXTERNAL_API_KEY`` wins over this default.
    external_api_key: str = ""
    #: External trigger — rubric-scoring pipeline ID (judge).
    #: env ``EXTERNAL_RUBRIC_PIPELINE_ID`` wins over this default.
    external_rubric_pipeline_id: str = ""
    #: External trigger — attribution pipeline ID (separate from judge).
    #: env ``EXTERNAL_ATTRIBUTION_PIPELINE_ID`` wins over this default.
    external_attribution_pipeline_id: str = ""


def load_benchmark_config(yaml_path: Path) -> BenchmarkConfig:
    """Load ``custom.benchmark`` from app yaml; missing section → defaults.

    If set, the ``BENCHMARK_WEBDEV_AGENT_URL`` env overrides YAML ``webdev_agent_url``,
    so a deploy can point at another instance without editing config.
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f) or {}
    raw = (data.get("custom") or {}).get("benchmark") or {}
    cfg = BenchmarkConfig.model_validate(raw)
    env_url = os.environ.get(WEBDEV_AGENT_URL_ENV)
    if env_url:
        cfg = cfg.model_copy(update={"webdev_agent_url": env_url})

    # External trigger env overrides (§5.1).
    _ext_cb = os.environ.get(EXTERNAL_CALLBACK_URL_BASE_ENV)
    if _ext_cb:
        cfg = cfg.model_copy(update={"external_callback_url_base": _ext_cb})
    _external_judge = os.environ.get(EXTERNAL_JUDGE_BASE_URL_ENV)
    if _external_judge:
        cfg = cfg.model_copy(update={"external_judge_base_url": _external_judge})
    _external_attr = os.environ.get(EXTERNAL_ATTRIBUTION_BASE_URL_ENV)
    if _external_attr:
        cfg = cfg.model_copy(update={"external_attribution_base_url": _external_attr})
    _notification_key = os.environ.get(EXTERNAL_API_KEY_ENV)
    if _notification_key:
        cfg = cfg.model_copy(update={"external_api_key": _notification_key})
    _external_pipeline = os.environ.get(EXTERNAL_RUBRIC_PIPELINE_ID_ENV)
    if _external_pipeline:
        cfg = cfg.model_copy(update={"external_rubric_pipeline_id": _external_pipeline})
    _external_attr_pipeline = os.environ.get(EXTERNAL_ATTRIBUTION_PIPELINE_ID_ENV)
    if _external_attr_pipeline:
        cfg = cfg.model_copy(update={"external_attribution_pipeline_id": _external_attr_pipeline})
    return cfg


def a1_cli_env_overrides() -> tuple[str | None, str | None]:
    """Read local-debug env overrides for the ``a1`` CLI.

    Returns ``(a1 executable path override, ``a1 link`` working directory)``.
    Both default to ``None`` so production behavior follows the injected
    environment / yaml ``badcase_attribution`` config — see
    ``badcase_attribution/aone.py`` and ``docs/config-and-deployment.md``.
    """
    return (
        os.environ.get(A1_CLI_PATH_ENV) or os.environ.get(A1_BIN_ENV),
        os.environ.get(A1_PROJECT_DIR_ENV) or None,
    )


def notification_workitem_url_env_overrides() -> tuple[str | None, str | None]:
    """Read the workitem URL template and fallback project in the config layer."""
    return (
        os.environ.get(NOTIFICATION_WORKITEM_URL_TEMPLATE_ENV),
        os.environ.get(NOTIFICATION_PROJECT_ID_ENV),
    )


__all__ = [
    "A1_BIN_ENV",
    "A1_CLI_PATH_ENV",
    "A1_PROJECT_DIR_ENV",
    "EXTERNAL_API_KEY_ENV",
    "NOTIFICATION_PROJECT_ID_ENV",
    "NOTIFICATION_WORKITEM_URL_TEMPLATE_ENV",
    "EXTERNAL_ATTRIBUTION_BASE_URL_ENV",
    "EXTERNAL_ATTRIBUTION_PIPELINE_ID_ENV",
    "EXTERNAL_JUDGE_BASE_URL_ENV",
    "EXTERNAL_RUBRIC_PIPELINE_ID_ENV",
    "BenchmarkConfig",
    "EXTERNAL_CALLBACK_URL_BASE_ENV",
    "FusedClassifierConfig",
    "WEBDEV_AGENT_URL_ENV",
    "a1_cli_env_overrides",
    "notification_workitem_url_env_overrides",
    "load_benchmark_config",
]
