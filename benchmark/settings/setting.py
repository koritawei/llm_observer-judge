"""Process-level config for agent_eval.

Environment variables are managed in one place with Pydantic Settings, following the
same pattern as ``services/amap-agent/app/config/env.py``: every process-level env var
is declared here, and business code reads :class:`Settings` instead of calling ``os.getenv``.
"""

import os
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Lenient map of common bool env-var typos to the intended value (crash-loop protection).
#: 2026-09-28 incident: BENCHMARK_STREAMING_ENABLED was written as 'fasle', pydantic bool
#: parsing failed, Settings() raised at import, and the service crash-looped.
#: Ordinary true/false/1/0/on/off still use pydantic's parser; this only covers known typos.
_ENV_BOOL_TYPOS: dict[str, bool] = {
    "ture": True,
    "flase": False,
    "fasle": False,
    "fals": False,
    "yes": True,
    "no": False,
}


def _coerce_env_bool(value: object) -> object:
    """Lenient bool env-var normalize: known typos are mapped, everything else goes to pydantic."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ENV_BOOL_TYPOS:
            return _ENV_BOOL_TYPOS[normalized]
    return value


_LOCAL_SPAN_LOG_DEFAULT_MAX_CHARS = 12_000


class LocalSpanLogSettings(BaseSettings):
    """Report-diagnostics switch read on use. Does not build other runtime config."""

    model_config = SettingsConfigDict(env_ignore_empty=True, extra="ignore")

    enabled: bool = Field(default=False, validation_alias="AGENT_EVAL_LOCAL_SPAN_LOG")
    max_chars: int = Field(
        default=_LOCAL_SPAN_LOG_DEFAULT_MAX_CHARS,
        validation_alias="AGENT_EVAL_LOCAL_SPAN_LOG_MAX_CHARS",
    )

    @field_validator("enabled", mode="before")
    @classmethod
    def _read_enabled(cls, value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @field_validator("max_chars", mode="before")
    @classmethod
    def _read_max_chars(cls, value: object) -> int:
        try:
            parsed = int(str(value).strip())
        except ValueError:
            return _LOCAL_SPAN_LOG_DEFAULT_MAX_CHARS
        return parsed if parsed > 0 else _LOCAL_SPAN_LOG_DEFAULT_MAX_CHARS


class Settings(BaseSettings):
    """Process-level environment variables.

    Field names are snake_case. pydantic-settings binds case-insensitively, so env vars
    such as ``LOG_DIR`` / ``DATABASE_URL`` map automatically.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_EVAL_",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    # -- Server ----------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # -- Database --------------------------------------------------------
    database_url: str = Field(default="sqlite:///./agent_eval.db")
    environment: str = Field(default="development")

    # -- Evaluator runtime ----------------------------------------------
    # Selects conf/app.<env>.yaml for the agentgraph AppRuntime (evaluators).
    # Distinct from `environment`, which drives the database backend.
    env: str = Field(default="daily")

    # -- API docs (OpenAPI/Swagger) --------------------------------------
    # Security line (ALIBABA-2026-46066077 swagger leak): pre/prod must not expose
    # /docs, /redoc, or /openapi.json. None=automatic (open only on daily). An explicit
    # true/false overrides it (a temporary pre enable needs a change approval, then turn it off).
    docs_enabled: bool | None = Field(default=None)

    # -- Logging ---------------------------------------------------------
    # Deploy: setenv.sh mirrors LOG_LEVEL ↔ AGENT_EVAL_LOG_LEVEL (prod→WARNING).
    # Runtime also honors conf/app.<env>.yaml observability.log_level when LOG_LEVEL unset.
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="")
    log_filename: str = Field(default="agent_eval.log")
    enable_file_log: bool = Field(default=True)

    # -- Debug flow tracing ---------------------------------------------
    flow_log_path: str = Field(default="")
    debug_flow_enabled: bool = Field(default=False)
    debug_flow_retention_seconds: int = Field(default=3600)
    debug_flow_max_events: int = Field(default=1000)

    # -- Cluster report (batch clustering report) -----------------------------------
    # Injected via env vars (production: Aone app config; local: .env). Names carry the
    # env_prefix AGENT_EVAL_. Do not hardcode bucket / endpoint / resource ids in code
    # (see cluster-report.md in the vehicle-agent-eval skill).
    cluster_report_local_root: str = Field(
        default="",
        description="聚类报告产物根目录；空=落服务目录下 tmp/cluster_reports/",
    )
    cluster_report_project_id: str = Field(
        default="",
        description=(
            "性能采集用的 Langfuse project_id；配了多个实例时必须显式指定，"
            "否则取第一个实例，可能因凭证不匹配导致性能块整体失败"
        ),
    )
    # OSS artifact archive: if bucket or endpoint is empty, fall back to local files only
    # (report generation is unaffected). The bucket is private and is reached by a signed URL.
    # Resource ids live in the ops docs, not in code.
    cluster_report_oss_bucket: str = Field(
        default="", description="聚类报告产物 OSS bucket；空=未配（纯本地）"
    )
    cluster_report_oss_endpoint: str = Field(
        default="",
        description=(
            "OSS endpoint（host，可带/不带 scheme）；生产用内网 -internal 域名，"
            "本地从公网访问私有桶需用公网域名"
        ),
    )
    cluster_report_oss_public_endpoint: str = Field(
        default="",
        description=(
            "签名 URL 使用的公网 endpoint；空=从 endpoint 推导（去 -internal 后缀）。"
            "上传走内网省流量，但签名链接要给 VPC 外的人打开，必须用公网域名；"
            "OSS 签名不含 host，同一签名在公网域名上同样有效"
        ),
    )
    cluster_report_oss_credential_mode: str = Field(
        default="provider",
        description=(
            "凭证模式：provider（缺省；Normandy 资源绑定，无需显式 AK/SK）/ "
            "env（本地兜底，读 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET）"
        ),
    )
    cluster_report_oss_key_prefix: str = Field(
        default="eval-reports",
        description="object key 前缀；key = {prefix}/{batch_tag}/{文件名}",
    )
    cluster_report_oss_sign_expires_s: int = Field(
        default=604800,
        description=(
            "签名 URL 有效期（秒），缺省 604800=7d。生产凭证是长期 AK（has_sts=False），"
            "签名不受 STS token 寿命封顶；报告常需跨日/跨周传阅，过短会让已发出的链接提前失效"
        ),
    )
    cluster_report_dingtalk_webhook: str = Field(
        default="",
        description=(
            "评测器隔离的 latest 指针推送钉钉机器人 Webhook；空=不推送。"
            "必须经受控运行时密钥配置注入，禁止写入代码、配置文件或日志。"
        ),
    )
    cluster_report_scheduler_enabled: bool = Field(
        default=False,
        description=(
            "本实例是否启动 Cluster Report 定时触发器；false 时忽略 RemoteConfig/YAML 的 schedule，"
            "不影响手动触发、已有 Job 的恢复接管与查询接口。"
        ),
    )
    cluster_report_evaluator_name: str = Field(
        default="vehicle_agent",
        description="cluster report 查询与 JobStore 使用的评测器名；VUI 日报设为 vui_agent。",
    )
    cluster_report_include_trace_meta: bool = Field(
        default=False,
        description=(
            "report.json 是否保留每个 case 的 trace_meta；默认 false，原始评测 IO 始终保留在 PG。"
        ),
    )
    cluster_report_perf_concurrency: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Langfuse 性能与路由采集的最大并发数。",
    )
    cluster_report_perf_min_interval_s: float = Field(
        default=0.5,
        ge=0,
        le=10,
        description="Langfuse 每条 trace 性能或路由采集的最小启动间隔（秒）。",
    )
    cluster_report_remote_config_data_id: str = Field(
        default="agent-eval.cluster-report",
        description="Cluster Report 运行时配置的 RemoteConfig data_id。",
    )
    cluster_report_remote_config_group: str = Field(
        default="agent-eval.service.group",
        description="Cluster Report 运行时配置的 RemoteConfig group。",
    )

    # -- OpenAI / LLM proxy ---------------------------------------------
    llm_proxy_timeout_seconds: float = Field(default=30.0)

    # -- Report insight LLM (today's summary + top issues; keys isolated from judge/example/proxy) --
    # Only AGENT_EVAL_INSIGHT_LLM_API_KEY / AMAP_LLM_API_KEY are read. Empty skips the model rewrite.
    insight_llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "AGENT_EVAL_INSIGHT_LLM_API_KEY",
            "AMAP_LLM_API_KEY",
        ),
        description="Report-insight-only Amap LLM API key (not judge/example).",
    )
    insight_llm_base_url: str = Field(
        # OpenAI-compatible chat completions live at /open_api/v1 (same as conf/*.yaml and the
        # setenv LLM_PROXY allowlist). A bare /v1 returns 404 and silently falls back to rule insights.
        default="https://ai-llm-gateway.amap.com/open_api/v1",
        validation_alias=AliasChoices(
            "AGENT_EVAL_INSIGHT_LLM_BASE_URL",
            "AMAP_LLM_BASE_URL",
        ),
        description="Report-insight-only Amap LLM gateway base URL.",
    )
    insight_llm_model: str = Field(
        default="gpt-5.6-sol",
        validation_alias=AliasChoices("AGENT_EVAL_INSIGHT_LLM_MODEL"),
        description="Primary insight model (ieval primary).",
    )
    insight_llm_model_fallback: str = Field(
        default="gpt-5.4-0305-global",
        validation_alias=AliasChoices("AGENT_EVAL_INSIGHT_LLM_MODEL_FALLBACK"),
        description="Fallback insight model (ieval fallback).",
    )
    insight_llm_timeout_s: float = Field(
        default=45.0,
        validation_alias=AliasChoices("AGENT_EVAL_INSIGHT_LLM_TIMEOUT_S"),
        description="Timeout seconds for one insight chat.completions call.",
    )

    # -- LLM concurrency limit (per-process semaphore, shared by Path A + Path B) --
    # Covers judge `_eval` + sync triage/attribution (`after_eval`) under one slot.
    # In multi-node deployments, each node has its own independent semaphore.
    # Set judge_kafka_concurrency per-node so that node_count × judge_kafka_concurrency
    # stays within the LLM provider rate limit.
    llm_max_concurrency: int = Field(default=30)

    # -- Per-instance concurrency buckets for one codebase (design §12.5 F1, in-process semaphore) --
    # Cap in-flight evals per internal agent name so one instance cannot fill the global slots
    # (a noisy neighbor). 0 = unlimited (legacy behavior). A multi-node cluster cap is
    # node count × this value; fold it together with llm_max_concurrency when configuring.
    # An explicit alias bypasses env_prefix, so the full AGENT_EVAL_ name is written out,
    # same as worker_profile in this MR. A name without the prefix does not take effect.
    instance_max_concurrency: int = Field(
        default=0,
        validation_alias=AliasChoices("AGENT_EVAL_INSTANCE_MAX_CONCURRENCY"),
    )
    # -- Worker queue split (design §12.7 L2 process isolation) --
    # all = consume both the default and L2 queues (single-process compatible).
    # default = consume only the default benchmark-cases queue. l2 = consume only
    # benchmark-cases-l2 (used when deployed as a separate process, for fault and resource
    # isolation). Other workers ignore this switch (multi-node deploy already consumes
    # more than once, which BullMQ allows).
    worker_profile: str = Field(
        default="all",
        validation_alias=AliasChoices("AGENT_EVAL_WORKER_PROFILE"),
    )

    # -- Langfuse trace resolver ----------------------------------------
    # Base64-encoded JSON array; decoded by load_trace_instances().
    trace_instances: str = Field(
        default="",
        validation_alias=AliasChoices("AGENT_EVAL_TRACE_INSTANCES"),
    )

    # -- Langfuse dataset fetch tunables --------------------------------
    # HTTP timeout (seconds) for Langfuse SDK get_dataset pagination.
    # The SDK default is 5s; pre/prod pagination of large datasets needs more.
    trace_http_timeout_s: int = Field(
        default=60,
        validation_alias=AliasChoices("AGENT_EVAL_TRACE_HTTP_TIMEOUT_S"),
    )
    # Retry attempts for transient HTTP errors during dataset fetch.
    trace_fetch_attempts: int = Field(
        default=3,
        validation_alias=AliasChoices("AGENT_EVAL_TRACE_FETCH_ATTEMPTS"),
    )
    # Page size for get_dataset item pagination.
    trace_fetch_page_size: int = Field(
        default=50,
        validation_alias=AliasChoices("AGENT_EVAL_TRACE_FETCH_PAGE_SIZE"),
    )

    # -- MetaQ eval-message consumer (mq_consumer) --------------------------------
    # The subscription is described by custom.mq_consumer in conf/app.<env>.yaml. The next
    # two fields are runtime overrides and are not committed.
    # AGENT_EVAL_MQ_ENABLED=false turns consumption off locally or offline. None (unset)
    # follows yaml enabled.
    mq_enabled: bool | None = Field(default=None)
    # AGENT_EVAL_MQ_NAMESRV_DOMAIN overrides yaml namesrv_domain (canary / unit cutover).
    mq_namesrv_domain: str = Field(default="")

    # -- sidecar-trace MetaQ notify + Lindorm/OSS readback (independent of mq_consumer) --
    sidecar_trace_enabled: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("SIDECAR_TRACE_ENABLED"),
        description="Override sidecar enabled; None keeps yaml/default (false).",
    )
    sidecar_trace_namesrv_domain: str = Field(
        default="",
        validation_alias=AliasChoices("SIDECAR_TRACE_NAMESRV_DOMAIN"),
    )
    sidecar_trace_lindorm_dsn: str = Field(
        default="",
        validation_alias=AliasChoices("SIDECAR_TRACE_LINDORM_DSN"),
        description="Read-only Lindorm DSN for sidecar trace PK get.",
    )
    sidecar_trace_lindorm_table: str = Field(
        default="",
        validation_alias=AliasChoices("SIDECAR_TRACE_LINDORM_TABLE"),
        description="Override default lindorm_table (sidecar_trace) when set.",
    )
    sidecar_trace_oss_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices("SIDECAR_TRACE_OSS_ENDPOINT"),
    )
    sidecar_trace_oss_access_key_id: str = Field(
        default="",
        validation_alias=AliasChoices("SIDECAR_TRACE_OSS_ACCESS_KEY_ID"),
    )
    sidecar_trace_oss_access_key_secret: str = Field(
        default="",
        validation_alias=AliasChoices("SIDECAR_TRACE_OSS_ACCESS_KEY_SECRET"),
    )

    # -- Label Studio / Kafka consumer switches --------------------------------
    # Startup now uses benchmark streaming. judge_service_* exists so the app/judge
    # package still on the tree can import it. main no longer starts JudgeService from it.
    judge_service_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("JUDGE_SERVICE_ENABLED"),
        description="是否启用 Judge Service Kafka 消费器（Path B）",
    )
    judge_service_watch_interval_s: int = Field(
        default=30,
        validation_alias=AliasChoices("JUDGE_SERVICE_WATCH_INTERVAL_S"),
        description=(
            "Background watch interval (seconds) for re-checking "
            "JUDGE_SERVICE_ENABLED at runtime. 0 = disable watch "
            "(startup value is fixed for the process lifetime)."
        ),
    )
    label_studio_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("AGENT_EVAL_LABEL_STUDIO_ENABLED"),
        description="是否启用 Label Studio 标注集成",
    )
    # -- Benchmark / dataset processing switches (two independent switches) --------
    # Switch state lives in RemoteConfig (dataId agent-eval.service.switches) and is
    # checked live on each action (see benchmark.domain.service_switches). These fields
    # are only the env-var fallback when RemoteConfig is unavailable or the field is
    # missing. Default True (do not process): new data of that type is not processed —
    # create-style HTTP endpoints reject with 503 and the daily schedule skips the
    # trigger. A run already created, and the jobs already in its queue, are "tasks
    # already running" and finish. The old combined field benchmark_dataset_disabled
    # is no longer read (not compatible).
    benchmark_disabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("AGENT_EVAL_BENCHMARK_DISABLED"),
        description=(
            "benchmark 处理开关的环境变量回退（主链路走 RemoteConfig "
            "agent-eval.service.switches 的 benchmark_disabled 字段）。"
            "true = 不处理新数据（默认）；in-flight run 继续跑完。"
        ),
    )
    dataset_disabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("AGENT_EVAL_DATASET_DISABLED"),
        description=(
            "dataset 处理开关的环境变量回退（主链路走 RemoteConfig "
            "agent-eval.service.switches 的 dataset_disabled 字段）。"
            "true = 不处理新数据（默认）；in-flight run 继续跑完。"
        ),
    )
    # RemoteConfig coordinates for the service switches (the values live in RemoteConfig,
    # not in env vars). Only the coordinates are declared here and may be overridden by
    # env — the same pattern as benchmark_oss / external_daily.
    service_switches_remote_config_data_id: str = Field(
        default="agent-eval.service.switches",
        validation_alias=AliasChoices("SERVICE_SWITCHES_REMOTE_CONFIG_DATA_ID"),
        description="RemoteConfig data_id for service switches.",
    )
    service_switches_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("SERVICE_SWITCHES_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for service switches.",
    )
    judge_evaluate_timeout_s: int = Field(
        default=300,
        validation_alias=AliasChoices("JUDGE_EVALUATE_TIMEOUT_S"),
        description=(
            "Hard timeout (seconds) for each evaluate call. "
            "0 = no timeout.  When > 0, asyncio.wait_for wraps the "
            "evaluate; timeouts are treated as transient (consumer retries)."
        ),
    )

    # -- Judge Service: RemoteConfig config -----------------------------------
    judge_eval_config_remote_config_data_id: str = Field(
        default="agent-eval.service.evaluators",
        validation_alias=AliasChoices("JUDGE_EVAL_CONFIG_REMOTE_CONFIG_DATA_ID"),
    )
    judge_eval_config_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("JUDGE_EVAL_CONFIG_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for judge eval configs.",
    )
    judge_eval_config_fallback_file: str = Field(
        default="",
        validation_alias=AliasChoices("JUDGE_EVAL_CONFIG_FALLBACK_FILE"),
    )

    # -- Benchmark OSS: RemoteConfig config source ---------------------------
    # OSS config values (bucket / endpoint / cdn / credential_mode, and so on) live in
    # RemoteConfig, not in env vars. Only the RemoteConfig coordinates are declared here
    # and may be overridden by env.
    benchmark_oss_remote_config_data_id: str = Field(
        default="agent-eval.benchmark.oss",
        validation_alias=AliasChoices("BENCHMARK_OSS_REMOTE_CONFIG_DATA_ID"),
        description="RemoteConfig data_id for benchmark report OSS config.",
    )
    benchmark_oss_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("BENCHMARK_OSS_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for benchmark report OSS config.",
    )

    # -- Benchmark external-daily scheduler: RemoteConfig config source -----------
    # Schedule config (schedule_time / the benchmarks list, and so on) lives in RemoteConfig,
    # not in env vars. Only the RemoteConfig coordinates are declared here and may be
    # overridden by env.
    benchmark_external_daily_remote_config_data_id: str = Field(
        default="agent-eval.benchmark.external-daily",
        validation_alias=AliasChoices("BENCHMARK_EXTERNAL_DAILY_REMOTE_CONFIG_DATA_ID"),
        description="RemoteConfig data_id for benchmark external-daily scheduler config.",
    )
    benchmark_external_daily_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("BENCHMARK_EXTERNAL_DAILY_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for benchmark external-daily scheduler config.",
    )

    # -- Trace daily report (T+1) scheduler: RemoteConfig config source -----------
    # Daily report schedule config for trace_pipeline_records (source / agent_name /
    # project_id / the reports list, and so on) lives in RemoteConfig, not in env vars.
    # Only the RemoteConfig coordinates are declared here and may be overridden by env.
    trace_daily_report_remote_config_data_id: str = Field(
        default="agent-eval.benchmark.trace-daily-report",
        validation_alias=AliasChoices("TRACE_DAILY_REPORT_REMOTE_CONFIG_DATA_ID"),
        description="RemoteConfig data_id for trace daily report (T+1) scheduler config.",
    )
    trace_daily_report_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("TRACE_DAILY_REPORT_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for trace daily report (T+1) scheduler config.",
    )

    # -- Benchmark external report providers: RemoteConfig seed (optional) -------
    # Business writes go to the DB through the API. RemoteConfig is only an ops seed / fallback when no DB row exists.
    benchmark_report_providers_remote_config_data_id: str = Field(
        default="agent-eval.benchmark.report-providers",
        validation_alias=AliasChoices("BENCHMARK_REPORT_PROVIDERS_REMOTE_CONFIG_DATA_ID"),
        description="RemoteConfig data_id for external report provider seed config.",
    )
    benchmark_report_providers_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("BENCHMARK_REPORT_PROVIDERS_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for external report provider seed config.",
    )

    # -- Eval plugins (benchmark components / dataset output providers) ----
    # Runtime enable/disable via RemoteConfig; empty / missing → allow all.
    eval_plugins_remote_config_data_id: str = Field(
        default="agent-eval.eval.plugins",
        validation_alias=AliasChoices("EVAL_PLUGINS_REMOTE_CONFIG_DATA_ID"),
        description="RemoteConfig data_id for eval plugin enable/disable config.",
    )
    eval_plugins_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("EVAL_PLUGINS_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for eval plugin enable/disable config.",
    )

    # -- Judge Service: Kafka consumer -----------------------------------
    # Consumes ``langfuse.records`` topic from Langfuse RecordStreamProducer.
    # Only ``record_type=trace`` messages are processed; observation and
    # score messages are silently skipped.
    #
    # .. versionadded:: 0.5.0
    #    Replaces BullMQ ``llm-as-a-judge-execution-queue`` consumption.
    #    Uses ``libs/message_queue`` ``MQConsumer`` + ``AioKafkaBackend``.
    judge_kafka_bootstrap_servers: str = Field(
        default="",
        validation_alias=AliasChoices("JUDGE_KAFKA_BOOTSTRAP_SERVERS"),
        description=(
            "Comma-separated Kafka broker list, e.g. "
            "'host1:9092,host2:9092'. Aligns with Langfuse "
            "LANGFUSE_RECORD_STREAM_KAFKA_BROKERS."
        ),
    )
    judge_kafka_topic: str = Field(
        default="langfuse.records",
        validation_alias=AliasChoices("JUDGE_KAFKA_TOPIC"),
        description=(
            "Kafka topic name(s). Comma-separated for multiple topics, "
            "e.g. 'langfuse.records,langfuse.records.v2'. "
            "Aligns with Langfuse LANGFUSE_RECORD_STREAM_KAFKA_TOPIC "
            "(default 'langfuse.records')."
        ),
    )
    judge_kafka_topic_filters: str = Field(
        default="",
        validation_alias=AliasChoices("JUDGE_KAFKA_TOPIC_FILTERS"),
        description=(
            "Per-topic filter rules as JSON string. "
            'Format: {"topic_name": {"record_types": ["trace"], '
            '"skip_deleted": true}}. '
            "Empty string = all topics use default "
            '(record_types=["trace"], skip_deleted=true).'
        ),
    )
    judge_kafka_group_id: str = Field(
        default="agent-eval-judge",
        validation_alias=AliasChoices("JUDGE_KAFKA_GROUP_ID"),
        description="Kafka consumer group ID for Judge Service.",
    )
    judge_kafka_security_protocol: str = Field(
        default="PLAINTEXT",
        validation_alias=AliasChoices("JUDGE_KAFKA_SECURITY_PROTOCOL"),
        description="Kafka security protocol (PLAINTEXT / SASL_PLAINTEXT / SSL / SASL_SSL).",
    )
    judge_kafka_sasl_mechanism: str = Field(
        default="",
        validation_alias=AliasChoices("JUDGE_KAFKA_SASL_MECHANISM"),
        description="SASL mechanism (PLAIN / SCRAM-SHA-256 / SCRAM-SHA-512).",
    )
    judge_kafka_sasl_username: str = Field(
        default="",
        validation_alias=AliasChoices("JUDGE_KAFKA_SASL_USERNAME"),
    )
    judge_kafka_sasl_password: str = Field(
        default="",
        validation_alias=AliasChoices("JUDGE_KAFKA_SASL_PASSWORD"),
    )
    judge_kafka_poll_timeout_s: float = Field(
        default=1.0,
        validation_alias=AliasChoices("JUDGE_KAFKA_POLL_TIMEOUT_S"),
        description="Kafka consumer poll timeout in seconds.",
    )
    judge_kafka_concurrency: int = Field(
        default=1,
        validation_alias=AliasChoices("JUDGE_KAFKA_CONCURRENCY"),
        description=(
            "Number of concurrent MQConsumer instances (asyncio tasks). "
            "Each instance runs its own poll loop. All share the same "
            "group_id; Kafka auto-assigns partitions."
        ),
    )
    judge_kafka_auto_commit: bool = Field(
        default=True,
        validation_alias=AliasChoices("JUDGE_KAFKA_AUTO_COMMIT"),
        description=(
            "MQConsumer auto_commit flag. Judge Kafka always "
            "commit_message() immediately when a message is delivered to "
            "the handler (before evaluate); judge failures do not block "
            "offset progress. This flag mainly affects framework end-of-"
            "handler commit when early commit did not run."
        ),
    )
    judge_kafka_client_id: str = Field(
        default="agent-eval-judge",
        validation_alias=AliasChoices("JUDGE_KAFKA_CLIENT_ID"),
        description="Kafka client_id for broker-side identification.",
    )
    judge_kafka_max_msgs_per_min: float = Field(
        default=0.0,
        validation_alias=AliasChoices("JUDGE_KAFKA_MAX_MSGS_PER_MIN"),
        description=(
            "Rate limit: max trace records processed per minute. "
            "0 = unlimited. When > 0, aiolimiter throttles consumption "
            "across all consumer instances in this process."
        ),
    )
    judge_kafka_max_retries: int = Field(
        default=0,
        validation_alias=AliasChoices("JUDGE_KAFKA_MAX_RETRIES"),
        description=(
            "Max retry attempts per message before discarding. "
            "0 = infinite retry (default). When > 0 and auto_commit=True, "
            "exhausted messages are discarded and their offset is committed "
            "so the consumer advances past poison-pill messages."
        ),
    )
    judge_kafka_max_poll_interval_ms: int | None = Field(
        default=600000,
        validation_alias=AliasChoices("JUDGE_KAFKA_MAX_POLL_INTERVAL_MS"),
        description=(
            "Kafka consumer max.poll.interval.ms. If the consumer doesn't "
            "poll within this interval the broker triggers a rebalance. "
            "Must exceed the longest evaluate + backoff cycle. "
            "Default 600000 (10 min) — safe for evaluate_timeout up to 300s "
            "plus 3 backoff retries."
        ),
    )
    judge_kafka_session_timeout_ms: int | None = Field(
        default=30000,
        validation_alias=AliasChoices("JUDGE_KAFKA_SESSION_TIMEOUT_MS"),
        description=(
            "Kafka consumer session.timeout.ms. If no heartbeat is received "
            "within this time the broker considers the consumer dead. "
            "Default 30000 (30s) — tolerant of GIL contention during "
            "CPU-intensive evaluation."
        ),
    )
    judge_kafka_heartbeat_interval_ms: int | None = Field(
        default=10000,
        validation_alias=AliasChoices("JUDGE_KAFKA_HEARTBEAT_INTERVAL_MS"),
        description=(
            "Kafka consumer heartbeat.interval.ms. Should be ≤ 1/3 of "
            "session_timeout_ms. Default 10000 (10s)."
        ),
    )

    # -- Benchmark streaming (the only Kafka realtime consume path) ----------------------
    # The old JUDGE_SERVICE_ENABLED is kept as a compatibility alias for one release.
    # After cutover, shadow defaults to off.
    benchmark_streaming_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "BENCHMARK_STREAMING_ENABLED",
            "JUDGE_SERVICE_ENABLED",
        ),
        description=(
            "是否启用 benchmark streaming Kafka 消费（唯一 Path B）。"
            "兼容旧环境变量 JUDGE_SERVICE_ENABLED；默认 false"
        ),
    )
    benchmark_streaming_shadow: bool = Field(
        default=False,
        validation_alias=AliasChoices("BENCHMARK_STREAMING_SHADOW"),
        description=(
            "streaming shadow 对账模式：S2–S7 全量执行打 shadow 日志，"
            "S8 起短路（不调 LLM / 不落库 / 不归因）；开启时强制独立 "
            "consumer group（<group>-shadow）。切主后默认 false"
        ),
    )
    benchmark_streaming_remote_config_data_id: str = Field(
        default="agent-eval.benchmark.streaming",
        validation_alias=AliasChoices("BENCHMARK_STREAMING_REMOTE_CONFIG_DATA_ID"),
        description="RemoteConfig data_id for benchmark streaming 声明式配置（§5.2）",
    )
    benchmark_streaming_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("BENCHMARK_STREAMING_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for benchmark streaming 声明式配置（§5.2）",
    )

    @field_validator("benchmark_streaming_enabled", "benchmark_streaming_shadow", mode="before")
    @classmethod
    def _coerce_streaming_bool(cls, value: object) -> object:
        """Lenient bool normalize for the streaming switch: a typo such as 'fasle' maps to the intended value.

        Stops a mistyped deploy env var from raising ValidationError while importing
        ``Settings()`` and crash-looping the service (2026-09-28 incident; see _ENV_BOOL_TYPOS).
        """
        return _coerce_env_bool(value)

    # -- Label Studio integration -----------------------------------------
    label_studio_base_url: str = Field(default="http://127.0.0.1:8080")
    label_studio_api_key: str | None = Field(default=None)
    label_studio_webhook_secret: str | None = Field(default=None)
    label_studio_auth_mode: str = Field(default="token")
    label_studio_email: str | None = Field(default=None)
    label_studio_password: str | None = Field(default=None)
    label_studio_callback_url: str = Field(default="http://127.0.0.1:8000")
    label_studio_default_label_config: str = Field(
        default="conf/label_studio/badcase_review.xml",
        description="默认 label config XML 路径（未在 map 中命中的 agent 使用此模板）",
    )
    label_studio_label_config_map: dict[str, str] = Field(
        default_factory=dict,
        description="agent_name → XML 路径映射",
    )

    # -- BullMQ eval-task dispatch (design doc §14) ─────────────────────────────
    # Redis connection
    eval_queue_redis_url: str = Field(
        default="",
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_URL"),
        description="Redis 直连 URL（本地开发；最高优先级）",
    )
    eval_queue_redis_kvstore_instance_id: str = Field(
        default="",
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_KVSTORE_INSTANCE_ID"),
        description="阿里云 KVStore 实例 ID（生产推荐；经 normandy SDK 获取凭证）",
    )
    eval_queue_redis_host: str = Field(
        default="",
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_HOST"),
    )
    eval_queue_redis_port: int = Field(
        default=6379,
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_PORT"),
    )
    eval_queue_redis_username: str = Field(
        default="",
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_USERNAME"),
    )
    eval_queue_redis_password: str = Field(
        default="",
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_PASSWORD"),
    )
    eval_queue_redis_scheme: str = Field(
        default="redis",
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_SCHEME"),
        description="redis / rediss",
    )
    eval_queue_redis_db: int = Field(
        default=0,
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_DB"),
    )
    eval_queue_redis_cluster_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("EVAL_QUEUE_REDIS_CLUSTER_ENABLED"),
        description=(
            "保留字段；BullMQ prefix 始终带 per-queue hash tag，"
            "以兼容阿里云 KvStore 的 cluster slot 校验"
        ),
    )

    # Queue prefix (separates daily / pre / prod).
    eval_queue_prefix: str = Field(
        default="",
        validation_alias=AliasChoices(
            "EVAL_QUEUE_PREFIX",
            "AGENT_EVAL_EVAL_QUEUE_PREFIX",
        ),
        description=(
            "BullMQ 队列名前缀。标准 envSign（staging/production 等）会被策略强制为 "
            "pre/prod/daily 并覆盖本字段；临时环境（长 envSign）必须设非保留值。"
        ),
    )
    eval_queue_deploy_kind: str = Field(
        default="",
        validation_alias=AliasChoices(
            "AGENT_EVAL_DEPLOY_KIND",
            "EVAL_QUEUE_DEPLOY_KIND",
        ),
        description=(
            "可选强制临时泳道：personal|business|dev|local|vibe；"
            "通常由平台 envSign 长度自动判定，无需手填。"
        ),
    )
    # Platform-injected: an Aone temporary envSign is long (>20). Standard values are testing/staging/gray/production.
    env_sign: str = Field(
        default="",
        validation_alias=AliasChoices("envSign", "ENV_SIGN", "AGENT_EVAL_ENV_SIGN"),
        description="平台 envSign；长度>20 视为临时/项目环境。",
    )
    sigma_app_stage: str = Field(
        default="",
        validation_alias=AliasChoices("SIGMA_APP_STAGE", "AGENT_EVAL_SIGMA_APP_STAGE"),
        description="ASI 无 envSign 时的阶段：DAILY/PRE_PUBLISH/GRAY/SMALLFLOW/PUBLISH。",
    )

    # Sync-model job spec.
    eval_queue_max_attempts: int = Field(
        default=1,
        validation_alias=AliasChoices("EVAL_QUEUE_MAX_ATTEMPTS"),
        description=(
            "失败累计次数上限；retry_count >= 此值则终败（D4）。"
            "默认 1：case 失败一次即终败，不再重试（env 可覆盖）"
        ),
    )
    eval_queue_backoff_initial_ms: int = Field(
        default=5000,
        validation_alias=AliasChoices("EVAL_QUEUE_BACKOFF_INITIAL_MS"),
    )
    eval_queue_backoff_max_ms: int = Field(
        default=25000,
        validation_alias=AliasChoices("EVAL_QUEUE_BACKOFF_MAX_MS"),
    )
    eval_queue_lock_duration_ms: int = Field(
        default=1200000,
        validation_alias=AliasChoices("EVAL_QUEUE_LOCK_DURATION_MS"),
        description="同步模型 lockDuration（20 分钟，对齐 LLM 重试上限）",
    )
    sync_slot_ttl_s: int = Field(
        default=600,
        validation_alias=AliasChoices("SYNC_SLOT_TTL_S"),
        description=(
            "同步 worker Redis 槽位 key TTL（秒）。"
            "独立于 lock_duration_ms，确保卡住的槽位在 10min 内自动过期，"
            "避免 per-node 计数器长期占位导致 slot_busy 死循环。"
        ),
    )
    eval_queue_stalled_interval_ms: int = Field(
        default=30000,
        validation_alias=AliasChoices("EVAL_QUEUE_STALLED_INTERVAL_MS"),
    )
    eval_queue_max_stalled_count: int = Field(
        default=10,
        validation_alias=AliasChoices("EVAL_QUEUE_MAX_STALLED_COUNT"),
        description=(
            "Worker maxStalledCount：进程重启 / 短暂卡死后 wait job 仍可被"
            " stalled 机制重投（§9.2）；BullMQ 默认 1 会在第二次 stall 后直接"
            " failed+remove，与「崩溃后继续执行」冲突。"
        ),
    )
    eval_queue_worker_concurrency: int = Field(
        default=5,
        validation_alias=AliasChoices("EVAL_QUEUE_WORKER_CONCURRENCY"),
    )

    # Sync-model slots (D7/D14/D15).
    sync_slots_per_run_per_node: int = Field(
        default=1,
        validation_alias=AliasChoices("SYNC_SLOTS_PER_RUN_PER_NODE"),
        description="每节点每 run 并发槽位上限（D7）；1 = 同 run 内串行，不同 run 仍可并行",
    )
    sync_cluster_soft_cap: int = Field(
        default=0,
        validation_alias=AliasChoices("SYNC_CLUSTER_SOFT_CAP"),
        description="同步模型集群级 LLM 软顶（≤0 禁用；对齐 external_cluster_aone_soft_cap）",
    )
    sync_slot_busy_delay_ms: int = Field(
        default=3000,
        validation_alias=AliasChoices("SYNC_SLOT_BUSY_DELAY_MS"),
        description="槽位忙时 wait delay 基准（D15）",
    )
    sync_slot_heal_interval_s: int = Field(
        default=60,
        validation_alias=AliasChoices("SYNC_SLOT_HEAL_INTERVAL_S"),
        description="槽位差额巡检间隔",
    )
    sync_retry_max_duration_ms: int = Field(
        default=7200000,
        validation_alias=AliasChoices("SYNC_RETRY_MAX_DURATION_MS"),
        description="同步模型重试时间窗上限（自 first_enqueued_at 起算，D18）",
    )
    max_slot_probes_per_run: int = Field(
        default=2,
        validation_alias=AliasChoices("MAX_SLOT_PROBES_PER_RUN"),
        description="同 run 同时探测上限（D15）",
    )

    # Guardrails.
    eval_max_cases_per_run: int = Field(
        default=5000,
        validation_alias=AliasChoices("EVAL_MAX_CASES_PER_RUN"),
    )
    eval_max_case_payload_kb: int = Field(
        default=0,
        validation_alias=AliasChoices("EVAL_MAX_CASE_PAYLOAD_KB"),
        description="单 case payload 大小护栏（0 = 不启用，D11）",
    )

    # Async-model (external) config.
    external_worker_concurrency: int = Field(
        default=2,
        validation_alias=AliasChoices("EXTERNAL_WORKER_CONCURRENCY"),
        description="external-wait Worker concurrency（YAML slots.external.worker_concurrency）",
    )
    external_wait_lock_duration_ms: int = Field(
        default=300000,
        validation_alias=AliasChoices("EXTERNAL_WAIT_LOCK_DURATION_MS"),
        description="异步 wait queue lockDuration（提交 Aone 单次调用）",
    )
    external_judge_submit_interval_min: int = Field(
        default=20,
        validation_alias=AliasChoices("EXTERNAL_JUDGE_SUBMIT_INTERVAL_MIN"),
        description="wait queue Worker limiter（Aone 提交全局限速）",
    )
    external_running_ttl_ms: int = Field(
        default=86400000,
        validation_alias=AliasChoices("EXTERNAL_RUNNING_TTL_MS"),
        description="running 队列 job TTL（24h）",
    )
    external_slot_ttl_s: int = Field(
        default=600,
        validation_alias=AliasChoices("EXTERNAL_SLOT_TTL_S"),
        description=(
            "Redis 槽位 key TTL（秒）。"
            "独立于 running_ttl_ms，确保卡住的槽位在 10min 内自动过期，"
            "避免 per-node 计数器长期占位导致 slot_busy 死循环。"
        ),
    )
    external_callback_timeout_ms: int = Field(
        default=600000,
        validation_alias=AliasChoices("EXTERNAL_CALLBACK_TIMEOUT_MS"),
        description=(
            "pending_callback 等待回调的超时阈值（默认 10min）。"
            "超过后 healer 查询 Aone job 状态："
            "running 则继续等；否则回插 wait（保留 first_enqueued_at）。"
            "逾单 case 1h 或 running TTL 则终败该 case。"
        ),
    )
    external_callback_concurrency: int = Field(
        default=20,
        validation_alias=AliasChoices("EXTERNAL_CALLBACK_CONCURRENCY"),
        description="回调 API 信号量",
    )
    external_attr_submit_interval_min: int = Field(
        default=20,
        validation_alias=AliasChoices("EXTERNAL_ATTR_SUBMIT_INTERVAL_MIN"),
    )
    external_local_attr_concurrency: int = Field(
        default=1,
        validation_alias=AliasChoices("EXTERNAL_LOCAL_ATTR_CONCURRENCY"),
    )
    # Async-model slots (D7/D14).
    external_slots_per_run_per_node: int = Field(
        default=1,
        validation_alias=AliasChoices("EXTERNAL_SLOTS_PER_RUN_PER_NODE"),
        description="异步每节点每 run 并发槽位上限；1 = 同 run 内串行，不同 run 仍可并行",
    )
    external_cluster_notification_soft_cap: int = Field(
        default=4,
        validation_alias=AliasChoices("EXTERNAL_CLUSTER_NOTIFICATION_SOFT_CAP"),
        description="集群级 Aone 提交软顶（≤0 禁用，prod 必须 >0）",
    )
    external_slot_busy_delay_ms: int = Field(
        default=3000,
        validation_alias=AliasChoices("EXTERNAL_SLOT_BUSY_DELAY_MS"),
        description="异步槽位忙时 wait delay 基准（D15）",
    )
    external_slot_heal_interval_s: int = Field(
        default=60,
        validation_alias=AliasChoices("EXTERNAL_SLOT_HEAL_INTERVAL_S"),
    )
    attr_worker_concurrency: int = Field(
        default=2,
        validation_alias=AliasChoices("ATTR_WORKER_CONCURRENCY"),
        description="归因 Worker concurrency 预算（与 case Worker 分离）",
    )
    attr_slots_per_run_per_node: int = Field(
        default=2,
        validation_alias=AliasChoices("ATTR_SLOTS_PER_RUN_PER_NODE"),
        description="归因每节点每 run 并发槽位上限（kind=attr，与 judge 隔离）",
    )
    attr_cluster_soft_cap: int = Field(
        default=4,
        validation_alias=AliasChoices("ATTR_CLUSTER_SOFT_CAP"),
        description="归因集群软顶（kind=attr；≤0 禁用）",
    )
    attr_callback_timeout_ms: int = Field(
        default=600000,
        validation_alias=AliasChoices("ATTR_CALLBACK_TIMEOUT_MS"),
        description=("归因 pending_callback 巡检间隔，默认与 Aone/judge 相同（10min）。"),
    )
    attr_callback_url: str = Field(
        default="",
        validation_alias=AliasChoices("ATTR_CALLBACK_URL"),
        description="归因回调 URL 模板（含 {run_id} 占位符）",
    )
    code_attribution_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("CODE_ATTRIBUTION_ENABLED"),
        description="归因回调之后在本服务跑代码归因并替换展示内容",
    )
    code_attribution_lock_duration_ms: int = Field(
        default=2_400_000,
        validation_alias=AliasChoices("CODE_ATTRIBUTION_LOCK_DURATION_MS"),
        description="代码归因队列 lockDuration（覆盖两次 900 秒云端等待）",
    )
    cloud_agent_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("CLOUD_AGENT_BASE_URL"),
        description="代码归因云端 Open API 根地址",
    )
    cloud_agent_token: str = Field(
        default="",
        validation_alias=AliasChoices("CLOUD_AGENT_TOKEN"),
        description="代码归因云端 token",
    )
    cloud_agent_project_path: str = Field(
        default="",
        validation_alias=AliasChoices("CLOUD_AGENT_PROJECT_PATH"),
        description="代码归因只读仓库在云端的路径",
    )
    notification_query_dedup_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("NOTIFICATION_QUERY_DEDUP_ENABLED"),
        description=(
            "通知派单的 Redis query 指纹去重开关。"
            "关闭后同 query 每次回调均建单（去重异常时快速止血用）。"
        ),
    )
    external_retry_max_duration_ms: int = Field(
        default=7200000,
        validation_alias=AliasChoices("EXTERNAL_RETRY_MAX_DURATION_MS"),
        description=(
            "单 case 时窗（默认 2h）。每个 case 以自己首次进入 running 的 "
            "first_enqueued_at 起算（各 case 时钟独立；wait 排队不占用）。"
            "回插 wait 不得改写该字段。judge 与归因各自 2h。"
        ),
    )
    external_run_max_duration_ms: int = Field(
        default=86400000,
        validation_alias=AliasChoices("EXTERNAL_RUN_MAX_DURATION_MS"),
        description=(
            "整个 external run 墙钟时窗（judge+归因共用，自 run.created_at 起算，默认 24h）。"
            "逾时 healer 失败剩余 case、释放槽位并 fail_run；与单 case 2h 时窗独立。"
        ),
    )
    eval_queue_retry_max_duration_ms: int = Field(
        default=7200000,
        validation_alias=AliasChoices("EVAL_QUEUE_RETRY_MAX_DURATION_MS"),
        description="重试时间窗兜底默认（优先读 sync/external 专用值）",
    )
    eval_queue_healer_disabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("EVAL_QUEUE_HEALER_DISABLED"),
        description=(
            "禁用 RunningHealer 后台巡检（崩溃恢复 / TTL / 心跳）。"
            "调试或排查时设为 true；生产环境不要设为 true。"
        ),
    )
    eval_queue_healer_leader_disabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("EVAL_QUEUE_HEALER_LEADER_DISABLED"),
        description=("禁用 D19 领选举（所有节点均巡检）。单节点环境可选；多节点环境必须为 false。"),
    )
    healer_leader_ttl_s: int = Field(
        default=120,
        validation_alias=AliasChoices("HEALER_LEADER_TTL_S"),
        description="D19 领选举锁 TTL（秒）。",
    )
    healer_leader_renew_s: int = Field(
        default=40,
        validation_alias=AliasChoices("HEALER_LEADER_RENEW_S"),
        description="D19 锁续约间隔（秒；须 < TTL/2）。",
    )
    # Process-level master switch: whether to start in-process scheduled tasks
    # (daily-schedule startup catch-up plus the fixed-time trigger).
    # Default false: do not run. RemoteConfig is not read. Only an explicit
    # AGENT_EVAL_SCHEDULED_TASKS_ENABLED=true turns it on.
    scheduled_tasks_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("AGENT_EVAL_SCHEDULED_TASKS_ENABLED"),
        description=(
            "是否启动进程内定时任务（DailyExternalBenchmarkScheduler / "
            "TraceDailyReportScheduler / OdpsExportScheduler）。默认 false 不执行；"
            "true 后仍受 EXTERNAL_DAILY_BENCHMARK_DISABLED / "
            "TRACE_DAILY_REPORT_DISABLED / ODPS_EXPORT_DISABLED 约束。"
        ),
    )
    external_daily_benchmark_disabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("EXTERNAL_DAILY_BENCHMARK_DISABLED"),
        description=(
            "禁用 DailyExternalBenchmarkScheduler（每日定点触发 external benchmark）。"
            "设为 true 时忽略 RemoteConfig external_daily.enabled，强制不启动调度器。"
        ),
    )
    trace_daily_report_disabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("TRACE_DAILY_REPORT_DISABLED"),
        description=(
            "禁用 TraceDailyReportScheduler（trace_pipeline_records 日级 T+1 报表）。"
            "设为 true 时忽略 RemoteConfig trace_daily_report.enabled，强制不启动调度器。"
        ),
    )

    # -- ODPS sample export (scripts/export_odps_to_oss.py, daily T-1 → CSV → OSS) ------
    # Credentials come from the resource-binding SDK: the app name declares the bound
    # ODPS resource, the SDK issues AK/SK, and the process never sees plaintext
    # credentials (CT-ODPS-001). OSS config reuses remote config agent-eval.benchmark.oss.
    odps_notification_app_name: str = Field(
        default="",
        validation_alias=AliasChoices("AGENT_EVAL_ODPS_NOTIFICATION_APP_NAME"),
        description="ODPS 资源绑定的应用名（export_odps_to_oss.py 必填，缺省脚本退出）",
    )
    odps_endpoint: str = Field(
        default="",
        description="ODPS 服务端点（export_odps_to_oss.py 必填，内网域名）",
    )
    odps_project: str = Field(
        default="autonavi_ai",
        description="ODPS 项目名",
    )

    # -- Realtime eval: environment-level fields, written to the DB on save, not sent by the page --
    evaluator_environment: str = Field(
        default="",
        description=(
            "trace environment 过滤，逗号分隔。空表示不限。"
            "对应 AGENT_EVAL_EVALUATOR_ENVIRONMENT，写入 runtime_config_records。"
        ),
    )
    evaluator_variable_mapping: str = Field(
        default="",
        description=(
            "评测变量映射 JSON 数组。空则使用 input/output/metadata 默认映射。"
            "对应 AGENT_EVAL_EVALUATOR_VARIABLE_MAPPING。"
        ),
    )
    evaluator_model_provider: str = Field(default="", description="评测模型 provider")
    evaluator_model_name: str = Field(default="", description="评测模型名")
    evaluator_model_base_url: str = Field(default="", description="评测模型 base URL")
    evaluator_model_api_key_env: str = Field(
        default="",
        description="存放评测模型 API key 的环境变量名，不存放密钥本身",
    )
    evaluator_model_temperature: float = Field(default=0.0, description="评测模型温度")
    evaluator_model_max_tokens: int | None = Field(default=None, description="评测模型 max tokens")
    evaluator_model_timeout: int | None = Field(default=None, description="评测模型超时秒数")

    # -- ODPS export scheduler (app/services/eval_workers/odps_export_scheduler.py) ------
    # Business fields come from the page and are stored. Timezone, object name, SQL, and
    # the default limit come from the env vars below and are written into
    # runtime_config_records (kind=odps_job) on save. RemoteConfig coordinates remain only
    # for old reads that have not been removed yet.
    odps_export_remote_config_data_id: str = Field(
        default="agent-eval.odps.export",
        validation_alias=AliasChoices("ODPS_EXPORT_REMOTE_CONFIG_DATA_ID"),
        description="RemoteConfig data_id for ODPS export scheduler config.",
    )
    odps_export_remote_config_group: str = Field(
        default="agent-eval.service.group",
        validation_alias=AliasChoices("ODPS_EXPORT_REMOTE_CONFIG_GROUP"),
        description="RemoteConfig group for ODPS export scheduler config.",
    )
    odps_export_disabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("ODPS_EXPORT_DISABLED"),
        description=(
            "禁用 OdpsExportScheduler（每日定点 ODPS 抽样导出 → CSV → OSS）。"
            "设为 true 时不启动调度器。"
        ),
    )
    odps_schedule_timezone: str = Field(
        default="Asia/Shanghai",
        description="ODPS 导出时区，写入每条 job。对应 AGENT_EVAL_ODPS_SCHEDULE_TIMEZONE。",
    )
    odps_object_name: str = Field(
        default="benchmark_final_results.csv",
        description="ODPS 导出文件名。对应 AGENT_EVAL_ODPS_OBJECT_NAME。",
    )
    odps_sql_template: str = Field(
        default="",
        description=(
            "ODPS 抽样 SQL。空则使用进程内置模板。对应 AGENT_EVAL_ODPS_SQL_TEMPLATE。"
            "保存任务时写入 DB，页面不传。"
        ),
    )
    odps_default_limit: int = Field(
        default=200,
        description="页面未传 limit 时写入 job 的抽样条数。对应 AGENT_EVAL_ODPS_DEFAULT_LIMIT。",
    )

    @property
    def resolved_log_path(self) -> Path:
        """Absolute path of the log file. An empty Path when file logging is off."""
        if not self.enable_file_log:
            return Path()
        filename = self.log_filename or "agent_eval.log"
        if self.log_dir:
            return Path(self.log_dir) / filename
        return Path(filename)

    @property
    def api_docs_enabled(self) -> bool:
        """Whether OpenAPI doc endpoints are open: daily only by default, docs_enabled overrides."""
        if self.docs_enabled is not None:
            return self.docs_enabled
        return self.env == "daily"


def configure_trace_environment(app_settings: Settings) -> None:
    """Write the runtime environment to the trace SDK."""

    os.environ["TRACE_TRACING_ENVIRONMENT"] = app_settings.env
    os.environ.setdefault("TRACE_TIMEOUT", "30")


settings = Settings()


def _env_switch_disabled(env_key: str) -> bool:
    """Read a disable-style env switch: unset → ``True`` (do not process by default). When set,
    parse true/1/yes leniently (only an explicit false/0/no turns processing on)."""
    raw = os.environ.get(env_key)
    if raw is None:
        return True
    return raw.lower() in ("true", "1", "yes")


def is_benchmark_disabled_runtime() -> bool:
    """Re-read AGENT_EVAL_BENCHMARK_DISABLED from environment at runtime.

    Env-var fallback: the main path is RemoteConfig (the ``benchmark_disabled`` field of
    agent-eval.service.switches; see benchmark.domain.service_switches). When RemoteConfig
    is unavailable or the field is missing, this function decides. Unset defaults to
    ``True`` (do not process new benchmark data; an in-flight run finishes).
    """
    return _env_switch_disabled("AGENT_EVAL_BENCHMARK_DISABLED")


def is_force_webdev_report_runtime() -> bool:
    """Whether to force the webdev skill for reports (skip the static unified template).

    - ``AGENT_EVAL_FORCE_WEBDEV_REPORT=1|true|yes|on`` → force
    - The old switch ``AGENT_EVAL_UNIFIED_REPORT=0|false|no|off`` → force
    - Unset → False (the static template is the main path)
    """
    force = os.environ.get("AGENT_EVAL_FORCE_WEBDEV_REPORT", "").strip().lower()
    if force in ("true", "1", "yes", "on"):
        return True
    legacy = os.environ.get("AGENT_EVAL_UNIFIED_REPORT")
    if legacy is not None and legacy.strip().lower() in ("0", "false", "no", "off"):
        return True
    return False


def is_dataset_disabled_runtime() -> bool:
    """Re-read AGENT_EVAL_DATASET_DISABLED from environment at runtime.

    Env-var fallback: the main path is RemoteConfig (the ``dataset_disabled`` field of
    agent-eval.service.switches; see benchmark.domain.service_switches). When RemoteConfig
    is unavailable or the field is missing, this function decides. Unset defaults to
    ``True`` (do not process new dataset data; an in-flight run finishes).
    """
    return _env_switch_disabled("AGENT_EVAL_DATASET_DISABLED")


def is_judge_service_enabled_runtime() -> bool:
    """Re-read JUDGE_SERVICE_ENABLED from environment at runtime.

    Transitional compatibility: until the JudgeService Kafka retirement branch lands,
    app/judge and app/config/__init__ still import this symbol. The retirement branch
    removes it together with those imports.
    """
    return os.environ.get("JUDGE_SERVICE_ENABLED", "").lower() in ("true", "1", "yes")
