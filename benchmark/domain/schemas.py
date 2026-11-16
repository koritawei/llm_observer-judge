"""Eval config Pydantic schemas — domain models for RemoteConfig eval configuration.

This module is intentionally lightweight (only depends on ``pydantic``) to
avoid circular imports.  Heavier modules like :mod:`benchmark.domain.eval_config`
(EvalConfigStore) import from here, and consumers that only need the schema
types (e.g. :mod:`benchmark.evaluators.base`) can import directly without
triggering the ``benchmark.settings`` dependency chain.

Moved from :mod:`benchmark.judge.schemas` to resolve §2 layering violation
(domain → judge reverse dependency).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

INGEST_KAFKA = "kafka"
INGEST_SIDECAR = "sidecar"

# ---------------------------------------------------------------------------
# Variable mapping
# ---------------------------------------------------------------------------


class VariableMapping(BaseModel):
    """Variable mapping item — observation field → evaluator prompt variable."""

    model_config = {"extra": "ignore"}

    template_variable: str = Field(
        alias="templateVariable", description="evaluator template 中的变量名"
    )
    trace_object: str = Field(
        alias="traceObject", description="数据源对象: trace / observation"
    )
    selected_column_id: str = Field(
        alias="selectedColumnId",
        description="数据源字段: input / output / metadata / tool_calls / trace_name / user_id / session_id / tags / name / type / environment",  # noqa: E501 — the enum value list must not be split further
    )
    json_selector: str | None = Field(
        default=None, alias="jsonSelector", description="JSONPath 表达式（可选）"
    )


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class RemoteConfigModelConfig(BaseModel):
    """Model configuration embedded in RemoteConfig eval config.

    All fields are optional.  When every critical field (``model_name``,
    ``base_url``, ``api_key_env``) is empty/None the RemoteConfig override is
    treated as a no-op and the evaluator falls back to the YAML-assembled
    default LLM (``models.<key>`` in ``app.<env>.yaml``).  ``api_key_env``
    holds an *environment variable name* (never the key itself).
    """

    model_config = {"extra": "allow"}

    provider: str | None = Field(
        default=None, description="LLM provider (e.g. openai)；为空时回退到 YAML provider"
    )
    model_name: str | None = Field(
        default=None, alias="modelName", description="模型名称；缺失时回退到 YAML model_name"
    )
    base_url: str | None = Field(
        default=None, alias="baseUrl", description="API base URL；缺失时回退到 YAML base_url"
    )
    api_key_env: str | None = Field(
        default=None,
        alias="apiKeyEnv",
        description="存放 API key 的环境变量名；缺失时回退到 YAML api_key_env",
    )
    temperature: float = Field(default=0.0, description="采样温度")
    max_tokens: int | None = Field(default=None, alias="maxTokens", description="最大 token 数")
    timeout: int | None = Field(default=None, description="超时秒数")

    @property
    def is_empty(self) -> bool:
        """True when no critical field is set — override carries no usable config.

        When True, the evaluator should fall back to the YAML-assembled
        default LLM instead of attempting to build one from this override.
        """
        return not any([self.model_name, self.base_url, self.api_key_env])


# ---------------------------------------------------------------------------
# Eval config entry + file
# ---------------------------------------------------------------------------


class ScoreWritebackPolicy(BaseModel):
    """Optional source-trace score writeback policy for one Kafka evaluator."""

    model_config = {"extra": "forbid"}

    enabled: bool = Field(default=False, description="是否将 judge 分写回原业务 Langfuse trace")
    score_name: str | None = Field(
        default=None,
        alias="scoreName",
        min_length=1,
        description="原业务 Langfuse score 名称；同 trace 同名会被最新分数覆盖",
    )

    @model_validator(mode="after")
    def _require_score_name_when_enabled(self) -> "ScoreWritebackPolicy":
        if self.enabled and self.score_name is None:
            raise ValueError("scoreName is required when scoreWriteback.enabled is true")
        return self


class EvalConfigEntry(BaseModel):
    """Single eval config entry — one per evaluator per projectId.

    ``evaluator_name`` doubles as the evaluator lookup key: the service maps it
    to ``judge_{evaluator_name}`` and queries ``agent_manager``.  It **must**
    match a registered evaluator — valid values are:

    ===========================  =========================================
    evaluator_name               evaluator
    ===========================  =========================================
    ``"example"``               Generic LLM-as-a-Judge (uses RemoteConfig prompt/model)
    ``"amap_agent"``            Amap-agent evaluator (includes trace evidence extraction)
    ``"competitor_pairwise"``   Competitor pairwise evaluator
    ===========================  =========================================

    Using an unregistered value raises ``model_not_found`` (HTTP 400 /
    Kafka permanent skip).

    ``prompt`` and ``model`` are optional RemoteConfig-level overrides.  When
    present **and non-empty** they take precedence over the YAML-assembled
    defaults at evaluate-time (restart-only; no hot reload).  When empty
    (None / empty string / all-None fields) the evaluator falls back to
    the YAML configuration (``prompts`` and ``models`` sections).

    ``environment`` is an optional list of environment names (e.g.
    ``["prod"]``).  When set, only traces whose ``environment`` field
    matches one of the listed values will be evaluated.  ``None`` or
    empty means all environments.

    ``name`` is an optional list of trace names.  When set, only traces
    whose ``name`` field matches one of the listed values will be
    evaluated.  ``None`` or empty means all names.

    ``trace_name_contains`` is an optional list of substrings.  When
    set, only traces whose ``name`` contains **at least one** of the
    listed substrings will be evaluated (OR semantics).  ``None`` or
    empty means all names pass.  Useful when trace names carry a
    business identifier like ``amap-agent.ws.vui_agent`` and the
    operator wants to route by substring rather than exact match.

    ``trace_name_excludes`` is an optional list of substrings.  When
    set, traces whose ``name`` contains **any** of the listed
    substrings are excluded (OR semantics, applied *after*
    ``trace_name_contains``).  ``None`` or empty means no exclusion.
    Typical use: ``["vui_agent"]`` to keep amap-agent main-dialog
    evaluator from picking up VUI traces sharing the same Langfuse
    project.

    ``tags`` is an optional list of tag strings.  When set, only traces
    that have at least one of the listed tags will be evaluated (set
    intersection must be non-empty).  ``None`` or empty means all tags.

    ``start_time`` / ``end_time`` are optional ISO 8601 timestamps.
    When set, only traces whose ``timestamp`` falls within the range
    ``[start_time, end_time]`` will be evaluated.  ``None`` means no
    bound on that side.

    ``ingest`` selects the realtime path. Default ``kafka`` keeps Path B
    sampling. ``sidecar`` skips Kafka sampling; only sidecar ingest
    extracts that entry.
    """

    model_config = {"extra": "ignore"}

    evaluator_name: str = Field(
        alias="evaluatorName",
        max_length=128,
        description=(
            "统一 agent 名，同时作为 evaluator agent 的查找 key。"
            "服务层映射为 judge_{evaluatorName} 从 agent_manager 取实例。"
            "合法值即已配置 evaluator 的 agent 名：example / amap_agent / "
            "vehicle_agent / search_ai_card / competitor_pairwise / memory_pairwise。"
            "非法值将报 model_not_found（HTTP 400 / Kafka permanent skip）。"
        ),
    )
    agent_name: str | None = Field(
        default=None,
        alias="agentName",
        description=(
            "流级业务归属 agent 名（一个 project 多 agent 场景）："
            "本 entry 的 name/tags 过滤命中的 trace 归属该 agent。"
            "值必须命中 RemoteConfig config.agent.registry 中该 project 登记的"
            "候选集（防拼写错；未命中时 entry 被跳过并 error 日志）。"
            "None = 不指定归属，回退 metadata 顶层 agent_name / registry 唯一候选。"
        ),
    )
    sampling: float = Field(ge=0.0, le=1.0, description="采样率 0..1")
    ingest: Literal["kafka", "sidecar"] = Field(
        default="kafka",
        description="实时入口。缺省 kafka；sidecar 时 Kafka 不采样，由 sidecar ingest 提取。",
    )
    environment: list[str] | None = Field(
        default=None, description='允许的环境列表（如 ["prod"]）；None 或空表示不限'
    )
    name: list[str] | None = Field(
        default=None, description="允许的 trace name 列表（精确匹配）；None 或空表示不限"
    )
    trace_name_contains: list[str] | None = Field(
        default=None,
        alias="traceNameContains",
        description=(
            "trace name 子串白名单（OR 语义）；None 或空表示不限。"
            "trace.name 包含其中任一子串即可通过。典型用途：按业务子图路由 "
            "（如 ['vui_agent'] 让该 config 只接 VUI trace）。"
        ),
    )
    trace_name_excludes: list[str] | None = Field(
        default=None,
        alias="traceNameExcludes",
        description=(
            "trace name 子串黑名单（OR 语义）；None 或空表示不排除。"
            "trace.name 包含其中任一子串即被排除。典型用途：把 VUI trace 排除在 "
            "amap-agent 主对话评测之外（如 ['vui_agent']）。"
        ),
    )
    tags: list[str] | None = Field(
        default=None, description="允许的 trace tag 列表（交集非空即通过）；None 或空表示不限"
    )
    start_time: str | None = Field(
        default=None, alias="startTime", description="时间范围下界（ISO 8601）；None 表示不限制"
    )
    end_time: str | None = Field(
        default=None, alias="endTime", description="时间范围上界（ISO 8601）；None 表示不限制"
    )
    variable_mapping: list[VariableMapping] = Field(
        default_factory=list, alias="variableMapping", description="变量映射列表"
    )
    prompt: str | None = Field(
        default=None, description="RemoteConfig prompt 覆盖（${variable} 语法，重启生效）"
    )
    model: RemoteConfigModelConfig | None = Field(
        default=None, description="RemoteConfig model 配置覆盖（重启生效）"
    )
    agent_registration_id: str | None = Field(
        default=None,
        alias="agentRegistrationId",
        max_length=128,
        description=(
            "业务定制注册 id：实时流槽位键第四段；同茎多 prompt 实例"
            "（example_v56/v57）并行各评的隔离主键"
        ),
    )
    stream_instance_id: str | None = Field(
        default=None,
        alias="streamInstanceId",
        max_length=160,
        description=(
            "实时流实例 id（通常 = 数据集 id，含创建时 ms 时间戳）："
            "同 topic/同 group 下多数据集隔离键；仅采样率不同亦可并存。"
            "写入 record_key 第四段 ``__si__`` 后缀、采样键与落库/报表命名空间。"
        ),
    )
    internal_agent_name: str | None = Field(
        default=None,
        alias="internalAgentName",
        description=(
            "多实例内部 agent 名（服务端分配，如 agx_example_…）："
            "数据面归属与日志展示名；不替代 evaluator_name 路由茎，"
            "也不替代 agentName 的 trace 过滤语义"
        ),
    )
    prompt_bundle_ref: dict[str, Any] | None = Field(
        default=None,
        alias="promptBundleRef",
        description="judge 阶段 prompt 定制引用（PromptBundleRef 形态，缺省无定制）",
    )
    skill_bundle_ref: dict[str, Any] | None = Field(
        default=None,
        alias="skillBundleRef",
        description="judge 阶段 skill 定制引用（SkillBundleRef 形态，缺省无定制）",
    )
    attribution_prompt_bundle_ref: dict[str, Any] | None = Field(
        default=None,
        alias="attributionPromptBundleRef",
        description=(
            "attribution 阶段 prompt 定制引用（与 promptBundleRef 分立；"
            "缺省则归因用平台默认模板，不继承 judge 定制）"
        ),
    )
    attribution_skill_bundle_ref: dict[str, Any] | None = Field(
        default=None,
        alias="attributionSkillBundleRef",
        description=("attribution 阶段 skill 定制引用（与 skillBundleRef 分立；缺省无定制）"),
    )
    attribution_execution: Literal["sync", "async", "none"] | None = Field(
        default=None,
        alias="attributionExecution",
        description="条目级归因执行覆盖；none 时消费侧跳过 triage/归因（注册关闭归因时写入）",
    )
    score_writeback: ScoreWritebackPolicy | None = Field(
        default=None,
        alias="scoreWriteback",
        description="Kafka judge 完成后向原业务 Langfuse trace 写分的可选策略",
    )

    @property
    def is_sidecar_ingest(self) -> bool:
        """True when this entry is extracted by sidecar ingest, not Kafka."""
        return self.ingest == INGEST_SIDECAR


class EvalConfigFile(BaseModel):
    """RemoteConfig config file root — version + projectId → config array mapping."""

    model_config = {"extra": "ignore"}

    version: str = Field(default="1.0", description="配置 schema 版本号")
    configs: dict[str, list[EvalConfigEntry]] = Field(
        default_factory=dict, description="projectId → eval config 数组的映射"
    )


__all__ = [
    "RemoteConfigModelConfig",
    "EvalConfigEntry",
    "EvalConfigFile",
    "INGEST_KAFKA",
    "INGEST_SIDECAR",
    "ScoreWritebackPolicy",
    "VariableMapping",
]
