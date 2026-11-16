"""BaseEvaluatorAgent — the development standard for an evaluator.

An evaluator is an ``agentgraph`` agent selected by the request's ``model`` field
(model = evaluator name). The base hides all framework plumbing (AgentRequest/
AgentResponse translation, error mapping, agent card); a new evaluator implements
only :meth:`from_config` (construct) and :meth:`evaluate` (the logic).

- In: :class:`EvaluatorRequest` — ``input`` / ``output`` (the agent request/response
  snapshots, format not fixed) + ``metadata`` / ``trace_id`` / ``agent_name`` /
  ``project_id``. Parsed once by the API layer; the evaluator just reads fields.
- Out: :class:`EvaluatorResult` — ``score`` (required) + ``reason`` / ``extra``.
- Errors: raise :class:`EvaluatorError` with a ``*_CODE`` constant (prompt-not-found
  → 400, llm-* → 502, else → 500).
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from benchmark.libs.agenthost.agents.base import BaseAgent
from benchmark.libs.agenthost.agents.card import AgentCard
from benchmark.libs.agenthost.agents.config.context import AgentProviderContext
from benchmark.libs.agenthost.agents.request import (
    AgentRequest,
    AgentResponse,
    ContentPart,
    ResponseStatus,
)
from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, JsonValue

from benchmark.domain.schemas import RemoteConfigModelConfig

# EvaluatorError + the *_CODE constants live in benchmark.utils.exceptions (single source);
# base imports EvaluatorError (used by call()) + SERVER_ERROR_STATUS — consumers import from there.
from benchmark.utils.exceptions import SERVER_ERROR_STATUS, EvaluatorError

logger = get_logger("agent_eval.evaluator.base")

#: Default agent-card description if a subclass / config does not set one.
DEFAULT_DESCRIPTION = "An agent-eval evaluator."


class PromptBundleRef(BaseModel, frozen=True):
    """业务 prompt 定制引用（只含定位与指纹，正文由 BundleStore 按 commit 物化）。"""

    stage: Literal["judge", "attribution"]
    git_path: str
    git_commit: str
    bundle_sha256: str
    content_version: int | None = Field(
        default=None,
        description="业务定制 contentVersion（目录 vN）；缺省时由 git_path 末段推导",
    )
    slots: list[str] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(default_factory=list)


class SkillBundleRef(BaseModel, frozen=True):
    """业务 skill 定制引用（系统级前置指令，正文由 BundleStore 物化）。"""

    stage: Literal["judge", "attribution"]
    git_path: str
    git_commit: str
    bundle_sha256: str
    content_version: int | None = Field(
        default=None,
        description="业务定制 contentVersion（目录 vN）；缺省时由 git_path 末段推导",
    )
    entry: str = "SKILL.md"
    files: list[dict[str, Any]] = Field(default_factory=list)


class EvaluatorRequest(BaseModel, frozen=True):
    """Pre-parsed request bundle (input to ``evaluate``); built once by the API layer.

    ``input`` / ``output`` are :class:`pydantic.JsonValue` — the agent request/response
    snapshots, format not fixed; each evaluator interprets them as it sees fit.

    Prompt 阶段隔离（互不影响，未指定则各用默认模板）：
    - judge：``judge_prompt_text`` → ``prompt_text``（RemoteConfig 打分覆盖）→
      ``prompt_bundle_ref``（stage=judge）→ 文件默认
    - attribution：``attribution_prompt_text`` →
      ``attribution_prompt_bundle_ref`` → 文件默认；**禁止**读取 judge 字段
    """

    input: JsonValue = None
    output: JsonValue = None
    # 可选原始 observation 列表；None 表示未提供，空列表表示已获取但无证据。
    # 由业务 evaluator 装配，不把正文塞进 metadata，也不预加工 input/output。
    observations: list[dict[str, Any]] | None = None
    # Langfuse dataset item 的期望输出（ground truth）；仅 dataset eval 路径填充。
    # ``output`` 始终是被评对象的实际输出，二者语义分离；evaluator 可选消费
    # （如 amap_agent 的 ideal_answer 对齐），不消费时行为不变。
    expected_output: JsonValue = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # 对齐 Langfuse variableMapping：按 config.variable_mapping 从 observation/trace
    # 提取的数据，key = template_variable（prompt 模板变量名），value = 提取值。
    # 为空时 evaluator 回退到硬编码解析逻辑（兼容无 variable_mapping 的旧配置）。
    variables: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    # Original source-trace request time, normalized once at the ingestion boundary.
    trace_timestamp: str | int | None = None
    agent_name: str | None = None
    project_id: str | None = None
    # Kafka consumer path (Path B) already has the Langfuse primary trace ID
    # (record.id); HTTP path (Path A) has a business trace ID needing resolution.
    # Dataset eval path also sets this to True (sourceTraceId is already a
    # Langfuse primary id). When True, ``before_eval`` skips metadata-based resolve.
    skip_trace_resolve: bool = False
    # Request-level prompt version alias (optional; evaluator 映射为具体 prompt
    # 模板，如 vui 的 ``vui_agent_jsonpath``)；白名单见各 evaluator 的版本表。
    prompt_version: str | int | None = None
    # RemoteConfig-level overrides (optional; when present, take precedence over
    # the YAML-assembled prompt/model at evaluate-time).
    #: RemoteConfig / 条目级 **judge** 打分模板覆盖（不进 attribution）。
    prompt_text: str | None = None
    judge_prompt_text: str | None = Field(
        default=None,
        description=(
            "本次请求携带的 judge 打分模板正文。"
            "example judge 非空时替换默认文件模板；空则回退 prompt_text，再回退文件。"
            "不参与 attribution。"
        ),
    )
    attribution_prompt_text: str | None = Field(
        default=None,
        description=(
            "本次请求携带的归因 system 正文。"
            "example 归因非空时替换默认文件模板；不参与 judge 渲染。"
        ),
    )
    model_config_override: RemoteConfigModelConfig | None = None
    #: judge 阶段业务定制 prompt（stage 必须为 judge；attribution 勿复用）
    prompt_bundle_ref: PromptBundleRef | None = None
    skill_bundle_ref: SkillBundleRef | None = None
    #: attribution 阶段业务定制（与 judge 字段分立，互不影响）
    attribution_prompt_bundle_ref: PromptBundleRef | None = None
    attribution_skill_bundle_ref: SkillBundleRef | None = None
    tool_check: dict[str, Any] | None = Field(
        default=None,
        description=(
            "仅 judge_example 消费。其它 evaluator 必须忽略。缺省 None 与未带该字段时一致。"
        ),
    )


class EvaluatorResult(BaseModel, frozen=True):
    """Verdict returned by :meth:`evaluate`: ``score`` (required) + ``reason`` / ``extra``
    / ``sub_scores`` (optional).

    ``extra`` carries anything triage may want (e.g. user_query / assistant_answer previews).
    ``sub_scores`` carries per-axis numeric ratings (业务自定义 axes，ext_schema 声明);
    None = 老行为，平台不消费。
    """

    score: float
    reason: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    sub_scores: dict[str, float] | None = None


class BaseEvaluatorAgent(BaseAgent, ABC):
    """Base for every evaluator.

    You implement :meth:`evaluate` + :meth:`from_config` (typed domain layer);
    ``call`` / ``run`` / ``get_agent_card`` are provided — do not override.
    Chain: ``EvaluatorService → agent.call() → self.evaluate()``.
    """

    #: Agent-card description; set per evaluator via this attr or the ``description=`` arg.
    default_description: ClassVar[str] = DEFAULT_DESCRIPTION

    def __init__(self, *, name: str, description: str | None = None) -> None:
        super().__init__(name=name)
        self.description = description or self.default_description

    # ════════════════════════════════════════════════════════════════════════
    # Domain layer — IMPLEMENT THESE TWO (and nothing else)
    # ════════════════════════════════════════════════════════════════════════

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: BaseModel | None, ctx: AgentProviderContext) -> "BaseEvaluatorAgent":
        """Construct the evaluator (once, at bootstrap; not per request).

        Read deps off ``ctx`` (e.g. ``ctx.runtime``) + ``cfg``, then return
        ``cls(name=ctx.name, ...)``.
        """

    @abstractmethod
    async def evaluate(self, request: EvaluatorRequest) -> EvaluatorResult:
        """The evaluation logic — your one method.

        Read the fields you need off ``request`` (already parsed); return an
        :class:`EvaluatorResult` (``score`` required). On a known failure raise
        :class:`EvaluatorError` with a ``*_CODE`` constant; other exceptions → 500.
        """

    # ── Framework layer — PROVIDED, DO NOT OVERRIDE ──────────────────────────

    async def call(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        """Framework entry the service invokes — do not override.

        Deserializes the :class:`EvaluatorRequest` from the message content, runs
        :meth:`evaluate`, and serializes the result (or the :class:`EvaluatorError` JSON
        payload on a caught failure) back into the response content.
        """

        parts = request.messages[0].content if request.messages else []
        eval_request = EvaluatorRequest.model_validate_json(decode_eval_text(parts) or "{}")
        try:
            result = await self.evaluate(eval_request)
        except EvaluatorError as exc:
            if exc.code >= SERVER_ERROR_STATUS:
                logger.error(
                    "evaluator.failed name=%s type=%s status=%s",
                    self.name,
                    exc.error_type,
                    exc.code,
                )
            else:
                logger.warning(
                    "evaluator.failed name=%s type=%s status=%s",
                    self.name,
                    exc.error_type,
                    exc.code,
                )
            return AgentResponse(
                traceid=request.traceid,
                status=ResponseStatus.FAILED,
                content_parts=encode_eval_error(exc),
            )
        return AgentResponse(
            traceid=request.traceid,
            status=ResponseStatus.SUCCESS,
            content_parts=encode_eval_json(result),
        )

    async def run(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        """Legacy ``BaseAgent`` entry (``@abstractmethod``) — do not override; delegates
        to :meth:`call`."""

        return await self.call(request, **kwargs)

    def get_agent_card(self) -> AgentCard:
        """Card from name + description (satisfies pre-push Agent Card governance)."""

        return AgentCard(name=self.name, description=self.description)


EVAL_CONTENT_TYPE = "text"


def encode_eval_json(model: BaseModel) -> list[ContentPart]:
    """Pack a pydantic model as a single text :class:`ContentPart` carrying its JSON."""

    return [ContentPart(type=EVAL_CONTENT_TYPE, data=model.model_dump_json())]


def encode_eval_error(exc: EvaluatorError) -> list[ContentPart]:
    """Pack an :class:`EvaluatorError` as the single text :class:`ContentPart` (its JSON)."""

    return [ContentPart(type=EVAL_CONTENT_TYPE, data=exc.to_payload())]


def decode_eval_text(parts: list[ContentPart]) -> str:
    """Return the JSON text from the first text :class:`ContentPart` (``""`` if none)."""

    for part in parts:
        if part.type == EVAL_CONTENT_TYPE and isinstance(part.data, str):
            return part.data
    return ""


__all__ = [
    "DEFAULT_DESCRIPTION",
    "EVAL_CONTENT_TYPE",
    "BaseEvaluatorAgent",
    "EvaluatorRequest",
    "EvaluatorResult",
    "PromptBundleRef",
    "SkillBundleRef",
    "decode_eval_text",
    "encode_eval_error",
    "encode_eval_json",
]
