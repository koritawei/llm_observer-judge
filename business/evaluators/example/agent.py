"""ExampleEvaluatorAgent — the reference evaluator.

Implements the two required methods (:meth:`from_config` + :meth:`evaluate`).
``evaluate`` runs a 3-node LangGraph: ``build_prompt`` → ``evaluate`` (LLM) →
``response`` (``{score, reason}``); any node may set ``error`` / ``error_code`` to
short-circuit to END, surfaced as an :class:`EvaluatorError`.

Assembled by ``AppRuntime`` from the ``agents.<name>`` block in ``conf/app.<env>.yaml``
(``type`` matches ``@agent_component(name=...)``): it references ``default_model`` →
``models.<key>`` and ``prompt`` → a template under ``prompts.template_dir``; it builds
no model/prompt infra itself.
"""

import json
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.agents.component_metadata import agent_component
from benchmark.libs.agenthost.agents.config.context import AgentProviderContext
from benchmark.libs.agenthost.models.schemas import ModelConfigSchema
from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.agenthost.prompts import PromptManager
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, JsonValue, ValidationError

from benchmark.evaluators.base import (
    DEFAULT_DESCRIPTION,
    BaseEvaluatorAgent,
    EvaluatorRequest,
    EvaluatorResult,
)
from benchmark.evaluators.customization_resolution import (
    prompt_provenance,
    resolve_stage_prompt,
)
from business.evaluators.example.checks_tool import ToolNameCheck, check_tool_names
from business.evaluators.example.llm import ChatModelLLM, EvaluatorLLM, build_remote_config_llm
from business.evaluators.example.nodes.build_prompt import DEFAULT_SCORE_MAX, BuildPromptNode
from business.evaluators.example.nodes.evaluate import EvaluateNode
from business.evaluators.example.nodes.response import ResponseNode
from business.evaluators.example.schema import build_case_variables, parse_output
from business.evaluators.example.scoring import EvaluationResult
from business.evaluators.example.state import (
    ERROR,
    ERROR_CODE,
    MODEL_OVERRIDE,
    PROMPT_OVERRIDE,
    RENDER_CONTEXT,
    RESULT,
    ExampleState,
)
from benchmark.evaluators.prompt_stages import judge_bundle_refs
from benchmark.runs.common.dynamic_rubric import resolve_dynamic_rubric
from business.evaluators.example.grading import grade_fused_score
from benchmark.trace.tools import TraceToolProvider
from benchmark.utils.exceptions import (
    EVALUATION_FAILED_CODE,
    EvaluatorError,
    evaluator_status,
)

logger = get_logger("agent_eval.evaluator.example")

# 运行时注入的模板变量（非 case 载荷字段）。
# case 侧宏由 build_case_variables 按 input/output 实际字段展平，可随 prompt 扩展。
# expected_output：有值才进袋；无值时 build_prompt 兜底「(未配置期望输出)」。
_VAR_TRACE_EVIDENCE = "trace_evidence"
_VAR_EXPECTED_OUTPUT = "expected_output"

_TRACE_PROVIDER = "trace"

# 工具说明不进可替换的打分正文。照抄派生时改这一段，不要改 BuildPromptNode。
_JUDGE_TOOL_PROMPT = """
# 工具说明
本段由评测器在正文之外附上。judge 调用 `fetch_trace_skeleton`
取本 trace 的 span 骨架，不调用 detail / find。

工具调用列表只有工具名称，用来判断是否调了该调的工具。没有工具返回内容时，不要因此认定事实无法核实或编造。

Trace Process Evidence 是可选辅助，不是评分门槛：
- 仅当 preview 中明确出现的事实与回答直接矛盾时，才可据此压分（可至 0）。
- preview 截断、没出现某事实、trace 不可用 → 一律忽略，
  不得记「无法核实」、不得封顶、不得当失败原因。
- 禁止因「事实无法用 trace / 工具返回核实」而封顶。
- 禁止因 trace 为空、`(trace 不可用)`、preview 截断而压分。
- 禁止把「无法核实」写成失败摘要。

## 工具调用列表（仅工具名称）
${used_tool}

## Trace Process Evidence（可选；仅用于发现明确矛盾；为空或 (trace 不可用) 时忽略）
${trace_evidence}
""".strip()

#: ${trace_evidence} 的兜底值 — 与 evaluator.prompt 的"(trace 不可用)"约定一致，
#: 避免 safe_substitute 把字面 ${trace_evidence} 占位符渲染给 LLM。
_TRACE_UNAVAILABLE = "(trace 不可用)"


class ExampleEvaluatorConfigSchema(BaseModel, extra="forbid"):
    """ext_schema for an ``agents.<name>`` block of ``type: example`` — only references
    into the standard sections (``default_model`` → ``models.<key>``, ``prompt`` → a template).
    """

    description: str = Field(default=DEFAULT_DESCRIPTION, description="Agent-card 描述")
    default_model: str = Field(description="引用 models.<key>（runtime.get_llm 解析）")
    prompt: str = Field(description="prompts.template_dir 下的模板名")
    sub_score_axes: list[str] = Field(
        default_factory=list,
        description=(
            "业务自定义分项维度名列表。空列表 = 不产分项（老行为）。"
            "LLM 需按此 axes 输出 sub_scores 字典；axes 顺序无关。"
            "不应包含 'score'（总分由顶层字段承载）。"
        ),
    )
    sub_score_scale: tuple[float, float] | None = Field(
        default=None,
        description=(
            "(min, max) 闭区间，约束 LLM 的 sub_scores 取值范围。"
            "None = 不强制约束（向后兼容）。如 (0, 1) / (0, 10)。"
        ),
    )
    dimension_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "分项权重（键须与 sub_score_axes 对齐，和非负）。空 = 不声明动态权重"
            "（框架回退静态 Rubric / LLM 顶层 score）。"
            "有值时注入 prompt，并在 LLM 未回写 weights 时回填 extra.weights。"
        ),
    )
    attribution_threshold: float | None = Field(
        default=None,
        description=(
            "归因门限（fused < 此值触发 attribution）。None = 不注入、不回填"
            "（框架回退业务注册的 AttributionTriggerPolicy / 默认策略）。"
            "有值时注入 prompt ${attribution_threshold}，LLM 未回写则回填 extra。"
        ),
    )


@dataclass(frozen=True)
class ExampleEvaluatorDeps:
    """``ExampleEvaluatorAgent`` 构造依赖（PLR0913 / FUNC-PARAMS 收敛）。"""

    llm: EvaluatorLLM
    prompt_manager: PromptManager
    prompt_name: str
    sub_score_axes: list[str] | None = None
    sub_score_scale: tuple[float, float] | None = None
    dimension_weights: dict[str, float] | None = None
    attribution_threshold: float | None = None
    runtime: object | None = None
    trace_provider: TraceToolProvider | None = None
    default_model_name: str = ""


@agent_component(name="judge_example", ext_schema=ExampleEvaluatorConfigSchema)
class ExampleEvaluatorAgent(BaseEvaluatorAgent):
    """Reference evaluator: render prompt → call LLM → parse ``{score, reason}``."""

    def __init__(
        self,
        *,
        name: str,
        description: str | None,
        deps: ExampleEvaluatorDeps,
    ) -> None:
        """LLM seam + prompt manager from the runtime; graph compiled lazily."""

        super().__init__(name=name, description=description)
        self._llm = deps.llm
        self._prompt_manager = deps.prompt_manager
        self._prompt_name = deps.prompt_name
        self._sub_score_axes = list(deps.sub_score_axes or [])
        self._sub_score_scale = deps.sub_score_scale
        self._dimension_weights = dict(deps.dimension_weights or {})
        self._attribution_threshold = deps.attribution_threshold
        self._runtime = deps.runtime
        self._trace_provider_override = deps.trace_provider
        self._default_model_name = deps.default_model_name
        self._compiled_graph: CompiledStateGraph | None = None

    @classmethod
    def from_config(
        cls, cfg: ExampleEvaluatorConfigSchema, ctx: AgentProviderContext
    ) -> "ExampleEvaluatorAgent":
        """Build from the parsed ext_schema + ``ctx``. LLM is resolved lazily via
        ``ctx.runtime.get_llm(cfg.default_model)`` (missing api_key → request-time
        ``llm_failed``, not a boot failure)."""

        runtime = ctx.runtime
        llm = ChatModelLLM(lambda: runtime.get_llm(cfg.default_model))

        # Read model_name from YAML config as fallback for RemoteConfig override
        # (when RemoteConfig model omits modelName, build_remote_config_llm uses this).
        default_model_name = ""
        try:
            model_schema = runtime.app_config.get_resource(
                "models", cfg.default_model, ModelConfigSchema
            )
            default_model_name = getattr(model_schema, "model_name", "") or ""
        except Exception:
            logger.debug("from_config: cannot read model_name from YAML for %s", cfg.default_model)

        return cls(
            name=ctx.name,
            description=cfg.description,
            deps=ExampleEvaluatorDeps(
                llm=llm,
                prompt_manager=runtime.prompt_manager,
                prompt_name=cfg.prompt,
                sub_score_axes=cfg.sub_score_axes,
                sub_score_scale=cfg.sub_score_scale,
                dimension_weights=cfg.dimension_weights,
                attribution_threshold=cfg.attribution_threshold,
                runtime=runtime,
                default_model_name=default_model_name,
            ),
        )

    # ── the one business method: evaluate ────────────────────────────────────

    async def evaluate(self, request: EvaluatorRequest) -> EvaluatorResult:
        # tool_check 仅本 agent 消费。None 走下方原路径。名称硬检失败返回 score=0，
        # 不拉 trace、不进图；形状不合法则 EvaluatorError。通过后 tool_check 不进渲染变量。
        if request.tool_check is not None:
            rejected = _reject_tool_names(request)
            if rejected is not None:
                return rejected
        # 1) input/output/expected 中存在的字段 → ${宏} 袋（prompt 宏可扩展）；
        # 2) variableMapping 提取的 request.variables 覆盖同名键；
        # 3) expected_output 仅在袋中尚无时补写（避免空串盖住 mapping）。
        variables = build_case_variables(
            request.input,
            request.output,
            expected_output=request.expected_output,
        )
        if request.variables:
            for key, value in request.variables.items():
                variables[key] = value if isinstance(value, str) else _as_text(value)
        expected_text = _as_text(request.expected_output).strip()
        if expected_text and _VAR_EXPECTED_OUTPUT not in variables:
            variables[_VAR_EXPECTED_OUTPUT] = expected_text
        # 预拉 trace 骨架（如可用）→ 注入 prompt，judge LLM 看到骨架后再判。
        # trace 不可用时注入 "(trace 不可用)" 哨兵，配合 prompt 里
        # "本节为 (trace 不可用) 时视为不可用、按无法核实封顶"的约定。
        trace_evidence = await self._fetch_trace_evidence(request)
        variables[_VAR_TRACE_EVIDENCE] = trace_evidence or _TRACE_UNAVAILABLE
        # 仅 judge-stage bundle；否则 judge_prompt_text / prompt_text；再否则文件默认。
        prompt_extra: dict[str, object] | None = None
        judge_prompt_ref, judge_skill_ref = judge_bundle_refs(request)
        if judge_prompt_ref is not None or judge_skill_ref is not None:
            # 通用字段可能混放 attribution ref；解析器只接收已过滤的 judge refs。
            judge_request = request.model_copy(
                update={
                    "prompt_bundle_ref": judge_prompt_ref,
                    "skill_bundle_ref": judge_skill_ref,
                }
            )
            prompt_override = await resolve_stage_prompt(
                judge_request, self._prompt_manager, "prompt", self._prompt_name
            )
            prompt_extra = prompt_provenance(judge_prompt_ref, judge_skill_ref)
        else:
            prompt_override = _judge_prompt_text_override(request)
            if prompt_override:
                # 请求正文直替路径：无 Git 版本号，仍落库可查询的来源标记。
                prompt_extra = {
                    "stage": "judge",
                    "content_version": None,
                    "source": "request_text",
                    "chars": len(prompt_override),
                }
        initial: ExampleState = {
            RENDER_CONTEXT: variables,
            PROMPT_OVERRIDE: prompt_override,
            MODEL_OVERRIDE: request.model_config_override,
        }
        final = await self._graph().ainvoke(initial)

        # 失败也带回 prompt 溯源，worker 写入 criterion_scores.prompt_customization。
        fail_details: dict[str, Any] | None = (
            {"prompt_customization": prompt_extra} if prompt_extra is not None else None
        )
        if final.get(ERROR):
            error_type = final.get(ERROR_CODE) or EVALUATION_FAILED_CODE
            error = EvaluatorError(
                code=evaluator_status(error_type),
                error_type=error_type,
                message=str(final.get(ERROR) or ""),
            )
            error.details = fail_details
            raise error
        verdict = final.get(RESULT)
        if verdict is None:
            error = EvaluatorError(
                code=evaluator_status(EVALUATION_FAILED_CODE),
                error_type=EVALUATION_FAILED_CODE,
                message="no verdict produced",
            )
            error.details = fail_details
            raise error
        extra = self._build_result_extra(verdict, prompt_extra)
        # 统一服务层、实时流和批量链均消费服务端计算分；模型分保留在 extra。
        return EvaluatorResult(
            score=float(extra["computed_fused_score"]),
            reason=verdict.reason,
            sub_scores=verdict.sub_scores,
            extra=extra,
        )

    def _build_result_extra(
        self,
        verdict: EvaluationResult,
        prompt_extra: dict[str, object] | None,
    ) -> dict[str, Any]:
        """组装 EvaluatorResult.extra：prompt 溯源 + 动态权重/归因门限/grade（框架写入）。"""
        extra: dict[str, Any] = {}
        if prompt_extra is not None:
            extra["prompt_customization"] = prompt_extra
        # example 显式 opt-in 动态协议；声明值与 LLM 返回值分离，供严格校验与审计。
        extra["dynamic_rubric_opt_in"] = True
        extra["dynamic_rubric_opt_in_source"] = "example_agent"
        extra["declared_weights"] = dict(self._dimension_weights)
        extra["model_score"] = float(verdict.score)
        if verdict.raw_json is not None:
            extra["raw_judge_json"] = verdict.raw_json
        extra["declared_sub_score_axes"] = list(self._sub_score_axes)
        # 分制由显式 scale 或 example 默认 10 分制决定；不使用归因门限定级。
        score_scale = (
            list(self._sub_score_scale)
            if self._sub_score_scale is not None
            else [0.0, float(DEFAULT_SCORE_MAX)]
        )
        extra["sub_score_scale"] = score_scale
        # weights / 返回门限只保存模型实际返回值；缺失由 resolver 统一回退并审计。
        weights = verdict.weights if verdict.weights else None
        if weights:
            extra["weights"] = weights
        attr_threshold = verdict.attribution_threshold
        if attr_threshold is not None:
            extra["attribution_threshold"] = float(attr_threshold)

        resolved = resolve_dynamic_rubric(
            {"score": verdict.score, "sub_scores": verdict.sub_scores, "extra": extra}
        )
        computed_score = resolved.fused_score if resolved.fused_score is not None else verdict.score
        extra["computed_fused_score"] = float(computed_score)
        if resolved.score_mismatch:
            extra["score_mismatch"] = True
        if resolved.fallback_reason is not None:
            extra["dynamic_rubric_fallback_reason"] = resolved.fallback_reason
        extra["grade"] = grade_fused_score(
            float(computed_score),
            score_scale_max=score_scale[1],
        )
        return extra

    # ── graph ────────────────────────────────────────────────────────────────

    def _build_state_graph(self) -> CompiledStateGraph:
        # 3 phases by purpose: build_prompt (frame) → evaluate (judge) → response (verdict).
        prompt_node = BuildPromptNode(
            self._prompt_manager,
            self._prompt_name,
            sub_score_axes=self._sub_score_axes,
            sub_score_scale=self._sub_score_scale,
            dimension_weights=self._dimension_weights,
            attribution_threshold=self._attribution_threshold,
            tool_prompt=_JUDGE_TOOL_PROMPT,
        )
        evaluate_node = EvaluateNode(
            self._llm,
            remote_config_model_builder=lambda override: build_remote_config_llm(
                override, default_model_name=self._default_model_name
            ),
        )
        response_node = ResponseNode(sub_score_axes=self._sub_score_axes)

        async def _build_prompt(state: ExampleState) -> dict[str, Any]:
            return await prompt_node(dict(state))

        async def _evaluate(state: ExampleState) -> dict[str, Any]:
            return await evaluate_node(dict(state))

        async def _response(state: ExampleState) -> dict[str, Any]:
            return await response_node(dict(state))

        def _route_after_prompt(state: ExampleState) -> str:
            return END if state.get(ERROR) else EvaluateNode.NODE_NAME

        def _route_after_evaluate(state: ExampleState) -> str:
            return END if state.get(ERROR) else ResponseNode.NODE_NAME

        builder = StateGraph(ExampleState)
        builder.add_node(BuildPromptNode.NODE_NAME, _build_prompt)
        builder.add_node(EvaluateNode.NODE_NAME, _evaluate)
        builder.add_node(ResponseNode.NODE_NAME, _response)
        builder.add_edge(START, BuildPromptNode.NODE_NAME)
        builder.add_conditional_edges(BuildPromptNode.NODE_NAME, _route_after_prompt)
        builder.add_conditional_edges(EvaluateNode.NODE_NAME, _route_after_evaluate)
        builder.add_edge(ResponseNode.NODE_NAME, END)
        return builder.compile()

    def _graph(self) -> CompiledStateGraph:
        if self._compiled_graph is None:
            self._compiled_graph = self._build_state_graph()
        return self._compiled_graph

    # ── trace skeleton pre-fetch (照搬 main 上 judge_amap_agent 模式) ────────────

    def _resolve_trace_provider(self) -> TraceToolProvider | None:
        """懒取框架级 ``trace``（YAML tools.providers 注册，业务无需再声明）。"""

        if self._trace_provider_override is not None:
            return self._trace_provider_override
        runtime = self._runtime
        if runtime is None:
            return None
        tool_manager = getattr(runtime, "tool_manager", None)
        if tool_manager is None:
            return None
        try:
            provider = tool_manager.get_provider(_TRACE_PROVIDER)
        except (KeyError, LookupError):
            return None
        if isinstance(provider, TraceToolProvider):
            return provider
        return None

    async def _fetch_trace_evidence(self, request: EvaluatorRequest) -> str | None:
        """拉 trace 骨架（仅 fetch_trace_skeleton，不调 detail/find）。

        Returns the skeleton text to inject into ``${trace_evidence}``, or
        ``None`` when trace is unavailable (no provider / no trace_id / no
        project_id / fetch failed). The caller falls back to "无法核实" via
        the prompt's empty-evidence convention.
        """

        provider = self._resolve_trace_provider()
        if provider is None or not request.trace_id or not request.project_id:
            return None

        try:
            bound = provider.create_bound(
                trace_id=request.trace_id,
                project_id=request.project_id,
            )
            result = await provider.call_tool(
                "fetch_trace_skeleton",
                {},
                bound=bound,
            )
        except Exception as exc:  # noqa: BLE001 — silent fallback to "unverifiable"
            logger.warning(
                "evaluator.example.trace_fetch_raised trace_id=%s project_id=%s error_type=%s",
                request.trace_id,
                request.project_id,
                type(exc).__name__,
            )
            return None

        if not getattr(result, "success", False):
            logger.warning(
                "evaluator.example.trace_fetch_failed trace_id=%s project_id=%s",
                request.trace_id,
                request.project_id,
            )
            return None

        body = result.to_text() if hasattr(result, "to_text") else str(result)
        return body or None


def _reject_tool_names(request: EvaluatorRequest) -> EvaluatorResult | None:
    """Apply ``request.tool_check`` before the graph.

    Returns a zero-score result when a name rule fails. Returns None when the
    spec is unconstrained or the names pass, so ``evaluate`` continues.
    Raises :class:`EvaluatorError` (400) when ``tool_check`` does not match
    :class:`ToolNameCheck`.
    """

    try:
        spec = ToolNameCheck.model_validate(request.tool_check)
    except ValidationError as exc:
        error_type = "invalid_request_body"
        raise EvaluatorError(
            code=evaluator_status(error_type),
            error_type=error_type,
            message=f"tool_check 无效：{exc}",
        ) from exc
    used_tool = parse_output(request.output).content_desc.used_tool
    result = check_tool_names(
        used_tool,
        required=spec.required,
        forbidden=spec.forbidden,
        must_be_empty=spec.must_be_empty,
    )
    if result.ok:
        return None
    return EvaluatorResult(score=0.0, reason=result.reason)


def _judge_prompt_text_override(request: EvaluatorRequest) -> str | None:
    """文本覆盖：``judge_prompt_text`` → ``prompt_text``；不读 attribution。"""

    for value in (request.judge_prompt_text, request.prompt_text):
        if isinstance(value, str) and value.strip():
            return value
    return None


def _as_text(value: JsonValue) -> str:
    """Render a JsonValue as text — a str verbatim, anything else JSON-encoded."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)
