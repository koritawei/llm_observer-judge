"""AttributionAgent（注册名 attribution_example）。

收 :class:`AgentRequest`，经框架级 ``trace`` ToolProvider 按需取证，
产出 :class:`AttributionResult` → :class:`AgentResponse`。
不再接受 ``raw_trace`` / 本地 evidence expand。
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.agents import AgentCard, AgentProviderContext, BaseAgent
from benchmark.libs.agenthost.agents.component_metadata import agent_component
from benchmark.libs.agenthost.agents.config.schemas import AgentConfigSchema
from benchmark.libs.agenthost.agents.request import AgentRequest, AgentResponse, ContentPart, ResponseStatus
from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import Field, ValidationError

from benchmark.paths import default_conf_dir
from benchmark.attribution.badcase.llm_attribution import run_llm_attribution
from benchmark.attribution.badcase.schemas import (
    DEFAULT_MAX_TOOL_CALLS_PER_ROUND,
    AttributionResult,
    BadcaseAttributionDecision,
    BadcaseAttributionRuntimeOptions,
    JudgeFailureSignal,
    TraceContextPack,
)
from benchmark.trace.tools import (
    TraceRequestBound,
    TraceToolProvider,
    bind_trace_request,
)
from business.agents.example._internal.llm import (
    AttributionLLM,
    ChatModelAttributionLLM,
)
from business.agents.example.constants import (
    DEFAULT_BADCASE_PROMPT_NAME,
    PROMPT_FILE_SUFFIX,
)

# default_conf_dir() 已经是 conf，这里只写其下的目录。
_PROMPT_DIR_UNDER_CONF = ("agents", "attribution_example", "prompts")

logger = get_logger(__name__)

#: tool 请求/返回 INFO 预览上限，避免 skeleton/detail 整包刷爆日志
_TOOL_LOG_PREVIEW_CHARS = 500
#: final pass 仍要 tool 时强制 synthesis 的 confidence 下调（Option A）
_FORCE_SYNTHESIS_CONFIDENCE_HAIRCUT = 0.1

_FINAL_PASS_CONSTRAINT = """
## Constraint
This is the FINAL pass. Tools are DISABLED.
You MUST return decision_type=final or human_review based on User Query,
Judge Failure Signal, Observed Output, and Tool Observations above.
Do NOT return decision_type=tool_calls.
Prefer final when any useful evidence exists; use human_review only if evidence is
empty, contradictory, or unsafe. Do not invent unobserved span content.
""".strip()

_FORCE_SYNTHESIS_CONSTRAINT = """
## Constraint (forced synthesis — tools disabled)
Tool budget is exhausted and a previous pass still requested tools.
You MUST NOT return decision_type=tool_calls.
Based ONLY on User Query, Judge Failure Signal, Observed Output, and Tool
Observations above, prefer decision_type=final with the best attribution you can justify.
Use human_review only if evidence is empty, contradictory, or unsafe to conclude.
Do not invent span content not present in Tool Observations.
""".strip()

_OUTPUT_CONTRACT = """
## Output Contract
Return exactly ONE JSON object (no markdown fence).
- `decision_type` is REQUIRED; allowed values: `tool_calls` | `final` | `human_review`.
- `decision_type=tool_calls` → include non-empty `tool_calls`; do NOT include `final`.
- `decision_type=final` or `human_review` → include `final` (AttributionResult);
  do NOT include `tool_calls`.
- Do NOT emit `score` or `reason` keys: the judge score above is INPUT only,
  never your output schema.
""".strip()


def _preview_for_log(value: Any, *, max_chars: int = _TOOL_LOG_PREVIEW_CHARS) -> str:  # noqa: ANN401
    """Serialize value for log lines; truncate long tool bodies."""

    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}…[truncated len={len(text)}]"


class AttributionAgentConfigSchema(AgentConfigSchema):
    """AttributionAgent 配置契约。"""

    type: str | None = Field(default=None, description="由 framework 消费,本 schema 不校验")
    default_model: str | None = Field(default=None, description="引用 models.<key>")
    badcase_prompt_name: str = DEFAULT_BADCASE_PROMPT_NAME
    badcase_runtime_options: BadcaseAttributionRuntimeOptions = Field(
        default_factory=BadcaseAttributionRuntimeOptions
    )


_TRACE_PROVIDER = "trace"


@agent_component(name="attribution_example", ext_schema=AttributionAgentConfigSchema)
class AttributionAgent(BaseAgent):
    """attribution 动态 agent — Langfuse tool calling ReAct。"""

    def __init__(
        self,
        *,
        name: str,
        llm: AttributionLLM | None,
        prompt_manager: object | None,
        prompt_name: str,
        max_react_rounds: int,
        runtime_options: BadcaseAttributionRuntimeOptions,
        runtime: object | None = None,
        tool_provider: TraceToolProvider | None = None,
    ) -> None:
        super().__init__(name=name)
        self._llm = llm
        self._prompt_manager = prompt_manager
        self._prompt_name = prompt_name
        self._max_react_rounds = max_react_rounds
        self._runtime_options = runtime_options
        self._runtime = runtime
        # 单测可显式注入；生产路径从 runtime.tool_manager 懒取框架级 trace
        self._tool_provider_override = tool_provider

    @classmethod
    def from_config(
        cls,
        cfg: AttributionAgentConfigSchema,
        ctx: AgentProviderContext,
    ) -> "AttributionAgent":
        runtime = ctx.runtime
        attribution_llm: AttributionLLM | None = None
        if cfg.default_model:
            attribution_llm = ChatModelAttributionLLM(lambda: runtime.get_llm(cfg.default_model))
        return cls(
            name=ctx.name,
            llm=attribution_llm,
            prompt_manager=runtime.prompt_manager,
            prompt_name=cfg.badcase_prompt_name,
            max_react_rounds=cfg.badcase_runtime_options.max_react_rounds,
            runtime_options=cfg.badcase_runtime_options,
            runtime=runtime,
            tool_provider=None,
        )

    def _resolve_trace_provider(self) -> TraceToolProvider | None:
        """懒取框架级 ``trace``（YAML tools.providers 注册，业务无需再声明）。"""

        if self._tool_provider_override is not None:
            return self._tool_provider_override
        runtime = self._runtime
        if runtime is None:
            return None
        tool_manager = getattr(runtime, "tool_manager", None)
        if tool_manager is None:
            return None
        provider = tool_manager.get_provider(_TRACE_PROVIDER)
        if isinstance(provider, TraceToolProvider):
            return provider
        return None

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.name,
            description=("agent-eval 归因动态 agent — Langfuse 按需 tool 取证 + ReAct 归因"),
        )

    async def run(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        return await self.call(request, **kwargs)

    async def call(self, request: AgentRequest, **kwargs: object) -> AgentResponse:
        try:
            result = await self._run_tool_react(request)
        except ValidationError as exc:
            # 模型输出未满足 BadcaseAttributionDecision 契约（典型形态：把 Judge
            # Failure Signal 的 score/reason 当模板抄回来，缺 decision_type）。
            # 这是模型侧格式漂移，不是链路故障：降级为 human_review，保住已落库的
            # judge 分与整条评测（attribution_failed 会被提升为 500，而 Kafka
            # offset 已提前提交，该条 trace 不会重试）。
            fallback = _human_review(
                _request_trace_id(request),
                "attribution LLM output did not match the decision schema "
                "(decision_type missing or invalid).",
            )
            logger.warning(
                "attribution_agent.schema_fallback trace_id=%s detail=%s",
                request.traceid,
                exc,
            )
            return AgentResponse(
                traceid=request.traceid,
                status=ResponseStatus.SUCCESS,
                content_parts=[ContentPart(type="text", data=fallback.model_dump_json())],
                metadata={
                    "attribution_result": fallback.model_dump(mode="json"),
                    "case_id": fallback.case_id,
                    "fallback": "schema_validation",
                },
            )
        except Exception as exc:
            logger.exception("attribution_agent.call failed trace_id=%s", request.traceid)
            return AgentResponse(
                traceid=request.traceid,
                status=ResponseStatus.FAILED,
                metadata={"error_code": "attribution_failed", "error": str(exc)},
            )
        return AgentResponse(
            traceid=request.traceid,
            status=ResponseStatus.SUCCESS,
            content_parts=[ContentPart(type="text", data=result.model_dump_json())],
            metadata={
                "attribution_result": result.model_dump(mode="json"),
                "case_id": result.case_id,
            },
        )

    async def _run_tool_react(  # noqa: PLR0911, PLR0912, PLR0915
        self, request: AgentRequest
    ) -> AttributionResult:
        metadata = dict(request.metadata or {})
        trace_id = str(metadata.get("trace_id") or request.traceid or "")
        project_id = metadata.get("project_id")
        judge = _extract_judge_signal(metadata)
        observed = _extract_observed_output(metadata)
        user_query = _extract_user_query(metadata)

        # trace_id 缺失是上游契约错误。显式返回 human_review 避免后续
        # 调 LLM / tool 浪费预算 + 给出可定位的 action_hint。
        if not trace_id:
            case_id = str(metadata.get("case_id") or request.traceid or "")
            return AttributionResult(
                trace_id="",
                case_id=case_id,
                category="unknown",
                root_cause="trace_id is required for badcase attribution.",
                confidence=0.0,
                reasoning="trace_id is required for badcase attribution.",
                evidence_refs=[],
                action_hint="fix_upstream_trace_id_contract",
                needs_human_review=True,
            )

        provider = self._resolve_trace_provider()
        bound: TraceRequestBound | None = None
        if provider is not None:
            try:
                bound = provider.bound_from_metadata(metadata, trace_id=trace_id or None)
            except ValueError as exc:
                logger.warning("attribution.bind_failed err=%s", exc)
                return _human_review(trace_id, f"cannot bind langfuse tools: {exc}")

        if self._llm is None:
            return _human_review(
                trace_id,
                "attribution LLM is not configured (set agents.attribution_example.default_model)",
            )

        system_prompt = await self._load_system_prompt(request)
        observations: list[str] = []
        seen_refs: set[str] = set()
        last_decision_type = "none"
        # max_react_rounds = max tool rounds; one extra LLM pass consumes tool results.
        max_tool_rounds = self._max_react_rounds
        total_llm_passes = max_tool_rounds + 1

        pack = TraceContextPack(
            trace_id=trace_id,
            case_id=trace_id,
        )
        threshold = self._runtime_options.confidence_threshold

        for round_idx in range(total_llm_passes):
            is_final_pass = round_idx >= max_tool_rounds
            evidence_prompt = _build_evidence_prompt(
                EvidencePromptInput(
                    trace_id=trace_id,
                    project_id=str(project_id) if project_id else "",
                    judge=judge,
                    observed_output=observed,
                    user_query=user_query,
                    observations=observations,
                    round_idx=round_idx + 1,
                    max_rounds=total_llm_passes,
                    tools_allowed=not is_final_pass,
                )
            )
            decision = await run_llm_attribution(
                llm=self._llm,
                system_prompt=system_prompt,
                evidence_prompt=evidence_prompt,
                context_pack=pack,
                llm_call_timeout_s=self._runtime_options.llm_call_timeout_s,
            )
            last_decision_type = decision.decision_type

            if decision.decision_type == "tool_calls":
                if is_final_pass:
                    # Option A: harness 事件不上屏；强制基于已有上下文再归因一次。
                    logger.warning(
                        "attribution.final_pass_still_requested_tools "
                        "trace_id=%s max_tool_rounds=%s — forcing synthesis",
                        trace_id,
                        max_tool_rounds,
                    )
                    force_prompt = _build_evidence_prompt(
                        EvidencePromptInput(
                            trace_id=trace_id,
                            project_id=str(project_id) if project_id else "",
                            judge=judge,
                            observed_output=observed,
                            user_query=user_query,
                            observations=observations,
                            round_idx=round_idx + 1,
                            max_rounds=total_llm_passes,
                            tools_allowed=False,
                            force_synthesis=True,
                        )
                    )
                    decision = await run_llm_attribution(
                        llm=self._llm,
                        system_prompt=system_prompt,
                        evidence_prompt=force_prompt,
                        context_pack=pack,
                        llm_call_timeout_s=self._runtime_options.llm_call_timeout_s,
                    )
                    last_decision_type = decision.decision_type
                    if decision.final is not None:
                        return _materialize_final(
                            decision=decision,
                            trace_id=trace_id,
                            seen_refs=seen_refs,
                            confidence_threshold=threshold,
                            force_synthesized=True,
                        )
                    return _human_review(
                        trace_id,
                        "insufficient evidence to conclude after tool budget; "
                        "could not synthesize a reliable root cause from "
                        "available context.",
                    )
                if provider is None or bound is None:
                    return _human_review(
                        trace_id,
                        "trace tool provider is not registered "
                        "(check tools.providers + AGENT_EVAL_LANGFUSE_INSTANCES)",
                    )
                # Fixed per-round cap (not a runtime YAML knob — avoids CR/config drift).
                per_round_cap = DEFAULT_MAX_TOOL_CALLS_PER_ROUND
                calls = list(decision.tool_calls)
                if len(calls) > per_round_cap:
                    dropped = len(calls) - per_round_cap
                    observations.append(
                        f"[round={round_idx + 1} truncated {dropped} tool_calls "
                        f"(per_round_cap={per_round_cap})]"
                    )
                    calls = calls[:per_round_cap]
                with bind_trace_request(bound):
                    for call in calls:
                        args = dict(call.arguments or {})
                        logger.info(
                            "attribution.tool_call trace_id=%s round=%s tool=%s args=%s",
                            trace_id,
                            round_idx + 1,
                            call.name,
                            _preview_for_log(args),
                        )
                        try:
                            result = await self._invoke_trace_tool(provider, call.name, args)
                        except Exception as exc:  # noqa: BLE001 — feed model, do not FAIL agent
                            logger.warning(
                                "attribution.tool_raised tool=%s err=%s",
                                call.name,
                                exc,
                            )
                            observations.append(
                                f"[round={round_idx + 1} tool={call.name} success=False]\n{exc}"
                            )
                            continue
                        body = result.to_text() if hasattr(result, "to_text") else str(result)
                        success = bool(getattr(result, "success", True))
                        logger.info(
                            "attribution.tool_result trace_id=%s round=%s tool=%s "
                            "success=%s body_chars=%s preview=%s",
                            trace_id,
                            round_idx + 1,
                            call.name,
                            success,
                            len(body),
                            _preview_for_log(body),
                        )
                        if success:
                            # evidence gate 以 bound.known_refs 为准，不重解析 body
                            seen_refs.update(bound.known_refs)
                        observations.append(
                            f"[round={round_idx + 1} tool={call.name} success={success}]\n{body}"
                        )
                continue

            if decision.final is not None:
                return _materialize_final(
                    decision=decision,
                    trace_id=trace_id,
                    seen_refs=seen_refs,
                    confidence_threshold=threshold,
                )

        return _human_review(
            trace_id,
            "insufficient evidence to conclude after tool budget; "
            f"last_decision={last_decision_type}.",
        )

    async def _invoke_trace_tool(
        self,
        provider: TraceToolProvider,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Execute via provider; bound comes from ``bind_trace_request`` ContextVar.

        Other callers may use ``ToolManager.call_tool`` the same way after
        ``bind_trace_request`` — AttributionAgent already holds the provider.
        """

        return await provider.call_tool(tool_name, arguments)

    async def _load_system_prompt(self, request: AgentRequest) -> str:
        """先选定正文，再在替换逻辑之外拼工具说明。"""

        body = await self._load_prompt_body(request)
        return f"{body.rstrip()}\n\n{_read_tool_prompt()}".strip()

    async def _load_prompt_body(self, request: AgentRequest) -> str:
        """选定归因 system 正文（工具说明在 _load_system_prompt 外层拼接）。

        优先级：``attribution_prompt_text`` → attribution-stage bundle（若可解析）
        → PromptManager → conf 文件。**不读** ``judge_prompt_text`` /
        ``prompt_text`` / judge-stage bundle；未指定则用默认模板。
        """

        metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        override = _attribution_prompt_text(metadata)
        if override:
            return override

        base = await self._default_prompt_body()
        return await _apply_attribution_bundle_if_present(metadata, base)

    async def _default_prompt_body(self) -> str:
        """平台默认归因正文：PromptManager → conf 文件。"""

        pm = self._prompt_manager
        if pm is not None:
            try:
                loaded = await pm.load_prompt(self._prompt_name)
                loaded_text = str(loaded or "")
                if loaded_text.strip():
                    return loaded_text
            except Exception:  # noqa: BLE001 — load 失败走 conf 文件
                pass
        path = default_conf_dir().joinpath(
            *_PROMPT_DIR_UNDER_CONF,
            f"{self._prompt_name}{PROMPT_FILE_SUFFIX}",
        )
        if path.is_file():
            return path.read_text(encoding="utf-8")
        logger.warning("attribution.example.prompt_file_missing path=%s", path)
        return ""


def _render_judge_signal(judge: JudgeFailureSignal) -> str:
    """判分结论以自然语言呈现，避免被模型当作输出结构模仿。"""

    return "\n".join(
        [
            "以下判分结论是**待归因的输入**，不是你应输出的结构：",
            f"- 判分：{judge.score}",
            f"- 判分器：{judge.evaluator or '-'}",
            f"- 判分理由：{judge.reason or '-'}",
        ]
    )


def _request_trace_id(request: AgentRequest) -> str:
    """归因请求的 trace id（与 _run_tool_react 同源，metadata 优先）。"""

    metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
    return str(metadata.get("trace_id") or request.traceid or "")


def _human_review(trace_id: str, reason: str) -> AttributionResult:
    return AttributionResult(
        trace_id=trace_id,
        case_id=trace_id,
        category="unknown",
        root_cause=reason,
        confidence=0.0,
        reasoning=reason,
        evidence_refs=[],
        action_hint="manual_review",
        needs_human_review=True,
        should_dispatch=False,
    )


def _materialize_final(
    *,
    decision: BadcaseAttributionDecision,
    trace_id: str,
    seen_refs: set[str],
    confidence_threshold: float,
    force_synthesized: bool = False,
) -> AttributionResult:
    """Apply evidence / confidence gates to an LLM final / human_review decision."""

    base = decision.final
    if base is None:
        return _human_review(trace_id, "LLM returned empty final payload.")

    if force_synthesized:
        haircut = max(0.0, float(base.confidence) - _FORCE_SYNTHESIS_CONFIDENCE_HAIRCUT)
        base = base.model_copy(update={"confidence": haircut})
        logger.info(
            "attribution.force_synthesized trace_id=%s confidence=%.3f (haircut=%.2f)",
            trace_id,
            haircut,
            _FORCE_SYNTHESIS_CONFIDENCE_HAIRCUT,
        )

    missing_refs = [ref for ref in base.evidence_refs if ref and ref not in seen_refs]
    reasons: list[str] = []
    if base.needs_human_review:
        reasons.append("llm_flagged")
    if base.confidence < confidence_threshold:
        reasons.append(
            f"confidence_below_threshold({base.confidence:.3f}<{confidence_threshold:.3f})"
        )
    if decision.decision_type == "human_review":
        reasons.append("decision_type_human_review")
    if missing_refs:
        reasons.append(f"missing_evidence_refs({','.join(missing_refs)})")

    escalated = bool(reasons)
    if escalated:
        logger.info(
            "attribution.escalated_to_human_review trace_id=%s "
            "reasons=%s confidence=%.3f threshold=%.3f",
            trace_id,
            ",".join(reasons),
            base.confidence,
            confidence_threshold,
        )
    update: dict[str, Any] = {
        "trace_id": base.trace_id or trace_id,
        "case_id": base.case_id or trace_id,
        "needs_human_review": base.needs_human_review or escalated,
        "raw_response": decision.raw_response,
    }
    if escalated:
        # 证据不安全时不得回写 triage_level（避免半成品推翻）。
        update["should_dispatch"] = False
        update["corrected_triage_level"] = None
    elif base.corrected_triage_level is not None:
        # 归因推翻 judge：以 corrected_triage_level 为准，拒派 Aone。
        update["should_dispatch"] = False
    return base.model_copy(update=update)


def _read_tool_prompt() -> str:
    """工具说明。不参与正文替换，读 conf 下 attribution_example 的 tool.prompt。"""

    path = default_conf_dir().joinpath(
        *_PROMPT_DIR_UNDER_CONF,
        f"tool{PROMPT_FILE_SUFFIX}",
    )
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    logger.warning("attribution.example.tool_prompt_missing path=%s", path)
    return ""


def _attribution_prompt_text(metadata: Mapping[str, Any]) -> str:
    """从评测请求快照取归因 system 正文。空串表示本次请求没带。"""

    er = metadata.get("evaluator_request")
    if not isinstance(er, Mapping):
        return ""
    text = er.get("attribution_prompt_text")
    if isinstance(text, str) and text.strip():
        return text
    return ""


async def _apply_attribution_bundle_if_present(
    metadata: Mapping[str, Any],
    default_text: str,
) -> str:
    """仅当 metadata 顶层 bundle 的 stage=attribution 时覆盖默认正文。

    judge-stage refs（历史错误地把打分定制带进归因）一律忽略，回退默认。
    customization_resolution 未部署时保持默认正文，不阻塞归因。
    """
    from benchmark.evaluators.prompt_stages import is_stage  # noqa: PLC0415

    prompt_ref = metadata.get("prompt_bundle_ref")
    skill_ref = metadata.get("skill_bundle_ref")
    if not (is_stage(prompt_ref, "attribution") or is_stage(skill_ref, "attribution")):
        return default_text
    try:
        from benchmark.evaluators.customization_resolution import (  # noqa: PLC0415
            resolve_stage_prompt_from_text,
        )
    except ImportError:
        return default_text
    try:
        return await resolve_stage_prompt_from_text(metadata, "prompt", default_text)
    except Exception as exc:  # noqa: BLE001 — bundle 失败不阻断，回退默认
        logger.warning(
            "attribution.example.bundle_resolve_failed err=%s — use default prompt",
            type(exc).__name__,
        )
        return default_text


def _extract_judge_signal(metadata: Mapping[str, Any]) -> JudgeFailureSignal:
    er = metadata.get("evaluator_result")
    if isinstance(er, Mapping):
        return JudgeFailureSignal.model_validate(
            {
                "score": er.get("score"),
                "reason": er.get("reason") or "",
                "evaluator": er.get("evaluator") or "",
            }
        )
    return JudgeFailureSignal()


def _extract_observed_output(metadata: Mapping[str, Any]) -> str:
    er = metadata.get("evaluator_request")
    if isinstance(er, Mapping):
        for key in ("output", "assistant_answer", "answer"):
            val = er.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return ""


def _extract_user_query(metadata: Mapping[str, Any]) -> str:
    """从 evaluator_request.input.request_body.query 抽用户原始问题。

    缺失时返回空串 — evidence prompt 显示 "-"，模型可按 prompt 约定
    通过 fetch_observation_detail 读 root span 兜底。
    """

    er = metadata.get("evaluator_request")
    if not isinstance(er, Mapping):
        return ""
    inp = er.get("input")
    if not isinstance(inp, Mapping):
        return ""
    body = inp.get("request_body")
    if not isinstance(body, Mapping):
        return ""
    query = body.get("query")
    if isinstance(query, str) and query.strip():
        return query
    return ""


@dataclass(frozen=True)
class EvidencePromptInput:
    """一轮归因取证 prompt 的组装素材（失败现场 + 轮次进度 + 工具权限）。"""

    trace_id: str
    project_id: str
    judge: JudgeFailureSignal
    observed_output: str
    user_query: str
    observations: list[str]
    round_idx: int
    max_rounds: int
    tools_allowed: bool = True
    force_synthesis: bool = False


def _build_evidence_prompt(prompt_input: EvidencePromptInput) -> str:
    trace_id = prompt_input.trace_id
    project_id = prompt_input.project_id
    judge = prompt_input.judge
    observed_output = prompt_input.observed_output
    user_query = prompt_input.user_query
    observations = prompt_input.observations
    round_idx = prompt_input.round_idx
    max_rounds = prompt_input.max_rounds
    tools_allowed = prompt_input.tools_allowed
    force_synthesis = prompt_input.force_synthesis
    sections = [
        "# Attribution Context",
        f"- trace_id: {trace_id or '-'}",
        f"- project_id: {project_id or '-'}",
        f"- round: {round_idx}/{max_rounds}",
        f"- tools_allowed: {'yes' if tools_allowed and not force_synthesis else 'no'}",
        "",
        "## User Query",
        user_query or "-",
        "",
        "## Judge Failure Signal",
        # 判分结论用自然语言呈现：此前 model_dump 的 JSON 会被模型当作输出模板，
        # 直接抄成 {"score": ..., "reasoning": ...} 并丢掉 decision_type。
        _render_judge_signal(judge),
        "",
        "## Observed Output",
        observed_output or "-",
    ]
    if observations:
        sections.extend(["", "## Tool Observations", *observations])
    else:
        sections.extend(
            [
                "",
                "## Tool Observations",
                (
                    "(none — conclude from Judge Failure Signal / Observed Output, "
                    "or human_review if unsafe)"
                    if force_synthesis or not tools_allowed
                    else "(none yet — call fetch_trace_skeleton first)"
                ),
            ]
        )
    if force_synthesis:
        sections.extend(["", _FORCE_SYNTHESIS_CONSTRAINT])
    elif not tools_allowed:
        sections.extend(["", _FINAL_PASS_CONSTRAINT])
    # 输出契约放最后：模型对末尾约束更敏感，且要与上面的 tool 状态约束一致。
    sections.extend(["", _OUTPUT_CONTRACT])
    return "\n".join(sections)


__all__ = ["AttributionAgent", "AttributionAgentConfigSchema"]
