"""observation → EvaluatorRequest。

单独成模块，避免 sidecar 生命周期 import runner 时和
``sidecar_ingest.converter`` 形成环。
"""

import re
from typing import Any

from benchmark.domain.enums import PIPELINE_SOURCE_METADATA_KEY, PipelineSource
from benchmark.domain.eval_config import EvalConfigEntry, VariableMapping
from benchmark.evaluators.base import EvaluatorRequest, PromptBundleRef, SkillBundleRef
from benchmark.streaming.metadata import (
    AGENT_NAME_METADATA_KEY,
    metadata_agent_name,
    set_metadata_agent_name,
)
from benchmark.streaming.schemas import ObservationForEval

# ---------------------------------------------------------------------------
# observation → EvaluatorRequest conversion
# ---------------------------------------------------------------------------


def to_evaluator_request(
    observation: ObservationForEval,
    config: EvalConfigEntry,
) -> EvaluatorRequest:
    """Convert observation + eval config → :class:`EvaluatorRequest`.

    Field mapping::

        input           ← observation.input
        output          ← observation.output
        trace_id        ← observation.trace_id
        trace_timestamp ← observation.trace_timestamp
        agent_name      ← observation.metadata['agent_name'] (legacy agentName; force-stamp)
        project_id      ← observation.project_id
        metadata        ← observation.metadata (copy) + forced agent_name
        variables       ← config.variable_mapping 按 variableMapping 提取

    ``evaluator_name`` is **not** a field of ``EvaluatorRequest`` — it is
    passed as a separate argument to ``EvaluatorService.evaluate()`` and
    comes from ``config.evaluator_name``.
    """
    metadata = dict(observation.metadata) if observation.metadata else {}
    # Stamp kafka source for trace_meta_json filtering (streaming 实时流形态).
    metadata[PIPELINE_SOURCE_METADATA_KEY] = PipelineSource.KAFKA.value
    # 多实例归属键（设计 §12.2）：注册 id 透传供日志/落库区分同茎多版本；
    # 内部名仍写入供实例限流分桶，日志侧一律用 registrationId。
    if config.agent_registration_id:
        metadata["agent_registration_id"] = config.agent_registration_id.strip()
    if config.stream_instance_id:
        metadata["stream_instance_id"] = config.stream_instance_id.strip()
    if config.internal_agent_name:
        metadata["internal_agent_name"] = config.internal_agent_name
    # Normalize identity key to snake_case before judge / attribution.
    resolved_agent = metadata_agent_name(metadata)
    agent_name: str | None = None
    if isinstance(resolved_agent, str) and resolved_agent.strip():
        agent_name = resolved_agent.strip()
        set_metadata_agent_name(metadata, agent_name)

    # 对齐 Langfuse：按 variable_mapping 从 observation 提取数据填充 variables
    variables = _build_variables_from_mapping(observation, config.variable_mapping)

    request = EvaluatorRequest(
        input=observation.input,
        output=observation.output,
        metadata=metadata,
        variables=variables,
        trace_id=observation.trace_id,
        trace_timestamp=observation.trace_timestamp,
        agent_name=agent_name,
        project_id=observation.project_id,
        # Kafka consumer already has the Langfuse primary trace ID (record.id);
        # skip before_eval metadata resolution that would fail with trace_not_found.
        skip_trace_resolve=True,
        # RemoteConfig prompt / judge bundle：仅 judge 阶段；attribution 用独立字段。
        prompt_text=config.prompt,
        model_config_override=config.model,
        # 业务定制：按 stage 分发；stage 不匹配则该阶段为 None（用平台默认）
        prompt_bundle_ref=_coerce_prompt_bundle(config.prompt_bundle_ref, "judge"),
        skill_bundle_ref=_coerce_skill_bundle(config.skill_bundle_ref, "judge"),
        attribution_prompt_bundle_ref=_coerce_prompt_bundle(
            config.attribution_prompt_bundle_ref,
            "attribution",
        ),
        attribution_skill_bundle_ref=_coerce_skill_bundle(
            config.attribution_skill_bundle_ref,
            "attribution",
        ),
    )

    # Force agent_name into metadata even when only present on Request top-level
    # (some converters / callers set agent_name without writing metadata).
    # EvaluatorRequest is frozen — use model_copy so the stamp is reliable.
    if isinstance(request.agent_name, str) and request.agent_name.strip():
        stamped_name = request.agent_name.strip()
        stamped_meta = dict(request.metadata) if request.metadata else {}
        if stamped_meta.get(AGENT_NAME_METADATA_KEY) != stamped_name:
            set_metadata_agent_name(stamped_meta, stamped_name)
            request = request.model_copy(
                update={"metadata": stamped_meta, "agent_name": stamped_name},
            )
        elif request.agent_name != stamped_name:
            request = request.model_copy(update={"agent_name": stamped_name})
    return request


# ---------------------------------------------------------------------------
# variable_mapping helpers — Langfuse-aligned observation field extraction
# ---------------------------------------------------------------------------


def _build_variables_from_mapping(
    observation: ObservationForEval,
    variable_mapping: list[VariableMapping],
) -> dict[str, Any]:
    """对齐 Langfuse ``variableMapping``：按映射列表从 observation/trace 提取数据。

    对每条 mapping：
      1. 按 ``selected_column_id`` 从 observation 取原始值；
      2. 若有 ``json_selector``（JSONPath），从原始值中提取子值；
      3. 以 ``template_variable`` 为 key 填入返回 dict。

    mapping 列表为空时返回空 dict（evaluator 回退到硬编码解析）。
    """
    variables: dict[str, Any] = {}
    for mapping in variable_mapping:
        value = _extract_field(observation, mapping.selected_column_id)
        if mapping.json_selector and value is not None:
            value = _apply_json_selector(value, mapping.json_selector)
        variables[mapping.template_variable] = value
    return variables


def _extract_field(observation: ObservationForEval, selected_column_id: str) -> object:
    """按 ``selected_column_id`` 从 observation 提取字段值。

    支持的字段名对齐 Langfuse ``ObservationForEval`` schema：
    - observation 级别: input / output / metadata / tool_calls
    - trace 级别: trace_name / user_id / session_id / tags / name / type / environment
    """
    field_map = {
        "input": observation.input,
        "output": observation.output,
        "metadata": observation.metadata,
        "tool_calls": observation.tool_calls,
        "trace_name": observation.trace_name,
        "user_id": observation.user_id,
        "session_id": observation.session_id,
        "tags": observation.tags,
        "name": observation.name,
        "type": observation.type,
        "environment": observation.environment,
    }
    return field_map.get(selected_column_id)


def _apply_json_selector(value: object, json_selector: str) -> object:  # noqa: PLR0911, PLR0912
    """简化 JSONPath 实现：支持 ``$.key.path[0].subkey`` 语法。

    对齐 Langfuse ``jsonSelector`` 语义——从复杂 JSON 中按路径提取子值。
    支持点号访问 (``.key``)、索引访问 (``[0]``)、通配符 (``[*]``)。
    不依赖第三方 JSONPath 库，仅覆盖 Langfuse 常见用法。
    """
    if not json_selector.startswith("$"):
        return value

    path = json_selector[1:]  # strip leading $
    current: object = value
    tokens = re.findall(r"\.(\w+)|\[(\d+)\]|(\[\*\])", path)
    for key, idx, wildcard in tokens:
        if wildcard == "[*]":
            if not isinstance(current, list):
                return None
            # [*] — map remaining path over each element
            remaining_start = path.index("[*]") + 3
            remaining = path[remaining_start:]
            if remaining:
                return [_apply_json_selector(item, "$" + remaining) for item in current]
            return current
        elif key:
            if isinstance(current, dict):
                current = current.get(key)
            elif hasattr(current, key):
                current = getattr(current, key, None)
            else:
                return None
        elif idx:
            if isinstance(current, list | tuple):
                i = int(idx)
                current = current[i] if i < len(current) else None
            else:
                return None
        if current is None:
            return None
    return current


def _coerce_prompt_bundle(
    raw: dict[str, Any] | PromptBundleRef | None,
    expected_stage: str,
) -> PromptBundleRef | None:
    """条目 dict → PromptBundleRef；stage 不匹配或非法则丢弃（该阶段用默认）。"""
    if raw is None:
        return None
    try:
        ref = raw if isinstance(raw, PromptBundleRef) else PromptBundleRef.model_validate(raw)
    except Exception:
        return None
    return ref if ref.stage == expected_stage else None


def _coerce_skill_bundle(
    raw: dict[str, Any] | SkillBundleRef | None,
    expected_stage: str,
) -> SkillBundleRef | None:
    """条目 dict → SkillBundleRef；stage 不匹配或非法则丢弃。"""
    if raw is None:
        return None
    try:
        ref = raw if isinstance(raw, SkillBundleRef) else SkillBundleRef.model_validate(raw)
    except Exception:
        return None
    return ref if ref.stage == expected_stage else None
