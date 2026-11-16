"""Example evaluator schema — input/output payload models + prompt rendering.

1. Payload models — the evaluator's ``input`` / ``output`` (agent request/response
   snapshots) arrive as parsed JSON carrying more than needed; only consumed fields
   are declared (extra ignored). ``parse_input`` / ``parse_output`` 统一接受：

   - dict 结构体
   - JSON 字符串（CSV / 传输层常见，与 dict 等价）
   - 纯文本（归一到标准形状，避免 ``${query}`` / ``${answer}`` 静默为空）

   标准形状：

   - input: ``{"request_body": {"query": "..."}}``
   - output: ``{"component": {"type": "markdown", "text": "..."}, "response_text": "..."}``

2. Case variables — :func:`build_case_variables` 把 input/output/expected 里**实际存在**
   的字段展平为 ``${name}`` 袋；prompt 宏可变，渲染时只替换袋中有的键
   （``safe_substitute``）。

3. Prompt rendering — :func:`render_template` fills ``${name}`` placeholders via the
   stdlib :class:`string.Template` (no regex).
"""

import json
import re
from string import Template

from pydantic import BaseModel, Field, JsonValue, ValidationError

#: ``string.Template`` 合法标识符；非法键无法写成 ``${...}``，展平时跳过。
_TEMPLATE_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


# ── input: the agent-request snapshot ─────────────────────────────────────────


class InputRequestBody(BaseModel):
    query: str = ""
    session_id: str = ""
    conversation_id: str = ""
    task_id: str = ""
    scene_id: str = ""
    dialog_type: str = ""


class EvaluatorInput(BaseModel):
    """The input payload — only ``request_body`` is consumed."""

    request_body: InputRequestBody = Field(default_factory=InputRequestBody)


# ── output: the agent-response snapshot ───────────────────────────────────────


class OutputComponent(BaseModel):
    type: str = ""
    text: str = ""
    is_done: bool = False


class OutputContentDesc(BaseModel):
    category: str = ""
    subcategory: str = ""
    used_tool: list[str] = Field(default_factory=list)


class EvaluatorOutput(BaseModel):
    """The output payload — ``component`` + ``response_text`` + ``content_desc``."""

    task_id: str = ""
    code: int = 0
    component: OutputComponent = Field(default_factory=OutputComponent)
    content_desc: OutputContentDesc = Field(default_factory=OutputContentDesc)
    response_text: str = ""
    session_id: str = ""
    conversation_id: str = ""


def _decode_json_string(value: JsonValue) -> JsonValue:
    """JSON 字符串 → 结构体；已是 dict/list 或非 JSON 纯文本则原样。"""

    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _text(value: object) -> str:
    if value is None:
        return ""
    return value.strip() if isinstance(value, str) else str(value).strip()


def parse_input(value: JsonValue) -> EvaluatorInput:
    """Validate ``input``；JSON 字符串 / dict / 纯文本均归一到 request_body.query。"""

    decoded = _decode_json_string(value)
    if isinstance(decoded, str):
        text = decoded.strip()
        if not text:
            return EvaluatorInput()
        return EvaluatorInput(request_body=InputRequestBody(query=text))
    if isinstance(decoded, dict):
        request_body = decoded.get("request_body")
        body = dict(request_body) if isinstance(request_body, dict) else {}
        query = _text(body.get("query")) or _text(decoded.get("query"))
        if query and not _text(body.get("query")):
            body = {**body, "query": query}
            decoded = {**decoded, "request_body": body}
    try:
        return EvaluatorInput.model_validate(decoded)
    except ValidationError:
        return EvaluatorInput()


def parse_output(value: JsonValue) -> EvaluatorOutput:
    """Validate ``output``；JSON/dict/纯文本归一，并双写 component.text ↔ response_text。"""

    decoded = _decode_json_string(value)
    if isinstance(decoded, str):
        text = decoded.strip()
        if not text:
            return EvaluatorOutput()
        return EvaluatorOutput(
            component=OutputComponent(type="markdown", text=text),
            response_text=text,
        )
    try:
        parsed = EvaluatorOutput.model_validate(decoded)
    except ValidationError:
        return EvaluatorOutput()

    text = _text(parsed.component.text) or _text(parsed.response_text)
    if not text:
        return parsed
    component = parsed.component.model_copy(
        update={
            "text": _text(parsed.component.text) or text,
            "type": _text(parsed.component.type) or "markdown",
        }
    )
    return parsed.model_copy(
        update={
            "component": component,
            "response_text": _text(parsed.response_text) or text,
        }
    )


# ── case → ${name} bag（prompt 宏可扩展：存在才注入）──────────────────────────


def _stringify_var(value: object) -> str:
    """模板变量统一成 str；str 原样，其余 JSON（ensure_ascii=False）。"""

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _put_var(bag: dict[str, str], key: str, value: object | None) -> None:
    """键合法且值非 None 时写入；空串视为「存在」仍注入。"""

    if value is None or not _TEMPLATE_IDENT_RE.match(key):
        return
    bag[key] = _stringify_var(value)


def _flatten_mapping(bag: dict[str, str], mapping: dict[str, object]) -> None:
    """一层 KV 写入袋；嵌套 dict/list 整段序列化，不递归炸开。"""

    for key, value in mapping.items():
        _put_var(bag, key, value)


def build_case_variables(
    input_value: JsonValue,
    output_value: JsonValue,
    *,
    expected_output: JsonValue = None,
) -> dict[str, str]:
    """从 input/output/expected 构建可扩展的 prompt 变量袋。

    规则：

    - input 为 dict：顶层键 + ``request_body`` 内键（同名时 request_body 覆盖顶层）；
    - output 为 dict：顶层键；
    - 语义别名始终优先写回：``query`` / ``answer`` / ``used_tool`` / ``response_text``；
    - 整包：``input`` / ``output``（有值才写）；
    - ``expected_output``：有非空文本才写（避免空串盖住 mapping 已提取值）；
    - 非法标识符键跳过；值为 ``None`` 视为不存在。

    prompt 含 ``${scene_id}`` 且 input.request_body 带 ``scene_id`` → 替换；
    prompt 无该宏 → 袋中多出的键被 Template 忽略；宏有、字段无 → 渲染为空串
    （``render_template``；运行时宏仍可由 build_prompt ``setdefault`` 注入哨兵）。
    """

    bag: dict[str, str] = {}

    decoded_input = _decode_json_string(input_value)
    if isinstance(decoded_input, dict):
        _flatten_mapping(bag, decoded_input)
        request_body = decoded_input.get("request_body")
        if isinstance(request_body, dict):
            _flatten_mapping(bag, request_body)
        _put_var(bag, "input", decoded_input)
    elif decoded_input is not None and _text(decoded_input):
        _put_var(bag, "input", decoded_input)
        _put_var(bag, "query", _text(decoded_input))

    decoded_output = _decode_json_string(output_value)
    if isinstance(decoded_output, dict):
        _flatten_mapping(bag, decoded_output)
        _put_var(bag, "output", decoded_output)
    elif decoded_output is not None and _text(decoded_output):
        _put_var(bag, "output", decoded_output)

    eval_input = parse_input(input_value)
    eval_output = parse_output(output_value)
    body = eval_input.request_body
    # 语义别名始终覆盖扁平键（含空串），避免默认 prompt 留下字面 ${query}/${answer}。
    _put_var(bag, "query", body.query)
    for field_name in (
        "session_id",
        "conversation_id",
        "task_id",
        "scene_id",
        "dialog_type",
    ):
        field_value = getattr(body, field_name, "") or ""
        if field_value:
            _put_var(bag, field_name, field_value)

    # 对齐旧逻辑：component.text → response_text → 整包 JSON（含 dict）。
    # 自定义字段（如 result/data.text）不进 EvaluatorOutput 时，仍用整包兜底，
    # 避免 ${answer} 静默变空导致默认 prompt 按「空回答」打分。
    answer = (
        eval_output.component.text
        or eval_output.response_text
        or (_stringify_var(decoded_output) if decoded_output is not None else "")
    )
    _put_var(bag, "answer", answer)
    _put_var(bag, "response_text", eval_output.response_text)
    _put_var(bag, "used_tool", ", ".join(eval_output.content_desc.used_tool))

    expected_text = _stringify_var(expected_output).strip() if expected_output is not None else ""
    if expected_text:
        _put_var(bag, "expected_output", expected_text)

    return bag


# ── prompt rendering — ${name} via stdlib string.Template ─────────────────────


def render_template(template: str, variables: dict[str, str]) -> str:
    """Fill ``${name}`` placeholders from ``variables`` (``$$`` is a literal ``$``).

    袋中无对应键的宏替换为空串（不保留字面 ``${name}``）。
    """

    tpl = Template(template)
    merged = {name: "" for name in tpl.get_identifiers()}
    merged.update(variables)
    return tpl.safe_substitute(merged)
