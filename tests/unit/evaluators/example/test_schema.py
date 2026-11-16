"""Unit tests for the example evaluator schema — payload models + ${name} rendering."""
# ruff: noqa: PLR2004  (test-only: literal asserts)

from business.evaluators.example.schema import (
    build_case_variables,
    parse_input,
    parse_output,
    render_template,
)

# ── payload models ──


def test_parse_input_extracts_query() -> None:
    value = {"request_body": {"query": "你是谁", "session_id": "s"}, "viewer_context": {}}
    parsed = parse_input(value)
    assert parsed.request_body.query == "你是谁"
    assert parsed.request_body.session_id == "s"


def test_parse_input_ignores_unknown_fields() -> None:
    value = {"request_body": {"query": "q", "libra_ab_test": {"x": "1"}}, "os": "ANDH"}
    parsed = parse_input(value)
    assert parsed.request_body.query == "q"


def test_parse_input_plain_text_becomes_query() -> None:
    parsed = parse_input("你是谁")
    assert parsed.request_body.query == "你是谁"


def test_parse_input_decodes_json_string() -> None:
    value = '{"request_body":{"query":"这周边到处乱碰，看碰得出去不？"}}'
    parsed = parse_input(value)
    assert parsed.request_body.query == "这周边到处乱碰，看碰得出去不？"


def test_parse_input_top_level_query_without_request_body() -> None:
    parsed = parse_input({"query": "顶层问题"})
    assert parsed.request_body.query == "顶层问题"


def test_parse_output_extracts_text_and_tools() -> None:
    value = {
        "component": {"type": "markdown", "text": "答案"},
        "content_desc": {"used_tool": ["get_weather", "search_place"]},
        "response_text": "答案",
    }
    parsed = parse_output(value)
    assert parsed.component.text == "答案"
    assert parsed.response_text == "答案"
    assert parsed.content_desc.used_tool == ["get_weather", "search_place"]


def test_parse_output_plain_text_becomes_component_and_response_text() -> None:
    parsed = parse_output("plain text answer")
    assert parsed.component.text == "plain text answer"
    assert parsed.component.type == "markdown"
    assert parsed.response_text == "plain text answer"


def test_parse_output_decodes_json_string() -> None:
    value = (
        '{"component":{"type":"markdown","text":"答案"},'
        '"response_text":"答案",'
        '"content_desc":{"used_tool":["get_order"]}}'
    )
    parsed = parse_output(value)
    assert parsed.component.text == "答案"
    assert parsed.response_text == "答案"
    assert parsed.content_desc.used_tool == ["get_order"]


def test_parse_output_syncs_response_text_from_component() -> None:
    parsed = parse_output({"component": {"text": "仅 component"}})
    assert parsed.component.text == "仅 component"
    assert parsed.component.type == "markdown"
    assert parsed.response_text == "仅 component"


def test_parse_output_syncs_component_from_response_text() -> None:
    parsed = parse_output({"response_text": "仅 response_text"})
    assert parsed.response_text == "仅 response_text"
    assert parsed.component.text == "仅 response_text"
    assert parsed.component.type == "markdown"


# ── case → ${name} bag ──


def test_build_case_variables_flattens_input_fields_and_aliases() -> None:
    bag = build_case_variables(
        {
            "request_body": {
                "query": "附近咖啡",
                "scene_id": "scene-1",
                "session_id": "s-1",
            },
            "viewer_context": {"os": "ios"},
        },
        {
            "component": {"type": "markdown", "text": "推荐 A"},
            "response_text": "推荐 A",
            "content_desc": {"used_tool": ["search"]},
        },
        expected_output="应推荐咖啡馆",
    )
    assert bag["query"] == "附近咖啡"
    assert bag["scene_id"] == "scene-1"
    assert bag["session_id"] == "s-1"
    assert bag["answer"] == "推荐 A"
    assert bag["response_text"] == "推荐 A"
    assert bag["used_tool"] == "search"
    assert bag["expected_output"] == "应推荐咖啡馆"
    assert '"query": "附近咖啡"' in bag["input"] or '"query":"附近咖啡"' in bag["input"]
    assert "viewer_context" in bag  # A field present at the top level is serialized into the bag as a whole


def test_build_case_variables_skips_missing_expected_and_absent_macros() -> None:
    bag = build_case_variables(
        {"request_body": {"query": "q"}},
        {"component": {"text": "a"}, "response_text": "a"},
    )
    assert "expected_output" not in bag
    assert "scene_id" not in bag
    # The prompt has a macro and the bag has no such key → substitute an empty string
    assert render_template("S=${scene_id} Q=${query}", bag) == "S= Q=q"


def test_build_case_variables_prompt_macro_extensible_via_existing_fields() -> None:
    """A new prompt macro whose name matches a field on input → it can be substituted, with no hardcoded change in the agent."""
    bag = build_case_variables(
        {"request_body": {"query": "q", "dialog_type": "navi", "task_id": "t-9"}},
        "纯文本回答",
    )
    rendered = render_template(
        "Q=${query} D=${dialog_type} T=${task_id} A=${answer}",
        bag,
    )
    assert rendered == "Q=q D=navi T=t-9 A=纯文本回答"


def test_build_case_variables_answer_falls_back_to_dict_json() -> None:
    """When output is a dict with no component.text/response_text, ${answer} falls back to the whole package as JSON."""
    bag = build_case_variables(
        {"request_body": {"query": "q"}},
        {"result": {"data": {"text": "自定义正文"}}, "code": 0},
    )
    assert bag["answer"]
    assert "自定义正文" in bag["answer"]
    assert bag["answer"] == bag["output"]


# ── ${name} template rendering ──


def test_fills_braced_variable() -> None:
    assert render_template("Q=${query}", {"query": "你是谁"}) == "Q=你是谁"


def test_fills_bare_variable() -> None:
    assert render_template("Q=$query!", {"query": "你是谁"}) == "Q=你是谁!"


def test_fills_multiple_variables() -> None:
    out = render_template("Q=${query} A=${answer}", {"query": "你是谁", "answer": "我是助手"})
    assert out == "Q=你是谁 A=我是助手"


def test_unknown_variable_renders_empty() -> None:
    assert render_template("X=${missing}", {"query": "y"}) == "X="


def test_double_dollar_is_literal() -> None:
    assert render_template("price $$5 ${query}", {"query": "q"}) == "price $5 q"


def test_json_example_braces_pass_through() -> None:
    # JSON examples in a prompt have no ``$`` and must survive untouched.
    tmpl = 'schema {"score": 0, "reason": "..."}'
    assert render_template(tmpl, {"score": "should-not-be-used"}) == tmpl


def test_empty_variable_renders_empty() -> None:
    assert render_template("v=${query}", {"query": ""}) == "v="
