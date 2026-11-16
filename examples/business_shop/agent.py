# -*- coding: utf-8 -*-
"""Store customer-service agent.

One question passes through four stages in order:

1. Intent recognition: first hit against the skill intent table.
2. Tool selection: map the intent to a tool name via the skill route table.
3. Tool execution: run every policy, shipping, price, and knowledge tool in the catalog.
4. Result composition: the script, a code filter, or the selected tool's return value decides the final answer.
"""

from pathlib import Path


def answer(query: str) -> dict[str, object]:
    """Run the four-stage flow and bring back the trace of each stage."""

    skill = _load_skill()
    keyword, intent = _recognize(query, skill["意图"])
    selected = _select_tool(intent, skill["路由"])
    catalog = _catalog(query)
    chosen = str(catalog.get(selected, ""))
    text, layer, action = _compose(query, intent, selected, chosen, skill["结果"])
    return {
        "text": text,
        "intent_keyword": keyword,
        "intent": intent,
        "selected_tool": selected,
        "tool_result": chosen,
        "catalog": catalog,
        "layer": layer,
        "action": action,
    }


def _recognize(query: str, rules: list[tuple[str, str]]) -> tuple[str, str]:
    """First hit on the intent table in file order. When nothing matches, the intent is the unknown label."""

    for keyword, intent in rules:
        if keyword in query:
            return keyword, intent
    return "", "未知"


def _select_tool(intent: str, routes: list[tuple[str, str]]) -> str:
    """The route table maps an intent to a tool name. With no route, no tool is called."""

    for name, tool in routes:
        if name == intent:
            return tool
    return "none"


def _compose(
    query: str,
    intent: str,
    selected: str,
    chosen: str,
    phrases: list[tuple[str, str]],
) -> tuple[str, str, str]:
    """The result layer decides the final text, the layer it came from, and the action name."""

    spoken = dict(phrases)
    if intent == "问候":
        return spoken["问候"], "result", "greet"
    if intent == "人工":
        return spoken["人工"], "intent", "wrong_intent"
    dropped = _apply_code(query, intent, chosen)
    if dropped is not None:
        return dropped, "code", "drop_result"
    if selected in {"", "none"}:
        return "不知道", "tool_select", "no_tool"
    if chosen in {"", "不知道"}:
        return "不知道", "tool_select", "wrong_route"
    return chosen, "tool", "use_result"


def _apply_code(query: str, intent: str, tool_result: str) -> str | None:
    """Result-layer filter. When a knowledge question contains the quantity interrogative, drop the tool result."""

    if intent == "知识" and "多少" in query:
        del tool_result
        return ""
    return None


def _load_skill() -> dict[str, list[tuple[str, str]]]:
    path = Path(__file__).with_name("skill.md")
    sections: dict[str, list[tuple[str, str]]] = {}
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if not current or ":" not in line:
            continue
        key, value = line.split(":", 1)
        sections[current].append((key.strip(), value.strip()))
    missing = [name for name in ("意图", "路由", "结果") if name not in sections]
    if missing:
        raise RuntimeError(f"skill.md missing sections: {missing}")
    return sections


def _catalog(query: str) -> dict[str, str]:
    path = Path(__file__).with_name("tools.py")
    namespace: dict[str, object] = {"__file__": str(path)}
    exec(path.read_text(encoding="utf-8"), namespace)  # noqa: S102
    run_catalog = namespace["run_catalog"]
    if not callable(run_catalog):
        raise RuntimeError("tools.py has no run_catalog")
    catalog = run_catalog(query)
    if not isinstance(catalog, dict):
        raise RuntimeError("run_catalog must return a dict")
    return {str(name): str(value) for name, value in catalog.items()}
