# -*- coding: utf-8 -*-
"""门店客服 agent。

一条问题依次经过四段：

1. 意图识别：按 skill 的意图表做第一次命中。
2. 工具选择：按 skill 的路由表把意图映射到工具名。
3. 工具执行：目录里的政策、运费、价格、知识工具全部运行。
4. 结果合成：话术、代码过滤或被选中工具的返回值决定最终答案。
"""

from pathlib import Path


def answer(query: str) -> dict[str, object]:
    """跑完四段流程并带回每一段的轨迹。"""

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
    """意图表按文件顺序第一次命中。没有命中时意图为「未知」。"""

    for keyword, intent in rules:
        if keyword in query:
            return keyword, intent
    return "", "未知"


def _select_tool(intent: str, routes: list[tuple[str, str]]) -> str:
    """路由表把意图映射成工具名。缺路由时不调用工具。"""

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
    """结果层决定最终文本、所在层和动作名。"""

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
    """结果层过滤。保留工具结果。"""

    del query, intent, tool_result
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
