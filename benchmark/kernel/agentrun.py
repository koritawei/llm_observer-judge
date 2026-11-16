"""In-process agent runtime.

Trajectory regeneration runs as a LangGraph inside this process. The graph
sees the user turn and optional tool call. It never receives credentials,
URLs, or the judge's expected answer.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph


class TurnState(TypedDict, total=False):
    user_text: str
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: str
    answer: str


Tool = Callable[[dict[str, Any]], str]


def build_agent_graph(tools: dict[str, Tool] | None = None):
    """Compile the default turn graph: read the turn, maybe call one tool, answer."""

    available = dict(tools or {})

    def read_turn(state: TurnState) -> dict[str, Any]:
        return {"user_text": str(state.get("user_text") or "")}

    def call_tool(state: TurnState) -> dict[str, Any]:
        name = str(state.get("tool_name") or "")
        tool = available.get(name)
        if tool is None:
            return {}
        return {"tool_result": str(tool(dict(state.get("tool_args") or {})))}

    def route(state: TurnState) -> str:
        name = str(state.get("tool_name") or "")
        if name and name in available:
            return "tool"
        return "answer"

    def answer(state: TurnState) -> dict[str, Any]:
        if state.get("tool_result"):
            return {"answer": state["tool_result"]}
        return {"answer": str(state.get("user_text") or "")}

    graph = StateGraph(TurnState)
    graph.add_node("read", read_turn)
    graph.add_node("tool", call_tool)
    graph.add_node("answer", answer)
    graph.add_edge(START, "read")
    graph.add_conditional_edges("read", route, {"tool": "tool", "answer": "answer"})
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def replay_turn(case_input: Any, *, tools: dict[str, Tool] | None = None) -> str:
    """Run one turn and return the agent answer. Expected labels are not an input."""

    text = ""
    tool_name = ""
    tool_args: dict[str, Any] = {}
    if isinstance(case_input, dict):
        text = str(case_input.get("text") or case_input.get("query") or "")
        call = case_input.get("tool_call")
        if isinstance(call, dict):
            tool_name = str(call.get("name") or "")
            args = call.get("arguments")
            tool_args = dict(args) if isinstance(args, dict) else {}
    elif case_input is not None:
        text = str(case_input)
    graph = build_agent_graph(tools)
    final = graph.invoke(
        {"user_text": text, "tool_name": tool_name, "tool_args": tool_args},
    )
    return str(final.get("answer") or "")
