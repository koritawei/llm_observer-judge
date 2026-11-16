from typing import Any

from langgraph.graph import END, START, StateGraph

from benchmark.libs.agenthost.agents.card import AgentCard
from benchmark.libs.agenthost.agents.request import AgentRequest, AgentResponse, ResponseStatus


class BaseAgent:
    """LangGraph agent. Subclasses add nodes; ``call`` runs the compiled graph."""

    name = "agent"
    description = ""

    def __init__(self, name: str | None = None, description: str | None = None, **_kwargs):
        if name:
            self.name = name
        if description:
            self.description = description
        self.graph = self.build_graph()

    def build_graph(self):
        graph = StateGraph(dict)

        def respond(state: dict) -> dict:
            state["answer"] = state.get("answer") or state.get("user_text") or ""
            return state

        graph.add_node("respond", respond)
        graph.add_edge(START, "respond")
        graph.add_edge("respond", END)
        return graph.compile()

    def card(self) -> AgentCard:
        return AgentCard(name=self.name, description=self.description)

    async def call(self, request: AgentRequest | dict[str, Any], *_args, **_kwargs) -> AgentResponse:
        state = request if isinstance(request, dict) else {"request": request, "input": request.input}
        result = await self.graph.ainvoke(state)
        text = str(result.get("answer") or result.get("output") or "")
        return AgentResponse(status=ResponseStatus.OK, output={"text": text, **result})
