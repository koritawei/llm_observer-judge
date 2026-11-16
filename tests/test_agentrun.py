from benchmark.kernel.agentrun import build_agent_graph, replay_turn


def test_replay_answers_from_the_user_turn():
    answer = replay_turn({"text": "yes", "expected": "secret-label"})
    assert answer == "yes"


def test_replay_runs_one_registered_tool():
    def lookup(arguments: dict) -> str:
        return "found:" + str(arguments.get("q"))

    answer = replay_turn(
        {"text": "search", "tool_call": {"name": "lookup", "arguments": {"q": "tide"}}},
        tools={"lookup": lookup},
    )
    assert answer == "found:tide"


def test_graph_stops_after_the_answer():
    graph = build_agent_graph()
    assert set(graph.get_graph().nodes) >= {"read", "tool", "answer"}
