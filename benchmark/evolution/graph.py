# -*- coding: utf-8 -*-
"""Self-evolution as a LangGraph. One design step is one node.

The graph sequences the steps. It does not merge code, grant release, or run a
tool-registration change. Release, rollback, archive, and repairer mutation stay
off the automatic path and run only when that step is requested.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from benchmark.evolution.policy import plan_change

EVOLUTION_NODES: tuple[str, ...] = (
    "s01_admit",
    "s02_qualify",
    "s03_freeze",
    "s04_diagnose",
    "s05_plan",
    "s06_lease",
    "s07_propose",
    "s08_seal",
    "s09_skill",
    "s10_tool",
    "s11_fast_check",
    "s12_trial",
    "s13_decide",
    "s14_confirm",
    "s15_archive",
    "s16_repairer",
    "s17_meta",
    "s18_release",
    "s19_rollback",
    "s20_stop",
)

_RESUME = {
    "admitted": "s03_freeze",
    "frozen": "s04_diagnose",
    "diagnosed": "s05_plan",
    "planning": "s06_lease",
    "candidate_sealed": "s12_trial",
    "trial_running": "s13_decide",
    "gate_accepted": "s14_confirm",
}


class EvolutionState(TypedDict, total=False):
    campaign_id: str
    body: dict[str, Any]
    requested_step: str
    status: str
    reason: str
    verdict: str
    stage_outcome: str
    release_authorization: str
    benchmark_run_id: str | None
    actionable: bool
    lease: str
    proposal: str
    skill_lifecycle: str
    tool_lifecycle: str
    fast_check: str
    trial_ready: bool
    qualified: bool
    applied: bool
    meta_validation: str
    replayed: bool
    visited: Annotated[list[str], operator.add]


def build_evolution_graph(service: Any) -> CompiledStateGraph:
    """Compile S01–S20. Automatic advance stops after heldout confirmation."""

    async def s01_admit(state: EvolutionState) -> dict[str, Any]:
        result = await service.admit(dict(state.get("body") or {}))
        return {
            "visited": ["s01_admit"],
            "campaign_id": result.get("campaign_id") or "",
            "status": result["status"],
            "reason": result["reason"],
            "replayed": bool(result.get("replayed")),
        }

    async def s02_qualify(state: EvolutionState) -> dict[str, Any]:
        return {"visited": ["s02_qualify"], "qualified": state.get("status") == "admitted"}

    async def s03_freeze(state: EvolutionState) -> dict[str, Any]:
        result = await service.freeze(state["campaign_id"], _body(state))
        return {"visited": ["s03_freeze"], "status": result["status"]}

    async def s04_diagnose(state: EvolutionState) -> dict[str, Any]:
        result = await service.diagnose_campaign(state["campaign_id"])
        return {
            "visited": ["s04_diagnose"],
            "status": "diagnosed",
            "actionable": bool(result["actionable"]),
            "reason": result["failure_class"],
        }

    async def s05_plan(state: EvolutionState) -> dict[str, Any]:
        result = await service.plan(state["campaign_id"], _body(state))
        return {"visited": ["s05_plan"], "status": result["status"], "reason": result["reason"]}

    async def s06_lease(state: EvolutionState) -> dict[str, Any]:
        if state.get("status") not in {"authorized", "planning"}:
            return {"visited": ["s06_lease"], "lease": "denied", "reason": "plan_not_authorized"}
        return {"visited": ["s06_lease"], "lease": "artifact_store"}

    async def s07_propose(state: EvolutionState) -> dict[str, Any]:
        diff_text = str(_body(state).get("diff_text") or "")
        if not diff_text.strip():
            return {"visited": ["s07_propose"], "proposal": "missing", "status": "blocked", "reason": "empty_diff"}
        return {"visited": ["s07_propose"], "proposal": "caller"}

    async def s08_seal(state: EvolutionState) -> dict[str, Any]:
        result = await service.seal_candidate(state["campaign_id"], _body(state))
        return {
            "visited": ["s08_seal"],
            "status": result.get("status") or "candidate_sealed",
            "reason": result.get("reason"),
        }

    async def s09_skill(state: EvolutionState) -> dict[str, Any]:
        return {"visited": ["s09_skill"], "skill_lifecycle": "recorded_with_candidate"}

    async def s10_tool(state: EvolutionState) -> dict[str, Any]:
        return {"visited": ["s10_tool"], "tool_lifecycle": "deferred"}

    async def s11_fast_check(state: EvolutionState) -> dict[str, Any]:
        body = _body(state)
        actionable = state.get("actionable")
        if actionable is None:
            actionable = state.get("status") in {"authorized", "planning", "candidate_sealed"}
        decision = plan_change(
            {"actionable": bool(actionable)},
            diff_text=str(body.get("diff_text") or ""),
            allowed_prefixes=list(body.get("allowed_prefixes") or []),
        )
        if decision["status"] != "authorized":
            return {
                "visited": ["s11_fast_check"],
                "fast_check": decision["status"],
                "status": decision["status"],
                "reason": decision["reason"],
            }
        return {"visited": ["s11_fast_check"], "fast_check": "authorized"}

    async def s12_trial(state: EvolutionState) -> dict[str, Any]:
        result = await service.start_trial(state["campaign_id"])
        ready = result.get("run_terminal") == "completed"
        return {
            "visited": ["s12_trial"],
            "status": result.get("status") or "trial_running",
            "reason": result.get("reason"),
            "benchmark_run_id": result.get("benchmark_run_id"),
            "trial_ready": ready,
        }

    async def s13_decide(state: EvolutionState) -> dict[str, Any]:
        result = await service.decide(state["campaign_id"], _body(state))
        word = result["verdict"]
        return {
            "visited": ["s13_decide"],
            "verdict": word,
            "status": result.get("status") or ("gate_accepted" if word == "hard_pass" else "gate_denied"),
            "reason": result.get("reason"),
            "benchmark_run_id": result.get("benchmark_run_id") or state.get("benchmark_run_id"),
        }

    async def s14_confirm(state: EvolutionState) -> dict[str, Any]:
        result = await service.confirm(state["campaign_id"])
        outcome = result["stage_outcome"]
        if outcome == "passed":
            status = "release_ready"
        elif outcome == "failed":
            status = "rejected"
        else:
            status = "blocked"
        return {
            "visited": ["s14_confirm"],
            "status": status,
            "reason": result.get("reason"),
            "stage_outcome": outcome,
            "release_authorization": result["release_authorization"],
            "benchmark_run_id": state.get("benchmark_run_id"),
        }

    async def s15_archive(state: EvolutionState) -> dict[str, Any]:
        result = await service.archive(state["campaign_id"])
        return {"visited": ["s15_archive"], "status": result["status"]}

    async def s16_repairer(state: EvolutionState) -> dict[str, Any]:
        result = await service.propose_repairer(state["campaign_id"], _body(state))
        return {"visited": ["s16_repairer"], "applied": bool(result["applied"])}

    async def s17_meta(state: EvolutionState) -> dict[str, Any]:
        return {"visited": ["s17_meta"], "meta_validation": "not_run", "applied": False}

    async def s18_release(state: EvolutionState) -> dict[str, Any]:
        result = await service.request_release(state["campaign_id"], _body(state))
        return {"visited": ["s18_release"], "status": result["status"]}

    async def s19_rollback(state: EvolutionState) -> dict[str, Any]:
        result = await service.rollback(state["campaign_id"])
        return {"visited": ["s19_rollback"], "status": result["status"]}

    async def s20_stop(state: EvolutionState) -> dict[str, Any]:
        result = await service.stop(state["campaign_id"], _body(state))
        return {"visited": ["s20_stop"], "status": result["status"], "reason": result["reason"]}

    graph = StateGraph(EvolutionState)
    nodes = {
        "s01_admit": s01_admit,
        "s02_qualify": s02_qualify,
        "s03_freeze": s03_freeze,
        "s04_diagnose": s04_diagnose,
        "s05_plan": s05_plan,
        "s06_lease": s06_lease,
        "s07_propose": s07_propose,
        "s08_seal": s08_seal,
        "s09_skill": s09_skill,
        "s10_tool": s10_tool,
        "s11_fast_check": s11_fast_check,
        "s12_trial": s12_trial,
        "s13_decide": s13_decide,
        "s14_confirm": s14_confirm,
        "s15_archive": s15_archive,
        "s16_repairer": s16_repairer,
        "s17_meta": s17_meta,
        "s18_release": s18_release,
        "s19_rollback": s19_rollback,
        "s20_stop": s20_stop,
    }
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.add_conditional_edges(START, _entry)
    graph.add_edge("s01_admit", "s02_qualify")
    graph.add_edge("s02_qualify", END)
    graph.add_edge("s03_freeze", "s04_diagnose")
    graph.add_conditional_edges("s04_diagnose", _after_diagnose)
    graph.add_conditional_edges("s05_plan", _after_plan)
    graph.add_conditional_edges("s06_lease", _after_lease)
    graph.add_conditional_edges("s07_propose", _after_propose)
    graph.add_conditional_edges("s08_seal", _after_seal)
    graph.add_edge("s09_skill", "s10_tool")
    graph.add_edge("s10_tool", "s11_fast_check")
    graph.add_conditional_edges("s11_fast_check", _after_fast_check)
    graph.add_conditional_edges("s12_trial", _after_trial)
    graph.add_conditional_edges("s13_decide", _after_decide)
    graph.add_edge("s14_confirm", END)
    graph.add_edge("s15_archive", END)
    graph.add_edge("s16_repairer", "s17_meta")
    graph.add_edge("s17_meta", END)
    graph.add_edge("s18_release", END)
    graph.add_edge("s19_rollback", END)
    graph.add_edge("s20_stop", END)
    return graph.compile()


def advance_result(campaign_id: str, state: EvolutionState) -> dict[str, Any]:
    """Public result of the automatic path. Manual release is not included."""

    status = str(state.get("status") or "")
    result: dict[str, Any] = {"campaign_id": campaign_id, "status": status}
    if state.get("benchmark_run_id"):
        result["benchmark_run_id"] = state.get("benchmark_run_id")
    if status == "diagnosed" or status in {"waiting_approval", "blocked", "trial_running"}:
        result["reason"] = state.get("reason")
    if status == "blocked":
        result["stage_outcome"] = state.get("stage_outcome")
        result["release_authorization"] = state.get("release_authorization")
    elif status == "gate_denied":
        result["verdict"] = state.get("verdict")
    elif status in {"release_ready", "rejected"}:
        result["stage_outcome"] = state.get("stage_outcome")
        result["release_authorization"] = state.get("release_authorization")
    return result


def _entry(state: EvolutionState) -> str:
    requested = state.get("requested_step") or "advance"
    manual = {
        "admit": "s01_admit",
        "archive": "s15_archive",
        "repairer": "s16_repairer",
        "release": "s18_release",
        "rollback": "s19_rollback",
        "stop": "s20_stop",
    }
    if requested in manual:
        return manual[requested]
    if requested != "advance":
        return END
    return _RESUME.get(str(state.get("status") or ""), END)


def _after_diagnose(state: EvolutionState) -> str:
    return "s05_plan" if state.get("actionable") else END


def _after_plan(state: EvolutionState) -> str:
    return "s06_lease" if state.get("status") == "authorized" else END


def _after_lease(state: EvolutionState) -> str:
    return "s07_propose" if state.get("lease") == "artifact_store" else END


def _after_propose(state: EvolutionState) -> str:
    return "s08_seal" if state.get("proposal") == "caller" else END


def _after_fast_check(state: EvolutionState) -> str:
    return "s12_trial" if state.get("fast_check") == "authorized" else END


def _after_seal(state: EvolutionState) -> str:
    return "s09_skill" if state.get("status") == "candidate_sealed" else END


def _after_trial(state: EvolutionState) -> str:
    return "s13_decide" if state.get("trial_ready") else END


def _after_decide(state: EvolutionState) -> str:
    return "s14_confirm" if state.get("verdict") == "hard_pass" else END


def _body(state: EvolutionState) -> dict[str, Any]:
    body = state.get("body") or {}
    return body if isinstance(body, dict) else {}
