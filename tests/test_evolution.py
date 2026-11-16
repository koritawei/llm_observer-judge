# -*- coding: utf-8 -*-
"""Evolution control plane. Candidates stay sealed; release stays manual."""

import pytest
from httpx import ASGITransport, AsyncClient
from psycopg.types.json import Jsonb

from benchmark.evolution.canonical import evolution_hash
from benchmark.evolution.policy import (
    classify_admission,
    confirm_heldout,
    decide_evidence,
    freeze_manifest,
    materialize_outputs,
    plan_change,
    scan_candidate,
)
from benchmark.evolution.service import EvolutionService
from benchmark.kernel.api import create_app
from benchmark.kernel.contract import content_hash, freeze_policy
from benchmark.kernel.errors import Conflict
from tests.conftest import CONF, wait_terminal

LABEL = "heldout-label"
DIFF = "\n".join([
    "diff --git a/business/agents/example/prompt.txt b/business/agents/example/prompt.txt",
    "--- a/business/agents/example/prompt.txt",
    "+++ b/business/agents/example/prompt.txt",
    "@@",
    "+repaired",
])
RISK_DIFF = "\n".join([
    "diff --git a/benchmark/kernel/gate.py b/benchmark/kernel/gate.py",
    "--- a/benchmark/kernel/gate.py",
    "+++ b/benchmark/kernel/gate.py",
    "@@",
    "+changed",
])


def _run(status="completed", **extra):
    body = {"run_id": "src-1", "status": status, "config_json": {}}
    body.update(extra)
    return body


def _item(item_id, status="failed", score=0, reason=None):
    return {
        "item_id": item_id,
        "status": status,
        "score": score,
        "reason": reason,
        "input": {"q": item_id},
        "output": {"text": "old"},
        "expected": {"text": LABEL if item_id == "hold" else "right"},
    }


def _policy():
    return freeze_policy({
        "enabled_authorize_edges": ["manual_authorize"],
        "isolation_requirement": "standard",
        "required_check_ids": ["repair"],
        "security_violations": 0,
        "primary_metric_id": "repair",
    })


def _checks(verdict="pass"):
    return [{"check_id": "repair", "applicability": "required", "verdict": verdict, "value": 1.0}]


def test_only_actionable_completed_run_is_admitted():
    admitted = classify_admission(_run(), [_item("bad"), _item("good", "success", 1)], {})
    assert admitted["status"] == "admitted"
    assert admitted["bad_ids"] == ["bad"]
    assert classify_admission(_run("failed"), [_item("bad")], {})["status"] == "diagnosis_only"
    assert classify_admission(_run("stopped"), [_item("bad")], {})["reason"] == "source_stopped"
    incomplete = _item("bad", status="running", score=None)
    assert classify_admission(_run(), [incomplete], {})["reason"] == "scores_incomplete"
    assert classify_admission(_run(), [_item("good", "success", 1)], {})["reason"] == "no_actionable_bad"
    infra = _item("bad", reason="provider timeout")
    assert classify_admission(_run(), [infra], {})["reason"] == "infrastructure_failure"


def test_parent_and_recursive_events_do_not_open_a_campaign():
    assert classify_admission(_run(), [], {"policy_id": "example.not_activatable"})["status"] == "rejected"
    parent = classify_admission(_run(), [], {"event_type": "parent", "decision": "pass"})
    assert parent == {"status": "quarantined", "reason": "parent_event_missing_words"}
    closed = classify_admission(_run(), [], {
        "event_type": "parent",
        "run_terminal": "completed",
        "stage_outcome": "passed",
        "release_authorization": "pending_confirmation",
    })
    assert closed["reason"] == "parent_event_does_not_open_campaign"
    recursive = classify_admission(
        _run(config_json={"source": "evolution_trial", "origin_chain": ["evolution_trial"]}),
        [_item("bad")],
        {},
    )
    assert recursive["reason"] == "recursive_trial"


def test_contract_hides_heldout_labels_and_hash_is_stable():
    items = [_item("bad"), _item("good", "success", 1), _item("hold", "success", 1)]
    policy = _policy()
    first = freeze_manifest(
        "evo-1",
        _run(),
        items,
        gate_policy=policy,
        heldout_ids=["hold"],
        subject_hash="subject",
        repairer_hash="repairer",
        evaluator_hash="evaluator",
    )
    second = freeze_manifest(
        "evo-1",
        _run(),
        items,
        gate_policy=policy,
        heldout_ids=["hold"],
        subject_hash="subject",
        repairer_hash="repairer",
        evaluator_hash="evaluator",
    )
    assert first["public"]["contract_hash"] == second["public"]["contract_hash"]
    assert LABEL not in str(first["public"])
    assert first["heldout"][0]["expected"]["text"] == LABEL
    assert "expected" not in first["public"]["visible_items"][0]


def test_missing_output_is_not_replaced_with_the_old_one():
    with pytest.raises(ValueError, match="missing_fresh_output:bad"):
        materialize_outputs([_item("bad")], {})


def test_heldout_answer_in_the_proposal_is_a_scanner_hit():
    rows = [{"item_id": "hold", "expected": {"text": LABEL}}]
    assert "scanner.answer_leakage" in scan_candidate(DIFF, {"hold": {"text": LABEL}}, rows)
    leaked = "heldout-answer-text-is-long-enough"
    assert "scanner.answer_leakage" in scan_candidate(f"+{leaked}\n", {}, [{
        "item_id": "hold",
        "expected": {"text": leaked},
    }])
    assert scan_candidate(DIFF, {"bad": {"text": "fixed"}}, rows) == []
    assert "format_character" in scan_candidate("+\u200bhidden\n", {}, [])


def test_evolution_hash_rejects_noncanonical_measurements():
    assert evolution_hash("contract", {"n": 1}) == evolution_hash("contract", {"n": 1})
    assert evolution_hash("contract", {"n": 1}) != evolution_hash("event", {"n": 1})
    with pytest.raises(ValueError, match="measurement_invalid"):
        evolution_hash("contract", {"n": float("nan")})
    with pytest.raises(ValueError, match="non_nfc"):
        evolution_hash("contract", {"name": "e\u0301"})


def test_risk_floor_waits_and_runner_does_not_receive_expected():
    diagnosis = {"actionable": True, "failure_class": "product_defect"}
    allowed = plan_change(diagnosis, diff_text=DIFF, allowed_prefixes=["business/agents/example"])
    assert allowed["status"] == "authorized"
    risk = plan_change(diagnosis, diff_text=RISK_DIFF, allowed_prefixes=["benchmark/kernel"])
    assert risk["status"] == "waiting_approval"
    assert risk["reason"] == "risk_floor"
    produced = materialize_outputs([_item("bad")], {"bad": {"text": "fixed"}})
    assert produced == [{"item_id": "bad", "input": {"q": "bad"}, "output": {"text": "fixed"}}]
    assert "expected" not in produced[0]


def test_missing_calibration_stays_inconclusive():
    judges = [{"kind": "llm", "artifact_hash": "sha256:judge", "participates_in": ["primary_metric"]}]
    assert decide_evidence(_policy(), _checks(), judges=judges) == "inconclusive"
    assert decide_evidence(_policy(), _checks()) == "hard_pass"
    assert decide_evidence(_policy(), _checks("fail")) == "hard_fail"
    aggregate = confirm_heldout([{"item_id": "hold", "output": {"text": LABEL}, "expected": {"text": LABEL}}])
    assert aggregate == {
        "stage_outcome": "passed",
        "release_authorization": "pending_confirmation",
        "checked": 1,
    }
    assert "expected" not in aggregate
    assert content_hash({"ids": ["hold"]})


def test_each_evolution_step_is_a_graph_node():
    from benchmark.evolution.graph import EVOLUTION_NODES, build_evolution_graph

    compiled = build_evolution_graph(_GraphService())
    names = set(compiled.get_graph().nodes)
    assert list(EVOLUTION_NODES) == [
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
    ]
    assert set(EVOLUTION_NODES) <= names


async def test_graph_stops_before_release_and_skips_unactionable_work():
    from benchmark.evolution.graph import advance_result, build_evolution_graph

    paused = _GraphService()
    paused_final = await build_evolution_graph(paused).ainvoke(_advance_state())
    assert paused.calls == ["freeze", "diagnose", "plan", "seal", "trial"]
    assert paused_final["status"] == "trial_running"
    assert "s13_decide" not in paused_final["visited"]
    assert advance_result("evo-1", paused_final)["benchmark_run_id"] == "run-trial"

    ready = _GraphService(terminal=True)
    final = await build_evolution_graph(ready).ainvoke(_advance_state())
    assert ready.calls == ["freeze", "diagnose", "plan", "seal", "trial", "decide", "confirm"]
    assert "release" not in ready.calls
    assert final["tool_lifecycle"] == "deferred"
    assert final["skill_lifecycle"] == "recorded_with_candidate"
    assert final["visited"] == [
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
    ]
    assert advance_result("evo-1", final)["status"] == "release_ready"
    assert advance_result("evo-1", final)["benchmark_run_id"] == "run-trial"

    blocked = _GraphService(actionable=False)
    stopped = await build_evolution_graph(blocked).ainvoke(_advance_state())
    assert blocked.calls == ["freeze", "diagnose"]
    assert advance_result("evo-1", stopped) == {
        "campaign_id": "evo-1",
        "status": "diagnosed",
        "reason": "not_actionable",
    }

    risky = _GraphService(plan_status="waiting_approval")
    held = await build_evolution_graph(risky).ainvoke(_advance_state())
    assert risky.calls == ["freeze", "diagnose", "plan"]
    assert advance_result("evo-1", held)["status"] == "waiting_approval"

    repair = _GraphService()
    mutated = await build_evolution_graph(repair).ainvoke({
        "campaign_id": "evo-1",
        "body": {"diff_text": DIFF},
        "requested_step": "repairer",
        "visited": [],
    })
    assert repair.calls == ["repairer"]
    assert not mutated["applied"]
    assert mutated["meta_validation"] == "not_run"
    assert mutated["visited"] == ["s16_repairer", "s17_meta"]


def _advance_state() -> dict:
    return {
        "campaign_id": "evo-1",
        "body": {
            "diff_text": DIFF,
            "allowed_prefixes": ["business/agents/example"],
            "proposed_outputs": {"bad": {"text": "fixed"}},
            "checks": _checks(),
            "gate_policy": _policy(),
        },
        "status": "admitted",
        "requested_step": "advance",
        "visited": [],
    }


class _GraphService:
    def __init__(self, *, actionable: bool = True, plan_status: str = "authorized", terminal: bool = False):
        self.calls: list[str] = []
        self.actionable = actionable
        self.plan_status = plan_status
        self.terminal = terminal

    async def freeze(self, campaign_id: str, body: dict):
        self.calls.append("freeze")
        return {"status": "frozen"}

    async def diagnose_campaign(self, campaign_id: str):
        self.calls.append("diagnose")
        return {
            "actionable": self.actionable,
            "failure_class": "product_defect" if self.actionable else "not_actionable",
        }

    async def plan(self, campaign_id: str, body: dict):
        self.calls.append("plan")
        reason = "in_scope" if self.plan_status == "authorized" else "risk_floor"
        return {"status": self.plan_status, "reason": reason}

    async def seal_candidate(self, campaign_id: str, body: dict):
        self.calls.append("seal")
        return {"status": "candidate_sealed"}

    async def start_trial(self, campaign_id: str):
        self.calls.append("trial")
        result = {"benchmark_run_id": "run-trial", "status": "trial_running"}
        if self.terminal:
            result["run_terminal"] = "completed"
        return result

    async def decide(self, campaign_id: str, body: dict):
        self.calls.append("decide")
        return {"verdict": "hard_pass"}

    async def confirm(self, campaign_id: str):
        self.calls.append("confirm")
        return {
            "stage_outcome": "passed",
            "release_authorization": "pending_confirmation",
            "reason": None,
        }

    async def propose_repairer(self, campaign_id: str, body: dict):
        self.calls.append("repairer")
        return {"status": "recorded", "applied": False}


def _case_policy():
    return freeze_policy({
        "enabled_authorize_edges": ["manual_authorize"],
        "isolation_requirement": "standard",
        "required_check_ids": ["bad", "good"],
        "original_good_ids": ["good"],
        "target_bad_ids": ["bad"],
        "security_violations": 0,
        "primary_metric_id": "bad",
    })


@pytest.mark.integration
async def test_campaign_stops_at_manual_release(live):
    kernel = live
    await _seed(kernel, "src-1")
    service = EvolutionService(kernel)
    opened = await service.admit({"run_id": "src-1", "report_revision": 1, "tenant_id": "default"})
    assert opened["status"] == "admitted"
    replayed = await service.admit({"run_id": "src-1", "report_revision": 1, "tenant_id": "default"})
    assert replayed["replayed"]
    assert replayed["campaign_id"] == opened["campaign_id"]

    denied = await service.admit({"run_id": "missing", "policy_id": "example.not_activatable"})
    assert denied["status"] == "rejected"
    assert denied["campaign_id"] is None
    parent = await service.admit({
        "run_id": "src-1",
        "event_type": "parent",
        "run_terminal": "completed",
        "stage_outcome": "passed",
        "release_authorization": "pending_confirmation",
    })
    assert parent["campaign_id"] is None

    body = {
        "gate_policy": _case_policy(),
        "heldout_ids": ["hold"],
        "subject_hash": "subject-1",
        "repairer_hash": "repairer-1",
        "evaluator_hash": "evaluator-1",
        "diff_text": DIFF,
        "allowed_prefixes": ["business/agents/example"],
        "proposed_outputs": {"bad": {"text": "right"}, "good": {"text": "right"}},
        "checks": _checks("fail"),
    }
    advanced = await service.advance(opened["campaign_id"], body)
    assert advanced["status"] == "trial_running"
    assert advanced["benchmark_run_id"]

    public = await service.public_contract(opened["campaign_id"])
    assert LABEL not in str(public)
    assert "visible_items" not in public
    async with await kernel.store.connect() as conn:
        cur = await conn.execute(
            "SELECT payload FROM input_manifest WHERE run_id = %s ORDER BY item_id",
            (advanced["benchmark_run_id"],),
        )
        payloads = [row["payload"] for row in await cur.fetchall()]
    assert [item["item_id"] for item in payloads] == ["bad", "good"]
    assert payloads[0]["output"] == {"text": "right"}
    assert payloads[0]["expected"] == {"text": "right"}
    assert LABEL not in str(payloads)

    source = await kernel.store.get_run("src-1")
    assert source["status"] == "completed"
    trial = await service.admit({"run_id": advanced["benchmark_run_id"], "report_revision": 1})
    assert trial["reason"] == "recursive_trial"
    assert trial["campaign_id"] is None

    await wait_terminal(kernel, advanced["benchmark_run_id"])
    decided = await service.advance(opened["campaign_id"], body)
    assert decided["status"] == "blocked"
    assert decided["stage_outcome"] == "inconclusive"
    assert decided["release_authorization"] == "denied"
    assert decided["reason"] == "heldout_not_collected"
    with pytest.raises(Conflict, match="hard pass"):
        await service.confirm(opened["campaign_id"])
    with pytest.raises(Conflict, match="confirmed"):
        await service.request_release(opened["campaign_id"], {
            "experiment_id": "exp-missing",
            "blast_radius_ref": "radius-1",
            "approval_ref": "person-1",
        })
    proposal = await service.propose_repairer(opened["campaign_id"], {"diff_text": DIFF})
    assert not proposal["applied"]
    with pytest.raises(Conflict, match="releasing"):
        await service.rollback(opened["campaign_id"])

    await _seed(kernel, "src-leak")
    leaked = await service.admit({"run_id": "src-leak", "report_revision": 1})
    leak_body = {
        **body,
        "proposed_outputs": {
            "bad": {"text": "right"},
            "good": {"text": "right"},
            "hold": {"text": LABEL},
        },
    }
    blocked = await service.advance(leaked["campaign_id"], leak_body)
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "scanner.answer_leakage"

    await _seed(kernel, "src-miss")
    missed = await service.admit({"run_id": "src-miss", "report_revision": 2})
    miss_body = {
        **body,
        "proposed_outputs": {"bad": {"text": "wrong"}, "good": {"text": "right"}},
        "checks": _checks("pass"),
    }
    started = await service.advance(missed["campaign_id"], miss_body)
    await wait_terminal(kernel, started["benchmark_run_id"])
    failed = await service.advance(missed["campaign_id"], miss_body)
    assert failed["status"] == "gate_denied"
    assert failed["verdict"] == "hard_fail"

    await _seed(kernel, "src-stop", items=[_item("bad")])
    stopped_open = await service.admit({"run_id": "src-stop", "report_revision": 1})
    stopped = await service.stop(stopped_open["campaign_id"], {"expected_epoch": 0, "reason": "operator"})
    assert stopped["status"] == "cancelled"
    with pytest.raises(Conflict, match="terminal"):
        await service.stop(stopped_open["campaign_id"], {"expected_epoch": 1})


@pytest.mark.integration
async def test_risk_floor_is_not_sealed(kernel):
    await _seed(kernel, "src-risk")
    service = EvolutionService(kernel)
    opened = await service.admit({"run_id": "src-risk"})
    body = {
        "gate_policy": _policy(),
        "heldout_ids": ["hold"],
        "diff_text": RISK_DIFF,
        "allowed_prefixes": ["benchmark/kernel"],
        "proposed_outputs": {"bad": {"text": "fixed"}},
    }
    await service.freeze(opened["campaign_id"], body)
    await service.diagnose_campaign(opened["campaign_id"])
    planned = await service.plan(opened["campaign_id"], body)
    assert planned["status"] == "waiting_approval"
    with pytest.raises(Conflict, match="not authorized"):
        await service.seal_candidate(opened["campaign_id"], body)


@pytest.mark.integration
async def test_evolution_event_route(kernel):
    app = create_app(CONF)
    app.state.kernel = kernel
    app.state.owns_kernel = False
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://benchmark") as client:
        response = await client.post("/v1/evolution/events", json={
            "run_id": "unused",
            "policy_id": "example.not_activatable",
        })
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"


async def _seed(kernel, run_id: str, items: list[dict] | None = None) -> None:
    rows = items or [
        _item("bad"),
        _item("good", "success", 1),
        _item("hold", "success", 1),
    ]
    async with await kernel.store.connect() as conn:
        await conn.execute(
            """
            INSERT INTO benchmark_runs (
                run_id, benchmark_name, status, total_samples, total_evaluators,
                completion_model, config_json
            ) VALUES (%s, 'demo', 'completed', %s, 1, 'count', %s)
            """,
            (run_id, len(rows), Jsonb({})),
        )
        for item in rows:
            await conn.execute(
                """
                INSERT INTO benchmark_run_items (
                    run_id, item_id, criterion_id, status, score, reason, output
                ) VALUES (%s, %s, 'fused', %s, %s, %s, %s)
                """,
                (run_id, item["item_id"], item["status"], item["score"], item.get("reason"), Jsonb(item["output"])),
            )
            await conn.execute(
                """
                INSERT INTO input_manifest (run_id, item_id, payload, checksum)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    run_id,
                    item["item_id"],
                    Jsonb({
                        "item_id": item["item_id"],
                        "input": item["input"],
                        "output": item["output"],
                        "expected": item["expected"],
                    }),
                    "checksum",
                ),
            )
        await conn.commit()
