"""Independent gate: calibration seal, binomial diagnostics, release edges."""

import math

from benchmark.kernel.contract import content_hash, freeze_contract
from benchmark.kernel.gate import (
    authorize_decision,
    pass_all_k,
    pass_at_k,
    seal_development,
    sign_receipt,
)


def _policy(**extra):
    body = {
        "enabled_authorize_edges": ["manual_authorize"],
        "isolation_requirement": "standard",
        "required_check_ids": ["skill"],
        "security_violations": 0,
        "primary_metric_id": "skill",
        "gate_receipt_trust": {"host_build_hash": "host-1"},
        "evolution_capabilities": ["evolution-kernel"],
    }
    body.update(extra)
    return body


def _check(verdict="pass"):
    return [{"check_id": "skill", "applicability": "required", "verdict": verdict, "value": 1.0}]


def test_deterministic_pass_is_pending_confirmation():
    verdict = seal_development(_policy(), _check())
    assert verdict["stage_outcome"] == "passed"
    assert verdict["release_authorization"] == "pending_confirmation"


def test_deterministic_hard_fail_is_denied():
    verdict = seal_development(_policy(), _check("fail"))
    assert verdict["stage_outcome"] == "failed"
    assert verdict["release_authorization"] == "denied"


def test_missing_llm_calibration_is_inconclusive_denied():
    judges = [{"kind": "llm", "artifact_hash": "sha256:judge", "participates_in": ["primary_metric"]}]
    verdict = seal_development(_policy(), _check(), judges=judges)
    assert verdict == {
        "stage_outcome": "inconclusive",
        "release_authorization": "denied",
        "reasons": ["judge_calibration"],
    }


def test_failed_spot_check_is_not_pending():
    judges = [{"kind": "llm", "artifact_hash": "sha256:judge", "participates_in": ["bad_repair"]}]
    record = {
        "judge_artifact_hash": "sha256:judge",
        "gold_set_hash": "sha256:gold",
        "spot_check_outcome": "fail",
    }
    record["record_hash"] = content_hash(record)
    verdict = seal_development(_policy(), _check(), calibration=record, judges=judges)
    assert verdict["stage_outcome"] == "inconclusive"
    assert verdict["release_authorization"] == "denied"
    assert "pending_confirmation" not in verdict.values()


def test_matching_calibration_can_pend_confirmation():
    judges = [{"kind": "llm", "artifact_hash": "sha256:judge", "metric_id": "skill"}]
    record = {
        "judge_artifact_hash": "sha256:judge",
        "gold_set_hash": "sha256:gold",
        "spot_check_outcome": "pass",
    }
    record["record_hash"] = content_hash(record)
    verdict = seal_development(_policy(), _check(), calibration=record, judges=judges)
    assert verdict["stage_outcome"] == "passed"
    assert verdict["release_authorization"] == "pending_confirmation"


def test_pass_at_k_domain_is_inconclusive():
    assert pass_at_k(3, 4, 1) is None
    assert pass_all_k(3, 1, 0) is None
    assert pass_at_k(4, 4, 2) == 1.0
    assert pass_all_k(4, 1, 2) == 0.0
    assert math.isclose(pass_at_k(2, 1, 1), 0.5)


def test_contract_hash_ignores_its_own_field():
    frozen = freeze_contract({"schema_version": 1, "evaluation_kind": "skill"})
    again = freeze_contract({**frozen, "evaluation_contract_hash": "tampered"})
    assert frozen["evaluation_contract_hash"] == again["evaluation_contract_hash"]


def test_authorize_rejects_auto_edge_on_standard_isolation():
    policy = _policy()
    experiment = {
        "experiment_id": "exp-1",
        "release_authorization": "pending_confirmation",
        "pending_epoch": 0,
        "candidate_hash": "cand",
        "contract": {"evaluation_contract_hash": "contract"},
        "gate_policy": policy,
        "bundle": {"bundle_hash": "bundle"},
    }
    receipt = sign_receipt(
        {
            "candidate_hash": "cand",
            "development_experiment_id": "exp-1",
            "bundle_hash": "bundle",
            "evaluation_contract_hash": "contract",
            "gate_policy_hash": "policy",
            "verdict": "hard_pass",
            "host_build_hash": "host-1",
            "issued_epoch": 0,
        },
        "secret",
    )
    experiment["gate_policy"] = {**policy, "gate_policy_hash": "policy"}
    denied = authorize_decision(
        experiment,
        {
            "expected_pending_epoch": 0,
            "caller_capability": "evolution-kernel",
            "basis": "exhaustive_spec",
            "stage_edge": "exhaustive_authorize",
            "blast_radius_ref": "rollback-1",
            "proof_refs": {"development_gate_receipt": receipt},
        },
        signer_key="secret",
    )
    assert denied["accepted"] is False
    assert denied["reason"] == "edge"
