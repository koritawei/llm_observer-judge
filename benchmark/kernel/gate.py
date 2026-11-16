"""Independent gate. Development seal and release CAS share one verdict."""

from __future__ import annotations

import hashlib
import hmac
import math
from typing import Any

from benchmark.kernel.confloader import canonical
from benchmark.kernel.contract import content_hash

_EDGES = {"heldout_confirm", "exhaustive_authorize", "manual_authorize"}
_BASIS_EDGE = {
    "confirmation_experiment": "heldout_confirm",
    "exhaustive_spec": "exhaustive_authorize",
    "manual_approval": "manual_authorize",
}
_RECEIPT_FIELDS = (
    "candidate_hash",
    "development_experiment_id",
    "bundle_hash",
    "evaluation_contract_hash",
    "gate_policy_hash",
    "verdict",
    "host_build_hash",
    "issued_epoch",
    "judge_calibration_record_hash",
)


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Diagnostic only. Illegal inputs are inconclusive, not zero."""

    if not _trial_domain(n, c, k):
        return None
    if n - c < k:
        return 1.0
    log_num = _log_comb(n - c, k)
    log_den = _log_comb(n, k)
    if log_num is None or log_den is None:
        return None
    return 1.0 - math.exp(log_num - log_den)


def pass_all_k(n: int, c: int, k: int) -> float | None:
    if not _trial_domain(n, c, k):
        return None
    if c < k:
        return 0.0
    log_num = _log_comb(c, k)
    log_den = _log_comb(n, k)
    if log_num is None or log_den is None:
        return None
    return math.exp(log_num - log_den)


def seal_development(
    policy: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    calibration: dict[str, Any] | None = None,
    judges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write one stage outcome and one release word. Missing calibration is not pending."""

    uses_llm = _llm_participates(judges or [], policy)
    if uses_llm and not _calibration_accepted(policy, calibration, judges or []):
        return {
            "stage_outcome": "inconclusive",
            "release_authorization": "denied",
            "reasons": ["judge_calibration"],
        }
    reasons = hard_failures(policy, checks)
    if reasons:
        return {
            "stage_outcome": "failed",
            "release_authorization": "denied",
            "reasons": reasons,
        }
    return {
        "stage_outcome": "passed",
        "release_authorization": "pending_confirmation",
        "reasons": [],
    }


def hard_failures(policy: dict[str, Any], checks: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    edges = set(policy.get("enabled_authorize_edges") or [])
    if not edges or not edges <= _EDGES:
        reasons.append("authorize_edges")
    if policy.get("isolation_requirement") == "standard" and edges != {"manual_authorize"}:
        reasons.append("standard_isolation")
    if policy.get("risk_acceptance_ref") and edges != {"manual_authorize"}:
        reasons.append("risk_acceptance")
    required = set(policy.get("required_check_ids") or [])
    by_id = {str(item.get("check_id")): item for item in checks}
    if required - set(by_id):
        reasons.append("missing_check")
    for check_id in required:
        item = by_id.get(check_id)
        if item is None:
            continue
        applicability = item.get("applicability") or "required"
        verdict = item.get("verdict")
        if applicability == "not_applicable" and item.get("expected_applicability") != "not_applicable":
            reasons.append("unexpected_not_applicable")
        elif applicability == "required" and verdict != "pass":
            reasons.append(f"check:{check_id}")
        if verdict in {"unknown", None}:
            reasons.append(f"unknown:{check_id}")
        if _bad_number(item.get("value")):
            reasons.append(f"measurement_invalid:{check_id}")
    if int(policy.get("security_violations") or 0) != 0:
        reasons.append("security")
    for case_id in policy.get("original_good_ids") or []:
        row = by_id.get(str(case_id))
        if row is None or row.get("verdict") != "pass":
            reasons.append(f"good:{case_id}")
    for case_id in policy.get("target_bad_ids") or []:
        row = by_id.get(str(case_id))
        if row is None or row.get("verdict") != "pass":
            reasons.append(f"bad:{case_id}")
    return reasons


def sign_receipt(fields: dict[str, Any], key: str) -> dict[str, Any]:
    body = {name: fields.get(name) for name in _RECEIPT_FIELDS if name in fields}
    body["signature"] = _mac(body, key)
    return body


def authorize_decision(
    experiment: dict[str, Any],
    request: dict[str, Any],
    *,
    signer_key: str,
) -> dict[str, Any]:
    policy = experiment["gate_policy"]
    if experiment.get("release_authorization") != "pending_confirmation":
        return {"accepted": False, "reason": "not_pending"}
    raw_epoch = request.get("expected_pending_epoch")
    if raw_epoch is None or int(raw_epoch) != int(experiment.get("pending_epoch") or 0):
        return {"accepted": False, "reason": "epoch"}
    capabilities = set(policy.get("evolution_capabilities") or [])
    if request.get("caller_capability") not in capabilities:
        return {"accepted": False, "reason": "caller"}
    edges = set(policy.get("enabled_authorize_edges") or [])
    if policy.get("isolation_requirement") == "standard" or policy.get("risk_acceptance_ref"):
        edges = {"manual_authorize"}
    edge = str(request.get("stage_edge") or "")
    basis = str(request.get("basis") or "")
    if edge not in edges or _BASIS_EDGE.get(basis) != edge:
        return {"accepted": False, "reason": "edge"}
    if not request.get("blast_radius_ref"):
        return {"accepted": False, "reason": "blast_radius"}
    receipt = (request.get("proof_refs") or {}).get("development_gate_receipt")
    if not isinstance(receipt, dict) or receipt.get("verdict") != "hard_pass":
        return {"accepted": False, "reason": "receipt"}
    if not _receipt_matches(experiment, receipt, signer_key):
        return {"accepted": False, "reason": "receipt"}
    return {"accepted": True, "reason": ""}


def _receipt_matches(experiment: dict[str, Any], receipt: dict[str, Any], signer_key: str) -> bool:
    signed = {key: value for key, value in receipt.items() if key != "signature"}
    if not hmac.compare_digest(_mac(signed, signer_key), str(receipt.get("signature") or "")):
        return False
    policy = experiment["gate_policy"]
    contract = experiment["contract"]
    expected = {
        "candidate_hash": experiment.get("candidate_hash"),
        "development_experiment_id": experiment.get("experiment_id"),
        "bundle_hash": (experiment.get("bundle") or {}).get("bundle_hash"),
        "evaluation_contract_hash": contract.get("evaluation_contract_hash"),
        "gate_policy_hash": policy.get("gate_policy_hash"),
        "verdict": "hard_pass",
        "host_build_hash": (policy.get("gate_receipt_trust") or {}).get("host_build_hash"),
        "issued_epoch": experiment.get("pending_epoch"),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            return False
    calibration = (experiment.get("bundle") or {}).get("calibration")
    if calibration:
        if receipt.get("judge_calibration_record_hash") != calibration.get("record_hash"):
            return False
    elif "judge_calibration_record_hash" in receipt:
        return False
    return True


def _mac(body: dict[str, Any], key: str) -> str:
    digest = hmac.new(key.encode("utf-8"), canonical(body).encode("utf-8"), hashlib.sha256).hexdigest()
    return digest


def _llm_participates(judges: list[dict[str, Any]], policy: dict[str, Any]) -> bool:
    primary = policy.get("primary_metric_id")
    for judge in judges:
        kind = str(judge.get("kind") or "")
        if kind not in {"llm", "llm_judge"}:
            continue
        roles = set(judge.get("participates_in") or [])
        if "primary_metric" in roles or "bad_repair" in roles or judge.get("metric_id") == primary:
            return True
    return False


def _calibration_accepted(
    policy: dict[str, Any],
    record: dict[str, Any] | None,
    judges: list[dict[str, Any]],
) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("spot_check_outcome") != "pass":
        return False
    expected = ""
    for judge in judges:
        if str(judge.get("kind") or "") in {"llm", "llm_judge"}:
            expected = str(judge.get("artifact_hash") or "")
            break
    if not expected or record.get("judge_artifact_hash") != expected:
        return False
    if record.get("record_hash") != content_hash(record):
        return False
    if not record.get("gold_set_hash"):
        return False
    del policy
    return True


def _trial_domain(n: int, c: int, k: int) -> bool:
    return isinstance(n, int) and isinstance(c, int) and isinstance(k, int) and 1 <= k <= n and 0 <= c <= n


def _log_comb(n: int, k: int) -> float | None:
    if k < 0 or k > n:
        return None
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _bad_number(value: Any) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))
