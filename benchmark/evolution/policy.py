# -*- coding: utf-8 -*-
"""Admission, contract freeze, and gate words for an evolution campaign.

The repair plugin never sees heldout labels. A candidate hash is not a release.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal
from typing import Any

from benchmark.evolution.canonical import evolution_hash
from benchmark.kernel.gate import seal_development

_RISK_MARKERS = ("auth", "payment", "signer", "secret", "credential", "gate")
_INFRA_MARKERS = ("timeout", "unavailable", "lease_lost", "enqueue_failed")


def classify_admission(
    run: dict[str, Any],
    items: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any]:
    """Return admitted, diagnosis_only, rejected, or quarantined. Never mutates the run."""

    if str(event.get("policy_id") or "") == "example.not_activatable":
        return _admission("rejected", "policy_not_activatable")
    if event.get("event_type") == "parent":
        words = (event.get("run_terminal"), event.get("stage_outcome"), event.get("release_authorization"))
        if any(not word for word in words) or "decision" in event:
            return _admission("quarantined", "parent_event_missing_words")
        return _admission("rejected", "parent_event_does_not_open_campaign")
    snapshot = run.get("config_json") or {}
    origin = list(event.get("origin_chain") or snapshot.get("origin_chain") or [])
    if snapshot.get("source") == "evolution_trial" or "evolution_trial" in origin:
        return _admission("rejected", "recursive_trial")
    status = str(run.get("status") or "")
    if status != "completed":
        if status in {"stopped", "failed"}:
            return _admission("diagnosis_only", f"source_{status}")
        return _admission("rejected", "source_not_terminal")
    if any(item.get("status") not in {"success", "failed"} for item in items):
        return _admission("diagnosis_only", "scores_incomplete")
    bads = [item for item in items if _is_bad(item)]
    if not bads:
        return _admission("rejected", "no_actionable_bad")
    if any(_is_infra(item) for item in bads):
        return _admission("diagnosis_only", "infrastructure_failure")
    return _admission("admitted", "actionable_bad", bad_ids=[str(item["item_id"]) for item in bads])


def freeze_manifest(
    campaign_id: str,
    run: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    gate_policy: dict[str, Any],
    heldout_ids: list[str],
    subject_hash: str,
    repairer_hash: str,
    evaluator_hash: str,
) -> dict[str, Any]:
    """Public contract plus separately stored heldout rows. Labels stay out of the public hash body."""

    _require_gate_alignment(gate_policy)
    heldout = set(heldout_ids)
    visible = []
    hidden = []
    for item in items:
        item_id = str(item["item_id"])
        if item_id in heldout:
            hidden.append(item)
            continue
        visible.append({key: value for key, value in item.items() if key != "expected"})
    bad_ids = _unique(str(item["item_id"]) for item in visible if _is_bad(item))
    good_ids = _unique(str(item["item_id"]) for item in visible if not _is_bad(item))
    public = {
        "campaign_id": campaign_id,
        "source_run_id": run["run_id"],
        "bad_target_ids": bad_ids,
        "good_required_ids": good_ids,
        "heldout_manifest_hash": "",
        "subject_hash": subject_hash,
        "repairer_hash": repairer_hash,
        "evaluator_hash": evaluator_hash,
        "isolation_requirement": gate_policy.get("isolation_requirement"),
        "enabled_authorize_edges": list(gate_policy.get("enabled_authorize_edges") or []),
        "risk_acceptance_ref": gate_policy.get("risk_acceptance_ref"),
        "gate_policy_hash": gate_policy.get("gate_policy_hash"),
        "visible_items": visible,
    }
    snapshot = run.get("config_json") or {}
    if snapshot.get("profile_id"):
        public["profile_id"] = snapshot["profile_id"]
    if snapshot.get("profile_hash"):
        public["profile_hash"] = snapshot["profile_hash"]
    domain = snapshot.get("eval_domain")
    if isinstance(domain, dict):
        domain_id = domain.get("domain_id") or domain.get("id")
        if domain_id:
            public["eval_domain_id"] = domain_id
    public["heldout_manifest_hash"] = evolution_hash("heldout_manifest", {"ids": sorted(heldout)})
    public["contract_hash"] = evolution_hash("contract", public)
    return {"public": public, "heldout": hidden}


def diagnose(items: list[dict[str, Any]], bad_ids: list[str]) -> dict[str, Any]:
    """Cluster failed items. Infrastructure failures are not code repairs."""

    clusters: dict[str, list[str]] = {}
    for item in items:
        if str(item.get("item_id")) not in set(bad_ids):
            continue
        kind = "infrastructure" if _is_infra(item) else "product_defect"
        clusters.setdefault(kind, []).append(str(item["item_id"]))
    actionable = bool(clusters.get("product_defect")) and not clusters.get("infrastructure")
    return {
        "failure_class": "product_defect" if actionable else "not_actionable",
        "clusters": clusters,
        "actionable": actionable,
    }


def plan_change(
    diagnosis: dict[str, Any],
    *,
    diff_text: str,
    allowed_prefixes: list[str],
) -> dict[str, Any]:
    """A plan inside scope can be sealed. Risk-floor paths wait for a person."""

    if not diagnosis.get("actionable"):
        return {"status": "blocked", "reason": "diagnosis_not_actionable"}
    paths = _diff_paths(diff_text)
    if not paths:
        return {"status": "blocked", "reason": "empty_diff"}
    if any(not _allowed(path, allowed_prefixes) for path in paths):
        return {"status": "waiting_approval", "reason": "path_outside_scope", "paths": paths}
    if any(_risk_path(path) for path in paths):
        return {"status": "waiting_approval", "reason": "risk_floor", "paths": paths}
    return {"status": "authorized", "reason": "in_scope", "paths": paths}


def materialize_outputs(
    cases: list[dict[str, Any]],
    proposed_outputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build trial inputs from fresh outputs. A missing output is not the old one."""

    produced = []
    for case in cases:
        item_id = str(case.get("item_id"))
        if item_id not in proposed_outputs:
            raise ValueError(f"missing_fresh_output:{item_id}")
        produced.append({
            "item_id": item_id,
            "input": case.get("input"),
            "output": proposed_outputs[item_id],
        })
    return produced


def scan_candidate(
    diff_text: str,
    proposed_outputs: dict[str, Any],
    heldout_rows: list[dict[str, Any]],
) -> list[str]:
    """Builtin scanners. A hit blocks sealing. Timeouts are not represented here."""

    findings: list[str] = []
    if _SECRET_LINE.search(diff_text):
        findings.append("scanner.secret")
    if any(unicodedata.category(char) == "Cf" for char in diff_text):
        findings.append("format_character")
    if re.search(r"^-.*\bdef test_", diff_text, re.MULTILINE):
        findings.append("scanner.test_deletion")
    if re.search(r"^\+.*\b(egress|side_effect_class)\b", diff_text, re.MULTILINE):
        findings.append("scanner.permission_diff")
    for row in heldout_rows:
        expected = row.get("expected")
        item_id = str(row.get("item_id"))
        claimed = proposed_outputs.get(item_id)
        if claimed is not None and claimed == expected:
            findings.append("scanner.answer_leakage")
        text = expected.get("text") if isinstance(expected, dict) else None
        if isinstance(text, str) and len(text) >= 32 and text in diff_text:
            findings.append("scanner.answer_leakage")
    return _unique(findings)


def trial_checks(
    contract: dict[str, Any],
    policy: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Checks the kernel can read from a finished trial. Caller checks are not an input."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get("item_id")), []).append(item)
    ordered: list[str] = []
    pools = [
        contract.get("bad_target_ids") or [],
        contract.get("good_required_ids") or [],
        policy.get("required_check_ids") or [],
        policy.get("original_good_ids") or [],
        policy.get("target_bad_ids") or [],
    ]
    for pool in pools:
        for item_id in pool:
            name = str(item_id)
            if name not in ordered:
                ordered.append(name)
    checks = []
    for item_id in ordered:
        rows = grouped.get(item_id) or []
        if not rows:
            checks.append({
                "check_id": item_id,
                "applicability": "required",
                "verdict": "unknown",
                "value": None,
            })
            continue
        passed = all(_score_passed(row) for row in rows)
        checks.append({
            "check_id": item_id,
            "applicability": "required",
            "verdict": "pass" if passed else "fail",
            "value": rows[0].get("score"),
        })
    return checks


def gate_policy_for_trial(policy: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Frozen good and bad cases stay mandatory even if the caller omits them."""

    goods = [str(item_id) for item_id in contract.get("good_required_ids") or []]
    bads = [str(item_id) for item_id in contract.get("bad_target_ids") or []]
    effective = dict(policy)
    effective["original_good_ids"] = _unique([*(policy.get("original_good_ids") or []), *goods])
    effective["target_bad_ids"] = _unique([*(policy.get("target_bad_ids") or []), *bads])
    effective["required_check_ids"] = _unique([*(policy.get("required_check_ids") or []), *bads, *goods])
    return effective


def decide_evidence(
    policy: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    calibration: dict[str, Any] | None = None,
    judges: list[dict[str, Any]] | None = None,
) -> str:
    """Return hard_pass, hard_fail, or inconclusive. This function does not write release words."""

    verdict = seal_development(policy, checks, calibration=calibration, judges=judges)
    if verdict["stage_outcome"] == "inconclusive":
        return "inconclusive"
    if verdict["stage_outcome"] == "passed" and verdict["release_authorization"] == "pending_confirmation":
        return "hard_pass"
    return "hard_fail"


def confirm_heldout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score heldout inside the kernel and return only the aggregate."""

    if not rows:
        return {"stage_outcome": "inconclusive", "release_authorization": "denied", "checked": 0}
    passed = 0
    for row in rows:
        output = row.get("output")
        expected = row.get("expected")
        if isinstance(output, dict) and isinstance(expected, dict) and "text" in expected:
            ok = output.get("text") == expected.get("text")
        else:
            ok = output == expected
        passed += int(ok)
    if passed == len(rows):
        return {
            "stage_outcome": "passed",
            "release_authorization": "pending_confirmation",
            "checked": len(rows),
        }
    return {"stage_outcome": "failed", "release_authorization": "denied", "checked": len(rows)}


def _unique(values: Any) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _admission(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, **extra}


def _is_bad(item: dict[str, Any]) -> bool:
    return item.get("status") == "failed" or item.get("score") == 0


def _is_infra(item: dict[str, Any]) -> bool:
    reason = str(item.get("reason") or item.get("error_message") or "").lower()
    return any(marker in reason for marker in _INFRA_MARKERS)


def _diff_paths(diff_text: str) -> list[str]:
    paths = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            paths.append(line[6:].strip())
    return paths


def _allowed(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in prefixes)


def _risk_path(path: str) -> bool:
    lowered = path.lower()
    return any(marker in lowered for marker in _RISK_MARKERS)


def _score_passed(item: dict[str, Any]) -> bool:
    score = item.get("score")
    if isinstance(score, Decimal):
        score = float(score)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    return float(score) >= 1


_SECRET_LINE = re.compile(
    r"(?i)(api[_-]?key|private[_-]?key|secret[_-]?token|access[_-]?key)\s*[:=]\s*\S+"
)


def _require_gate_alignment(policy: dict[str, Any]) -> None:
    edges = set(policy.get("enabled_authorize_edges") or [])
    if policy.get("isolation_requirement") == "standard" and edges != {"manual_authorize"}:
        raise ValueError("standard isolation requires manual authorization")
    if policy.get("risk_acceptance_ref") and edges != {"manual_authorize"}:
        raise ValueError("risk acceptance requires manual authorization")
    if not policy.get("gate_policy_hash"):
        raise ValueError("gate policy hash is required")
