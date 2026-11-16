"""Content hashes for an evaluation contract. The hash never includes itself."""

from __future__ import annotations

from typing import Any

from benchmark.kernel.confloader import canonical, sha256_text

_SELF_KEYS = {
    "evaluation_contract_hash",
    "check_plan_hash",
    "gate_policy_hash",
    "bundle_hash",
    "record_hash",
    "signature",
    "status",
    "stage_outcome",
    "release_authorization",
    "pending_epoch",
}


def content_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key not in _SELF_KEYS}
    return sha256_text(canonical(body))


def freeze_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if int(contract.get("schema_version") or 0) != 1:
        raise ValueError("schema_version must be 1")
    frozen = dict(contract)
    frozen["evaluation_contract_hash"] = content_hash(frozen)
    return frozen


def freeze_policy(policy: dict[str, Any]) -> dict[str, Any]:
    frozen = dict(policy)
    frozen["gate_policy_hash"] = content_hash(frozen)
    return frozen
