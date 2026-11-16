"""Parse a ResultPack from the final aoned text."""

import json
import re

from benchmark.attribution.code.models import LABEL_IDS, ResultPack

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def strip_fence(text: str) -> str:
    raw = text.strip()
    matched = _FENCE.search(raw)
    if matched:
        return matched.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]
    return raw


def parse_result(text: str) -> ResultPack:
    try:
        data = json.loads(strip_fence(text))
    except json.JSONDecodeError as exc:
        return ResultPack(status="failed", error_message=f"json: {exc}")
    if not isinstance(data, dict):
        return ResultPack(status="failed", error_message="json: not object")

    commit_sha = str(data.get("commit_sha") or "").strip()
    label_id = str(data.get("label_id") or "").strip()
    if label_id not in LABEL_IDS:
        return ResultPack(
            status="failed",
            label_id=label_id,
            commit_sha=commit_sha,
            error_message="label_id not in closed set",
        )

    root = str(data.get("module_root_cause") or "").strip()
    chain = str(data.get("evidence_chain") or "").strip()
    plan = str(data.get("fix_plan") or "").strip()
    missing = [
        name
        for name, value in (
            ("module_root_cause", root),
            ("evidence_chain", chain),
            ("fix_plan", plan),
        )
        if not value
    ]
    if missing:
        return ResultPack(
            status="failed",
            label_id=label_id,
            commit_sha=commit_sha,
            error_message="missing " + ",".join(missing),
        )

    tool_name = str(data.get("tool_name") or "").strip()
    if not label_id.startswith("tool."):
        tool_name = ""

    gate = data.get("gate")
    gate_text = gate.strip() if isinstance(gate, str) else ""
    if label_id.startswith("gate."):
        gate_value: str | None = label_id
    elif gate_text in ("gate.eval_misjudge", "gate.non_agent"):
        return ResultPack(
            status="failed",
            label_id=label_id,
            commit_sha=commit_sha,
            error_message="gate set but label_id is not gate",
        )
    else:
        gate_value = None

    return ResultPack(
        status="ok",
        label_id=label_id,
        tool_name=tool_name,
        module_root_cause=root,
        evidence_chain=chain,
        fix_plan=plan,
        commit_sha=commit_sha,
        gate=gate_value,
    )
