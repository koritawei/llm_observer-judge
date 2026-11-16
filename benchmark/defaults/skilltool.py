"""Default skill and tool judges. They score a case and do not write the database."""

from __future__ import annotations

from typing import Any

from benchmark.kernel.types import Completed, PhaseContext


def _row(criterion_id: str, score: float, reason: str) -> dict[str, Any]:
    return {
        "criterion_id": criterion_id,
        "score": score,
        "reason": reason,
        "status": "success",
    }


class SkillExec:
    submits = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> Completed:
        expected = (ctx.case.get("expected") or {}).get("skill_id")
        used = (ctx.case.get("output") or {}).get("skills_used") or []
        ok = expected in used
        return Completed(
            {
                "rows": [
                    {
                        **_row("skill", 1.0 if ok else 0.0, "skill_used" if ok else "skill_missing"),
                        "checks": [
                            {
                                "check_id": "skill",
                                "applicability": "required",
                                "verdict": "pass" if ok else "fail",
                                "value": 1.0 if ok else 0.0,
                                "reason_code": "skill_used" if ok else "skill_missing",
                            }
                        ],
                    }
                ]
            }
        )


class ToolExec:
    submits = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> Completed:
        expected = ctx.case.get("expected") or {}
        calls = (ctx.case.get("output") or {}).get("tool_calls") or []
        call = calls[0] if calls else {}
        name_ok = call.get("name") == expected.get("tool_name")
        args_ok = expected.get("arguments") is None or call.get("arguments") == expected.get("arguments")
        ok = bool(calls) and name_ok and args_ok
        return Completed(
            {
                "rows": [
                    {
                        **_row("tool_call", 1.0 if ok else 0.0, "tool_match" if ok else "tool_mismatch"),
                        "checks": [
                            {
                                "check_id": "tool_call",
                                "applicability": "required",
                                "verdict": "pass" if ok else "fail",
                                "value": 1.0 if ok else 0.0,
                                "reason_code": "tool_match" if ok else "tool_mismatch",
                            }
                        ],
                    }
                ]
            }
        )


class ToolContract:
    submits = False

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def execute(self, ctx: PhaseContext) -> Completed:
        expected = ctx.case.get("expected") or {}
        required = list(expected.get("required_fields") or [])
        calls = (ctx.case.get("output") or {}).get("tool_calls") or []
        arguments = (calls[0].get("arguments") if calls else None) or {}
        missing = [field for field in required if field not in arguments]
        ok = bool(calls) and not missing
        reason = "contract_ok" if ok else "missing:" + ",".join(missing)
        return Completed(
            {
                "rows": [
                    {
                        **_row("tool_contract", 1.0 if ok else 0.0, reason),
                        "checks": [
                            {
                                "check_id": "tool_contract",
                                "applicability": "required",
                                "verdict": "pass" if ok else "fail",
                                "value": 1.0 if ok else 0.0,
                                "reason_code": reason,
                            }
                        ],
                    }
                ]
            }
        )
