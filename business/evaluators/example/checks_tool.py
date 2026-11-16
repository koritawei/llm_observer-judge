"""Name-level tool check consumed by ``judge_example``.

Compares tool names only. Does not inspect arguments.
"""

from pydantic import BaseModel, Field


class CheckResult(BaseModel, frozen=True):
    """``ok=True`` and empty ``reason`` means the name check passed."""

    ok: bool
    reason: str


class ToolNameCheck(BaseModel, frozen=True, extra="forbid"):
    """Constraints on ``used_tool`` names.

    An instance whose three fields stay at their defaults is unconstrained.
    """

    required: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    must_be_empty: bool = False


def check_tool_names(
    used_tool: list[str],
    *,
    required: list[str] | None = None,
    forbidden: list[str] | None = None,
    must_be_empty: bool = False,
) -> CheckResult:
    """Check ``used_tool`` names.

    Order: non-empty list when ``must_be_empty``; any missing ``required``
    name; any present ``forbidden`` name. Failure ``reason`` uses the prefix
    emitted by this function and names the missing or present tool. Pass returns
    ``ok=True`` and ``reason=""``.
    """

    names = list(used_tool)
    if must_be_empty and names:
        return CheckResult(ok=False, reason=f"硬检失败：出现 {'、'.join(names)}")

    missing = [name for name in (required or []) if name not in names]
    if missing:
        return CheckResult(ok=False, reason=f"硬检失败：缺少 {'、'.join(missing)}")

    hit = [name for name in (forbidden or []) if name in names]
    if hit:
        return CheckResult(ok=False, reason=f"硬检失败：出现 {'、'.join(hit)}")

    return CheckResult(ok=True, reason="")
