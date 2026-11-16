"""ContextVar 跨 Hook 传递 judge + triage 上下文。

TriageAttributionService.run() 在 before_hook 返回后写入，
LsCaseReviewAfterRunHook / Dispatcher 在同一协程内读取。
"""

import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TriageJudgeContext:
    """run() 作用域内的 judge + triage 上下文快照。"""

    score: str
    reason: str
    triage_level: str
    extra: str


_triage_judge_ctx: ContextVar[TriageJudgeContext | None] = ContextVar(
    "triage_judge_ctx",
    default=None,
)


def set_triage_judge_context(
    *,
    evaluator_result: Any,  # noqa: ANN401 — evaluator 结果为动态载荷（不同 evaluator 类型不同）
    triage_level: str | None,
) -> None:
    score = getattr(evaluator_result, "score", None)
    reason = getattr(evaluator_result, "reason", None)
    extra_raw = getattr(evaluator_result, "extra", None)
    extra = json.dumps(extra_raw, ensure_ascii=False, default=str) if extra_raw else ""
    _triage_judge_ctx.set(
        TriageJudgeContext(
            score=str(score) if score is not None else "",
            reason=str(reason) if reason is not None else "",
            triage_level=triage_level or "",
            extra=extra,
        )
    )


def get_triage_judge_context() -> TriageJudgeContext | None:
    return _triage_judge_ctx.get()


def clear_triage_judge_context() -> None:
    _triage_judge_ctx.set(None)
