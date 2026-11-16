"""Unified daily-report ReportPayload: the fill data structure shared by every business.

Field-aligned with the live ``report-data`` / ``cases-data`` so injection and UX patches can be reused.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class CaseColumnSpec(BaseModel):
    """Custom columns for case details (written to ``#case-columns-data``)."""

    model_config = {"extra": "ignore"}

    id: str
    header: str
    #: Dotted path into the case dict, for example ``extra.biz_tag``
    path: str
    placement: Literal["after_standard", "before_trace"] = "after_standard"


class ReportPayload(BaseModel):
    """Shared fill payload (produced by a business plugin, injected into the template by the filler)."""

    model_config = {"extra": "allow"}

    run_id: str = ""
    benchmark_name: str = ""
    service_name: str = ""
    dataset_name: str = ""
    rule_link_url: str | None = None
    final_score: float | None = None
    aggregation_method: str = "weighted_avg"
    total_samples: int = 0
    total_failed: int = 0
    total_success: int = 0
    total_duration_ms: int | None = None
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    criterion_metrics: dict[str, Any] = Field(default_factory=dict)
    attribution_metrics: dict[str, Any] = Field(default_factory=dict)
    react_rounds_summary: dict[str, Any] = Field(default_factory=dict)
    #: Performance and efficiency metrics ``{latency, ttft, tokens, react}`` (a missing segment is absent)
    performance_metrics: dict[str, Any] = Field(default_factory=dict)
    case_results: list[dict[str, Any]] = Field(default_factory=list)
    grade_distribution: dict[str, int] = Field(default_factory=dict)
    #: Yesterday's tier counts (accepts floats after JSON deserialization, such as ``10.0``)
    yesterday_grade_distribution: dict[str, float] = Field(default_factory=dict)
    trend_7d: dict[str, Any] | None = None
    today_criterion_means: dict[str, float] = Field(default_factory=dict)
    yesterday_criterion_means: dict[str, float] = Field(default_factory=dict)
    insight_text: str | None = None
    #: LLM summary for today: ``{conclusion, bullets, action}``
    insight_summary: dict[str, Any] | None = None
    #: LLM deep-dive list of top issues (1 to 3 items)
    top_problems: list[dict[str, Any]] | None = None
    #: Report display date ``YYYY-MM-DD`` (template ``#meta-date``; do not use the browser's local today)
    report_date: str | None = None
    score_loss_waterfall: dict[str, Any] | None = None
    criterion_deltas: list[dict[str, Any]] | None = None
    root_cause_dim_matrix: dict[str, Any] | None = None
    #: Primary field for overall attribution (ieval=primary_issue_category; search=deduction dimension)
    fused_attr_field: str = "primary_issue_category"
    #: scores keys preferred when the detail reads "score reason" (exposed by the plugin; template/JS share the source)
    judge_reason_preferred_keys: list[str] = Field(default_factory=list)
    case_columns: list[CaseColumnSpec] = Field(default_factory=list)

    @field_validator("yesterday_grade_distribution", mode="before")
    @classmethod
    def _coerce_yesterday_grade_distribution(cls, value: object) -> dict[str, float]:
        """Accept float / numeric-string counts; drop invalid items."""
        if not isinstance(value, dict):
            return {}
        out: dict[str, float] = {}
        for key, raw in value.items():
            label = str(key or "").strip()
            if not label or isinstance(raw, bool):
                continue
            if isinstance(raw, (int, float)):
                out[label] = float(int(round(float(raw))))
                continue
            if isinstance(raw, str):
                try:
                    out[label] = float(int(round(float(raw.strip()))))
                except ValueError:
                    continue
        return out

    def to_report_spec(self) -> dict[str, Any]:
        """Emit a dict compatible with the live ``report-data`` (includes case_results)."""
        data = self.model_dump(exclude_none=False)
        if not data.get("report_date"):
            data["report_date"] = derive_report_date(data)
        return data

    def report_data_json(self) -> dict[str, Any]:
        """``report-data`` tag contents (excludes case_results)."""
        data = self.to_report_spec()
        data.pop("case_results", None)
        return data

    def cases_json(self) -> list[dict[str, Any]]:
        return list(self.case_results or [])

    def case_columns_json(self) -> list[dict[str, Any]]:
        return [c.model_dump() for c in self.case_columns]

    @classmethod
    def from_report_spec(
        cls,
        spec: dict[str, Any],
        *,
        fused_attr_field: str = "primary_issue_category",
        case_columns: list[CaseColumnSpec] | None = None,
    ) -> "ReportPayload":
        """Build from the existing assemble_report_spec plus _attach_* results."""
        known = set(cls.model_fields.keys())
        raw = {k: v for k, v in spec.items() if k in known}
        raw["fused_attr_field"] = fused_attr_field
        if case_columns is not None:
            raw["case_columns"] = case_columns
        if not raw.get("report_date"):
            raw["report_date"] = derive_report_date(spec)
        return cls.model_validate(raw)


def derive_report_date(spec: dict[str, Any]) -> str | None:
    """Derive the report date from the spec: explicit field, then yyyymmdd embedded in run_id."""
    for key in ("report_date", "report_day", "date"):
        val = spec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:10]
    run_id = str(spec.get("run_id") or "")
    # ``{name}-20260916`` / ``{name}-20260916-suffix``
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", run_id)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


__all__ = ["CaseColumnSpec", "ReportPayload", "derive_report_date"]
