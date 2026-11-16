"""Shared case-display parsers (reused by plugins, not exposed to the WebUI)."""

import json
import math
from collections.abc import Mapping

from benchmark.runs.case_display.views import (
    CaseAttributionView,
    CaseJudgeDimension,
    CaseJudgeView,
)
from benchmark.runs.schema import FUSED_CRITERION_ID


def as_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, dict) and value:
        return value
    return None


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def as_finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def as_score(value: object) -> float | str | None:
    number = as_finite(value)
    if number is not None:
        return number
    text = as_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def first_present_score(*values: object) -> float | str | None:
    """Take the first parseable score in order; keep a legitimate ``0``, and do not fall back with truthy ``or``."""
    for value in values:
        score = as_score(value)
        if score is not None:
            return score
    return None


def numeric_score(value: object) -> float | None:
    """Parse with the same rules as ``as_score``, but dimension scores keep only numbers (including numeric strings)."""
    parsed = as_score(value)
    return parsed if isinstance(parsed, float) else None


def dimension_of(key: object, payload: object) -> CaseJudgeDimension | None:
    name = as_text(key)
    if not name:
        return None
    if isinstance(payload, Mapping):
        score = numeric_score(payload.get("score"))
        reason = as_text(payload.get("reason"))
        if score is None and not reason:
            return None
        return CaseJudgeDimension(id=name, score=score, reason=reason)
    score = numeric_score(payload)
    if score is None:
        return None
    return CaseJudgeDimension(id=name, score=score, reason="")


def dimensions_from_mapping(raw: object) -> list[CaseJudgeDimension]:
    if not isinstance(raw, Mapping):
        return []
    out: list[CaseJudgeDimension] = []
    for key, payload in raw.items():
        item = dimension_of(key, payload)
        if item is not None:
            out.append(item)
    return out


def dimensions_from_extra(extra: Mapping[str, object]) -> list[CaseJudgeDimension]:
    """Treat object keys under ``extra`` that contain ``score`` as dimensions."""
    out: list[CaseJudgeDimension] = []
    for key, payload in extra.items():
        if not isinstance(payload, Mapping) or "score" not in payload:
            continue
        item = dimension_of(key, payload)
        if item is not None:
            out.append(item)
    return out


def raw_judge_blob(case: Mapping[str, object]) -> dict[str, object] | None:
    """Prefer the top-level judge, then ``scores.fused.judge``, then any scores.*.judge."""
    top = as_mapping(case.get("judge"))
    if top is not None:
        return top
    scores = case.get("scores")
    if not isinstance(scores, Mapping):
        return None
    fused = scores.get(FUSED_CRITERION_ID)
    if isinstance(fused, Mapping):
        judge = as_mapping(fused.get("judge"))
        if judge is not None:
            return judge
    for row in scores.values():
        if not isinstance(row, Mapping):
            continue
        judge = as_mapping(row.get("judge"))
        if judge is not None:
            return judge
    return None


def raw_attribution_blob(case: Mapping[str, object]) -> dict[str, object] | None:
    top = as_mapping(case.get("attribution"))
    if top is not None:
        return top
    scores = case.get("scores")
    if not isinstance(scores, Mapping):
        return None
    fused = scores.get(FUSED_CRITERION_ID)
    if isinstance(fused, Mapping):
        attribution = as_mapping(fused.get("attribution"))
        if attribution is not None:
            return attribution
    for row in scores.values():
        if not isinstance(row, Mapping):
            continue
        attribution = as_mapping(row.get("attribution"))
        if attribution is not None:
            return attribution
    return None


def component_display_text(blob: Mapping[str, object]) -> str:
    component = blob.get("component")
    if isinstance(component, Mapping):
        text = as_text(component.get("displayText") or component.get("text"))
        if text:
            return text
    return as_text(blob.get("displayText") or blob.get("text"))


def stringify_nested(value: object) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(value)
    return as_text(value)


def parse_jsonish(raw: object) -> object:
    """Try to parse a string as JSON; on failure return it unchanged."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not (text.startswith("{") or text.startswith("[")):
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text


def batch_output_text(case: Mapping[str, object]) -> str:
    """Shared batch-pipeline output: prefer ``result`` / ``response_text``."""
    raw: object = parse_jsonish(case.get("output"))
    if isinstance(raw, str):
        return raw
    blob = as_mapping(raw)
    if blob is None:
        return as_text(raw)
    for key in ("result", "response_text", "displayText", "text"):
        value = blob.get(key)
        if value is None:
            continue
        rendered = stringify_nested(value)
        if rendered:
            return rendered
    text = component_display_text(blob)
    if text:
        return text
    return as_text(case.get("display_output"))


def empty_judge(view: CaseJudgeView) -> bool:
    return view.score is None and not view.reason and not view.dimensions


def empty_attribution(view: CaseAttributionView) -> bool:
    if view.format == "markdown":
        return not view.markdown
    return not any(
        [
            view.category,
            view.confidence is not None,
            view.evidence,
            view.reasoning,
        ]
    )


def first_raw_bucket(
    judge: Mapping[str, object],
    *,
    require_key: str | None = None,
) -> Mapping[str, object] | None:
    """Take the first Mapping bucket under ``raw_json``; optionally require a key inside that bucket."""
    raw = judge.get("raw_json")
    if not isinstance(raw, Mapping):
        return None
    for _key, payload in raw.items():
        if not isinstance(payload, Mapping):
            continue
        if require_key is not None and require_key not in payload:
            continue
        return payload
    return None


def judge_from_extra_bucket(judge: Mapping[str, object]) -> CaseJudgeView:
    """``raw_json.<stem>.extra`` dimensions plus the stem score/reason."""
    bucket = first_raw_bucket(judge, require_key="extra")
    if bucket is None:
        return CaseJudgeView(
            score=first_present_score(judge.get("fused_score"), judge.get("score")),
            reason="",
            dimensions=dimensions_from_mapping(judge.get("criterion_scores")),
        )
    extra = bucket.get("extra")
    dimensions = dimensions_from_extra(extra) if isinstance(extra, Mapping) else []
    return CaseJudgeView(
        score=first_present_score(bucket.get("score"), judge.get("fused_score")),
        reason=as_text(bucket.get("reason")),
        dimensions=dimensions,
    )


def judge_from_sub_scores(judge: Mapping[str, object]) -> CaseJudgeView:
    """``raw_json.<stem>.sub_scores`` or ``criterion_scores.*.sub_scores``."""
    bucket = first_raw_bucket(judge, require_key="sub_scores")
    if bucket is not None:
        sub = bucket.get("sub_scores")
        return CaseJudgeView(
            score=first_present_score(bucket.get("score"), judge.get("fused_score")),
            reason=as_text(bucket.get("reason")),
            dimensions=dimensions_from_mapping(sub),
        )
    cs = judge.get("criterion_scores")
    if isinstance(cs, Mapping):
        for _key, payload in cs.items():
            if not isinstance(payload, Mapping):
                continue
            sub = payload.get("sub_scores")
            if isinstance(sub, Mapping):
                return CaseJudgeView(
                    score=first_present_score(payload.get("score"), judge.get("fused_score")),
                    reason=as_text(payload.get("reason")),
                    dimensions=dimensions_from_mapping(sub),
                )
    return CaseJudgeView(
        score=first_present_score(judge.get("fused_score"), judge.get("score")),
        reason=as_text(judge.get("reason")),
        dimensions=[],
    )


def judge_from_criterion_scores(judge: Mapping[str, object]) -> CaseJudgeView:
    raw = judge.get("raw_json")
    raw_map = raw if isinstance(raw, Mapping) else {}
    dims_src = raw_map.get("criterion_scores")
    if not isinstance(dims_src, Mapping):
        dims_src = judge.get("criterion_scores")
    return CaseJudgeView(
        score=first_present_score(judge.get("score"), judge.get("fused_score")),
        reason=as_text(judge.get("reason")),
        dimensions=dimensions_from_mapping(dims_src),
    )


def judge_from_extra_top(judge: Mapping[str, object]) -> CaseJudgeView:
    """Streaming: top-level score/reason plus extra dimensions."""
    extra = judge.get("extra")
    dimensions = dimensions_from_extra(extra) if isinstance(extra, Mapping) else []
    return CaseJudgeView(
        score=as_score(judge.get("score")),
        reason=as_text(judge.get("reason")),
        dimensions=dimensions,
    )


def attribution_table(attr: Mapping[str, object]) -> CaseAttributionView:
    evidence = attr.get("evidence_raw")
    if evidence is None:
        evidence = attr.get("evidence")
    if isinstance(evidence, list):
        evidence_text = "\n".join(as_text(item) for item in evidence if as_text(item))
    else:
        evidence_text = as_text(evidence) or None
    reasoning = as_text(
        attr.get("reasoning_raw") or attr.get("reasoning") or attr.get("root_cause")
    )
    category = as_text(
        attr.get("category") or attr.get("primary_issue_category") or attr.get("owner_domain")
    )
    return CaseAttributionView(
        format="table",
        category=category or None,
        confidence=as_finite(attr.get("confidence")),
        evidence=evidence_text,
        reasoning=reasoning or None,
    )


def attribution_markdown(attr: Mapping[str, object]) -> CaseAttributionView:
    markdown = as_text(attr.get("attribution_md"))
    if not markdown:
        raw = attr.get("raw_json")
        if isinstance(raw, Mapping):
            markdown = as_text(raw.get("attribution_md"))
            if not markdown:
                meta = raw.get("metadata")
                if isinstance(meta, Mapping):
                    nested = meta.get("attribution")
                    if isinstance(nested, Mapping):
                        markdown = as_text(nested.get("attribution_md"))
    if markdown:
        return CaseAttributionView(format="markdown", markdown=markdown)
    return attribution_table(attr)


def batch_attribution_view(case: Mapping[str, object]) -> CaseAttributionView | None:
    """Shared batch-pipeline attribution: prefer markdown, otherwise None."""
    blob = raw_attribution_blob(case)
    if blob is None:
        return None
    view = attribution_markdown(blob)
    return None if empty_attribution(view) else view


def raw_json_has_key_prefix(judge: Mapping[str, object], needle: str) -> bool:
    raw = judge.get("raw_json")
    if not isinstance(raw, Mapping):
        return False
    return any(needle in str(key) for key in raw)


__all__ = [
    "as_finite",
    "as_mapping",
    "as_score",
    "as_text",
    "attribution_markdown",
    "attribution_table",
    "batch_attribution_view",
    "batch_output_text",
    "component_display_text",
    "dimension_of",
    "dimensions_from_extra",
    "dimensions_from_mapping",
    "empty_attribution",
    "empty_judge",
    "first_present_score",
    "first_raw_bucket",
    "numeric_score",
    "judge_from_criterion_scores",
    "judge_from_extra_bucket",
    "judge_from_extra_top",
    "judge_from_sub_scores",
    "parse_jsonish",
    "raw_attribution_blob",
    "raw_judge_blob",
    "raw_json_has_key_prefix",
    "stringify_nested",
]
