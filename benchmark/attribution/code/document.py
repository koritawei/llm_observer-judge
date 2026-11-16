"""Write a code-attribution result into the attribution document. The analysis stage only receives an already built TaskPack."""

import json
from dataclasses import dataclass
from typing import Any

from benchmark.attribution.code.models import TaskPack, label_brief
from benchmark.attribution.code.trace_summary import build_trace_summary

_QUERY_LIMIT = 2000
_RESPONSE_LIMIT = 8000
_LAYER = {"tool": "tool", "plan": "planning", "gen": "generation", "gate": "none"}
_EVIDENCE_KEYS = (
    ("tool_calls_json", "full_tool_calls_json"),
    ("llm_calls_json", "full_llm_calls_json"),
    ("last_generation_input", "full_last_generation_input"),
)


def symptom_source(document: dict[str, Any]) -> dict[str, Any]:
    """The symptom field prefers the callback's original text. On persist, root_cause is overwritten by chain text."""
    raw = document.get("raw_json")
    if isinstance(raw, dict):
        return raw
    return document


def evidence_text(sources: list[dict[str, Any]], canonical: str, alias: str) -> str:
    """Take evidence in order. The field may sit at the document top level, beside evidence, or in metadata.evidence."""
    for source in sources:
        metadata = source.get("metadata")
        nested = metadata.get("evidence") if isinstance(metadata, dict) else None
        for blob in (source, source.get("evidence"), nested):
            if not isinstance(blob, dict):
                continue
            text = _as_payload_text(blob.get(canonical) or blob.get(alias))
            if text.strip():
                return text
    return ""


def merge_evidence(detail: dict[str, Any], *sources: dict[str, Any]) -> dict[str, Any]:
    """Write the three callback bodies into the persisted document and raw_json."""
    found = {
        canonical: evidence_text(list(sources), canonical, alias)
        for canonical, alias in _EVIDENCE_KEYS
    }
    found = {key: value for key, value in found.items() if value}
    detail.update(found)
    raw = detail.get("raw_json")
    if isinstance(raw, dict):
        raw.update(found)
    return detail


@dataclass(frozen=True)
class TaskSource:
    """Case fields used to build a TaskPack. document is passed separately so it is not mixed with the original text in one argument."""

    case_id: str
    query: str
    tool_calls_json: str
    llm_calls_json: str = ""
    last_generation_input: str = ""
    response_text: str = ""


def build_task(document: dict[str, Any], source: TaskSource) -> TaskPack:
    """Build a TaskPack from the attribution document, callback evidence, and the local reply."""
    symptom = symptom_source(document)
    sources = [symptom, document]
    trace = build_trace_summary(
        tool_calls_json=source.tool_calls_json or evidence_text(sources, *_EVIDENCE_KEYS[0]),
        llm_calls_json=source.llm_calls_json or evidence_text(sources, *_EVIDENCE_KEYS[1]),
        last_generation_input=(
            source.last_generation_input or evidence_text(sources, *_EVIDENCE_KEYS[2])
        ),
        response_text=_clip(source.response_text, _RESPONSE_LIMIT),
    )
    return TaskPack(
        case_id=source.case_id,
        query=_clip(source.query, _QUERY_LIMIT),
        symptom_layer=str(symptom.get("primary_issue_layer") or ""),
        symptom_category=str(symptom.get("primary_issue_category") or ""),
        symptom_root_cause=str(symptom.get("root_cause") or ""),
        symptom_chain=str(symptom.get("root_cause_chain") or ""),
        symptom_suggestion=str(symptom.get("improvement_suggestion") or ""),
        trace_summary=trace,
    )


def _as_payload_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def apply_code_attribution(attribution: dict[str, Any], result: dict[str, Any]) -> bool:
    """On success, rewrite attribution_md and replace primary_issue_category with the label's short display name."""
    if result.get("status") != "ok":
        return False
    label_id = str(result.get("label_id") or "").strip()
    prefix = label_id.split(".", 1)[0]
    layer = _LAYER.get(prefix)
    root = str(result.get("module_root_cause") or "").strip()
    chain = str(result.get("evidence_chain") or "").strip()
    plan = str(result.get("fix_plan") or "").strip()
    brief = label_brief(label_id)
    if layer is None or not brief or not root or not chain or not plan or len(root) >= len(chain):
        return False

    view = dict(attribution)
    view["primary_issue_layer"] = layer
    view["primary_issue_category"] = label_id
    view["root_cause"] = root
    view["root_cause_chain"] = chain
    view["improvement_suggestion"] = plan
    attribution["attribution_md"] = build_attribution_md(view)
    attribution["primary_issue_category"] = brief
    raw = attribution.get("raw_json")
    if isinstance(raw, dict):
        raw["attribution_md"] = attribution["attribution_md"]
        raw["primary_issue_category"] = brief
    return True


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _as_markdown_text(value: object) -> str:
    """Field text as stored. Only turn a literal backslash-n into a newline."""
    if not isinstance(value, str):
        return ""
    return value.replace("\\n", "\n").strip()


def _detail_as_markdown_list(detail_text: str) -> str:
    """Turn the problem description into a Markdown list, one line per item. A line that already starts with "- " is not wrapped again."""
    lines: list[str] = []
    for raw_line in detail_text.splitlines():
        item = raw_line.strip()
        if not item:
            continue
        if item.startswith("- "):
            lines.append(item)
        elif item.startswith("-"):
            lines.append("- " + item[1:].strip())
        else:
            lines.append(f"- {item}")
    return "\n".join(lines)


def build_attribution_md(attribution: dict[str, Any]) -> str:
    """Assemble Markdown from the field text. Blank line between sections."""
    detail_text = _as_markdown_text(attribution.get("primary_issue_detail"))
    problem = _detail_as_markdown_list(detail_text)
    root_text = _as_markdown_text(attribution.get("root_cause"))
    chain_text = _as_markdown_text(attribution.get("root_cause_chain"))
    suggestion_text = _as_markdown_text(attribution.get("improvement_suggestion"))
    dims = attribution.get("affected_dims")
    if isinstance(dims, list):
        dims_text = "/".join(str(item) for item in dims if str(item).strip())
    else:
        dims_text = ""
    layer = attribution.get("primary_issue_layer") or ""
    category = attribution.get("primary_issue_category") or ""
    severity = attribution.get("severity") or ""
    priority = attribution.get("improvement_priority") or ""
    tag = (
        f"标签：问题层 {layer} · 类别 {category} · 严重程度 {severity}"
        f" · 影响维度 {dims_text} · 优先级 {priority}"
    )
    return "\n".join(
        [
            "## 问题说明",
            "",
            problem,
            "",
            "## 根因",
            "",
            f"**{root_text}**",
            "",
            chain_text,
            "",
            "## 怎么改",
            "",
            suggestion_text,
            "",
            tag,
        ]
    )
