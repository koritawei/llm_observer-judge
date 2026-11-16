"""Write a ReportPayload into the placeholder tags of the unified HTML template."""

import json
import re
from html import escape as html_escape

from benchmark.reporting.framework.unified_report.payload import ReportPayload

_REPORT_DATA_RE = re.compile(
    r'<script\s+type="application/json"\s+id="report-data"\s*>.*?</script>',
    re.IGNORECASE | re.DOTALL,
)
_CASES_DATA_RE = re.compile(
    r'<script\s+type="application/json"\s+id="cases-data"\s*>.*?</script>',
    re.IGNORECASE | re.DOTALL,
)
_CASE_COLUMNS_RE = re.compile(
    r'<script\s+type="application/json"\s+id="case-columns-data"\s*>.*?</script>',
    re.IGNORECASE | re.DOTALL,
)
_RULE_LINK_SLOT_RE = re.compile(
    r'<span[^>]*class="[^"]*rule-link-slot[^"]*"[^>]*>.*?</span>',
    re.IGNORECASE | re.DOTALL,
)
_BODY_CLOSE_RE = re.compile(r"</body\s*>", re.IGNORECASE)


def _json_script(tag_id: str, payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f'<script type="application/json" id="{tag_id}">{text}</script>'


def _replace_or_inject(html: str, pattern: re.Pattern[str], tag: str) -> str:
    m = pattern.search(html)
    if m:
        return html[: m.start()] + tag + html[m.end() :]
    body = _BODY_CLOSE_RE.search(html)
    if body:
        return html[: body.start()] + tag + html[body.start() :]
    return html + tag


def _fill_rule_link(html: str, rule_link_url: str | None) -> str:
    """Replace the scoring-rule slot at the HTML level (template JS still falls back to REPORT.rule_link_url)."""
    url = (rule_link_url or "").strip()
    if not url or not _RULE_LINK_SLOT_RE.search(html):
        return html
    link = (
        f'<a class="rule-link" href="{html_escape(url, quote=True)}"'
        ' target="_blank" rel="noopener">评分规则 ↗</a>'
    )
    # Use a callable replacement so backslashes in the link are not treated as backrefs/escapes by re.sub
    return _RULE_LINK_SLOT_RE.sub(lambda _m: link, html, count=1)


def fill_report_template(html: str, payload: ReportPayload) -> str:
    """Inject report-data / cases-data / case-columns-data and fill the scoring-rule link."""
    out = _replace_or_inject(
        html,
        _REPORT_DATA_RE,
        _json_script("report-data", payload.report_data_json()),
    )
    out = _replace_or_inject(
        out,
        _CASES_DATA_RE,
        _json_script("cases-data", payload.cases_json()),
    )
    out = _replace_or_inject(
        out,
        _CASE_COLUMNS_RE,
        _json_script("case-columns-data", payload.case_columns_json()),
    )
    out = _fill_rule_link(out, payload.rule_link_url)
    return out


__all__ = ["fill_report_template"]
