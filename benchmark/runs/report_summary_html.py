"""Take the summary that precedes the case details from a published report HTML page.

In the report template ``benchmark_report.html``, ``#case-section`` starts the case details;
the all-cases page embeds that summary above the table so the case table below is not rendered twice.
"""

import asyncio
import contextvars
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger(__name__)

#: Do not use asyncio.to_thread / the default executor (DEFAULT-EXECUTOR-001).
_REPORT_FETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="report-summary-fetch",
)
_ALLOWED_REPORT_SCHEMES = frozenset({"http", "https"})

_CASE_SECTION_HIDE_CSS = (
    '<style id="agent-eval-case-summary-hide">'
    "#case-section,.detail-overlay,#detail-modal{display:none!important}"
    "</style>"
)
_CASES_DATA_RE = re.compile(
    r'(<script\s+type="application/json"\s+id="cases-data">)(.*?)(</script>)',
    re.IGNORECASE | re.DOTALL,
)
_CASE_COLUMNS_RE = re.compile(
    r'(<script\s+type="application/json"\s+id="case-columns-data">)(.*?)(</script>)',
    re.IGNORECASE | re.DOTALL,
)
_HEAD_CLOSE_RE = re.compile(r"</head>", re.IGNORECASE)
_FETCH_TIMEOUT_S = 20.0
_MAX_REPORT_BYTES = 8 * 1024 * 1024


def strip_case_details_from_report_html(html: str) -> str:
    """Clear the cases data and hide the case details / detail modal, keeping summary blocks such as KPIs."""
    text = html
    text = _CASES_DATA_RE.sub(r"\1[]\3", text, count=1)
    text = _CASE_COLUMNS_RE.sub(r"\1[]\3", text, count=1)
    if _HEAD_CLOSE_RE.search(text):
        return _HEAD_CLOSE_RE.sub(f"{_CASE_SECTION_HIDE_CSS}</head>", text, count=1)
    return f"{_CASE_SECTION_HIDE_CSS}{text}"


def _assert_http_report_url(report_url: str) -> str:
    """Allow only absolute http(s) URLs; reject file/data schemes and other SSRF surfaces."""
    text = (report_url or "").strip()
    parsed = urlparse(text)
    if parsed.scheme not in _ALLOWED_REPORT_SCHEMES or not parsed.netloc:
        msg = f"unsupported report url scheme: {parsed.scheme or '-'}"
        raise ValueError(msg)
    return text


def _download_report_html(report_url: str) -> str:
    safe_url = _assert_http_report_url(report_url)
    request = Request(safe_url, headers={"Accept": "text/html,*/*;q=0.8"})
    with urlopen(request, timeout=_FETCH_TIMEOUT_S) as response:  # noqa: S310
        raw = response.read(_MAX_REPORT_BYTES + 1)
        if len(raw) > _MAX_REPORT_BYTES:
            msg = "report html too large"
            raise ValueError(msg)
        charset = response.headers.get_content_charset() or "utf-8"
        content_type = (response.headers.get_content_type() or "").lower()
    body = raw.decode(charset, errors="replace")
    if not body.strip():
        msg = "empty report html"
        raise ValueError(msg)
    if "html" not in content_type and "<html" not in body[:500].lower():
        msg = f"unexpected content-type: {content_type or '-'}"
        raise ValueError(msg)
    return body


async def fetch_report_summary_html(report_url: str) -> str:
    """Fetch the published report HTML and keep the part before the case details.

    Raises:
        URLError / HTTPError: network failure.
        ValueError: the URL is invalid, or the body is empty, too large, or not HTML.
    """
    safe_url = _assert_http_report_url(report_url)
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    try:
        body = await loop.run_in_executor(
            _REPORT_FETCH_EXECUTOR,
            ctx.run,
            _download_report_html,
            safe_url,
        )
    except HTTPError as exc:
        logger.warning(
            "report_summary.fetch_http_error url=%s status=%s",
            report_url,
            exc.code,
        )
        raise
    except URLError:
        logger.warning("report_summary.fetch_url_error url=%s", report_url, exc_info=True)
        raise
    return strip_case_details_from_report_html(body)


__all__ = [
    "fetch_report_summary_html",
    "strip_case_details_from_report_html",
]
