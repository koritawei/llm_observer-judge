"""ODPS sample-export core: SQL render → pyodps query → CSV serialization → OSS upload.

The in-service scheduler (:mod:`benchmark.workers.odps_export_scheduler`) and the
manual backfill script (``scripts/export_odps_to_oss.py``) share this module, so the
two implementations do not drift.

- SQL template source: RemoteConfig ``agent-eval.odps.export`` (``sql_template``, literal
  replace of ``{ds}`` / ``{limit}``; if both placeholders are absent the SQL is treated as
  self-contained and executed as-is; optional ``{project_id}`` is injected from RemoteConfig
  ``jobs[].project_id`` and used to filter ``$.record.project_id``; the CSV ``project``
  column is taken from that JSON field). This module ships :data:`DEFAULT_SQL_TEMPLATE`
  (with ``{ds}``/``{limit}``/``{project_id}``) as the fallback for ``--no-remote_config``
  or when RemoteConfig is unavailable.
- **Date convention**: ODPS partition ``odps_ds`` = **T-1**; OSS path basis ``oss_ds`` = **T**.
  Classic layout ``{prefix}/{oss_ds}/benchmark_final_results.csv``. When the prefix contains
  a date macro or its last segment is already ``yyyymmdd``, ``oss_ds`` is not appended.
- CSV dialect: header row + UTF-8 BOM + every field quoted (aligned with
  ``docs/benchmark_final_results.csv``, so Excel opens it without mojibake).
- OSS: reuse :func:`build_data_storage_client` (RemoteConfig
  ``agent-eval.benchmark.oss``, Normandy credentials).
- ODPS credentials: Normandy resource binding (``normandy-credential-sdk[extension]``,
  resource name delivered via ``AGENT_EVAL_ODPS_AONE_APP_NAME``; CT-ODPS-001 forbids hardcoded AK/SK).
"""

import asyncio
import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.settings.setting import settings
from benchmark.domain.odps_export_config import (
    DEFAULT_SQL_TEMPLATE,
    normalize_sql_project_id,
)
from benchmark.runs.common.oss_date_macro import (
    expand_date_macros,
    has_date_macro,
    object_key_for_dated_prefix,
    validate_date_macros,
)
from benchmark.reporting.framework.report_storage import build_data_storage_client

if TYPE_CHECKING:
    from benchmark.libs.storage import StorageClient

logger = get_logger("agent_eval.eval_framework.odps_export")

#: Sample-count cap, so a mistake cannot pull the whole table.
MAX_LIMIT = 10000
#: Default OSS object-key prefix (used when RemoteConfig ``oss_prefix`` is blank).
DEFAULT_OSS_PREFIX = "agent-eval/search-agent-overview"

_DS_PATTERN = re.compile(r"^\d{8}$")

#: search_eval / Amap Langfuse project (the SQL ``{project_id}`` filter value; default for ``--no-remote_config``)
DEFAULT_PROJECT_ID = "cmpwln0qs04fo6j8eryfdgznb"

# DEFAULT_SQL_TEMPLATE is defined in the domain layer and re-exported here, so existing script and scheduler import paths stay valid.


class OdpsExportError(RuntimeError):
    """Base class for export failures."""


class OdpsExportEmptyError(OdpsExportError):
    """The query result is empty (the partition has no data): the caller must not write an artifact (protects against overwriting with an empty file)."""


class OdpsExportStorageError(OdpsExportError):
    """The OSS client is unavailable or the upload failed."""


@dataclass(frozen=True)
class OdpsExportResult:
    """Export result."""

    odps_ds: str
    oss_ds: str
    rows: int
    columns: list[str] = field(default_factory=list)
    csv_bytes: bytes = b""
    object_key: str = ""
    oss_url: str | None = None

    @property
    def ds(self) -> str:
        """Compatibility field: same as ``odps_ds`` (ODPS partition day T-1)."""
        return self.odps_ds


@dataclass(frozen=True)
class OdpsExportRequest:
    """One export request (ODPS partition T-1 + OSS path date T)."""

    sql_template: str
    odps_ds: str
    oss_ds: str
    limit: int
    oss_prefix: str
    object_key: str | None = None
    #: Object filename (joined into the key only when ``object_key`` is empty); default benchmark_final_results.csv
    object_name: str = "benchmark_final_results.csv"
    #: Business job id (logs / lock; optional)
    job_id: str = "default"
    #: Langfuse project; when the template contains ``{project_id}``, inject the SQL filter from RemoteConfig
    project_id: str = ""


def _validate_ds(ds: str, *, label: str = "ds") -> str:
    if not _DS_PATTERN.match(ds):
        msg = f"{label} 必须是 YYYYMMDD 格式: {ds!r}"
        raise ValueError(msg)
    return ds


def default_odps_ds(now: datetime) -> str:
    """ODPS partition day = **T-1** ``YYYYMMDD`` (the caller passes now in the schedule timezone)."""
    return (now - timedelta(days=1)).strftime("%Y%m%d")


def default_oss_ds(now: datetime) -> str:
    """OSS path date = **T** (today) ``YYYYMMDD``."""
    return now.strftime("%Y%m%d")


def default_ds(now: datetime) -> str:
    """Compatibility alias: same as :func:`default_odps_ds` (T-1)."""
    return default_odps_ds(now)


def shift_ds(ds: str, *, days: int) -> str:
    """Shift ``YYYYMMDD`` by ``days`` days (may be negative)."""
    _validate_ds(ds)
    base = datetime.strptime(ds, "%Y%m%d")
    return (base + timedelta(days=days)).strftime("%Y%m%d")


def render_sql(
    template: str,
    ds: str,
    limit: int,
    *,
    project_id: str = "",
) -> str:
    """Render the sample SQL: literal replace of ``{ds}`` / ``{limit}`` / optional ``{project_id}``.

    ``ds`` is the **ODPS partition day (T-1)** and is strictly checked against ``^\\d{8}$``.
    ``{ds}``/``{limit}`` follow an all-or-nothing rule:

    - Both placeholders present → replace and return (the recommended RemoteConfig shape;
      any partition can be backfilled);
    - Neither present → treat as self-contained SQL (``GETDATE()`` and similar) and return
      it unchanged; **``ds``/``limit`` do not enter the SQL**, so a backfill should use a
      placeholder template instead;
    - Only one present → fail explicitly as a config error (a half replacement would run
      the wrong partition or a full-table scan).

    ``{project_id}`` is independent: when the template contains that placeholder a valid
    ``project_id`` is required (RemoteConfig ``jobs[].project_id``) and is used as an SQL
    filter (for example
    ``GET_JSON_OBJECT(..., '$.record.project_id') = '{project_id}'``).
    If the template does not contain it, the argument is ignored.
    """
    _validate_ds(ds, label="odps_ds")
    if not 0 < limit <= MAX_LIMIT:
        msg = f"limit 必须在 1..{MAX_LIMIT} 之间: {limit}"
        raise ValueError(msg)
    has_ds = "{ds}" in template
    has_limit = "{limit}" in template
    if has_ds != has_limit:
        missing = "{limit}" if has_ds else "{ds}"
        msg = (
            f"sql_template 缺少占位符 {missing}（RemoteConfig agent-eval.odps.export 配置错误；"
            "占位符须成对出现，或全部缺失走自包含 SQL）"
        )
        raise ValueError(msg)
    sql = template if not has_ds else template.replace("{ds}", ds).replace("{limit}", str(limit))
    if "{project_id}" not in sql:
        return sql
    pid = normalize_sql_project_id(project_id)
    if not pid:
        msg = "sql_template 含 {project_id} 但未提供 project_id（RemoteConfig jobs[].project_id）"
        raise ValueError(msg)
    return sql.replace("{project_id}", pid)


def rows_to_csv_bytes(columns: list[str], rows: list[list[Any]]) -> bytes:
    """Serialize to CSV bytes: header row + UTF-8 BOM + every field quoted (Excel opens it directly)."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if value is None else str(value) for value in row])
    return buf.getvalue().encode("utf-8-sig")


def resolve_object_key(
    oss_ds: str,
    prefix: str,
    object_key: str | None = None,
    *,
    object_name: str = "benchmark_final_results.csv",
) -> str:
    """Target object key: ``object_key`` wins, otherwise assemble it from the prefix semantics.

    - Prefix contains ``${yyyymmdd±n}``: expand it with ``oss_ds`` (T) and that is the data
      directory; do not append ``oss_ds`` again (same macro semantics as the evaluation pull side).
    - The prefix's last segment is already ``yyyymmdd``: treat it as a located date partition
      and drop the file there.
    - Otherwise the classic layout ``{prefix}/{oss_ds}/{object_name}``.

    ``oss_ds`` is **T (today)**, not the ODPS partition day.
    """
    if object_key:
        return object_key.strip().lstrip("/")
    _validate_ds(oss_ds, label="oss_ds")
    cleaned = (prefix or DEFAULT_OSS_PREFIX).strip().strip("/")
    name = (object_name or "benchmark_final_results.csv").strip().replace("\\", "/").split("/")[-1]
    if not name or name in {".", ".."}:
        name = "benchmark_final_results.csv"
    if has_date_macro(cleaned) or "${" in cleaned:
        validate_date_macros(cleaned)
        cleaned = expand_date_macros(cleaned, base_date=oss_ds).strip().strip("/")
    pinned = object_key_for_dated_prefix(cleaned, file_name=name)
    if pinned:
        return pinned
    return f"{cleaned}/{oss_ds}/{name}"


def execute_query(sql: str) -> tuple[list[str], list[list[Any]]]:
    """Run SQL through pyodps with Normandy credentials and return ``(column names, rows)``.

    The credential path is the same Normandy mode as ``libs/storage`` OSS:
    ``names.of_odps_account_aone_app`` builds the ODPS resource URN, and
    ``extension.factory.get_acs_credential_provider`` produces a provider that pyodps
    ``CredentialProviderAccount`` can consume (the SDK caches and rotates credentials).
    pyodps and the SDK are optional dependencies used only for export and are imported
    lazily inside the function (the service does not load them at runtime). A missing
    credential fails explicitly (silently running an empty partition would return wrong
    data and be hard to notice).

    Runtime prerequisite: the process must be able to resolve the app name (``APP_NAME`` /
    ``SIGMA_APP_NAME``, or a ``*.release`` file in cwd) — a hard requirement of Normandy SDK init.
    """
    try:
        # Optional intranet dependency, not installed locally/in CI; a bare ignore covers both import-not-found and import-untyped
        from benchmark.libs.credential import names  # type: ignore  # noqa: PLC0415
        from benchmark.libs.credential.extension import factory  # type: ignore  # noqa: PLC0415
        from benchmark.libs.warehouse import ODPS  # type: ignore[import-untyped]  # noqa: PLC0415
        from benchmark.libs.warehouse.accounts import CredentialProviderAccount  # type: ignore  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "pyodps 或 normandy-credential-sdk[extension] 未安装：pip install "
            "'pyodps>=0.13.1,<0.14' 'normandy-credential-sdk[extension]==1.1.1'（artlab 源）"
        )
        raise OdpsExportError(msg) from exc

    missing = [
        name
        for name, value in (
            ("AGENT_EVAL_ODPS_NOTIFICATION_APP_NAME", settings.odps_notification_app_name),
            ("AGENT_EVAL_ODPS_ENDPOINT", settings.odps_endpoint),
            ("AGENT_EVAL_ODPS_PROJECT", settings.odps_project),
        )
        if not value
    ]
    if missing:
        msg = f"ODPS 配置缺失，缺少环境变量: {', '.join(missing)}"
        raise OdpsExportError(msg)

    notification_app = settings.odps_notification_app_name
    project = settings.odps_project
    endpoint = settings.odps_endpoint
    urn = names.of_odps_account_notification_app(notification_app)
    try:
        provider = factory.get_acs_credential_provider(urn)
    except Exception as exc:
        msg = (
            f"Normandy ODPS 凭证获取失败（urn={urn}）：请确认该 ODPS 资源已绑定到"
            "当前应用环境，且进程可解析应用名（APP_NAME 环境变量或 cwd 下 *.release 文件）"
        )
        raise OdpsExportError(msg) from exc

    client = ODPS(
        account=CredentialProviderAccount(provider),
        project=project,
        endpoint=endpoint,
    )
    logger.info(
        "odps_export.execute_sql aone_app=%s project=%s endpoint=%s sql_chars=%d",
        notification_app,
        project,
        endpoint,
        len(sql),
    )
    instance = client.execute_sql(sql)
    # Wait for the terminal state explicitly, then open the Tunnel; on some environments open_reader reads an empty set if wait was skipped
    wait_for_success = getattr(instance, "wait_for_success", None)
    if callable(wait_for_success):
        wait_for_success()
    instance_id = getattr(instance, "id", None) or getattr(instance, "instance_id", "")
    status = getattr(instance, "status", None)
    logview = _safe_logview(instance)

    columns, rows, reader_count, reader_mode = _read_instance_rows(instance)

    logger.info(
        "odps_export.reader_done instance_id=%s status=%s reader_mode=%s "
        "reader_count=%s rows=%d columns=%d logview=%s",
        instance_id,
        status,
        reader_mode,
        reader_count,
        len(rows),
        len(columns),
        logview or "(none)",
    )
    if not rows:
        task_diag = _format_task_diagnostics(instance)
        sql_head = " ".join(sql.split())[:240]
        hint = (
            f"查询结果为空（instance={instance_id} status={status} "
            f"reader_mode={reader_mode} aone_app={notification_app} project={project} "
            f"endpoint={endpoint}"
            f"{f' logview={logview}' if logview else ''}）。"
            f" task_diag=[{task_diag}] sql_head={sql_head!r}。"
            "请打开 logview 核对 Output Records；"
            "Inputs 有量而 Output=0 → JOIN/过滤后真空，或 JSON 列 Label Security "
            "读出 NULL（个人账号有行而本账号 0 行即为列级不可见实锤）。"
            "可跑：bash scripts/export_odps_to_oss.sh --oss-ds <T> --probe "
            "（p0 段自动判读行/列可见性）"
        )
        raise OdpsExportEmptyError(hint)
    return columns, rows


def build_overview_probe_sqls(odps_ds: str) -> list[tuple[str, str]]:
    """Staged COUNTs for an empty search_agent.overview result (same filters as the DEFAULT/RemoteConfig SQL).

    Returns a ``(label, sql)`` list, executed in order. The p0 stage has no JSON
    extraction and exists to distinguish "rows invisible / columns invisible / truly no
    data": when Label Security reads the ``content`` column as NULL for the Normandy-bound
    account, p1 is also 0, so p1/p2 alone cannot locate a permission problem. p1b prints
    the name distribution (this account's view). When p1=0 it confirms no traffic, a
    rename, or field-level redaction (overview disappears from the distribution, or the
    name value was rewritten). After the run, hand ``{label: count}`` to
    :func:`diagnose_probe_results` for the summary reading.
    """
    _validate_ds(odps_ds, label="odps_ds")
    raw_rows = f"""\
SELECT  COUNT(1) AS c
FROM    autonavi_ai.s_tt_dwd_gd_ai_chat_req_langfuse_trace_raw_ri_tt4
WHERE   ds = '{odps_ds}'
"""
    raw_visible = f"""\
SELECT  COUNT(1) AS c
FROM    autonavi_ai.s_tt_dwd_gd_ai_chat_req_langfuse_trace_raw_ri_tt4
WHERE   ds = '{odps_ds}'
  AND   content IS NOT NULL
  AND   LENGTH(content) > 0
"""
    dwd_visible = f"""\
SELECT  COUNT(1) AS c
FROM    autonavi_ai.dwd_gd_ai_chat_langfuse_trace_di
WHERE   ds = '{odps_ds}'
  AND   root_input IS NOT NULL
  AND   LENGTH(root_input) > 0
"""
    raw = f"""\
SELECT  COUNT(1) AS c
FROM    autonavi_ai.s_tt_dwd_gd_ai_chat_req_langfuse_trace_raw_ri_tt4
WHERE   ds = '{odps_ds}'
  AND   GET_JSON_OBJECT(content, '$.record.name') = 'search_agent.stream.overview'
  AND   GET_JSON_OBJECT(content, '$.record.id') IS NOT NULL
  AND   GET_JSON_OBJECT(content, '$.record.output') IS NOT NULL
  AND   GET_JSON_OBJECT(content, '$.record.output') <> ''
"""
    name_dist = f"""\
SELECT  GET_JSON_OBJECT(content, '$.record.name') AS name
       ,COUNT(1) AS cnt
FROM    autonavi_ai.s_tt_dwd_gd_ai_chat_req_langfuse_trace_raw_ri_tt4
WHERE   ds = '{odps_ds}'
GROUP BY GET_JSON_OBJECT(content, '$.record.name')
ORDER BY cnt DESC
LIMIT   15
"""
    join = f"""\
WITH overview_raw AS (
    SELECT  GET_JSON_OBJECT(content, '$.record.id') AS record_id
    FROM    autonavi_ai.s_tt_dwd_gd_ai_chat_req_langfuse_trace_raw_ri_tt4
    WHERE   ds = '{odps_ds}'
      AND   GET_JSON_OBJECT(content, '$.record.name') = 'search_agent.stream.overview'
      AND   GET_JSON_OBJECT(content, '$.record.id') IS NOT NULL
      AND   GET_JSON_OBJECT(content, '$.record.output') IS NOT NULL
      AND   GET_JSON_OBJECT(content, '$.record.output') <> ''
)
SELECT  COUNT(1) AS c
FROM    overview_raw s
JOIN    autonavi_ai.dwd_gd_ai_chat_langfuse_trace_di t
ON      t.ds = '{odps_ds}'
 AND   t.langfuse_trace_id = s.record_id
WHERE   GET_JSON_OBJECT(t.root_input, '$.query') IS NOT NULL
  AND   GET_JSON_OBJECT(t.root_input, '$.query') <> ''
"""
    sample = f"""\
WITH overview_raw AS (
    SELECT  GET_JSON_OBJECT(content, '$.record.id') AS record_id
    FROM    autonavi_ai.s_tt_dwd_gd_ai_chat_req_langfuse_trace_raw_ri_tt4
    WHERE   ds = '{odps_ds}'
      AND   GET_JSON_OBJECT(content, '$.record.name') = 'search_agent.stream.overview'
      AND   GET_JSON_OBJECT(content, '$.record.id') IS NOT NULL
      AND   GET_JSON_OBJECT(content, '$.record.output') IS NOT NULL
      AND   GET_JSON_OBJECT(content, '$.record.output') <> ''
)
SELECT  s.record_id
       ,t.langfuse_trace_id
       ,GET_JSON_OBJECT(t.root_input, '$.query') AS query
FROM    overview_raw s
JOIN    autonavi_ai.dwd_gd_ai_chat_langfuse_trace_di t
ON      t.ds = '{odps_ds}'
 AND   t.langfuse_trace_id = s.record_id
WHERE   GET_JSON_OBJECT(t.root_input, '$.query') IS NOT NULL
  AND   GET_JSON_OBJECT(t.root_input, '$.query') <> ''
LIMIT   5
"""
    return [
        ("p0_raw_partition_rows", raw_rows),
        ("p0_raw_content_visible", raw_visible),
        ("p0_dwd_rootjson_visible", dwd_visible),
        ("p1_raw_filter_count", raw),
        ("p1b_name_distribution", name_dist),
        ("p2_join_count", join),
        ("p3_join_sample5", sample),
    ]


def diagnose_probe_results(counts: dict[str, int | None]) -> str:
    """Read the empty-result root cause from the ``--probe`` staged COUNTs (column invisible vs truly no data).

    ``counts`` is ``{label: count}``. Pass ``None`` when a step failed to execute (that
    step is skipped). Reading matrix:

    - ``p0_raw_partition_rows=0`` → the raw partition has no data (or row-level policy hides it);
    - The partition has rows but ``p0_raw_content_visible=0`` → the ``content`` column is
      invisible to the current account (Label Security/redaction reads NULL). Typical
      sign: a personal account has rows and the service account has 0 → request column-level
      authorization for the ODPS account bound to aone_app;
    - p0 is normal but ``p1_raw_filter_count=0`` → no search_agent.stream.overview trace
      that day (no traffic / name changed), unrelated to permissions;
    - p1>0 but ``p2_join_count=0`` → the JOIN misses (record_id ↔ langfuse_trace_id);
    - p2>0 but ``p0_dwd_rootjson_visible=0`` → the dwd ``root_input``/``root_output``
      columns are invisible; the JOIN hits but ``$.query`` extracts as NULL.
    """
    raw_rows = counts.get("p0_raw_partition_rows")
    p1 = counts.get("p1_raw_filter_count")
    p2 = counts.get("p2_join_count")
    if raw_rows is None:
        diagnosis = "probe 未完整执行（p0_raw_partition_rows 缺失）：请人工逐条核对上方各阶段结果"
    elif raw_rows == 0:
        diagnosis = (
            "raw 分区无数据：ds 分区尚未产出，或行级策略对当前账号不可见；"
            "用个人账号跑同分区 COUNT 对照区分"
        )
    elif counts.get("p0_raw_content_visible") == 0:
        diagnosis = (
            "content 列对当前账号不可见（Label Security/脱敏读出 NULL）："
            "分区有数据但 content 全空，是个人账号有行而服务账号 0 行的典型特征 → "
            "找表 owner 确认列 label 等级，给 aone_app 绑定 ODPS 账号申请列级授权"
        )
    elif p1 == 0:
        diagnosis = (
            "raw 表当天无 search_agent.stream.overview trace（列可见但 name 过滤为空）："
            "链路无流量 / name 变更 / 字段级脱敏改写 name 值；"
            "核对上方 p1b_name_distribution 输出确认本账号视角的 name 分布"
        )
    elif p1 is not None and p1 > 0 and p2 == 0:
        diagnosis = (
            "JOIN 不命中：raw $.record.id 与 dwd langfuse_trace_id 对不上；"
            "核对两侧 id 形态（引号/大小写/前缀）"
        )
    elif (
        p1 is not None
        and p1 > 0
        and p2 is not None
        and p2 > 0
        and counts.get("p0_dwd_rootjson_visible") == 0
    ):
        diagnosis = (
            "dwd root_input/root_output 列对当前账号不可见：JOIN 命中但 "
            "$.query 提取为 NULL → 给绑定账号申请 dwd 表列级授权"
        )
    else:
        diagnosis = (
            "行/列可见性与 JOIN 各阶段均有数据：空结果出在主查询其他过滤或读结果环节，"
            "请对照 logview Output Records 与落盘 .sql 逐字比对"
        )
    return diagnosis


def _safe_logview(instance: Any) -> str:  # noqa: ANN401 — pyodps Instance is an optional dynamic dependency
    get_logview = getattr(instance, "get_logview_address", None)
    if not callable(get_logview):
        return ""
    try:
        return str(get_logview() or "")
    except Exception:  # noqa: BLE001 — diagnostic helper
        return ""


def _read_instance_rows(
    instance: Any,  # noqa: ANN401 — pyodps Instance is an optional dynamic dependency
) -> tuple[list[str], list[list[Any]], Any, str]:
    """Prefer the Instance Tunnel; if it returns 0 rows, try the Result API, so a truly empty result can be told apart from a read that came back empty."""
    columns: list[str] = []
    rows: list[list[Any]] = []
    reader_count: Any = None

    with instance.open_reader(tunnel=True, limit=False) as reader:
        columns = [column.name for column in reader.schema.columns]
        rows = [list(record.values) for record in reader]
        reader_count = getattr(reader, "count", None)

    if rows:
        return columns, rows, reader_count, "tunnel"

    # Tunnel empty: try Result once more (small result sets); treat the SQL as having no rows only when both are empty
    try:
        with instance.open_reader() as reader:
            alt_columns = [column.name for column in reader.schema.columns]
            alt_rows = [list(record.values) for record in reader]
            alt_count = getattr(reader, "count", None)
    except Exception as exc:  # noqa: BLE001 — diagnosis / fallback only
        logger.warning(
            "odps_export.reader_fallback_failed err=%s",
            exc,
            exc_info=True,
        )
        return columns, rows, reader_count, "tunnel"

    if alt_rows:
        logger.warning(
            "odps_export.reader_tunnel_empty_but_result_ok "
            "tunnel_count=%s result_count=%s rows=%d — 使用 Result 接口数据",
            reader_count,
            alt_count,
            len(alt_rows),
        )
        return alt_columns, alt_rows, alt_count, "result-fallback"

    return columns, rows, reader_count, "tunnel+result-empty"


def _format_task_diagnostics(instance: Any) -> str:  # noqa: ANN401 — pyodps Instance is an optional dynamic dependency
    """Extract as much of the task summary / result summary as possible, for empty-result investigation."""
    parts: list[str] = []
    get_names = getattr(instance, "get_task_names", None)
    if not callable(get_names):
        return "n/a"
    try:
        names = list(get_names() or [])
    except Exception as exc:  # noqa: BLE001
        return f"task_names_err={exc}"

    get_summary = getattr(instance, "get_task_summary", None)
    get_results = getattr(instance, "get_task_results", None)
    results: dict[str, Any] = {}
    if callable(get_results):
        try:
            results = dict(get_results() or {})
        except Exception:  # noqa: BLE001
            results = {}

    for name in names[:5]:
        summary = ""
        if callable(get_summary):
            try:
                summary = str(get_summary(name) or "")[:200]
            except Exception:  # noqa: BLE001
                summary = ""
        result_preview = ""
        if name in results:
            result_preview = str(results[name])[:120]
        parts.append(f"{name}:summary={summary!r}:result={result_preview!r}")
    return "; ".join(parts) if parts else "no-tasks"


async def run_odps_export(
    request: OdpsExportRequest,
    *,
    storage_client: "StorageClient | None" = None,
) -> OdpsExportResult:
    """Run one export: render SQL (T-1) → query → CSV → OSS (path date T).

    Raises:
        ValueError: ds / limit / a template placeholder is invalid.
        OdpsExportEmptyError: the query result is empty (no artifact is written).
        OdpsExportStorageError: the OSS client is unavailable or the upload failed.
        OdpsExportError: pyodps is missing / credentials are missing.
    """
    sql = render_sql(
        request.sql_template,
        request.odps_ds,
        request.limit,
        project_id=request.project_id,
    )
    columns, rows = await asyncio.to_thread(execute_query, sql)
    logger.info(
        "odps_export.queried odps_ds=%s oss_ds=%s rows=%d columns=%d",
        request.odps_ds,
        request.oss_ds,
        len(rows),
        len(columns),
    )
    # execute_query already raises OdpsExportEmptyError on an empty result (includes project/logview)
    payload = rows_to_csv_bytes(columns, rows)
    key = resolve_object_key(
        request.oss_ds,
        request.oss_prefix,
        request.object_key,
        object_name=request.object_name,
    )

    client = storage_client if storage_client is not None else build_data_storage_client()
    if client is None:
        raise OdpsExportStorageError(
            "OSS client 不可用（RemoteConfig agent-eval.benchmark.oss 配置缺失或 storage 依赖未装）",
        )
    url = await client.upload(key, payload, content_type="text/csv")
    if url is None:
        raise OdpsExportStorageError(f"OSS upload returned None: {key}")

    logger.info(
        "odps_export.done odps_ds=%s oss_ds=%s rows=%d bytes=%d key=%s url=%s",
        request.odps_ds,
        request.oss_ds,
        len(rows),
        len(payload),
        key,
        url,
    )
    return OdpsExportResult(
        odps_ds=request.odps_ds,
        oss_ds=request.oss_ds,
        rows=len(rows),
        columns=columns,
        csv_bytes=payload,
        object_key=key,
        oss_url=url,
    )


__all__ = [
    "DEFAULT_OSS_PREFIX",
    "DEFAULT_PROJECT_ID",
    "DEFAULT_SQL_TEMPLATE",
    "MAX_LIMIT",
    "OdpsExportEmptyError",
    "OdpsExportError",
    "OdpsExportRequest",
    "OdpsExportResult",
    "OdpsExportStorageError",
    "build_overview_probe_sqls",
    "default_ds",
    "diagnose_probe_results",
    "default_odps_ds",
    "default_oss_ds",
    "execute_query",
    "render_sql",
    "resolve_object_key",
    "rows_to_csv_bytes",
    "run_odps_export",
    "shift_ds",
]
