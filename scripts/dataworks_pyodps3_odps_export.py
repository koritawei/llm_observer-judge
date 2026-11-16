"""DataWorks「PyODPS 3」节点脚本：部门账号访问 ODPS → CSV → OSS。

用法（粘贴到 DataWorks PyODPS 3 节点，勿在本仓库当服务入口跑）：
  官方说明：https://help.aliyun.com/zh/dataworks/user-guide/create-a-pyodps-3-node
  - 节点内已有全局 ``o`` / ``odps``（工作空间绑定的 MaxCompute / 部门账号），
    **不要**再 ``ODPS(access_id, ...)`` 手建入口。
  - 调度参数经全局 ``args`` 注入（不会替换代码里的 ``${...}``）。

调度参数（节点右侧「调度配置 → 参数」）建议::

  ds=${yyyymmdd}
  limit=200
  project_id=cmpwln0qs04fo6j8eryfdgznb
  oss_prefix=agent-eval/search-agent-overview
  oss_access_id=<部门账号 AccessKeyId>
  oss_access_key=<部门账号 AccessKeySecret>
  oss_endpoint=<OSS endpoint，如 https://oss-cn-hangzhou-zmf.aliyuncs.com>
  oss_bucket=<bucket 名>

日期约定（与 agent-eval 服务一致）::

  - ``args['ds']`` = **T**（OSS 路径日 / 业务日）
  - ODPS 分区 = **T-1**
  - object key = ``{oss_prefix}/{T}/benchmark_final_results.csv``

若工作空间业务日期 **已经是分区日**（常见：跑日 9 号、bizdate=8 号），请额外配置::

  ds_is_odps_partition=true

此时 ``args['ds']`` 当作 ODPS 分区，OSS 路径日 = ds+1。

注意::

  - 空结果主动 ``raise``，便于运维中心重跑；勿写空 CSV 覆盖。
  - 读结果开 Instance Tunnel，避免 Result 接口 1 万行截断
    （见官方「执行 SQL / open_reader」说明）。
  - 资源组需能访问 OSS；跨 VPC 请用 Serverless 资源组并打通网络。
  - **禁止**把真实 AK/SK 写进本文件或 Git（CT-ODPS-001）。
"""

import csv
import io
from datetime import datetime, timedelta

# DataWorks 注入：o / odps / args（本地静态检查用占位，节点内会被覆盖）
try:
    o  # type: ignore[name-defined]  # noqa: B018, F821
except NameError:  # pragma: no cover
    o = None  # type: ignore[assignment]
try:
    args  # type: ignore[name-defined]  # noqa: B018, F821
except NameError:  # pragma: no cover
    args = {}  # type: ignore[assignment]

MAX_LIMIT = 10000
DEFAULT_LIMIT = 200
DEFAULT_OSS_PREFIX = "agent-eval/search-agent-overview"
DEFAULT_PROJECT_ID = "cmpwln0qs04fo6j8eryfdgznb"

#: 结构与 RemoteConfig ``agent-eval.odps.export`` / 服务内置 DEFAULT_SQL_TEMPLATE 一致
#: （``{ds}``/``{limit}``/``{project_id}`` 由调度参数注入，支持按业务日期补跑历史分区）。
#: ``{project_id}`` 过滤 ``$.record.project_id``；CSV ``project`` 列取自该 JSON 字段。
#: 勿改成 ``GETDATE()`` 自包含形态，否则节点 bizdate 无法驱动分区。
SQL_TEMPLATE = """\
WITH overview_raw AS (
    SELECT  GET_JSON_OBJECT(content_json, '$.record.id') AS record_id
           ,GET_JSON_OBJECT(content_json, '$.record.output') AS raw_output
           ,GET_JSON_OBJECT(content_json, '$.record.project_id') AS project
    FROM    (
                SELECT  SPLIT_PART(content, CHR(1), 1) AS content_json
                FROM    autonavi_ai.s_tt_dwd_gd_ai_chat_req_langfuse_trace_raw_ri_tt4
                WHERE   ds = '{ds}'
            )
    WHERE   GET_JSON_OBJECT(content_json, '$.record.name') = 'search_agent.stream.overview'
      AND   GET_JSON_OBJECT(content_json, '$.record.id') IS NOT NULL
      AND   GET_JSON_OBJECT(content_json, '$.record.output') IS NOT NULL
      AND   GET_JSON_OBJECT(content_json, '$.record.output') <> ''
      AND   GET_JSON_OBJECT(content_json, '$.record.project_id') = '{project_id}'
)
,joined AS (
    SELECT  GET_JSON_OBJECT(t.root_input, '$.query') AS query
           ,t.langfuse_trace_id AS trace_id
           ,GET_JSON_OBJECT(t.root_input, '$.adiu') AS adiu
           ,GET_JSON_OBJECT(t.root_input, '$.longitude') AS longitude
           ,GET_JSON_OBJECT(t.root_input, '$.latitude') AS latitude
           ,GET_JSON_OBJECT(t.root_output, '$.address') AS address
           ,CONCAT(
                TO_CHAR(
                    FROM_UNIXTIME(CAST(t.trace_start_time / 1000 AS BIGINT))
                   ,'yyyy-mm-dd hh:mi:ss'
                )
               ,'.'
               ,LPAD(CAST(t.trace_start_time % 1000 AS STRING), 3, '0')
            ) AS current_time
           ,GET_JSON_OBJECT(t.root_output, '$.user_profile') AS user_profile
           ,GET_JSON_OBJECT(t.root_output, '$.app_use_context') AS app_use_context
           ,COALESCE(
                NULLIF(GET_JSON_OBJECT(t.root_output, '$.result'), '')
               ,NULLIF(s.raw_output, '')
            ) AS result
           ,t.tool_calls_json AS tool_call_json
           ,s.project AS project
    FROM    overview_raw s
    JOIN    autonavi_ai.dwd_gd_ai_chat_langfuse_trace_di t
    ON      t.ds = '{ds}'
     AND   t.langfuse_trace_id = s.record_id
    WHERE   GET_JSON_OBJECT(t.root_input, '$.query') IS NOT NULL
      AND   GET_JSON_OBJECT(t.root_input, '$.query') <> ''
)
SELECT  *
FROM    joined
ORDER BY RAND()
LIMIT   {limit}
"""


def _require_ds(raw: str, *, label: str) -> str:
    text = (raw or "").strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"{label} 必须是 YYYYMMDD，收到: {raw!r}")
    return text


def _shift_ds(ds: str, days: int) -> str:
    base = datetime.strptime(ds, "%Y%m%d")
    return (base + timedelta(days=days)).strftime("%Y%m%d")


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_dates(schedule_args: dict) -> tuple[str, str]:
    """返回 ``(odps_ds, oss_ds)``。"""
    raw = schedule_args.get("ds")
    if raw is None or str(raw).strip() == "":
        raise ValueError("调度参数缺少 ds（请配置 ds=${yyyymmdd}）")
    ds = _require_ds(str(raw), label="ds")
    if _truthy(schedule_args.get("ds_is_odps_partition")):
        return ds, _shift_ds(ds, 1)
    return _shift_ds(ds, -1), ds


def _resolve_limit(schedule_args: dict) -> int:
    raw = schedule_args.get("limit", DEFAULT_LIMIT)
    limit = int(raw)
    if not 0 < limit <= MAX_LIMIT:
        raise ValueError(f"limit 必须在 1..{MAX_LIMIT}: {limit}")
    return limit


def _resolve_project_id(schedule_args: dict) -> str:
    text = str(schedule_args.get("project_id") or DEFAULT_PROJECT_ID).strip()
    if not text or not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"project_id 非法: {text!r}（仅允许 [A-Za-z0-9_-]）")
    return text


def _rows_to_csv_bytes(columns: list[str], rows: list[list[object]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if value is None else str(value) for value in row])
    return buf.getvalue().encode("utf-8-sig")


def _run_sql(client: object, sql: str) -> tuple[list[str], list[list[object]]]:
    """执行 SQL；Instance Tunnel 读全量（受 SQL LIMIT 约束）。"""
    instance = client.execute_sql(sql)
    # tunnel=True / limit=False：见 DataWorks PyODPS 3「执行 SQL」说明
    with instance.open_reader(tunnel=True, limit=False) as reader:
        columns = [column.name for column in reader.schema.columns]
        rows = [list(record.values) for record in reader]
    return columns, rows


def _upload_oss(
    *,
    payload: bytes,
    object_key: str,
    schedule_args: dict,
) -> str:
    """用部门账号 AccessId 上传 OSS；返回 ``oss://bucket/key``。"""
    try:
        from benchmark.libs.storage import osscompat as oss2
    except ImportError as exc:
        raise RuntimeError(
            "资源组未安装 oss2：Serverless 请用自定义镜像 / 运维助手 pip install oss2",
        ) from exc

    access_id = str(schedule_args.get("oss_access_id") or "").strip()
    access_key = str(schedule_args.get("oss_access_key") or "").strip()
    endpoint = str(schedule_args.get("oss_endpoint") or "").strip()
    bucket_name = str(schedule_args.get("oss_bucket") or "").strip()
    missing = [
        name
        for name, value in (
            ("oss_access_id", access_id),
            ("oss_access_key", access_key),
            ("oss_endpoint", endpoint),
            ("oss_bucket", bucket_name),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"OSS 调度参数缺失: {', '.join(missing)}")

    auth = oss2.Auth(access_id, access_key)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    bucket.put_object(object_key, payload, headers={"Content-Type": "text/csv"})
    return f"oss://{bucket_name}/{object_key}"


def main() -> None:
    if o is None:
        raise RuntimeError("未找到全局 odps/o：请在 DataWorks PyODPS 3 节点内运行")

    schedule_args = dict(args) if args is not None else {}
    odps_ds, oss_ds = _resolve_dates(schedule_args)
    limit = _resolve_limit(schedule_args)
    project_id = _resolve_project_id(schedule_args)
    oss_prefix = (
        str(schedule_args.get("oss_prefix") or DEFAULT_OSS_PREFIX).strip().strip("/")
        or DEFAULT_OSS_PREFIX
    )
    object_key = f"{oss_prefix}/{oss_ds}/benchmark_final_results.csv"
    sql = (
        SQL_TEMPLATE.replace("{ds}", odps_ds)
        .replace("{limit}", str(limit))
        .replace("{project_id}", project_id)
    )

    print(
        f"export start odps_ds={odps_ds} oss_ds={oss_ds} limit={limit} "
        f"project_id={project_id} key={object_key}",
    )
    columns, rows = _run_sql(o, sql)
    print(f"export queried rows={len(rows)} columns={len(columns)}")
    if not rows:
        raise RuntimeError(
            f"查询结果为空（odps_ds={odps_ds} 分区无数据？），跳过写 OSS",
        )

    payload = _rows_to_csv_bytes(columns, rows)
    uri = _upload_oss(payload=payload, object_key=object_key, schedule_args=schedule_args)
    print(
        f"export done rows={len(rows)} bytes={len(payload)} uri={uri}",
    )


main()
