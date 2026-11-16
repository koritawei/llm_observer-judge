#!/usr/bin/env python3
"""运维补跑：ODPS 抽样导出 → CSV → OSS（支持多业务 job）。

核心逻辑在 ``benchmark.reporting.framework.odps_export``；本脚本仅提供 CLI。
线上日调度请用 ``OdpsExportScheduler``（RemoteConfig ``agent-eval.odps.export``）。

日期约定：ODPS 分区 ``odps_ds`` = T-1；OSS ``oss_ds`` = T。

用法::

  python3 scripts/export_odps_to_oss.py --oss-ds 20260909
  python3 scripts/export_odps_to_oss.py --job search-agent-overview --no-upload
  python3 scripts/export_odps_to_oss.py --dry-run
  python3 scripts/export_odps_to_oss.py --no-remote_config   # 内置 default job

线上 PYTHONPATH 对齐请用 ``scripts/export_odps_to_oss.sh``。
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import _bootstrap  # sys.path/.env/禁 MQ，须先于 app.* 执行
from benchmark.settings.setting import settings as _settings
from benchmark.domain.odps_export_config import (
    OdpsExportJob,
    fetch_odps_export_config,
)
from benchmark.reporting.framework.odps_export import (
    DEFAULT_OSS_PREFIX,
    DEFAULT_PROJECT_ID,
    DEFAULT_SQL_TEMPLATE,
    MAX_LIMIT,
    OdpsExportEmptyError,
    OdpsExportError,
    OdpsExportRequest,
    OdpsExportStorageError,
    build_overview_probe_sqls,
    default_odps_ds,
    default_oss_ds,
    diagnose_probe_results,
    execute_query,
    render_sql,
    resolve_object_key,
    rows_to_csv_bytes,
    run_odps_export,
    shift_ds,
)

del _bootstrap

_DEFAULT_LIMIT = 200
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "exports"


def _builtin_default_job(
    *, limit: int, prefix: str | None, project_id: str | None
) -> OdpsExportJob:
    return OdpsExportJob(
        id="default",
        enabled=True,
        limit=limit,
        oss_prefix=prefix or DEFAULT_OSS_PREFIX,
        project_id=project_id or DEFAULT_PROJECT_ID,
        sql_template=DEFAULT_SQL_TEMPLATE,
    )


def _load_jobs(  # noqa: PLR0912 — CLI 覆盖矩阵分支集中，拆分反而割裂配置来源语义
    *,
    use_remote_config: bool,
    job_id: str | None,
    limit: int,
    prefix: str | None,
    project_id: str | None,
) -> tuple[list[OdpsExportJob], str]:
    """加载要执行的 job 列表与来源标记。"""
    if not use_remote_config:
        job = _builtin_default_job(limit=limit, prefix=prefix, project_id=project_id)
        if job_id and job_id != job.id:
            msg = f"未知 --job={job_id!r}（--no-remote_config 仅有 id=default）"
            raise ValueError(msg)
        return [job], "default"

    try:
        config = fetch_odps_export_config(use_cache=False)
    except Exception as exc:
        print(f"WARN: RemoteConfig odps.export 拉取失败，用内置默认: {exc}", file=sys.stderr)
        job = _builtin_default_job(limit=limit, prefix=prefix, project_id=project_id)
        return [job], "remote_config-fallback"

    # CLI 覆盖顶层缺省后再 resolved
    overrides: dict[str, object] = {}
    if limit != _DEFAULT_LIMIT:
        overrides["limit"] = limit
    if prefix is not None:
        overrides["oss_prefix"] = prefix
    if project_id:
        overrides["project_id"] = project_id
    if overrides:
        config = config.model_copy(update=overrides)

    if not config.resolved_jobs():
        data_id = _settings.odps_export_remote_config_data_id
        group = _settings.odps_export_remote_config_group
        print(
            "WARN: RemoteConfig 无 jobs 且 sql_template 为空，回落内置 default；"
            f"当前拉取坐标 data_id={data_id!r} group={group!r} "
            f"（snapshot 形如 {data_id}+{group}+）。"
            "正确应为 data_id='agent-eval.odps.export' "
            "group='agent-eval.service.group'；"
            "请检查进程 env / antx 是否把 ODPS_EXPORT_REMOTE_CONFIG_* 改成了 odps_export",
            file=sys.stderr,
        )
        job = _builtin_default_job(limit=limit, prefix=prefix, project_id=project_id)
        return [job], "remote_config-fallback"

    if job_id:
        found = config.get_job(job_id)
        if found is None:
            known = [j.id for j in config.resolved_jobs()]
            msg = f"未知 --job={job_id!r}；可选: {known}"
            raise ValueError(msg)
        if not found.sql_template.strip():
            msg = f"job={job_id!r} sql_template 为空"
            raise ValueError(msg)
        # CLI prefix/limit/project_id 覆盖单 job
        updates: dict[str, object] = {}
        if prefix is not None:
            updates["oss_prefix"] = prefix
        if limit != _DEFAULT_LIMIT:
            updates["limit"] = limit
        if project_id:
            updates["project_id"] = project_id
        if updates:
            found = found.model_copy(update=updates)
        return [found], "remote_config"

    jobs = config.enabled_jobs()
    if not jobs:
        msg = "没有 enabled 且 sql 非空的 job"
        raise ValueError(msg)
    if prefix is not None or limit != _DEFAULT_LIMIT or project_id:
        patched: list[OdpsExportJob] = []
        for job in jobs:
            updates = {}
            if prefix is not None:
                updates["oss_prefix"] = prefix
            if limit != _DEFAULT_LIMIT:
                updates["limit"] = limit
            if project_id:
                updates["project_id"] = project_id
            patched.append(job.model_copy(update=updates) if updates else job)
        jobs = patched
    return jobs, "remote_config"


def _resolve_dates(
    *,
    odps_ds: str | None,
    oss_ds: str | None,
    now: datetime | None = None,
) -> tuple[str, str]:
    now = now or datetime.now()
    if odps_ds and oss_ds:
        return odps_ds, oss_ds
    if odps_ds:
        return odps_ds, shift_ds(odps_ds, days=1)
    if oss_ds:
        return shift_ds(oss_ds, days=-1), oss_ds
    return default_odps_ds(now), default_oss_ds(now)


def _export_local(sql: str, local_path: Path) -> bytes | None:
    try:
        columns, rows = execute_query(sql)
    except OdpsExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None
    if not rows:
        print("ERROR: 查询结果为空，跳过写产物（空文件覆盖保护）", file=sys.stderr)
        return None
    payload = rows_to_csv_bytes(columns, rows)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(payload)
    print(f"local: {local_path} rows={len(rows)} bytes={len(payload)}")
    return payload


async def _run_one_job(
    *,
    job: OdpsExportJob,
    odps_ds: str,
    oss_ds: str,
    object_key_override: str | None,
    output_dir: Path,
    no_upload: bool,
    dry_run: bool,
    sql_source: str,
) -> int:
    limit = int(job.limit or _DEFAULT_LIMIT)
    try:
        sql = render_sql(job.sql_template, odps_ds, limit, project_id=job.project_id)
    except ValueError as exc:
        print(f"ERROR: job={job.id} {exc}", file=sys.stderr)
        return 2

    object_key = resolve_object_key(
        oss_ds,
        job.oss_prefix,
        object_key_override,
        object_name=job.object_name,
    )
    stem = Path(job.object_name).stem or job.id
    # 本地固定路径 {output_dir}/{oss_prefix}/{object_name}：每次执行先删上一份，
    # 避免空结果/失败后误读旧导出
    prefix_dir = job.oss_prefix.strip("/")
    local_dir = output_dir / prefix_dir if prefix_dir else output_dir
    local_path = local_dir / job.object_name
    placeholders = "{ds}" in job.sql_template and "{limit}" in job.sql_template

    if dry_run:
        print(
            f"[dry-run] job={job.id} sql_source={sql_source} placeholders={placeholders} "
            f"odps_ds={odps_ds}(T-1) oss_ds={oss_ds}(T) limit={limit} "
            f"project_id={job.project_id!r}",
        )
        print(f"[dry-run] local={local_path}")
        print(f"[dry-run] object_key={object_key}")
        print("[dry-run] SQL:")
        print(sql)
        print("---")
        return 0

    print(
        f"export: job={job.id} sql_source={sql_source} placeholders={placeholders} "
        f"odps_ds={odps_ds} oss_ds={oss_ds} limit={limit} "
        f"langfuse_project_id={job.project_id!r} odps_project={_settings.odps_project!r}",
    )
    # 落盘渲染后 SQL，便于与 DataWorks / logview 逐字对照（空结果时必看）
    sql_path = output_dir / f"{stem}_{job.id}_{odps_ds}.sql"
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    sql_path.write_text(sql, encoding="utf-8")
    print(f"sql: {sql_path} chars={len(sql)}")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.unlink(missing_ok=True)

    if no_upload:
        payload = _export_local(sql, local_path)
        return 0 if payload is not None else 1

    try:
        result = await run_odps_export(
            OdpsExportRequest(
                sql_template=job.sql_template,
                odps_ds=odps_ds,
                oss_ds=oss_ds,
                limit=limit,
                oss_prefix=job.oss_prefix,
                object_key=object_key_override,
                object_name=job.object_name,
                job_id=job.id,
                project_id=job.project_id,
            ),
        )
    except (OdpsExportEmptyError, OdpsExportStorageError, OdpsExportError) as exc:
        print(f"ERROR: job={job.id} {exc}", file=sys.stderr)
        return 1

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(result.csv_bytes)
    print(f"local: {local_path}")
    print(f"oss: {result.oss_url}")
    print(
        f"DONE job={job.id} odps_ds={result.odps_ds} oss_ds={result.oss_ds} "
        f"rows={result.rows} -> {result.object_key}",
    )
    return 0


def _probe_count(rows: list[list[Any]]) -> int:
    """提取 probe 结果计数：COUNT 类取首行首列，sample 类取行数。"""
    if not rows:
        return 0
    first = rows[0][0] if rows[0] else None
    if isinstance(first, (int, float)):
        return int(first)
    return len(rows)


def _run_probes(*, odps_ds: str, output_dir: Path) -> int:
    """分阶段 COUNT/抽样，定位 overview 导出空结果卡在 raw 还是 JOIN。

    p0 段区分行/列可见性（Label Security 读出 NULL 时 p1 同样为 0），
    跑完由 diagnose_probe_results 汇总判读结论。
    """
    probes = build_overview_probe_sqls(odps_ds)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] odps_ds={odps_ds} steps={len(probes)}")
    rc = 0
    counts: dict[str, int | None] = {}
    for label, sql in probes:
        sql_path = output_dir / f"probe_{label}_{odps_ds}.sql"
        sql_path.write_text(sql, encoding="utf-8")
        print(f"[probe] --- {label} ---")
        print(f"[probe] sql={sql_path}")
        try:
            columns, rows = execute_query(sql)
        except OdpsExportEmptyError as exc:
            print(f"[probe] EMPTY {label}: {exc}", file=sys.stderr)
            counts[label] = None
            rc = 1
            continue
        except OdpsExportError as exc:
            print(f"[probe] ERROR {label}: {exc}", file=sys.stderr)
            counts[label] = None
            rc = 1
            continue
        counts[label] = _probe_count(rows)
        print(f"[probe] OK {label} count={counts[label]} columns={columns} rows={rows}")
    print(f"[probe] diagnosis: {diagnose_probe_results(counts)}")
    return rc


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Export ODPS T-1 partition to CSV and upload OSS (multi-job). "
        "Credentials: Normandy (AGENT_EVAL_ODPS_AONE_APP_NAME); "
        "OSS: RemoteConfig agent-eval.benchmark.oss.",
    )
    parser.add_argument("--odps-ds", type=str, default=None)
    parser.add_argument("--oss-ds", type=str, default=None)
    parser.add_argument("--ds", type=str, default=None, help="兼容旧参数：等同 --odps-ds")
    parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="覆盖 jobs[].project_id（注入 SQL {project_id}，过滤 $.record.project_id）",
    )
    parser.add_argument("--object-key", type=str, default=None)
    parser.add_argument(
        "--job",
        type=str,
        default=None,
        help="只跑指定 job id；不传则跑全部 enabled jobs",
    )
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-remote_config", action="store_true")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="不导出：跑 overview 分阶段 COUNT/抽样（定位空结果）",
    )
    args = parser.parse_args()

    if not 0 < args.limit <= MAX_LIMIT and args.limit != _DEFAULT_LIMIT:
        # allow default; validate when rendering
        pass

    try:
        odps_ds, oss_ds = _resolve_dates(
            odps_ds=args.odps_ds or args.ds,
            oss_ds=args.oss_ds,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "[export_odps] odps "
        f"app={_settings.odps_notification_app_name!r} "
        f"project={_settings.odps_project!r} "
        f"endpoint={_settings.odps_endpoint!r}",
        file=sys.stderr,
    )

    if args.probe:
        print(
            f"[export_odps] probe mode odps_ds={odps_ds} oss_ds={oss_ds}",
            file=sys.stderr,
        )
        return _run_probes(odps_ds=odps_ds, output_dir=args.output_dir)

    try:
        jobs, sql_source = _load_jobs(
            use_remote_config=not args.no_remote_config,
            job_id=args.job,
            limit=args.limit,
            prefix=args.prefix,
            project_id=args.project_id,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "[export_odps] remote_config "
        f"data_id={_settings.odps_export_remote_config_data_id!r} "
        f"group={_settings.odps_export_remote_config_group!r} "
        f"sql_source={sql_source} jobs={[j.id for j in jobs]}",
        file=sys.stderr,
    )
    if args.dry_run:
        print(f"[dry-run] upload={not args.no_upload}")

    # object-key 仅在单 job 时允许覆盖
    if args.object_key and len(jobs) > 1:
        print("ERROR: --object-key 仅可与单个 --job 联用", file=sys.stderr)
        return 2

    rc = 0
    for job in jobs:
        code = await _run_one_job(
            job=job,
            odps_ds=odps_ds,
            oss_ds=oss_ds,
            object_key_override=args.object_key,
            output_dir=args.output_dir,
            no_upload=args.no_upload,
            dry_run=args.dry_run,
            sql_source=sql_source,
        )
        if code != 0:
            rc = code
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
