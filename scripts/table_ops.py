#!/usr/bin/env python3
"""运维：表级数据/表结构操作（删指定 ID 行 / 清空表 / 删表 / 传建表 SQL 建表 / 加列 / 删列）。

六种子命令，均默认 dry-run，加 ``--execute`` 才真正执行：

1. ``delete-rows``：删除指定表、指定 ID 的行（ID 列名可配，缺省 ``id``）。
   ID 支持正整数主键与字符串业务 ID（如 ``run_id`` 的 UUID）。按 ID
   逐行计数预览，单次操作行数 < 10000（DB-RDB-003）。
2. ``truncate-table``：清空表数据但保留表结构（TRUNCATE TABLE）。需
   ``--execute`` 且 ``--force`` 双确认——数据不可恢复，生产环境优先走
   DMS 变更单（DB-RDB-003：TRUNCATE 线上表必须以工单方式执行）。
3. ``drop-table``：删除整张表（DROP TABLE）。需 ``--execute`` 且 ``--force``
   双确认——删表不可恢复，生产环境优先走 DMS 变更单（DB-RDB-003）。
4. ``create-table``：读取建表 SQL 文件（或 ``-`` 读 stdin）并逐语句执行。
   仅放行 CREATE TABLE / CREATE INDEX / CREATE UNIQUE INDEX /
   CREATE TYPE / COMMENT ON 语句，拒绝其余 DDL/DML（防误执行 DROP/UPDATE）。
5. ``add-column``：给已有表加一列（``ALTER TABLE ... ADD COLUMN
   IF NOT EXISTS``）。仅这一种 ALTER 形态：表/列名过标识符白名单，
   列类型过类型白名单（基础类型 + 可选长度/精度），不带 DEFAULT /
   NOT NULL 等附加子句；列已存在时幂等跳过。为加列型 schema 变更
   （如 benchmark_runs.deleted_at）提供可审计的落地通道，不再依赖
   人工直连数据库执行 ALTER。
6. ``drop-column``：从已有表删一列（``ALTER TABLE ... DROP COLUMN
   IF EXISTS``）。与 drop-table 同级防护：需 ``--execute`` 且
   ``--force`` 双确认，dry-run 先打印该列的类型与非空行数（将丢失的
   数据量）；列不存在时幂等跳过。主键列 ``id`` 直接拒绝（毁表走
   drop-table）。为删列型 schema 变更（如 runtime_config_records 去
   ``enabled`` 列）提供落地通道；删列不可恢复，生产环境优先走 DMS
   变更单（DB-RDB-003），且必须等旧版本实例全部下线后再执行。

表名与标识符一律校验 ``^[A-Za-z_][A-Za-z0-9_]*$``，拒绝拼接注入。
DELETE 恒带 WHERE（按 id 列过滤），满足 DB-RDB-003「禁止无 WHERE 删除」。

用法（优先包装脚本，线上会拼 agentgraph PYTHONPATH）::

  # 删指定表指定 ID 的行（dry-run 预览）
  bash scripts/table_ops.sh delete-rows --table runtime_config_records --ids 1,2,3
  # 确认执行
  bash scripts/table_ops.sh delete-rows --table runtime_config_records --ids 1,2,3 --execute

  # 字符串业务 ID（如 benchmark_runs.run_id 的 UUID）需指定 ID 列
  bash scripts/table_ops.sh delete-rows --table benchmark_runs \
    --id-column run_id --ids 4bc42a6a-610e-475c-96ae-ad8056607d99

  # 清空表数据但保留表结构（truncate 需 --execute --force 双确认）
  bash scripts/table_ops.sh truncate-table --table tmp_xxx --execute --force

  # 删整张表（drop 需 --execute --force 双确认）
  bash scripts/table_ops.sh drop-table --table tmp_xxx --execute --force

  # 按建表 SQL 文件建表（可多语句，分号分隔）
  bash scripts/table_ops.sh create-table --sql-file ./sql/tmp_table.sql --execute
  # 或从 stdin
  cat tmp.sql | bash scripts/table_ops.sh create-table --sql-file - --execute

  # 给已有表加列（先 dry-run 看现状，再 --execute 落地）
  bash scripts/table_ops.sh add-column --table benchmark_runs \
    --column deleted_at --type TIMESTAMPTZ
  bash scripts/table_ops.sh add-column --table benchmark_runs \
    --column deleted_at --type TIMESTAMPTZ --execute

  # 从已有表删一列（不可恢复，需 --execute --force 双确认；先 dry-run 看影响）
  bash scripts/table_ops.sh drop-column --table runtime_config_records \
    --column enabled
  bash scripts/table_ops.sh drop-column --table runtime_config_records \
    --column enabled --execute --force
"""

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _bootstrap  # sys.path/.env/禁 MQ 副作用，须先于 app.* 执行
from benchmark.settings import Settings
from benchmark.service.main import create_app

del _bootstrap  # 仅为副作用导入：已执行，释放模块名

#: Tortoise 连接对象（execute_query 提供方；仅用到鸭子类型，不引具体类避免循环依赖）。
_Conn = Any

#: DB-RDB-003：单次操作行数必须小于 10000；达到上限拒绝删除。
_MAX_DELETE_ROWS = 10000

#: 表名 / 列名等标识符白名单（防拼接注入）。
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: 字符串业务 ID 最大长度（UUID / run_id；SQL 已参数化，长度仅防误输）。
_MAX_ID_LENGTH = 128

#: create-table 放行的语句前缀（白名单外 DDL/DML 一律拒绝）。
_CREATE_ALLOWED_PREFIXES = (
    "create table",
    "create unique index",
    "create index",
    "create type",
    "comment on",
)

#: create-table 拒绝的高危关键字（出现在语句内即拒绝，防白名单前缀伪装）。
_FORBIDDEN_KEYWORDS = ("drop ", "delete ", "truncate ", "update ", "insert ", "alter ")

#: add-column 列类型白名单形态：基础类型（可多词，如 DOUBLE PRECISION /
#: TIMESTAMP WITH TIME ZONE）+ 可选 (n) / (n,m) 长度精度；大写归一后校验，
#: 禁止任何子句/表达式（DEFAULT / NOT NULL / 引用等一律不带）。
_COLUMN_TYPE_RE = re.compile(
    r"^[A-Z][A-Z0-9_]*( [A-Z][A-Z0-9_]*)*(\([0-9]+(,[0-9]+)?\))?$",
)

#: 列类型中禁止出现的子句关键字（类型名多词形态会放行 TEXT NOT NULL 这类
#: 伪装，必须显式拒绝；WITH TIME ZONE 等合法多词类型不含这些词）。
_COLUMN_TYPE_FORBIDDEN_WORDS = frozenset(
    {"NOT", "NULL", "DEFAULT", "REFERENCES", "PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT"},
)


def _check_ident(name: str, label: str) -> str:
    """校验标识符合法（字母/下划线/数字），不合法直接退出。"""
    if not _IDENT_RE.match(name):
        print(f"ERROR: illegal {label}: {name!r}", file=sys.stderr)
        raise SystemExit(2)
    return name


def _check_column_type(raw: str) -> str:
    """校验并归一列类型（大写、单空格），不合法直接退出。

    只接受「类型名 [+ 长度/精度]」形态，拒绝分号、引号、子句关键字，
    保证拼出的 ADD COLUMN 语句只有加列一种语义。
    """
    normalized = " ".join(raw.strip().upper().split())
    if not _COLUMN_TYPE_RE.match(normalized) or (
        _COLUMN_TYPE_FORBIDDEN_WORDS & set(normalized.split())
    ):
        print(f"ERROR: illegal column type: {raw!r}", file=sys.stderr)
        raise SystemExit(2)
    return normalized


def _parse_ids(raw: str) -> list[int | str]:
    """解析逗号分隔的 ID 列表：纯数字转 int，其余按字符串业务 ID（UUID 等）。

    SQL 一律参数化（``$1..$n`` 绑定），字符串 ID 不会拼进 SQL；此处只做
    非空/长度/控制字符校验，防止误输空串或换行造成不可见条件。
    """
    ids: list[int | str] = []
    for piece in raw.split(","):
        part = piece.strip()
        if not part:
            continue
        if part.isdigit():
            ids.append(int(part))
            continue
        if len(part) > _MAX_ID_LENGTH or any(ch.isspace() or ord(ch) < 32 for ch in part):
            print(f"ERROR: illegal id: {part!r}", file=sys.stderr)
            raise SystemExit(2)
        ids.append(part)
    if not ids:
        print("ERROR: --ids is empty", file=sys.stderr)
        raise SystemExit(2)
    if len(ids) >= _MAX_DELETE_ROWS:
        print(
            f"ERROR: ids count {len(ids)} >= {_MAX_DELETE_ROWS}"
            "（DB-RDB-003 单次操作行数上限），请分批删除",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return ids


def _split_statements(sql_text: str) -> list[str]:
    """把 SQL 文本按分号拆成语句（去注释与空白；不解析字符串内分号——
    建表 DDL 不含数据字面量，此简化可接受）。"""
    lines: list[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)
    joined = "\n".join(lines)
    return [stmt.strip() for stmt in joined.split(";") if stmt.strip()]


def _check_statements_allowed(statements: list[str]) -> None:
    """白名单校验：每条语句必须以放行前缀开头，且不含高危关键字。"""
    for stmt in statements:
        lowered = stmt.lower()
        if not lowered.startswith(_CREATE_ALLOWED_PREFIXES):
            print(
                f"ERROR: statement not allowed (only CREATE TABLE/INDEX/TYPE "
                f"and COMMENT ON):\n  {stmt[:120]}...",
                file=sys.stderr,
            )
            raise SystemExit(2)
        for kw in _FORBIDDEN_KEYWORDS:
            if kw in lowered:
                print(
                    f"ERROR: forbidden keyword {kw.strip()!r} in statement:\n  {stmt[:120]}...",
                    file=sys.stderr,
                )
                raise SystemExit(2)


@dataclass(frozen=True)
class _RowsQuery:
    """按 ID 删行的一次操作描述（表 + ID 列 + 待删 ID 集合，已过标识符校验）。"""

    table: str
    id_column: str
    ids: list[int | str]


async def _delete_rows(conn: _Conn, query: _RowsQuery, *, execute: bool) -> int:
    """删除指定表、指定 ID 的行（整数主键或字符串业务 ID）。DELETE 恒带 WHERE。"""
    table = query.table
    id_column = query.id_column
    ids = query.ids
    # 表存在性检查（不存在直接退出，避免拼出无效 SQL）。
    # 注：asyncpg 后端占位符为 $1/$2（PostgreSQL 原生），非 %s；
    # execute_query 返回 (affected, rows) 元组——rows 在第二位。
    _, exists_rows = await conn.execute_query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1) AS exists_",
        [table],
    )
    if not exists_rows[0][0]:
        print(f"ERROR: table not found: {table}", file=sys.stderr)
        return 1
    # id 列存在性检查
    _, col_rows = await conn.execute_query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = $1 AND column_name = $2) AS exists_",
        [table, id_column],
    )
    if not col_rows[0][0]:
        print(
            f"ERROR: column {id_column!r} not found on table {table}",
            file=sys.stderr,
        )
        return 1

    placeholders = ", ".join([f"${i + 1}" for i in range(len(ids))])
    _, found = await conn.execute_query(
        f"SELECT {id_column} FROM {table} WHERE {id_column} IN ({placeholders})",
        ids,
    )
    found_ids = [row[0] for row in found]
    missing = [i for i in ids if i not in found_ids]
    print(f"Target: table={table} id_column={id_column}")
    print(f"  ids requested: {len(ids)}, found: {len(found_ids)}, missing: {missing or 'none'}")
    if not found_ids:
        print("\n[dry-run] nothing to delete.")
        return 0
    if not execute:
        print(f"\n[dry-run] would delete {len(found_ids)} row(s). Add --execute to apply.")
        return 0
    deleted, _ = await conn.execute_query(
        f"DELETE FROM {table} WHERE {id_column} IN ({placeholders})",
        ids,
    )
    # affected 行数在元组首位（不同后端返回形态已归一为元组）
    print(f"  Deleted {deleted} row(s) from {table}")
    print("\nDone.")
    return 0


async def _truncate_table(conn: _Conn, table: str, *, execute: bool, force: bool) -> int:
    """TRUNCATE TABLE（清空数据保留表结构；--execute --force 双确认）。"""
    # 注：asyncpg 后端占位符为 $1（PostgreSQL 原生），非 %s；
    # execute_query 返回 (affected, rows) 元组——rows 在第二位。
    _, exists_rows = await conn.execute_query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1) AS exists_",
        [table],
    )
    if not exists_rows[0][0]:
        print(f"ERROR: table not found: {table}", file=sys.stderr)
        return 1
    _, count = await conn.execute_query(f"SELECT COUNT(*) AS c FROM {table}")
    rows = count[0][0]
    print(f"Target: TRUNCATE TABLE {table}")
    print(f"  current rows: {rows}")
    print("  WARNING: 数据清空不可恢复（表结构保留）；生产环境优先走 DMS 变更单（DB-RDB-003）")
    if not execute or not force:
        print("\n[dry-run] no changes made. Add --execute --force to apply.")
        return 0
    await conn.execute_query(f'TRUNCATE TABLE "{table}"')
    print(f"  Truncated table {table} (structure kept)")
    print("\nDone.")
    return 0


async def _drop_table(conn: _Conn, table: str, *, execute: bool, force: bool) -> int:
    """DROP TABLE（--execute --force 双确认）。"""
    # 注：asyncpg 后端占位符为 $1（PostgreSQL 原生），非 %s；
    # execute_query 返回 (affected, rows) 元组——rows 在第二位。
    _, exists_rows = await conn.execute_query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1) AS exists_",
        [table],
    )
    if not exists_rows[0][0]:
        print(f"ERROR: table not found: {table}", file=sys.stderr)
        return 1
    _, count = await conn.execute_query(f"SELECT COUNT(*) AS c FROM {table}")
    rows = count[0][0]
    print(f"Target: DROP TABLE {table}")
    print(f"  current rows: {rows}")
    print("  WARNING: DROP TABLE 不可恢复；生产环境优先走 DMS 变更单（DB-RDB-003）")
    if not execute or not force:
        print("\n[dry-run] no changes made. Add --execute --force to apply.")
        return 0
    await conn.execute_query(f'DROP TABLE IF EXISTS "{table}"')
    print(f"  Dropped table {table}")
    print("\nDone.")
    return 0


async def _create_table(
    conn: _Conn,
    statements: list[str],
    *,
    source: str,
    execute: bool,
) -> int:
    """按已校验的语句列表建表（白名单校验在 CLI 层提前完成）。"""
    print(f"Target: {len(statements)} statement(s) from {source}")
    for i, stmt in enumerate(statements, 1):
        preview = " ".join(stmt.split())
        print(f"  [{i}] {preview[:100]}{'…' if len(preview) > 100 else ''}")
    if not execute:
        print("\n[dry-run] no changes made. Add --execute to apply.")
        return 0
    for i, stmt in enumerate(statements, 1):
        await conn.execute_query(stmt)
        preview = " ".join(stmt.split())
        print(f"  [{i}] executed: {preview[:80]}{'…' if len(preview) > 80 else ''}")
    print("\nDone.")
    return 0


@dataclass(frozen=True)
class _ColumnTarget:
    """一次列结构变更的目标列（表 + 列 + 计划列类型；删列时 column_type 为 None）。"""

    table: str
    column: str
    column_type: str | None = None


async def _add_column(conn: _Conn, target: _ColumnTarget, *, execute: bool) -> int:
    """给已有表加一列（ADD COLUMN IF NOT EXISTS；列已存在幂等跳过）。

    只加可空、无默认值的列：在线加列不锁表、不回填存量行。为
    benchmark_runs.deleted_at 这类软删除标记列的落地通道。
    """
    table = target.table
    column = target.column
    column_type = target.column_type
    assert column_type is not None  # 加列必须过类型白名单（CLI 层保证）
    _, exists_rows = await conn.execute_query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1) AS exists_",
        [table],
    )
    if not exists_rows[0][0]:
        print(f"ERROR: table not found: {table}", file=sys.stderr)
        return 1
    _, col_rows = await conn.execute_query(
        "SELECT data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = $1 AND column_name = $2",
        [table, column],
    )
    print(f"Target: ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {column_type}")
    if col_rows:
        data_type, is_nullable = col_rows[0][0], col_rows[0][1]
        print(f"  column already exists: type={data_type} nullable={is_nullable} — no-op")
        print("\nDone.")
        return 0
    print("  column missing")
    if not execute:
        print("\n[dry-run] no changes made. Add --execute to apply.")
        return 0
    await conn.execute_query(
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {column_type}",
    )
    _, verify_rows = await conn.execute_query(
        "SELECT data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = $1 AND column_name = $2",
        [table, column],
    )
    if not verify_rows:
        print("ERROR: column still missing after ALTER", file=sys.stderr)
        return 1
    print(
        f"  column added: type={verify_rows[0][0]} nullable={verify_rows[0][1]}",
    )
    print("\nDone.")
    return 0


async def _drop_column(conn: _Conn, target: _ColumnTarget, *, execute: bool, force: bool) -> int:
    """从已有表删一列（DROP COLUMN IF EXISTS；--execute --force 双确认）。

    与 drop-table 同级的不可逆操作：删前打印列类型与非空行数（将永久
    丢失的数据量），删后复查 information_schema 确认列已消失。列不存在
    时幂等跳过。主键列 ``id`` 由 CLI 层直接拒绝（毁表走 drop-table）。
    """
    table = target.table
    column = target.column
    # 注：asyncpg 后端占位符为 $1（PostgreSQL 原生），非 %s；
    # execute_query 返回 (affected, rows) 元组——rows 在第二位。
    _, exists_rows = await conn.execute_query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1) AS exists_",
        [table],
    )
    if not exists_rows[0][0]:
        print(f"ERROR: table not found: {table}", file=sys.stderr)
        return 1
    _, col_rows = await conn.execute_query(
        "SELECT data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = $1 AND column_name = $2",
        [table, column],
    )
    print(f"Target: ALTER TABLE {table} DROP COLUMN IF EXISTS {column}")
    print(
        "  WARNING: DROP COLUMN 不可恢复，该列的现存数据将永久丢失；"
        "生产环境优先走 DMS 变更单（DB-RDB-003）"
    )
    if not col_rows:
        print("  column already missing — no-op")
        print("\nDone.")
        return 0
    print(
        f"  column exists: type={col_rows[0][0]} nullable={col_rows[0][1]}",
    )
    _, count_rows = await conn.execute_query(
        f"SELECT COUNT(*) FROM {table} WHERE {column} IS NOT NULL",
    )
    print(f"  rows with non-null value (will lose data): {count_rows[0][0]}")
    if not execute or not force:
        print("\n[dry-run] no changes made. Add --execute --force to apply.")
        return 0
    await conn.execute_query(f'ALTER TABLE "{table}" DROP COLUMN IF EXISTS "{column}"')
    _, verify_rows = await conn.execute_query(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = $1 AND column_name = $2) AS exists_",
        [table, column],
    )
    if verify_rows[0][0]:
        print("ERROR: column still exists after ALTER", file=sys.stderr)
        return 1
    print(f"  column dropped: {table}.{column}")
    print("\nDone.")
    return 0


async def _amain(args: argparse.Namespace) -> int:
    settings = Settings()
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        # lifespan 会拉起 worker/调度器；只取默认连接后立即走操作路径。
        from tortoise import connections

        conn = connections.get("default")
        if args.command == "add-column":
            return await _add_column(
                conn,
                _ColumnTarget(table=args.table, column=args.column, column_type=args.column_type),
                execute=args.execute,
            )
        if args.command == "drop-column":
            return await _drop_column(
                conn,
                _ColumnTarget(table=args.table, column=args.column),
                execute=args.execute,
                force=args.force,
            )
        if args.command == "delete-rows":
            return await _delete_rows(
                conn,
                _RowsQuery(table=args.table, id_column=args.id_column, ids=args.ids),
                execute=args.execute,
            )
        if args.command == "truncate-table":
            return await _truncate_table(
                conn,
                args.table,
                execute=args.execute,
                force=args.force,
            )
        if args.command == "drop-table":
            return await _drop_table(
                conn,
                args.table,
                execute=args.execute,
                force=args.force,
            )
        return await _create_table(
            conn,
            args.sql_statements,
            source=args.sql_file,
            execute=args.execute,
        )


def _validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """CLI 层前置校验（不依赖 DB；非法参数立即退出，不进 lifespan）。

    create-table 且非 execute：dry-run 预览不触库，直接在此输出并退出
    （--execute 才进 lifespan 连 DB 执行）。
    """
    if args.command == "delete-rows":
        args.table = _check_ident(args.table, "table name")
        args.id_column = _check_ident(args.id_column, "id column")
        args.ids = _parse_ids(args.ids)
    elif args.command == "add-column":
        args.table = _check_ident(args.table, "table name")
        args.column = _check_ident(args.column, "column name")
        args.column_type = _check_column_type(args.column_type)
    elif args.command == "drop-column":
        args.table = _check_ident(args.table, "table name")
        args.column = _check_ident(args.column, "column name")
        if args.column.lower() == "id":
            # 删主键列等于毁表（级联约束/索引无从处理），必须走 drop-table 语义
            print(
                "ERROR: refusing to drop primary-key column 'id'; "
                "use drop-table if the whole table should go",
                file=sys.stderr,
            )
            raise SystemExit(2)
    elif args.command in ("drop-table", "truncate-table"):
        args.table = _check_ident(args.table, "table name")
    else:
        # create-table：SQL 白名单校验提前（读文件/stdin 在 lifespan 前完成）。
        path = args.sql_file.strip()
        if path == "-":
            sql_text = sys.stdin.read()
        else:
            p = Path(path)
            if not p.is_file():
                print(f"ERROR: sql file not found: {path}", file=sys.stderr)
                raise SystemExit(2)
            sql_text = p.read_text(encoding="utf-8")
        statements = _split_statements(sql_text)
        if not statements:
            print("ERROR: no statements found in input", file=sys.stderr)
            raise SystemExit(2)
        _check_statements_allowed(statements)
        args.sql_file = path
        args.sql_statements = statements
        if not args.execute:
            print(f"Target: {len(statements)} statement(s) from {path}")
            for i, stmt in enumerate(statements, 1):
                preview = " ".join(stmt.split())
                print(f"  [{i}] {preview[:100]}{'…' if len(preview) > 100 else ''}")
            print("\n[dry-run] no changes made. Add --execute to apply.")
            raise SystemExit(0)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Table-level ops: delete rows by id / drop table / add-drop "
        "column / create table from SQL file (PG; dry-run by default)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_del = sub.add_parser("delete-rows", help="delete rows by primary ids")
    p_del.add_argument("--table", type=str, required=True, help="table name")
    p_del.add_argument("--ids", type=str, required=True, help="comma-separated ids")
    p_del.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="id column name (default: id)",
    )
    p_del.add_argument("--execute", action="store_true", help="actually execute")

    p_trunc = sub.add_parser("truncate-table", help="truncate: clear rows, keep table")
    p_trunc.add_argument("--table", type=str, required=True, help="table name")
    p_trunc.add_argument("--execute", action="store_true", help="actually execute")
    p_trunc.add_argument(
        "--force",
        action="store_true",
        help="double-confirm for TRUNCATE TABLE (required with --execute)",
    )

    p_drop = sub.add_parser("drop-table", help="drop an entire table")
    p_drop.add_argument("--table", type=str, required=True, help="table name")
    p_drop.add_argument("--execute", action="store_true", help="actually execute")
    p_drop.add_argument(
        "--force",
        action="store_true",
        help="double-confirm for DROP TABLE (required with --execute)",
    )

    p_create = sub.add_parser("create-table", help="run CREATE TABLE sql file")
    p_create.add_argument(
        "--sql-file",
        type=str,
        required=True,
        help="path to .sql file, or '-' for stdin",
    )
    p_create.add_argument("--execute", action="store_true", help="actually execute")

    p_addcol = sub.add_parser(
        "add-column",
        help="ALTER TABLE ADD COLUMN IF NOT EXISTS (one nullable column)",
    )
    p_addcol.add_argument("--table", type=str, required=True, help="table name")
    p_addcol.add_argument("--column", type=str, required=True, help="column name")
    p_addcol.add_argument(
        "--type",
        dest="column_type",
        type=str,
        required=True,
        help="column type, e.g. TIMESTAMPTZ / TEXT / VARCHAR(128)",
    )
    p_addcol.add_argument("--execute", action="store_true", help="actually execute")

    p_dropcol = sub.add_parser(
        "drop-column",
        help="ALTER TABLE DROP COLUMN IF EXISTS (one column; irreversible)",
    )
    p_dropcol.add_argument("--table", type=str, required=True, help="table name")
    p_dropcol.add_argument("--column", type=str, required=True, help="column name")
    p_dropcol.add_argument("--execute", action="store_true", help="actually execute")
    p_dropcol.add_argument(
        "--force",
        action="store_true",
        help="double-confirm for DROP COLUMN (required with --execute)",
    )

    args = _validate_args(parser.parse_args())
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
