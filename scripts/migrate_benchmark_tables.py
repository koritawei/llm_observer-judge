#!/usr/bin/env python3
"""把 bullMQ 分支的 benchmark 表数据迁入 main 分支的表结构。

源表（bullMQ 分支 ORM）:
- ``benchmark_runs_bullmq``
- ``benchmark_run_items_bullmq``

目标表（main 分支 ORM / ``sql/schema.sql``）:
- ``benchmark_runs``
- ``benchmark_run_items``

两边模型列一致，仅表名不同。运行时仍按 information_schema 取列交集：
源表多出的列跳过并打印；目标表 NOT NULL 且无默认值、源表又没有的列直接失败。
只插入目标表还没有的行，已有数据不更新：
runs 以 ``run_id`` 判重，items 以 ``run_id, item_id, criterion_id`` 判重；
``id`` 已在目标表中的行同样跳过。``ON CONFLICT DO NOTHING`` 兜底，避免改写已有行。
可重复执行。单批行数 < 10000（DB-RDB-003），且同时满足
PostgreSQL 扩展协议单语句绑定参数 ≤ 65535（``行数 × 列数``）。

两张表所在库地址不同，必须分别传入 ``--source-url``（bullMQ 库）和
``--target-url``（main 库）。脚本各开一条 psycopg3 连接：从目标库读取已有键，
再从源库分页拉行，只把 diff 写回目标库。

``benchmark_kafka_tasks`` / ``benchmark_report_providers`` 在两条分支上同名同结构，不迁移。

用法（默认 dry-run；加 ``--execute`` 才写库）::

  bash scripts/migrate_benchmark_tables.sh \\
    --source-url 'postgresql://USER:PASSWORD@bullmq-host:5432/agent_eval' \\
    --target-url 'postgresql://USER:PASSWORD@main-host:5432/agent_eval'
  bash scripts/migrate_benchmark_tables.sh \\
    --source-url 'postgresql://USER:PASSWORD@bullmq-host:5432/agent_eval' \\
    --target-url 'postgresql://USER:PASSWORD@main-host:5432/agent_eval' \\
    --execute
"""

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import _bootstrap  # .env + 禁 MQ，须先于后续可选 app 导入执行

del _bootstrap

_BATCH_LIMIT = 9999
# psycopg3 默认扩展协议：单条语句绑定参数上限（PostgreSQL）。
_PG_BIND_PARAM_LIMIT = 65535
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_POSTGRES_PREFIXES = (
    "postgresql+psycopg://",
    "postgresql+asyncpg://",
    "postgres+psycopg://",
)


def insert_batch_row_limit(column_count: int) -> int:
    """单批最大行数：同时满足 DB-RDB-003（<10000）与 PG 绑定参数上限。"""
    if column_count <= 0:
        raise ValueError(f"列数必须为正整数: {column_count}")
    by_params = max(1, _PG_BIND_PARAM_LIMIT // column_count)
    return min(_BATCH_LIMIT, by_params)


@dataclass(frozen=True)
class TablePair:
    """一对源表 / 目标表，以及用来判断“已迁过”的冲突列。"""

    source: str
    target: str
    conflict_columns: tuple[str, ...]


@dataclass(frozen=True)
class PartitionColumns:
    """一页源行做判重切分所需的列布局（全体列 + 冲突列）。"""

    columns: Sequence[str]
    conflict_columns: Sequence[str]


@dataclass(frozen=True)
class ColumnInfo:
    """information_schema 里迁移需要的列元数据。"""

    name: str
    nullable: bool
    has_default: bool


class _Cursor(Protocol):
    """psycopg cursor 里本脚本用到的方法。"""

    rowcount: int

    def execute(self, query: str, params: object = None) -> object:
        """执行 SQL。"""

    def fetchone(self) -> tuple[object, ...] | None:
        """取一行。"""

    def fetchall(self) -> list[tuple[object, ...]]:
        """取全部行。"""

    def __enter__(self) -> "_Cursor":
        """进入上下文。"""

    def __exit__(self, *args: object) -> None:
        """退出上下文。"""


class _Transaction(Protocol):
    """psycopg transaction 上下文。"""

    def __enter__(self) -> object:
        """开始事务。"""

    def __exit__(self, *args: object) -> None:
        """提交或回滚。"""


class _Conn(Protocol):
    """psycopg connection 里本脚本用到的方法。"""

    def cursor(self) -> _Cursor:
        """打开游标。"""

    def transaction(self) -> _Transaction:
        """打开事务。"""


#: bullMQ 表 → main 表。先 runs 再 items，保证明细的 run_id 先落到目标表。
BENCHMARK_TABLES: tuple[TablePair, ...] = (
    TablePair(
        source="benchmark_runs_bullmq",
        target="benchmark_runs",
        conflict_columns=("run_id",),
    ),
    TablePair(
        source="benchmark_run_items_bullmq",
        target="benchmark_run_items",
        conflict_columns=("run_id", "item_id", "criterion_id"),
    ),
)


def quote_ident(name: str) -> str:
    """只允许普通 SQL 标识符，再加双引号。"""
    if _IDENT_RE.fullmatch(name) is None:
        raise ValueError(f"非法标识符: {name!r}")
    return f'"{name}"'


def normalize_postgres_url(url: str) -> str:
    """把 SQLAlchemy / tortoise 风格的 PostgreSQL URL 收成 psycopg3 能连的形式。"""
    stripped = url.strip()
    for prefix in _POSTGRES_PREFIXES:
        if stripped.startswith(prefix):
            return "postgresql://" + stripped[len(prefix) :]
    if stripped.startswith("postgres://"):
        return "postgresql://" + stripped[len("postgres://") :]
    if stripped.startswith("postgresql://"):
        return stripped
    raise ValueError("数据库地址必须是 PostgreSQL 连接串")


def plan_copy_columns(
    source_columns: Sequence[ColumnInfo],
    target_columns: Sequence[ColumnInfo],
) -> list[str]:
    """按目标表列顺序返回可复制的列名。

    目标表上 NOT NULL 且没有默认值、源表又不存在的列无法填，直接失败。
    """
    source_names = {column.name for column in source_columns}
    missing = [
        column.name
        for column in target_columns
        if column.name not in source_names and not column.nullable and not column.has_default
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"目标表必填列在源表中不存在: {joined}")
    return [column.name for column in target_columns if column.name in source_names]


def _require_conflict_columns(pair: TablePair, columns: Sequence[str]) -> None:
    if not columns:
        raise ValueError(f"{pair.source} -> {pair.target} 没有可复制的列")
    missing_keys = [name for name in pair.conflict_columns if name not in columns]
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(f"冲突列不在可复制列中: {joined}")


def _distinct_order(pair: TablePair, columns: Sequence[str]) -> str:
    """DISTINCT ON 的 ORDER BY：冲突列在前，再用 updated_at/created_at/id 取较新行。"""
    parts = [f"s.{quote_ident(name)}" for name in pair.conflict_columns]
    for name in ("updated_at", "created_at", "id"):
        if name in columns and name not in pair.conflict_columns:
            parts.append(f"s.{quote_ident(name)} DESC NULLS LAST")
            break
    return ", ".join(parts)


def build_target_keys_sql(pair: TablePair) -> str:
    """从目标库取出已有冲突键。"""
    cols = ", ".join(quote_ident(name) for name in pair.conflict_columns)
    return f"SELECT {cols} FROM {quote_ident(pair.target)}"


def build_target_ids_sql(table: str) -> str:
    """从目标库取出已有主键，避免跨库插入时撞上 id。"""
    quoted = quote_ident(table)
    ident = quote_ident("id")
    return f"SELECT {ident} FROM {quoted} WHERE {ident} IS NOT NULL"


def build_source_page_sql(pair: TablePair, columns: Sequence[str], *, has_cursor: bool) -> str:
    """从源库按冲突键分页读取。``has_cursor`` 时追加键比较，避免 OFFSET。"""
    _require_conflict_columns(pair, columns)
    source = quote_ident(pair.source)
    distinct = ", ".join(f"s.{quote_ident(name)}" for name in pair.conflict_columns)
    selected = ", ".join(f"s.{quote_ident(name)}" for name in columns)
    where = ""
    if has_cursor:
        key_cols = ", ".join(f"s.{quote_ident(name)}" for name in pair.conflict_columns)
        placeholders = ", ".join(["%s"] * len(pair.conflict_columns))
        where = f"WHERE ({key_cols}) > ({placeholders}) "
    return (
        f"SELECT DISTINCT ON ({distinct}) {selected} "
        f"FROM {source} AS s "
        f"{where}"
        f"ORDER BY {_distinct_order(pair, columns)} "
        "LIMIT %s"
    )


def build_insert_values_sql(target: str, columns: Sequence[str], row_count: int) -> str:
    """生成只写目标库的批量 INSERT。已有唯一键由 ``ON CONFLICT DO NOTHING`` 跳过。"""
    if not columns:
        raise ValueError(f"{target} 没有可插入的列")
    max_rows = insert_batch_row_limit(len(columns))
    if row_count <= 0 or row_count > max_rows:
        raise ValueError(
            f"单次插入行数必须在 1..{max_rows} 之间"
            f"（列数={len(columns)}，PG 绑定参数上限 {_PG_BIND_PARAM_LIMIT}）: {row_count}",
        )
    col_sql = ", ".join(quote_ident(name) for name in columns)
    one_row = "(" + ", ".join(["%s"] * len(columns)) + ")"
    values = ", ".join([one_row] * row_count)
    return f"INSERT INTO {quote_ident(target)} ({col_sql}) VALUES {values} ON CONFLICT DO NOTHING"


def partition_new_rows(
    rows: Sequence[Sequence[object]],
    partition_columns: PartitionColumns,
    existing_keys: set[tuple[object, ...]],
    existing_ids: set[object],
) -> tuple[list[Sequence[object]], list[Sequence[object]]]:
    """把一页源行分成已存在（跳过）和 diff（待插入）。不修改传入的集合。"""
    seen_keys = set(existing_keys)
    seen_ids = set(existing_ids)
    columns = partition_columns.columns
    conflict_columns = partition_columns.conflict_columns
    key_indexes = [columns.index(name) for name in conflict_columns]
    id_index = columns.index("id") if "id" in columns and "id" not in conflict_columns else None
    skipped: list[Sequence[object]] = []
    fresh: list[Sequence[object]] = []
    for row in rows:
        key = tuple(row[index] for index in key_indexes)
        row_id = row[id_index] if id_index is not None else None
        if key in seen_keys or (row_id is not None and row_id in seen_ids):
            skipped.append(row)
            continue
        fresh.append(row)
        seen_keys.add(key)
        if row_id is not None:
            seen_ids.add(row_id)
    return skipped, fresh


def _load_columns(conn: _Conn, table: str) -> list[ColumnInfo]:
    """读取 search_path 上该表的列。表不存在时抛出 ValueError。"""
    quote_ident(table)
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        row = cur.fetchone()
        if row is None or row[0] is None:
            raise ValueError(f"表不存在: {table}")
        cur.execute(
            """
            SELECT n.nspname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.oid = to_regclass(%s)
            """,
            (table,),
        )
        schema_row = cur.fetchone()
        if schema_row is None:
            raise ValueError(f"表不存在: {table}")
        schema = schema_row[0]
        cur.execute(
            """
            SELECT column_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (schema, table),
        )
        rows = cur.fetchall()
    return [
        ColumnInfo(
            name=name,
            nullable=nullable == "YES",
            has_default=default is not None,
        )
        for name, nullable, default in rows
    ]


def _scalar(conn: _Conn, sql: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
    if row is None:
        return 0
    return int(row[0])


def _fetch_all(conn: _Conn, sql: str) -> list[tuple[object, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql)
        return [tuple(row) for row in cur.fetchall()]


def _adapt_cell(value: object) -> object:
    """dict/list 按 JSON 写回，避免 psycopg 把 list 当成 PG 数组。"""
    if isinstance(value, (dict, list)):
        from psycopg.types.json import Jsonb

        return Jsonb(value)
    return value


def _insert_rows(
    target: _Conn,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> int:
    if not rows:
        return 0
    batch_rows = insert_batch_row_limit(len(columns))
    inserted = 0
    with target.transaction():
        with target.cursor() as cur:
            for start in range(0, len(rows), batch_rows):
                chunk = rows[start : start + batch_rows]
                sql = build_insert_values_sql(table, columns, len(chunk))
                params = tuple(_adapt_cell(cell) for row in chunk for cell in row)
                cur.execute(sql, params)
                inserted += int(cur.rowcount)
    return inserted


def migrate_table(source: _Conn, target: _Conn, pair: TablePair, *, execute: bool) -> int:
    """跨库只插入目标表没有的行。返回实际插入行数；dry-run 返回 0。"""
    source_columns = _load_columns(source, pair.source)
    target_columns = _load_columns(target, pair.target)
    columns = plan_copy_columns(source_columns, target_columns)
    _require_conflict_columns(pair, columns)
    skipped_columns = [column.name for column in source_columns if column.name not in columns]
    source_rows = _scalar(source, f"SELECT count(*) FROM {quote_ident(pair.source)}")
    existing_keys = {tuple(row) for row in _fetch_all(target, build_target_keys_sql(pair))}
    track_ids = "id" in columns and "id" not in pair.conflict_columns
    existing_ids = (
        {row[0] for row in _fetch_all(target, build_target_ids_sql(pair.target))}
        if track_ids
        else set()
    )

    already = 0
    diff = 0
    moved = 0
    page_limit = insert_batch_row_limit(len(columns))
    cursor_key: tuple[object, ...] | None = None
    while True:
        page_sql = build_source_page_sql(pair, columns, has_cursor=cursor_key is not None)
        params: tuple[object, ...]
        if cursor_key is None:
            params = (page_limit,)
        else:
            params = (*cursor_key, page_limit)
        with source.cursor() as cur:
            cur.execute(page_sql, params)
            page = [tuple(row) for row in cur.fetchall()]
        if not page:
            break
        key_indexes = [columns.index(name) for name in pair.conflict_columns]
        cursor_key = tuple(page[-1][index] for index in key_indexes)
        skipped, fresh = partition_new_rows(
            page,
            PartitionColumns(columns=columns, conflict_columns=pair.conflict_columns),
            existing_keys,
            existing_ids,
        )
        already += len(skipped)
        diff += len(fresh)
        for row in fresh:
            existing_keys.add(tuple(row[index] for index in key_indexes))
            if track_ids:
                existing_ids.add(row[columns.index("id")])
        if execute and fresh:
            inserted = _insert_rows(target, pair.target, columns, fresh)
            moved += inserted
            print(f"  inserted {inserted} (total {moved})")
        if len(page) < page_limit:
            break

    print(
        f"{pair.source} -> {pair.target}: "
        f"source={source_rows} already={already} diff={diff} "
        f"columns={len(columns)} skipped_columns={skipped_columns or '-'}",
    )
    if diff == 0:
        print("  existing rows skipped, nothing to move")
    return moved if execute else 0


def migrate_benchmark_tables(source: _Conn, target: _Conn, *, execute: bool) -> int:
    """按 runs → items 的顺序迁移。返回插入总行数。"""
    total = 0
    for pair in BENCHMARK_TABLES:
        total += migrate_table(source, target, pair, execute=execute)
    return total


def _parse_url(flag: str, raw_url: str) -> str:
    try:
        return normalize_postgres_url(raw_url)
    except ValueError as exc:
        print(f"ERROR: {flag}: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy missing benchmark rows from a bullMQ database into a main database. "
            "Existing keys and ids are skipped (psycopg3)."
        ),
    )
    parser.add_argument(
        "--source-url",
        required=True,
        help="bullMQ PostgreSQL URL (benchmark_runs_bullmq / benchmark_run_items_bullmq)",
    )
    parser.add_argument(
        "--target-url",
        required=True,
        help="main PostgreSQL URL (benchmark_runs / benchmark_run_items)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually insert rows into the target database (default: dry-run)",
    )
    args = parser.parse_args()
    source_url = _parse_url("--source-url", args.source_url)
    target_url = _parse_url("--target-url", args.target_url)

    import psycopg

    with (
        psycopg.connect(source_url, autocommit=True) as source,
        psycopg.connect(target_url, autocommit=True) as target,
    ):
        try:
            inserted = migrate_benchmark_tables(source, target, execute=args.execute)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    if args.execute:
        print(f"\nDone. inserted={inserted}")
        return
    print("\n[dry-run] no changes made. Add --execute to apply.")


if __name__ == "__main__":
    main()
