"""Pure-logic unit tests for migrate_benchmark_tables (no database)."""

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import migrate_benchmark_tables as mig  # noqa: E402


def _col(name: str, *, nullable: bool = True, has_default: bool = False) -> mig.ColumnInfo:
    return mig.ColumnInfo(name=name, nullable=nullable, has_default=has_default)


def test_plan_copy_columns_follows_target_order_and_skips_source_only() -> None:
    source = [_col("id"), _col("run_id"), _col("legacy_only")]
    target = [_col("run_id", nullable=False), _col("id", nullable=False), _col("note")]
    assert mig.plan_copy_columns(source, target) == ["run_id", "id"]


def test_plan_copy_columns_rejects_missing_required_target_column() -> None:
    source = [_col("run_id", nullable=False)]
    target = [_col("run_id", nullable=False), _col("status", nullable=False)]
    with pytest.raises(ValueError, match="status"):
        mig.plan_copy_columns(source, target)


def test_plan_copy_columns_allows_missing_nullable_or_defaulted_column() -> None:
    source = [_col("run_id", nullable=False)]
    target = [
        _col("run_id", nullable=False),
        _col("note", nullable=True),
        _col("created_at", nullable=False, has_default=True),
    ]
    assert mig.plan_copy_columns(source, target) == ["run_id"]


def test_normalize_postgres_url_strips_driver_and_accepts_postgres_scheme() -> None:
    assert (
        mig.normalize_postgres_url("postgresql+psycopg://user:pw@db/agent")
        == "postgresql://user:pw@db/agent"
    )
    assert mig.normalize_postgres_url("postgres://db/agent") == "postgresql://db/agent"


def test_normalize_postgres_url_rejects_sqlite() -> None:
    with pytest.raises(ValueError, match="PostgreSQL"):
        mig.normalize_postgres_url("sqlite:///./agent_eval.db")


def test_quote_ident_rejects_injection() -> None:
    with pytest.raises(ValueError, match="非法标识符"):
        mig.quote_ident("run_id; drop table")


def test_source_page_sql_reads_only_the_source_table() -> None:
    pair = mig.BENCHMARK_TABLES[0]
    sql = mig.build_source_page_sql(pair, ["id", "run_id", "updated_at"], has_cursor=True)
    assert 'FROM "benchmark_runs_bullmq" AS s' in sql
    assert 'benchmark_runs"' not in sql.replace("benchmark_runs_bullmq", "")
    assert 'WHERE (s."run_id") > (%s)' in sql
    assert "LIMIT %s" in sql
    assert mig._BATCH_LIMIT < 10000


def test_insert_batch_row_limit_caps_by_pg_bind_param_ceiling() -> None:
    # 10 columns × 9999 rows = 99990 > 65535; it should cut to 6553 rows at the parameter limit.
    assert mig.insert_batch_row_limit(10) == 6553
    assert mig.insert_batch_row_limit(10) * 10 <= mig._PG_BIND_PARAM_LIMIT
    # Even with few columns it is still constrained by DB-RDB-003 (<10000).
    assert mig.insert_batch_row_limit(1) == mig._BATCH_LIMIT


def test_insert_values_sql_rejects_batch_that_exceeds_pg_bind_params() -> None:
    columns = [f"c{i}" for i in range(10)]
    too_many = mig.insert_batch_row_limit(len(columns)) + 1
    with pytest.raises(ValueError, match="PG 绑定参数上限"):
        mig.build_insert_values_sql("benchmark_runs", columns, too_many)


def test_insert_values_sql_writes_only_the_target_table() -> None:
    sql = mig.build_insert_values_sql("benchmark_runs", ["id", "run_id"], 2)
    assert sql.startswith('INSERT INTO "benchmark_runs"')
    assert "benchmark_runs_bullmq" not in sql
    assert sql.count("%s") == 4
    assert "ON CONFLICT DO NOTHING" in sql
    assert "UPDATE" not in sql


def test_partition_new_rows_skips_existing_key_and_id() -> None:
    columns = ["id", "run_id", "updated_at"]
    rows = [
        ("id-old", "run-old", 1),
        ("id-taken", "run-new", 2),
        ("id-fresh", "run-fresh", 3),
        ("id-dup", "run-fresh", 4),
    ]
    skipped, fresh = mig.partition_new_rows(
        rows,
        mig.PartitionColumns(columns=columns, conflict_columns=("run_id",)),
        {("run-old",)},
        {"id-taken"},
    )
    assert [row[1] for row in skipped] == ["run-old", "run-new", "run-fresh"]
    assert [row[1] for row in fresh] == ["run-fresh"]


def test_target_keys_sql_uses_item_conflict_columns() -> None:
    pair = mig.BENCHMARK_TABLES[1]
    sql = mig.build_target_keys_sql(pair)
    assert 'FROM "benchmark_run_items"' in sql
    assert '"run_id"' in sql
    assert '"item_id"' in sql
    assert '"criterion_id"' in sql
