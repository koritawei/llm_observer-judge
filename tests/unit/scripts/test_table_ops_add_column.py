"""CLI-layer validation for table_ops add-column / drop-column (the database is not touched).

Add-column allows only the shape "ADD COLUMN IF NOT EXISTS <column> <type>": both identifiers
and column types are allowlisted, and injection or extra clauses are rejected. Drop-column validation refuses to drop the primary-key column ``id``
as an ordinary column (destroying the table must go through drop-table).
"""

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "table_ops.py"


def _load_module():  # noqa: ANN202
    # Test only the pure validation function: stub the script's startup side effects (_bootstrap / the app import).
    # patch.dict temporarily replaces sys.modules inside the with block and restores it — the old form used
    # setdefault to shove an empty benchmark.service.main into sys.modules permanently, and modules collected later
    # such as test_main then got a fake benchmark.service.main and raised AttributeError (latent for a long time).
    stubs = {
        name: types.ModuleType(name) for name in ("_bootstrap", "app", "benchmark.settings", "benchmark.service.main")
    }
    stubs["benchmark.settings"].Settings = object  # type: ignore[attr-defined]
    stubs["benchmark.service.main"].create_app = lambda **_kw: None  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location("table_ops_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


table_ops = _load_module()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("TIMESTAMPTZ", "TIMESTAMPTZ"),
        ("timestamptz", "TIMESTAMPTZ"),
        ("TEXT", "TEXT"),
        ("VARCHAR(128)", "VARCHAR(128)"),
        ("DOUBLE PRECISION", "DOUBLE PRECISION"),
        ("NUMERIC(10,2)", "NUMERIC(10,2)"),
    ],
)
def test_column_type_accepted(raw: str, expected: str) -> None:
    assert table_ops._check_column_type(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "TIMESTAMPTZ; DROP TABLE benchmark_runs",
        "TIMESTAMPTZ DEFAULT now()",
        "TEXT NOT NULL",
        "INT REFERENCES other(id)",
        "",
        "VARCHAR()",
        "VARCHAR(abc)",
    ],
)
def test_column_type_rejected(raw: str) -> None:
    with pytest.raises(SystemExit):
        table_ops._check_column_type(raw)


def test_ident_validation() -> None:
    assert table_ops._check_ident("benchmark_runs", "table") == "benchmark_runs"
    assert table_ops._check_ident("deleted_at", "column") == "deleted_at"
    with pytest.raises(SystemExit):
        table_ops._check_ident("benchmark_runs; DROP TABLE x", "table")


def _drop_column_args(column: str) -> Namespace:
    return Namespace(
        command="drop-column",
        table="runtime_config_records",
        column=column,
        execute=False,
        force=False,
    )


def test_drop_column_accepts_regular_column() -> None:
    args = table_ops._validate_args(_drop_column_args("enabled"))
    assert args.table == "runtime_config_records"
    assert args.column == "enabled"


def test_drop_column_rejects_primary_key() -> None:
    with pytest.raises(SystemExit):
        table_ops._validate_args(_drop_column_args("id"))


def test_drop_column_rejects_illegal_identifiers() -> None:
    with pytest.raises(SystemExit):
        table_ops._validate_args(_drop_column_args("enabled; DROP TABLE x"))
    with pytest.raises(SystemExit):
        table_ops._validate_args(
            Namespace(
                command="drop-column",
                table="runtime_config_records; DROP TABLE x",
                column="enabled",
                execute=False,
                force=False,
            ),
        )
