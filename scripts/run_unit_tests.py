"""Run SQLite tests first, then PG tests, with a clean subprocess boundary.

The Tortoise ORM caches backend dialect state in module-level globals,
so once an event loop has bound to SQLite the asyncpg client cannot
re-attach cleanly within the same Python process.  Splitting the two
groups across two ``pytest`` invocations gives each backend a fresh
interpreter.

Usage::

    # SQLite pass + PG pass (PG_* must be set in the env for the PG
    # pass to actually exercise anything; otherwise it is a no-op
    # because the PG-tagged test classes ``pytest.skip`` cleanly).
    .venv/bin/python scripts/run_unit_tests.py

The script exits 0 only when both passes succeed.
"""

import os
import subprocess
import sys


SQLITE_ARGS = ["-m", "pytest", "tests/unit/", "-v"]
PG_ARGS = ["-m", "pytest", "tests/unit/", "-v", "-k", "TestPg"]


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = [sys.executable]

    # 1) SQLite pass — unset PG_* so autouse fixtures stay on the
    #    sqlite path (per-test ``_tortoise_init`` re-creates schema).
    sqlite_env = {k: v for k, v in os.environ.items() if not k.startswith("PG_")}
    print("=== SQLite pass (no PG_*) ===", flush=True)
    sqlite_rc = subprocess.run(py + SQLITE_ARGS, cwd=repo, env=sqlite_env).returncode

    # 2) PG pass — only meaningful when PG_* is configured.  When
    #    unset, the PG-tagged test classes skip cleanly so this is a
    #    no-op.
    pg_env = {k: v for k, v in os.environ.items() if not k.startswith("PG_")}
    pg_env.update({k: os.environ[k] for k in os.environ if k.startswith("PG_")})
    if any(k.startswith("PG_") for k in pg_env):
        print("=== PG pass (PG_* set, TestPg only) ===", flush=True)
        pg_rc = subprocess.run(py + PG_ARGS, cwd=repo, env=pg_env).returncode
    else:
        print("=== PG pass skipped (no PG_* env) ===", flush=True)
        pg_rc = 0

    return 0 if (sqlite_rc == 0 and pg_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
