"""Programmatic Alembic helpers + CLI (llmj-db)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from alembic import command
from alembic.config import Config


def repo_root() -> Path:
    """src/llmobserverjudge/shared/migrate.py → repo root."""
    return Path(__file__).resolve().parents[3]


def alembic_config(
    database_url: str | None = None,
    *,
    ini_path: Path | None = None,
) -> Config:
    root = repo_root()
    ini = ini_path or Path(os.environ.get("LLMJUDGE_ALEMBIC_INI") or root / "alembic.ini")
    if not ini.is_file():
        raise FileNotFoundError(
            f"alembic.ini not found at {ini}; run from repo root or set LLMJUDGE_ALEMBIC_INI"
        )
    cfg = Config(str(ini))
    cfg.set_main_option("script_location", str(root / "alembic"))
    url = database_url or os.environ.get("LLMJUDGE_DATABASE_URL")
    if url:
        cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def upgrade_head(database_url: str | None = None) -> None:
    command.upgrade(alembic_config(database_url), "head")


def downgrade_base(database_url: str | None = None) -> None:
    command.downgrade(alembic_config(database_url), "base")


def current_revision(database_url: str | None = None) -> None:
    command.current(alembic_config(database_url))


def main(argv: list[str] | None = None) -> None:
    from llmobserverjudge.shared.env import load_env

    load_env()
    parser = argparse.ArgumentParser(prog="llmj-db", description="LLMJudge DB migrations")
    parser.add_argument(
        "command",
        choices=["upgrade", "downgrade", "current", "history"],
        help="Alembic command",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Database URL (default: LLMJUDGE_DATABASE_URL / alembic.ini)",
    )
    parser.add_argument(
        "--revision",
        default="head",
        help="Target revision for upgrade/downgrade (default: head / base)",
    )
    args = parser.parse_args(argv)
    cfg = alembic_config(args.url)
    if args.command == "upgrade":
        command.upgrade(cfg, args.revision)
    elif args.command == "downgrade":
        rev = args.revision if args.revision != "head" else "base"
        command.downgrade(cfg, rev)
    elif args.command == "current":
        command.current(cfg)
    elif args.command == "history":
        command.history(cfg)


if __name__ == "__main__":
    main()
