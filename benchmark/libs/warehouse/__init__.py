"""Local table reader with the ODPS call shape used by the export job."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path


class CredentialProviderAccount:
    def __init__(self, provider=None):
        self.provider = provider


@dataclass
class Instance:
    sql: str
    project: str
    id: str = "local"
    status: str = "success"
    rows: list[dict] = field(default_factory=list)

    def wait_for_success(self) -> None:
        return None

    def open_reader(self, tunnel: bool = False, limit: bool = False):
        del tunnel, limit
        return iter(self.rows)


class ODPS:
    def __init__(self, account=None, project: str = "local", endpoint: str = ""):
        self.account = account
        self.project = project
        self.endpoint = endpoint
        self.root = Path(os.environ.get("BENCHMARK_WAREHOUSE_DIR", "var/warehouse")) / project

    def execute_sql(self, sql: str) -> Instance:
        rows = []
        if self.root.is_dir():
            for path in sorted(self.root.glob("*.csv")):
                with path.open(encoding="utf-8", newline="") as handle:
                    rows.extend(csv.DictReader(handle))
        return Instance(sql=sql, project=self.project, rows=rows)
