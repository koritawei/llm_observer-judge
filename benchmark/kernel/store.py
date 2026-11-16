from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCHEMA = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


class Store:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def connect(self):
        return psycopg.AsyncConnection.connect(self.dsn, row_factory=dict_row)

    async def setup(self) -> None:
        script = SCHEMA.read_text(encoding="utf-8")
        statements = [part.strip() for part in script.split(";") if part.strip()]
        async with await self.connect() as conn:
            async with conn.cursor() as cur:
                for statement in statements:
                    await cur.execute(statement)
            await conn.commit()

    async def truncate(self) -> None:
        async with await self.connect() as conn:
            await conn.execute(
                """
                TRUNCATE
                    benchmark_runs,
                    benchmark_run_items,
                    input_manifest,
                    admissions,
                    phase_cursors,
                    run_progress_events,
                    attribution_decisions,
                    outbox,
                    submission_intents,
                    lease_ledger,
                    trace_pipeline_records,
                    agent_registrations,
                    eval_experiments,
                    eval_release_events,
                    evolution_requests,
                    evolution_campaigns,
                    evolution_contracts,
                    evolution_heldout,
                    evolution_steps,
                    evolution_candidates,
                    evolution_trials,
                    evolution_confirmations,
                    evolution_releases
                """
            )
            await conn.commit()

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with await self.connect() as conn:
            cur = await conn.execute("SELECT * FROM benchmark_runs WHERE run_id = %s", (run_id,))
            return await cur.fetchone()

    async def list_runs(self) -> list[dict[str, Any]]:
        async with await self.connect() as conn:
            cur = await conn.execute(
                "SELECT * FROM benchmark_runs ORDER BY created_at DESC LIMIT 200"
            )
            return await cur.fetchall()

    async def list_items(self, run_id: str) -> list[dict[str, Any]]:
        async with await self.connect() as conn:
            cur = await conn.execute(
                """
                SELECT * FROM benchmark_run_items
                WHERE run_id = %s
                ORDER BY item_id, criterion_id
                """,
                (run_id,),
            )
            return await cur.fetchall()

    async def list_manifest(self, run_id: str) -> list[dict[str, Any]]:
        async with await self.connect() as conn:
            cur = await conn.execute(
                """
                SELECT item_id, payload FROM input_manifest
                WHERE run_id = %s
                ORDER BY item_id
                """,
                (run_id,),
            )
            return await cur.fetchall()

    async def list_cursors(self, run_id: str) -> list[dict[str, Any]]:
        async with await self.connect() as conn:
            cur = await conn.execute(
                """
                SELECT item_id, phase_id, output FROM phase_cursors
                WHERE run_id = %s
                ORDER BY item_id, phase_id
                """,
                (run_id,),
            )
            return await cur.fetchall()

    async def find_by_idempotency(self, key: str) -> dict[str, Any] | None:
        async with await self.connect() as conn:
            cur = await conn.execute(
                "SELECT * FROM benchmark_runs WHERE idempotency_key = %s",
                (key,),
            )
            return await cur.fetchone()
