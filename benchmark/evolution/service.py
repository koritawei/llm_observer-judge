# -*- coding: utf-8 -*-
"""Evolution campaigns. Candidates are sealed records; this module does not merge them."""

from __future__ import annotations

import uuid
from typing import Any

from psycopg.types.json import Jsonb

from benchmark.evolution.canonical import evolution_hash
from benchmark.evolution.graph import advance_result, build_evolution_graph
from benchmark.evolution.policy import (
    classify_admission,
    decide_evidence,
    diagnose,
    freeze_manifest,
    gate_policy_for_trial,
    materialize_outputs,
    plan_change,
    scan_candidate,
    trial_checks,
)
from benchmark.kernel.errors import Conflict, EvalError, NotFound


def _decision(
    campaign_id: str,
    verdict: str,
    status: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "verdict": verdict,
        "status": status,
        "benchmark_run_id": run_id,
    }


def _domain_id(snapshot: dict[str, Any]) -> str | None:
    domain = snapshot.get("eval_domain") or {}
    if not isinstance(domain, dict):
        return None
    found = domain.get("domain_id") or domain.get("id")
    return None if found is None else str(found)


_TERMINAL = {"rejected", "quarantined", "cancelled", "completed", "rolled_back", "archived"}
_EARLY_REASONS = {
    "policy_not_activatable",
    "parent_event_missing_words",
    "parent_event_does_not_open_campaign",
}


class EvolutionService:
    def __init__(self, kernel: Any):
        self.kernel = kernel

    async def admit(self, event: dict[str, Any]) -> dict[str, Any]:
        tenant = str(event.get("tenant_id") or "default")
        run_id = str(event.get("run_id") or "")
        revision = int(event.get("report_revision") or 1)
        kind = "parent" if event.get("event_type") == "parent" else "source"
        source_key = evolution_hash("source_event", {
            "tenant_id": tenant,
            "kind": kind,
            "run_id": run_id,
            "report_revision": revision,
            "admission_policy_hash": event.get("admission_policy_hash") or event.get("policy_id") or "",
        })
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                existing = await conn.execute(
                    "SELECT * FROM evolution_requests WHERE source_key = %s",
                    (source_key,),
                )
                prior = await existing.fetchone()
                if prior is not None:
                    return {
                        "request_id": source_key,
                        "campaign_id": prior["campaign_id"],
                        "status": prior["status"],
                        "reason": prior["reason"],
                        "replayed": True,
                    }
                decision = classify_admission({}, [], event)
                if decision["reason"] not in _EARLY_REASONS:
                    run = await self.kernel.store.get_run(run_id)
                    if run is None:
                        raise NotFound("source run not found")
                    items = await self._source_items(run_id)
                    decision = classify_admission(run, items, event)
                campaign_id = None
                if decision["status"] == "admitted":
                    campaign_id = f"evo-{uuid.uuid4().hex[:12]}"
                    await conn.execute(
                        """
                        INSERT INTO evolution_campaigns (
                            campaign_id, tenant_id, source_run_id, status, budget_json
                        ) VALUES (%s,%s,%s,'admitted',%s)
                        """,
                        (campaign_id, tenant, run_id, Jsonb(event.get("budget") or {"max_candidates": 1})),
                    )
                await conn.execute(
                    """
                    INSERT INTO evolution_requests (
                        source_key, tenant_id, run_id, report_revision, status, reason, campaign_id, payload
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        source_key,
                        tenant,
                        run_id,
                        revision,
                        decision["status"],
                        decision["reason"],
                        campaign_id,
                        Jsonb({"event_type": event.get("event_type") or "source"}),
                    ),
                )
        return {
            "request_id": source_key,
            "campaign_id": campaign_id,
            "status": decision["status"],
            "reason": decision["reason"],
            "replayed": False,
        }

    async def freeze(self, campaign_id: str, body: dict[str, Any]) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        if campaign["status"] not in {"admitted", "freezing"}:
            raise Conflict("campaign is not waiting for a contract")
        run = await self.kernel.store.get_run(campaign["source_run_id"])
        if run is None:
            raise NotFound("source run not found")
        items = await self._source_items(campaign["source_run_id"])
        try:
            sealed = freeze_manifest(
                campaign_id,
                run,
                items,
                gate_policy=dict(body.get("gate_policy") or {}),
                heldout_ids=list(body.get("heldout_ids") or []),
                subject_hash=str(body.get("subject_hash") or "subject-unspecified"),
                repairer_hash=str(body.get("repairer_hash") or "repairer-unspecified"),
                evaluator_hash=str(body.get("evaluator_hash") or "evaluator-unspecified"),
            )
        except ValueError as exc:
            raise EvalError(str(exc)) from exc
        public = sealed["public"]
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO evolution_contracts (contract_hash, campaign_id, manifest)
                    VALUES (%s,%s,%s)
                    ON CONFLICT (contract_hash) DO NOTHING
                    """,
                    (public["contract_hash"], campaign_id, Jsonb(public)),
                )
                for item in sealed["heldout"]:
                    await conn.execute(
                        """
                        INSERT INTO evolution_heldout (campaign_id, item_id, payload)
                        VALUES (%s,%s,%s)
                        ON CONFLICT (campaign_id, item_id) DO NOTHING
                        """,
                        (campaign_id, str(item["item_id"]), Jsonb(item)),
                    )
                await conn.execute(
                    """
                    UPDATE evolution_campaigns
                    SET status = 'frozen', contract_hash = %s
                    WHERE campaign_id = %s
                    """,
                    (public["contract_hash"], campaign_id),
                )
                await self._step(conn, campaign_id, "freeze", "contract", public["contract_hash"], public)
        return {"campaign_id": campaign_id, "contract_hash": public["contract_hash"], "status": "frozen"}

    async def diagnose_campaign(self, campaign_id: str) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        contract = await self._contract(campaign["contract_hash"])
        items = await self._source_items(campaign["source_run_id"])
        record = diagnose(items, list(contract["bad_target_ids"]))
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE evolution_campaigns
                    SET status = 'diagnosed', diagnosis_json = %s
                    WHERE campaign_id = %s
                    """,
                    (Jsonb(record), campaign_id),
                )
                await self._step(
                    conn,
                    campaign_id,
                    "diagnose",
                    "diagnosis",
                    evolution_hash("diagnosis", record),
                    record,
                )
        return {"campaign_id": campaign_id, **record}

    async def plan(self, campaign_id: str, body: dict[str, Any]) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        diagnosis = campaign.get("diagnosis_json") or {}
        decision = plan_change(
            diagnosis,
            diff_text=str(body.get("diff_text") or ""),
            allowed_prefixes=list(body.get("allowed_prefixes") or []),
        )
        status = "planning" if decision["status"] == "authorized" else decision["status"]
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE evolution_campaigns SET status = %s, plan_json = %s WHERE campaign_id = %s",
                    (status, Jsonb({**body, **decision}), campaign_id),
                )
                await self._step(conn, campaign_id, "plan", "plan", evolution_hash("plan", decision), decision)
        return {"campaign_id": campaign_id, **decision}

    async def seal_candidate(self, campaign_id: str, body: dict[str, Any]) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        if campaign["status"] != "planning":
            raise Conflict("plan is not authorized")
        diff_text = str(body.get("diff_text") or "")
        raw_outputs = dict(body.get("proposed_outputs") or {})
        hidden_rows = await self._heldout_rows(campaign_id)
        findings = scan_candidate(diff_text, raw_outputs, hidden_rows)
        if findings:
            await self._mark(campaign_id, "blocked", "seal", {"findings": findings})
            return {"campaign_id": campaign_id, "status": "blocked", "reason": findings[0]}
        hidden_ids = {str(row.get("item_id")) for row in hidden_rows}
        proposed = {
            str(key): value
            for key, value in raw_outputs.items()
            if str(key) not in hidden_ids
        }
        try:
            artifact_hash = evolution_hash("subject", {"diff": diff_text, "outputs": proposed, "kind": "subject"})
        except ValueError as exc:
            await self._mark(campaign_id, "blocked", "seal", {"reason": str(exc)})
            return {"campaign_id": campaign_id, "status": "blocked", "reason": str(exc)}
        candidate_id = f"cand-{artifact_hash[-12:]}"
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO evolution_candidates (
                        candidate_id, campaign_id, artifact_hash, artifact_kind, status, diff_text, proposed_outputs
                    ) VALUES (%s,%s,%s,'subject','sealed',%s,%s)
                    ON CONFLICT (campaign_id, artifact_hash) DO NOTHING
                    """,
                    (candidate_id, campaign_id, artifact_hash, diff_text, Jsonb(proposed)),
                )
                await conn.execute(
                    "UPDATE evolution_campaigns SET status = 'candidate_sealed' WHERE campaign_id = %s",
                    (campaign_id,),
                )
                await self._step(
                    conn, campaign_id, "seal", artifact_hash, artifact_hash, {"candidate_id": candidate_id}
                )
        return {
            "campaign_id": campaign_id,
            "candidate_id": candidate_id,
            "artifact_hash": artifact_hash,
            "status": "candidate_sealed",
        }

    async def start_trial(self, campaign_id: str) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        if campaign["status"] != "candidate_sealed":
            raise Conflict("a sealed candidate is required")
        candidate = await self._candidate(campaign_id)
        items = await self._source_items(campaign["source_run_id"])
        hidden = await self._heldout_ids(campaign_id)
        visible = []
        seen: set[str] = set()
        for item in items:
            item_id = str(item["item_id"])
            if item_id in hidden or item_id in seen:
                continue
            seen.add(item_id)
            visible.append(item)
        contract = await self._contract(campaign["contract_hash"])
        source = await self.kernel.store.get_run(campaign["source_run_id"])
        snapshot = (source or {}).get("config_json") or {}
        proposed = candidate["proposed_outputs"] or {}
        try:
            fresh = materialize_outputs(visible, proposed)
        except ValueError as exc:
            await self._mark(campaign_id, "blocked", "trial", {"reason": str(exc)})
            return {
                "campaign_id": campaign_id,
                "status": "blocked",
                "reason": str(exc),
                "benchmark_run_id": None,
            }
        cases = []
        for item, produced in zip(visible, fresh, strict=True):
            case = dict(produced)
            if "expected" in item:
                case["expected"] = item.get("expected")
            cases.append(case)
        profile_id = str(contract.get("profile_id") or snapshot.get("profile_id") or "benchmark_sync")
        domain = snapshot.get("eval_domain") if isinstance(snapshot.get("eval_domain"), dict) else None
        created = await self.kernel.publish_run(
            {
                "benchmark_name": "evolution-trial",
                "profile_id": profile_id,
                "business": str(snapshot.get("business") or "default"),
                "cases": cases,
                "origin_chain": ["evolution_trial", campaign_id],
            },
            eval_domain=domain,
        )
        trial_id = f"trial-{uuid.uuid4().hex[:12]}"
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO evolution_trials (
                        trial_id, campaign_id, candidate_id, benchmark_run_id, stage, status
                    ) VALUES (%s,%s,%s,%s,'development','running')
                    """,
                    (trial_id, campaign_id, candidate["candidate_id"], created["run_id"]),
                )
                await conn.execute(
                    "UPDATE evolution_campaigns SET status = 'trial_running' WHERE campaign_id = %s",
                    (campaign_id,),
                )
        run = await self.kernel.store.get_run(created["run_id"])
        return {
            "campaign_id": campaign_id,
            "trial_id": trial_id,
            "benchmark_run_id": created["run_id"],
            "status": "trial_running",
            "run_terminal": None if run is None else run["status"],
        }

    async def decide(self, campaign_id: str, body: dict[str, Any]) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        if campaign["status"] != "trial_running":
            raise Conflict("decision requires a running trial")
        contract = await self._contract(campaign["contract_hash"])
        policy = dict(body.get("gate_policy") or {})
        if policy.get("gate_policy_hash") != contract.get("gate_policy_hash"):
            raise EvalError("gate policy hash does not match the frozen contract")
        trial = await self._development_trial(campaign_id)
        run_id = str(trial["benchmark_run_id"])
        run = await self.kernel.store.get_run(run_id)
        if run is None or run["status"] != "completed":
            return {
                "campaign_id": campaign_id,
                "verdict": "inconclusive",
                "status": "trial_running",
                "reason": "trial_not_completed",
                "benchmark_run_id": run_id,
            }
        snapshot = run.get("config_json") or {}
        if contract.get("profile_id") and snapshot.get("profile_id") != contract["profile_id"]:
            await self._mark(campaign_id, "blocked", "decide", {"reason": "profile_mismatch"})
            return {**_decision(campaign_id, "inconclusive", "blocked", run_id), "reason": "profile_mismatch"}
        pinned_domain = contract.get("eval_domain_id")
        if pinned_domain and _domain_id(snapshot) != pinned_domain:
            await self._mark(campaign_id, "blocked", "decide", {"reason": "eval_domain_mismatch"})
            return {**_decision(campaign_id, "inconclusive", "blocked", run_id), "reason": "eval_domain_mismatch"}
        items = await self.kernel.store.list_items(run_id)
        effective = gate_policy_for_trial(policy, contract)
        checks = trial_checks(contract, effective, items)
        if any(item.get("verdict") == "unknown" for item in checks):
            word = "inconclusive"
        else:
            word = decide_evidence(
                effective,
                checks,
                calibration=body.get("calibration"),
                judges=list(body.get("judges") or []),
            )
        status = {"hard_pass": "gate_accepted", "hard_fail": "gate_denied"}.get(word, "blocked")
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE evolution_campaigns SET status = %s WHERE campaign_id = %s",
                    (status, campaign_id),
                )
                await self._step(conn, campaign_id, "decide", "gate", word, {"verdict": word})
        return _decision(campaign_id, word, status, run_id)

    async def confirm(self, campaign_id: str) -> dict[str, Any]:
        """Heldout is not scored from caller-supplied outputs. No collector means inconclusive."""

        campaign = await self._campaign(campaign_id)
        if campaign["status"] != "gate_accepted":
            raise Conflict("confirmation requires a hard pass")
        async with await self.kernel.store.connect() as conn:
            existing = await conn.execute(
                "SELECT confirmation_key FROM evolution_confirmations WHERE campaign_id = %s",
                (campaign_id,),
            )
            if await existing.fetchone() is not None:
                raise Conflict("heldout confirmation was already reserved")
        result = {
            "stage_outcome": "inconclusive",
            "release_authorization": "denied",
            "checked": 0,
            "reason": "heldout_not_collected",
        }
        await self._mark(campaign_id, "blocked", "confirm", {
            "stage_outcome": result["stage_outcome"],
            "release_authorization": result["release_authorization"],
            "reason": result["reason"],
        })
        return {"campaign_id": campaign_id, **result}

    async def request_release(self, campaign_id: str, body: dict[str, Any]) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        if campaign["status"] != "release_ready":
            raise Conflict("release requires a confirmed campaign")
        experiment_id = str(body.get("experiment_id") or campaign.get("experiment_id") or "")
        if not experiment_id:
            raise EvalError("experiment_id is required")
        experiment = await self.kernel._load_experiment(experiment_id)
        if experiment["release_authorization"] != "approved":
            raise EvalError("release requires an approved authorization", code="release_rejected")
        if not body.get("blast_radius_ref") or not body.get("approval_ref"):
            raise EvalError("blast radius and approval are required")
        candidate = await self._candidate(campaign_id)
        release_id = f"rel-{uuid.uuid4().hex[:12]}"
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO evolution_releases (
                        release_id, campaign_id, artifact_hash, status, approval_ref
                    ) VALUES (%s,%s,%s,'manual_pr',%s)
                    """,
                    (release_id, campaign_id, candidate["artifact_hash"], body["approval_ref"]),
                )
                await conn.execute(
                    """
                    UPDATE evolution_campaigns
                    SET status = 'releasing', experiment_id = %s
                    WHERE campaign_id = %s
                    """,
                    (experiment_id, campaign_id),
                )
        return {
            "campaign_id": campaign_id,
            "release_id": release_id,
            "status": "manual_pr",
            "artifact_hash": candidate["artifact_hash"],
        }

    async def propose_repairer(self, campaign_id: str, body: dict[str, Any]) -> dict[str, Any]:
        await self._campaign(campaign_id)
        record = {
            "kind": "repairer_mutation",
            "diff_text": str(body.get("diff_text") or ""),
            "applied": False,
        }
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await self._step(conn, campaign_id, "repairer", "proposal", evolution_hash("repairer", record), record)
        return {"campaign_id": campaign_id, "status": "recorded", "applied": False}

    async def archive(self, campaign_id: str) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        if campaign["status"] not in {
            "release_ready",
            "releasing",
            "rejected",
            "gate_denied",
            "blocked",
            "rolled_back",
        }:
            raise Conflict("campaign cannot be archived")
        async with await self.kernel.store.connect() as conn:
            await conn.execute(
                "UPDATE evolution_campaigns SET status = 'archived' WHERE campaign_id = %s",
                (campaign_id,),
            )
            await conn.commit()
        return {"campaign_id": campaign_id, "status": "archived"}

    async def rollback(self, campaign_id: str) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        if campaign["status"] != "releasing":
            raise Conflict("only a releasing campaign can roll back")
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE evolution_releases SET status = 'rolled_back' WHERE campaign_id = %s",
                    (campaign_id,),
                )
                await conn.execute(
                    "UPDATE evolution_campaigns SET status = 'rolled_back' WHERE campaign_id = %s",
                    (campaign_id,),
                )
        return {"campaign_id": campaign_id, "status": "rolled_back"}

    async def stop(self, campaign_id: str, body: dict[str, Any]) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        if campaign["status"] in _TERMINAL:
            raise Conflict("campaign is already terminal")
        raw_epoch = body.get("expected_epoch")
        if raw_epoch is None or int(raw_epoch) != int(campaign["execution_epoch"]):
            raise Conflict("epoch mismatch")
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    """
                    UPDATE evolution_campaigns
                    SET status = 'cancelled', execution_epoch = execution_epoch + 1
                    WHERE campaign_id = %s AND execution_epoch = %s
                    RETURNING campaign_id
                    """,
                    (campaign_id, int(raw_epoch)),
                )
                if await cur.fetchone() is None:
                    raise Conflict("epoch mismatch")
        return {"campaign_id": campaign_id, "status": "cancelled", "reason": body.get("reason") or "stopped"}

    async def advance(self, campaign_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Walk the evolution graph from the current step through heldout confirmation."""

        campaign = await self._campaign(campaign_id)
        final = await build_evolution_graph(self).ainvoke({
            "campaign_id": campaign_id,
            "body": dict(body),
            "status": campaign["status"],
            "requested_step": "advance",
            "visited": [],
        })
        return advance_result(campaign_id, final)

    async def public_contract(self, campaign_id: str) -> dict[str, Any]:
        campaign = await self._campaign(campaign_id)
        manifest = await self._contract(campaign["contract_hash"])
        return {key: value for key, value in manifest.items() if key != "visible_items"}

    async def _mark(self, campaign_id: str, status: str, kind: str, output: dict[str, Any]) -> None:
        async with await self.kernel.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE evolution_campaigns SET status = %s WHERE campaign_id = %s",
                    (status, campaign_id),
                )
                await self._step(conn, campaign_id, kind, status, evolution_hash("step", output), output)

    async def _heldout_rows(self, campaign_id: str) -> list[dict[str, Any]]:
        async with await self.kernel.store.connect() as conn:
            cur = await conn.execute(
                "SELECT item_id, payload FROM evolution_heldout WHERE campaign_id = %s ORDER BY item_id",
                (campaign_id,),
            )
            rows = await cur.fetchall()
        plain = []
        for row in rows:
            payload = dict(row["payload"] or {})
            payload["item_id"] = str(row["item_id"])
            plain.append(payload)
        return plain

    async def _development_trial(self, campaign_id: str) -> dict[str, Any]:
        async with await self.kernel.store.connect() as conn:
            cur = await conn.execute(
                """
                SELECT * FROM evolution_trials
                WHERE campaign_id = %s AND stage = 'development'
                ORDER BY trial_id DESC
                LIMIT 1
                """,
                (campaign_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise NotFound("development trial not found")
        return row

    async def _source_items(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self.kernel.store.list_items(run_id)
        async with await self.kernel.store.connect() as conn:
            cur = await conn.execute(
                "SELECT item_id, payload FROM input_manifest WHERE run_id = %s",
                (run_id,),
            )
            manifests = {row["item_id"]: row["payload"] for row in await cur.fetchall()}
        plain = []
        for item in rows:
            payload = manifests.get(item["item_id"]) or {}
            plain.append({
                "item_id": str(item["item_id"]),
                "status": item.get("status"),
                "score": item.get("score"),
                "reason": item.get("reason"),
                "error_message": item.get("error_message"),
                "input": payload.get("input", item.get("input")),
                "output": payload.get("output", item.get("output")),
                "expected": payload.get("expected"),
            })
        return plain

    async def _heldout_ids(self, campaign_id: str) -> set[str]:
        async with await self.kernel.store.connect() as conn:
            cur = await conn.execute(
                "SELECT item_id FROM evolution_heldout WHERE campaign_id = %s",
                (campaign_id,),
            )
            return {str(row["item_id"]) for row in await cur.fetchall()}

    async def _campaign(self, campaign_id: str) -> dict[str, Any]:
        async with await self.kernel.store.connect() as conn:
            cur = await conn.execute(
                "SELECT * FROM evolution_campaigns WHERE campaign_id = %s",
                (campaign_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise NotFound("campaign not found")
        return row

    async def _contract(self, contract_hash: str | None) -> dict[str, Any]:
        if not contract_hash:
            raise Conflict("contract is not frozen")
        async with await self.kernel.store.connect() as conn:
            cur = await conn.execute(
                "SELECT manifest FROM evolution_contracts WHERE contract_hash = %s",
                (contract_hash,),
            )
            row = await cur.fetchone()
        if row is None:
            raise NotFound("contract not found")
        return row["manifest"]

    async def _candidate(self, campaign_id: str) -> dict[str, Any]:
        async with await self.kernel.store.connect() as conn:
            cur = await conn.execute(
                """
                SELECT * FROM evolution_candidates
                WHERE campaign_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (campaign_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise NotFound("candidate not found")
        return row

    async def _step(
        self,
        conn: Any,
        campaign_id: str,
        kind: str,
        scope: str,
        output_hash: str,
        output: dict[str, Any],
    ) -> None:
        await conn.execute(
            """
            INSERT INTO evolution_steps (
                step_id, campaign_id, step_kind, scope_key, status, output_hash, output
            ) VALUES (%s,%s,%s,%s,'completed',%s,%s)
            ON CONFLICT (campaign_id, step_kind, scope_key) DO NOTHING
            """,
            (f"{campaign_id}:{kind}:{scope}", campaign_id, kind, scope, output_hash, Jsonb(output)),
        )
