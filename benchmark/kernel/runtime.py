"""Batch pipeline kernel.

Redis/BullMQ owns dispatch. PostgreSQL owns state, results, and completion.
Plugins return outcomes; this module is the only writer of run status.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bullmq import WaitingChildrenError, Worker
from psycopg.types.json import Jsonb

from benchmark.kernel.casebook import assemble_case_book
from benchmark.kernel.confloader import Conf, canonical, load_conf, sha256_text
from benchmark.kernel.dispatch import JUDGE_QUEUES, RUNNING_QUEUE, Dispatch
from benchmark.kernel.errors import Conflict, EvalError, NotFound, Unauthorized, Unavailable
from benchmark.kernel.lease import LeaseManager, LeasePair, now_ms
from benchmark.kernel.pluginhost import REQUIRED_SLOTS, PluginHost
from benchmark.kernel.store import Store
from benchmark.kernel.workflow import compose_profile
from benchmark.kernel.types import (
    OUTCOMES,
    Completed,
    InvalidateAndReset,
    PermanentError,
    PhaseContext,
    ProviderAcceptedPendingFailed,
    ProviderAcceptedPendingRecorded,
    ProviderResultUnknown,
    ProviderSubmitRejected,
    ProviderSyncCompleted,
    Retryable,
    RetryableError,
    SubmitRequest,
    TerminalFailure,
)

log = logging.getLogger("benchmark")

TERMINAL_RUN = {"completed", "failed", "stopped"}
_BANNED_REQUEST_KEYS = {"url", "base_url", "url_key", "domain_id", "endpoint", "isolation_domain"}


class SlotBusy(Exception):
    pass


def _case_job_id(run_id: str, item_id: str) -> str:
    return f"{run_id}:{item_id}"


def _attr_job_id(run_id: str, item_id: str, criterion_id: str) -> str:
    return f"{run_id}:{item_id}:{criterion_id}"


def _new_run_id(stem: str) -> str:
    day = datetime.now().strftime("%Y%m%d")
    return f"{stem}-{day}-{uuid.uuid4().hex[:6]}"


class Kernel:
    def __init__(self, conf: Conf, *, node_id: str = "node-1", queue_prefix: str | None = None):
        self.conf = conf
        self.node_id = node_id
        self.queue_prefix = queue_prefix or conf.queue_prefix()
        self.store = Store(conf.database_dsn())
        self.dispatch = Dispatch(conf.redis_url(), self.queue_prefix, str(conf.app["keyspace_generation"]))
        self.leases = LeaseManager(conf.redis_url(), f"{self.queue_prefix}:cap")
        self.host = PluginHost(conf)
        self.workers: list[Worker] = []
        self.report_dir = Path(conf.app["object_store_dir"])
        if not self.report_dir.is_absolute():
            self.report_dir = Path.cwd() / self.report_dir

    async def setup(self) -> None:
        await self.store.setup()

    async def close(self) -> None:
        for worker in self.workers:
            await worker.close(force=True)
        self.workers.clear()
        await self.dispatch.close()
        await self.leases.close()

    def start_workers(self) -> None:
        profile = str(self.conf.app.get("worker_profile") or "default")
        names = ["benchmark-cases-l2", "attribution-wait"] if profile == "l2" else [
            "benchmark-cases",
            "dataset-cases",
            "attribution-wait",
        ]
        slots = self.conf.app["slots"]
        for name in names:
            concurrency = int(slots["attr" if name.startswith("attribution") else "sync"]["worker_concurrency"])
            worker = Worker(
                name,
                self._process,
                {
                    **self.dispatch.options(name),
                    "concurrency": concurrency,
                    "lockDuration": 120000,
                    "drainDelay": 0.05,
                    "autorun": True,
                },
            )
            self.workers.append(worker)

    async def _process(self, job, token):
        try:
            if job.name == "attribution":
                await self.handle_attribution(job, token)
            else:
                await self.handle(job, token)
        except WaitingChildrenError:
            raise
        except Exception:
            log.exception("processor failed job=%s", getattr(job, "id", ""))
            await self._safe_terminal(job, "processor_error")

    async def publish_run(
        self,
        body: dict[str, Any],
        *,
        criteria_plan: list[dict[str, Any]] | None = None,
        eval_domain: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._reject_transport_overrides(body)
        profile_id = str(body.get("profile_id") or "")
        workflow = body.get("workflow")
        if not isinstance(workflow, dict) and profile_id in {"playground_replay", "playground_rejudge"}:
            raise EvalError("replay is only created from a finished parent run", code="replay_entry")
        idem = body.get("idempotency_key")
        if idem:
            existing = await self.store.find_by_idempotency(str(idem))
            if existing:
                return self._public_run(existing, replayed=True)
        cases = [_normalize_case(case) for case in list(body.get("cases") or [])]
        business = str(body.get("business") or "default")
        if isinstance(workflow, dict):
            base = str(workflow.get("base_profile") or profile_id or "benchmark_sync")
            if base in {"playground_replay", "playground_rejudge"}:
                raise EvalError("replay is only created from a finished parent run", code="replay_entry")
            profile = compose_profile(
                self.conf,
                base,
                business=business,
                plugins=workflow.get("plugins") or {},
            )
            profile_id = str(body.get("benchmark_name") or "workflow")
            snapshot = self._freeze_profile(profile_id, profile, business=business)
        else:
            snapshot = self._freeze(profile_id, business=business)
        origin = body.get("origin_chain")
        if origin:
            if not isinstance(origin, list) or any(not isinstance(item, str) for item in origin):
                raise EvalError("origin_chain must be a list of strings")
            snapshot["origin_chain"] = list(origin)
            snapshot["source"] = "evolution_trial"
        if criteria_plan is not None:
            snapshot["criteria_plan"] = criteria_plan
            snapshot.setdefault("rubric", {})["criteria"] = criteria_plan
            if snapshot.get("eval_domain") is not None or eval_domain is not None:
                snapshot["profile_hash"] = sha256_text(canonical({
                    "profile_id": profile_id,
                    "criteria_plan": snapshot["criteria_plan"],
                    "eval_domain": eval_domain,
                }))
        if eval_domain is not None:
            snapshot["eval_domain"] = eval_domain
            snapshot["eval_domain"]["profile_hash"] = snapshot["profile_hash"]
        self._check_cases(cases, snapshot)
        run_id = _new_run_id(str(body.get("benchmark_name") or profile_id))
        queue = self._queue_for(snapshot, business)
        criteria = snapshot["criteria_plan"]
        counting = [item for item in criteria if item.get("counts_toward_progress")]
        if not counting:
            raise EvalError("profile has no scoring criteria")
        total_evaluators = len(counting)
        expected = len(cases) * total_evaluators
        async with await self.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO benchmark_runs (
                        run_id, benchmark_name, status, total_samples, total_evaluators,
                        judge_expected_count, input_sealed, completion_model, config_json,
                        idempotency_key, owner
                    ) VALUES (%s,%s,'initializing',%s,%s,%s,true,%s,%s,%s,%s)
                    """,
                    (
                        run_id,
                        str(body.get("benchmark_name") or profile_id),
                        len(cases),
                        total_evaluators,
                        expected,
                        snapshot["completion_model"],
                        Jsonb(snapshot),
                        idem,
                        body.get("owner"),
                    ),
                )
                for case in cases:
                    await conn.execute(
                        "INSERT INTO input_manifest (run_id, item_id, payload, checksum) VALUES (%s,%s,%s,%s)",
                        (run_id, case["item_id"], Jsonb(case), sha256_text(canonical(case))),
                    )
        try:
            for case in cases:
                await self.dispatch.enqueue(
                    queue,
                    _case_job_id(run_id, case["item_id"]),
                    "case",
                    self._job_data(run_id, case, queue, snapshot),
                    keep=False,
                )
        except Exception as exc:
            await self._fail(run_id, f"enqueue_failed:{exc}")
            raise Unavailable("queue unavailable") from exc
        async with await self.store.connect() as conn:
            await conn.execute(
                """
                UPDATE benchmark_runs
                SET status = 'running', started_at = now(), heartbeat_at = now()
                WHERE run_id = %s AND status = 'initializing'
                """,
                (run_id,),
            )
            await conn.commit()
        run = await self.store.get_run(run_id)
        return self._public_run(run)

    async def publish_dataset(self, body: dict[str, Any]) -> dict[str, Any]:
        domain_id = str(body.get("eval_domain_id") or "")
        if self.conf.domain(domain_id) and not self.conf.eval_domain(domain_id):
            raise EvalError("isolation domain is not an eval domain", code="unknown_eval_domain")
        domain = self.conf.eval_domain(domain_id)
        if not domain:
            raise EvalError("unknown eval domain", code="unknown_eval_domain")
        evaluators = list(domain.get("evaluators") or [])
        if not evaluators:
            raise EvalError("eval domain has no evaluators")
        profile_id = str(domain["profile_id"])
        criteria = [
            {"id": name, "weight": 1, "counts_toward_progress": True}
            for name in evaluators
        ]
        eval_domain = {
            "domain_id": domain_id,
            "domain": domain.get("domain"),
            "dimension": domain.get("dimension") or domain_id,
            "portion": domain.get("portion"),
            "conf_hash": self.conf.file_hashes["eval_domains"],
            "profile_id": profile_id,
            "evaluators": evaluators,
            "rubric_id": domain.get("rubric_id"),
            "score_scale": domain.get("score_scale"),
        }
        created = await self.publish_run(
            {
                **body,
                "profile_id": profile_id,
                "benchmark_name": body.get("benchmark_name") or domain_id,
            },
            criteria_plan=criteria,
            eval_domain=eval_domain,
        )
        return created

    async def publish_domain(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        datasets = self.conf.domain_datasets(name)
        if not datasets:
            raise EvalError("unknown dataset domain", code="unknown_eval_domain")
        runs = []
        for item in datasets:
            runs.append(
                await self.publish_dataset(
                    {
                        **body,
                        "eval_domain_id": item["id"],
                        "benchmark_name": body.get("benchmark_name") or item["id"],
                    }
                )
            )
        return {
            "domain": name,
            "dimensions": [item.get("dimension") for item in datasets],
            "runs": runs,
        }

    async def playground_replay(self, parent_run_id: str, body: dict[str, Any], authorization: str) -> dict[str, Any]:
        self._reject_transport_overrides(body)
        parent = await self.store.get_run(parent_run_id)
        if parent is None:
            raise NotFound("parent run not found")
        if parent["status"] not in TERMINAL_RUN:
            raise Conflict("parent run is not terminal")
        if not body.get("agent_id") or not body.get("triggered_by"):
            raise EvalError("agent_id and triggered_by are required")
        replay_flow = body.get("replay_agent_flow", True)
        profile_id = "playground_replay" if replay_flow else "playground_rejudge"
        domain_id = self.conf.isolation.get("replay_domain")
        domain = self.conf.domain(str(domain_id)) if domain_id else None
        if not domain:
            raise Unavailable("replay domain is not configured")
        if domain.get("queue_prefix") != self.conf.queue_prefix():
            raise Unavailable("replay domain prefix is not served by this process")
        selected = await self._select_parent_cases(parent_run_id, body)
        limit = int(self.conf.app["limits"]["replay_case_limit"])
        if len(selected) > limit:
            raise EvalError(f"replay selection exceeds {limit}")
        snapshot = self._freeze(profile_id, business=str(body.get("business") or "default"))
        parent_snapshot = parent["config_json"]
        score = next((phase for phase in parent_snapshot.get("execution_plan") or [] if phase.get("role") == "score"), None)
        if score:
            for phase in snapshot["execution_plan"]:
                if phase["role"] == "score":
                    phase["execution"] = score["execution"]
                    phase["request_builder"] = score.get("request_builder") or phase["request_builder"]
        if parent_snapshot.get("criteria_plan"):
            snapshot["criteria_plan"] = parent_snapshot["criteria_plan"]
            snapshot["rubric"] = parent_snapshot.get("rubric") or snapshot["rubric"]
        if parent_snapshot.get("plugins", {}).get("attribution"):
            snapshot["plugins"]["attribution"] = parent_snapshot["plugins"]["attribution"]
        snapshot["source"] = profile_id
        snapshot["parent_run_id"] = parent_run_id
        snapshot["replay_agent_id"] = body.get("agent_id")
        snapshot["replay_subscription_id"] = body.get("subscription_id")
        snapshot["replay_agent_name"] = body.get("agent_name")
        snapshot["triggered_by"] = body.get("triggered_by")
        snapshot["replay_header_overrides"] = body.get("header_overrides") or {}
        snapshot["replay_body_overrides"] = body.get("body_overrides") or {}
        snapshot["dispatch_domain"] = {
            "domain_id": domain["id"],
            "conf_hash": self.conf.file_hashes["isolation"],
            "queue_prefix": self.queue_prefix,
        }
        snapshot["profile_hash"] = sha256_text(canonical({
            "profile_id": profile_id,
            "execution_plan": snapshot["execution_plan"],
            "criteria_plan": snapshot["criteria_plan"],
            "parent_run_id": parent_run_id,
        }))
        token = _bearer(authorization)
        if "user_bearer_checkout" in snapshot["credential_requirements"] and not token:
            raise Unauthorized("bearer token is required")
        day = datetime.now().strftime("%Y%m%d")
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS n FROM benchmark_runs WHERE parent_run_id = %s",
                (parent_run_id,),
            )
            count = int((await cur.fetchone())["n"])
            await conn.rollback()
        child_id = f"{parent['benchmark_name']}-{day}-p{count + 1}"
        counting = [item for item in snapshot["criteria_plan"] if item.get("counts_toward_progress")]
        async with await self.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO benchmark_runs (
                        run_id, benchmark_name, status, total_samples, total_evaluators,
                        judge_expected_count, input_sealed, completion_model, config_json,
                        parent_run_id, owner
                    ) VALUES (%s,%s,'initializing',%s,%s,%s,true,%s,%s,%s,%s)
                    """,
                    (
                        child_id,
                        parent["benchmark_name"],
                        len(selected),
                        len(counting),
                        len(selected) * len(counting),
                        snapshot["completion_model"],
                        Jsonb(snapshot),
                        parent_run_id,
                        body.get("triggered_by"),
                    ),
                )
                for case in selected:
                    await conn.execute(
                        "INSERT INTO input_manifest (run_id, item_id, payload, checksum) VALUES (%s,%s,%s,%s)",
                        (child_id, case["item_id"], Jsonb(case), sha256_text(canonical(case))),
                    )
        if token:
            await self.leases.redis.set(self._cred_key(child_id), token, ex=3600)
        queue = snapshot["queue"]
        for case in selected:
            await self.dispatch.enqueue(
                queue,
                _case_job_id(child_id, case["item_id"]),
                "case",
                self._job_data(child_id, case, queue, snapshot),
                keep=False,
            )
        async with await self.store.connect() as conn:
            await conn.execute(
                """
                UPDATE benchmark_runs
                SET status = 'running', started_at = now()
                WHERE run_id = %s
                """,
                (child_id,),
            )
            await conn.commit()
        return {
            "run_id": child_id,
            "parent_run_id": parent_run_id,
            "total_items": len(selected),
            "status": "running",
            "dispatch_domain_id": domain["id"],
            "profile_id": profile_id,
        }

    async def stop(self, run_id: str) -> dict[str, Any]:
        async with await self.store.connect() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT * FROM benchmark_runs WHERE run_id = %s FOR UPDATE",
                    (run_id,),
                )
                run = await cur.fetchone()
                if run is None:
                    raise NotFound("run not found")
                if run["status"] != "running":
                    raise Conflict("run is not running")
                admitted = await conn.execute(
                    "SELECT item_id FROM admissions WHERE run_id = %s",
                    (run_id,),
                )
                admitted_ids = [row["item_id"] for row in await admitted.fetchall()]
                keys = _expand_keys(admitted_ids, run["config_json"]["criteria_plan"])
                await conn.execute(
                    """
                    UPDATE benchmark_runs
                    SET admission_epoch = admission_epoch + 1,
                        admission_closed = true,
                        stop_requested_at = now(),
                        stop_admit_snapshot = %s,
                        stop_progress_keys = %s,
                        stop_target_count = %s,
                        judge_wait_purged = false
                    WHERE run_id = %s AND status = 'running'
                    """,
                    (Jsonb(admitted_ids), Jsonb(keys), len(keys), run_id),
                )
        clean = await self._purge_unadmitted(run_id, set(admitted_ids))
        if clean:
            async with await self.store.connect() as conn:
                await conn.execute(
                    "UPDATE benchmark_runs SET judge_wait_purged = true WHERE run_id = %s AND admission_closed",
                    (run_id,),
                )
                await conn.commit()
            await self.tick(run_id)
        run = await self.store.get_run(run_id)
        return self._public_run(run)

    async def callback(self, external_job_id: str, body: bytes, headers: dict[str, str]) -> dict[str, Any]:
        if await self.leases.redis.get(self._callback_key(external_job_id)):
            return {"applied": False, "replayed": True}
        existing = await self._item_by_external(external_job_id)
        if existing and existing["status"] in {"success", "failed"}:
            return {"applied": False, "replayed": True}
        job = None
        queue_name = ""
        for name in ("benchmark-running", "dataset-running"):
            job = await self.dispatch.get(name, external_job_id)
            if job is not None:
                queue_name = name
                break
        if job is None:
            raise NotFound("callback job not found")
        data = dict(job.data)
        run = await self.store.get_run(data["run_id"])
        if run is None:
            raise NotFound("run not found")
        auth = self.host.load(run["config_json"]["plugins"]["callback_authenticator"])
        if not auth.verify(body, headers):
            raise Unauthorized("callback signature rejected")
        payload = json.loads(body.decode("utf-8"))
        lease = _lease_from(data.get("lease"))
        if lease is None or not await self.leases.still_held(lease):
            return {"applied": False, "reason": "lease_lost"}
        applied = await self._commit_rows(run, data["item_id"], payload.get("rows") or [], external_job_id)
        await self.leases.release(lease)
        try:
            await job.remove()
        except Exception:
            log.info("pending ledger already gone id=%s", external_job_id)
        await self.leases.redis.set(self._callback_key(external_job_id), "1", ex=86400)
        await self._consume_intent(run["run_id"], data["item_id"], data.get("phase_id") or "judge", int(data.get("attempt_epoch") or 1))
        await self.tick(run["run_id"])
        return {"applied": applied, "replayed": not applied, "queue": queue_name}

    async def ingest_message(self, message: dict[str, Any]) -> dict[str, Any]:
        snapshot = self._freeze("streaming_message", business=str(message.get("business") or "default"))
        plugin = self.host.load(snapshot["plugins"]["stream_filter"])
        decision = plugin.decide(message)
        status = decision["status"]
        score = None
        if decision.get("sample"):
            judge = self.host.load(snapshot["execution_plan"][0]["execution"])
            ctx = PhaseContext(
                run_id="",
                item_id=str(message.get("trace_id")),
                phase=snapshot["execution_plan"][0],
                case=message,
                snapshot=snapshot,
            )
            outcome = judge.execute(ctx)
            if isinstance(outcome, Completed):
                rows = outcome.result.get("rows") or []
                score = rows[0]["score"] if rows else None
                status = "recorded"
        async with await self.store.connect() as conn:
            await conn.execute(
                """
                INSERT INTO trace_pipeline_records
                    (trace_id, project_id, agent_name, evaluator_name, status, score, trace_meta_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (trace_id, project_id, agent_name, evaluator_name)
                DO NOTHING
                """,
                (
                    str(message.get("trace_id")),
                    str(message.get("project_id") or "default"),
                    str(message.get("agent_name") or "default"),
                    str(message.get("evaluator_name") or "default"),
                    status,
                    score,
                    Jsonb({"message": message, "decision": decision}),
                ),
            )
            await conn.commit()
        return {"status": status, "score": score, "run_created": False}

    async def handle(self, job, token: str) -> None:
        data = dict(job.data)
        owned: LeasePair | None = None
        admitted = False
        hold = False
        busy_view: dict[str, Any] = self.conf.app
        try:
            drop = await self._must_drop(data)
            if drop == "ignore":
                return
            if drop == "require_terminal":
                await self._terminal(data, "run_terminal_cancel")
                await self.tick(data["run_id"])
                return
            run = await self.store.get_run(data["run_id"])
            if run is None or run["status"] != "running":
                raise SlotBusy()
            busy_view = run["config_json"]
            if self._budget_exhausted(data, run["config_json"]):
                await self._terminal(data, "retry_window_exhausted")
                await self.tick(data["run_id"])
                return
            manifest = await self._manifest(data["run_id"], data["item_id"])
            if manifest["checksum"] != data["checksum"]:
                await self._terminal(data, "checksum_mismatch")
                await self.tick(data["run_id"])
                return
            snapshot = run["config_json"]
            plan = snapshot["execution_plan"]
            while True:
                phase = _phase(plan, data.get("phase_id"))
                if await self._phase_done(data["run_id"], data["item_id"], phase["phase_id"]):
                    nxt = _next_phase(plan, phase["phase_id"])
                    if owned:
                        await self.leases.release(owned)
                        owned = None
                    if nxt is None:
                        await self.tick(data["run_id"])
                        return
                    data["phase_id"] = nxt["phase_id"]
                    await job.updateData(data)
                    continue
                if owned is None:
                    owned = await self._acquire(snapshot, data)
                    if owned is None:
                        raise SlotBusy()
                    stamped = data.get("first_running_at") or now_ms()
                    data["first_running_at"] = stamped
                    await job.updateData(data)
                    if not await self._register_lease(owned, data):
                        await self.leases.release(owned)
                        owned = None
                        raise SlotBusy()
                    await self._register_ledger(data, owned, "inflight")
                    claim = await self._claim(data["run_id"], data["item_id"])
                    if claim == "epoch_closed":
                        await self.leases.release(owned)
                        owned = None
                        await self._clear_ledger(data)
                        return
                    if claim != "admitted":
                        raise Unavailable("admission unavailable")
                    admitted = True
                if not await self.leases.still_held(owned):
                    await self._requeue(job, token, data, snapshot, Retryable("lease_lost"), owned)
                    owned = None
                    return
                ctx = PhaseContext(
                    run_id=data["run_id"],
                    item_id=data["item_id"],
                    phase=phase,
                    case=manifest["payload"],
                    snapshot=snapshot,
                    credential_ready=bool(await self.leases.redis.exists(self._cred_key(data["run_id"]))),
                )
                from benchmark.kernel.casegraph import execute_phase

                outcome = await execute_phase(self.host, ctx)
                if not isinstance(outcome, OUTCOMES):
                    raise PermanentError("unknown_outcome")
                if isinstance(outcome, SubmitRequest):
                    if not await self.leases.still_held(owned):
                        outcome = Retryable("lease_lost")
                    else:
                        await self._write_intent(data, phase, owned, outcome.request)
                        hold = True
                        held = await self._provider_submit(data, phase, owned, outcome.request)
                        if isinstance(held, ProviderSyncCompleted):
                            await self._consume_intent(data["run_id"], data["item_id"], phase["phase_id"], data["attempt_epoch"])
                            hold = False
                            outcome = Completed(held.payload)
                        elif isinstance(held, ProviderAcceptedPendingRecorded):
                            hold = True
                            owned = None
                            return
                        elif isinstance(held, ProviderSubmitRejected):
                            try:
                                await self._abort_intent(data, phase)
                                hold = False
                            except Exception:
                                await self._mark_intent(data, phase, "abort_failed")
                                owned = None
                                return
                            await self.leases.release(owned)
                            owned = None
                            await self._requeue(job, token, data, snapshot, Retryable("provider_rejected"), None)
                            return
                        else:
                            await self._mark_intent(data, phase, "non_enum_hold")
                            owned = None
                            return
                if isinstance(outcome, TerminalFailure):
                    if not await self.leases.still_held(owned):
                        await self._requeue(job, token, data, snapshot, Retryable("lease_lost_before_commit"), owned)
                        owned = None
                        return
                    await self._terminal(data, outcome.error)
                    await self.leases.release(owned)
                    owned = None
                    await self.tick(data["run_id"])
                    return
                if isinstance(outcome, Completed):
                    if not await self.leases.still_held(owned):
                        await self._requeue(job, token, data, snapshot, Retryable("lease_lost_before_commit"), owned)
                        owned = None
                        return
                    if phase["role"] == "produce_output":
                        if phase.get("counts_toward_progress"):
                            raise PermanentError("produce_output_counts_progress")
                        await self._commit_output(data, phase, outcome.result, _detach_graph(outcome))
                        nxt = _next_phase(plan, phase["phase_id"])
                        await self.leases.release(owned)
                        owned = None
                        if nxt is None:
                            await self._terminal(data, "execution_plan_missing_score")
                            await self.tick(data["run_id"])
                            return
                        data["phase_id"] = nxt["phase_id"]
                        await job.updateData(data)
                        manifest = await self._manifest(data["run_id"], data["item_id"])
                        continue
                    if phase["role"] == "score":
                        await self._commit_rows(run, data["item_id"], outcome.result.get("rows") or [])
                        await self._save_graph(data, phase, _detach_graph(outcome))
                        await self.leases.release(owned)
                        owned = None
                        await self._clear_ledger(data)
                        await self.tick(data["run_id"])
                        return
                    raise PermanentError("unknown_phase_role")
                if isinstance(outcome, (Retryable, InvalidateAndReset)):
                    await self.leases.release(owned)
                    owned = None
                    await self._requeue(job, token, data, snapshot, outcome, None)
                    return
        except SlotBusy:
            if owned is not None:
                await self.leases.release(owned)
                owned = None
                if not admitted:
                    await self._clear_ledger(data)
            await self._delay(job, token, data, busy_view)
        except WaitingChildrenError:
            raise
        except Exception as exc:
            if hold:
                await self._mark_intent(data, {"phase_id": data.get("phase_id")}, str(exc))
                return
            if owned is not None:
                await self.leases.release(owned)
                owned = None
            snapshot = (await self.store.get_run(data["run_id"]) or {}).get("config_json") or {}
            await self._requeue(job, token, data, snapshot, self._classify(exc), None)
        finally:
            if owned is not None and not hold:
                await self.leases.release(owned)
                if not admitted:
                    await self._clear_ledger(data)

    async def handle_attribution(self, job, token: str) -> None:
        data = dict(job.data)
        run = await self.store.get_run(data["run_id"])
        if run is None or run["status"] not in {"running"}:
            return
        lease = await self._acquire(run["config_json"], data, kind="attr")
        if lease is None:
            await self._delay(job, token, data, run["config_json"])
            return
        try:
            if not await self._register_lease(lease, data):
                return
            manifest = await self._manifest(data["run_id"], data["item_id"])
            plugin = self.host.load(run["config_json"]["plugins"]["attribution"])
            ctx = PhaseContext(
                run_id=data["run_id"],
                item_id=data["item_id"],
                phase={"phase_id": "attribution", "role": "score"},
                case=manifest["payload"],
                snapshot=run["config_json"],
            )
            detail = await _call(plugin, "attribute", ctx)
            await self._record_attribution(run, data["item_id"], data["criterion_id"], detail)
            await self.tick(data["run_id"])
        finally:
            await self.leases.release(lease)

    async def tick(self, run_id: str) -> bool:
        await self._drain_outbox(run_id)
        run = await self.store.get_run(run_id)
        if run is None or run["status"] != "running":
            if run and run["status"] in {"aggregating", "generating_report"}:
                await self._finish(run_id)
            return False
        snapshot = run["config_json"]
        keys = await self._required_keys(run)
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                "SELECT item_id, criterion_id FROM attribution_decisions WHERE run_id = %s",
                (run_id,),
            )
            decided = {(row["item_id"], row["criterion_id"]) for row in await cur.fetchall()}
            sealed = all(key in decided for key in keys)
            kinds = ["attribution_schedule", "cleanup"]
            if (snapshot.get("plugins", {}).get("dispatch_policy", {}).get("config") or {}).get("blocks_completion"):
                kinds.append("dispatch")
            cur = await conn.execute(
                """
                SELECT count(*) AS n FROM outbox
                WHERE run_id = %s AND status <> 'delivered' AND event_kind = ANY(%s)
                """,
                (run_id, kinds),
            )
            pending = int((await cur.fetchone())["n"])
            extra_plugin = self.host.load(snapshot["plugins"]["completion_extra"])
            extra = bool(extra_plugin.ready({
                "run_id": run_id,
                "input_sealed": run["input_sealed"],
                "evaluated_items": run["evaluated_items"],
                "failed_items": run["failed_items"],
                "total_samples": run["total_samples"],
            }))
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE benchmark_runs
                    SET attribution_decisions_sealed = %s,
                        required_outbox_delivered = %s,
                        completion_extra_ready = %s
                    WHERE run_id = %s AND status = 'running'
                    """,
                    (sealed, pending == 0, extra, run_id),
                )
                cur = await conn.execute(
                    """
                    UPDATE benchmark_runs
                    SET status = 'aggregating', heartbeat_at = now()
                    WHERE run_id = %s
                      AND status = 'running'
                      AND input_sealed
                      AND attribution_decisions_sealed
                      AND required_outbox_delivered
                      AND completion_extra_ready
                      AND (
                        (
                          admission_closed
                          AND judge_wait_purged
                          AND stop_admit_snapshot IS NOT NULL
                          AND NOT EXISTS (
                            SELECT 1
                            FROM jsonb_array_elements(COALESCE(stop_progress_keys, '[]'::jsonb)) AS k
                            WHERE NOT EXISTS (
                              SELECT 1 FROM benchmark_run_items i
                              WHERE i.run_id = benchmark_runs.run_id
                                AND i.item_id = k->>'item_id'
                                AND i.criterion_id = k->>'criterion_id'
                                AND i.status IN ('success', 'failed')
                            )
                          )
                          AND NOT EXISTS (
                            SELECT 1
                            FROM jsonb_array_elements(COALESCE(stop_progress_keys, '[]'::jsonb)) AS k
                            WHERE NOT EXISTS (
                              SELECT 1 FROM attribution_decisions d
                              WHERE d.run_id = benchmark_runs.run_id
                                AND d.item_id = k->>'item_id'
                                AND d.criterion_id = k->>'criterion_id'
                            )
                          )
                        )
                        OR (
                          NOT admission_closed
                          AND (
                            CASE
                              WHEN completion_model = 'external' THEN
                                judge_received_count >= judge_expected_count
                                AND attribution_received_count >= attribution_expected_count
                              ELSE
                                evaluated_items + failed_items >= total_samples * GREATEST(total_evaluators, 1)
                            END
                          )
                        )
                      )
                    RETURNING run_id
                    """,
                    (run_id,),
                )
                won = await cur.fetchone()
        if won:
            await self._finish(run_id)
            return True
        return False

    async def healer_tick(self) -> None:
        lease_ms = int(self.conf.app["healer"]["lease_ms"])
        if not await self.leases.try_lead(lease_ms):
            return
        grace = int(self.conf.app["healer"]["orphan_grace_ms"])
        current = await self.leases.redis_now_ms()
        for key in await self.leases.indexed_keys():
            for member, score in await self.leases.members(key):
                if current - score <= grace:
                    continue
                async with await self.store.connect() as conn:
                    cur = await conn.execute(
                        """
                        INSERT INTO lease_ledger (lease_id, capacity_key, abandoned)
                        VALUES (%s, %s, true)
                        ON CONFLICT (lease_id) DO NOTHING
                        RETURNING lease_id
                        """,
                        (member, key),
                    )
                    inserted = await cur.fetchone()
                    await conn.commit()
                if inserted:
                    await self.leases.release_one(key, member)
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                "SELECT run_id FROM benchmark_runs WHERE status IN ('aggregating', 'generating_report')"
            )
            stuck = await cur.fetchall()
        for row in stuck:
            await self._finish(row["run_id"])

    async def save_registration(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body["name"])
        business = str(body["business"])
        variant = str(body["variant_key"])
        scope = str(body.get("agent_scope") or "default")
        customization = body.get("customization") or {}
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                """
                SELECT * FROM agent_registrations
                WHERE name = %s AND business = %s AND variant_key = %s AND agent_scope = %s
                ORDER BY version DESC
                """,
                (name, business, variant, scope),
            )
            rows = await cur.fetchall()
            for row in rows:
                if row["customization"] == customization:
                    await conn.rollback()
                    return {"registration_id": row["registration_id"], "version": row["version"], "replayed": True}
            if body.get("registration_id"):
                raise Conflict("customization is immutable")
            version = (rows[0]["version"] + 1) if rows else 1
            internal = rows[0]["internal_agent_name"] if rows else f"agx_{_slug(name)}_{version:010d}"
            if not rows:
                cur = await conn.execute("SELECT count(*) AS n FROM agent_registrations")
                seq = int((await cur.fetchone())["n"]) + 1
                internal = f"agx_{_slug(name)}_{seq:010d}"
            registration_id = f"{_slug(name)}--{_slug(scope)}--{_slug(business)}--{_slug(variant)}--v{version}"
            if body.get("content_ref"):
                path = Path("var/bundles") / str(body["content_ref"]) / "manifest.json"
                if not path.is_file():
                    raise Unavailable("bundle is not available")
            await conn.execute(
                """
                INSERT INTO agent_registrations (
                    registration_id, name, business, variant_key, agent_scope, version,
                    internal_agent_name, is_default, customization, content_ref
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    registration_id,
                    name,
                    business,
                    variant,
                    scope,
                    version,
                    internal,
                    not rows,
                    Jsonb(customization),
                    body.get("content_ref"),
                ),
            )
            await conn.commit()
        return {"registration_id": registration_id, "version": version, "internal_agent_name": internal}

    def _freeze(self, profile_id: str, *, business: str) -> dict[str, Any]:
        return self._freeze_profile(profile_id, self.conf.profile(profile_id), business=business)

    def _freeze_profile(self, profile_id: str, profile: dict[str, Any], *, business: str) -> dict[str, Any]:
        plugins = {}
        for slot in REQUIRED_SLOTS:
            name = (profile.get("plugins") or {}).get(slot)
            if not name:
                raise EvalError(f"profile {profile_id} is missing plugin slot {slot}")
            plugins[slot] = self.host.freeze(slot, name)
        if profile.get("plugins", {}).get("stream_filter"):
            plugins["stream_filter"] = self.host.freeze("stream_filter", profile["plugins"]["stream_filter"])
        plan = []
        for phase in profile.get("execution_plan") or []:
            if phase["role"] not in {"produce_output", "score"}:
                raise EvalError("unknown phase role")
            if phase["role"] == "produce_output" and phase.get("counts_toward_progress"):
                raise EvalError("produce_output cannot count toward progress")
            execution = self.host.freeze("execution", phase["execution"])
            builder = self.host.freeze("request_builder", phase["request_builder"])
            plan.append({
                **phase,
                "request_builder": builder,
                "execution": execution,
            })
        if not plan:
            raise EvalError("execution_plan is empty")
        auth = plugins["callback_authenticator"]["plugin_id"]
        if any(phase["execution"].get("config", {}).get("submits") for phase in plan):
            pass
        loaded_exec = [self.host.load(phase["execution"]) for phase in plan]
        if any(getattr(plugin, "submits", False) for plugin in loaded_exec) and auth == "local_only":
            raise EvalError("async profile requires a callback authenticator")
        criteria = list((profile.get("rubric") or {}).get("criteria") or [])
        snapshot = {
            "snapshot_schema_version": 1,
            "profile_id": profile_id,
            "plugins": plugins,
            "execution_plan": plan,
            "credential_requirements": list(profile.get("credential_requirements") or ["none"]),
            "remote_execution": profile.get("remote_execution") or "forbidden",
            "completion_model": profile.get("completion_model") or "count",
            "slot_kind": profile.get("slot_kind") or "sync",
            "queue": profile.get("queue") or "benchmark-cases",
            "triage_mode": profile.get("triage_mode") or "fused",
            "rubric": profile.get("rubric") or {},
            "criteria_plan": criteria,
            "business": business,
            "retry": self.conf.app["retry"],
            "slots": self.conf.app["slots"],
            "slot_busy": self.conf.app["slot_busy"],
            "limits": self.conf.app["limits"],
            "conf_hash": self.conf.file_hashes,
            "blocks_completion": bool(plugins["dispatch_policy"]["config"].get("blocks_completion", False)),
        }
        snapshot["profile_hash"] = sha256_text(canonical({
            "profile_id": profile_id,
            "plugins": {key: value["artifact_hash"] for key, value in plugins.items()},
            "execution_plan": [
                {
                    "phase_id": phase["phase_id"],
                    "role": phase["role"],
                    "execution": phase["execution"]["artifact_hash"],
                }
                for phase in plan
            ],
            "criteria_plan": criteria,
        }))
        return snapshot

    def _queue_for(self, snapshot: dict[str, Any], business: str) -> str:
        queue = snapshot["queue"]
        if queue == "benchmark-cases" and business in set(self.conf.l2_business):
            return "benchmark-cases-l2"
        return queue

    def _check_cases(self, cases: list[dict[str, Any]], snapshot: dict[str, Any]) -> None:
        limit = int(snapshot["limits"]["max_cases"])
        if not cases:
            raise EvalError("cases are required")
        if len(cases) > limit:
            raise EvalError("too many cases")
        max_bytes = int(snapshot["limits"]["max_case_bytes"])
        for case in cases:
            if len(canonical(case).encode("utf-8")) > max_bytes:
                raise EvalError("case exceeds the size limit")

    def _job_data(self, run_id: str, case: dict[str, Any], queue: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "item_id": case["item_id"],
            "queue": queue,
            "phase_id": snapshot["execution_plan"][0]["phase_id"],
            "attempt_epoch": 1,
            "retry_count": 0,
            "first_running_at": None,
            "checksum": sha256_text(canonical(case)),
            "business": snapshot.get("business"),
        }

    async def _must_drop(self, data: dict[str, Any]) -> str | None:
        run = await self.store.get_run(data["run_id"])
        if run is None:
            return "ignore"
        if run["status"] == "initializing":
            raise SlotBusy()
        if run["status"] == "running":
            return None
        admitted = await self._is_admitted(data["run_id"], data["item_id"])
        snapshot = run.get("stop_admit_snapshot") or []
        if admitted or data["item_id"] in snapshot:
            return "require_terminal"
        return "ignore"

    async def _is_admitted(self, run_id: str, item_id: str) -> bool:
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM admissions WHERE run_id = %s AND item_id = %s",
                (run_id, item_id),
            )
            return await cur.fetchone() is not None

    async def _claim(self, run_id: str, item_id: str) -> str:
        async with await self.store.connect() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT admission_closed, stop_admit_snapshot FROM benchmark_runs WHERE run_id = %s FOR UPDATE",
                    (run_id,),
                )
                run = await cur.fetchone()
                if run is None:
                    return "unavailable"
                cur = await conn.execute(
                    "SELECT 1 FROM admissions WHERE run_id = %s AND item_id = %s",
                    (run_id, item_id),
                )
                if await cur.fetchone() is not None or item_id in (run["stop_admit_snapshot"] or []):
                    return "admitted"
                if run["admission_closed"]:
                    return "epoch_closed"
                await conn.execute(
                    "INSERT INTO admissions (run_id, item_id, epoch) VALUES (%s,%s,0)",
                    (run_id, item_id),
                )
                return "admitted"

    async def _acquire(self, snapshot: dict[str, Any], data: dict[str, Any], kind: str | None = None) -> LeasePair | None:
        kind = kind or snapshot.get("slot_kind") or "sync"
        slots = snapshot["slots"][kind]
        return await self.leases.acquire_pair(
            kind=kind,
            node_id=self.node_id,
            run_id=data["run_id"],
            node_limit=int(slots["per_node_per_run"]),
            cluster_limit=int(slots["cluster_soft_cap"]),
        )

    async def _register_lease(self, lease: LeasePair, data: dict[str, Any]) -> bool:
        async with await self.store.connect() as conn:
            async with conn.transaction():
                for key, lease_id in lease.ids():
                    cur = await conn.execute(
                        """
                        INSERT INTO lease_ledger (lease_id, capacity_key, run_id, item_id, abandoned)
                        VALUES (%s,%s,%s,%s,false)
                        ON CONFLICT (lease_id) DO NOTHING
                        RETURNING lease_id
                        """,
                        (lease_id, key, data["run_id"], data["item_id"]),
                    )
                    if await cur.fetchone() is None:
                        cur = await conn.execute(
                            "SELECT abandoned FROM lease_ledger WHERE lease_id = %s",
                            (lease_id,),
                        )
                        row = await cur.fetchone()
                        if row and row["abandoned"]:
                            return False
                await conn.execute(
                    "UPDATE benchmark_runs SET heartbeat_at = now(), owner = %s WHERE run_id = %s",
                    (self.node_id, data["run_id"]),
                )
            return True

    async def _register_ledger(self, data: dict[str, Any], lease: LeasePair, name: str) -> None:
        running = RUNNING_QUEUE[data["queue"]]
        payload = {**data, "lease": _lease_dict(lease), "ledger": name}
        job_id = _case_job_id(data["run_id"], data["item_id"])
        existing = await self.dispatch.get(running, job_id)
        if existing is not None:
            await existing.updateData(payload)
            return
        await self.dispatch.enqueue(running, job_id, name, payload, keep=True)

    async def _clear_ledger(self, data: dict[str, Any]) -> None:
        running = RUNNING_QUEUE.get(data.get("queue") or "", "benchmark-running")
        existing = await self.dispatch.get(running, _case_job_id(data["run_id"], data["item_id"]))
        if existing is not None:
            try:
                await existing.remove()
            except Exception:
                return

    async def _manifest(self, run_id: str, item_id: str) -> dict[str, Any]:
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                "SELECT * FROM input_manifest WHERE run_id = %s AND item_id = %s",
                (run_id, item_id),
            )
            row = await cur.fetchone()
        if row is None:
            raise PermanentError("manifest_missing")
        return row

    async def _phase_done(self, run_id: str, item_id: str, phase_id: str) -> bool:
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM phase_cursors WHERE run_id = %s AND item_id = %s AND phase_id = %s",
                (run_id, item_id, phase_id),
            )
            return await cur.fetchone() is not None

    async def _commit_output(
        self,
        data: dict[str, Any],
        phase: dict[str, Any],
        result: dict[str, Any],
        graph: dict[str, Any] | None = None,
    ) -> None:
        stored = dict(result)
        if graph:
            stored["_graph"] = graph
        async with await self.store.connect() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT payload, checksum FROM input_manifest WHERE run_id = %s AND item_id = %s FOR UPDATE",
                    (data["run_id"], data["item_id"]),
                )
                row = await cur.fetchone()
                payload = dict(row["payload"])
                payload["output"] = result.get("output")
                await conn.execute(
                    "UPDATE input_manifest SET payload = %s WHERE run_id = %s AND item_id = %s",
                    (Jsonb(payload), data["run_id"], data["item_id"]),
                )
                await conn.execute(
                    """
                    INSERT INTO phase_cursors (run_id, item_id, phase_id, output)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (run_id, item_id, phase_id)
                    DO UPDATE SET output = EXCLUDED.output
                    """,
                    (data["run_id"], data["item_id"], phase["phase_id"], Jsonb(stored)),
                )

    async def _save_graph(self, data: dict[str, Any], phase: dict[str, Any], graph: dict[str, Any] | None) -> None:
        if not graph:
            return
        async with await self.store.connect() as conn:
            await conn.execute(
                """
                INSERT INTO phase_cursors (run_id, item_id, phase_id, output)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (run_id, item_id, phase_id)
                DO UPDATE SET output = COALESCE(phase_cursors.output, '{}'::jsonb) || EXCLUDED.output
                """,
                (data["run_id"], data["item_id"], phase["phase_id"], Jsonb({"_graph": graph})),
            )
            await conn.commit()

    async def _commit_rows(self, run: dict[str, Any], item_id: str, rows: list[dict[str, Any]], external_job_id: str | None = None) -> bool:
        snapshot = run["config_json"]
        planned = [item for item in snapshot["criteria_plan"] if item.get("counts_toward_progress")]
        by_id = {row["criterion_id"]: row for row in rows}
        normalized = []
        for criterion in planned:
            row = by_id.get(criterion["id"])
            if row is None:
                normalized.append({
                    "criterion_id": criterion["id"],
                    "status": "failed",
                    "score": None,
                    "reason": "missing_criterion",
                })
            else:
                normalized.append(row)
        manifest = await self._manifest(run["run_id"], item_id)
        trigger = self.host.load(snapshot["plugins"]["trigger_policy"])
        applied = False
        async with await self.store.connect() as conn:
            async with conn.transaction():
                fresh = await conn.execute(
                    "SELECT * FROM benchmark_runs WHERE run_id = %s FOR UPDATE",
                    (run["run_id"],),
                )
                current = await fresh.fetchone()
                for row in normalized:
                    status = row.get("status") or "success"
                    if status not in {"success", "failed"}:
                        status = "failed"
                    await conn.execute(
                        """
                        INSERT INTO benchmark_run_items (
                            run_id, item_id, criterion_id, evaluator_ref, status, input, output, external_job_id
                        ) VALUES (%s,%s,%s,%s,'running',%s,%s,%s)
                        ON CONFLICT (run_id, item_id, criterion_id) DO NOTHING
                        """,
                        (
                            run["run_id"],
                            item_id,
                            row["criterion_id"],
                            row.get("evaluator_ref") or row["criterion_id"],
                            Jsonb(manifest["payload"].get("input")),
                            Jsonb(manifest["payload"].get("output")),
                            external_job_id,
                        ),
                    )
                    cur = await conn.execute(
                        """
                        UPDATE benchmark_run_items
                        SET status = %s, score = %s, reason = %s, judge_json = %s,
                            error_message = %s, external_job_id = COALESCE(%s, external_job_id),
                            updated_at = now()
                        WHERE run_id = %s AND item_id = %s AND criterion_id = %s
                          AND status NOT IN ('success', 'failed')
                        RETURNING criterion_id
                        """,
                        (
                            status,
                            row.get("score"),
                            row.get("reason"),
                            Jsonb(row),
                            row.get("reason") if status == "failed" else None,
                            external_job_id,
                            run["run_id"],
                            item_id,
                            row["criterion_id"],
                        ),
                    )
                    changed = await cur.fetchone()
                    if changed is None:
                        continue
                    applied = True
                    dedupe = f"{item_id}|{row['criterion_id']}"
                    cur = await conn.execute(
                        """
                        INSERT INTO run_progress_events (run_id, stage, dedupe_key)
                        VALUES (%s, 'judge', %s)
                        ON CONFLICT DO NOTHING
                        RETURNING dedupe_key
                        """,
                        (run["run_id"], dedupe),
                    )
                    if await cur.fetchone() is not None:
                        column = "failed_items" if status == "failed" else "evaluated_items"
                        await conn.execute(
                            f"""
                            UPDATE benchmark_runs
                            SET {column} = {column} + 1,
                                judge_received_count = judge_received_count + 1
                            WHERE run_id = %s
                            """,
                            (run["run_id"],),
                        )
                    if trigger.should_trigger({**row, "status": status}, snapshot.get("rubric") or {}):
                        await conn.execute(
                            """
                            INSERT INTO outbox (run_id, event_kind, dedupe_key, status, payload)
                            VALUES (%s, 'attribution_schedule', %s, 'pending', %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                run["run_id"],
                                dedupe,
                                Jsonb({"item_id": item_id, "criterion_id": row["criterion_id"]}),
                            ),
                        )
                        await conn.execute(
                            """
                            UPDATE benchmark_runs
                            SET attribution_expected_count = attribution_expected_count + 1
                            WHERE run_id = %s
                              AND NOT EXISTS (
                                SELECT 1 FROM run_progress_events
                                WHERE run_id = %s AND stage = 'attr_expected' AND dedupe_key = %s
                              )
                            """,
                            (run["run_id"], run["run_id"], dedupe),
                        )
                        await conn.execute(
                            """
                            INSERT INTO run_progress_events (run_id, stage, dedupe_key)
                            VALUES (%s, 'attr_expected', %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (run["run_id"], dedupe),
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO attribution_decisions (run_id, item_id, criterion_id, detail)
                            VALUES (%s,%s,%s,%s)
                            ON CONFLICT DO NOTHING
                            """,
                            (run["run_id"], item_id, row["criterion_id"], Jsonb({"triggered": False})),
                        )
                if current is None:
                    return False
        return applied

    async def _terminal(self, data: dict[str, Any], error: str) -> None:
        run = await self.store.get_run(data["run_id"])
        if run is None:
            return
        criteria = [item for item in run["config_json"]["criteria_plan"] if item.get("counts_toward_progress")]
        rows = [
            {"criterion_id": item["id"], "status": "failed", "score": None, "reason": error}
            for item in criteria
        ]
        await self._commit_rows(run, data["item_id"], rows)

    async def _safe_terminal(self, job, error: str) -> None:
        try:
            await self._terminal(dict(job.data), error)
            await self.tick(job.data["run_id"])
        except Exception:
            log.exception("safe terminal failed")

    async def _record_attribution(self, run: dict[str, Any], item_id: str, criterion_id: str, detail: dict[str, Any]) -> None:
        before = await self._item(run["run_id"], item_id, criterion_id)
        async with await self.store.connect() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE benchmark_run_items
                    SET attribution_detail_json = %s, updated_at = now()
                    WHERE run_id = %s AND item_id = %s AND criterion_id = %s
                    """,
                    (Jsonb(detail), run["run_id"], item_id, criterion_id),
                )
                cur = await conn.execute(
                    """
                    INSERT INTO attribution_decisions (run_id, item_id, criterion_id, detail)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT DO NOTHING
                    RETURNING criterion_id
                    """,
                    (run["run_id"], item_id, criterion_id, Jsonb(detail)),
                )
                if await cur.fetchone() is not None:
                    await conn.execute(
                        """
                        UPDATE benchmark_runs
                        SET attribution_received_count = attribution_received_count + 1
                        WHERE run_id = %s
                        """,
                        (run["run_id"],),
                    )
        after = await self._item(run["run_id"], item_id, criterion_id)
        if before and after and (before["status"] != after["status"] or before["score"] != after["score"]):
            raise PermanentError("attribution_changed_judge")

    async def _item_by_external(self, external_job_id: str) -> dict[str, Any] | None:
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                """
                SELECT status, score, run_id FROM benchmark_run_items
                WHERE external_job_id = %s
                LIMIT 1
                """,
                (external_job_id,),
            )
            return await cur.fetchone()

    async def _item(self, run_id: str, item_id: str, criterion_id: str) -> dict[str, Any] | None:
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                """
                SELECT status, score FROM benchmark_run_items
                WHERE run_id = %s AND item_id = %s AND criterion_id = %s
                """,
                (run_id, item_id, criterion_id),
            )
            return await cur.fetchone()

    async def _drain_outbox(self, run_id: str) -> None:
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                """
                SELECT * FROM outbox
                WHERE run_id = %s AND status = 'pending' AND event_kind = 'attribution_schedule'
                """,
                (run_id,),
            )
            rows = await cur.fetchall()
        for row in rows:
            payload = row["payload"]
            await self.dispatch.enqueue(
                "attribution-wait",
                _attr_job_id(run_id, payload["item_id"], payload["criterion_id"]),
                "attribution",
                {
                    "run_id": run_id,
                    "item_id": payload["item_id"],
                    "criterion_id": payload["criterion_id"],
                    "queue": "attribution-wait",
                    "phase_id": "attribution",
                    "attempt_epoch": 1,
                    "retry_count": 0,
                    "checksum": "",
                },
                keep=False,
            )
            async with await self.store.connect() as conn:
                await conn.execute(
                    """
                    UPDATE outbox SET status = 'delivered'
                    WHERE run_id = %s AND event_kind = %s AND dedupe_key = %s
                    """,
                    (run_id, row["event_kind"], row["dedupe_key"]),
                )
                await conn.commit()

    async def _required_keys(self, run: dict[str, Any]) -> list[tuple[str, str]]:
        criteria = [item["id"] for item in run["config_json"]["criteria_plan"] if item.get("counts_toward_progress")]
        if run["admission_closed"]:
            ids = list(run["stop_admit_snapshot"] or [])
        else:
            async with await self.store.connect() as conn:
                cur = await conn.execute(
                    "SELECT item_id FROM input_manifest WHERE run_id = %s",
                    (run["run_id"],),
                )
                ids = [row["item_id"] for row in await cur.fetchall()]
        return [(item_id, criterion_id) for item_id in ids for criterion_id in criteria]

    async def _finish(self, run_id: str) -> None:
        run = await self.store.get_run(run_id)
        if run is None or run["status"] in TERMINAL_RUN:
            return
        if run["status"] == "aggregating":
            items = await self.store.list_items(run_id)
            plugin = self.host.load(run["config_json"]["plugins"]["aggregation"])
            score = plugin.aggregate(items, run["config_json"].get("rubric") or {})
            metrics = {
                "total_samples": run["total_samples"],
                "evaluated_items": run["evaluated_items"],
                "failed_items": run["failed_items"],
                "unevaluated": max(run["total_samples"] - _distinct_items(items), 0),
                "final_score": score,
                "stopped": bool(run["admission_closed"]),
            }
            report_body = self.host.load(run["config_json"]["plugins"]["report"]).render({"metrics": metrics, "items": items})
            async with await self.store.connect() as conn:
                cur = await conn.execute(
                    """
                    UPDATE benchmark_runs
                    SET status = 'generating_report', metrics_json = %s, final_score = %s
                    WHERE run_id = %s AND status = 'aggregating'
                    RETURNING run_id
                    """,
                    (Jsonb(metrics), score, run_id),
                )
                won = await cur.fetchone()
                await conn.commit()
            if not won:
                return
            run = await self.store.get_run(run_id)
            report_body = report_body
        else:
            metrics = run["metrics_json"] or {}
            report_body = {"summary": metrics}
            score = run["final_score"]
        if run["status"] != "generating_report":
            return
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / f"{run_id}.json"
        path.write_text(json.dumps({"metrics": metrics, "report": report_body}, default=str), encoding="utf-8")
        final_status = "stopped" if run["admission_closed"] else "completed"
        meta = {"path": str(path), "bytes": path.stat().st_size}
        async with await self.store.connect() as conn:
            await conn.execute(
                """
                UPDATE benchmark_runs
                SET status = %s, report_meta_json = %s, finished_at = now(), final_score = %s
                WHERE run_id = %s AND status = 'generating_report'
                """,
                (final_status, Jsonb(meta), score, run_id),
            )
            await conn.commit()
        await self.leases.redis.delete(self._cred_key(run_id))

    async def _fail(self, run_id: str, message: str) -> None:
        async with await self.store.connect() as conn:
            await conn.execute(
                """
                UPDATE benchmark_runs
                SET status = 'failed', error_message = %s, finished_at = now()
                WHERE run_id = %s AND status NOT IN ('completed', 'stopped', 'failed')
                """,
                (message, run_id),
            )
            await conn.commit()

    def _budget_exhausted(self, data: dict[str, Any], snapshot: dict[str, Any]) -> bool:
        retry = snapshot.get("retry") or {}
        if int(data.get("retry_count") or 0) >= int(retry.get("max_attempts") or 1):
            return True
        started = data.get("first_running_at")
        if not started:
            return False
        return now_ms() - int(started) >= int(retry.get("retry_max_duration_ms") or 0)

    def _classify(self, exc: Exception):
        if isinstance(exc, PermanentError):
            return TerminalFailure(exc.code)
        if isinstance(exc, (RetryableError, TimeoutError, ConnectionError)):
            return Retryable(str(exc))
        return TerminalFailure(exc.__class__.__name__)

    async def _requeue(self, job, token: str, data: dict[str, Any], snapshot: dict[str, Any], outcome, lease: LeasePair | None) -> None:
        if lease is not None:
            await self.leases.release(lease)
        phase = _phase(snapshot.get("execution_plan") or [{"phase_id": data.get("phase_id"), "role": "score", "side_effect_class": "read_only"}], data.get("phase_id"))
        if isinstance(outcome, Retryable) and phase.get("side_effect_class") == "external_write":
            outcome = InvalidateAndReset(outcome.error)
        if isinstance(outcome, TerminalFailure) or isinstance(outcome, PermanentError):
            await self._terminal(data, getattr(outcome, "error", "terminal"))
            await self.tick(data["run_id"])
            return
        if not data.get("first_running_at"):
            await self._terminal(data, "missing_first_running_at")
            await self.tick(data["run_id"])
            return
        if self._budget_exhausted(data, snapshot):
            await self._terminal(data, "retry_window_exhausted")
            await self.tick(data["run_id"])
            return
        if isinstance(outcome, InvalidateAndReset):
            data["attempt_epoch"] = int(data.get("attempt_epoch") or 1) + 1
        data["retry_count"] = int(data.get("retry_count") or 0) + 1
        retry = snapshot.get("retry") or self.conf.app["retry"]
        delay = min(
            int(retry["backoff_max_ms"]),
            int(retry["backoff_initial_ms"]) * (2 ** max(data["retry_count"] - 1, 0)),
        )
        await self._move(job, token, data, delay)

    async def _delay(self, job, token: str, data: dict[str, Any], snapshot: dict[str, Any]) -> None:
        busy = snapshot.get("slot_busy") or self.conf.app["slot_busy"]
        probes = await self.leases.note_probe(
            data.get("queue") or "judge",
            self.node_id,
            data["run_id"],
            int(busy["probe_ttl_s"]),
        )
        base = int(busy["base_ms"])
        delay = base + random.randint(0, base)
        if probes > int(busy["max_probes_per_run"]):
            delay *= 3
        await self._move(job, token, data, delay)

    async def _move(self, job, token: str, data: dict[str, Any], delay: int) -> None:
        await job.updateData(data)
        try:
            await job.scripts.moveToDelayed(job.id, now_ms() + delay, delay, token)
        except Exception:
            await self._terminal(data, "delay_failed")
            await self.tick(data["run_id"])
            return
        raise WaitingChildrenError()

    async def _purge_unadmitted(self, run_id: str, admitted: set[str]) -> bool:
        for _ in range(3):
            dirty = False
            for name in JUDGE_QUEUES:
                for job in await self.dispatch.waiting_and_delayed(name):
                    payload = job.data or {}
                    if payload.get("run_id") != run_id:
                        continue
                    if payload.get("item_id") in admitted:
                        continue
                    dirty = True
                    try:
                        await job.remove()
                    except Exception:
                        return False
            if not dirty:
                return True
        return False

    async def _select_parent_cases(self, parent_run_id: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                "SELECT item_id, payload FROM input_manifest WHERE run_id = %s ORDER BY item_id",
                (parent_run_id,),
            )
            rows = await cur.fetchall()
        by_id = {row["item_id"]: row["payload"] for row in rows}
        if body.get("select_all_matching"):
            chosen = list(by_id)
            exclude = set(body.get("case_ids") or [])
            filt = body.get("filter") or {"status": "failed,poor"}
            items = await self.store.list_items(parent_run_id)
            allowed = _filter_items(items, filt)
            chosen = [item_id for item_id in chosen if item_id in allowed and item_id not in exclude]
        else:
            chosen = list(body.get("case_ids") or [])
        if not chosen:
            raise EvalError("replay selection is empty")
        missing = [item_id for item_id in chosen if item_id not in by_id]
        if missing:
            raise EvalError("selection contains cases outside the parent run")
        copied = []
        for item_id in chosen:
            payload = dict(by_id[item_id])
            payload["item_id"] = item_id
            if body.get("replay_agent_flow", True):
                payload.pop("output", None)
            copied.append(payload)
        return copied

    async def _provider_submit(self, data, phase, lease: LeasePair, request: dict[str, Any]):
        external_id = "ext_" + uuid.uuid4().hex
        try:
            await self._clear_ledger(data)
            payload = {
                **data,
                "external_job_id": external_id,
                "lease": _lease_dict(lease),
                "ledger": "pending_callback",
                "phase_id": phase["phase_id"],
            }
            running = RUNNING_QUEUE[data["queue"]]
            await self.dispatch.enqueue(running, external_id, "pending_callback", payload, keep=True)
            return ProviderAcceptedPendingRecorded(external_id)
        except Exception as exc:
            return ProviderAcceptedPendingFailed(str(exc))

    async def _write_intent(self, data, phase, lease: LeasePair, request: dict[str, Any]) -> None:
        key = f"{data['run_id']}:{data['item_id']}:{phase['phase_id']}:{data['attempt_epoch']}"
        async with await self.store.connect() as conn:
            await conn.execute(
                """
                INSERT INTO submission_intents (
                    idempotency_key, run_id, item_id, phase_id, attempt_epoch,
                    lease_node_id, lease_cluster_id, status, request
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,'open',%s)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (
                    key,
                    data["run_id"],
                    data["item_id"],
                    phase["phase_id"],
                    int(data["attempt_epoch"]),
                    lease.node_id,
                    lease.cluster_id,
                    Jsonb(request),
                ),
            )
            await conn.commit()

    async def _consume_intent(self, run_id: str, item_id: str, phase_id: str, epoch: int) -> None:
        key = f"{run_id}:{item_id}:{phase_id}:{epoch}"
        async with await self.store.connect() as conn:
            await conn.execute(
                "UPDATE submission_intents SET status = 'consumed' WHERE idempotency_key = %s",
                (key,),
            )
            await conn.commit()

    async def _abort_intent(self, data, phase) -> None:
        key = f"{data['run_id']}:{data['item_id']}:{phase['phase_id']}:{data['attempt_epoch']}"
        async with await self.store.connect() as conn:
            await conn.execute(
                "UPDATE submission_intents SET status = 'aborted' WHERE idempotency_key = %s",
                (key,),
            )
            await conn.commit()

    async def _mark_intent(self, data, phase, reason: str) -> None:
        phase_id = phase.get("phase_id") or data.get("phase_id") or "judge"
        key = f"{data['run_id']}:{data['item_id']}:{phase_id}:{data.get('attempt_epoch') or 1}"
        async with await self.store.connect() as conn:
            await conn.execute(
                "UPDATE submission_intents SET status = 'need_healer' WHERE idempotency_key = %s",
                (key,),
            )
            await conn.commit()
        log.warning("intent needs healer key=%s reason=%s", key, reason)

    async def _public_from_id(self, run_id: str) -> dict[str, Any]:
        return self._public_run(await self.store.get_run(run_id))

    async def case_book(self, run_id: str) -> dict[str, Any]:
        run = await self.store.get_run(run_id)
        if run is None:
            raise NotFound("run not found")
        snapshot = run.get("config_json") or {}
        domain = snapshot.get("eval_domain") or {}
        public = self._public_run(run)
        public["metrics"] = run.get("metrics_json") or {}
        public["eval_domain_id"] = domain.get("domain_id")
        public["business"] = snapshot.get("business") or "default"
        items = await self.store.list_items(run_id)
        manifests = await self.store.list_manifest(run_id)
        cursors = await self.store.list_cursors(run_id)
        report = None
        path = self.report_dir / f"{run_id}.json"
        if path.is_file():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = None
        return assemble_case_book(
            public,
            [{"item_id": row["item_id"], "payload": row.get("payload") or {}} for row in manifests],
            [
                {
                    "item_id": item.get("item_id"),
                    "criterion_id": item.get("criterion_id"),
                    "status": item.get("status"),
                    "score": item.get("score"),
                    "reason": item.get("reason"),
                    "error_message": item.get("error_message"),
                    "input": item.get("input"),
                    "output": item.get("output"),
                    "judge": item.get("judge_json"),
                }
                for item in items
            ],
            report if isinstance(report, dict) else None,
            [
                {
                    "item_id": row.get("item_id"),
                    "phase_id": row.get("phase_id"),
                    "output": row.get("output") or {},
                }
                for row in cursors
            ],
        )

    def _public_run(self, run: dict[str, Any] | None, replayed: bool = False) -> dict[str, Any]:
        if run is None:
            raise NotFound("run not found")
        snapshot = run["config_json"] or {}
        return {
            "run_id": run["run_id"],
            "benchmark_name": run["benchmark_name"],
            "status": run["status"],
            "profile_id": snapshot.get("profile_id"),
            "parent_run_id": run.get("parent_run_id"),
            "total_samples": run["total_samples"],
            "total_evaluators": run["total_evaluators"],
            "evaluated_items": run["evaluated_items"],
            "failed_items": run["failed_items"],
            "final_score": run["final_score"],
            "source": snapshot.get("source"),
            "replayed": replayed,
            "dispatch_domain_id": (snapshot.get("dispatch_domain") or {}).get("domain_id"),
        }

    async def open_experiment(self, body: dict[str, Any]) -> dict[str, Any]:
        from benchmark.kernel.contract import freeze_contract, freeze_policy

        self._reject_transport_overrides(body)
        try:
            contract = freeze_contract(dict(body.get("contract") or {}))
            policy = freeze_policy(dict(body.get("gate_policy") or {}))
        except ValueError as exc:
            raise EvalError(str(exc)) from exc
        experiment_id = str(body.get("experiment_id") or "").strip()
        stage = str(body.get("evaluation_stage") or "").strip()
        if not experiment_id or stage not in {"development", "confirmation", "fast"}:
            raise EvalError("experiment_id and evaluation_stage are required")
        async with await self.store.connect() as conn:
            await conn.execute(
                """
                INSERT INTO eval_experiments (
                    experiment_id, evaluation_stage, status, contract_json, gate_policy_json,
                    source_run_id, candidate_hash, conflict_fingerprint
                ) VALUES (%s,%s,'running',%s,%s,%s,%s,%s)
                """,
                (
                    experiment_id,
                    stage,
                    Jsonb(contract),
                    Jsonb(policy),
                    body.get("source_run_id"),
                    body.get("candidate_hash"),
                    Jsonb(list(body.get("conflict_fingerprint") or [])),
                ),
            )
            await conn.commit()
        return {"experiment_id": experiment_id, "status": "running", "evaluation_stage": stage}

    async def seal_experiment(self, experiment_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        from benchmark.kernel.contract import content_hash
        from benchmark.kernel.gate import seal_development, sign_receipt

        body = body or {}
        async with await self.store.connect() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT * FROM eval_experiments WHERE experiment_id = %s FOR UPDATE",
                    (experiment_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise NotFound("experiment not found")
                if row["status"] == "evaluated":
                    return self._public_experiment(row)
                policy = row["gate_policy_json"]
                contract = row["contract_json"]
                checks = list(body.get("checks") or [])
                if not checks and row["source_run_id"]:
                    checks = await self._checks_from_run(conn, row["source_run_id"])
                judges = list(body.get("judges") or policy.get("judges") or [])
                calibration = body.get("calibration") if "calibration" in body else policy.get("judge_calibration_record")
                if isinstance(calibration, dict) and not calibration.get("record_hash"):
                    calibration = {**calibration, "record_hash": content_hash(calibration)}
                verdict = seal_development(policy, checks, calibration=calibration, judges=judges)
                run_terminal = body.get("run_terminal") or "completed"
                bundle = {
                    "experiment_id": experiment_id,
                    "evaluation_contract_hash": contract.get("evaluation_contract_hash"),
                    "checks": checks,
                    "calibration": calibration,
                    "validity": "valid" if verdict["stage_outcome"] == "passed" else "invalid",
                }
                bundle["bundle_hash"] = content_hash(bundle)
                receipt = None
                if verdict["stage_outcome"] == "passed":
                    trust = policy.get("gate_receipt_trust") or {}
                    receipt = sign_receipt(
                        {
                            "candidate_hash": row["candidate_hash"],
                            "development_experiment_id": experiment_id,
                            "bundle_hash": bundle["bundle_hash"],
                            "evaluation_contract_hash": contract.get("evaluation_contract_hash"),
                            "gate_policy_hash": policy.get("gate_policy_hash"),
                            "verdict": "hard_pass",
                            "host_build_hash": trust.get("host_build_hash"),
                            "issued_epoch": row["pending_epoch"],
                            **(
                                {"judge_calibration_record_hash": calibration.get("record_hash")}
                                if calibration
                                else {}
                            ),
                        },
                        self._gate_key(),
                    )
                    bundle["development_gate_receipt"] = receipt
                await conn.execute(
                    """
                    UPDATE eval_experiments
                    SET status = 'evaluated',
                        bundle_json = %s,
                        stage_outcome = %s,
                        release_authorization = %s,
                        run_terminal = %s
                    WHERE experiment_id = %s
                    """,
                    (
                        Jsonb(bundle),
                        verdict["stage_outcome"],
                        verdict["release_authorization"],
                        run_terminal,
                        experiment_id,
                    ),
                )
                await conn.execute(
                    """
                    INSERT INTO outbox (run_id, event_kind, dedupe_key, status, payload)
                    VALUES (%s, 'experiment_evaluated', %s, 'pending', %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        experiment_id,
                        experiment_id,
                        Jsonb(
                            {
                                "experiment_id": experiment_id,
                                "evaluation_stage": row["evaluation_stage"],
                                "stage_outcome": verdict["stage_outcome"],
                                "release_authorization": verdict["release_authorization"],
                                "run_terminal": run_terminal,
                                "bundle_hash": bundle["bundle_hash"],
                                "evaluation_contract_hash": contract.get("evaluation_contract_hash"),
                                "source_run_id": row["source_run_id"],
                            }
                        ),
                    ),
                )
        fresh = await self._load_experiment(experiment_id)
        return self._public_experiment(fresh)

    async def authorize_release(self, experiment_id: str, body: dict[str, Any]) -> dict[str, Any]:
        from benchmark.kernel.gate import authorize_decision

        async with await self.store.connect() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT * FROM eval_experiments WHERE experiment_id = %s FOR UPDATE",
                    (experiment_id,),
                )
                row = await cur.fetchone()
                if row is None:
                    raise NotFound("experiment not found")
                view = self._experiment_view(row)
                decision = authorize_decision(view, body, signer_key=self._gate_key())
                if not decision["accepted"]:
                    raise EvalError(decision["reason"], code="release_rejected")
                fingerprint = set(row["conflict_fingerprint"] or [])
                if fingerprint:
                    others = await conn.execute(
                        """
                        SELECT experiment_id, conflict_fingerprint
                        FROM eval_experiments
                        WHERE experiment_id <> %s
                          AND release_authorization IN ('pending_confirmation', 'approved')
                        FOR UPDATE
                        """,
                        (experiment_id,),
                    )
                    for other in await others.fetchall():
                        if fingerprint & set(other["conflict_fingerprint"] or []):
                            await conn.execute(
                                """
                                UPDATE eval_experiments
                                SET release_authorization = 'superseded'
                                WHERE experiment_id = %s
                                """,
                                (other["experiment_id"],),
                            )
                await conn.execute(
                    """
                    UPDATE eval_experiments
                    SET release_authorization = 'approved',
                        pending_epoch = pending_epoch + 1
                    WHERE experiment_id = %s
                      AND release_authorization = 'pending_confirmation'
                      AND pending_epoch = %s
                    """,
                    (experiment_id, int(body["expected_pending_epoch"])),
                )
                await conn.execute(
                    """
                    INSERT INTO eval_release_events (event_id, experiment_id, payload)
                    VALUES (%s, %s, %s)
                    """,
                    (uuid.uuid4().hex, experiment_id, Jsonb({"basis": body.get("basis"), "stage_edge": body.get("stage_edge")})),
                )
        fresh = await self._load_experiment(experiment_id)
        return self._public_experiment(fresh)

    async def _checks_from_run(self, conn, run_id: str) -> list[dict[str, Any]]:
        cur = await conn.execute(
            "SELECT item_id, criterion_id, status, score, judge_json FROM benchmark_run_items WHERE run_id = %s",
            (run_id,),
        )
        checks = []
        for item in await cur.fetchall():
            embedded = (item["judge_json"] or {}).get("checks") or []
            if embedded:
                checks.extend(embedded)
                continue
            passed = item["status"] == "success" and item["score"] == 1
            checks.append(
                {
                    "check_id": item["criterion_id"],
                    "applicability": "required",
                    "verdict": "pass" if passed else "fail",
                    "value": item["score"],
                }
            )
        return checks

    async def _load_experiment(self, experiment_id: str) -> dict[str, Any]:
        async with await self.store.connect() as conn:
            cur = await conn.execute(
                "SELECT * FROM eval_experiments WHERE experiment_id = %s",
                (experiment_id,),
            )
            row = await cur.fetchone()
        if row is None:
            raise NotFound("experiment not found")
        return row

    def _experiment_view(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "experiment_id": row["experiment_id"],
            "release_authorization": row["release_authorization"],
            "pending_epoch": row["pending_epoch"],
            "candidate_hash": row["candidate_hash"],
            "contract": row["contract_json"],
            "gate_policy": row["gate_policy_json"],
            "bundle": row["bundle_json"] or {},
        }

    def _public_experiment(self, row: dict[str, Any]) -> dict[str, Any]:
        bundle = row["bundle_json"] or {}
        return {
            "experiment_id": row["experiment_id"],
            "status": row["status"],
            "evaluation_stage": row["evaluation_stage"],
            "stage_outcome": row["stage_outcome"],
            "release_authorization": row["release_authorization"],
            "run_terminal": row["run_terminal"],
            "bundle_hash": bundle.get("bundle_hash"),
            "source_run_id": row["source_run_id"],
        }

    def _gate_key(self) -> str:
        from benchmark.kernel.confloader import secret

        return secret(self.conf.app["plugin_signature"]["secret_ref"])

    def _cred_key(self, run_id: str) -> str:
        return f"{self.queue_prefix}:cred:{run_id}"

    def _callback_key(self, external_job_id: str) -> str:
        return f"{self.queue_prefix}:callback-done:{external_job_id}"

    def _reject_transport_overrides(self, body: dict[str, Any]) -> None:
        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    lowered = str(key).lower()
                    if lowered in _BANNED_REQUEST_KEYS or lowered.endswith(("_url", "_base_url", "url_key")):
                        raise EvalError("request must not carry a url or domain", code="transport_override")
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(body)


def _normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    item_id = str(case.get("item_id") or uuid.uuid4().hex[:8])
    normalized = {
        "item_id": item_id,
        "input": case.get("input"),
        "output": case.get("output"),
    }
    if "expected" in case:
        normalized["expected"] = case.get("expected")
    return normalized


def _phase(plan: list[dict[str, Any]], phase_id: str | None) -> dict[str, Any]:
    if not plan:
        raise PermanentError("missing_plan")
    if phase_id:
        for phase in plan:
            if phase["phase_id"] == phase_id:
                return phase
    return plan[0]


def _next_phase(plan: list[dict[str, Any]], phase_id: str) -> dict[str, Any] | None:
    ids = [phase["phase_id"] for phase in plan]
    if phase_id not in ids:
        return None
    index = ids.index(phase_id) + 1
    if index >= len(plan):
        return None
    return plan[index]


def _expand_keys(item_ids: list[str], criteria: list[dict[str, Any]]) -> list[dict[str, str]]:
    keys = []
    for item_id in item_ids:
        for criterion in criteria:
            if criterion.get("counts_toward_progress"):
                keys.append({"item_id": item_id, "criterion_id": criterion["id"]})
    return keys


def _lease_dict(lease: LeasePair) -> dict[str, Any]:
    return {
        "node_key": lease.node_key,
        "node_id": lease.node_id,
        "cluster_key": lease.cluster_key,
        "cluster_id": lease.cluster_id,
        "kind": lease.kind,
    }


def _detach_graph(outcome: Any) -> dict[str, Any] | None:
    if isinstance(outcome, Completed) and isinstance(outcome.result, dict):
        graph = outcome.result.pop("_graph", None)
        return graph if isinstance(graph, dict) else None
    return None


def _lease_from(payload: dict[str, Any] | None) -> LeasePair | None:
    if not payload:
        return None
    return LeasePair(
        node_key=payload["node_key"],
        node_id=payload["node_id"],
        cluster_key=payload.get("cluster_key"),
        cluster_id=payload.get("cluster_id"),
        kind=payload.get("kind") or "sync",
    )


def _distinct_items(items: list[dict[str, Any]]) -> int:
    return len({row["item_id"] for row in items if row["status"] in {"success", "failed"}})


def _filter_items(items: list[dict[str, Any]], filt: dict[str, Any]) -> set[str]:
    raw = str(filt.get("status") or "failed,poor")
    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    query = str(filt.get("query") or "").lower()
    allowed = set()
    for row in items:
        status = str(row["status"]).lower()
        poor = row["status"] == "success" and row["score"] is not None and float(row["score"]) < 0.5
        match = ("failed" in wanted and status == "failed") or ("poor" in wanted and poor) or (status in wanted)
        if query and query not in json.dumps(row, default=str).lower():
            match = False
        if match:
            allowed.add(row["item_id"])
    return allowed


def _bearer(header: str) -> str:
    if not header:
        return ""
    parts = header.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return cleaned[:40] or "agent"


async def _call(plugin: Any, method: str, *args):
    async_name = "a" + method
    target = getattr(plugin, async_name, None)
    if target is not None:
        return await target(*args)
    target = getattr(plugin, method)
    result = target(*args)
    if hasattr(result, "__await__"):
        return await result
    return result


def open_kernel(conf_dir: str | Path, *, node_id: str = "node-1", worker_profile: str | None = None) -> Kernel:
    conf = load_conf(conf_dir)
    if worker_profile:
        conf.app["worker_profile"] = worker_profile
    return Kernel(conf, node_id=node_id)
