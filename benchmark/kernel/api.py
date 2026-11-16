"""HTTP API. The browser only talks to this process."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from benchmark.evolution.service import EvolutionService
from benchmark.kernel.errors import EvalError
from benchmark.kernel.runtime import Kernel, open_kernel
from benchmark.kernel.workflow import plugin_catalog, profile_ids_for

FRONT = Path(__file__).resolve().parents[2] / "front"
CONF = Path(os.environ.get("EVALBENCH_CONF", "conf"))


def create_app(conf_dir: Path | None = None) -> FastAPI:
    conf_dir = conf_dir or CONF

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "kernel", None) is None:
            kernel = open_kernel(conf_dir)
            await kernel.setup()
            kernel.start_workers()
            app.state.kernel = kernel
            app.state.owns_kernel = True
        yield
        if getattr(app.state, "owns_kernel", False):
            await app.state.kernel.close()

    app = FastAPI(title="Eval Benchmark", lifespan=lifespan)

    @app.exception_handler(EvalError)
    async def on_eval_error(_: Request, exc: EvalError):
        return JSONResponse({"error": exc.code, "message": exc.message}, status_code=exc.status)

    @app.get("/")
    async def index():
        return FileResponse(FRONT / "index.html")

    @app.get("/front/{name}")
    async def asset(name: str):
        path = (FRONT / name).resolve()
        if not str(path).startswith(str(FRONT.resolve())) or not path.is_file():
            return JSONResponse({"error": "not_found"}, status_code=404)
        return FileResponse(path, headers={"cache-control": "no-cache"})

    @app.get("/health")
    async def health(request: Request):
        kernel: Kernel = request.app.state.kernel
        await kernel.leases.redis.ping()
        async with await kernel.store.connect() as conn:
            await conn.execute("SELECT 1")
        return {"status": "ok"}

    @app.get("/v1/eval-domains")
    async def eval_domains(request: Request):
        domains = request.app.state.kernel.conf.eval_domains.get("eval_domains") or {}
        return {"eval_domains": domains}

    @app.get("/v1/profiles")
    async def profiles(request: Request, business: str = "default"):
        return {"profiles": profile_ids_for(request.app.state.kernel.conf, business)}

    @app.get("/v1/plugins")
    async def plugins(request: Request, business: str = "default"):
        return plugin_catalog(request.app.state.kernel.conf, business)

    @app.post("/v1/workflows")
    async def create_workflow(request: Request):
        body = await request.json()
        payload = {
            **body,
            "workflow": {
                "base_profile": body.get("base_profile") or "benchmark_sync",
                "plugins": body.get("plugins") or {},
            },
        }
        return await request.app.state.kernel.publish_run(payload)

    @app.post("/v1/runs")
    async def create_run(request: Request):
        body = await request.json()
        return await request.app.state.kernel.publish_run(body)

    @app.get("/v1/runs")
    async def list_runs(request: Request):
        rows = await request.app.state.kernel.store.list_runs()
        return {"runs": [request.app.state.kernel._public_run(row) for row in rows]}

    @app.get("/v1/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        return await request.app.state.kernel._public_from_id(run_id)

    @app.get("/v1/runs/{run_id}/items")
    async def get_items(run_id: str, request: Request):
        items = await request.app.state.kernel.store.list_items(run_id)
        return {"items": items}

    @app.get("/v1/runs/{run_id}/book")
    async def case_book(run_id: str, request: Request):
        return await request.app.state.kernel.case_book(run_id)

    @app.post("/v1/runs/{run_id}/stop")
    async def stop_run(run_id: str, request: Request):
        return await request.app.state.kernel.stop(run_id)

    @app.post("/v1/datasets")
    async def create_dataset(request: Request):
        return await request.app.state.kernel.publish_dataset(await request.json())

    @app.post("/v1/benchmarks/runs/{parent_run_id}/playground-replay")
    async def replay(parent_run_id: str, request: Request):
        body = await request.json()
        result = await request.app.state.kernel.playground_replay(
            parent_run_id,
            body,
            request.headers.get("authorization", ""),
        )
        return JSONResponse(result, status_code=201)

    @app.post("/v1/experiments")
    async def open_experiment(request: Request):
        return await request.app.state.kernel.open_experiment(await request.json())

    @app.post("/v1/experiments/{experiment_id}/seal")
    async def seal_experiment(experiment_id: str, request: Request):
        raw = await request.body()
        payload = {} if not raw else json.loads(raw)
        return await request.app.state.kernel.seal_experiment(experiment_id, payload)

    @app.post("/v1/experiments/{experiment_id}/authorize-release")
    async def authorize(experiment_id: str, request: Request):
        return await request.app.state.kernel.authorize_release(experiment_id, await request.json())

    @app.post("/v1/callbacks/{external_job_id}")
    async def callback(external_job_id: str, request: Request):
        body = await request.body()
        return await request.app.state.kernel.callback(external_job_id, body, dict(request.headers))

    @app.post("/v1/messages")
    async def message(request: Request):
        return await request.app.state.kernel.ingest_message(await request.json())

    @app.post("/v1/registrations")
    async def registration(request: Request):
        return await request.app.state.kernel.save_registration(await request.json())

    @app.post("/v1/evolution/gate-policy")
    async def evolution_gate_policy(request: Request):
        from benchmark.kernel.contract import freeze_policy

        try:
            return freeze_policy(await _json_body(request))
        except ValueError as exc:
            raise EvalError(str(exc)) from exc

    @app.post("/v1/evolution/events")
    async def evolution_event(request: Request):
        return await _evolution(request).admit(await request.json())

    @app.post("/v1/evolution/campaigns/{campaign_id}/freeze")
    async def evolution_freeze(campaign_id: str, request: Request):
        return await _evolution(request).freeze(campaign_id, await _json_body(request))

    @app.post("/v1/evolution/campaigns/{campaign_id}/diagnose")
    async def evolution_diagnose(campaign_id: str, request: Request):
        return await _evolution(request).diagnose_campaign(campaign_id)

    @app.post("/v1/evolution/campaigns/{campaign_id}/plan")
    async def evolution_plan(campaign_id: str, request: Request):
        return await _evolution(request).plan(campaign_id, await _json_body(request))

    @app.post("/v1/evolution/campaigns/{campaign_id}/seal-candidate")
    async def evolution_seal(campaign_id: str, request: Request):
        return await _evolution(request).seal_candidate(campaign_id, await _json_body(request))

    @app.post("/v1/evolution/campaigns/{campaign_id}/trial")
    async def evolution_trial(campaign_id: str, request: Request):
        return await _evolution(request).start_trial(campaign_id)

    @app.post("/v1/evolution/campaigns/{campaign_id}/decide")
    async def evolution_decide(campaign_id: str, request: Request):
        return await _evolution(request).decide(campaign_id, await _json_body(request))

    @app.post("/v1/evolution/campaigns/{campaign_id}/confirm")
    async def evolution_confirm(campaign_id: str, request: Request):
        return await _evolution(request).confirm(campaign_id)

    @app.post("/v1/evolution/campaigns/{campaign_id}/release")
    async def evolution_release(campaign_id: str, request: Request):
        return await _evolution(request).request_release(campaign_id, await _json_body(request))

    @app.post("/v1/evolution/campaigns/{campaign_id}/stop")
    async def evolution_stop(campaign_id: str, request: Request):
        return await _evolution(request).stop(campaign_id, await _json_body(request))

    @app.post("/v1/evolution/campaigns/{campaign_id}/advance")
    async def evolution_advance(campaign_id: str, request: Request):
        return await _evolution(request).advance(campaign_id, await _json_body(request))

    @app.post("/v1/evolution/campaigns/{campaign_id}/repairer-proposal")
    async def evolution_repairer(campaign_id: str, request: Request):
        return await _evolution(request).propose_repairer(campaign_id, await _json_body(request))

    @app.post("/v1/evolution/campaigns/{campaign_id}/archive")
    async def evolution_archive(campaign_id: str, request: Request):
        return await _evolution(request).archive(campaign_id)

    @app.post("/v1/evolution/campaigns/{campaign_id}/rollback")
    async def evolution_rollback(campaign_id: str, request: Request):
        return await _evolution(request).rollback(campaign_id)

    @app.get("/v1/evolution/campaigns/{campaign_id}/contract")
    async def evolution_contract(campaign_id: str, request: Request):
        return await _evolution(request).public_contract(campaign_id)

    return app


def _evolution(request: Request) -> EvolutionService:
    service = getattr(request.app.state, "evolution", None)
    kernel = request.app.state.kernel
    if service is None or service.kernel is not kernel:
        service = EvolutionService(kernel)
        request.app.state.evolution = service
    return service


async def _json_body(request: Request) -> dict:
    raw = await request.body()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise EvalError("json object is required")
    return payload


app = create_app()
