"""Multi-process worker entrypoints (Redis Streams MQ + SQL outbox).

Roles share Redis Streams; Outbox/stores use ``LLMJUDGE_PERSISTENCE=sql``
so Dispatch and Ingest can run in separate processes.
Gateway mounts Dashboard / Annotation / Prompt / Benchmark BFF.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Literal

from llmobserverjudge.dispatch.config_loader import DispatchConfigLoader, load_dispatch_config
from llmobserverjudge.dispatch.publisher import DispatchPublisher
from llmobserverjudge.eval_worker.worker import EvalWorker
from llmobserverjudge.ingest.worker import IngestWorker
from llmobserverjudge.annotation_worker.worker import AnnotationWorker
from llmobserverjudge.attribution_worker.worker import AttributionWorker
from llmobserverjudge.runtime.stack import _resolve_dispatch_config_path, parse_api_keys
from llmobserverjudge.shared.mq import MessageQueue, create_mq_from_env
from llmobserverjudge.shared.persistence import create_persistence
from llmobserverjudge.shared.redis_event_bus import create_event_bus_from_env

logger = logging.getLogger(__name__)

Role = Literal["ingest", "dispatch", "eval", "attribution", "annotation", "gateway"]


def _force_sample() -> bool:
    return os.environ.get("LLMJUDGE_FORCE_SAMPLE", "1").strip() not in ("0", "false", "no")


def _poll_interval() -> float:
    return float(os.environ.get("LLMJUDGE_POLL_INTERVAL", "0.05"))


async def _loop(name: str, tick, stop: asyncio.Event, interval: float) -> None:  # noqa: ANN001
    while not stop.is_set():
        try:
            n = await tick()
        except Exception:
            logger.exception("worker_tick_failed role=%s", name)
            n = 0
        if not n:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except TimeoutError:
                pass


def _dispatch_loader() -> DispatchConfigLoader:
    path = _resolve_dispatch_config_path()
    if path is None:
        logger.warning("dispatch config missing; using defaults")
        return DispatchConfigLoader()
    return load_dispatch_config(path)


async def run_ingest(*, mq: MessageQueue | None = None) -> None:
    mq = mq or create_mq_from_env(consumer_group="ingest")
    bundle = await create_persistence()
    event_bus = create_event_bus_from_env()
    start = getattr(event_bus, "start", None)
    if start is not None:
        await start()
    worker = IngestWorker(
        mq,
        outbox=bundle.outbox,
        store=bundle.spans,
        idempotency=bundle.idempotency,
        force_sample=_force_sample(),
        sampling_stats=bundle.sampling,
        event_bus=event_bus,
    )
    stop = asyncio.Event()
    logger.info("worker_started role=ingest persistence=%s", bundle.driver)
    try:
        await _loop("ingest", worker.process_once, stop, _poll_interval())
    finally:
        stop_fn = getattr(event_bus, "stop", None)
        if stop_fn is not None:
            await stop_fn()


async def run_dispatch(*, mq: MessageQueue | None = None) -> None:
    mq = mq or create_mq_from_env(consumer_group="dispatch")
    bundle = await create_persistence()
    pub = DispatchPublisher(mq, bundle.outbox, config_loader=_dispatch_loader())
    loader = pub.config_loader
    stop = asyncio.Event()

    async def tick() -> int:
        return await pub.publish_pending()

    logger.info("worker_started role=dispatch persistence=%s", bundle.driver)
    await asyncio.gather(
        _loop("dispatch", tick, stop, _poll_interval()),
        loader.watch_loop(interval_s=2.0),
    )


async def run_eval(*, mq: MessageQueue | None = None) -> None:
    from llmobserverjudge.dispatch.client import DispatchClient
    from llmobserverjudge.eval_worker.judges import build_judge_router

    mq = mq or create_mq_from_env(consumer_group="eval")
    bundle = await create_persistence()
    event_bus = create_event_bus_from_env()
    start = getattr(event_bus, "start", None)
    if start is not None:
        await start()
    loader = _dispatch_loader()
    worker = EvalWorker(
        mq,
        dispatch_client=DispatchClient(mq, config_loader=loader),
        judge_router=build_judge_router(include_env_http=True),
        eval_results=bundle.eval_results,
        event_bus=event_bus,
        prompt_service=bundle.prompts,
        idempotency=bundle.idempotency,
    )
    stop = asyncio.Event()

    async def tick() -> int:
        return len(await worker.process_once())

    logger.info("worker_started role=eval")
    try:
        await _loop("eval", tick, stop, _poll_interval())
    finally:
        stop_fn = getattr(event_bus, "stop", None)
        if stop_fn is not None:
            await stop_fn()


async def run_attribution(*, mq: MessageQueue | None = None) -> None:
    mq = mq or create_mq_from_env(consumer_group="attribution")
    bundle = await create_persistence()
    worker = AttributionWorker(
        mq,
        store=bundle.attributions,
        checkpoint=bundle.checkpoints,
        use_env_react_policy=True,
    )
    stop = asyncio.Event()

    async def tick() -> int:
        return len(await worker.process_once())

    logger.info("worker_started role=attribution")
    await _loop("attribution", tick, stop, _poll_interval())


async def run_annotation(*, mq: MessageQueue | None = None) -> None:
    mq = mq or create_mq_from_env(consumer_group="annotation")
    bundle = await create_persistence()
    worker = AnnotationWorker(mq, store=bundle.annotations)
    stop = asyncio.Event()

    async def tick() -> int:
        return len(await worker.process_once())

    logger.info("worker_started role=annotation")
    await _loop("annotation", tick, stop, _poll_interval())


async def run_gateway() -> None:
    """Gateway HTTP + Dashboard/Annotation/Prompt/Benchmark BFF (split deploy)."""
    import uvicorn

    from llmobserverjudge.annotation_worker.api import create_annotation_app
    from llmobserverjudge.benchmark.api import create_benchmark_app
    from llmobserverjudge.benchmark.redis_signal import create_batch_signal_from_env
    from llmobserverjudge.benchmark.service import BenchmarkService
    from llmobserverjudge.dashboard.api import create_dashboard_app
    from llmobserverjudge.dashboard.store import DashboardContext
    from llmobserverjudge.eval_worker.prompt import seed_default_prompts
    from llmobserverjudge.gateway.app import create_app
    from llmobserverjudge.gateway.auth import ApiKeyAuthenticator
    from llmobserverjudge.prompt_admin.api import create_prompt_admin_app
    from llmobserverjudge.shared.project_api import create_project_app
    from llmobserverjudge.shared.redis_rate_limit import create_rate_limiter_from_env

    mq = create_mq_from_env(consumer_group="gateway")
    bundle = await create_persistence()
    await seed_default_prompts(bundle.prompts)
    event_bus = create_event_bus_from_env()
    start = getattr(event_bus, "start", None)
    if start is not None:
        await start()
    rate_limiter = create_rate_limiter_from_env()
    signal = await create_batch_signal_from_env()
    benchmark = BenchmarkService(store=bundle.benchmark_reports, signal=signal)
    authenticator = ApiKeyAuthenticator(parse_api_keys())

    async def on_shutdown(_app: Any) -> None:
        stop_fn = getattr(event_bus, "stop", None)
        if stop_fn is not None:
            await stop_fn()

    app = create_app(
        mq=mq,
        authenticator=authenticator,
        rate_limiter=rate_limiter,
        project_store=bundle.projects,
        on_shutdown=on_shutdown,
    )

    async def stack_info() -> dict[str, Any]:
        return {
            "started": True,
            "persistence": bundle.driver,
            "role": "gateway",
            "mq_driver": os.environ.get("LLMJUDGE_MQ") or "memory",
        }

    dash_ctx = DashboardContext(
        span_store=bundle.spans,
        eval_results=bundle.eval_results,
        sampling=bundle.sampling,
        annotations=bundle.annotations,
        attributions=bundle.attributions,
        benchmark_store=bundle.benchmark_reports,
        event_bus=event_bus,
        mq=mq,
        telemetry=app.state.telemetry,
        stack_info=stack_info,
        prompt_service=bundle.prompts,
    )
    app.mount("/dashboard", create_dashboard_app(dash_ctx))
    app.mount("/annotation", create_annotation_app(bundle.annotations))
    app.mount("/benchmark", create_benchmark_app(benchmark))
    app.mount("/prompt", create_prompt_admin_app(bundle.prompts))
    app.mount("/projects", create_project_app(bundle.projects, authenticator=authenticator))

    @app.get("/stackz")
    async def stackz() -> dict[str, Any]:
        from llmobserverjudge.contracts.topics import Topic

        return {
            "status": "ok",
            "role": "gateway",
            "persistence": bundle.driver,
            "mq_driver": os.environ.get("LLMJUDGE_MQ") or "memory",
            "mq_depths": {
                "spans": await mq.depth(Topic.SPANS),
                "eval_jobs": await mq.depth(Topic.EVAL_JOBS),
                "attribution_jobs": await mq.depth(Topic.ATTRIBUTION_JOBS),
                "annotation_jobs": await mq.depth(Topic.ANNOTATION_JOBS),
            },
        }

    host = os.environ.get("LLMJUDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("LLMJUDGE_PORT", "8080"))
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    logger.info(
        "worker_started role=gateway mq=%s persistence=%s",
        os.environ.get("LLMJUDGE_MQ") or "memory",
        bundle.driver,
    )
    await server.serve()


_RUNNERS = {
    "ingest": run_ingest,
    "dispatch": run_dispatch,
    "eval": run_eval,
    "attribution": run_attribution,
    "annotation": run_annotation,
    "gateway": run_gateway,
}


def main(argv: list[str] | None = None) -> None:
    from llmobserverjudge.shared.env import load_env

    load_env()
    parser = argparse.ArgumentParser(description="LLMJudge multi-process worker")
    parser.add_argument(
        "role",
        choices=sorted(_RUNNERS),
        help="Worker role (compose split services)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LLMJUDGE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not os.environ.get("LLMJUDGE_DISPATCH_CONFIG"):
        p = Path("configs/dispatch.yaml")
        if p.is_file():
            os.environ["LLMJUDGE_DISPATCH_CONFIG"] = str(p.resolve())
    asyncio.run(_RUNNERS[args.role]())
