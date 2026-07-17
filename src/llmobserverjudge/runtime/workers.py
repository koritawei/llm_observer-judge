"""Multi-process worker entrypoints (Redis Streams MQ + SQL outbox).

Roles share Redis Streams; Outbox/stores use ``LLMJUDGE_PERSISTENCE=sql``
so Dispatch and Ingest can run in separate processes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

from llmobserverjudge.dispatch.config_loader import DispatchConfigLoader, load_dispatch_config
from llmobserverjudge.dispatch.publisher import DispatchPublisher
from llmobserverjudge.eval_worker.worker import EvalWorker
from llmobserverjudge.ingest.worker import IngestWorker
from llmobserverjudge.annotation_worker.worker import AnnotationWorker
from llmobserverjudge.attribution_worker.worker import AttributionWorker
from llmobserverjudge.runtime.stack import _resolve_dispatch_config_path, parse_api_keys
from llmobserverjudge.shared.mq import MessageQueue, create_mq_from_env
from llmobserverjudge.shared.persistence import create_persistence

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
    worker = IngestWorker(mq, outbox=bundle.outbox, force_sample=_force_sample())
    stop = asyncio.Event()
    logger.info("worker_started role=ingest persistence=%s", bundle.driver)
    await _loop("ingest", worker.process_once, stop, _poll_interval())


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
    from llmobserverjudge.eval_worker.judges import build_judge_router

    mq = mq or create_mq_from_env(consumer_group="eval")
    worker = EvalWorker(mq, judge_router=build_judge_router(include_env_http=True))
    stop = asyncio.Event()

    async def tick() -> int:
        return len(await worker.process_once())

    logger.info("worker_started role=eval")
    await _loop("eval", tick, stop, _poll_interval())


async def run_attribution(*, mq: MessageQueue | None = None) -> None:
    mq = mq or create_mq_from_env(consumer_group="attribution")
    bundle = await create_persistence()
    worker = AttributionWorker(
        mq, store=bundle.attributions, use_env_react_policy=True
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
    """Gateway HTTP only — publishes spans to MQ (no in-process workers)."""
    import uvicorn

    from llmobserverjudge.gateway.app import create_app
    from llmobserverjudge.gateway.auth import ApiKeyAuthenticator

    mq = create_mq_from_env(consumer_group="gateway")
    app = create_app(mq=mq, authenticator=ApiKeyAuthenticator(parse_api_keys()))
    host = os.environ.get("LLMJUDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("LLMJUDGE_PORT", "8080"))
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    logger.info("worker_started role=gateway mq=%s", os.environ.get("LLMJUDGE_MQ") or "memory")
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
    # Ensure dispatch config path visible when cwd is /app
    if not os.environ.get("LLMJUDGE_DISPATCH_CONFIG"):
        p = Path("configs/dispatch.yaml")
        if p.is_file():
            os.environ["LLMJUDGE_DISPATCH_CONFIG"] = str(p.resolve())
    asyncio.run(_RUNNERS[args.role]())
