"""In-process Compose stack: Gateway + workers on a shared InMemoryMQ.

Multi-container RocketMQ is out of scope; this process hosts the full
ingest → outbox → dispatch → eval (+ attribution/annotation) path for local
compose / E2E. Postgres + Redis attach via env when available.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from llmobserverjudge.annotation_worker.api import create_annotation_app
from llmobserverjudge.annotation_worker.worker import AnnotationWorker
from llmobserverjudge.attribution_worker.worker import AttributionWorker
from llmobserverjudge.benchmark.api import create_benchmark_app
from llmobserverjudge.benchmark.redis_signal import create_batch_signal_from_env
from llmobserverjudge.benchmark.service import BenchmarkService
from llmobserverjudge.contracts.topics import Topic
from llmobserverjudge.dashboard.api import create_dashboard_app
from llmobserverjudge.dashboard.events import PipelineEventBus
from llmobserverjudge.dashboard.store import (
    DashboardContext,
    EvalResultStore,
    SamplingStatsStore,
)
from llmobserverjudge.dispatch.client import DispatchClient
from llmobserverjudge.dispatch.config_loader import DispatchConfigLoader, load_dispatch_config
from llmobserverjudge.dispatch.publisher import DispatchPublisher
from llmobserverjudge.eval_worker.judges import build_judge_router
from llmobserverjudge.eval_worker.prompt import seed_default_prompts
from llmobserverjudge.eval_worker.worker import EvalWorker
from llmobserverjudge.gateway.app import create_app
from llmobserverjudge.gateway.auth import ApiKeyAuthenticator
from llmobserverjudge.ingest.worker import IngestWorker
from llmobserverjudge.prompt_admin.api import create_prompt_admin_app
from llmobserverjudge.shared.mq import InMemoryMQ, MessageQueue, create_mq_from_env
from llmobserverjudge.shared.persistence import PersistenceBundle, create_persistence
from llmobserverjudge.shared.storage import InMemorySpanStore

logger = logging.getLogger(__name__)


def parse_api_keys(raw: str | None = None) -> dict[str, str]:
    """Parse ``key:tenant,key2:tenant2`` (default ``dev-key:tenant-dev``)."""
    text = (raw if raw is not None else os.environ.get("LLMJUDGE_API_KEYS") or "").strip()
    if not text:
        return {"dev-key": "tenant-dev"}
    out: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid API key entry (want key:tenant): {part}")
        key, tenant = part.split(":", 1)
        key, tenant = key.strip(), tenant.strip()
        if not key or not tenant:
            raise ValueError(f"invalid API key entry: {part}")
        out[key] = tenant
    return out or {"dev-key": "tenant-dev"}


def _resolve_dispatch_config_path() -> Path | None:
    env = (os.environ.get("LLMJUDGE_DISPATCH_CONFIG") or "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path("configs/dispatch.yaml"),
            Path(__file__).resolve().parents[3] / "configs" / "dispatch.yaml",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


class DevStack:
    """Shared-MQ local stack used by ``llmj-stack`` and compose."""

    def __init__(
        self,
        *,
        mq: MessageQueue | None = None,
        persistence: PersistenceBundle | None = None,
        force_sample: bool | None = None,
        poll_interval: float = 0.05,
        api_keys: dict[str, str] | None = None,
        dispatch_config_path: Path | str | None = None,
    ) -> None:
        # Prefer explicit mq; else env (redis streams) when LLMJUDGE_MQ set; else memory.
        if mq is not None:
            self.mq = mq
        elif (os.environ.get("LLMJUDGE_MQ") or "").strip():
            self.mq = create_mq_from_env(consumer_group="stack")
        else:
            self.mq = InMemoryMQ()
        self.persistence = persistence
        self.force_sample = (
            force_sample
            if force_sample is not None
            else (os.environ.get("LLMJUDGE_FORCE_SAMPLE", "1").strip() not in ("0", "false", "no"))
        )
        self.poll_interval = poll_interval
        self.api_keys = api_keys or parse_api_keys()
        self._dispatch_path = (
            Path(dispatch_config_path)
            if dispatch_config_path
            else _resolve_dispatch_config_path()
        )
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._started = False

        self.ingest: IngestWorker | None = None
        self.dispatch: DispatchPublisher | None = None
        self.eval_worker: EvalWorker | None = None
        self.attribution: AttributionWorker | None = None
        self.annotation: AnnotationWorker | None = None
        self.benchmark: BenchmarkService | None = None
        self.config_loader: DispatchConfigLoader | None = None
        self.app: FastAPI | None = None
        self.span_store = InMemorySpanStore()
        self.eval_results = EvalResultStore()
        self.sampling_stats = SamplingStatsStore()
        self.event_bus = PipelineEventBus()
        self.dashboard_ctx: DashboardContext | None = None

    @classmethod
    async def create_from_env(cls, **kwargs: Any) -> DevStack:
        stack = cls(**kwargs)
        await stack.build()
        return stack

    async def build(self) -> DevStack:
        if self.persistence is None:
            self.persistence = await create_persistence()
        signal = await create_batch_signal_from_env()
        assert self.persistence is not None

        if self._dispatch_path is not None:
            self.config_loader = load_dispatch_config(self._dispatch_path)
        else:
            self.config_loader = DispatchConfigLoader()
            logger.warning("dispatch config not found; using built-in defaults")

        self.ingest = IngestWorker(
            self.mq,
            outbox=self.persistence.outbox,
            store=self.span_store,
            force_sample=self.force_sample,
            sampling_stats=self.sampling_stats,
            event_bus=self.event_bus,
        )
        self.dispatch = DispatchPublisher(
            self.mq,
            self.persistence.outbox,
            config_loader=self.config_loader,
        )
        await seed_default_prompts(self.persistence.prompts)
        self.eval_worker = EvalWorker(
            self.mq,
            dispatch_client=DispatchClient(self.mq, config_loader=self.config_loader),
            judge_router=build_judge_router(include_env_http=True),
            eval_results=self.eval_results,
            event_bus=self.event_bus,
            prompt_service=self.persistence.prompts,
        )
        self.attribution = AttributionWorker(
            self.mq,
            store=self.persistence.attributions,
            use_env_react_policy=True,
        )
        self.annotation = AnnotationWorker(self.mq, store=self.persistence.annotations)
        self.benchmark = BenchmarkService(
            store=self.persistence.benchmark_reports,
            signal=signal,
        )
        self.app = self._build_app()
        return self

    def _build_app(self) -> FastAPI:
        assert self.persistence is not None
        assert self.benchmark is not None

        async def on_startup(_app: FastAPI) -> None:
            await self.start_workers()

        async def on_shutdown(_app: FastAPI) -> None:
            await self.stop_workers()

        gw = create_app(
            mq=self.mq,
            authenticator=ApiKeyAuthenticator(self.api_keys),
            on_startup=on_startup,
            on_shutdown=on_shutdown,
        )
        ann = create_annotation_app(self.persistence.annotations)
        bench = create_benchmark_app(self.benchmark)
        prompt_admin = create_prompt_admin_app(self.persistence.prompts)

        async def stack_info() -> dict[str, Any]:
            return {
                "started": self._started,
                "persistence": self.persistence.driver if self.persistence else None,
                "force_sample": self.force_sample,
                "mq_driver": os.environ.get("LLMJUDGE_MQ") or "memory",
            }

        self.dashboard_ctx = DashboardContext(
            span_store=self.span_store,
            eval_results=self.eval_results,
            sampling=self.sampling_stats,
            annotations=self.persistence.annotations,
            attributions=self.persistence.attributions,
            benchmark_store=self.persistence.benchmark_reports,
            event_bus=self.event_bus,
            mq=self.mq,
            telemetry=gw.state.telemetry,
            stack_info=stack_info,
            prompt_service=self.persistence.prompts,
        )
        dash = create_dashboard_app(self.dashboard_ctx)
        gw.mount("/annotation", ann)
        gw.mount("/benchmark", bench)
        gw.mount("/prompt", prompt_admin)
        gw.mount("/dashboard", dash)

        @gw.get("/stackz")
        async def stackz() -> dict[str, Any]:
            depths = {
                "spans": await self.mq.depth(Topic.SPANS),
                "eval_jobs": await self.mq.depth(Topic.EVAL_JOBS),
                "attribution_jobs": await self.mq.depth(Topic.ATTRIBUTION_JOBS),
                "annotation_jobs": await self.mq.depth(Topic.ANNOTATION_JOBS),
            }
            return {
                "status": "ok",
                "started": self._started,
                "persistence": self.persistence.driver if self.persistence else None,
                "force_sample": self.force_sample,
                "mq_driver": os.environ.get("LLMJUDGE_MQ") or "memory",
                "mq_depths": depths,
            }

        return gw

    async def start_workers(self) -> None:
        if self._started:
            return
        if self.ingest is None:
            await self.build()
        assert self.ingest and self.dispatch and self.eval_worker
        assert self.attribution and self.annotation and self.config_loader
        self._stop.clear()
        self._tasks = [
            asyncio.create_task(self._loop("ingest", self._tick_ingest), name="llmj-ingest"),
            asyncio.create_task(self._loop("dispatch", self._tick_dispatch), name="llmj-dispatch"),
            asyncio.create_task(self._loop("eval", self._tick_eval), name="llmj-eval"),
            asyncio.create_task(
                self._loop("attribution", self._tick_attribution), name="llmj-attribution"
            ),
            asyncio.create_task(
                self._loop("annotation", self._tick_annotation), name="llmj-annotation"
            ),
            asyncio.create_task(
                self.config_loader.watch_loop(interval_s=2.0), name="llmj-dispatch-watch"
            ),
        ]
        self._started = True
        logger.info("dev_stack_workers_started tasks=%d", len(self._tasks))

    async def stop_workers(self) -> None:
        self._stop.set()
        if self.ingest is not None:
            self.ingest.stop()
        if self.config_loader is not None:
            self.config_loader.stop()
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._started = False
        logger.info("dev_stack_workers_stopped")

    async def _loop(self, name: str, tick) -> None:  # noqa: ANN001
        while not self._stop.is_set():
            try:
                n = await tick()
            except Exception:
                logger.exception("dev_stack_tick_failed name=%s", name)
                n = 0
            if not n:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                except TimeoutError:
                    pass

    async def _tick_ingest(self) -> int:
        assert self.ingest is not None
        return await self.ingest.process_once()

    async def _tick_dispatch(self) -> int:
        assert self.dispatch is not None
        return await self.dispatch.publish_pending()

    async def _tick_eval(self) -> int:
        assert self.eval_worker is not None
        return len(await self.eval_worker.process_once())

    async def _tick_attribution(self) -> int:
        assert self.attribution is not None
        return len(await self.attribution.process_once())

    async def _tick_annotation(self) -> int:
        assert self.annotation is not None
        return len(await self.annotation.process_once())

    async def tick_once(self) -> dict[str, int]:
        """Single synchronous round (tests / pump)."""
        if self.ingest is None:
            await self.build()
        return {
            "ingest": await self._tick_ingest(),
            "dispatch": await self._tick_dispatch(),
            "eval": await self._tick_eval(),
            "attribution": await self._tick_attribution(),
            "annotation": await self._tick_annotation(),
        }

    async def pump(self, *, max_rounds: int = 100) -> dict[str, int]:
        """Drain pipeline until a quiet round or max_rounds."""
        totals = {
            "ingest": 0,
            "dispatch": 0,
            "eval": 0,
            "attribution": 0,
            "annotation": 0,
            "rounds": 0,
        }
        for _ in range(max_rounds):
            step = await self.tick_once()
            totals["rounds"] += 1
            for k, v in step.items():
                totals[k] += v
            if sum(step.values()) == 0:
                break
        return totals


async def create_stack_app() -> FastAPI:
    stack = await DevStack.create_from_env()
    assert stack.app is not None
    return stack.app


def main() -> None:
    import uvicorn

    from llmobserverjudge.shared.env import load_env

    load_env()
    logging.basicConfig(
        level=os.environ.get("LLMJUDGE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    host = os.environ.get("LLMJUDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("LLMJUDGE_PORT", "8080"))
    # Build before uvicorn owns the event loop; workers start on ASGI lifespan.
    app = asyncio.run(create_stack_app())
    uvicorn.run(app, host=host, port=port)
