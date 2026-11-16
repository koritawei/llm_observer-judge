"""Sidecar ingest lifecycle: one MetaQ consumer per eval config."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.mq import (
    BackendType,
    CommitError,
    Configuration,
    ConsumeError,
    Env,
    RawMessage,
)
from benchmark.libs.mq._backends.metaq.backend import MetaqConsumerBackend

from benchmark.domain.eval_config import EvalConfigStore
from benchmark.domain.runtime_config_store import (
    register_eval_snapshot_listener,
    unregister_eval_snapshot_listener,
)
from benchmark.domain.schemas import EvalConfigEntry
from benchmark.ingest.sidecar.config import SidecarTraceConfig, resolve_namesrv_domain
from benchmark.ingest.sidecar.consumer_group import (
    SidecarConsumerIdentity,
    consumer_group_for,
    iter_sidecar_eval_entries,
    live_sidecar_entries,
)
from benchmark.ingest.sidecar.funnel import SidecarFunnel
from benchmark.ingest.sidecar.oss_blob import OssBlob
from benchmark.ingest.sidecar.processor import (
    BenchmarkJudgeProtocol,
    PermanentSkip,
    SidecarProcessor,
)
from benchmark.ingest.sidecar.trace_reader import SidecarTraceReader

logger = get_logger("agent_eval.sidecar_ingest.service")

_ENV_NAME_MAP = {"daily": "daily", "pre": "pre", "prod": "prod"}
_ACK_TIMEOUT_SLACK_S = 60.0
_POLL_ERROR_BACKOFF_S = 1.0
_RECONCILE_INTERVAL_S = 30.0
_CONSUMER_STOP_TIMEOUT_S = 1.0


@dataclass(slots=True)
class _EvalConsumer:
    identity: SidecarConsumerIdentity
    group: str
    backend: MetaqConsumerBackend
    task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True)
class _FanoutWork:
    """同 CID 下单条 stream entry 的 fan-out 入参（FUNC-PARAMS-001）。"""

    raw: RawMessage
    entry: EvalConfigEntry
    project_id: str
    group: str


@dataclass(frozen=True)
class SidecarIngestDeps:
    """``SidecarIngestService`` 构造依赖（FUNC-PARAMS-001 收敛）。"""

    benchmark_judge: BenchmarkJudgeProtocol
    config: SidecarTraceConfig
    env: str
    lindorm_dsn: str
    oss: OssBlob | None = None
    config_store: EvalConfigStore | None = None
    reader: SidecarTraceReader | None = None
    backend_factory: Callable[[], MetaqConsumerBackend] | None = None


class SidecarIngestService:
    """MetaQ 消费 ``ingest=sidecar``，按稳定 identity 幂等热对账。"""

    def __init__(self, deps: SidecarIngestDeps) -> None:
        self._benchmark_judge = deps.benchmark_judge
        self._config = deps.config
        self._env = deps.env
        self._lindorm_dsn = deps.lindorm_dsn
        self._config_store = deps.config_store or EvalConfigStore()
        if deps.reader is not None:
            self._reader = deps.reader
        else:
            self._reader = SidecarTraceReader(
                deps.lindorm_dsn,
                deps.config.lindorm_table,
                oss=deps.oss or OssBlob(),
            )
        self._owns_reader = deps.reader is None
        self._backend_factory = deps.backend_factory
        self._funnel = SidecarFunnel()
        self._processor: SidecarProcessor | None = None
        self._consumers: dict[SidecarConsumerIdentity, _EvalConsumer] = {}
        self._reconcile_task: asyncio.Task[None] | None = None
        self._reconcile_lock = asyncio.Lock()
        self._snapshot_listener_registered = False
        self._pending_snapshot_reconciles: set[asyncio.Task[None]] = set()
        self._stopping = False

    @property
    def active_consumer_count(self) -> int:
        """当前已订阅的 MetaQ consumer 数（0 = MQ 控制台无连接）。"""
        return len(self._consumers)

    async def start(self) -> None:
        if self._reconcile_task is not None:
            logger.warning("SidecarIngestService already started; skipping")
            return
        self._stopping = False
        try:
            await self.reconcile_once(strict=True)
        except Exception:
            await self.stop()
            raise
        self._log_reconcile_summary(initial=True)
        register_eval_snapshot_listener(self._on_eval_snapshot_refreshed)
        self._snapshot_listener_registered = True
        self._reconcile_task = asyncio.create_task(
            self._reconcile_loop(),
            name="sidecar-ingest-reconcile",
        )

    async def _on_eval_snapshot_refreshed(self) -> None:
        """快照刷新后 fire-and-forget 热对账，不阻塞 API / 30s 刷新循环。

        终态与同步 await reconcile 一致（增删 consumer）；仅返回时机提前。
        """
        if self._stopping or self._reconcile_task is None:
            return
        task = asyncio.create_task(
            self._reconcile_after_snapshot(),
            name="sidecar-reconcile-after-snapshot",
        )
        self._pending_snapshot_reconciles.add(task)
        task.add_done_callback(self._pending_snapshot_reconciles.discard)

    async def _reconcile_after_snapshot(self) -> None:
        try:
            await self.reconcile_once()
        except Exception:
            logger.exception("sidecar.reconcile_after_snapshot_failed")

    async def _reconcile_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(_RECONCILE_INTERVAL_S)
            try:
                await self.reconcile_once()
            except Exception:
                logger.exception("sidecar.reconcile_failed")

    async def reconcile_once(self, *, strict: bool = False) -> None:
        """desired 与 running 做集合差异；单条失败不影响其他 consumer。

        支持 evaluator 动态上下线：``ingest=sidecar`` 新增 → 订 Topic；
        删除 / 改非 sidecar → 停 consumer。identity 不变的条目不重启。
        持锁后再判 ``_stopping``，避免 stop 并发留下孤儿 consumer。
        """
        async with self._reconcile_lock:
            if self._stopping:
                return
            before = set(self._consumers)
            bindings = iter_sidecar_eval_entries(self._config_store.get_config_file())
            # ACL CID 仅 project+evaluator：多 stream/registration 合并为一个订阅。
            desired = {
                SidecarConsumerIdentity.from_entry(project_id, entry)
                for project_id, entry in bindings
            }
            for identity in tuple(self._consumers):
                if identity not in desired:
                    await self._stop_consumer(identity)
            for identity in desired:
                if identity in self._consumers:
                    continue
                if self._stopping:
                    return
                try:
                    await self._start_consumer(identity)
                except Exception:
                    if strict:
                        raise
                    logger.exception(
                        "sidecar.consumer_start_failed project_id=%s evaluator=%s",
                        identity.project_id,
                        identity.evaluator_name,
                    )
            after = set(self._consumers)
            added = after - before
            removed = before - after
            if added or removed:
                logger.info(
                    "sidecar.consumers_changed added=%d removed=%d active=%d desired=%d",
                    len(added),
                    len(removed),
                    len(after),
                    len(desired),
                )
            if not desired and before and not after:
                self._log_reconcile_summary(initial=False)

    def _log_reconcile_summary(self, *, initial: bool) -> None:
        """开关已开但无 ``ingest=sidecar`` 条目时告警：否则会静默零连接。"""
        count = self.active_consumer_count
        if count == 0:
            logger.warning(
                "sidecar.enabled_but_no_ingest_entries initial=%s "
                "SIDECAR_TRACE_ENABLED 已开但 runtime 快照无 ingest=sidecar；"
                "不创建 MetaQ consumer（控制台无连接、无处理日志）。"
                "请确认实时数据集/评测配置写入 ingest=sidecar 并刷新快照",
                initial,
            )
            return
        logger.info(
            "SidecarIngestService reconcile ok initial=%s active_consumers=%d",
            initial,
            count,
        )

    async def _ensure_processor(self) -> None:
        if self._processor is not None:
            return
        if not self._lindorm_dsn:
            raise RuntimeError("SIDECAR_TRACE_LINDORM_DSN is required")
        await self._reader.open()
        self._processor = SidecarProcessor(
            reader=self._reader,
            benchmark_judge=self._benchmark_judge,
            evaluate_timeout_s=self._config.evaluate_timeout_s,
            funnel=self._funnel,
        )

    async def _start_consumer(self, identity: SidecarConsumerIdentity) -> None:
        await self._ensure_processor()
        group = consumer_group_for(self._config.consumer_group, identity)
        topic = f"{self._config.topic_prefix}{identity.project_id}"
        backend = self._new_backend()
        try:
            await backend.start()
            await backend.subscribe_with_expressions({topic: None}, group)
        except BaseException:
            # CancelledError 等不继承 Exception：停机取消时也必须 close，
            # 否则 backend 未入 _consumers，stop 扫不到 → 孤儿建连。
            await _close_backend(backend, group=group)
            raise
        consumer = _EvalConsumer(identity=identity, group=group, backend=backend)
        self._consumers[identity] = consumer
        consumer.task = asyncio.create_task(
            self._run(consumer),
            name=f"sidecar-trace-ingest-{group}",
        )
        logger.info(
            "SidecarIngestService started topic=%s group=%s evaluator=%s",
            topic,
            group,
            identity.evaluator_name,
        )

    async def _stop_consumer(self, identity: SidecarConsumerIdentity) -> None:
        consumer = self._consumers.pop(identity, None)
        if consumer is None:
            return
        if consumer.task is not None:
            consumer.task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(consumer.task, return_exceptions=True),
                    timeout=_CONSUMER_STOP_TIMEOUT_S,
                )
            except TimeoutError:
                logger.warning("sidecar.consumer_stop_timeout group=%s", consumer.group)
        await _close_backend(consumer.backend, group=consumer.group)
        logger.info(
            "SidecarIngestService consumer stopped project_id=%s evaluator=%s group=%s",
            identity.project_id,
            identity.evaluator_name,
            consumer.group,
        )

    async def stop(self, timeout: float = 30.0) -> None:
        self._stopping = True
        if self._snapshot_listener_registered:
            unregister_eval_snapshot_listener(self._on_eval_snapshot_refreshed)
            self._snapshot_listener_registered = False
        pending = tuple(self._pending_snapshot_reconciles)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._pending_snapshot_reconciles.clear()
        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(self._reconcile_task, return_exceptions=True),
                    timeout=timeout,
                )
            except TimeoutError:
                pass
            self._reconcile_task = None
        for identity in tuple(self._consumers):
            await self._stop_consumer(identity)
        if self._owns_reader:
            await _close_reader(self._reader)
        self._funnel.flush()
        logger.info("SidecarIngestService stopped")

    def _new_backend(self) -> MetaqConsumerBackend:
        if self._backend_factory is not None:
            return self._backend_factory()
        return MetaqConsumerBackend(_to_configuration(self._config, self._env))

    async def _handle_fanout_entry(self, work: _FanoutWork) -> None:
        """单条 stream entry 的 fan-out 处理；PermanentSkip / 其它异常不向外抛。"""
        assert self._processor is not None
        try:
            await self._processor.handle(
                work.raw.value,
                config=work.entry,
                project_id=work.project_id,
            )
        except PermanentSkip:
            return
        except Exception:
            logger.exception(
                "sidecar.handle_failed topic=%s group=%s offset=%s",
                work.raw.topic,
                work.group,
                work.raw.offset,
            )

    async def _run(self, consumer: _EvalConsumer) -> None:
        assert self._processor is not None
        identity = consumer.identity
        while not self._stopping:
            try:
                raw = await consumer.backend.poll(self._config.poll_timeout_s)
            except ConsumeError:
                logger.exception("sidecar.poll_failed group=%s", consumer.group)
                await asyncio.sleep(_POLL_ERROR_BACKOFF_S)
                continue
            if raw is None:
                # 空 poll 立即返回时也必须让出事件循环，保证 cancel/timeout 可调度。
                await asyncio.sleep(0)
                continue
            entries = live_sidecar_entries(
                self._config_store.get_configs(identity.project_id),
                identity,
            )
            if not entries:
                self._funnel.bump("msg_total")
                self._funnel.bump("received")
                self._funnel.bump("filter_no_config")
                logger.info(
                    "sidecar.entry_gone project_id=%s evaluator_name=%s",
                    identity.project_id,
                    identity.evaluator_name,
                )
                await _settle(consumer.backend, raw, accepted=True)
                continue
            # 先 ack，再评测。评测和归因可以超过 MetaQ callback 等待时间；
            # 不在这里 wait_for，避免把还在跑的 LLM 收成 CancelledError。
            # 同 CID 下多 stream：已 ack，用 gather 并发 copy fan-out（各自采样/过滤），
            # 避免串行把下一轮 poll 拖到 evaluate_timeout_s * N。
            await _settle(consumer.backend, raw, accepted=True)
            if len(entries) > 1:
                logger.info(
                    "sidecar.fanout_concurrent topic=%s group=%s entries=%d",
                    raw.topic,
                    consumer.group,
                    len(entries),
                )
            await asyncio.gather(
                *(
                    self._handle_fanout_entry(
                        _FanoutWork(
                            raw=raw,
                            entry=entry,
                            project_id=identity.project_id,
                            group=consumer.group,
                        )
                    )
                    for entry in entries
                )
            )


def _to_configuration(cfg: SidecarTraceConfig, env: str) -> Configuration:
    mq_env = Env(_ENV_NAME_MAP.get(env, "local"))
    return Configuration(
        env=mq_env,
        backend=BackendType.METAQ,
        service_name="agent-eval",
        extra={
            "namesrv_domain": resolve_namesrv_domain(cfg, env),
            "max_buffer_size": str(cfg.max_buffer_size),
            "ack_timeout": str(cfg.handler_timeout_s + _ACK_TIMEOUT_SLACK_S),
            "orderly_consume": "false",
        },
    )


async def _close_backend(backend: MetaqConsumerBackend, *, group: str) -> None:
    try:
        await backend.close()
    except Exception:
        logger.exception("sidecar.backend_close_failed group=%s", group)


async def _close_reader(reader: SidecarTraceReader) -> None:
    try:
        await reader.close()
    except Exception:
        logger.exception("sidecar.reader_close_failed")


async def _settle(
    backend: MetaqConsumerBackend,
    raw: RawMessage,
    *,
    accepted: bool,
) -> None:
    try:
        await backend.settle_message(
            raw.topic,
            raw.partition,
            raw.offset,
            delivery_token=raw.delivery_token,
            accepted=accepted,
        )
    except CommitError:
        logger.warning(
            "sidecar.settle_missed accepted=%s topic=%s offset=%s",
            accepted,
            raw.topic,
            raw.offset,
        )


__all__ = ["SidecarIngestService"]
