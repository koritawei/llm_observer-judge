"""MQ 消费端顶层编排 —— QueueManager + 有界并发 worker 池 + handler + 生命周期。

装配链路::

    RocketMQ PushConsumer (C 线程) × N 个来源
        └─ MetaqConsumer buffer ──poll──▶ MQConsumer.run() (每来源一个 async 循环)
                                              └─ _on_message: 阻塞式入队 worker 池（背压）
    EvalWorkerPool (固定 N workers, 并发, 队列有界；所有来源共享)
        └─ _process_task: 失败退避重试直到成功 → commit_message（成功才推进）

路由按 **topic → source → converter**：``config.consumers`` 里每个 source 绑定一个 topic，
本服务为每个启用订阅注册一个 consumer，并给它绑定 :class:`TolerantEvalCodec` (source)；
消息从哪个 topic 收到就用哪个 source 的转换器解析，所有来源汇入同一个 worker 池。

消费语义（本次要求）：**池中每个任务都必须消费成功后，该 worker 才推进到下一条**。

- **并发但有界**：固定 ``worker_concurrency`` 个 worker + 有界 ``worker_queue_size`` 队列；
  入队阻塞式（满则等待 worker 腾容量）→ 对上游 poll 背压，进程内最多持有 queue+并发 条，
  **绝不无限增长**（禁止内存无界堆积）。
- **每任务重试直到成功**：``_process_task`` 用退避 performer 反复调用 handler，直到成功
  （或永久性坏消息被 handler 主动跳过）；期间该 worker **不取下一条**。
  ``retry_max_attempts=0`` → 无限重试；>0 → 重试耗尽则跳过（避免坏消息永久占用 worker）。
- **per-attempt 超时**：单次评测包在 ``asyncio.wait_for(handler_timeout_s)`` 内，挂死 →
  超时抛错 → 重试，避免单任务永久卡住某个 worker 而毫无进展。
- **成功后才提交**：``auto_commit=False``；worker 在成功/跳过后 ``commit_message`` 精确提交
  该条（关停被 cancel 时不提交，交 broker 重投）。

并发下不保证**全局**顺序（每个 worker 对自己顺序处理的任务保序）；如需严格全局有序，
把 ``worker_concurrency`` 设为 1。

可靠性边界：进程内重试**不覆盖**"重试过程中进程崩溃"——MetaQ 常规 subscribe 在缓冲即
auto-ack，崩溃时在途消息可能丢；跨崩溃严格不丢需 broker 手动 ack（共享库未暴露该路径）。

扩展点（可扩展性）：新增来源只需在 ``conf`` 的 ``consumers`` 里加一个 source→topic，并在
``consumers/<source>`` 下实现并自注册转换器/补充器；本服务与 handler / 评测服务均无需改动。
"""

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.libs.mq import (
    BackoffPolicy,
    DefaultBackoffPerformer,
    RetryExhaustedError,
    TopicDefinition,
)

from benchmark.ingest.mq.enrichment import aclose_enrichers
from benchmark.ingest.mq.executor import EvalMessageExecutor
from benchmark.ingest.mq.protocol import BaseTaskMessage, TolerantEvalCodec
from benchmark.ingest.mq.runtime.config import EvalMQConfig, to_backend_configuration
from benchmark.ingest.mq.runtime.queue_manager import ConsumerOnlyMetaqQueueManager
from benchmark.ingest.mq.runtime.worker import EvalWorkerPool
from benchmark.services.evaluation import EvaluatorService

if TYPE_CHECKING:
    from benchmark.libs.mq import BackoffPerformer, ConsumedMessage, MQConsumer, QueueManager

logger = get_logger("agent_eval.mq.service")

#: 关停时等待 worker 池 drain 的上限（秒）。
_DRAIN_TIMEOUT_S = 10.0
#: 关停时等待消费循环退出的上限（秒）。
_CONSUMER_STOP_TIMEOUT_S = 5.0
#: 无限重试（``retry_max_attempts=0``）时判定“卡死”的尝试次数；有限重试时改取上限的
#: 一半（见 :func:`_alert_attempts`）。
_RETRY_ALERT_ATTEMPTS = 20
#: 单条消息累计卡住超过该秒数（即使次数未到）→ 同样升级 ERROR。
_RETRY_ALERT_STUCK_SECONDS = 600.0
#: 触发后 ERROR 的最小间隔（秒），避免长时间卡死刷屏；首次触发立即打。
_RETRY_ALERT_INTERVAL_S = 60.0


def _alert_attempts(retry_max_attempts: int) -> int:
    """判定“卡死”的尝试次数：有限重试取上限一半，无限重试用固定值。

    告警必须**早于**放弃，否则 ERROR 与“消息被丢弃”同时发生、彻底失去预警意义。而
    ``retry_max_attempts`` 在 ``conf/app.<env>.yaml``、触发次数在本模块常量——二者语义硬耦合却
    分处两地，各自硬编码时只要改一边就会静默失效（两者都是 20 时，ERROR 恰好在最后
    一次尝试才打出）。因此由配置派生，把约束写进代码。

    取一半而不是更小的比例：指数退避封顶 30s，上限 20 时一半（第 10 次）约在 2 分钟处，
    既能在放弃（约 7 分钟）前留出足够的介入窗口，又能排除 LLM 限流类瞬时抖动
    （那类通常在前 6 次、约 15s 内就恢复）带来的误报。
    """

    if retry_max_attempts <= 0:
        return _RETRY_ALERT_ATTEMPTS
    return max(1, retry_max_attempts // 2)


@dataclass(slots=True)
class _EvalTask:
    """一个待处理任务：解码后的 payload + 原始 ConsumedMessage（用于成功后 commit）。"""

    payload: BaseTaskMessage
    consumed: "ConsumedMessage[BaseTaskMessage]"


class EvalMQConsumerService:
    """管理评测消息 consumer 的完整生命周期（多来源订阅 + 有界并发池 + 每任务重试直到成功）。

    Args:
        evaluator_service: 评测服务（消息最终触发它的 ``evaluate``）。
        config: 已解析的 :class:`EvalMQConfig`（``consumers`` 描述多来源多 topic 订阅）。
        env: agent-eval ``Settings.env``（daily/pre/prod/其他）；决定 backend 与默认 namesrv。
        queue_manager: 可选注入（测试用 InMemory manager）；None 时按 config 自建 MetaQ manager。
        message_handler: 可选注入（测试用）；None 时用 :class:`EvalMessageExecutor`。
    """

    def __init__(
        self,
        *,
        evaluator_service: EvaluatorService,
        config: EvalMQConfig,
        env: str,
        queue_manager: "QueueManager | None" = None,
        message_handler: EvalMessageExecutor | None = None,
    ) -> None:
        self._config = config
        self._env = env
        self._handler = message_handler or EvalMessageExecutor(evaluator_service)
        self._queue_manager = queue_manager
        self._owns_manager = queue_manager is None
        #: 每个启用订阅一个 consumer；``(source, consumer)`` 顺序即注册顺序。
        self._consumers: list[tuple[str, MQConsumer]] = []
        #: 按 topic 名索引 consumer，供成功后精确 commit。
        self._consumers_by_topic: dict[str, MQConsumer] = {}
        self._consumer_tasks: list[asyncio.Task[None]] = []
        self._pool: EvalWorkerPool[_EvalTask] | None = None
        self._backoff: BackoffPerformer | None = None

    # ── lifecycle ───────────────────────────────────────────────────────

    async def start(self) -> None:
        """构建 backend + worker 池 + 注册各来源 consumer + 启动消费循环。"""

        cfg = self._config
        if self._queue_manager is None:
            # 用只消费的 QueueManager:避免 METAQ 消费链路创建两个同 group 的 producer
            # 导致 rocketmq "producer group has been created before"。见该模块 docstring。
            self._queue_manager = ConsumerOnlyMetaqQueueManager(
                to_backend_configuration(cfg, self._env)
            )

        # 每任务重试用的退避器（无状态可跨 worker/任务共享：perform 的 attempt 为局部变量）。
        self._backoff = DefaultBackoffPerformer(
            policy=BackoffPolicy.EXPONENTIAL,
            initial_backoff_ms=cfg.retry_initial_backoff_ms,
            maximum_backoff_ms=cfg.retry_max_backoff_ms,
            maximum_retries=cfg.retry_max_attempts,
        )
        self._pool = EvalWorkerPool(
            handler=self._process_task,
            concurrency=cfg.worker_concurrency,
            queue_size=cfg.worker_queue_size,
        )
        self._register_consumers(self._queue_manager)

        self._pool.start()
        await self._queue_manager.bootstrap()
        # QueueManager.bootstrap 只启动 backend；每个 consumer 需显式 start() 完成 backend
        # 订阅（MetaQ 侧即创建并启动 RocketMQ PushConsumer），之后 run() 才有消息可拉。
        for source, consumer in self._consumers:
            await consumer.start()
            task = asyncio.create_task(consumer.run(), name=f"eval-mq-consumer-{source}")
            task.add_done_callback(self._on_consumer_task_done)
            self._consumer_tasks.append(task)
        active = cfg.active_consumers()
        logger.info(
            "EvalMQConsumerService started consumers=%s env=%s "
            "concurrency=%d queue_size=%d retry_max_attempts=%d",
            [f"{name}->{c.topic}" for name, c in active],
            self._env,
            cfg.worker_concurrency,
            cfg.worker_queue_size,
            cfg.retry_max_attempts,
        )

    def _register_consumers(self, manager: "QueueManager") -> None:
        """为每个启用订阅注册一个 consumer：``auto_commit=False``（成功后由 worker 精确提交）。

        每个 consumer 绑定 :class:`TolerantEvalCodec` (source)：非 JSON/无法解码的坏消息 /
        未被该来源认领的结构 → ``BaseTaskMessage.rejected``（携 decode_error，由 handler 校验门
        跳过并提交），避免共享库 ERROR + 不提交导致重投。

        run() 的退避对 ``_on_message``（仅入队）生效，但入队为阻塞式、基本不失败，故消费
        循环不做重试——真正的"重试直到成功"在每个 worker 的 :meth:`_process_task` 内。
        """

        for source, topic_cfg in self._config.active_consumers():
            topic: TopicDefinition[BaseTaskMessage] = TopicDefinition(
                name=topic_cfg.topic, payload_type=BaseTaskMessage
            )
            consumer = manager.register_consumer(
                topic,
                self._on_message,
                group_id=topic_cfg.resolved_group(),
                poll_timeout_s=self._config.poll_timeout_s,
                auto_commit=False,
                codec=TolerantEvalCodec(source),
            )
            self._consumers.append((source, consumer))
            self._consumers_by_topic[topic_cfg.topic] = consumer

    async def stop(self) -> None:
        """优雅关停：停各消费循环 → drain worker → 关 QueueManager（仅自建时）。"""

        for _source, consumer in self._consumers:
            consumer.stop()
        if self._consumer_tasks:
            await self._await_consumer_tasks()
            self._consumer_tasks = []

        if self._pool is not None:
            drained = await self._pool.drain(_DRAIN_TIMEOUT_S)
            if not drained:
                logger.warning(
                    "EvalMQConsumerService drain timeout remaining=%d",
                    self._pool.queue_size,
                )
            await self._pool.stop()
            self._pool = None

        # worker 已停,补充器不再被调用:释放各来源补充器进程级复用的连接（如 luoshu httpx）。
        await aclose_enrichers()

        if self._queue_manager is not None and self._owns_manager:
            await self._queue_manager.shutdown()
            self._queue_manager = None
        self._consumers = []
        self._consumers_by_topic = {}
        logger.info("EvalMQConsumerService stopped")

    # ── message flow ────────────────────────────────────────────────────

    async def _on_message(self, msg: "ConsumedMessage[BaseTaskMessage]") -> None:
        """MQConsumer.run 回调（所有来源共用）：把消息**阻塞式**入队 worker 池。

        队列满则在此等待 worker 腾容量（背压）→ 消费循环放慢 poll、内存有界、不丢消息。
        """

        assert self._pool is not None  # start() 后必然存在
        await self._pool.enqueue(_EvalTask(payload=msg.payload, consumed=msg))

    async def _process_task(self, task: _EvalTask) -> None:
        """worker 执行体：对**当前任务**重试直到成功，成功/跳过后才 commit 推进。

        - 成功 / 永久性坏消息被跳过 → 正常返回 → commit（该 worker 取下一条）。
        - 有限重试耗尽（``retry_max_attempts>0``）→ 记录后跳过 + commit（不永久占用 worker）。
        - 关停被 cancel → 不 commit，交 broker 重投。
        """

        try:
            await self._run_until_success(task.payload)
        except asyncio.CancelledError:
            raise
        except RetryExhaustedError:
            logger.warning(
                "mq.task.retry_exhausted trace_id=%s -> skip",
                getattr(task.payload, "trace_id", None),
            )
        await self._commit(task.consumed)

    async def _run_until_success(self, payload: BaseTaskMessage) -> None:
        """反复尝试处理该任务，直到 handler 正常返回（成功或永久性跳过）。

        单次尝试包 ``handler_timeout_s`` 超时；handler 抛异常（可重试 5xx / 未预期）→
        本次尝试失败 → 退避重试；handler 正常返回 → 成功。``retry_max_attempts=0`` 时
        无限重试；>0 且耗尽 → performer 抛 :class:`RetryExhaustedError`。
        """

        assert self._backoff is not None  # start() 后必然存在

        # 无限重试下的可观测性：累计尝试次数 + 起始时间，越过阈值（次数或卡住时长）判定为
        # “卡死”并升级为 ERROR，暴露毒消息/系统性故障导致的 head-of-line blocking——避免
        # worker 被无声钉死、只能等吞吐掉底才发现。ERROR 首次越阈即打、之后按间隔节流防刷屏。
        attempts = 0
        started = time.monotonic()
        last_alert = 0.0
        alert_attempts = _alert_attempts(self._config.retry_max_attempts)
        trace_id = getattr(payload, "trace_id", None)

        async def _attempt() -> bool:
            nonlocal attempts, last_alert
            attempts += 1
            try:
                await asyncio.wait_for(
                    self._handler.handle(payload),
                    timeout=self._config.handler_timeout_s,
                )
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                now = time.monotonic()
                elapsed = now - started
                stuck = attempts >= alert_attempts or elapsed >= _RETRY_ALERT_STUCK_SECONDS
                if stuck and (last_alert == 0.0 or now - last_alert >= _RETRY_ALERT_INTERVAL_S):
                    last_alert = now
                    logger.error(
                        "mq.task.retry_stuck trace_id=%s attempts=%d elapsed_s=%.0f err=%r",
                        trace_id,
                        attempts,
                        elapsed,
                        exc,
                    )
                else:
                    logger.warning(
                        "mq.task.retry trace_id=%s attempts=%d err=%r",
                        trace_id,
                        attempts,
                        exc,
                    )
                return False

        await self._backoff.perform(_attempt)

    async def _commit(self, consumed: "ConsumedMessage[BaseTaskMessage]") -> None:
        # 按消息所属 topic 找到对应 consumer 精确提交（多来源共享一个 worker 池）。
        consumer = self._consumers_by_topic.get(consumed.topic)
        if consumer is None:
            return
        try:
            await consumer.commit_message(consumed)
        except Exception:
            logger.warning(
                "EvalMQConsumerService commit failed topic=%s partition=%s offset=%s",
                consumed.topic,
                consumed.partition,
                consumed.offset,
                exc_info=True,
            )

    # ── internals ───────────────────────────────────────────────────────

    async def _await_consumer_tasks(self) -> None:
        tasks = [t for t in self._consumer_tasks if t is not None]
        if not tasks:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*tasks, return_exceptions=True)),
                timeout=_CONSUMER_STOP_TIMEOUT_S,
            )
        except TimeoutError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    def _on_consumer_task_done(self, task: asyncio.Task[None]) -> None:
        """消费循环意外退出（非正常 stop）时告警，便于可用性监控。"""

        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("EvalMQConsumerService consumer loop exited with error: %r", exc)


__all__ = ["EvalMQConsumerService"]
