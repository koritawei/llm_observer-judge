"""通用周期心跳（benchmark runner 多节点模式）。"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

#: run 级心跳间隔（秒）—— 多节点 worker 模式
RUN_HEARTBEAT_INTERVAL_S = 15.0


@dataclass(frozen=True)
class HeartbeatSpec:
    """心跳循环规格（``heartbeat_loop`` 参数对象）。"""

    heartbeat_fn: Callable[[], Awaitable[Any]]
    logger: logging.Logger
    fail_log_message: str
    fail_log_args: tuple[Any, ...] = ()


async def heartbeat_loop(
    spec: HeartbeatSpec,
    *,
    interval_s: float,
) -> None:
    """周期心跳循环：``sleep(interval) → heartbeat_fn()``，异常仅 debug 不中断。"""
    while True:
        await asyncio.sleep(interval_s)
        try:
            await spec.heartbeat_fn()
        except Exception:
            spec.logger.debug(
                spec.fail_log_message,
                *spec.fail_log_args,
                exc_info=True,
            )


async def cancel_and_wait(task: asyncio.Task[Any]) -> None:
    """``cancel()`` + 吞 ``CancelledError`` 的标准收尾模板。"""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


__all__ = [
    "RUN_HEARTBEAT_INTERVAL_S",
    "HeartbeatSpec",
    "cancel_and_wait",
    "heartbeat_loop",
]
