"""AsyncSender — httpx batch ingest + telemetry (design §3.1.3)."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from llmobserverjudge.sdk.buffer import BoundedRingBuffer
from llmobserverjudge.sdk.config import SDKConfig

logger = logging.getLogger(__name__)


class AsyncSender:
    def __init__(
        self,
        config: SDKConfig,
        buffer: BoundedRingBuffer,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.buffer = buffer
        base = config.endpoint.rstrip("/")
        self.ingest_url = f"{base}/api/v1/ingest"
        self.telemetry_url = f"{base}/api/v1/telemetry"
        self._owns_http = http is None
        self.http = http or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            timeout=httpx.Timeout(connect=2.0, read=5.0, write=3.0, pool=10.0),
        )
        self._consecutive_failures = 0
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.config.api_key:
            h["X-API-Key"] = self.config.api_key
        return h

    async def flush_batch(self, spans: list[dict[str, Any]]) -> None:
        if not spans or not self.config.enabled:
            return
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                resp = await self.http.post(
                    self.ingest_url,
                    json={"spans": spans},
                    headers=self._headers(),
                )
                if resp.status_code == 200:
                    self.buffer.metrics.sent_spans_total += len(spans)
                    self._consecutive_failures = 0
                    return
                if resp.status_code == 429:
                    self.buffer.metrics.dropped_spans_total += len(spans)
                    retry_after = int(resp.headers.get("Retry-After", "10"))
                    logger.warning(
                        "sdk_rate_limited count=%d retry_after=%ds",
                        len(spans),
                        retry_after,
                    )
                    return
                if resp.status_code == 503 and attempt < max_retries:
                    delay = (2**attempt) * 0.1 + random.uniform(0, 0.05)
                    await asyncio.sleep(delay)
                    continue
                self.buffer.metrics.dropped_spans_total += len(spans)
                self._consecutive_failures += 1
                logger.warning(
                    "sdk_send_failed status=%d count=%d", resp.status_code, len(spans)
                )
                return
            except httpx.HTTPError as exc:
                if attempt < max_retries:
                    delay = (2**attempt) * 0.1 + random.uniform(0, 0.05)
                    logger.warning("sdk_send_error_retry attempt=%d err=%s", attempt + 1, exc)
                    await asyncio.sleep(delay)
                    continue
                self.buffer.metrics.dropped_spans_total += len(spans)
                self._consecutive_failures += 1
                logger.warning("sdk_send_error err=%s count=%d", exc, len(spans))
                return
        self.buffer.metrics.dropped_spans_total += len(spans)
        self._consecutive_failures += 1

    async def flush(self) -> int:
        batch = self.buffer.drain(self.config.flush_max_batch)
        if not batch:
            return 0
        payloads = [
            s if isinstance(s, dict) else s.to_ingest_dict()  # type: ignore[union-attr]
            for s in batch
        ]
        await self.flush_batch(payloads)
        return len(payloads)

    async def _flush_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.flush_interval_s
                )
                break
            except TimeoutError:
                await self.flush()

    async def _telemetry_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.telemetry_interval_s
                )
                break
            except TimeoutError:
                try:
                    await self.http.post(
                        self.telemetry_url,
                        json={
                            "sent_spans_total": self.buffer.metrics.sent_spans_total,
                            "dropped_spans_total": self.buffer.metrics.dropped_spans_total,
                            "buffer_usage": self.buffer.metrics.current_usage,
                            "buffer_capacity": self.buffer.metrics.buffer_size,
                        },
                        headers=self._headers(),
                    )
                except httpx.HTTPError:
                    pass

    def start_background(self) -> None:
        if self._tasks:
            return
        self._stop.clear()
        self._tasks = [
            asyncio.create_task(self._flush_loop(), name="llmj-flush"),
            asyncio.create_task(self._telemetry_loop(), name="llmj-telemetry"),
        ]

    async def shutdown(self) -> None:
        self._stop.set()
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
        while len(self.buffer) > 0:
            await self.flush()
        if self._owns_http:
            await self.http.aclose()
