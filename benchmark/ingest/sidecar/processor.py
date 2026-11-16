"""Parse → thin-filter → Lindorm+OSS readback → unpack → benchmark judge."""

import asyncio
import threading
from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import JsonValue

from benchmark.domain.eval_config import EvalConfigEntry, evaluator_slot_label
from benchmark.domain.trace_filters import should_skip_test_adiu
from benchmark.evaluators.base import EvaluatorRequest
from benchmark.ingest.sidecar.converter import (
    SidecarIdentityError,
    to_sidecar_evaluator_request,
)
from benchmark.ingest.sidecar.filter import notify_match_reason
from benchmark.ingest.sidecar.funnel import SidecarFunnel, _hour_key_cn
from benchmark.ingest.sidecar.notify import (
    SidecarNotifyParseError,
    SidecarTraceNotify,
    parse_notify,
)
from benchmark.ingest.sidecar.observations import extract_trace_fields
from benchmark.ingest.sidecar.trace_reader import SidecarTraceReader
from benchmark.ingest.sidecar.unpack import UnpackError, unpack_observations
from benchmark.utils.exceptions import EvaluatorError

logger = get_logger("agent_eval.sidecar_ingest.processor")

_HTTP_SERVER_ERROR = 500


class BenchmarkJudgeProtocol(Protocol):
    """``BenchmarkStreamingEvaluator.judge_ingested``. Does not call ``evaluate``."""

    async def judge_ingested(
        self,
        config: EvalConfigEntry,
        request: EvaluatorRequest,
        observations: list[dict[str, JsonValue]],
    ) -> object: ...


class PermanentSkip(Exception):  # noqa: N818
    """Ack without retry: parse miss, filter miss, or missing eval identity."""


class SidecarProcessor:
    def __init__(
        self,
        *,
        reader: SidecarTraceReader,
        benchmark_judge: BenchmarkJudgeProtocol,
        evaluate_timeout_s: float = 300.0,
        funnel: SidecarFunnel | None = None,
    ) -> None:
        self._reader = reader
        self._benchmark_judge = benchmark_judge
        self._evaluate_timeout_s = evaluate_timeout_s
        self._funnel = funnel or SidecarFunnel()
        self._sample_hour = ""
        self._seen_reasons: set[str] = set()
        self._sample_lock = threading.Lock()

    async def handle(
        self,
        raw: bytes | str | dict[str, Any],
        *,
        config: EvalConfigEntry,
        project_id: str,
    ) -> None:
        self._funnel.bump("msg_total")
        try:
            notify = parse_notify(raw)
        except SidecarNotifyParseError as exc:
            self._funnel.bump("skip_invalid")
            logger.warning("sidecar.notify_invalid error=%s", exc)
            raise PermanentSkip from exc

        self._funnel.bump("received")
        reason = notify_match_reason(notify, config, project_id=project_id)
        if reason != "hit":
            self._funnel.bump(reason)
            if reason == "sample_miss":
                self._funnel.bump("eligible")
            self._log_filter_sample(reason, notify, config)
            logger.info(
                "sidecar.filter_miss trace_id=%s project_id=%s slot=%s reason=%s",
                notify.trace_id,
                notify.project_id,
                evaluator_slot_label(config),
                reason,
            )
            raise PermanentSkip

        self._funnel.bump("eligible")
        self._funnel.bump("sample_hit")
        await self._evaluate_hit(notify, config)

    def _log_filter_sample(
        self,
        reason: str,
        notify: SidecarTraceNotify,
        config: EvalConfigEntry,
    ) -> None:
        """One WARNING example per drop reason per hour, with the compared fields."""
        hour = _hour_key_cn()
        with self._sample_lock:
            if hour != self._sample_hour:
                self._seen_reasons.clear()
                self._sample_hour = hour
            if reason in self._seen_reasons:
                return
            self._seen_reasons.add(reason)
        logger.warning(
            "sidecar.filter_sample reason=%s trace_id=%s project_id=%s "
            "notify_name=%s config_name=%s notify_env=%s config_env=%s "
            "notify_agent=%s sampling=%s",
            reason,
            notify.trace_id,
            notify.project_id,
            notify.name,
            config.name,
            notify.environment,
            config.environment,
            notify.agent_name,
            config.sampling,
        )

    async def _evaluate_hit(
        self,
        notify: SidecarTraceNotify,
        config: EvalConfigEntry,
    ) -> None:
        try:
            record = await self._reader.fetch_by_trace_id(notify.trace_id)
        except Exception:
            self._funnel.bump("read_fail")
            raise
        if record is None:
            self._funnel.bump("read_fail")
            raise LookupError(f"sidecar trace miss trace_id={notify.trace_id}")
        try:
            observations = unpack_observations(record.observations)
        except UnpackError as exc:
            self._funnel.bump("read_fail")
            logger.warning(
                "sidecar.unpack_failed trace_id=%s error=%s",
                notify.trace_id,
                exc,
            )
            raise PermanentSkip from exc
        if not observations:
            self._funnel.bump("read_fail")
            logger.warning("sidecar.empty_observations trace_id=%s", notify.trace_id)
            raise PermanentSkip

        inp, _out, _metadata = extract_trace_fields(observations)
        if should_skip_test_adiu(user_id=notify.user_id, input_value=inp):
            logger.info(
                "sidecar.test_adiu_skip trace_id=%s project_id=%s user_id=%s",
                notify.trace_id,
                notify.project_id,
                notify.user_id,
            )
            raise PermanentSkip

        try:
            request = to_sidecar_evaluator_request(notify, observations, config)
        except SidecarIdentityError as exc:
            logger.warning(
                "sidecar.identity_missing trace_id=%s error=%s",
                notify.trace_id,
                exc,
            )
            raise PermanentSkip from exc
        await self._evaluate_one(config, request, notify.trace_id, observations)

    async def _evaluate_one(
        self,
        config: EvalConfigEntry,
        request: EvaluatorRequest,
        trace_id: str,
        observations: list[Any],
    ) -> None:
        slot = evaluator_slot_label(config)
        try:
            call = self._benchmark_judge.judge_ingested(config, request, observations)
            if self._evaluate_timeout_s > 0:
                await asyncio.wait_for(call, timeout=self._evaluate_timeout_s)
            else:
                await call
        except TimeoutError:
            self._funnel.bump("eval_fail_timeout")
            logger.warning(
                "sidecar.evaluate_timeout trace_id=%s slot=%s",
                trace_id,
                slot,
            )
            return
        except EvaluatorError as exc:
            if exc.code < _HTTP_SERVER_ERROR:
                self._funnel.bump("eval_fail_permanent")
                logger.warning(
                    "sidecar.evaluate_permanent trace_id=%s slot=%s code=%s error_type=%s",
                    trace_id,
                    slot,
                    exc.code,
                    exc.error_type,
                )
                return
            self._funnel.bump("eval_fail_transient")
            raise
        self._funnel.bump("eval_ok")
        logger.info(
            "sidecar.evaluated trace_id=%s slot=%s",
            trace_id,
            slot,
        )


__all__ = ["PermanentSkip", "SidecarProcessor"]
