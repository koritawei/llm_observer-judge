"""run(TaskPack) -> ResultPack. Serial; returns only after aoned finishes."""

import fcntl
import logging
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from benchmark.settings.setting import settings
from benchmark.attribution.code.cloud_agent import (
    AuthError,
    Events,
    HttpTransport,
    TransportError,
    is_readonly,
)
from benchmark.attribution.code.models import ModuleConfig, ResultPack, RunTimings, TaskPack
from benchmark.attribution.code.parse import parse_result
from benchmark.attribution.code.prompts import render_system, render_user

logger = logging.getLogger(__name__)

_RUN_LOCK = threading.Semaphore(1)
_LOCK_PATH = Path(tempfile.gettempdir()) / "code-attribution.lock"


@contextmanager
def _machine_lock() -> Iterator[None]:
    handle = _LOCK_PATH.open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def module_config_from_settings() -> ModuleConfig:
    """Build the aoned connection from process config. When the switch is off, run returns failed."""
    return ModuleConfig(
        base_url=settings.cloud_agent_base_url,
        token=settings.cloud_agent_token,
        project_path=settings.cloud_agent_project_path,
        live_enabled=settings.code_attribution_enabled,
    )


class AnyTransport:
    """Structural convention so the runner does not depend on a concrete HTTP class."""

    def create_session(self, *, prompt: str) -> str: ...

    def poll_events(self, session_id: str) -> Events: ...

    def answer_approval(self, session_id: str, approval_id: str, *, approve: bool) -> None: ...

    def stop(self, session_id: str) -> None: ...


@dataclass
class ModuleRuntime:
    """Transport and clock injected by tests. The default uses real aoned and the system clock."""

    transport: AnyTransport | None = None
    clock: Callable[[], float] | None = None
    sleep: Callable[[float], None] | None = None


class CodeAttributionModule:
    def __init__(
        self,
        config: ModuleConfig | None = None,
        runtime: ModuleRuntime | None = None,
    ) -> None:
        runtime = runtime if runtime is not None else ModuleRuntime()
        self.config = config if config is not None else module_config_from_settings()
        self._transport = runtime.transport
        self._clock = runtime.clock or time.monotonic
        self._sleep = runtime.sleep or time.sleep
        self.last_timings = RunTimings()

    def run(self, task: TaskPack) -> ResultPack:
        with _RUN_LOCK, _machine_lock():
            started = self._clock()
            result = self._run(task)
            self.last_timings.total_sec = self._clock() - started
            return result

    def _run(self, task: TaskPack) -> ResultPack:
        self.last_timings = RunTimings()
        if not self.config.live_enabled:
            return ResultPack(status="failed", error_message="aoned live call disabled")
        if self._transport is None and (not self.config.base_url or not self.config.token):
            return ResultPack(status="failed", error_message="aoned base url or token missing")
        if self._transport is None and not self.config.project_path:
            return ResultPack(status="failed", error_message="aoned project path missing")
        try:
            return self._analyze(task)
        except AuthError as exc:
            return ResultPack(status="failed", error_message=str(exc))
        except (TransportError, OSError) as exc:
            logger.warning("code attribution failed case_id=%s error=%s", task.case_id, exc)
            return ResultPack(status="failed", error_message=str(exc))

    def _analyze(self, task: TaskPack) -> ResultPack:
        transport = self._transport or HttpTransport(self.config)
        repo_root = task.repo_root or self.config.project_path
        prompt = (
            render_system(repo_root=repo_root, repo=task.repo)
            + "\n\n"
            + render_user(task, repo_root=repo_root)
        )
        for attempt in (1, 2):
            self.last_timings.attempts = attempt
            session_started = self._clock()
            session_id = transport.create_session(prompt=prompt)
            self.last_timings.session_sec += self._clock() - session_started
            try:
                outcome = self._poll(transport, session_id)
            finally:
                self._close_session(transport, session_id)
            if outcome is not None:
                return outcome
        return ResultPack(status="timeout", error_message="aoned timeout")

    def _close_session(self, transport: AnyTransport, session_id: str) -> None:
        """A finished session is kept for review. A still-running one is stopped first to free the serial slot."""
        try:
            if transport.poll_events(session_id).is_processing:
                transport.stop(session_id)
        except (AuthError, TransportError, OSError) as exc:
            logger.warning("aoned stop failed session_id=%s error=%s", session_id, exc)

    def _poll(self, transport: AnyTransport, session_id: str) -> ResultPack | None:
        """Returning a ResultPack means this round is done; None means timeout, and the caller decides whether to retry."""
        deadline = self._clock() + self.config.timeout_sec
        while True:
            events = transport.poll_events(session_id)
            self.last_timings.poll_count += 1
            for approval in events.pending_approvals:
                if not approval.id:
                    continue
                transport.answer_approval(
                    session_id,
                    approval.id,
                    approve=is_readonly(approval),
                )
            if not events.is_processing:
                return parse_result(events.final_text)
            if self._clock() >= deadline:
                return None
            self._sleep(self.config.poll_interval_sec)


def run(task: TaskPack) -> ResultPack:
    return CodeAttributionModule().run(task)
