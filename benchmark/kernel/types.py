"""Shared outcome types. Plugins may return only these values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Completed:
    result: dict[str, Any]


@dataclass(frozen=True)
class SubmitRequest:
    request: dict[str, Any]


@dataclass(frozen=True)
class TerminalFailure:
    error: str


@dataclass(frozen=True)
class Retryable:
    error: str


@dataclass(frozen=True)
class InvalidateAndReset:
    reason: str


@dataclass(frozen=True)
class ProviderSyncCompleted:
    payload: dict[str, Any]


@dataclass(frozen=True)
class ProviderAcceptedPendingRecorded:
    external_job_id: str


@dataclass(frozen=True)
class ProviderAcceptedPendingFailed:
    error: str


@dataclass(frozen=True)
class ProviderResultUnknown:
    error: str


@dataclass(frozen=True)
class ProviderSubmitRejected:
    error: str


OUTCOMES = (
    Completed,
    SubmitRequest,
    TerminalFailure,
    Retryable,
    InvalidateAndReset,
)

PROVIDER_RESULTS = (
    ProviderSyncCompleted,
    ProviderAcceptedPendingRecorded,
    ProviderAcceptedPendingFailed,
    ProviderResultUnknown,
    ProviderSubmitRejected,
)


@dataclass
class PhaseContext:
    run_id: str
    item_id: str
    phase: dict[str, Any]
    case: dict[str, Any]
    snapshot: dict[str, Any]
    request: dict[str, Any] | None = None
    credential_ready: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)


class PermanentError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class RetryableError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code
