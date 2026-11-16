"""Aone work-item dispatch decision policy — shared policy protocol + default implementation.

Grade scale follows ``services/agent-eval/docs/评测rubric_v2.md`` §2.3
(weighted score on a 0–2 scale): EXCELLENT ≥ 1.8 / GOOD [1.4,1.8) /
ACCEPTABLE [1.0,1.4) / POOR < 1.0 / FAIL (gate override 0).

Usage:

- The shared policy protocol :class:`AoneDispatchPolicy` only defines ``should_dispatch``;
- Each business (service_name) registers its own
  dispatch-policy implementation via :func:`register_dispatch_policy` (same name overwrites; idempotent);
- RemoteConfig ``eval.plugins.dispatch_policies.bindings`` can switch a service_name to
  ``remote`` (HTTP) or ``class_ref`` (a local class without a release), which outranks the registry;
- An unregistered business falls back to :class:`RubricV2DispatchPolicy`
  (dispatch only POOR / FAIL; a missing grade is allowed = fail-open, so missing data does not skip dispatch).

Async call sites use :func:`should_dispatch_aone` (a remote policy uses async HTTP).
"""

from typing import Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.plugins.host import PluginResolution, call_plugin, resolve_plugin
from benchmark.plugins.remote_adapters import RemoteDispatchPolicy

logger = get_logger(__name__)

#: Default dispatch grade set — rubric_v2 creates a work item only for POOR / FAIL
DISPATCH_GRADES_RUBRIC_V2 = frozenset({"POOR", "FAIL"})


@runtime_checkable
class NotificationDispatchPolicy(Protocol):
    """Aone dispatch policy protocol — the business decides whether to allow dispatch from the grade.

    ``grade`` is the grade label precomputed by the business (for example the rubric_v2
    EXCELLENT/GOOD/ACCEPTABLE/POOR/FAIL scale); ``score`` is the raw score,
    only a fallback input (a business policy may decide from a score threshold directly).

    Return-value meaning (two states; callers must not invent a third):

    - ``True`` = allow — passed this layer's grade gate and enters the caller's later dispatch flow;
      **this is not "create the work item now"** (enabled / false-positive blocking and other earlier layers belong to the caller).
    - ``False`` = blocked at this layer — the caller skips dispatch and records ``notification_skip_reason``.
    """

    def should_dispatch(
        self,
        *,
        grade: str | None,
        score: float | None = None,
    ) -> bool: ...


class RubricV2DispatchPolicy:
    """Default dispatch policy — allow only when the rubric_v2 grade is in {POOR, FAIL}.

    ``grade=None`` (missing fused row / score lookup failed) is allowed — fail-open,
    keeping the existing "do not skip dispatch because data is missing" meaning. Note: severity layering
    is currently paused and does not block (see attribution_dispatch.should_dispatch_attribution),
    so None that passes this layer's gate is pushed as a work item; when severity returns it is the backstop for the None case.
    """

    def __init__(
        self,
        dispatch_grades: frozenset[str] = DISPATCH_GRADES_RUBRIC_V2,
    ) -> None:
        self._dispatch_grades = frozenset(dispatch_grades)

    @property
    def dispatch_grades(self) -> frozenset[str]:
        """Grade set that triggers dispatch (for logs / investigation)."""
        return self._dispatch_grades

    def should_dispatch(
        self,
        *,
        grade: str | None,
        score: float | None = None,
    ) -> bool:
        if grade is None:
            return True
        return grade in self._dispatch_grades


_DEFAULT_POLICY = RubricV2DispatchPolicy()

#: Business policy registry — service_name → policy (unregistered names fall back to the default)
_POLICY_REGISTRY: dict[str, NotificationDispatchPolicy] = {}


def get_default_dispatch_policy() -> NotificationDispatchPolicy:
    """Default policy singleton (rubric_v2: dispatch POOR / FAIL)."""
    return _DEFAULT_POLICY


def register_dispatch_policy(
    service_name: str,
    policy: NotificationDispatchPolicy,
) -> None:
    """A business registers its own dispatch policy (same name overwrites; idempotent)."""
    _POLICY_REGISTRY[service_name] = policy
    logger.info(
        "aone_dispatch_policy.registered service=%s policy=%s dispatch_grades=%s",
        service_name,
        type(policy).__name__,
        getattr(policy, "dispatch_grades", None),
    )


def resolve_dispatch_policy(service_name: str | None) -> NotificationDispatchPolicy:
    """Resolve a dispatch policy by business name: RemoteConfig binding → registry → default implementation."""
    return resolve_plugin(
        "dispatch_policies",
        service_name,
        resolution=PluginResolution(
            registry=_POLICY_REGISTRY,
            default=_DEFAULT_POLICY,
            remote_factory=lambda binding: RemoteDispatchPolicy(binding, fallback=_DEFAULT_POLICY),
        ),
    )


async def should_dispatch_notification(
    policy: NotificationDispatchPolicy,
    *,
    grade: str | None,
    score: float | None = None,
) -> bool:
    """Entry for async call sites: a remote policy uses async HTTP; a local policy decides synchronously."""
    return bool(await call_plugin(policy, "should_dispatch", grade=grade, score=score))


def clear_dispatch_policies() -> None:
    """Clear the business registry (the default implementation is unaffected; for tests / config reload)."""
    _POLICY_REGISTRY.clear()


__all__ = [
    "NotificationDispatchPolicy",
    "DISPATCH_GRADES_RUBRIC_V2",
    "RubricV2DispatchPolicy",
    "clear_dispatch_policies",
    "get_default_dispatch_policy",
    "register_dispatch_policy",
    "resolve_dispatch_policy",
    "should_dispatch_notification",
]
