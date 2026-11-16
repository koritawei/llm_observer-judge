"""Host for notification publish plugins.

Resolution order matches the other eval plugins: RemoteConfig
``eval.plugins.notification_providers`` → in-process registry → default publisher.
``class_ref`` points at a zero-arg constructible business class that implements ``create_task``;
``mode=remote`` POSTs ``action=create_task``.
"""

from collections.abc import Mapping
from typing import Any, Protocol

from benchmark.attribution.badcase.schemas import NotificationDispatchResult
from benchmark.domain.plugin_binding import PluginBinding
from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.plugins.host import PluginResolution, resolve_plugin
from benchmark.plugins.remote_client import RemotePluginClient, RemotePluginError

logger = get_logger(__name__)


class NotificationPublisher(Protocol):
    """Create one notification. Arguments match the default publisher."""

    async def create_task(
        self,
        *,
        notification_config: object,
        attribution_result: object,
        resolved_route: object = None,
    ) -> NotificationDispatchResult: ...


def _plain(value: object) -> object:
    if value is None:
        return None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return {key: getattr(value, key) for key in value.__dataclass_fields__}
    return value


class LocalNotificationPublisher:
    """Default publisher: uses this process's work-item publish implementation."""

    async def create_task(
        self,
        *,
        notification_config: object,
        attribution_result: object,
        resolved_route: object = None,
    ) -> NotificationDispatchResult:
        from benchmark.attribution.badcase.publisher import create_notification_task

        return await create_notification_task(
            notification_config=notification_config,
            attribution_result=attribution_result,
            resolved_route=resolved_route,
        )


class RemoteNotificationPublisher:
    """Hand the create-work-item request to the business's own HTTP platform."""

    def __init__(
        self,
        binding: PluginBinding,
        *,
        fallback: NotificationPublisher,
        client: RemotePluginClient | None = None,
    ) -> None:
        self._binding = binding
        self._fallback = fallback
        self._client = client or RemotePluginClient()

    async def create_task(
        self,
        *,
        notification_config: object,
        attribution_result: object,
        resolved_route: object = None,
    ) -> NotificationDispatchResult:
        payload = {
            "config": _plain(notification_config),
            "attribution": _plain(attribution_result),
            "route": _plain(resolved_route),
        }
        try:
            data = await self._client.invoke(
                binding=self._binding,
                kind="notification_provider",
                action="create_task",
                payload=payload,
            )
        except RemotePluginError:
            logger.warning(
                "notification.remote_failed name=%s fallback=%s",
                self._binding.name,
                self._binding.fallback_local,
            )
            if self._binding.fallback_local:
                return await self._fallback.create_task(
                    notification_config=notification_config,
                    attribution_result=attribution_result,
                    resolved_route=resolved_route,
                )
            raise
        body: Mapping[str, Any] | Any = data.get("result", data)
        if not isinstance(body, Mapping):
            body = {"status": "failed", "message": "remote notification response is not an object"}
        return NotificationDispatchResult.model_validate(body)


_REGISTRY: dict[str, NotificationPublisher] = {}
_DEFAULT = LocalNotificationPublisher()


def register_notification_publisher(name: str, publisher: NotificationPublisher) -> None:
    """Register a business publisher in-process. The same name overwrites."""
    _REGISTRY[name] = publisher


def resolve_notification_publisher(name: str | None) -> NotificationPublisher:
    """Resolve a publisher by business name. With no binding and no registration, use the default publisher."""
    return resolve_plugin(
        "notification_providers",
        name or "default",
        resolution=PluginResolution(
            registry=_REGISTRY,
            default=_DEFAULT,
            remote_factory=lambda binding: RemoteNotificationPublisher(binding, fallback=_DEFAULT),
        ),
    )


__all__ = [
    "LocalNotificationPublisher",
    "NotificationPublisher",
    "RemoteNotificationPublisher",
    "register_notification_publisher",
    "resolve_notification_publisher",
]
