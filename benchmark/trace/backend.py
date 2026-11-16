"""Trace read plugins.

The default implementation is the HTTP observation backend in this process. A business switches to its own platform via RemoteConfig
``eval.plugins.trace_backends``: a ``class_ref`` takes the same constructor arguments as
the default backend, ``(endpoint, http, *, max_retries, resolve_lookback_days)``;
``mode=remote`` forwards ``resolve_primary_id`` / ``get_trace`` /
``list_observations_for_trace`` / ``get_observation`` by action.
"""

from collections.abc import Callable
from typing import Any

from benchmark.domain.eval_plugin_config import resolve_plugin_binding
from benchmark.domain.plugin_binding import PluginBinding
from benchmark.libs.agenthost.observability.logger import get_logger
from benchmark.plugins.class_loader import PluginClassLoadError, load_plugin_class
from benchmark.plugins.remote_client import RemotePluginClient
from benchmark.trace.schema import TraceDocument, TraceEndpoint

logger = get_logger(__name__)

TraceBackendFactory = Callable[..., object]

_REGISTRY: dict[str, TraceBackendFactory] = {}


def register_trace_backend(name: str, factory: TraceBackendFactory) -> None:
    """Register a trace-backend factory in-process. The same name overwrites."""
    _REGISTRY[name] = factory


class RemoteTraceBackend:
    """Hand trace reads to the business's own HTTP platform."""

    def __init__(
        self,
        binding: PluginBinding,
        endpoint: TraceEndpoint,
        *,
        client: RemotePluginClient | None = None,
    ) -> None:
        self._binding = binding
        self._endpoint = endpoint
        self._client = client or RemotePluginClient()

    async def _call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "name": self._endpoint.name,
            "base_url": self._endpoint.base_url,
            "metadata_key": self._endpoint.metadata_key,
            **payload,
        }
        return await self._client.invoke(
            binding=self._binding,
            kind="trace_backend",
            action=action,
            payload=body,
        )

    async def resolve_primary_id(self, trace_id: str) -> str | None:
        data = await self._call("resolve_primary_id", {"trace_id": trace_id})
        primary = data.get("primary_id")
        return primary if isinstance(primary, str) and primary else None

    async def get_trace(
        self,
        primary_id: str,
        *,
        fields: str | None = "core",
        expand_metadata: str | None = None,
    ) -> TraceDocument | None:
        data = await self._call(
            "get_trace",
            {
                "primary_id": primary_id,
                "fields": fields,
                "expand_metadata": expand_metadata,
            },
        )
        raw = data.get("trace")
        if not isinstance(raw, dict):
            return None
        return TraceDocument.model_validate(raw)

    async def list_observations_for_trace(self, *args: object, **kwargs: object) -> list[dict]:
        data = await self._call(
            "list_observations_for_trace",
            {"args": list(args), "kwargs": kwargs},
        )
        rows = data.get("observations")
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]

    async def get_observation(self, observation_id: str) -> dict | None:
        data = await self._call("get_observation", {"observation_id": observation_id})
        raw = data.get("observation")
        return raw if isinstance(raw, dict) else None


def open_trace_backend(
    endpoint: TraceEndpoint,
    http: object,
    *,
    name: str = "default",
    max_retries: int,
    resolve_lookback_days: int,
) -> object:
    """Open the trace backend that should be used now. With no binding and no registration, use the default HTTP backend."""
    from benchmark.trace.http_backend import TraceHttpBackend

    binding = resolve_plugin_binding("trace_backends", name)
    if binding is not None and binding.enabled and binding.is_remote_active():
        return RemoteTraceBackend(binding, endpoint)
    if binding is not None and binding.enabled and binding.class_ref:
        try:
            backend_cls = load_plugin_class(binding.class_ref, what="trace_backends")
        except PluginClassLoadError:
            logger.warning(
                "trace_backend.class_ref_failed name=%s fallback=%s",
                binding.name,
                binding.fallback_local,
                exc_info=True,
            )
            if not binding.fallback_local:
                raise
        else:
            return backend_cls(
                endpoint,
                http,
                max_retries=max_retries,
                resolve_lookback_days=resolve_lookback_days,
            )
    factory = _REGISTRY.get(name)
    if factory is not None:
        return factory(
            endpoint,
            http,
            max_retries=max_retries,
            resolve_lookback_days=resolve_lookback_days,
        )
    return TraceHttpBackend(
        endpoint,
        http,
        max_retries=max_retries,
        resolve_lookback_days=resolve_lookback_days,
    )


__all__ = [
    "RemoteTraceBackend",
    "open_trace_backend",
    "register_trace_backend",
]
