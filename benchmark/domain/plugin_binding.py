"""Plugin binding: a shared ``local`` / ``remote`` execution mode.

Every eval extension point uses this model. The default ``mode=local`` loads an in-process
registration or a ``class_ref`` class; ``mode=remote`` calls an external service over HTTP.

Remote contract (shared)::

    POST {endpoint_url}
    Authorization: Bearer <token>   # optional
    Content-Type: application/json

    {
      "plugin_kind": "<kind>",
      "plugin_name": "<registry-or-ref>",
      "action": "<action>",
      "payload": { ... }
    }
"""

from typing import Literal

from pydantic import BaseModel, field_validator

PluginMode = Literal["local", "remote"]

#: All plugin kinds (aligned with RemoteConfig ``eval.plugins`` categories / remote ``plugin_kind``).
PluginKind = Literal[
    "output_provider",
    "parse_interceptor",
    "report",
    "report_payload",
    "attribution_trigger_policy",
    "dispatch_policy",
    "attribution_stats_provider",
    "trace_identity_extractor",
    "item_trace_identity_extractor",
    "fusion_function",
    "notification_provider",
    "trace_backend",
]

_VALID_MODES = frozenset({"local", "remote"})
_DEFAULT_TIMEOUT_MS = 60_000


class PluginBinding(BaseModel):
    """Execution binding for one plugin (local class / process registry, or remote HTTP)."""

    model_config = {"extra": "ignore"}

    name: str
    mode: PluginMode = "local"
    #: Optional when ``mode=local``: load ``module:Class`` directly, no release-time registration.
    class_ref: str | None = None
    endpoint_url: str = ""
    bearer_token: str | None = None
    timeout_ms: int = _DEFAULT_TIMEOUT_MS
    fallback_local: bool = False
    enabled: bool = True
    description: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("class_ref", mode="before")
    @classmethod
    def _strip_class_ref(cls, value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: object) -> str:
        text = str(value or "local").strip().lower()
        if text in ("platform", "inprocess", "in_process"):
            return "local"
        if text in ("external_http", "http", "rpc"):
            return "remote"
        if text not in _VALID_MODES:
            return "local"
        return text

    @field_validator("endpoint_url", mode="before")
    @classmethod
    def _strip_url(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("timeout_ms", mode="before")
    @classmethod
    def _clamp_timeout(cls, value: object) -> int:
        try:
            n = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _DEFAULT_TIMEOUT_MS
        return max(1_000, min(n, 600_000))

    def is_remote_active(self) -> bool:
        """Whether this binding should be invoked over remote HTTP."""
        return self.enabled and self.mode == "remote" and bool(self.endpoint_url)

    def is_local_active(self) -> bool:
        """Whether this binding should be invoked via the local implementation."""
        return self.enabled and self.mode == "local"


__all__ = [
    "PluginBinding",
    "PluginKind",
    "PluginMode",
]
