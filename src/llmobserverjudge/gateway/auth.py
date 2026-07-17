"""Minimal API key auth (G-06 Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tenant:
    id: str
    api_key: str


class ApiKeyAuthenticator:
    """Maps X-API-Key → tenant. Replace with OIDC in Phase 3."""

    def __init__(self, keys: dict[str, str] | None = None) -> None:
        # api_key -> tenant_id
        self._keys = keys or {"dev-key": "tenant-dev"}

    def authenticate(self, api_key: str | None) -> Tenant:
        if not api_key or api_key not in self._keys:
            raise PermissionError("invalid or missing API key")
        return Tenant(id=self._keys[api_key], api_key=api_key)
