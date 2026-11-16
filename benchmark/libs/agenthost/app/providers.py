"""Component registry. Discovery reads classes already imported by the registries."""

from __future__ import annotations

import sys
from typing import Any


class ProviderRegistry:
    _discovered = False
    _scan_namespaces: list[str] = []
    _components: dict[tuple[str, str], type] = {}
    _metadata_adapters: dict[str, Any] = {}

    def __init__(self) -> None:
        self.providers: dict[str, object] = {}

    def register(self, name: str, provider: object) -> None:
        self.providers[name] = provider

    def get(self, name: str) -> object:
        return self.providers[name]

    @classmethod
    def add_scan_namespace(cls, name: str) -> None:
        if name not in cls._scan_namespaces:
            cls._scan_namespaces.append(name)

    @classmethod
    def register_metadata_adapter(cls, kind: str, adapter: Any) -> None:
        cls._metadata_adapters[kind] = adapter

    @classmethod
    def discover(cls) -> None:
        if cls._discovered:
            raise RuntimeError("discover already run")
        cls._discovered = True
        cls._components.clear()
        for module in list(sys.modules.values()):
            if module is None:
                continue
            for obj in vars(module).values():
                if not isinstance(obj, type):
                    continue
                try:
                    metadata = getattr(obj, "component_metadata", None)
                except Exception:
                    continue
                if not isinstance(metadata, dict):
                    continue
                kind = str(metadata.get("kind") or "agent")
                kwargs = metadata.get("kwargs") or {}
                name = kwargs.get("name") or getattr(obj, "name", None)
                if name:
                    cls._components[(kind, str(name))] = obj

    @classmethod
    def has(cls, kind: str, name: str) -> bool:
        return (kind, name) in cls._components

    @classmethod
    def get_component(cls, kind: str, name: str) -> type:
        return cls._components[(kind, name)]

    @classmethod
    def _coerce_to_component_provider(cls, component: type) -> Any:
        metadata = getattr(component, "component_metadata", None) or {}
        kwargs = metadata.get("kwargs") or {}

        class _Coerced:
            kind = str(metadata.get("kind") or "")
            name = str(kwargs.get("name") or getattr(component, "name", ""))

        return _Coerced
