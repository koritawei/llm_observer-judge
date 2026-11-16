"""Evidence Provider protocol and explicit registration.

An Evidence Provider only produces neutral ``EvidenceFact`` values (it must not score), separate from a Checker's
deterministic verdict duty. The registry only accepts factories registered explicitly in code. A YAML Profile only refers to
a registration name and cannot refer to a module path.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from benchmark.pipeline.checker_registry import (
    CheckerSpec,
    ComponentDescriptor,
)
from benchmark.pipeline.contracts import EvidenceFact

ContextT = TypeVar("ContextT")


class EvidenceProvider(Generic[ContextT]):
    """Neutral evidence producer: observe only, do not score, do not decide."""

    name: str

    def extract(self, context: ContextT) -> list[EvidenceFact]:
        """Return neutral evidence facts for the given context."""
        raise NotImplementedError


EvidenceProviderFactory = Callable[[CheckerSpec], EvidenceProvider[Any]]


@dataclass(frozen=True)
class _RegisteredProvider:
    descriptor: ComponentDescriptor
    factory: EvidenceProviderFactory


class EvidenceProviderRegistry:
    """Explicitly register Evidence Provider factories and validate references before instantiation."""

    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredProvider] = {}

    def register(
        self,
        type_name: str,
        descriptor: ComponentDescriptor,
        factory: EvidenceProviderFactory,
    ) -> None:
        if not type_name.strip():
            raise ValueError("evidence provider type must not be empty")
        if descriptor.component_kind != "evidence_provider":
            raise ValueError(
                f"evidence provider {type_name} descriptor must declare "
                "component_kind='evidence_provider'"
            )
        if type_name in self._entries:
            raise ValueError(f"evidence provider type already registered: {type_name}")
        self._entries[type_name] = _RegisteredProvider(descriptor, factory)

    def descriptor(self, type_name: str) -> ComponentDescriptor:
        try:
            return self._entries[type_name].descriptor
        except KeyError as exc:
            raise ValueError(f"unknown evidence provider type: {type_name}") from exc

    def resolve(
        self,
        specs: Sequence[CheckerSpec],
        *,
        allowed_dimensions: set[str] | frozenset[str] | None = None,
        available_fields: set[str] | frozenset[str] | None = None,
    ) -> tuple[EvidenceProvider[Any], ...]:
        resolved: list[tuple[int, int, EvidenceProvider[Any]]] = []
        identities: set[tuple[str, str]] = set()
        for index, spec in enumerate(specs):
            entry = self._entries.get(spec.type)
            if entry is None:
                raise ValueError(f"unknown evidence provider type: {spec.type}")
            descriptor = entry.descriptor
            if not spec.enabled or not descriptor.enabled:
                continue
            name = spec.name or descriptor.name
            version = spec.version or descriptor.version
            identity = (name, version)
            if identity in identities:
                raise ValueError(f"duplicate evidence provider identity: {name}@{version}")
            identities.add(identity)
            if allowed_dimensions is not None:
                unknown = set(descriptor.dimensions) - set(allowed_dimensions)
                if unknown:
                    values = ", ".join(sorted(unknown))
                    raise ValueError(
                        f"evidence provider {name} references unknown dimensions: {values}"
                    )
            if available_fields is not None:
                unknown_fields = set(descriptor.required_fields) - set(available_fields)
                if unknown_fields:
                    values = ", ".join(sorted(unknown_fields))
                    raise ValueError(
                        f"evidence provider {name} references unknown required fields: {values}"
                    )
            priority = spec.priority if spec.priority is not None else descriptor.priority
            resolved.append((priority, index, entry.factory(spec)))
        resolved.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in resolved)

    def registered_types(self) -> tuple[str, ...]:
        return tuple(self._entries)


__all__ = [
    "EvidenceProvider",
    "EvidenceProviderFactory",
    "EvidenceProviderRegistry",
]
