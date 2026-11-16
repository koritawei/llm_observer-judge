"""Component-metadata registration and a compatibility adapter for the legacy Checker set."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, overload

from pydantic import BaseModel, ConfigDict, Field, field_validator

from benchmark.pipeline.checker import BaseChecker, run_checkers
from benchmark.pipeline.signals import EvaluationSignal, JsonValue

ContextT = TypeVar("ContextT")
ComponentKind = Literal[
    "evidence_provider",
    "checker",
    "judge",
    "metric_collector",
    "attribution_investigator",
]


class ComponentDescriptor(BaseModel):
    """Stable identity and capability declaration for an execution component."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str = "v1"
    component_kind: ComponentKind = "checker"
    dimensions: tuple[str, ...] = ()
    supported_context: str = ""
    emitted_failure_modes: tuple[str, ...] = ()
    emitted_risk_flags: tuple[str, ...] = ()
    emitted_metrics: tuple[str, ...] = ()
    emitted_diagnostics: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    priority: int = 100
    enabled: bool = True
    parallel_safe: bool = False

    @field_validator("name", "version")
    @classmethod
    def _require_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("component name and version must not be empty")
        return value


CheckerDescriptor = ComponentDescriptor


class CheckerSpec(BaseModel):
    """Checker constructor arguments declared by a business plugin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    name: str | None = None
    version: str | None = None
    priority: int | None = None
    enabled: bool = True
    params: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _require_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("checker type must not be empty")
        return value


CheckerFactory = Callable[[CheckerSpec], BaseChecker[Any]]


@dataclass(frozen=True)
class _RegisteredChecker:
    descriptor: CheckerDescriptor
    factory: CheckerFactory


class CheckerRegistry:
    """Explicitly register Checker factories and validate references before instantiation."""

    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredChecker] = {}

    def register(
        self, type_name: str, descriptor: CheckerDescriptor, factory: CheckerFactory
    ) -> None:
        if not type_name.strip():
            raise ValueError("checker type must not be empty")
        if type_name in self._entries:
            raise ValueError(f"checker type already registered: {type_name}")
        self._entries[type_name] = _RegisteredChecker(descriptor, factory)

    def descriptor(self, type_name: str) -> CheckerDescriptor:
        try:
            return self._entries[type_name].descriptor
        except KeyError as exc:
            raise ValueError(f"unknown checker type: {type_name}") from exc

    def resolve(
        self,
        specs: Sequence[CheckerSpec],
        *,
        allowed_dimensions: set[str] | frozenset[str] | None = None,
        allowed_failure_modes: set[str] | frozenset[str] | None = None,
        allowed_risk_flags: set[str] | frozenset[str] | None = None,
        available_fields: set[str] | frozenset[str] | None = None,
    ) -> tuple[BaseChecker[Any], ...]:
        resolved: list[tuple[int, int, BaseChecker[Any]]] = []
        identities: set[tuple[str, str]] = set()
        for index, spec in enumerate(specs):
            entry = self._entries.get(spec.type)
            if entry is None:
                raise ValueError(f"unknown checker type: {spec.type}")
            descriptor = entry.descriptor
            if not spec.enabled or not descriptor.enabled:
                continue
            name = spec.name or descriptor.name
            version = spec.version or descriptor.version
            identity = (name, version)
            if identity in identities:
                raise ValueError(f"duplicate checker identity: {name}@{version}")
            identities.add(identity)
            self._validate_references(
                descriptor,
                name=name,
                allowed_dimensions=allowed_dimensions,
                allowed_failure_modes=allowed_failure_modes,
                allowed_risk_flags=allowed_risk_flags,
                available_fields=available_fields,
            )
            priority = spec.priority if spec.priority is not None else descriptor.priority
            resolved.append((priority, index, entry.factory(spec)))
        resolved.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in resolved)

    @staticmethod
    def _validate_references(
        descriptor: CheckerDescriptor,
        *,
        name: str,
        allowed_dimensions: set[str] | frozenset[str] | None,
        allowed_failure_modes: set[str] | frozenset[str] | None,
        allowed_risk_flags: set[str] | frozenset[str] | None,
        available_fields: set[str] | frozenset[str] | None,
    ) -> None:
        checks = (
            ("dimensions", descriptor.dimensions, allowed_dimensions),
            ("failure modes", descriptor.emitted_failure_modes, allowed_failure_modes),
            ("risk flags", descriptor.emitted_risk_flags, allowed_risk_flags),
            ("required fields", descriptor.required_fields, available_fields),
        )
        for label, references, allowed in checks:
            if allowed is None:
                continue
            unknown = set(references) - set(allowed)
            if unknown:
                values = ", ".join(sorted(unknown))
                raise ValueError(f"checker {name} references unknown {label}: {values}")

    def registered_types(self) -> tuple[str, ...]:
        return tuple(self._entries)


class LegacyCheckerSetAdapter(Generic[ContextT], Sequence[BaseChecker[ContextT]]):
    """Wrap an existing Checker set behind the unified interface, keeping declaration order and degrade semantics."""

    def __init__(self, checkers: Sequence[BaseChecker[ContextT]]) -> None:
        self._checkers = tuple(checkers)

    @overload
    def __getitem__(self, index: int) -> BaseChecker[ContextT]: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[BaseChecker[ContextT], ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> BaseChecker[ContextT] | tuple[BaseChecker[ContextT], ...]:
        return self._checkers[index]

    def __len__(self) -> int:
        return len(self._checkers)

    def __iter__(self) -> Iterator[BaseChecker[ContextT]]:
        return iter(self._checkers)

    def run(self, context: ContextT) -> list[EvaluationSignal]:
        return run_checkers(context, self._checkers)


__all__ = [
    "CheckerDescriptor",
    "CheckerRegistry",
    "CheckerSpec",
    "ComponentDescriptor",
    "ComponentKind",
    "LegacyCheckerSetAdapter",
]
