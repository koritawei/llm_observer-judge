"""业务 Profile 模型与 Loader。

Profile 是业务装配清单，不是可编程 DSL：YAML 只引用显式注册表中的插件名，
不能引用模块路径，也不允许任意条件表达式。Loader 在加载阶段校验未知插件、
重复实例、非法维度与策略，并生成 ``config_digest``；任何校验失败都立即抛出，
不带病启动。

合并优先级（首期只落前两层）::

    Catalog defaults
      < Business Profile YAML
      < RemoteConfig allowlisted runtime overrides（本期仅白名单占位）
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from benchmark.pipeline.checker import BaseChecker
from benchmark.pipeline.checker_registry import (
    CheckerRegistry,
    CheckerSpec,
    ComponentDescriptor,
)
from benchmark.pipeline.evidence import EvidenceProviderRegistry

#: RemoteConfig 运行参数覆盖白名单：本期仅允许按组件开关，禁止改类型/维度/参数。
RUNTIME_OVERRIDE_ALLOWLIST = frozenset({"component_enabled"})

ExtensionRequired = Literal["required", "optional"]


class ExtensionBinding(BaseModel):
    """Profile 对业务扩展契约的声明：公共层只校验存在性，不解析 payload。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extension_type: str
    required: ExtensionRequired = "required"


class DimensionBinding(BaseModel):
    """单个维度下挂载的组件引用（YAML 中可为字符串或 ``{type: ...}`` 映射）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_providers: tuple[CheckerSpec, ...] = ()
    checkers: tuple[CheckerSpec, ...] = ()

    @field_validator("evidence_providers", "checkers", mode="before")
    @classmethod
    def _coerce_entries(cls, value: object) -> object:
        if value is None:
            return ()
        if isinstance(value, (str, dict)):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"dimension binding entries must be a list, got {type(value).__name__}"
            )
        return [{"type": item} if isinstance(item, str) else item for item in value]


class MetricBinding(BaseModel):
    """Profile 中的运行指标声明；阈值缺失时仅记录观测值，不判断超限。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collector: str
    source_field: str
    threshold: float | int | None = None
    threshold_ref: str | None = None
    unavailable_policy: Literal["unknown", "skip"] = "unknown"


class ProfileSpec(BaseModel):
    """业务装配清单的结构真源。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    profile_version: str
    plugin: str
    catalog_version: str
    extension: ExtensionBinding
    enabled_dimensions: tuple[str, ...]
    dimensions: dict[str, DimensionBinding] = Field(default_factory=dict)
    metrics: dict[str, MetricBinding] = Field(default_factory=dict)

    @field_validator("profile_id", "profile_version", "plugin", "catalog_version")
    @classmethod
    def _require_value(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("profile identity fields must not be empty")
        return value

    @field_validator("enabled_dimensions")
    @classmethod
    def _require_dimensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("enabled_dimensions must be non-empty and unique")
        return value


def validate_profile(
    profile: ProfileSpec,
    *,
    checker_registry: CheckerRegistry,
    evidence_registry: EvidenceProviderRegistry,
    allowed_failure_modes: set[str] | frozenset[str] | None = None,
    allowed_risk_flags: set[str] | frozenset[str] | None = None,
) -> None:
    """加载期引用校验：未知插件、重复实例、非法维度、缺失声明。"""

    seen_checkers: set[tuple[str, str]] = set()
    seen_providers: set[tuple[str, str]] = set()
    enabled = set(profile.enabled_dimensions)
    for dimension, binding in profile.dimensions.items():
        if dimension not in enabled:
            raise ValueError(
                f"profile {profile.profile_id} binds dimension {dimension} "
                "outside enabled_dimensions"
            )
        for spec in binding.checkers:
            identity = _validate_checker_binding(
                spec,
                checker_registry=checker_registry,
                enabled=enabled,
                seen=seen_checkers,
                allowed_failure_modes=allowed_failure_modes,
                allowed_risk_flags=allowed_risk_flags,
            )
            seen_checkers.add(identity)
        for spec in binding.evidence_providers:
            identity = _validate_provider_binding(
                spec,
                evidence_registry=evidence_registry,
                enabled=enabled,
                seen=seen_providers,
            )
            seen_providers.add(identity)


def _validate_checker_binding(
    spec: CheckerSpec,
    *,
    checker_registry: CheckerRegistry,
    enabled: set[str],
    seen: set[tuple[str, str]],
    allowed_failure_modes: set[str] | frozenset[str] | None,
    allowed_risk_flags: set[str] | frozenset[str] | None,
) -> tuple[str, str]:
    descriptor = checker_registry.descriptor(spec.type)
    if descriptor.component_kind != "checker":
        raise ValueError(f"profile references non-checker component as checker: {spec.type}")
    identity = (spec.name or descriptor.name, spec.version or descriptor.version)
    if identity in seen:
        raise ValueError(f"duplicate checker instance in profile: {identity}")
    unknown = set(descriptor.dimensions) - enabled
    if unknown:
        values = ", ".join(sorted(unknown))
        raise ValueError(f"checker {identity[0]} declares dimensions outside profile: {values}")
    _validate_emissions(
        identity[0],
        descriptor,
        allowed_failure_modes=allowed_failure_modes,
        allowed_risk_flags=allowed_risk_flags,
    )
    return identity


def _validate_provider_binding(
    spec: CheckerSpec,
    *,
    evidence_registry: EvidenceProviderRegistry,
    enabled: set[str],
    seen: set[tuple[str, str]],
) -> tuple[str, str]:
    descriptor = evidence_registry.descriptor(spec.type)
    if descriptor.component_kind != "evidence_provider":
        raise ValueError(
            f"profile references non-evidence component as evidence provider: {spec.type}"
        )
    identity = (spec.name or descriptor.name, spec.version or descriptor.version)
    if identity in seen:
        raise ValueError(f"duplicate evidence provider instance in profile: {identity}")
    unknown = set(descriptor.dimensions) - enabled
    if unknown:
        values = ", ".join(sorted(unknown))
        raise ValueError(
            f"evidence provider {identity[0]} declares dimensions outside profile: {values}"
        )
    return identity


def _validate_emissions(
    name: str,
    descriptor: ComponentDescriptor,
    *,
    allowed_failure_modes: set[str] | frozenset[str] | None,
    allowed_risk_flags: set[str] | frozenset[str] | None,
) -> None:
    if allowed_failure_modes is not None:
        unknown_modes = set(descriptor.emitted_failure_modes) - set(allowed_failure_modes)
        if unknown_modes:
            values = ", ".join(sorted(unknown_modes))
            raise ValueError(f"checker {name} emits unknown failure modes: {values}")
    if allowed_risk_flags is not None:
        unknown_flags = set(descriptor.emitted_risk_flags) - set(allowed_risk_flags)
        if unknown_flags:
            values = ", ".join(sorted(unknown_flags))
            raise ValueError(f"checker {name} emits unknown risk flags: {values}")


def apply_runtime_overrides(profile: ProfileSpec, overrides: dict[str, Any] | None) -> ProfileSpec:
    """RemoteConfig 白名单覆盖：本期仅支持 ``component_enabled`` 组件开关。"""

    if not overrides:
        return profile
    unknown = set(overrides) - RUNTIME_OVERRIDE_ALLOWLIST
    if unknown:
        values = ", ".join(sorted(unknown))
        raise ValueError(f"runtime override keys not allowlisted: {values}")
    toggles = overrides.get("component_enabled") or {}
    if not isinstance(toggles, dict):
        raise ValueError("component_enabled overrides must be a mapping")
    payload = profile.model_dump(mode="json")
    for dimension_binding in payload.get("dimensions", {}).values():
        for key in ("checkers", "evidence_providers"):
            for spec in dimension_binding.get(key, ()):  # type: ignore[union-attr]
                type_name = spec.get("type")
                if type_name in toggles:
                    spec["enabled"] = bool(toggles[type_name])
    return ProfileSpec.model_validate(payload)


def compute_config_digest(profile: ProfileSpec) -> str:
    """对 Profile 结构真源生成稳定摘要，随结果一起写入信封。"""

    canonical = json.dumps(profile.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_profile(
    source: str | Path,
    *,
    checker_registry: CheckerRegistry,
    evidence_registry: EvidenceProviderRegistry,
    runtime_overrides: dict[str, Any] | None = None,
    allowed_failure_modes: set[str] | frozenset[str] | None = None,
    allowed_risk_flags: set[str] | frozenset[str] | None = None,
) -> ProfileSpec:
    """解析 Profile YAML 并完成全部加载期校验。"""

    path = Path(source)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        # 插件用 ``Path(__file__).with_name`` 定位 Profile，源码树永远读得到，
        # 因此缺失几乎只有一种成因：YAML 未被声明进部署构建的 runfiles。
        raise FileNotFoundError(
            f"profile yaml not found: {path}; "
            "in deploy builds ensure it is declared in the service BUILD data (runfiles)"
        ) from exc
    profile = ProfileSpec.model_validate(yaml.safe_load(raw))
    validate_profile(
        profile,
        checker_registry=checker_registry,
        evidence_registry=evidence_registry,
        allowed_failure_modes=allowed_failure_modes,
        allowed_risk_flags=allowed_risk_flags,
    )
    return apply_runtime_overrides(profile, runtime_overrides)


def _ordered_specs(
    profile: ProfileSpec, key: Literal["checkers", "evidence_providers"]
) -> list[CheckerSpec]:
    """按 enabled_dimensions 声明顺序 flatten 组件引用，按 (name, version) 去重保持首个声明位置。"""

    ordered: list[CheckerSpec] = []
    seen: set[tuple[str, str]] = set()
    for dimension in profile.enabled_dimensions:
        binding = profile.dimensions.get(dimension)
        if binding is None:
            continue
        for spec in getattr(binding, key):
            identity = (spec.name or spec.type, spec.version or "")
            if identity in seen:
                continue
            seen.add(identity)
            ordered.append(spec)
    return ordered


def assemble_checkers(
    profile: ProfileSpec, registry: CheckerRegistry
) -> tuple[BaseChecker[Any], ...]:
    """按 Profile 顺序装配 Checker 实例（同 priority 下声明顺序即执行顺序，priority 小者优先）。"""

    return registry.resolve(
        _ordered_specs(profile, "checkers"),
        allowed_dimensions=set(profile.enabled_dimensions),
    )


def assemble_evidence_providers(
    profile: ProfileSpec, registry: EvidenceProviderRegistry
) -> tuple[Any, ...]:
    """按 Profile 顺序装配 Evidence Provider 实例。"""

    return registry.resolve(
        _ordered_specs(profile, "evidence_providers"),
        allowed_dimensions=set(profile.enabled_dimensions),
    )


__all__ = [
    "RUNTIME_OVERRIDE_ALLOWLIST",
    "DimensionBinding",
    "ExtensionBinding",
    "MetricBinding",
    "ProfileSpec",
    "apply_runtime_overrides",
    "assemble_checkers",
    "assemble_evidence_providers",
    "compute_config_digest",
    "load_profile",
    "validate_profile",
]
