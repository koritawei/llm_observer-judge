"""Profile Loader contract tests: closed-set validation, config_digest, and allowlist coverage."""

from pathlib import Path

import pytest
from benchmark.pipeline.checker_registry import (
    CheckerRegistry,
    ComponentDescriptor,
)
from benchmark.pipeline.evidence import EvidenceProviderRegistry
from benchmark.pipeline.profiles import (
    compute_config_digest,
    load_profile,
)

_CHECKER_YAML = """
profile_id: sample.business
profile_version: v1
plugin: sample
catalog_version: d0-d6.v1

extension:
  extension_type: sample_eval
  required: required

enabled_dimensions: [D4, D2]

dimensions:
  D4:
    evidence_providers:
      - sample_evidence
    checkers:
      - sample_checker
  D2:
    checkers:
      - second_checker
"""


def _registries() -> tuple[CheckerRegistry, EvidenceProviderRegistry]:
    checkers = CheckerRegistry()
    checkers.register(
        "sample_checker",
        ComponentDescriptor(name="sample_checker", dimensions=("D4",)),
        lambda spec: object(),
    )
    checkers.register(
        "second_checker",
        ComponentDescriptor(name="second_checker", dimensions=("D2",)),
        lambda spec: object(),
    )
    evidence = EvidenceProviderRegistry()
    evidence.register(
        "sample_evidence",
        ComponentDescriptor(name="sample_evidence", component_kind="evidence_provider"),
        lambda spec: object(),
    )
    return checkers, evidence


def _write_profile(tmp_path: Path, content: str = _CHECKER_YAML) -> Path:
    path = tmp_path / "profile.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_profile_accepts_valid_yaml(tmp_path: Path) -> None:
    checker_registry, evidence_registry = _registries()
    profile = load_profile(
        _write_profile(tmp_path),
        checker_registry=checker_registry,
        evidence_registry=evidence_registry,
    )
    assert profile.profile_id == "sample.business"
    assert profile.extension.extension_type == "sample_eval"
    assert profile.extension.required == "required"


def test_config_digest_is_stable_and_sensitive(tmp_path: Path) -> None:
    checker_registry, evidence_registry = _registries()
    path = _write_profile(tmp_path)
    first = load_profile(
        path, checker_registry=checker_registry, evidence_registry=evidence_registry
    )
    second = load_profile(
        path, checker_registry=checker_registry, evidence_registry=evidence_registry
    )
    assert compute_config_digest(first) == compute_config_digest(second)
    assert len(compute_config_digest(first)) == 64

    changed = load_profile(
        _write_profile(
            tmp_path, _CHECKER_YAML.replace("profile_version: v1", "profile_version: v2")
        ),
        checker_registry=checker_registry,
        evidence_registry=evidence_registry,
    )
    assert compute_config_digest(changed) != compute_config_digest(first)


def test_unknown_checker_type_fails_loading(tmp_path: Path) -> None:
    checker_registry, evidence_registry = _registries()
    path = _write_profile(tmp_path, _CHECKER_YAML.replace("sample_checker", "ghost_checker"))
    with pytest.raises(ValueError, match="unknown checker type"):
        load_profile(
            path,
            checker_registry=checker_registry,
            evidence_registry=evidence_registry,
        )


def test_duplicate_checker_instance_fails_loading(tmp_path: Path) -> None:
    checker_registry, evidence_registry = _registries()
    duplicated = _CHECKER_YAML.replace("- second_checker", "- sample_checker")
    with pytest.raises(ValueError, match="duplicate checker instance"):
        load_profile(
            _write_profile(tmp_path, duplicated),
            checker_registry=checker_registry,
            evidence_registry=evidence_registry,
        )


def test_checker_dimension_outside_profile_fails_loading(tmp_path: Path) -> None:
    checker_registry, evidence_registry = _registries()
    checker_registry.register(
        "wide_checker",
        ComponentDescriptor(name="wide_checker", dimensions=("D9",)),
        lambda spec: object(),
    )
    broken = _CHECKER_YAML.replace("- second_checker", "- wide_checker")
    with pytest.raises(ValueError, match="dimensions outside profile"):
        load_profile(
            _write_profile(tmp_path, broken),
            checker_registry=checker_registry,
            evidence_registry=evidence_registry,
        )


def test_binding_dimension_outside_enabled_fails_loading(tmp_path: Path) -> None:
    checker_registry, evidence_registry = _registries()
    broken = _CHECKER_YAML.replace("D2:", "D6:")
    with pytest.raises(ValueError, match="outside enabled_dimensions"):
        load_profile(
            _write_profile(tmp_path, broken),
            checker_registry=checker_registry,
            evidence_registry=evidence_registry,
        )


def test_extension_binding_is_required(tmp_path: Path) -> None:
    checker_registry, evidence_registry = _registries()
    broken = _CHECKER_YAML.replace(
        "extension:\n  extension_type: sample_eval\n  required: required\n", ""
    )
    with pytest.raises(ValueError):
        load_profile(
            _write_profile(tmp_path, broken),
            checker_registry=checker_registry,
            evidence_registry=evidence_registry,
        )


def test_runtime_override_outside_allowlist_is_rejected(tmp_path: Path) -> None:
    checker_registry, evidence_registry = _registries()
    with pytest.raises(ValueError, match="not allowlisted"):
        load_profile(
            _write_profile(tmp_path),
            checker_registry=checker_registry,
            evidence_registry=evidence_registry,
            runtime_overrides={"dimensions": {"D4": {}}},
        )


def test_component_enabled_override_disables_checker(tmp_path: Path) -> None:
    from benchmark.pipeline.profiles import assemble_checkers

    checker_registry, evidence_registry = _registries()
    profile = load_profile(
        _write_profile(tmp_path),
        checker_registry=checker_registry,
        evidence_registry=evidence_registry,
        runtime_overrides={"component_enabled": {"second_checker": False}},
    )
    assembled = assemble_checkers(profile, checker_registry)
    assert len(assembled) == 1


def test_evidence_registry_rejects_non_provider_descriptor() -> None:
    registry = EvidenceProviderRegistry()
    with pytest.raises(ValueError, match="component_kind"):
        registry.register(
            "mislabelled",
            ComponentDescriptor(name="mislabelled"),
            lambda spec: object(),
        )


def test_missing_profile_points_at_runfiles_declaration(tmp_path: Path) -> None:
    """The error for a missing Profile must point at the BUILD data declaration; that is the only cause in the deployed state."""

    checker_registry, evidence_registry = _registries()
    with pytest.raises(FileNotFoundError, match="runfiles"):
        load_profile(
            tmp_path / "profile.yaml",
            checker_registry=checker_registry,
            evidence_registry=evidence_registry,
        )
