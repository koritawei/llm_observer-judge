"""Unit tests for the ``ExternalBenchmarkRequest`` contract migration window (§9).

Locks: the alias contract is parsed first; the v1 compatibility fields (judge / attribution / rubric_ref /
attribution_provider) fall back to being parsed when the alias is omitted; when both are present the alias wins;
``extra=forbid`` still fails fast on an unknown field; the old profile's native contract
(``BenchmarkRuntimeProfile``) can be normalized by the v2 endpoint.
"""

import logging

import pytest
from benchmark.runs.schema import (
    AttributionProfileConfig,
    BenchmarkRuntimeProfile,
    ExternalBenchmarkRequest,
    JudgeProfileConfig,
)
from benchmark.workers.external_runner import profile_from_external_request
from pydantic import ValidationError

_IEVAL_JUDGE_REF = "business.runs.ieval.external_execution:IevalExternalJudge"
_IEVAL_ATTR_REF = "business.runs.ieval.external_execution:IevalExternalAttribution"
_AMAP_RUBRIC_REF = "business.runs.ieval.amap_chat.rubric:AmapChatRubricV2"
_CUSTOM_RUBRIC_REF = "benchmark.runs.example.services:ExampleRubric"


def _alias_payload(**overrides: object) -> ExternalBenchmarkRequest:
    base: dict[str, object] = {
        "benchmark_name": "ieval_external",
        "judge_class": "ieval_external_judge",
        "attribution_class": "ieval_external_attribution",
        "rubric_class": "amap_chat_rubric_v2",
    }
    base.update(overrides)
    return ExternalBenchmarkRequest(**base)


class TestAliasContract:
    def test_alias_resolves_to_class_refs(self) -> None:
        profile = profile_from_external_request(_alias_payload())
        assert profile.judge.class_ref == _IEVAL_JUDGE_REF
        assert profile.attribution.class_ref == _IEVAL_ATTR_REF
        assert profile.rubric_ref == _AMAP_RUBRIC_REF
        assert profile.component_rubric is False


class TestLegacyFallback:
    def test_legacy_judge_structured_passthrough(self) -> None:
        legacy_judge = JudgeProfileConfig(execution="sync", provider="local")
        profile = profile_from_external_request(
            ExternalBenchmarkRequest(benchmark_name="ieval_external", judge=legacy_judge),
        )
        assert profile.judge.execution == "sync"
        assert profile.judge.provider == "local"
        assert profile.judge.class_ref is None

    def test_legacy_attribution_structured_passthrough(self) -> None:
        legacy_attr = AttributionProfileConfig(execution="sync", provider="local")
        profile = profile_from_external_request(
            ExternalBenchmarkRequest(benchmark_name="ieval_external", attribution=legacy_attr),
        )
        assert profile.attribution.execution == "sync"
        assert profile.attribution.provider == "local"
        assert profile.attribution.class_ref is None

    def test_legacy_rubric_ref_passthrough(self) -> None:
        profile = profile_from_external_request(
            ExternalBenchmarkRequest(
                benchmark_name="ieval_external",
                rubric_ref=_CUSTOM_RUBRIC_REF,
            ),
        )
        assert profile.rubric_ref == _CUSTOM_RUBRIC_REF
        assert profile.component_rubric is False

    def test_attribution_provider_default_mapping(self) -> None:
        # v1 default semantics: no attribution field is passed → map attribution_provider=example
        profile = profile_from_external_request(
            ExternalBenchmarkRequest(benchmark_name="ieval_external"),
        )
        assert profile.attribution.provider == "example"

    def test_attribution_provider_explicit_mapping(self) -> None:
        profile = profile_from_external_request(
            ExternalBenchmarkRequest(
                benchmark_name="ieval_external",
                attribution_provider="external",
            ),
        )
        assert profile.attribution.provider == "external"

    def test_minimal_request_defaults(self) -> None:
        # v1 minimal shape: only benchmark_name → the process default (sync judge / example attribution /
        # the benchmark's default Rubric mapping)
        profile = profile_from_external_request(
            ExternalBenchmarkRequest(benchmark_name="ieval_external"),
        )
        assert profile.judge.execution is None
        assert profile.judge.target is None
        assert profile.judge.class_ref is None
        assert profile.attribution.provider == "example"
        assert profile.rubric_ref is None
        assert profile.component_rubric is False


class TestPrecedenceAndGuards:
    def test_alias_wins_over_legacy(self) -> None:
        # The alias and the old fields are both present → the alias wins (the structured old fields are ignored)
        profile = profile_from_external_request(
            _alias_payload(
                judge=JudgeProfileConfig(execution="sync", provider="local"),
                rubric_ref=_CUSTOM_RUBRIC_REF,
            ),
        )
        assert profile.judge.class_ref == _IEVAL_JUDGE_REF
        assert profile.judge.provider == "external"
        assert profile.rubric_ref == _AMAP_RUBRIC_REF

    def test_extra_forbid_still_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            ExternalBenchmarkRequest(
                benchmark_name="ieval_external",
                judge_class="ieval_external_judge",
                typo_field="nope",  # type: ignore[call-arg]
            )

    def test_legacy_fields_emit_deprecation_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            profile_from_external_request(
                ExternalBenchmarkRequest(
                    benchmark_name="ieval_external",
                    judge=JudgeProfileConfig(execution="sync"),
                    attribution_provider="external",
                ),
            )
        assert any("legacy_contract_fields" in record.message for record in caplog.records)


class TestLegacyV2ProfileContract:
    """The compatibility surface of the old profile's native contract (``BenchmarkRuntimeProfile``)."""

    def test_legacy_profile_shape_rejected_by_strict_model(self) -> None:
        # extra=forbid: an old v2 shape that contains data_date cannot pass the alias-contract model
        with pytest.raises(ValidationError):
            ExternalBenchmarkRequest(
                benchmark_name="ieval_external",
                data_date="20260923",  # type: ignore[call-arg]
            )

    def test_legacy_profile_shape_still_validates_profile_model(self) -> None:
        # The same old v2 shape can still be parsed under the old model (the v2 endpoint normalizes from that)
        profile = BenchmarkRuntimeProfile(
            benchmark_name="ieval_external",
            data_date="20260923",
            judge=JudgeProfileConfig(execution="async", target="https://aone.example.com/t"),
        )
        assert profile.data_date == "20260923"
        assert profile.judge is not None and profile.judge.target == "https://aone.example.com/t"
