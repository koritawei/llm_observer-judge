"""Profile assembly equivalence tests: parsed instance types and order match the historical hand-written tuple item by item.

This is the zero-regression guard for CR-A: the class sequence of Vehicle's 5 Checkers + 5 Extractors stays the same;
VUI assembles five Checkers: rejection, navigation parameters, place constraints, output shape, and completion.
"""

DEFAULT_CHECKERS = None
CmdChecker = None
HistoryReferenceDetector = None
OutputShapeChecker = None
PoiIdChecker = None
SchemaChecker = None
EVIDENCE_EXTRACTORS = None
EvidenceBoundaryExtractor = None
FactualEvidenceExtractor = None
OptimalityExtractor = None
PoiStatusExtractor = None
RouteConstraintExtractor = None
load_vehicle_profile = None
vehicle_checker_registry = None
vehicle_evidence_registry = None
CompletionChecker = None
LocalityChecker = None
NavigationCommandChecker = None
VuiOutputShapeChecker = None
RejectionChecker = None
vui_default_checkers = None
vui_default_evidence_providers = None
load_vui_profile = None

#: Historical assembly order before Profile was wired in (golden baseline).
LEGACY_VEHICLE_CHECKERS = (
    SchemaChecker,
    OutputShapeChecker,
    HistoryReferenceDetector,
    CmdChecker,
    PoiIdChecker,
)
LEGACY_VEHICLE_EXTRACTORS = (
    EvidenceBoundaryExtractor,
    FactualEvidenceExtractor,
    PoiStatusExtractor,
    RouteConstraintExtractor,
    OptimalityExtractor,
)
LEGACY_VUI_CHECKERS = (
    RejectionChecker,
    NavigationCommandChecker,
    LocalityChecker,
    VuiOutputShapeChecker,
    CompletionChecker,
)


def test_vehicle_checker_assembly_matches_legacy_tuple() -> None:
    assert tuple(type(checker) for checker in DEFAULT_CHECKERS) == LEGACY_VEHICLE_CHECKERS


def test_vehicle_extractor_assembly_matches_legacy_tuple() -> None:
    assert tuple(type(provider) for provider in EVIDENCE_EXTRACTORS) == LEGACY_VEHICLE_EXTRACTORS


def test_vehicle_instances_carry_descriptor_identity() -> None:
    """Instance name/dimensions match the Descriptor single source of truth, so the two sources cannot fork."""

    for checker in DEFAULT_CHECKERS:
        descriptor = vehicle_checker_registry.descriptor(checker.name)
        assert descriptor.component_kind == "checker"
        assert descriptor.name == checker.name
        assert descriptor.dimensions == checker.dimensions
    for provider in EVIDENCE_EXTRACTORS:
        descriptor = vehicle_evidence_registry.descriptor(provider.name)
        assert descriptor.component_kind == "evidence_provider"
        assert descriptor.name == provider.name


def test_vehicle_profile_identity_fields() -> None:
    profile = load_vehicle_profile()
    assert profile.profile_id == "vehicle_agent.navigation"
    assert profile.plugin == "vehicle_agent"
    assert profile.extension.extension_type == "vehicle_eval"
    assert profile.extension.required == "required"


def test_vui_checker_assembly_matches_legacy_tuple() -> None:
    assert tuple(type(checker) for checker in vui_default_checkers()) == LEGACY_VUI_CHECKERS


def test_vui_instances_carry_descriptor_identity() -> None:
    vui_checker_registry = None

    for checker in vui_default_checkers():
        descriptor = vui_checker_registry.descriptor(checker.name)
        assert descriptor.component_kind == "checker"
        assert descriptor.name == checker.name
        assert descriptor.dimensions == checker.dimensions


def test_vui_profile_identity_fields() -> None:
    profile = load_vui_profile()
    assert profile.profile_id == "vui_agent.standard"
    assert profile.plugin == "vui_agent"
    assert profile.extension.extension_type == "vui_eval"
    assert profile.profile_version == "v2"
    assert profile.enabled_dimensions == ("D0", "D1", "D2", "D3", "D4", "D5", "D6")
    assert profile.metrics["first_markdown_latency_ms"].threshold == 3000
    assert profile.metrics["last_markdown_latency_ms"].threshold is None


def test_vui_evidence_providers_are_empty_today() -> None:
    assert vui_default_evidence_providers() == ()
