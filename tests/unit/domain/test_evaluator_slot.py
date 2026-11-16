"""Slot labels and sampling keys for several registrations of the same stem."""

from benchmark.domain.eval_config import (
    EvalConfigEntry,
    evaluator_sample_key,
    evaluator_slot_label,
)


def test_slot_label_uses_registration_id() -> None:
    """When several versions in a group share an internal name, the label distinguishes them only by registrationId."""
    shared_internal = "agx_example_0000000004"
    v56 = EvalConfigEntry(
        evaluatorName="example",
        agentRegistrationId="example--biz--default--v56",
        internalAgentName=shared_internal,
        sampling=1.0,
    )
    v57 = EvalConfigEntry(
        evaluatorName="example",
        agentRegistrationId="example--biz--default--v57",
        internalAgentName=shared_internal,
        sampling=1.0,
    )
    assert evaluator_slot_label(v56) == "example@example--biz--default--v56"
    assert evaluator_slot_label(v57) == "example@example--biz--default--v57"
    assert evaluator_slot_label(v56) != evaluator_slot_label(v57)


def test_slot_label_ignores_internal_falls_back_to_stem() -> None:
    """With no registrationId, fall back to the stem; never use internal_agent_name."""
    with_internal_only = EvalConfigEntry(
        evaluatorName="example",
        internalAgentName="agx_example_0000000001",
        sampling=1.0,
    )
    assert evaluator_slot_label(with_internal_only) == "example"
    bare = EvalConfigEntry(evaluatorName="amap_agent", sampling=1.0)
    assert evaluator_slot_label(bare) == "amap_agent"


def test_slot_from_request_uses_registration_in_metadata() -> None:
    from benchmark.domain.eval_config import evaluator_slot_from_request

    assert (
        evaluator_slot_from_request(
            "example",
            {"agent_registration_id": "example--biz--default--v56"},
        )
        == "example@example--biz--default--v56"
    )
    assert evaluator_slot_from_request("amap_agent", {}) == "amap_agent"
    assert evaluator_slot_from_request("example", None) == "example"


def test_persist_evaluator_name_prefers_registration() -> None:
    from benchmark.domain.eval_config import (
        eval_name_to_criterion_ids,
        persist_evaluator_name,
        sample_evaluator_names_for_report,
    )

    assert (
        persist_evaluator_name(
            "example",
            {"agent_registration_id": "example--biz--default--v56"},
        )
        == "example@example--biz--default--v56"
    )
    assert persist_evaluator_name("example", {}) == "example"
    assert sample_evaluator_names_for_report(
        rubric_refs=["example"],
        registration_id="reg-v56",
    ) == ["example@reg-v56", "reg-v56", "example"]
    assert (
        persist_evaluator_name(
            "amap_agent",
            {
                "agent_registration_id": "ba-1793104298670",
                "stream_instance_id": "ds-rt-1793090866482",
            },
        )
        == "amap_agent@ba-1793104298670@ds-rt-1793090866482"
    )
    assert sample_evaluator_names_for_report(
        rubric_refs=["amap_agent"],
        registration_id="ba-1793104298670",
        stream_instance_id="ds-rt-1793090866482",
    ) == ["amap_agent@ba-1793104298670@ds-rt-1793090866482"]
    mapping = eval_name_to_criterion_ids(
        rubric_refs_to_ids={"example": ["c1", "c2"]},
        registration_id="reg-v56",
    )
    assert mapping["example"] == ["c1", "c2"]
    assert mapping["example@reg-v56"] == ["c1", "c2"]
    assert mapping["reg-v56"] == ["c1", "c2"]
    stream_mapping = eval_name_to_criterion_ids(
        rubric_refs_to_ids={"amap_agent": ["amap_agent_overall"]},
        registration_id="ba-1793104298670",
        stream_instance_id="ds-rt-1793090866482",
    )
    assert stream_mapping["amap_agent"] == ["amap_agent_overall"]
    assert stream_mapping["amap_agent@ba-1793104298670@ds-rt-1793090866482"] == [
        "amap_agent_overall"
    ]
    assert stream_mapping["amap_agent@ba-1793104298670"] == ["amap_agent_overall"]
    assert stream_mapping["ba-1793104298670"] == ["amap_agent_overall"]


def test_benchmark_run_id_stem_strips_registration() -> None:
    from benchmark.domain.eval_config import benchmark_run_id_stem

    assert (
        benchmark_run_id_stem("ieval_external@ba-58417e58-c9cb-429e-baba-5367aecaee60")
        == "ieval_external"
    )
    assert (
        benchmark_run_id_stem("search_eval_v2@ba-73117431-04cf-4b47-b8af-d0f6adbee5c9")
        == "search_eval_v2"
    )
    assert benchmark_run_id_stem("search_eval") == "search_eval"
    assert benchmark_run_id_stem("ieval_external") == "ieval_external"
    assert benchmark_run_id_stem("") == ""


def test_stored_benchmark_run_name_prefers_registration() -> None:
    from benchmark.domain.eval_config import (
        logical_benchmark_run_name,
        parse_registration_eval_key,
        run_matches_benchmark_name,
        stored_benchmark_run_name,
    )

    assert (
        stored_benchmark_run_name(
            logical_name="example_benchmark",
            registration_id="example--biz--default--v56",
            evaluator_stem="example",
        )
        == "example@example--biz--default--v56"
    )
    assert (
        stored_benchmark_run_name(logical_name="example_benchmark", registration_id=None)
        == "example_benchmark"
    )
    assert parse_registration_eval_key("example@example--biz--default--v56") == (
        "example",
        "example--biz--default--v56",
    )
    cfg = {
        "benchmark_services": "example_benchmark",
        "agent_registration": {
            "id": "example--biz--default--v56",
            "name": "example_benchmark",
        },
    }
    assert (
        logical_benchmark_run_name(
            column_name="example@example--biz--default--v56",
            config=cfg,
        )
        == "example_benchmark"
    )
    assert run_matches_benchmark_name(
        column_name="example@example--biz--default--v56",
        config=cfg,
        query_name="example_benchmark",
    )
    assert run_matches_benchmark_name(
        column_name="example@example--biz--default--v56",
        config=cfg,
        query_name="example--biz--default--v56",
    )
    assert run_matches_benchmark_name(
        column_name="example@example--biz--default--v56",
        config=cfg,
        query_name="example@example--biz--default--v56",
    )


def test_sample_keys_differ_across_registrations() -> None:
    a = EvalConfigEntry(
        evaluatorName="example",
        agentRegistrationId="reg-v56",
        sampling=1.0,
    )
    b = EvalConfigEntry(
        evaluatorName="example",
        agentRegistrationId="reg-v57",
        sampling=1.0,
    )
    assert evaluator_sample_key("t1", a) != evaluator_sample_key("t1", b)
    assert evaluator_sample_key("t1", a) == "t1:example:reg-v56:_"
