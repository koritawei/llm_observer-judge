"""Realtime-stream instance identity: isolation of several datasets under the same topic/group, and the sampling key."""

from benchmark.domain.eval_config import (
    EvalConfigEntry,
    evaluator_sample_key,
    evaluator_slot_label,
    parse_eval_identity_key,
    parse_registration_eval_key,
    persist_evaluator_name,
    run_matches_benchmark_name,
    sample_evaluator_names_for_report,
)
from benchmark.domain.trace_report_config import (
    registered_dataset_name_from_report_name,
    versioned_trace_report_name,
)


def test_sample_keys_differ_when_only_stream_differs() -> None:
    a = EvalConfigEntry(
        evaluatorName="example",
        sampling=0.1,
        streamInstanceId="ds-rt-1",
    )
    b = EvalConfigEntry(
        evaluatorName="example",
        sampling=0.01,
        streamInstanceId="ds-rt-2",
    )
    assert evaluator_sample_key("t1", a) != evaluator_sample_key("t1", b)
    assert evaluator_slot_label(a) == "example@stream:ds-rt-1"
    assert evaluator_slot_label(b) == "example@stream:ds-rt-2"


def test_persist_evaluator_name_includes_stream() -> None:
    assert (
        persist_evaluator_name(
            "example",
            {"agent_registration_id": "reg-v56", "stream_instance_id": "ds-rt-9"},
        )
        == "example@reg-v56@ds-rt-9"
    )


def test_trace_report_name_appends_ts_stream() -> None:
    name = versioned_trace_report_name("amap-sampling-2026-10-27", "", "ds-rt-1730000000123")
    assert name == "amap-sampling-2026-10-27__ts__ds-rt-1730000000123"
    with_reg = versioned_trace_report_name("ds", "reg-v56", "ds-rt-1")
    assert with_reg == "ds__reg__reg-v56__ts__ds-rt-1"
    assert registered_dataset_name_from_report_name(with_reg) == "ds"
    assert registered_dataset_name_from_report_name(name) == "amap-sampling-2026-10-27"


def test_parse_eval_identity_strips_stream_from_registration() -> None:
    assert parse_eval_identity_key("example@reg-v56@ds-rt-1") == (
        "example",
        "reg-v56",
        "ds-rt-1",
    )
    assert parse_registration_eval_key("example@reg-v56@ds-rt-1") == ("example", "reg-v56")
    assert parse_eval_identity_key("example@stream:ds-rt-9") == (
        "example",
        None,
        "ds-rt-9",
    )
    assert parse_eval_identity_key("example@example--biz--v56") == (
        "example",
        "example--biz--v56",
        None,
    )


def test_run_matches_same_registration_across_streams() -> None:
    # A query with a bare registration / no stream: hits across streams (the proxy card's total)
    assert run_matches_benchmark_name(
        column_name="example@reg-v56@ds-rt-1",
        config={"agent_registration_id": "reg-v56"},
        query_name="reg-v56",
    )
    assert run_matches_benchmark_name(
        column_name="example@reg-v56@ds-rt-1",
        config={},
        query_name="example@reg-v56",
    )
    # Both sides carry a stream and they differ: no hit (exact lookup by stream)
    assert not run_matches_benchmark_name(
        column_name="example@reg-v56@ds-rt-1",
        config={},
        query_name="example@reg-v56@ds-rt-2",
    )
    assert not run_matches_benchmark_name(
        column_name="example@reg-v56@ds-rt-1",
        config={},
        query_name="example@reg-v57@ds-rt-1",
    )


def test_sample_evaluator_names_include_stream_slot() -> None:
    keys = sample_evaluator_names_for_report(
        rubric_refs=["example"],
        registration_id="reg-v56",
        stream_instance_id="ds-rt-1",
    )
    assert keys == ["example@reg-v56@ds-rt-1"]
