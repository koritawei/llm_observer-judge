"""Pure-function surface of runtime_config_record: kind constants, and building and restoring record keys.

It does not depend on the ORM (no sqlite fixture). Envelope behavior once stored (soft delete / revive /
snapshot) is covered by test_runtime_config_record.py (the runtime store MR).
"""

import pytest
from benchmark.records.models.runtime_config_record import (
    KIND_AGENT_SCOPE,
    KIND_AGENT_VIEW,
    KIND_BENCHMARK_AGENT,
    KIND_BENCHMARK_DATASET,
    KIND_CUSTOMIZATION_SEQUENCE,
    KIND_EVAL_TASK,
    KIND_EVALUATOR,
    KIND_INSTANCE_SEQUENCE,
    KIND_KAFKA_TASK,
    KIND_ODPS_JOB,
    KIND_OSS_UPLOAD_SEQUENCE,
    KIND_SCHEDULED_TASK,
    KIND_TRACE_REPORT,
    SCHEMA_VERSION,
    SCOPE_GLOBAL,
    CustomizationKey,
    agent_scope_key,
    agent_scope_record_key,
    agent_view_record_key,
    benchmark_agent_record_key,
    benchmark_dataset_record_key,
    benchmark_scope_key,
    customization_sequence_record_key,
    eval_task_record_key,
    evaluator_record_key,
    instance_sequence_record_key,
    odps_record_key,
    oss_upload_sequence_record_key,
    project_scope_key,
    scheduled_task_record_key,
    split_evaluator_record_key,
    trace_report_record_key,
)


def test_evaluator_record_key_joins_parts() -> None:
    assert evaluator_record_key("proj", "example", "agent-a") == "evaluator:proj:example:agent-a"


def test_evaluator_record_key_empty_agent_keeps_trailing_segment() -> None:
    assert evaluator_record_key("proj", "example", "") == "evaluator:proj:example:"


def test_split_evaluator_record_key_round_trips() -> None:
    key = evaluator_record_key("proj", "example", "agent-a")
    assert split_evaluator_record_key(key) == ("proj", "example", "agent-a", "")


def test_split_evaluator_record_key_round_trips_with_registration() -> None:
    key = evaluator_record_key("proj", "example", "agent-a", "reg-1")
    assert split_evaluator_record_key(key) == ("proj", "example", "agent-a", "reg-1")


def test_evaluator_record_key_isolates_stream_instance() -> None:
    from benchmark.records.models.runtime_config_record import (
        compose_evaluator_key_tail,
        parse_evaluator_key_tail,
    )

    a = evaluator_record_key(
        "proj", "example", "agent-a", compose_evaluator_key_tail("", "ds-rt-1")
    )
    b = evaluator_record_key(
        "proj", "example", "agent-a", compose_evaluator_key_tail("", "ds-rt-2")
    )
    assert a != b
    assert a.endswith("__si__ds-rt-1")
    assert parse_evaluator_key_tail(split_evaluator_record_key(a)[3]) == ("", "ds-rt-1")
    with_reg = evaluator_record_key(
        "proj",
        "example",
        "agent-a",
        compose_evaluator_key_tail("reg-v56", "ds-rt-1"),
    )
    assert parse_evaluator_key_tail(split_evaluator_record_key(with_reg)[3]) == (
        "reg-v56",
        "ds-rt-1",
    )
    assert compose_evaluator_key_tail("reg-v56", "ds-rt-1") == "reg-v56__si__ds-rt-1"


def test_split_evaluator_record_key_supports_empty_agent() -> None:
    assert split_evaluator_record_key("evaluator:proj:example:") == ("proj", "example", "", "")


def test_split_evaluator_record_key_rejects_other_kinds() -> None:
    # A non-evaluator key with the same number of segments is rejected by the prefix check
    with pytest.raises(ValueError, match="不是评测配置键"):
        split_evaluator_record_key(f"{KIND_ODPS_JOB}:a:b:c")


def test_split_evaluator_record_key_rejects_short_keys() -> None:
    # When there are too few segments, unpacking fails first (also a ValueError, but not the prefix-rejection meaning)
    with pytest.raises(ValueError):
        split_evaluator_record_key(f"{KIND_ODPS_JOB}:job-1")


def test_plain_kind_record_keys() -> None:
    assert odps_record_key("job-1") == f"{KIND_ODPS_JOB}:job-1"
    assert scheduled_task_record_key("task-9") == f"{KIND_SCHEDULED_TASK}:task-9"
    assert benchmark_agent_record_key("agent-1") == f"{KIND_BENCHMARK_AGENT}:agent-1"
    assert benchmark_dataset_record_key("ds-1") == f"{KIND_BENCHMARK_DATASET}:ds-1"
    assert eval_task_record_key("eval-1") == f"{KIND_EVAL_TASK}:eval-1"
    assert agent_scope_record_key("scope-1") == f"{KIND_AGENT_SCOPE}:scope-1"
    assert trace_report_record_key("amap_chat_eval") == f"{KIND_TRACE_REPORT}:amap_chat_eval"
    assert agent_view_record_key("agent-a") == f"{KIND_AGENT_VIEW}:agent-a"


def test_sequence_record_keys() -> None:
    assert oss_upload_sequence_record_key("20261026") == f"{KIND_OSS_UPLOAD_SEQUENCE}:20261026"
    assert instance_sequence_record_key() == f"{KIND_INSTANCE_SEQUENCE}:global"


def test_customization_record_key_and_dataclass_agree() -> None:
    key = CustomizationKey(
        agent_name="agent-a",
        business="report",
        variant_key="default",
        stage="prod",
    )
    expected = f"{KIND_CUSTOMIZATION_SEQUENCE}:agent-a:report:default:prod"
    assert key.record_key() == expected
    assert customization_sequence_record_key(key) == expected


def test_scope_keys_normalize_and_namespace() -> None:
    assert agent_scope_key(" scope-1 ") == "agent:scope-1"
    assert benchmark_scope_key(" ieval_external ") == "benchmark:ieval_external"
    assert project_scope_key(" proj-1 ") == "project:proj-1"


def test_empty_project_scope_is_global() -> None:
    assert project_scope_key("") == SCOPE_GLOBAL
    assert project_scope_key("   ") == SCOPE_GLOBAL


def test_schema_version_is_positive_int() -> None:
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_kind_constants_are_distinct_nonempty_strings() -> None:
    kinds = {
        KIND_EVALUATOR,
        KIND_ODPS_JOB,
        KIND_SCHEDULED_TASK,
        KIND_KAFKA_TASK,
        KIND_BENCHMARK_AGENT,
        KIND_BENCHMARK_DATASET,
        KIND_EVAL_TASK,
        KIND_AGENT_SCOPE,
        KIND_TRACE_REPORT,
        KIND_OSS_UPLOAD_SEQUENCE,
        KIND_CUSTOMIZATION_SEQUENCE,
        KIND_INSTANCE_SEQUENCE,
    }
    assert len(kinds) == len([k for k in kinds if k])
