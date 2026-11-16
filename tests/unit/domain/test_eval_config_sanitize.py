"""EvalConfigStore sanitization + evaluators agentName gate helpers."""

from unittest.mock import patch

from benchmark.settings.remote_config_store import clear_cache
from benchmark.domain._cache import clear_parsed_cache
from benchmark.domain.eval_config import (
    EvalConfigEntry,
    EvalConfigStore,
    agent_declared_in_evaluators,
    declared_agent_names,
    eval_name_to_criterion_ids,
    registration_eval_key,
    sanitize_eval_config_raw,
)


def setup_function() -> None:
    clear_parsed_cache()
    clear_cache()


def test_sanitize_drops_non_list_config_keys() -> None:
    raw = {
        "version": "1.0",
        "configs": {
            "_comment_projectId": "do not put comments inside configs",
            "proj-a": [{"evaluatorName": "example", "sampling": 0.1}],
        },
    }
    cleaned = sanitize_eval_config_raw(raw)
    assert "_comment_projectId" not in cleaned["configs"]
    assert "proj-a" in cleaned["configs"]


def test_store_reads_snapshot_not_remote_config() -> None:
    """Evaluation config reads only the DB snapshot, and no longer fetches from RemoteConfig."""
    from benchmark.domain.runtime_config_store import replace_eval_config_file
    from benchmark.domain.schemas import EvalConfigFile

    poisoned = {
        "version": "1.0",
        "configs": {
            "_comment_projectId": "noise",
            "cmpwln0qs04fo6j8eryfdgznb": [
                {
                    "evaluatorName": "amap_agent",
                    "sampling": 0.0005,
                    "agentName": "amap-agent",
                    "environment": ["prod"],
                }
            ],
        },
    }
    replace_eval_config_file(EvalConfigFile.model_validate(sanitize_eval_config_raw(poisoned)))
    try:
        with patch("benchmark.settings.remote_config_store.fetch_cached_settings") as fetch:
            store = EvalConfigStore(data_id="x", group="y", fallback_file="")
            configs = store.get_configs("cmpwln0qs04fo6j8eryfdgznb")
        fetch.assert_not_called()
    finally:
        replace_eval_config_file(EvalConfigFile())
    assert len(configs) == 1
    assert configs[0].evaluator_name == "amap_agent"
    assert configs[0].agent_name == "amap-agent"


def test_agent_declared_in_evaluators_gate() -> None:
    entries = [
        EvalConfigEntry.model_validate(
            {
                "evaluatorName": "amap_agent",
                "sampling": 0.1,
                "agentName": "amap-agent",
            }
        )
    ]
    # The declared set keeps the registration's original value (S3' missing-name injection writes amap-agent verbatim); when comparing, both sides
    # are normalized: the page scope's amap-agent and the SDK-reported amap_agent match each other.
    assert declared_agent_names(entries) == {"amap-agent"}
    assert agent_declared_in_evaluators("amap-agent", entries) is True
    assert agent_declared_in_evaluators("amap_agent", entries) is True
    assert agent_declared_in_evaluators("search_ai_card", entries) is False
    # When agentName is not declared, the gate is closed
    open_entries = [EvalConfigEntry.model_validate({"evaluatorName": "example", "sampling": 0.1})]
    assert agent_declared_in_evaluators("search_ai_card", open_entries) is True


def test_eval_name_to_criterion_ids_compound_key_keeps_per_stem_ids() -> None:
    """With several evaluator_refs, a composite key maps only that stem's own criterion ids."""
    mapping = eval_name_to_criterion_ids(
        rubric_refs_to_ids={
            "stem_a": ["crit_a1", "crit_a2"],
            "stem_b": ["crit_b1"],
        },
        registration_id="reg-1",
        stream_instance_id="stream-1",
    )
    key_a = registration_eval_key("stem_a", "reg-1", "stream-1")
    key_b = registration_eval_key("stem_b", "reg-1", "stream-1")
    assert mapping[key_a] == ["crit_a1", "crit_a2"]
    assert mapping[key_b] == ["crit_b1"]
    # A bare registration_id is still the full union (the caller does not know the stem)
    assert mapping["reg-1"] == ["crit_a1", "crit_a2", "crit_b1"]
