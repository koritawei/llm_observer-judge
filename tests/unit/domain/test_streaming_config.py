"""Unit tests for the RemoteConfig streaming declarative-config domain model (§5.2 / §11)."""

import pytest
from benchmark.domain import streaming_config as sc
from benchmark.domain.streaming_config import (
    StreamingConfig,
    StreamingTaskConfig,
    fetch_streaming_config,
)


@pytest.fixture(autouse=True)
def _reset_last_good():
    """The module-level last-known-good is reset between tests (§11 process-level state)."""
    sc._last_good_cache.clear()
    yield
    sc._last_good_cache.clear()


def _task_payload(**updates: object) -> dict:
    payload: dict[str, object] = {
        "name": "amap_agent_daily",
        "topics": ["langfuse.records"],
        "group_id": None,
        "state": "running",
        "judge_class": None,
        "judge_execution": None,
        "triage_class": None,
        "attribution_class": None,
        "attribution_execution": None,
        "rubric_class": "amap_agent_kafka_rubric",
        "sampling": None,
        "daily_report": {"enabled": True},
    }
    payload.update(updates)
    return payload


class TestStreamingTaskConfig:
    def test_full_fields_parse(self) -> None:
        task = StreamingTaskConfig.model_validate(_task_payload())
        assert task.name == "amap_agent_daily"
        assert task.topics == ["langfuse.records"]
        assert task.group_id is None
        assert task.state == "running"
        assert task.rubric_class == "amap_agent_kafka_rubric"
        assert task.sampling is None
        assert task.daily_report.enabled is True

    def test_amap_kafka_recommended_explicit_judge_class(self) -> None:
        """amap-agent Kafka recommends an explicit three-stage alias plus the sync combination."""
        task = StreamingTaskConfig.model_validate(
            _task_payload(
                judge_class="evaluator_service_judge",
                judge_execution="sync",
                triage_class="amap_agent_triage",
                attribution_class="amap_agent_attribution",
                attribution_execution="sync",
            ),
        )
        assert task.judge_class == "evaluator_service_judge"
        assert task.judge_execution == "sync"
        assert task.triage_class == "amap_agent_triage"
        assert task.attribution_class == "amap_agent_attribution"
        assert task.attribution_execution == "sync"

    def test_blank_strings_normalize_to_none(self) -> None:
        task = StreamingTaskConfig.model_validate(
            _task_payload(group_id="  ", judge_class="", rubric_class=" "),
        )
        assert task.group_id is None
        assert task.judge_class is None
        assert task.rubric_class is None

    def test_state_normalizes_case_and_none(self) -> None:
        assert StreamingTaskConfig.model_validate(_task_payload(state="PAUSED")).state == "paused"
        assert StreamingTaskConfig.model_validate(_task_payload(state=None)).state == "running"

    def test_invalid_sampling_falls_back_to_none(self) -> None:
        task = StreamingTaskConfig.model_validate(_task_payload(sampling="abc"))
        assert task.sampling is None
        task = StreamingTaskConfig.model_validate(_task_payload(sampling=0.5))
        assert task.sampling == 0.5

    def test_non_list_topics_normalize_to_empty(self) -> None:
        task = StreamingTaskConfig.model_validate(_task_payload(topics="langfuse.records"))
        assert task.topics == []

    def test_daily_report_null_falls_back_to_default(self) -> None:
        task = StreamingTaskConfig.model_validate(_task_payload(daily_report=None))
        assert task.daily_report.enabled is False


class TestStreamingConfig:
    def test_top_level_defaults(self) -> None:
        cfg = StreamingConfig.model_validate({})
        assert cfg.enabled is False
        assert cfg.allow_empty_tasks is False
        assert cfg.reconcile_interval_s == 30
        assert cfg.tasks == []

    def test_invalid_reconcile_interval_falls_back(self) -> None:
        cfg = StreamingConfig.model_validate({"reconcile_interval_s": "abc"})
        assert cfg.reconcile_interval_s == 30
        cfg = StreamingConfig.model_validate({"reconcile_interval_s": None})
        assert cfg.reconcile_interval_s == 30


class TestFetchStreamingConfig:
    def test_remote_config_absent_returns_none_without_last_good(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sc, "fetch_cached_settings", lambda *a, **k: None)
        assert fetch_streaming_config(use_cache=False) is None

    def test_parse_success_updates_last_good(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw = {"enabled": True, "tasks": [_task_payload()]}
        monkeypatch.setattr(sc, "fetch_cached_settings", lambda *a, **k: raw)
        cfg = fetch_streaming_config(use_cache=False)
        assert cfg is not None
        assert cfg.enabled is True
        assert len(cfg.tasks) == 1
        assert sc._last_good_cache.get("cfg") is cfg

    def test_parse_failure_keeps_last_good(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        good = {"enabled": True, "tasks": [_task_payload()]}
        monkeypatch.setattr(sc, "fetch_cached_settings", lambda *a, **k: good)
        first = fetch_streaming_config(use_cache=False)
        assert first is not None

        # tasks must be a list — an illegal structure raises ValidationError
        monkeypatch.setattr(
            sc,
            "fetch_cached_settings",
            lambda *a, **k: {"enabled": True, "tasks": "not-a-list"},
        )
        second = fetch_streaming_config(use_cache=False)
        assert second is first

    def test_remote_config_emptied_keeps_last_good(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sc,
            "fetch_cached_settings",
            lambda *a, **k: {"enabled": True, "tasks": [_task_payload()]},
        )
        first = fetch_streaming_config(use_cache=False)
        assert first is not None

        monkeypatch.setattr(sc, "fetch_cached_settings", lambda *a, **k: None)
        assert fetch_streaming_config(use_cache=False) is first

    def test_parse_failure_without_last_good_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            sc,
            "fetch_cached_settings",
            lambda *a, **k: {"enabled": True, "tasks": "not-a-list"},
        )
        assert fetch_streaming_config(use_cache=False) is None
