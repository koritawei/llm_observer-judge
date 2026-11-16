"""Process-level scheduled-task switch (``AGENT_EVAL_SCHEDULED_TASKS_ENABLED``; default is not to run)."""

import pytest
from benchmark.settings.setting import Settings

_ENV_KEY = "AGENT_EVAL_SCHEDULED_TASKS_ENABLED"


class TestScheduledTasksEnabled:
    """Environment-variable parsing: unset defaults to False; only an explicit true/1/yes allows it."""

    def test_default_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_ENV_KEY, raising=False)
        assert Settings().scheduled_tasks_enabled is False

    def test_field_default(self) -> None:
        assert Settings.model_fields["scheduled_tasks_enabled"].default is False

    @pytest.mark.parametrize("raw", ["true", "1", "yes", "True"])
    def test_env_true_enables(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_ENV_KEY, raw)
        assert Settings().scheduled_tasks_enabled is True

    def test_env_false_stays_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV_KEY, "false")
        assert Settings().scheduled_tasks_enabled is False
