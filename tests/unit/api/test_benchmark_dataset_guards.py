"""benchmark / dataset processing-switch tests (RemoteConfig ``agent-eval.service.switches``).

Two independent switches: ``benchmark_disabled`` / ``dataset_disabled``, both defaulting to
``True`` (do not process by default). The legacy merged field ``benchmark_dataset_disabled``
is no longer read (incompatible).

Covers the RemoteConfig priority chain (RemoteConfig field > env-var fallback > default True),
lenient bool parsing, independence of the two switches, and the API guard (503).
"""

from collections.abc import Iterator

import pytest
from benchmark.routes._guards import raise_if_benchmark_disabled, raise_if_dataset_disabled
from benchmark.settings.remote_config_store import clear_cache
from benchmark.settings.setting import (
    Settings,
    is_benchmark_disabled_runtime,
    is_dataset_disabled_runtime,
)
from benchmark.domain import service_switches
from benchmark.domain.service_switches import is_benchmark_disabled, is_dataset_disabled
from fastapi import HTTPException

_BM_ENV_KEY = "AGENT_EVAL_BENCHMARK_DISABLED"
_DS_ENV_KEY = "AGENT_EVAL_DATASET_DISABLED"


@pytest.fixture(autouse=True)
def _clear_remote_config_cache() -> Iterator[None]:
    """Clear the RemoteConfig cache before and after each case so cases do not pollute each other."""
    clear_cache()
    yield
    clear_cache()


def _mock_remote_config(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    """Stub the RemoteConfig fetch: ``fetch_cached_settings`` returns a fixed payload."""
    monkeypatch.setattr(
        service_switches,
        "fetch_cached_settings",
        lambda *a, **kw: payload,
    )


class TestSettingsField:
    """Settings fields: default True plus env-var alias parsing (env is the RemoteConfig fallback path)."""

    def test_default_disabled_true(self) -> None:
        assert Settings.model_fields["benchmark_disabled"].default is True
        assert Settings.model_fields["dataset_disabled"].default is True

    def test_legacy_field_removed(self) -> None:
        """The legacy merged config field is no longer kept (incompatible)."""
        assert "benchmark_dataset_disabled" not in Settings.model_fields

    def test_remote_config_coords_defaults(self) -> None:
        fields = Settings.model_fields
        assert fields["service_switches_remote_config_data_id"].default == "agent-eval.service.switches"
        assert fields["service_switches_remote_config_group"].default == "agent-eval.service.group"

    def test_env_alias_switches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_BM_ENV_KEY, "false")
        monkeypatch.setenv(_DS_ENV_KEY, "false")
        assert Settings().benchmark_disabled is False
        assert Settings().dataset_disabled is False


class TestBenchmarkSwitchResolution:
    """Benchmark switch fallback chain: RemoteConfig field > environment variable > default True."""

    def test_remote_config_false_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_BM_ENV_KEY, raising=False)
        _mock_remote_config(monkeypatch, {"benchmark_disabled": False})
        assert is_benchmark_disabled() is False

    def test_remote_config_true_overrides_env_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit RemoteConfig true takes priority over env=false (RemoteConfig > environment variable)."""
        monkeypatch.setenv(_BM_ENV_KEY, "false")
        _mock_remote_config(monkeypatch, {"benchmark_disabled": True})
        assert is_benchmark_disabled() is True

    def test_remote_config_missing_field_falls_back_to_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RemoteConfig has no such field → fall back to the environment variable."""
        monkeypatch.setenv(_BM_ENV_KEY, "false")
        _mock_remote_config(monkeypatch, {"judge_service_enabled": True})
        assert is_benchmark_disabled() is False

    def test_remote_config_unavailable_falls_back_to_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RemoteConfig unavailable (empty dict) → fall back to the environment variable."""
        monkeypatch.setenv(_BM_ENV_KEY, "0")
        _mock_remote_config(monkeypatch, {})
        assert is_benchmark_disabled() is False

    def test_all_unset_defaults_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RemoteConfig empty and env unset → True (do not process the benchmark by default)."""
        monkeypatch.delenv(_BM_ENV_KEY, raising=False)
        _mock_remote_config(monkeypatch, {})
        assert is_benchmark_disabled() is True

    def test_remote_config_string_value_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RemoteConfig values accept string spellings ("false"/"0")."""
        monkeypatch.delenv(_BM_ENV_KEY, raising=False)
        _mock_remote_config(monkeypatch, {"benchmark_disabled": "false"})
        assert is_benchmark_disabled() is False

    def test_legacy_remote_config_field_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The legacy merged field benchmark_dataset_disabled is no longer read: when RemoteConfig has only
        the legacy field, treat it as unset → fall back to env → default True."""
        monkeypatch.delenv(_BM_ENV_KEY, raising=False)
        _mock_remote_config(monkeypatch, {"benchmark_dataset_disabled": False})
        assert is_benchmark_disabled() is True


class TestDatasetSwitchResolution:
    """Dataset switch fallback chain: RemoteConfig field > environment variable > default True."""

    def test_remote_config_false_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_DS_ENV_KEY, raising=False)
        _mock_remote_config(monkeypatch, {"dataset_disabled": False})
        assert is_dataset_disabled() is False

    def test_remote_config_true_overrides_env_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(_DS_ENV_KEY, "false")
        _mock_remote_config(monkeypatch, {"dataset_disabled": True})
        assert is_dataset_disabled() is True

    def test_remote_config_missing_field_falls_back_to_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(_DS_ENV_KEY, "false")
        _mock_remote_config(monkeypatch, {"judge_service_enabled": True})
        assert is_dataset_disabled() is False

    def test_all_unset_defaults_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RemoteConfig empty and env unset → True (do not process the dataset by default)."""
        monkeypatch.delenv(_DS_ENV_KEY, raising=False)
        _mock_remote_config(monkeypatch, {})
        assert is_dataset_disabled() is True


class TestSwitchIndependence:
    """The two switches are independent: one field's value does not affect the other's decision."""

    def test_benchmark_open_dataset_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_BM_ENV_KEY, raising=False)
        monkeypatch.delenv(_DS_ENV_KEY, raising=False)
        _mock_remote_config(monkeypatch, {"benchmark_disabled": False})
        assert is_benchmark_disabled() is False
        assert is_dataset_disabled() is True

    def test_dataset_open_benchmark_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_BM_ENV_KEY, raising=False)
        monkeypatch.delenv(_DS_ENV_KEY, raising=False)
        _mock_remote_config(monkeypatch, {"dataset_disabled": False})
        assert is_benchmark_disabled() is True
        assert is_dataset_disabled() is False


class TestRuntimeEnvFallback:
    """Environment-variable fallback (the decision source when RemoteConfig is unavailable): unset defaults to True."""

    @pytest.mark.parametrize("value", ["false", "0", "no", "FALSE", "False"])
    def test_falsy_values_enable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        monkeypatch.setenv(_BM_ENV_KEY, value)
        monkeypatch.setenv(_DS_ENV_KEY, value)
        assert is_benchmark_disabled_runtime() is False
        assert is_dataset_disabled_runtime() is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE"])
    def test_truthy_values_disable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        monkeypatch.setenv(_BM_ENV_KEY, value)
        monkeypatch.setenv(_DS_ENV_KEY, value)
        assert is_benchmark_disabled_runtime() is True
        assert is_dataset_disabled_runtime() is True

    def test_unset_defaults_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_BM_ENV_KEY, raising=False)
        monkeypatch.delenv(_DS_ENV_KEY, raising=False)
        assert is_benchmark_disabled_runtime() is True
        assert is_dataset_disabled_runtime() is True


class TestApiGuard:
    """Create/trigger entry guards: switch on (the default) → 503; explicitly off → allow."""

    def test_benchmark_guard_raises_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_remote_config(monkeypatch, {"benchmark_disabled": True})
        with pytest.raises(HTTPException) as exc_info:
            raise_if_benchmark_disabled()
        assert exc_info.value.status_code == 503
        assert "benchmark_disabled" in str(exc_info.value.detail)

    def test_benchmark_guard_passes_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _mock_remote_config(monkeypatch, {"benchmark_disabled": False})
        raise_if_benchmark_disabled()  # does not raise

    def test_dataset_guard_raises_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mock_remote_config(monkeypatch, {"dataset_disabled": True})
        with pytest.raises(HTTPException) as exc_info:
            raise_if_dataset_disabled()
        assert exc_info.value.status_code == 503
        assert "dataset_disabled" in str(exc_info.value.detail)

    def test_dataset_guard_passes_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _mock_remote_config(monkeypatch, {"dataset_disabled": False})
        raise_if_dataset_disabled()  # does not raise

    def test_guards_independent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When benchmark is allowed through, dataset is still blocked by its own switch (they do not affect each other)."""
        _mock_remote_config(monkeypatch, {"benchmark_disabled": False})
        raise_if_benchmark_disabled()
        with pytest.raises(HTTPException) as exc_info:
            raise_if_dataset_disabled()
        assert exc_info.value.status_code == 503
