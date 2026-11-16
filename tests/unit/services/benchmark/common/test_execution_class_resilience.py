"""Execution-class resilience regression: degrade when class loading fails on the callback path, and validate the class_ref protocol.

Three contracts:
1. ``normalize_judge_callback`` / ``normalize_attribution_callback`` return the
   payload unchanged when class_ref fails to load (rolling-deploy window: the class
   was removed after the snapshot was frozen), and do not block the callback path;
2. When the hook exists, normalize passes through normally (regression);
3. ``ProfileResolver._validate_class_ref`` rejects a class that does not meet the
   protocol (missing execution/provider declarations, or async missing a required method),
   fail-fast to the same standard as alias resolution.
"""

from types import SimpleNamespace

import pytest
from benchmark.runs.common import execution_classes as ec
from benchmark.runs.common import profile_resolver as pr
from benchmark.runs.common.profile_resolver import ProfileResolutionError, ProfileResolver
from benchmark.plugins.class_loader import PluginClassLoadError


class _FakeCallback(SimpleNamespace):
    """Callback payload test double (normalize only passes it through and does not read fields)."""


def test_normalize_judge_callback_degrades_on_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(ref: str, *, what: str) -> type:
        raise PluginClassLoadError(f"module removed: {ref}")

    monkeypatch.setattr(ec, "load_plugin_class", _boom)
    payload = _FakeCallback(job_id="j1", criterion_scores={"d0": 1})
    assert ec.normalize_judge_callback("benchmark.runs.x:Gone", payload) is payload


def test_normalize_attribution_callback_degrades_on_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(ref: str, *, what: str) -> type:
        raise PluginClassLoadError(f"module removed: {ref}")

    monkeypatch.setattr(ec, "load_plugin_class", _boom)
    payload = _FakeCallback(job_id="a1")
    assert ec.normalize_attribution_callback("benchmark.runs.x:Gone", payload) is payload


def test_normalize_judge_callback_invokes_hook_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WithHook:
        @staticmethod
        def normalize_callback(payload: object) -> object:
            return _FakeCallback(normalized=True)

    monkeypatch.setattr(ec, "load_plugin_class", lambda ref, *, what: _WithHook)
    out = ec.normalize_judge_callback("benchmark.runs.x:WithHook", _FakeCallback())
    assert out.normalized is True


def test_normalize_callback_passthrough_without_class_ref() -> None:
    payload = _FakeCallback(job_id="j1")
    assert ec.normalize_judge_callback(None, payload) is payload
    assert ec.normalize_attribution_callback(None, payload) is payload


class _NoProtocolClass:
    """Non-conforming class with no execution/provider declaration."""


class _AsyncMissingSubmit:
    execution = "async"
    provider = "external"


class _CompliantSyncJudge:
    execution = "sync"
    provider = "local"


def _patch_load(monkeypatch: pytest.MonkeyPatch, cls: type) -> None:
    monkeypatch.setattr(pr, "load_class_ref", lambda ref, *, what: cls)


def test_validate_class_ref_rejects_missing_execution_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, _NoProtocolClass)
    with pytest.raises(ProfileResolutionError, match="execution/provider"):
        ProfileResolver._validate_class_ref(
            "benchmark.runs.x:NoProtocol", what="judge_class"
        )


def test_validate_class_ref_rejects_async_missing_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, _AsyncMissingSubmit)
    with pytest.raises(ProfileResolutionError, match="submit"):
        ProfileResolver._validate_class_ref(
            "benchmark.runs.x:AsyncMiss", what="judge_class"
        )


def test_validate_class_ref_rejects_attribution_missing_query_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AttrAsyncPartial:
        execution = "async"
        provider = "external"

        async def submit(self, req: object) -> str:
            return "job-1"

    _patch_load(monkeypatch, _AttrAsyncPartial)
    with pytest.raises(ProfileResolutionError, match="query_job_status"):
        ProfileResolver._validate_class_ref(
            "benchmark.runs.x:AttrPartial",
            what="attribution_class",
        )


def test_validate_class_ref_accepts_compliant_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, _CompliantSyncJudge)
    ref = "benchmark.runs.x:Compliant"
    assert ProfileResolver._validate_class_ref(ref, what="judge_class") == ref


def test_validate_class_ref_passthrough_none() -> None:
    assert ProfileResolver._validate_class_ref(None, what="judge_class") is None
    assert ProfileResolver._validate_class_ref("  ", what="attribution_class") is None
