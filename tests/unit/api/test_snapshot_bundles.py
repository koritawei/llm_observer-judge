"""Accept-time snapshot bundle prefetch: skip the attribution stage when attribution is off."""

from typing import Any

import pytest
from benchmark.routes.benchmark import _ensure_snapshot_bundles, _resolve_run_registration
from benchmark.runs.schema import ExternalBenchmarkRequest
from benchmark.customization import bundle_store as bundle_store_module


class _RecordingStore:
    def __init__(self) -> None:
        self.ensured: list[dict[str, Any]] = []

    async def ensure(self, ref: dict[str, Any]) -> None:
        self.ensured.append(ref)


def _snapshot(**overrides: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "customization": {
            "judge": {"git_path": "p/judge", "git_commit": "c1"},
            "attribution": {"git_path": "p/attribution", "git_commit": "c2"},
        },
    }
    snapshot.update(overrides)
    return snapshot


@pytest.fixture
def _store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    store = _RecordingStore()
    monkeypatch.setattr(bundle_store_module, "get_bundle_store", lambda: store)
    return store


async def test_bundles_ensured_for_both_stages_by_default(_store: _RecordingStore) -> None:
    await _ensure_snapshot_bundles(_snapshot())
    assert [ref["git_path"] for ref in _store.ensured] == ["p/judge", "p/attribution"]


async def test_attribution_bundle_skipped_when_disabled(_store: _RecordingStore) -> None:
    # Config exists (the snapshot carries attribution customization) but settings do not enable attribution: pull judge only
    await _ensure_snapshot_bundles(_snapshot(attribution_enabled=False))
    assert [ref["git_path"] for ref in _store.ensured] == ["p/judge"]


async def test_run_registration_forces_no_attribution_when_disabled(
    _store: _RecordingStore,
) -> None:
    # The snapshot carries attribution customization, but the registration turns attribution off: the payload is forced to no_attribution,
    # and the attribution bundle is not prefetched (config being present still skips execution)
    payload = ExternalBenchmarkRequest(benchmark_name="example_benchmark")
    snapshot, resolved = await _resolve_run_registration(
        payload, _snapshot(attribution_enabled=False)
    )
    assert snapshot is not None
    assert resolved.attribution_class == "no_attribution"
    assert [ref["git_path"] for ref in _store.ensured] == ["p/judge"]
