"""A realtime sampling dataset takes the realtime evaluation flow (manual-runs branch).

When the dataset ``source=realtime``, the accept entry does not create a batch run. It registers the
dataset sampling conditions as a ``kind=evaluator`` config and refreshes the snapshot (the Kafka realtime
chain takes over). Non-realtime / unregistered datasets stay on batch replay, unchanged.
"""

import json
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from benchmark.routes.benchmark import _accept_external_run
from benchmark.domain.benchmark_console import DatasetDocument, TraceSamplingConditions
from benchmark.runs.schema import ExternalBenchmarkRequest
from fastapi import HTTPException
from fastapi.responses import JSONResponse


def _realtime_dataset(
    *,
    status: Literal["parsing", "ready", "sampling", "collecting"] = "ready",
    sampling_ratio: float | None = 0.5,
    ingest: str = "kafka",
    names: list[str] | None = None,
    tags: list[str] | None = None,
    project_id: str | None = None,
    environments: list[str] | None = None,
    **overrides: Any,
) -> DatasetDocument:
    return DatasetDocument(
        id="ds-rt",
        agentScope="amap-agent",
        name="线上实时采样集",
        source="realtime",
        itemsCount=0,
        status=status,
        createdAt="2026-10-19 10:00:00",
        samplingRatio=sampling_ratio,
        projectId=project_id,
        **overrides,
        traceSampling=TraceSamplingConditions(
            ingest=ingest,  # type: ignore[arg-type]
            **({"name": names} if names else {}),
            **({"tags": tags} if tags else {}),
            **({"environment": environments} if environments else {}),
        ),
    )


def _criterion(ref: str) -> Any:
    criterion = MagicMock()
    criterion.evaluator_ref = ref
    return criterion


def _benchmark_services(refs: list[str]) -> Any:
    services = MagicMock()
    meta = services.get_metadata.return_value
    meta.rubric = [_criterion(ref) for ref in refs]
    meta.service_name = "amap_agent"
    return services


@pytest.fixture()
def _stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Every dependency on the branch path is stubbed, and call arguments are recorded for assertions."""
    calls: dict[str, Any] = {}

    async def _find(dataset_name: str) -> DatasetDocument | None:
        calls["dataset_name"] = dataset_name
        return calls.get("document")

    async def _resolve(runtime: Any, name: str) -> Any:
        refs = calls.get("refs") or ["amap_agent"]
        return _benchmark_services(refs)

    async def _upsert(**kwargs: Any) -> MagicMock:
        calls.setdefault("upserts", []).append(kwargs)
        if kwargs.get("kind") == "evaluator":
            calls["upsert"] = kwargs
        row = MagicMock()
        row.record_key = kwargs["record_key"]
        return row

    async def _refresh() -> None:
        calls["refreshed"] = True

    async def _save_dataset(document: Any) -> dict[str, Any]:
        calls["save_dataset"] = document
        return {}

    async def _retire_siblings(
        project_id: str,
        evaluator_name: str,
        keep_agent: str,
        *,
        keep_registration_id: str = "",
    ) -> int:
        calls["retire_siblings"] = {
            "project_id": project_id,
            "evaluator_name": evaluator_name,
            "keep_agent": keep_agent,
            "keep_registration_id": keep_registration_id,
        }
        return 0

    class _NoReportRows:
        async def first(self) -> None:
            return None

    class _FakeRuntimeConfigRecord:
        """Unit-test environment with no Tortoise connection: the report-item existence query is always "no row"."""

        @staticmethod
        def filter(**kwargs: Any) -> _NoReportRows:
            return _NoReportRows()

    async def _store_upsert(**kwargs: Any) -> MagicMock:
        calls.setdefault("upserts", []).append(kwargs)
        row = MagicMock()
        row.record_key = kwargs["record_key"]
        return row

    async def _no_registration(name: str, registration_id: str | None = None) -> None:
        return None

    monkeypatch.setattr("benchmark.routes.benchmark._resolve_benchmark_services", _resolve)
    monkeypatch.setattr("benchmark.routes.benchmark.resolve_agent_registration", _no_registration)
    monkeypatch.setattr("benchmark.routes.benchmark.find_dataset_document_by_name", _find)
    monkeypatch.setattr("benchmark.routes.benchmark.upsert_runtime_config", _upsert)
    monkeypatch.setattr("benchmark.routes.benchmark.refresh_eval_config_snapshot", _refresh)
    monkeypatch.setattr("benchmark.routes.benchmark.save_dataset", _save_dataset)
    monkeypatch.setattr("benchmark.routes.benchmark.retire_sibling_evaluator_entries", _retire_siblings)
    # ensure_default_trace_report lazily imports these two; patch the module attributes so the real
    # registration logic runs in a unit-test environment with no Tortoise connection (assert payload shape, not a bare stub).
    monkeypatch.setattr(
        "benchmark.records.models.runtime_config_record.RuntimeConfigRecord",
        _FakeRuntimeConfigRecord,
    )
    monkeypatch.setattr(
        "benchmark.domain.runtime_config_store.upsert_runtime_config",
        _store_upsert,
    )
    monkeypatch.setattr(
        "benchmark.routes.benchmark.get_agent_config",
        lambda agent_name: MagicMock(projects=["proj-amap"]),
    )
    return calls


async def _run_accept(payload: ExternalBenchmarkRequest) -> Any:
    from benchmark.routes.benchmark import AcceptExternalRunInput

    return await _accept_external_run(
        payload,
        AcceptExternalRunInput(
            benchmark_service=MagicMock(),
            runtime=MagicMock(),
            external_runner=MagicMock(),
        ),
    )


@pytest.mark.asyncio
async def test_realtime_dataset_activates_streaming_eval(
    _stubs: dict[str, Any],
) -> None:
    """realtime dataset: register the evaluator config and turn sampling on; do not create a batch run."""
    _stubs["document"] = _realtime_dataset(names=["chat"], tags=["daily"])
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="线上实时采样集",
    )
    response = await _run_accept(payload)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 202
    body: Any = json.loads(response.body)
    assert body["mode"] == "realtime"
    assert body["evaluator_name"] == "amap_agent"
    assert body["project_id"] == "proj-amap"
    assert body["run_id"] == ""

    upsert = _stubs["upsert"]
    assert upsert["kind"] == "evaluator"
    assert upsert["record_key"] == "evaluator:proj-amap:amap_agent::__si__ds-rt"
    stamped = upsert["payload"]
    assert stamped["evaluatorName"] == "amap_agent"
    assert stamped["sampling"] == 0.5
    assert stamped["name"] == ["chat"]
    assert stamped["tags"] == ["daily"]
    assert stamped["streamInstanceId"] == "ds-rt"
    assert _stubs["refreshed"] is True
    # Auto-register the T+1 report item: the namespace contains __ts__{stream}, and source follows ingest
    report_upsert = next(
        (kw for kw in _stubs["upserts"] if kw["kind"] == "trace_report"),
        None,
    )
    assert report_upsert is not None
    assert report_upsert["record_key"] == (f"trace_report:{_stubs['document'].name}__ts__ds-rt")
    assert report_upsert["payload"]["benchmark_name"] == "amap_agent_benchmark"
    assert report_upsert["payload"]["source"] == "kafka"
    assert report_upsert["payload"]["project_id"] == "proj-amap"
    assert report_upsert["payload"]["stream_instance_id"] == "ds-rt"
    # The dataset document is written back: evaluator_name / stream_instance_id are persisted
    # and ready→sampling
    saved = _stubs["save_dataset"]
    assert saved.evaluator_name == "amap_agent"
    assert saved.stream_instance_id == "ds-rt"
    assert saved.status == "sampling"


@pytest.mark.asyncio
async def test_realtime_dataset_already_sampling_keeps_status(
    _stubs: dict[str, Any],
) -> None:
    """Dataset already sampling: activate idempotently and do not switch status again."""
    _stubs["document"] = _realtime_dataset(status="sampling")
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="线上实时采样集",
    )
    response = await _run_accept(payload)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 202
    # Already sampling: status stays sampling, but evaluatorName is still written back
    saved = _stubs["save_dataset"]
    assert saved.status == "sampling"
    assert saved.evaluator_name == "amap_agent"
    assert _stubs["upsert"]["payload"]["sampling"] == 0.5


@pytest.mark.asyncio
async def test_realtime_dataset_without_ratio_defaults_full_sampling(
    _stubs: dict[str, Any],
) -> None:
    """When no sample ratio is configured, default to full sampling (1.0)."""
    _stubs["document"] = _realtime_dataset(sampling_ratio=None)
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="线上实时采样集",
    )
    response = await _run_accept(payload)
    assert isinstance(response, JSONResponse)
    assert _stubs["upsert"]["payload"]["sampling"] == 1.0


@pytest.mark.asyncio
async def test_realtime_dataset_page_overrides_forwarded(
    _stubs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Page-level overrides are forwarded with registration: environment / agentName / prompt / model."""
    # The env-var stamp path uses empty values, so everything is verified to come from the dataset document
    from benchmark.settings.setting import settings as app_settings

    for field in (
        "evaluator_environment",
        "evaluator_model_name",
        "evaluator_model_provider",
        "evaluator_model_api_key_env",
        "evaluator_model_base_url",
    ):
        monkeypatch.setattr(app_settings, field, "", raising=False)

    _stubs["document"] = _realtime_dataset(
        environments=["pre"],
        agentName="amap-chat",
        prompt="按安全规范打分 ${input}",
        evaluatorModel={"modelName": "page-judge", "apiKeyEnv": "PAGE_KEY"},
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="线上实时采样集",
    )
    response = await _run_accept(payload)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 202
    stamped = _stubs["upsert"]["payload"]
    assert stamped["environment"] == ["pre"]
    assert stamped["agentName"] == "amap-chat"
    assert stamped["prompt"] == "按安全规范打分 ${input}"
    assert stamped["model"]["modelName"] == "page-judge"
    assert stamped["model"]["apiKeyEnv"] == "PAGE_KEY"
    # agentName goes into the third segment of record_key; stream goes into the fourth segment as an __si__ suffix
    assert _stubs["upsert"]["record_key"] == (
        "evaluator:proj-amap:amap_agent:amap-chat:__si__ds-rt"
    )
    # Before registration, clear old agent_name variants in the same stream slot (other streams / regs are kept)
    assert _stubs["retire_siblings"] == {
        "project_id": "proj-amap",
        "evaluator_name": "amap_agent",
        "keep_agent": "amap-chat",
        "keep_registration_id": "__si__ds-rt",
    }


@pytest.mark.asyncio
async def test_realtime_dataset_project_from_document(
    _stubs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dataset document's projectId (configured on the page sampling form) wins over a registry reverse lookup."""
    seen: dict[str, str] = {}

    def _no_registry(agent_name: str) -> None:
        seen["agent"] = agent_name

    monkeypatch.setattr("benchmark.routes.benchmark.get_agent_config", _no_registry)
    _stubs["document"] = _realtime_dataset(project_id="proj-dataset")
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="线上实时采样集",
    )
    response = await _run_accept(payload)
    assert isinstance(response, JSONResponse)
    body: Any = json.loads(response.body)
    assert body["project_id"] == "proj-dataset"
    assert _stubs["upsert"]["record_key"] == ("evaluator:proj-dataset:amap_agent::__si__ds-rt")
    assert "agent" not in seen  # When the dataset already carries projectId, do not query the registry


@pytest.mark.asyncio
async def test_realtime_dataset_explicit_project_id_wins(
    _stubs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit project_id wins over a registry reverse lookup."""
    seen: dict[str, str] = {}
    monkeypatch.setattr(
        "benchmark.routes.benchmark.get_agent_config",
        lambda agent_name: (seen.setdefault("agent", agent_name), MagicMock(projects=["other"]))[1],
    )
    _stubs["document"] = _realtime_dataset()
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="线上实时采样集",
        project_id="proj-explicit",
    )
    response = await _run_accept(payload)
    assert isinstance(response, JSONResponse)
    assert "agent" not in seen  # When project_id is passed explicitly, do not query the registry
    assert _stubs["upsert"]["record_key"] == ("evaluator:proj-explicit:amap_agent::__si__ds-rt")


@pytest.mark.asyncio
async def test_realtime_dataset_project_unresolvable_rejected(
    _stubs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry has no entry and no explicit project was passed: 400 fail-fast."""
    monkeypatch.setattr(
        "benchmark.routes.benchmark.get_agent_config",
        lambda agent_name: None,
    )
    _stubs["document"] = _realtime_dataset()
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="线上实时采样集",
    )
    with pytest.raises(HTTPException) as exc:
        await _run_accept(payload)
    assert exc.value.status_code == 400
    assert "project_id" in str(exc.value.detail)
    assert "upsert" not in _stubs


@pytest.mark.asyncio
async def test_realtime_dataset_multi_evaluator_rejected(
    _stubs: dict[str, Any],
) -> None:
    """A multi-evaluator rubric cannot route realtime evaluation: 400."""
    _stubs["refs"] = ["amap_agent", "example"]
    _stubs["document"] = _realtime_dataset()
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="线上实时采样集",
    )
    with pytest.raises(HTTPException) as exc:
        await _run_accept(payload)
    assert exc.value.status_code == 400
    assert "单一 evaluator" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_non_realtime_dataset_keeps_batch_flow(
    _stubs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-realtime (or unregistered) dataset does not activate the realtime chain and continues through batch-replay acceptance."""
    _stubs["document"] = None

    def _batch_marker(date: str | None) -> str:
        raise RuntimeError("batch-path-reached")

    monkeypatch.setattr(
        "benchmark.routes.benchmark.resolve_external_data_date",
        _batch_marker,
    )
    payload = ExternalBenchmarkRequest(
        benchmark_name="amap_agent_benchmark",
        dataset_name="离线上传集",
    )
    # The branch did not hit realtime (document=None) → continue down the batch branch until
    # date parsing (blocked by the marked function) proves realtime activation did not run.
    with pytest.raises(RuntimeError, match="batch-path-reached"):
        await _run_accept(payload)
    assert "upsert" not in _stubs
