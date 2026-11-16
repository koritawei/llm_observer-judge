"""OSS storage upload API unit tests."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.routes import storage as storage_api
from benchmark.routes.storage import (
    _MAX_UPLOAD_BYTES,
    StorageUploadForm,
    _resolve_object_key,
    _safe_basename,
    upload_file,
)
DailyDataLoader = None
DailyLoadSpec = None
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.testclient import TestClient


def test_safe_basename_strips_path() -> None:
    assert _safe_basename("../../etc/passwd") == "passwd"
    assert _safe_basename(None) == "upload.bin"
    assert _safe_basename("a/b/c.csv") == "c.csv"


def test_resolve_object_key_explicit() -> None:
    key = _resolve_object_key(
        filename="a.csv",
        prefix="ignored",
        object_key="misc/dumps/a.csv",
    )
    assert key == "misc/dumps/a.csv"


def test_resolve_object_key_rejects_traversal() -> None:
    with pytest.raises(HTTPException) as ei:
        _resolve_object_key(filename="a.csv", prefix=None, object_key="../etc/passwd")
    assert ei.value.status_code == 400


def _chunked_read(*chunks: bytes) -> AsyncMock:
    """Simulate a chunked upload stream: return chunks in order, and the last one must be empty bytes to end the read loop."""
    return AsyncMock(side_effect=[*chunks, b""])


@pytest.fixture(autouse=True)
def _fake_upload_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same-day sequence number uses an in-memory counter stand-in (incrementing from 1) and does not depend on a real DB counter row."""
    state = {"next": 0}

    async def _next(partition_date: str) -> int:
        state["next"] += 1
        return state["next"]

    monkeypatch.setattr("benchmark.routes.storage.next_oss_upload_id", _next)


@pytest.mark.asyncio
async def test_upload_503_when_storage_unavailable() -> None:
    fake = MagicMock(spec=UploadFile)
    fake.filename = "a.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read(b"x")
    with patch("benchmark.routes.storage.build_data_storage_client", return_value=None):
        with pytest.raises(HTTPException) as ei:
            await upload_file(file=fake, form=StorageUploadForm())
    assert ei.value.status_code == 503


@pytest.mark.asyncio
async def test_upload_rejects_empty_file() -> None:
    fake = MagicMock(spec=UploadFile)
    fake.filename = "a.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read()
    client = MagicMock()
    with patch("benchmark.routes.storage.build_data_storage_client", return_value=client):
        with pytest.raises(HTTPException) as ei:
            await upload_file(file=fake, form=StorageUploadForm(object_key="misc/a.csv"))
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_oversized() -> None:
    fake = MagicMock(spec=UploadFile)
    fake.filename = "big.bin"
    fake.content_type = "application/octet-stream"
    # The first chunk already exceeds the limit; no need to return an empty chunk (the read loop stops before the raise)
    fake.read = AsyncMock(return_value=b"x" * (_MAX_UPLOAD_BYTES + 1))
    client = MagicMock()
    with patch("benchmark.routes.storage.build_data_storage_client", return_value=client):
        with pytest.raises(HTTPException) as ei:
            await upload_file(file=fake, form=StorageUploadForm(object_key="misc/big.bin"))
    assert ei.value.status_code == 413


@pytest.mark.asyncio
async def test_upload_success_with_sign() -> None:
    fake = MagicMock(spec=UploadFile)
    fake.filename = "a.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read(b"hello")
    client = MagicMock()
    client.upload = AsyncMock(return_value="https://cdn.example/misc/a.csv")
    client.sign_url = AsyncMock(return_value="https://signed.example/misc/a.csv")
    with patch("benchmark.routes.storage.build_data_storage_client", return_value=client):
        out = await upload_file(
            file=fake,
            form=StorageUploadForm(object_key="misc/a.csv", sign=True, sign_expires_seconds=3600),
        )
    assert out["data"]["object_key"] == "misc/a.csv"
    assert out["data"]["url"] == "https://cdn.example/misc/a.csv"
    assert out["data"]["bytes"] == 5
    assert out["data"]["signed_url"].startswith("https://signed")
    client.upload.assert_awaited_once()
    client.sign_url.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_502_when_upload_returns_none() -> None:
    fake = MagicMock(spec=UploadFile)
    fake.filename = "a.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read(b"hello")
    client = MagicMock()
    client.upload = AsyncMock(return_value=None)
    with patch("benchmark.routes.storage.build_data_storage_client", return_value=client):
        with pytest.raises(HTTPException) as ei:
            await upload_file(file=fake, form=StorageUploadForm(object_key="misc/a.csv"))
    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_upload_oss_prefix_normalizes_like_fetch_side() -> None:
    """oss_prefix is stored as given: no common prefix is joined; the response backfills a full path with scheme.

    Live-requirement lock: the page fills amap_agent_ieval, stored as
    amap_agent_ieval/{date}/benchmark_final_results.csv; the backfill is
    oss://agent-eval-oss/amap_agent_ieval (the fetch side keeps the scheme as-is,
    and after the loader strips the scheme it matches the stored location).
    """
    fake = MagicMock(spec=UploadFile)
    fake.filename = "benchmark_final_results.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read(b"id,query")
    client = MagicMock()
    client.upload = AsyncMock(return_value="https://cdn.example/ok")
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/ieval-service/benchmark/benchmark_result",
        ),
    ):
        out = await upload_file(
            file=fake,
            form=StorageUploadForm(
                oss_prefix="amap_agent_ieval", content_type="text/csv", sign=False
            ),
        )

    # Stored object_key = the prefix as given + today + the upload-batch segment + the filename (no common prefix);
    # the server-generated batch segment comes back on the response, so repeated same-prefix same-day uploads must differ.
    key = out["data"]["object_key"]
    today = datetime.now(UTC).strftime("%Y%m%d")
    upload_id = out["data"]["upload_id"]
    assert upload_id
    assert key == f"amap_agent_ieval/{today}/{upload_id}/benchmark_final_results.csv"
    # The response backfills a full path with scheme: after the frontend writes dataset.ossPrefix, manual-runs forwards it,
    # the fetch side keeps it as-is → the loader strips the scheme → amap_agent_ieval/{date}/... hits
    assert out["data"]["oss_prefix"] == "oss://agent-eval-oss/amap_agent_ieval"
    # The partition date comes back on the response (server clock): the frontend writes it onto the dataset document
    # as ossPartitionDate, and at trigger time the server uses it to build the full object key.
    assert out["data"]["partition_date"] == today
    client.upload.assert_awaited_once_with(
        key,
        b"id,query",
        content_type="text/csv",
    )


@pytest.mark.asyncio
async def test_upload_same_prefix_twice_yields_distinct_keys() -> None:
    """Two uploads of the same prefix on the same day: the sequence number increments (1, 2), and the keys neither collide nor overwrite."""

    async def _once() -> dict[str, Any]:
        fake = MagicMock(spec=UploadFile)
        fake.filename = "benchmark_final_results.csv"
        fake.content_type = "text/csv"
        fake.read = _chunked_read(b"id,query")
        client = MagicMock()
        client.upload = AsyncMock(return_value="https://cdn.example/ok")
        with patch("benchmark.routes.storage.build_data_storage_client", return_value=client):
            out: dict[str, Any] = await upload_file(
                file=fake,
                form=StorageUploadForm(
                    oss_prefix="amap_agent_ieval", content_type="text/csv", sign=False
                ),
            )
            return out

    first, second = await _once(), await _once()
    # The same-day sequence number starts at 1 and increments by 1, so upload order is visible
    assert first["data"]["upload_id"] == 1
    assert second["data"]["upload_id"] == 2
    assert first["data"]["object_key"].endswith("/1/benchmark_final_results.csv")
    assert second["data"]["object_key"].endswith("/2/benchmark_final_results.csv")
    assert first["data"]["partition_date"] == second["data"]["partition_date"]


@pytest.mark.asyncio
async def test_upload_oss_prefix_strips_scheme_keeps_path() -> None:
    """An oss_prefix that includes a scheme: strip the scheme, store the path as-is, and backfill the full path."""
    fake = MagicMock(spec=UploadFile)
    fake.filename = "benchmark_final_results.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read(b"input,expected_output")
    client = MagicMock()
    client.upload = AsyncMock(return_value="https://cdn.example/ok")
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        out = await upload_file(
            file=fake,
            form=StorageUploadForm(
                oss_prefix="oss://agent-eval-oss/x/y", content_type="text/csv", sign=False
            ),
        )
    assert out["data"]["object_key"].startswith("x/y/")
    assert out["data"]["oss_prefix"] == "oss://agent-eval-oss/x/y"


@pytest.mark.asyncio
async def test_upload_oss_prefix_rejects_traversal() -> None:
    """An oss_prefix containing a .. segment → 400 (the resolve_oss_prefix validation chain takes effect)."""
    fake = MagicMock(spec=UploadFile)
    fake.filename = "a.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read(b"x")
    client = MagicMock()
    with patch("benchmark.routes.storage.build_data_storage_client", return_value=client):
        with pytest.raises(HTTPException) as ei:
            await upload_file(file=fake, form=StorageUploadForm(oss_prefix="../escape", sign=False))
    assert ei.value.status_code == 400


def _benchmark_upload_client() -> MagicMock:
    client = MagicMock()
    client.upload = AsyncMock(return_value="https://cdn.example/ok")
    return client


async def _upload_benchmark(
    content: bytes, *, oss_prefix: str, filename: str = "benchmark_final_results.csv"
) -> dict:  # type: ignore[type-arg]
    fake = MagicMock(spec=UploadFile)
    fake.filename = filename
    fake.content_type = "text/csv"
    fake.read = _chunked_read(content)
    client = _benchmark_upload_client()
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        out = await upload_file(
            file=fake,
            form=StorageUploadForm(oss_prefix=oss_prefix, content_type="text/csv", sign=False),
        )
    client.upload.assert_awaited_once()
    return out


@pytest.mark.asyncio
async def test_upload_benchmark_csv_with_required_columns_ok() -> None:
    """A benchmark dataset with input and expected_output columns (including a BOM) → stored normally."""
    out = await _upload_benchmark(
        b"\xef\xbb\xbfinput, expected_output ,note\nq1,a1,x\n",
        oss_prefix="example_benchmark/benchmark",
    )
    assert out["data"]["object_key"].startswith("example_benchmark/benchmark/")


@pytest.mark.asyncio
async def test_upload_benchmark_csv_missing_expected_output_rejected() -> None:
    """Missing the expected_output column → 400, and nothing is written to OSS."""
    fake = MagicMock(spec=UploadFile)
    fake.filename = "benchmark_final_results.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read(b"input,output\nq1,a1\n")
    client = _benchmark_upload_client()
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        with pytest.raises(HTTPException) as ei:
            await upload_file(
                file=fake,
                form=StorageUploadForm(
                    oss_prefix="example_benchmark/benchmark",
                    content_type="text/csv",
                    sign=False,
                ),
            )
    assert ei.value.status_code == 400
    assert "expected_output" in str(ei.value.detail)
    client.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_benchmark_csv_missing_input_rejected() -> None:
    """Missing the input column → 400, and detail names the missing column."""
    fake = MagicMock(spec=UploadFile)
    fake.filename = "benchmark_final_results.csv"
    fake.content_type = "text/csv"
    fake.read = _chunked_read(b"expected_output,output\na1,x\n")
    client = _benchmark_upload_client()
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        with pytest.raises(HTTPException) as ei:
            await upload_file(
                file=fake,
                form=StorageUploadForm(
                    oss_prefix="example_benchmark/benchmark",
                    content_type="text/csv",
                    sign=False,
                ),
            )
    assert ei.value.status_code == 400
    assert "input" in str(ei.value.detail)
    client.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_amap_agent_benchmark_exempt_from_column_check() -> None:
    """amap-agent family prefixes (trace semantics) are exempt from column checks: id/query columns upload normally."""
    out = await _upload_benchmark(b"id,query\n1,hi\n", oss_prefix="amap-agent/benchmark")
    assert out["data"]["object_key"].startswith("amap-agent/benchmark/")
    out = await _upload_benchmark(b"id,query\n1,hi\n", oss_prefix="amap_agent_ieval")
    assert out["data"]["object_key"].startswith("amap_agent_ieval/")


@pytest.mark.asyncio
async def test_upload_source_format_json_requires_oss_prefix() -> None:
    """raw mode with an explicit source_format=json → 400, so it is not silently ignored."""
    fake = MagicMock(spec=UploadFile)
    fake.filename = "cases.json"
    fake.content_type = "application/json"
    fake.read = _chunked_read(b'[{"id":"c1","query":"hello","response_text":"world"}]')
    client = _benchmark_upload_client()
    with patch("benchmark.routes.storage.build_data_storage_client", return_value=client):
        with pytest.raises(HTTPException) as ei:
            await upload_file(
                file=fake,
                form=StorageUploadForm(object_key="misc/cases.json", source_format="json"),
            )
    assert ei.value.status_code == 400
    assert "oss_prefix" in str(ei.value.detail)
    client.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_benchmark_json_extension_without_source_format_keeps_raw() -> None:
    """A *.json under oss_prefix without an explicit source_format=json → stored as-is, not silently converted to CSV."""
    fake = MagicMock(spec=UploadFile)
    fake.filename = "cases.json"
    fake.content_type = "application/json"
    payload = b'[{"id":"c1","query":"hello","response_text":"world","reference":"world"}]'
    fake.read = _chunked_read(payload)
    client = _benchmark_upload_client()
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        out = await upload_file(
            file=fake,
            form=StorageUploadForm(
                oss_prefix="example_benchmark/benchmark",
                content_type="application/json",
                sign=False,
            ),
        )
    assert out["data"]["filename"] == "cases.json"
    assert out["data"]["content_type"] == "application/json"
    uploaded = client.upload.await_args.args[1]
    assert uploaded == payload


@pytest.mark.asyncio
async def test_upload_benchmark_json_normalizes_to_csv() -> None:
    """A benchmark JSON upload is normalized to CSV, and the filename is fixed as benchmark_final_results.csv."""
    fake = MagicMock(spec=UploadFile)
    fake.filename = "cases.json"
    fake.content_type = "application/json"
    payload = b'[{"id":"c1","query":"hello","response_text":"world","reference":"world"}]'
    fake.read = _chunked_read(payload)
    client = _benchmark_upload_client()
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        out = await upload_file(
            file=fake,
            form=StorageUploadForm(
                oss_prefix="example_benchmark/benchmark",
                source_format="json",
                sign=False,
            ),
        )
    assert out["data"]["filename"] == "benchmark_final_results.csv"
    assert out["data"]["content_type"] == "text/csv"
    uploaded = client.upload.await_args.args[1]
    assert b"case_id" in uploaded
    assert b"input" in uploaded


@pytest.mark.asyncio
async def test_upload_benchmark_invalid_json_rejected() -> None:
    """An illegal JSON dataset → 400, and nothing is written to OSS."""
    fake = MagicMock(spec=UploadFile)
    fake.filename = "cases.json"
    fake.content_type = "application/json"
    fake.read = _chunked_read(b"not-json")
    client = _benchmark_upload_client()
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        with pytest.raises(HTTPException) as ei:
            await upload_file(
                file=fake,
                form=StorageUploadForm(
                    oss_prefix="example_benchmark/benchmark",
                    source_format="json",
                    sign=False,
                ),
            )
    assert ei.value.status_code == 400
    client.upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_benchmark_non_csv_skips_column_check() -> None:
    """In benchmark mode, non-CSV files skip the column check."""
    out = await _upload_benchmark(
        b"plain text", oss_prefix="example_benchmark/benchmark", filename="notes.txt"
    )
    assert out["data"]["filename"] == "notes.txt"


def _http_client() -> TestClient:
    """Assemble the real router and go through HTTP multipart rather than calling the function directly."""
    app = FastAPI()
    app.include_router(storage_api.router)
    return TestClient(app)


def test_multipart_text_fields_reach_endpoint() -> None:
    """HTTP layer: text fields such as oss_prefix / content_type / sign are actually forwarded.

    The old form (``Annotated[Model, Form()]`` on the same endpoint as ``File``) treated the model as a
    single field named ``form``, and every text field silently fell back to its default.
    """
    client = _benchmark_upload_client()
    client.sign_url = AsyncMock(return_value="https://signed.example/ok")
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        response = _http_client().post(
            "/v1/storage/upload",
            data={
                "oss_prefix": "amap_agent_ieval/benchmark",
                "content_type": "text/csv",
                "sign": "true",
                "sign_expires_seconds": "3600",
            },
            files={"file": ("notes.txt", b"plain text", "text/plain")},
        )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["object_key"].startswith("amap_agent_ieval/benchmark/")
    assert data["content_type"] == "text/csv"
    assert data["sign_expires_seconds"] == 3600
    assert data["signed_url"] == "https://signed.example/ok"


@pytest.mark.asyncio
async def test_multipart_json_normalizes_csv_and_daily_loader_reads_all_rows() -> None:
    payload = json.dumps(
        {
            "rows": [
                {"id": "case-1", "query": "杭州", "response_text": "晴", "reference": "晴"},
                {"id": "case-2", "query": "南京", "response_text": "阴", "reference": "阴"},
            ]
        },
        ensure_ascii=False,
    ).encode()
    client = _benchmark_upload_client()
    with (
        patch("benchmark.routes.storage.build_data_storage_client", return_value=client),
        patch(
            "benchmark.routes.storage.get_default_download_prefix",
            return_value="oss://agent-eval-oss/common",
        ),
    ):
        response = _http_client().post(
            "/v1/storage/upload",
            data={"oss_prefix": "example_benchmark/benchmark", "source_format": "json"},
            files={"file": ("dataset.json", payload, "application/json")},
        )
    assert response.status_code == 200, response.text
    result = response.json()["data"]
    assert result["object_key"].endswith("/benchmark_final_results.csv")
    assert result["content_type"] == "text/csv"
    uploaded = client.upload.await_args.args[1]
    rows = list(csv.DictReader(io.StringIO(uploaded.decode("utf-8"))))
    assert len(rows) == 2
    assert json.loads(rows[0]["input"]) == {"query": "杭州"}
    assert json.loads(rows[0]["output"]) == {"response_text": "晴"}

    loader_client = MagicMock()
    loader_client.download = AsyncMock(return_value=uploaded)
    parsed = await DailyDataLoader(loader_client).load_daily(
        DailyLoadSpec(
            run_id="run-json-upload",
            oss_prefix="",
            date="",
            object_key="unused",
        )
    )
    assert parsed.total_count == 2
    assert parsed.items[1]["metadata"]["csv_row"]["case_id"] == "case-2"


@pytest.mark.asyncio
async def test_upload_json_invalid_does_not_write_oss() -> None:
    fake = MagicMock(spec=UploadFile)
    fake.filename = "dataset.json"
    fake.content_type = "application/json"
    fake.read = _chunked_read(b'[{"output":"missing input"}]')
    client = _benchmark_upload_client()
    with patch("benchmark.routes.storage.build_data_storage_client", return_value=client):
        with pytest.raises(HTTPException) as ei:
            await upload_file(
                file=fake,
                form=StorageUploadForm(
                    oss_prefix="example_benchmark/benchmark",
                    source_format="json",
                ),
            )
    assert ei.value.status_code == 400
    assert "missing input/query" in str(ei.value.detail)
    client.upload.assert_not_awaited()
