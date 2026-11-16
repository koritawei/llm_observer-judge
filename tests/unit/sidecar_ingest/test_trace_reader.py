"""Lindorm PK get then OSS GET; miss never touches OSS."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from benchmark.ingest.sidecar.oss_blob import OssBlob
from benchmark.ingest.sidecar.trace_reader import (
    SidecarTraceReader,
    SidecarTraceRecord,
    _run_on_pool,
    parse_lindorm_dsn,
)


def test_parse_lindorm_dsn() -> None:
    parsed = parse_lindorm_dsn("mysql://user:pass@host.example:33060/default")
    assert parsed["host"] == "host.example"
    assert parsed["port"] == 33060
    assert parsed["user"] == "user"
    assert parsed["password"] == "pass"
    assert parsed["database"] == "default"


def test_parse_lindorm_scheme_alias() -> None:
    parsed = parse_lindorm_dsn("lindorm://u:p@h/db")
    assert parsed["host"] == "h"
    assert parsed["port"] == 33060
    assert parsed["database"] == "db"


def test_run_on_pool_closes_connection_if_cursor_fails() -> None:
    connection = MagicMock()
    connection.cursor.side_effect = RuntimeError("cursor fail")
    pool = MagicMock()
    pool.get_connection.return_value = connection
    with pytest.raises(RuntimeError, match="cursor fail"):
        _run_on_pool(pool, "SELECT 1", ())
    connection.close.assert_called_once()


def _reader(
    *,
    lindorm: MagicMock,
    oss: MagicMock,
) -> SidecarTraceReader:
    return SidecarTraceReader(
        dsn="mysql://u:p@h:33060/default",
        table="sidecar_trace",
        oss=oss,
        lindorm=lindorm,
    )


@pytest.mark.asyncio
async def test_lindorm_miss_does_not_get_oss() -> None:
    lindorm = MagicMock()
    lindorm.get = AsyncMock(return_value=None)
    oss = MagicMock()
    oss.get = AsyncMock()
    reader = _reader(lindorm=lindorm, oss=oss)
    assert await reader.fetch_by_trace_id("t-miss") is None
    lindorm.get.assert_awaited_once_with("t-miss")
    oss.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_hit_gets_oss_using_row_pointer() -> None:
    packed = b"zst-bytes"
    key = "sidecar-trace-worker/prod/p-1/2026/09/16/t-1.obs.zst"
    lindorm = MagicMock()
    lindorm.get = AsyncMock(
        return_value={
            "trace_id": "t-1",
            "finish_reason": "finish",
            "oss_bucket": "trace-bkt",
            "oss_key": key,
        }
    )
    oss = MagicMock()
    oss.get = AsyncMock(return_value=packed)
    reader = _reader(lindorm=lindorm, oss=oss)
    record = await reader.fetch_by_trace_id("t-1")
    assert record == SidecarTraceRecord(finish_reason="finish", observations=packed)
    oss.get.assert_awaited_once_with("trace-bkt", key)


@pytest.mark.asyncio
async def test_missing_oss_key_raises() -> None:
    lindorm = MagicMock()
    lindorm.get = AsyncMock(
        return_value={
            "trace_id": "t-1",
            "finish_reason": "timeout",
            "oss_bucket": "trace-bkt",
            "oss_key": "",
        }
    )
    oss = MagicMock()
    oss.get = AsyncMock()
    reader = _reader(lindorm=lindorm, oss=oss)
    with pytest.raises(RuntimeError, match="oss_key"):
        await reader.fetch_by_trace_id("t-1")
    oss.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_oss_get_failure_raises() -> None:
    lindorm = MagicMock()
    lindorm.get = AsyncMock(
        return_value={
            "trace_id": "t-1",
            "finish_reason": "finish",
            "oss_bucket": "trace-bkt",
            "oss_key": "k.obs.zst",
        }
    )
    oss = MagicMock()
    oss.get = AsyncMock(side_effect=RuntimeError("OSS download failed"))
    reader = _reader(lindorm=lindorm, oss=oss)
    with pytest.raises(RuntimeError, match="OSS download failed"):
        await reader.fetch_by_trace_id("t-1")


@pytest.mark.asyncio
async def test_oss_blob_static_get(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    fake_oss2 = types.ModuleType("oss2")
    fake_exceptions = types.ModuleType("oss2.exceptions")

    class OssError(Exception):
        pass

    fake_exceptions.OssError = OssError
    fake_bucket = MagicMock()
    fake_result = MagicMock()
    fake_result.read.return_value = b"packed"
    fake_bucket.get_object.return_value = fake_result
    fake_oss2.Bucket = MagicMock(return_value=fake_bucket)
    fake_oss2.Auth = MagicMock(return_value="auth_obj")
    fake_oss2.exceptions = fake_exceptions
    monkeypatch.setitem(__import__("sys").modules, "oss2", fake_oss2)
    monkeypatch.setitem(__import__("sys").modules, "oss2.exceptions", fake_exceptions)

    blob = OssBlob(
        endpoint="oss-cn-hangzhou.aliyuncs.com",
        access_key_id="AK",
        access_key_secret="SK",
    )
    data = await blob.get("trace-bkt", "k.obs.zst")
    assert data == b"packed"
    fake_oss2.Auth.assert_called_once_with("AK", "SK")
    fake_oss2.Bucket.assert_called_once()
    assert fake_oss2.Bucket.call_args[0][1] == "https://oss-cn-hangzhou.aliyuncs.com"
    assert fake_oss2.Bucket.call_args[0][2] == "trace-bkt"
    fake_bucket.get_object.assert_called_once_with("k.obs.zst")
