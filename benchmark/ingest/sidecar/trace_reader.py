"""Read-only Lindorm PK get + OSS blob fetch for sidecar traces."""

import asyncio
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import unquote, urlsplit

DEFAULT_SQL_PORT = 33060


class OssGet(Protocol):
    async def get(self, bucket: str, key: str) -> bytes: ...


class LindormGet(Protocol):
    async def get(self, trace_id: str) -> dict[str, object] | None: ...


@dataclass(frozen=True, slots=True)
class SidecarTraceRecord:
    finish_reason: str
    observations: bytes


def parse_lindorm_dsn(dsn: str) -> dict[str, object]:
    parsed = urlsplit(dsn.strip())
    if parsed.scheme not in ("mysql", "lindorm"):
        raise ValueError(f"unsupported lindorm dsn scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise ValueError("lindorm dsn host is required")
    database = unquote(parsed.path.lstrip("/") or "default")
    if "/" in database:
        database = database.split("/", 1)[0]
    return {
        "host": parsed.hostname,
        "port": parsed.port or DEFAULT_SQL_PORT,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": database or "default",
    }


def _quote(identifier: str) -> str:
    return f"`{identifier.replace('`', '``')}`"


class _MysqlLindorm:
    def __init__(self, dsn: str, table: str) -> None:
        self._dsn = dsn
        self._table = table
        self._pool: object | None = None

    async def open(self) -> None:
        parsed = parse_lindorm_dsn(self._dsn)
        try:
            pooling = importlib.import_module("mysql.connector.pooling")
        except ImportError as exc:
            raise RuntimeError("lindorm requires mysql-connector-python>=8.0.15,<9.0") from exc
        kwargs = {
            "pool_name": "sidecar_trace_eval",
            "pool_size": 4,
            "host": parsed["host"],
            "port": parsed["port"],
            "user": parsed["user"],
            "password": parsed["password"],
            "database": parsed["database"],
            "pool_reset_session": True,
            "autocommit": True,
            "connection_timeout": 5,
            "use_pure": True,
        }
        self._pool = await asyncio.to_thread(pooling.MySQLConnectionPool, **kwargs)

    async def close(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is None:
            return
        remove = getattr(pool, "_remove_connections", None)
        if callable(remove):
            await asyncio.to_thread(remove)

    async def get(self, trace_id: str) -> dict[str, object] | None:
        if self._pool is None:
            raise RuntimeError("SidecarTraceReader is not open")
        sql = f"SELECT * FROM {_quote(self._table)} WHERE {_quote('trace_id')} = %s LIMIT 1"
        rows = await asyncio.to_thread(_run_on_pool, self._pool, sql, (trace_id,))
        return rows[0] if rows else None


def _run_on_pool(pool: object, sql: str, params: tuple[object, ...]) -> list[dict[str, object]]:
    connection = pool.get_connection()
    try:
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            names = [str(item[0]).lower() for item in (cursor.description or ())]
            return [_mysql_row(row, names) for row in rows]
        finally:
            cursor.close()
    finally:
        connection.close()


def _mysql_row(row: object, names: list[str]) -> dict[str, object]:
    if isinstance(row, Mapping):
        return {str(key).lower(): value for key, value in row.items()}
    if isinstance(row, (list, tuple)):
        return {name: value for name, value in zip(names, row, strict=False)}
    raise TypeError(f"unexpected mysql row type: {type(row)!r}")


class SidecarTraceReader:
    """Lindorm PK get then OSS GET. DSN is not AGENT_EVAL_DATABASE_URL."""

    def __init__(
        self,
        dsn: str,
        table: str,
        *,
        oss: OssGet,
        lindorm: LindormGet | None = None,
    ) -> None:
        self._oss = oss
        self._lindorm: LindormGet | _MysqlLindorm | None = lindorm
        self._mysql: _MysqlLindorm | None = None
        if lindorm is None:
            self._mysql = _MysqlLindorm(dsn, table)

    async def open(self) -> None:
        if self._mysql is not None:
            await self._mysql.open()
            return
        opener = getattr(self._lindorm, "open", None)
        if callable(opener):
            result = opener()
            if asyncio.iscoroutine(result):
                await result

    async def close(self) -> None:
        if self._mysql is not None:
            await self._mysql.close()
            return
        closer = getattr(self._lindorm, "close", None)
        if callable(closer):
            result = closer()
            if asyncio.iscoroutine(result):
                await result

    async def fetch_by_trace_id(self, trace_id: str) -> SidecarTraceRecord | None:
        row = await self._get_row(trace_id)
        if row is None:
            return None
        oss_key = str(row.get("oss_key") or "").strip()
        if not oss_key:
            raise RuntimeError(f"missing oss_key trace_id={trace_id}")
        bucket = str(row.get("oss_bucket") or "").strip()
        if not bucket:
            raise RuntimeError(f"missing oss_bucket trace_id={trace_id}")
        packed = await self._oss.get(bucket, oss_key)
        return SidecarTraceRecord(
            finish_reason=str(row.get("finish_reason") or ""),
            observations=bytes(packed) if packed is not None else b"",
        )

    async def _get_row(self, trace_id: str) -> dict[str, object] | None:
        if self._mysql is not None:
            return await self._mysql.get(trace_id)
        if self._lindorm is None:
            raise RuntimeError("SidecarTraceReader is not open")
        return await self._lindorm.get(trace_id)


__all__ = [
    "SidecarTraceReader",
    "SidecarTraceRecord",
    "parse_lindorm_dsn",
]
