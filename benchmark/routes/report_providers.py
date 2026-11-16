"""External report Provider CRUD API.

A business registers its own report service through this API. When a report is produced,
the platform resolves the Provider by ``benchmark_name`` and, for ``external_http``, POSTs
the ReportSpec to the business URL.

Path prefix: ``/v1/benchmarks/report-providers``
"""

import ipaddress
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from benchmark.records.models.benchmark_report_provider import BenchmarkReportProvider
from benchmark.domain.report_provider_config import (
    entry_from_row,
    entry_public_dict,
    fetch_remote_config_report_providers,
    resolve_report_provider,
)

router = APIRouter(prefix="/v1/benchmarks/report-providers", tags=["report-providers"])

_VALID = frozenset({"platform", "external_http"})
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata",
    }
)


class ReportProviderUpsertRequest(BaseModel):
    """Register or update an external report service."""

    model_config = {"extra": "forbid"}

    provider: str = Field(default="external_http", description="platform | external_http")
    endpoint_url: str = Field(default="", description="POST ReportSpec 的业务 URL")
    bearer_token: str | None = Field(default=None, description="可选 Bearer token")
    timeout_ms: int = Field(default=120_000, ge=1000, le=600_000)
    fallback_to_platform: bool = True
    enabled: bool = True
    description: str | None = None
    clear_bearer_token: bool = Field(
        default=False,
        description="true 时清空已存 token（更新场景）",
    )

    @field_validator("provider")
    @classmethod
    def _provider_ok(cls, value: str) -> str:
        text = value.strip().lower()
        if text not in _VALID:
            raise ValueError(f"provider must be one of {sorted(_VALID)}")
        return text

    @field_validator("endpoint_url")
    @classmethod
    def _url_strip(cls, value: str) -> str:
        return value.strip()


def _row_to_public(row: BenchmarkReportProvider) -> dict[str, Any]:
    return entry_public_dict(entry_from_row(row))


def _is_blocked_endpoint_url(url: str) -> bool:
    """Reject loopback / private / link-local / reserved addresses (SSRF against intranet and metadata).

    On save, only literal IPs and known dangerous hostnames are checked. There is no
    synchronous DNS, so the save API cannot hang on resolution. Whether a hostname
    resolves to an intranet address is left to egress network policy.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return True
    if parsed.scheme not in ("http", "https"):
        return True
    host = (parsed.hostname or "").strip().lower()
    if not host or host in _BLOCKED_HOSTNAMES:
        return True
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        literal.is_private
        or literal.is_loopback
        or literal.is_link_local
        or literal.is_reserved
        or literal.is_multicast
        or literal.is_unspecified
    )


def _require_benchmark_name(benchmark_name: str) -> str:
    name = benchmark_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="benchmark_name is required")
    return name


@router.get("")
async def list_report_providers() -> dict[str, Any]:
    """List DB-registered Providers plus RemoteConfig seeds (read-only merged view)."""
    rows = await BenchmarkReportProvider.all().order_by("benchmark_name")
    db_items = [_row_to_public(r) for r in rows]
    db_names = {i["benchmark_name"] for i in db_items}
    remote_config_items = []
    for entry in fetch_remote_config_report_providers().providers:
        if entry.benchmark_name in db_names:
            continue
        remote_config_items.append({**entry_public_dict(entry), "source": "remote_config"})
    for item in db_items:
        item["source"] = "db"
    return {"data": db_items + remote_config_items, "meta": {"total": len(db_items) + len(remote_config_items)}}


@router.get("/{benchmark_name}")
async def get_report_provider(benchmark_name: str) -> dict[str, Any]:
    """Look up the Provider in effect for a benchmark (including the resolved result)."""
    name = _require_benchmark_name(benchmark_name)
    row = await BenchmarkReportProvider.filter(benchmark_name=name).first()
    resolved = await resolve_report_provider(name)
    return {
        "data": {
            "registered": _row_to_public(row) if row else None,
            "resolved": entry_public_dict(resolved),
            "external_active": resolved.is_external_active(),
        }
    }


@router.put("/{benchmark_name}")
async def upsert_report_provider(
    benchmark_name: str,
    body: ReportProviderUpsertRequest,
) -> dict[str, Any]:
    """Register or update an external report service (idempotent upsert)."""
    name = _require_benchmark_name(benchmark_name)
    if body.provider == "external_http" and not body.endpoint_url:
        raise HTTPException(
            status_code=400,
            detail="endpoint_url is required when provider=external_http",
        )
    if body.endpoint_url and not (
        body.endpoint_url.startswith("http://") or body.endpoint_url.startswith("https://")
    ):
        raise HTTPException(status_code=400, detail="endpoint_url must be http(s)")
    if body.endpoint_url and _is_blocked_endpoint_url(body.endpoint_url):
        raise HTTPException(
            status_code=400,
            detail="endpoint_url must not target loopback/private/link-local addresses",
        )

    row = await BenchmarkReportProvider.filter(benchmark_name=name).first()
    if row is None:
        row = await BenchmarkReportProvider.create(
            benchmark_name=name,
            provider=body.provider,
            endpoint_url=body.endpoint_url,
            bearer_token=body.bearer_token,
            timeout_ms=body.timeout_ms,
            fallback_to_platform=body.fallback_to_platform,
            enabled=body.enabled,
            description=body.description,
        )
    else:
        row.provider = body.provider
        row.endpoint_url = body.endpoint_url
        row.timeout_ms = body.timeout_ms
        row.fallback_to_platform = body.fallback_to_platform
        row.enabled = body.enabled
        row.description = body.description
        if body.clear_bearer_token:
            row.bearer_token = None
        elif body.bearer_token is not None:
            row.bearer_token = body.bearer_token
        await row.save()

    return {"data": _row_to_public(row)}


@router.delete("/{benchmark_name}")
async def delete_report_provider(benchmark_name: str) -> dict[str, Any]:
    """Delete the DB registration (RemoteConfig seeds are unaffected)."""
    name = _require_benchmark_name(benchmark_name)
    deleted = await BenchmarkReportProvider.filter(benchmark_name=name).delete()
    if not deleted:
        raise HTTPException(status_code=404, detail="report provider not found")
    return {"data": {"benchmark_name": name, "deleted": True}}


__all__ = ["router"]
