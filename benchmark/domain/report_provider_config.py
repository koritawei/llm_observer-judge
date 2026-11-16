"""External report Provider domain model and resolution.

Resolution priority (high → low):
1. DB ``benchmark_report_providers`` (API registration, enabled=true)
2. RemoteConfig ``agent-eval.benchmark.report-providers`` (ops seed config)
3. Default ``platform`` (webdev-agent)

Usage::

    from benchmark.domain.report_provider_config import resolve_report_provider

    cfg = await resolve_report_provider("search_eval")
    if cfg.provider == "external_http":
        ...
"""

from typing import Any, Literal

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field, field_validator

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.settings.setting import settings
from benchmark.records.models.benchmark_report_provider import BenchmarkReportProvider
from benchmark.domain._cache import get_or_compute
from benchmark.domain.eval_plugin_config import resolve_plugin_binding

logger = get_logger(__name__)

REPORT_PROVIDER_DATA_ID = "agent-eval.benchmark.report-providers"
REPORT_PROVIDER_GROUP = "agent-eval.service.group"

ProviderKind = Literal["platform", "external_http", "local", "remote"]
_VALID_PROVIDERS = frozenset({"platform", "external_http", "local", "remote"})
_DEFAULT_TIMEOUT_MS = 120_000


class ReportProviderEntry(BaseModel):
    """Report Provider config for one benchmark."""

    model_config = {"extra": "ignore"}

    benchmark_name: str
    provider: ProviderKind = "platform"
    endpoint_url: str = ""
    bearer_token: str | None = None
    timeout_ms: int = _DEFAULT_TIMEOUT_MS
    fallback_to_platform: bool = True
    enabled: bool = True
    description: str | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: object) -> str:
        text = str(value or "platform").strip().lower()
        # Aligned with the shared plugin mode: local↔platform, remote↔external_http.
        if text in ("local", "inprocess", "in_process"):
            return "platform"
        if text in ("remote", "http", "rpc"):
            return "external_http"
        if text not in _VALID_PROVIDERS:
            logger.warning("invalid report provider %r, fallback to platform", value)
            return "platform"
        return text

    @field_validator("endpoint_url", mode="before")
    @classmethod
    def _strip_url(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("timeout_ms", mode="before")
    @classmethod
    def _clamp_timeout(cls, value: object) -> int:
        try:
            n = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _DEFAULT_TIMEOUT_MS
        return max(1_000, min(n, 600_000))

    def is_external_active(self) -> bool:
        """Whether report generation should go out over external HTTP."""
        return self.enabled and self.provider == "external_http" and bool(self.endpoint_url)


class ReportProvidersConfig(BaseModel):
    """RemoteConfig top level: ``{ "providers": [ {...}, ... ] }``."""

    model_config = {"extra": "ignore"}

    providers: list[ReportProviderEntry] = Field(default_factory=list)


def _platform_default(benchmark_name: str) -> ReportProviderEntry:
    return ReportProviderEntry(
        benchmark_name=benchmark_name,
        provider="platform",
        enabled=False,
    )


def fetch_remote_config_report_providers(
    *,
    data_id: str | None = None,
    group: str | None = None,
    use_cache: bool = True,
) -> ReportProvidersConfig:
    """Load report Provider seed config from RemoteConfig; an empty list on failure."""
    data_id = data_id or settings.benchmark_report_providers_remote_config_data_id
    group = group or settings.benchmark_report_providers_remote_config_group

    def _load() -> ReportProvidersConfig:
        raw = fetch_cached_settings(data_id, group, use_cache=use_cache)
        if not raw:
            return ReportProvidersConfig()
        try:
            return ReportProvidersConfig.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "RemoteConfig report-providers parse failed (%s/%s): %s",
                data_id,
                group,
                exc,
            )
            return ReportProvidersConfig()

    return get_or_compute((data_id, group), _load, use_cache=use_cache)


def entry_from_row(row: BenchmarkReportProvider) -> ReportProviderEntry:
    """DB row → :class:`ReportProviderEntry` (shared by API / resolve so behavior does not diverge)."""
    return ReportProviderEntry(
        benchmark_name=str(row.benchmark_name),
        provider=str(row.provider or "external_http"),  # type: ignore[arg-type]
        endpoint_url=str(row.endpoint_url or ""),
        bearer_token=row.bearer_token,
        timeout_ms=int(row.timeout_ms or _DEFAULT_TIMEOUT_MS),
        fallback_to_platform=bool(row.fallback_to_platform),
        enabled=bool(row.enabled),
        description=row.description,
    )


async def resolve_report_provider(benchmark_name: str) -> ReportProviderEntry:
    """Resolve the report Provider for a benchmark (DB > RemoteConfig > platform)."""
    name = (benchmark_name or "").strip()
    if not name:
        return _platform_default("")

    try:
        row = await BenchmarkReportProvider.filter(benchmark_name=name).first()
        if row is not None:
            entry = entry_from_row(row)
            # A DB row is authoritative: enabled=false → force platform.
            if not entry.enabled or entry.provider == "platform":
                return _platform_default(name)
            return entry
    except Exception as exc:
        logger.warning(
            "report_provider.db_lookup_failed benchmark=%s err=%s",
            name,
            exc,
        )

    remote_config = fetch_remote_config_report_providers()
    for item in remote_config.providers:
        if item.benchmark_name == name and item.is_external_active():
            return item

    # Shared-plugin RemoteConfig (agent-eval.eval.plugins.report_providers.bindings).
    try:
        binding = resolve_plugin_binding("report_providers", name)
        if binding is not None and binding.is_remote_active():
            return ReportProviderEntry(
                benchmark_name=name,
                provider="external_http",
                endpoint_url=binding.endpoint_url,
                bearer_token=binding.bearer_token,
                timeout_ms=binding.timeout_ms,
                fallback_to_platform=binding.fallback_local,
                enabled=True,
                description=binding.description,
            )
    except Exception as exc:
        logger.warning(
            "report_provider.eval_plugins_lookup_failed benchmark=%s err=%s",
            name,
            exc,
        )

    return _platform_default(name)


def entry_public_dict(entry: ReportProviderEntry) -> dict[str, Any]:
    """Public API serialization: hide the full token and only mark whether one is set."""
    data = entry.model_dump()
    token = data.pop("bearer_token", None)
    data["has_bearer_token"] = bool(token)
    return data


__all__ = [
    "REPORT_PROVIDER_DATA_ID",
    "REPORT_PROVIDER_GROUP",
    "ReportProviderEntry",
    "ReportProvidersConfig",
    "entry_from_row",
    "entry_public_dict",
    "fetch_remote_config_report_providers",
    "resolve_report_provider",
]
