"""Aone Pages publish config for benchmark reports — loaded from RemoteConfig, no DB storage.

Same pattern as :mod:`benchmark.domain.report_storage_config`: config lives in RemoteConfig
with a TTL cache. The sensitive ``token`` field lives only in RemoteConfig and must not
enter the repo, yaml, or logs.

RemoteConfig JSON shape::

    {
        "enabled": false,
        "site_base_url": "https://<site>.io.alibaba-inc.com",
        "git_api_base": "https://code.alibaba-inc.com",
        "repo": "amap-llm/eval-report-pages",
        "branch": "master",
        "token": "<repo PAT>",
        "git_username": "<domain-account>",
        "timeout_s": 30
    }

``git_api_base`` **must use ``https://``**. An ``http://`` value is 302-redirected to
https by the intranet gateway (httpx does not follow redirects by default), so probe and
push both fail. Prefer ``https://code.alibaba-inc.com`` (all API versions).
``gitlab.alibaba-inc.com`` is a legacy host and supports only api/v3 (the publisher
implements the v3 contract, so either host works), but a v4 path is answered 302/405.

``enabled`` defaults to ``False``: when RemoteConfig is unset or unavailable, publish is
disabled (graceful degradation; the existing OSS report path is unaffected).

``git_username`` is the git https push username (domain account). Alicode auth matrix
(code push and download): a privateToken (PAT) must be paired with the domain-account
username; the fixed ``oauth2`` username must be paired with an OAuth2 token. They are
not interchangeable.

Usage::

    from benchmark.domain.report_site_config import fetch_report_site_config

    cfg = fetch_report_site_config()
    if not cfg.enabled:
        # publish disabled
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.domain._cache import get_or_compute

logger = get_logger(__name__)

#: RemoteConfig data_id for the Aone Pages publish config.
REPORT_SITE_DATA_ID = "agent-eval.benchmark.report-site"
#: RemoteConfig group for the Aone Pages publish config.
REPORT_SITE_GROUP = "agent-eval.service.group"


class ReportSiteConfig(BaseModel):
    """Aone Pages publish config — loaded from RemoteConfig and validated by Pydantic.

    Every field has a default; ``enabled`` defaults to ``False``. ``token`` is a Git
    platform PAT (least privilege: repository write) and lives only in RemoteConfig.
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False
    #: Pages site base URL (used to derive URLs), e.g. ``https://<site>.io.alibaba-inc.com``.
    site_base_url: str = ""
    #: Git platform API base, e.g. ``https://code.alibaba-inc.com`` (must be https;
    #: http is 302'd to https and probe/push fail; the legacy gitlab host supports only v3).
    git_api_base: str = ""
    #: Reports repo (``group/project`` form).
    repo: str = ""
    #: Push branch (must match deploy-pages production-branch; a push is the production release).
    branch: str = "master"
    #: Git platform PAT (PRIVATE-TOKEN). Do not commit it or write it to logs.
    token: str = ""
    #: git https push username (domain account; a PAT needs the domain account, an oauth2
    #: username needs an OAuth2 token; they are not interchangeable). Empty = git push of
    #: large reports (>20MB) is unavailable; the small-report Files API path is unaffected.
    git_username: str = ""
    timeout_s: float = Field(default=30.0, ge=1)


def fetch_report_site_config(
    *,
    data_id: str | None = None,
    group: str | None = None,
    use_cache: bool = True,
) -> ReportSiteConfig:
    """Load and parse Aone Pages config from RemoteConfig; unavailable returns defaults (disabled).

    Results use the parsed-result cache by default (TTL 300s, see :mod:`benchmark.domain._cache`).
    Pass ``use_cache=False`` to force a refresh.

    Args:
        data_id: RemoteConfig data_id, default ``AONE_PAGES_DATA_ID``.
        group: RemoteConfig group, default ``AONE_PAGES_GROUP``.
        use_cache: Whether to use the cache, default ``True``.
    """
    data_id = data_id or REPORT_SITE_DATA_ID
    group = group or REPORT_SITE_GROUP

    def _load_and_parse() -> ReportSiteConfig:
        raw = fetch_cached_settings(data_id, group, use_cache=use_cache)
        if not raw:
            return ReportSiteConfig()
        try:
            return ReportSiteConfig.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "RemoteConfig aone-pages config parse failed (%s/%s): %s",
                data_id,
                group,
                exc,
            )
            return ReportSiteConfig()

    return get_or_compute(
        (data_id, group),
        _load_and_parse,
        use_cache=use_cache,
    )


__all__ = [
    "REPORT_SITE_DATA_ID",
    "REPORT_SITE_GROUP",
    "ReportSiteConfig",
    "fetch_report_site_config",
]
