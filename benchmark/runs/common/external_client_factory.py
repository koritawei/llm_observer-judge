"""AoneClientFactory — Aone client factory keyed by the per-run target URL (§6.3).

Construction receives the process-level ``base_url`` / ``api_key`` / default pipeline id
once (all fed by the live settings-read path in ``main.py``; **that read path is unchanged**).
At runtime, clients are cached and reused only by the full target URL — the cache key is the
URL. ``api_key`` is globally unique, is not part of the key, and is only injected into headers.

SSRF guard (§6.3 constraint 1): the target URL host must equal the host of the process-level
``judge.base_url`` (or the attribution base_url). A per-run override may change the
pipeline / path, but not the instance domain.
"""

import re
from collections import OrderedDict
from dataclasses import dataclass
from urllib.parse import urlparse

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.runs.common.external_client import (
    ExternalAttributionClient,
    ExternalJudgeClient,
    ExternalPipelineConfig,
    normalize_pipeline_base_url,
)

logger = get_logger("agent_eval.benchmark.common.aone_client_factory")

#: Trigger URL shape: ``{base}/api/pipelines/{pipeline_id}/trigger``
#: (base may carry an optional path prefix, for a gateway mount).
_TRIGGER_URL_PATTERN = re.compile(
    r"^(?P<base>https?://[^/]+(?:/[\w\-._~]*)?)/api/pipelines/(?P<pipeline_id>[^/]+)/trigger/?$",
)

#: Cap on the cache-key set (past the cap, LRU evicts the least recently used).
_DEFAULT_MAX_CACHE_SIZE = 64


class InvalidTargetUrlError(ValueError):
    """The judge / attribution target URL is invalid (bad format / host outside the allowlist)."""


@dataclass(frozen=True)
class ExternalClientFactoryConfig:
    """Construction config for AoneClientFactory (§6.3; fed by process-level settings, read path unchanged)."""

    judge_base_url: str
    api_key: str
    default_judge_pipeline_id: str = ""
    attribution_base_url: str = ""
    default_attribution_pipeline_id: str = ""
    max_cache_size: int = _DEFAULT_MAX_CACHE_SIZE


class ExternalClientFactory:
    """Build or reuse an Aone judge / attribution client by the full target URL.

    - ``default_judge_url`` / ``default_attribution_url``: the default target built from
      base_url plus the live default pipeline_id (equivalent to the live singleton);
    - ``get_*_client(None)`` returns the client for the default URL;
    - the cache has a cap and LRU eviction; hits / evictions are counted in ``aone.client_factory`` logs.
    """

    def __init__(self, config: ExternalClientFactoryConfig) -> None:
        self._api_key = config.api_key
        # base_url normalization matches the live client (config may carry an /api/pipelines suffix)
        self._judge_base_url = normalize_pipeline_base_url(config.judge_base_url)
        # Attribution base URL: explicit value wins; otherwise fall back to judge (same as live main.py)
        self._attribution_base_url = normalize_pipeline_base_url(
            config.attribution_base_url or config.judge_base_url,
        )
        self._default_judge_url = (
            f"{self._judge_base_url}/api/pipelines/{config.default_judge_pipeline_id}/trigger"
            if self._judge_base_url and config.default_judge_pipeline_id
            else None
        )
        # Attribution default URL: fall back to the judge default URL (same as live F3)
        default_attr_pid = (
            config.default_attribution_pipeline_id or config.default_judge_pipeline_id
        )
        self._default_attribution_url = (
            f"{self._attribution_base_url}/api/pipelines/{default_attr_pid}/trigger"
            if self._attribution_base_url and default_attr_pid
            else None
        )
        # Host allowlist source = hosts of the process-level judge / attribution base_url (no new env var)
        self._allowed_hosts = frozenset(
            host
            for host in (
                urlparse(self._judge_base_url).netloc,
                urlparse(self._attribution_base_url).netloc,
            )
            if host
        )
        self._max_cache_size = config.max_cache_size
        self._judge_clients: OrderedDict[str, ExternalJudgeClient] = OrderedDict()
        self._attr_clients: OrderedDict[str, ExternalAttributionClient] = OrderedDict()

    @property
    def default_judge_url(self) -> str | None:
        """Process-level default judge trigger URL (base_url + default pipeline_id)."""
        return self._default_judge_url

    @property
    def default_attribution_url(self) -> str | None:
        """Process-level default attribution trigger URL (falls back to the judge default URL)."""
        return self._default_attribution_url

    @property
    def allowed_hosts(self) -> frozenset[str]:
        """Target URL host allowlist (the set of process-level base_url hosts)."""
        return self._allowed_hosts

    def validate_target_url(self, target_url: str) -> None:
        """Validate the target URL shape plus the host allowlist (invalid → ``InvalidTargetUrlError``).

        A per-run override may change the pipeline / path, but not the instance domain (§6.3 SSRF guard).
        """
        if _TRIGGER_URL_PATTERN.match(target_url.strip()) is None:
            raise InvalidTargetUrlError(
                f"target url must match '<base>/api/pipelines/<id>/trigger': {target_url!r}",
            )
        host = urlparse(target_url.strip()).netloc
        if self._allowed_hosts and host not in self._allowed_hosts:
            raise InvalidTargetUrlError(
                f"target url host {host!r} not in allowed hosts {sorted(self._allowed_hosts)}",
            )

    def _parse_url(self, target_url: str) -> tuple[str, str]:
        """(After validation) split out ``(base_url, pipeline_id)`` so the live client constructor can be reused."""
        self.validate_target_url(target_url)
        matched = _TRIGGER_URL_PATTERN.match(target_url.strip())
        # validate_target_url already guaranteed a match; this only narrows the type
        assert matched is not None  # noqa: S101
        return matched.group("base"), matched.group("pipeline_id")

    def _evict_if_needed(self, cache: OrderedDict, *, kind: str) -> None:
        while len(cache) > self._max_cache_size:
            evicted_url, _ = cache.popitem(last=False)
            logger.info("aone.client_factory evict kind=%s url=%s", kind, evicted_url)

    def get_judge_client(self, target_url: str | None = None) -> ExternalJudgeClient:
        """Get the judge client for a full target URL (``None`` → the process default URL)."""
        url = target_url or self._default_judge_url
        if not url:
            raise InvalidTargetUrlError(
                "no judge target url given and no default judge pipeline configured",
            )
        cached = self._judge_clients.get(url)
        if cached is not None:
            self._judge_clients.move_to_end(url)
            logger.info("aone.client_factory hit kind=judge url=%s", url)
            return cached
        base_url, pipeline_id = self._parse_url(url)
        client = ExternalJudgeClient(
            ExternalPipelineConfig(
                base_url=base_url,
                api_key=self._api_key,
                pipeline_id=pipeline_id,
            ),
        )
        self._judge_clients[url] = client
        self._evict_if_needed(self._judge_clients, kind="judge")
        logger.info("aone.client_factory miss kind=judge url=%s", url)
        return client

    def get_attribution_client(self, target_url: str | None = None) -> ExternalAttributionClient:
        """Get the attribution client for a full target URL (``None`` → the process default URL)."""
        url = target_url or self._default_attribution_url
        if not url:
            raise InvalidTargetUrlError(
                "no attribution target url given and no default attribution pipeline configured",
            )
        cached = self._attr_clients.get(url)
        if cached is not None:
            self._attr_clients.move_to_end(url)
            logger.info("aone.client_factory hit kind=attribution url=%s", url)
            return cached
        base_url, pipeline_id = self._parse_url(url)
        client = ExternalAttributionClient(
            ExternalPipelineConfig(
                base_url=base_url,
                api_key=self._api_key,
                pipeline_id=pipeline_id,
            ),
        )
        self._attr_clients[url] = client
        self._evict_if_needed(self._attr_clients, kind="attribution")
        logger.info("aone.client_factory miss kind=attribution url=%s", url)
        return client


__all__ = [
    "ExternalClientFactory",
    "ExternalClientFactoryConfig",
    "InvalidTargetUrlError",
]
