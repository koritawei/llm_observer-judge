"""Unit tests for AoneClientFactory (design doc §6.3).

Locked contract:
- target URL shape + host allowlist (SSRF protection: the pipeline may change, the instance domain may not)
- default URL = base_url + the default pipeline_id (equivalent to the current production singleton)
- reuse by full URL cache + LRU eviction
- api_key is not part of the cache key and is injected only into the request header (it enters no key or log semantics)
"""

import pytest
from benchmark.runs.common.external_client import ExternalAttributionClient, ExternalJudgeClient
from benchmark.runs.common.external_client_factory import (
    ExternalClientFactory,
    ExternalClientFactoryConfig,
    InvalidTargetUrlError,
)

_BASE = "https://aone.example.com"
_ATTR_BASE = "https://aone-attr.example.com"


def _factory(**overrides: object) -> ExternalClientFactory:
    kwargs: dict[str, object] = {
        "judge_base_url": _BASE,
        "api_key": "ak-test",
        "default_judge_pipeline_id": "100",
        "attribution_base_url": "",
        "default_attribution_pipeline_id": "200",
    }
    kwargs.update(overrides)
    return ExternalClientFactory(ExternalClientFactoryConfig(**kwargs))  # type: ignore[arg-type]


def test_default_urls_built_from_process_config() -> None:
    """Default target URL = base_url + the production default pipeline_id (behavior matches the production singleton)."""
    factory = _factory()
    assert factory.default_judge_url == f"{_BASE}/api/pipelines/100/trigger"
    assert factory.default_attribution_url == f"{_BASE}/api/pipelines/200/trigger"


def test_base_url_with_pipelines_suffix_normalized() -> None:
    """Production pre-env shape regression: a base_url with an /api/pipelines suffix → after normalization the default
    URL is not double-joined, and get_judge_client() does not raise at startup (production-incident regression)."""
    factory = _factory(judge_base_url=f"{_BASE}/api/pipelines")
    assert factory.default_judge_url == f"{_BASE}/api/pipelines/100/trigger"
    client = factory.get_judge_client()
    assert isinstance(client, ExternalJudgeClient)
    # attribution falls back to the judge base by default, and is normalized the same way
    assert factory.default_attribution_url == f"{_BASE}/api/pipelines/200/trigger"
    # The host allowlist is not affected by suffix normalization
    assert "aone.example.com" in factory.allowed_hosts


def test_attribution_default_falls_back_to_judge_pipeline() -> None:
    """Default attribution URL: when the attr base is omitted, fall back to the judge base + judge pipeline (F3 semantics)."""
    factory = _factory(default_attribution_pipeline_id="")
    assert factory.default_attribution_url == f"{_BASE}/api/pipelines/100/trigger"


def test_get_client_caches_per_url() -> None:
    """The same URL reuses one client instance; different URLs each get their own instance."""
    factory = _factory()
    c1 = factory.get_judge_client()
    c2 = factory.get_judge_client()
    assert c1 is c2
    custom = f"{_BASE}/api/pipelines/999/trigger"
    c3 = factory.get_judge_client(custom)
    assert isinstance(c3, ExternalJudgeClient)
    assert c3 is factory.get_judge_client(custom)
    assert c3 is not c1


def test_get_attribution_client_type_and_cache() -> None:
    factory = _factory()
    client = factory.get_attribution_client()
    assert isinstance(client, ExternalAttributionClient)
    assert client is factory.get_attribution_client()


def test_target_url_host_outside_whitelist_rejected() -> None:
    """SSRF protection: a host outside the process base_url host → InvalidTargetUrlError."""
    factory = _factory()
    with pytest.raises(InvalidTargetUrlError, match="not in allowed hosts"):
        factory.validate_target_url("https://evil.example.com/api/pipelines/1/trigger")


def test_target_url_bad_form_rejected() -> None:
    factory = _factory()
    with pytest.raises(InvalidTargetUrlError, match="trigger"):
        factory.validate_target_url("https://aone.example.com/api/jobs/1")
    with pytest.raises(InvalidTargetUrlError):
        factory.validate_target_url("not-a-url")


def test_attribution_host_whitelisted_when_configured() -> None:
    """The attribution base_url host is also on the allowlist (§6.3 same-origin check)."""
    factory = _factory(attribution_base_url=_ATTR_BASE)
    factory.validate_target_url(f"{_ATTR_BASE}/api/pipelines/7/trigger")  # Does not raise
    assert factory.default_attribution_url == f"{_ATTR_BASE}/api/pipelines/200/trigger"


def test_lru_eviction() -> None:
    """Cache over the limit → evict the least recently used (hit / miss / evict are observable)."""
    factory = _factory(max_cache_size=2)
    factory.get_judge_client(f"{_BASE}/api/pipelines/1/trigger")
    factory.get_judge_client(f"{_BASE}/api/pipelines/2/trigger")
    factory.get_judge_client(f"{_BASE}/api/pipelines/1/trigger")  # touch → 2 is the oldest
    factory.get_judge_client(f"{_BASE}/api/pipelines/3/trigger")
    assert f"{_BASE}/api/pipelines/2/trigger" not in factory._judge_clients
    assert f"{_BASE}/api/pipelines/1/trigger" in factory._judge_clients
    assert f"{_BASE}/api/pipelines/3/trigger" in factory._judge_clients


def test_missing_default_and_no_target_raises() -> None:
    """Neither a target nor a process default → an explicit error (rather than a silent None)."""
    factory = _factory(default_judge_pipeline_id="")
    assert factory.default_judge_url is None
    with pytest.raises(InvalidTargetUrlError, match="no judge target"):
        factory.get_judge_client()
