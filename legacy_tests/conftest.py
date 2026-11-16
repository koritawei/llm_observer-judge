"""Shared pytest fixtures for agent-eval unit tests."""
# ruff: noqa: PLC0415  (test-only: lazy import inside fixture)

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _runtime_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the runtime build the `models` section LLM cleanly at bootstrap.

    Dummy api_keys satisfy eager `runtime.get_llm` builds (models are never actually
    called — tests swap evaluator LLM seams first), and SOCKS/HTTP proxy vars are
    cleared because they break model-client construction.
    """

    monkeypatch.setenv("AGENT_EVAL_MODEL_API_KEY", "sk-test-dummy")
    monkeypatch.setenv(
        "AGENT_EVAL_JUDGE_AMAP_AGENT_MODEL_API_KEY",
        "test-placeholder-not-used",
    )
    monkeypatch.setenv(
        "AGENT_EVAL_JUDGE_AMAP_SEARCH_AGENT_MODEL_API_KEY",
        "test-placeholder-not-used",
    )
    for var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def reset_provider_registry() -> Iterator[None]:
    """Reset ProviderRegistry discovery around each test so every app boot re-discovers.

    ``ProviderRegistry.discover()`` is process-global + one-shot; without this the
    second ``create_app()`` bootstrap in a session raises 'discover already run'.
    """

    try:
        from benchmark.libs.agenthost.app.providers import ProviderRegistry
    except ImportError:
        yield
        return
    ProviderRegistry._discovered = False
    yield
    ProviderRegistry._discovered = False


@pytest.fixture(autouse=True)
def _clear_domain_parsed_cache() -> None:
    """Clear domain-layer parsed-result cache before each test.

    Domain modules (``agent_config``, ``owner_mapping``, ``aone_dispatch_config``,
    ``eval_config``) cache parsed dataclass results by ``(data_id, group)``.
    Without this fixture, a test that patches ``fetch_cached_settings`` with
    one payload and a subsequent test using the same ``(data_id, group)`` but
    a different payload would receive stale cached results.
    """
    from benchmark.domain._cache import clear_parsed_cache

    clear_parsed_cache()
