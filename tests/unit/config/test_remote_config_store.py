"""Regression tests for RemoteConfig caching bucketed by (data_id, group).

Historical bug: ``fetch_cached_settings`` used a single global slot ``_settings_cache``
and did not distinguish ``data_id`` / ``group``. Domain modules shared the cache,
so whatever was fetched first became "fake data" for later modules (within the 300s TTL).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from benchmark.settings.remote_config_store import (
    clear_cache,
    fetch_cached_settings,
)
from benchmark.domain.agent_config import fetch_agent_configs
from benchmark.domain.notification_dispatch_config import fetch_notification_dispatch_profiles
from benchmark.domain.owner_mapping import fetch_owner_mappings

# ---------------------------------------------------------------------------
# Unit layer: the bucket contract of the cache structure itself
# ---------------------------------------------------------------------------


def test_cache_isolated_by_data_id_and_group() -> None:
    """Different (data_id, group) pairs must cache separately and must not overwrite each other."""
    clear_cache()
    payloads = {
        ("config.owner.mapping", "group.owner_mapping"): {"mappings": [{"id": "owner"}]},
        ("config.agent.eval.notification", "group.agent_eval"): {"agent-eval": {"notification_project_id": "1"}},
        ("config.agent.registry", "group.agent_eval"): {"agents": [{"name": "x"}]},
    }

    def _fake_fetch(data_id: str, group: str) -> dict:
        return payloads.get((data_id, group), {})

    with patch(
        "benchmark.settings.remote_config_store.fetch_remote_settings",
        side_effect=_fake_fetch,
    ) as fetch_mock:
        fetch_cached_settings("config.owner.mapping", "group.owner_mapping")
        fetch_cached_settings("config.agent.eval.notification", "group.agent_eval")
        fetch_cached_settings("config.agent.registry", "group.agent_eval")

        assert fetch_mock.call_count == 3
        called_keys = {(c.args[0], c.args[1]) for c in fetch_mock.call_args_list}
        assert called_keys == set(payloads.keys())

        # Cache hit: do not hit the network again
        fetch_cached_settings("config.owner.mapping", "group.owner_mapping")
        fetch_cached_settings("config.agent.eval.notification", "group.agent_eval")
        assert fetch_mock.call_count == 3


def test_use_cache_false_bypasses_cache() -> None:
    """use_cache=False must force a refresh and must not read the cache."""
    clear_cache()

    with patch(
        "benchmark.settings.remote_config_store.fetch_remote_settings",
        return_value={"k": "v"},
    ) as fetch_mock:
        fetch_cached_settings("id", "grp")  # write the cache
        fetch_cached_settings("id", "grp")  # hit
        assert fetch_mock.call_count == 1

        fetch_cached_settings("id", "grp", use_cache=False)
        fetch_cached_settings("id", "grp", use_cache=False)
        assert fetch_mock.call_count == 3


def test_clear_cache_empties_all_buckets() -> None:
    """clear_cache() must empty every bucket (not only the current key)."""
    clear_cache()

    with patch(
        "benchmark.settings.remote_config_store.fetch_remote_settings",
        return_value={"v": 1},
    ) as fetch_mock:
        fetch_cached_settings("a", "g1")
        fetch_cached_settings("b", "g2")
        assert fetch_mock.call_count == 2

        fetch_cached_settings("a", "g1")  # hit
        fetch_cached_settings("b", "g2")  # hit
        assert fetch_mock.call_count == 2

        clear_cache()
        fetch_cached_settings("a", "g1")
        fetch_cached_settings("b", "g2")
        assert fetch_mock.call_count == 4


def test_cached_payload_isolates_top_level_mutation() -> None:
    """The dict returned on a cache hit: a caller mutating the top level must not pollute the cache."""
    clear_cache()

    with patch(
        "benchmark.settings.remote_config_store.fetch_remote_settings",
        return_value={"k": "v"},
    ):
        first = fetch_cached_settings("id", "grp")
        first["new"] = "x"  # mutate the top level

        second = fetch_cached_settings("id", "grp")
        assert "new" not in second  # the cache was not polluted


# ---------------------------------------------------------------------------
# Integration layer: simulate the path that triggered the real bug
#
# This is the test that actually catches a regression of the single-slot shared bug: several domain modules
# all call fetch_cached_settings for different keys, and must not pollute each other.
# If someone later changes the cache back to a single slot, this test fails.
# ---------------------------------------------------------------------------


def test_domain_modules_do_not_share_cache_payload() -> None:
    """Regression: different domain modules fetch different (data_id, group) pairs and must each get their own payload.

    Historical bug: a single global cache slot made a later module treat an earlier module's data as its own return value.
    """
    clear_cache()

    owner_payload = {"mappings": [{"mapping_id": "m1", "match_key": "a/b"}]}
    notification_payload = {"agent-eval": {"notification_project_id": "2165583", "assignee": "529504"}}
    agent_payload = {
        "agents": [{"agent_name": "vehicle-agent", "project_id": "p1", "owner": "alice"}]
    }

    # Simulate RemoteConfig returning different payloads (dispatched by key)
    def _fake_fetch(data_id: str, group: str) -> dict:
        if (data_id, group) == ("config.owner.mapping", "group.owner_mapping"):
            return owner_payload
        if (data_id, group) == ("config.agent.eval.notification", "group.agent_eval"):
            return notification_payload
        if (data_id, group) == ("config.agent.registry", "group.agent_eval"):
            return agent_payload
        return {}

    with patch(
        "benchmark.settings.remote_config_store.fetch_remote_settings",
        side_effect=_fake_fetch,
    ):
        # A plausible historical call order: owner first, aone next, agent last
        owners = fetch_owner_mappings(data_id="config.owner.mapping", group="group.owner_mapping")
        notifications = fetch_notification_dispatch_profiles(
            data_id="config.agent.eval.notification", group="group.agent_eval"
        )
        agents = fetch_agent_configs(data_id="config.agent.registry", group="group.agent_eval")

        # each receives its own payload
        assert len(owners) == 1
        assert owners[0].mapping_id == "m1"

        assert "agent-eval" in notifications
        assert notifications["agent-eval"].notification_project_id == "2165583"
        # Key regression assertion: the aone payload is not polluted by the owner / agent payload
        assert "mappings" not in notifications
        assert "agents" not in notifications

        assert len(agents) == 1
        assert agents[0].agent_name == "vehicle-agent"
        # Key regression assertion: the agent payload is not polluted by the owner / aone payload
        assert "mappings" not in agents
        assert "agent-eval" not in agents


def test_ttl_expiry_triggers_refetch() -> None:
    """After the TTL expires it must fetch again and must not reuse the old value."""
    clear_cache()

    call_count = {"n": 0}

    def _fake_fetch(data_id: str, group: str) -> dict:
        call_count["n"] += 1
        return {"v": call_count["n"]}

    with patch(
        "benchmark.settings.remote_config_store.fetch_remote_settings",
        side_effect=_fake_fetch,
    ):
        # TTL=0 is equivalent to disabling the cache; every call fetches
        assert fetch_cached_settings("id", "grp", ttl=0) == {"v": 1}
        assert fetch_cached_settings("id", "grp", ttl=0) == {"v": 2}
        assert call_count["n"] == 2


def test_short_ttl_honored_independently_of_global_cache() -> None:
    """A short ttl passed by the caller must control expiry and must not be held back by the global 300s."""
    clear_cache()
    call_count = {"n": 0}

    def _fake_fetch(data_id: str, group: str) -> dict:
        call_count["n"] += 1
        return {"v": call_count["n"]}

    with (
        patch(
            "benchmark.settings.remote_config_store.fetch_remote_settings",
            side_effect=_fake_fetch,
        ),
        patch("benchmark.settings.remote_config_store.time.monotonic") as mono,
    ):
        mono.return_value = 1000.0
        assert fetch_cached_settings("switch", "grp", ttl=10) == {"v": 1}
        assert fetch_cached_settings("switch", "grp", ttl=10) == {"v": 1}
        assert call_count["n"] == 1

        # Under 10s: still a hit
        mono.return_value = 1009.0
        assert fetch_cached_settings("switch", "grp", ttl=10) == {"v": 1}
        assert call_count["n"] == 1

        # Past 10s: fetch again
        mono.return_value = 1010.0
        assert fetch_cached_settings("switch", "grp", ttl=10) == {"v": 2}
        assert call_count["n"] == 2


def test_zero_ttl_does_not_pollute_cache() -> None:
    """When ttl=0, do not write the cache; a later ttl>0 call must fetch again."""
    clear_cache()

    with patch(
        "benchmark.settings.remote_config_store.fetch_remote_settings",
        return_value={"k": 1},
    ) as fetch_mock:
        fetch_cached_settings("id", "grp", ttl=0)
        fetch_cached_settings("id", "grp", ttl=0)
        assert fetch_mock.call_count == 2  # ttl=0 fetches every time

        # Switch back to ttl>0 and it must fetch again (because ttl=0 did not write the cache)
        fetch_cached_settings("id", "grp", ttl=300)
        assert fetch_mock.call_count == 3


def test_fetch_remote_settings_logs_raw_and_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The real fetch path: log the original text; empty content raises the log to WARNING and returns {}."""
    import logging

    from benchmark.settings import remote_config_store as store

    clear_cache()
    client = type("C", (), {"get_config": staticmethod(lambda **_: "")})()
    with (
        patch.object(store, "_get_remote_config_client", return_value=client),
        caplog.at_level(logging.INFO, logger="benchmark.settings.remote_config_store"),
    ):
        result = store.fetch_remote_settings(
            "agent-eval.service.evaluators", "agent-eval.service.group"
        )

    assert result == {}
    joined = "\n".join(r.message for r in caplog.records)
    assert "RemoteConfig get_config raw:" in joined
    assert "agent-eval.service.evaluators" in joined
    assert "returned empty content" in joined


def test_fetch_remote_settings_logs_full_parsed_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-empty config: the original text plus the parsed JSON are logged in full."""
    import logging

    from benchmark.settings import remote_config_store as store

    clear_cache()
    body = '{"version":"1.0","configs":{"p1":[{"evaluatorName":"example","sampling":0.1}]}}'
    client = type("C", (), {"get_config": staticmethod(lambda **_: body)})()
    with (
        patch.object(store, "_get_remote_config_client", return_value=client),
        caplog.at_level(logging.INFO, logger="benchmark.settings.remote_config_store"),
    ):
        result = store.fetch_remote_settings("agent-eval.service.evaluators", "g")

    assert "p1" in result.get("configs", {})
    joined = "\n".join(r.message for r in caplog.records)
    assert "RemoteConfig get_config raw:" in joined
    assert body in joined
    assert "RemoteConfig get_config parsed:" in joined
    assert "evaluatorName" in joined


def test_fetch_remote_settings_masks_token_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Config that contains a token: the return value keeps the real value, but raw/parsed logs must never show the token in clear text."""
    import logging

    from benchmark.settings import remote_config_store as store

    clear_cache()
    secret = "FAKE_TOKEN_VALUE_FOR_LOG_MASK_TEST"
    body = (
        '{"enabled":true,"repo":"agent_eval_prompt_skill",'
        f'"token":"{secret}","git_username":"weiwei.qw",'
        '"nested":{"access_key":"AKSECRETVALUE","plain":"visible"}}'
    )
    client = type("C", (), {"get_config": staticmethod(lambda **_: body)})()
    with (
        patch.object(store, "_domain_yaml_section", return_value=None),
        patch.object(store, "_get_remote_config_client", return_value=client),
        caplog.at_level(logging.INFO, logger="benchmark.settings.remote_config_store"),
    ):
        result = store.fetch_remote_settings(
            "agent-eval.benchmark.customization", "agent-eval.service.group"
        )

    # The caller still receives the full config (redaction applies only to logs)
    assert result["token"] == secret
    joined = "\n".join(r.message for r in caplog.records)
    assert secret not in joined
    assert "AKSECRETVALUE" not in joined
    assert "***" in joined
    # Non-sensitive fields stay diagnosable
    assert "agent_eval_prompt_skill" in joined
    assert "visible" in joined


# ---------------------------------------------------------------------------
# Redaction contract: a token never enters the log (parsed redaction / placeholder fallback)
# ---------------------------------------------------------------------------


def test_mask_sensitive_text_string_value() -> None:
    """String values: redact after parsing; the sensitive value does not appear, and the key name is kept."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = '{"api_key": "sk-123", "name": "example"}'
    masked = _mask_sensitive_text(raw)
    assert "sk-123" not in masked
    assert "api_key" in masked
    assert '"***"' in masked
    assert "example" in masked


def test_mask_sensitive_text_non_string_values() -> None:
    """Non-string values (numbers / arrays / nested): the parse path redacts uniformly, and clear text no longer leaks."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = '{"api_key": 123456, "api_keys": ["sk-1", "sk-2"], "nested": {"password": {"k": "v"}}}'
    masked = _mask_sensitive_text(raw)
    assert "123456" not in masked
    assert "sk-1" not in masked
    assert "sk-2" not in masked
    # Nested sensitive key: its value (the whole inner object) is redacted as a whole; the outer structure is kept
    assert '"nested"' in masked
    assert '"password"' in masked


def test_mask_sensitive_text_escaped_quote_value() -> None:
    """A value containing escaped quotes: the parse path redacts it fully, and a regex no longer truncates and leaks the tail."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = '{"secret": "a\\"b", "plain": "ok"}'
    masked = _mask_sensitive_text(raw)
    assert 'a\\"b' not in masked
    assert '"***"' in masked
    assert "ok" in masked


def test_mask_sensitive_text_no_sensitive_keys_returns_verbatim() -> None:
    """No sensitive keys: the original text is returned verbatim (the existing full diagnostic-log definition is unchanged)."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = '{"version":"1.0","configs":{"p1":[{"evaluatorName":"example"}]}}'
    assert _mask_sensitive_text(raw) == raw


def test_mask_sensitive_text_unparsed_with_sensitive_key_withheld() -> None:
    """Illegal JSON that contains a sensitive key name: log only a length placeholder, not the original text."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = '{"secret": "a\\"b'  # Truncated JSON; a regex replace would miss the tail
    masked = _mask_sensitive_text(raw)
    assert raw not in masked
    assert "sk" not in masked
    assert masked.startswith("<unparsed config with sensitive keys")
    assert str(len(raw)) in masked


def test_mask_sensitive_text_unparsed_without_sensitive_keys_verbatim() -> None:
    """Illegal JSON with no sensitive key: log the original text (a non-JSON config is not harmed by mistake)."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = "not-json name=example"
    assert _mask_sensitive_text(raw) == raw


def test_mask_sensitive_text_jsonc_comments_stripped_before_parse() -> None:
    """JSONC comments plus a trailing comma: the parse path still redacts."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = '{\n  // comment with token wording\n  "token": "tok-1",\n}'
    masked = _mask_sensitive_text(raw)
    assert "tok-1" not in masked
    assert '"***"' in masked


def test_mask_sensitive_text_http_header_credential_keys() -> None:
    """HTTP-header-style credential keys: Authorization / Bearer / session id / cookie are redacted too."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = (
        '{"Authorization": "Bearer sk-xxx", "x-bearer-token": "bt-1",'
        '"SESSION_ID": "s-9", "Cookie": "session=abc", "credentials": "c-1"}'
    )
    masked = _mask_sensitive_text(raw)
    assert "sk-xxx" not in masked
    assert "bt-1" not in masked
    assert "s-9" not in masked
    assert "session=abc" not in masked
    assert "c-1" not in masked
    assert masked.count("***") == 5


def test_mask_sensitive_text_camel_case_and_joined_credential_keys() -> None:
    """Camel-case and concatenated variants: secretKey / apiToken match after normalization."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = '{"secretKey": "sk-9", "apiToken": "t-9"}'
    masked = _mask_sensitive_text(raw)
    assert "sk-9" not in masked
    assert "t-9" not in masked


def test_mask_sensitive_text_token_like_budget_keys_visible() -> None:
    """Diagnostic keys for token counts and budgets are not harmed by mistake: the value stays readable."""
    from benchmark.settings.remote_config_store import _mask_sensitive_text

    raw = (
        '{"max_tokens": 4096, "total_tokens": 1024, "token_count": 3,'
        '"tokenizer": "qwen", "api_key_env": "AGENT_EVAL_MODEL_API_KEY"}'
    )
    masked = _mask_sensitive_text(raw)
    assert "4096" in masked
    assert "1024" in masked
    assert masked.count("3") >= 1
    assert "qwen" in masked
    # The environment-variable name itself is not a credential and stays diagnosable
    assert "AGENT_EVAL_MODEL_API_KEY" in masked
    assert "***" not in masked


# ---------------------------------------------------------------------------
# clean up the fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache_after_each() -> None:
    """Clear the cache after each test so tests do not pollute each other."""
    yield
    clear_cache()
