"""RemoteConfig loader — shared module, zero import side effects.

Provides the RemoteConfig client singleton, a TTL config cache, and JSON fetch plus parse.
Any module that reads RemoteConfig can reuse this one.

Contract:
  - :func:`fetch_remote_settings` fetches and parses config and returns a JSON dict.
  - :func:`fetch_cached_settings` is the cached fetch. A hit returns a shallow copy.
  - :func:`reset_remote_config_client` clears the client singleton (credential rotation / tests).
  - :func:`clear_cache` clears the config cache (config change / tests).

Runtime errors are logged at WARN and do not interrupt the caller, so local development
works without a RemoteConfig network.

Usage::

    from benchmark.settings.remote_config_store import fetch_cached_settings

    settings = fetch_cached_settings("config.owner.mapping", "group.owner_mapping")
"""

import json
import re
import threading
import time
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger
from cachetools import TTLCache

from benchmark.constant import CACHE_TTL_SEC

#: Regexes: line comments (// ...) and block comments (/* ... */) inside JSON.
#: They do not handle // inside a string literal. RemoteConfig string values usually
#: do not contain //. For a stricter parse, add json5 or commentjson later.
_LINE_COMMENT_RE = re.compile(r'"(?:\\.|[^"\\])*"|//[^\n]*', re.DOTALL)
_BLOCK_COMMENT_RE = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/', re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_jsonc_comments(text: str) -> str:
    """Strip JSONC-style comments and trailing commas so the text is legal JSON.

    The RemoteConfig center lets users write ``//`` line comments and ``/* */`` block
    comments for readability, but ``json.loads`` does not accept comments. This function
    does a light cleanup before parse:

    1. Remove ``//`` line comments (not ``//`` inside a string literal)
    2. Remove ``/* */`` block comments
    3. Remove trailing commas (``,}`` / ``,]``)

    For a stricter JSONC parse, add ``json5`` or ``commentjson`` later.
    """
# Two steps: block comments first, then line comments, so // inside a block is not mishandled.
# The `"(?:\\.|[^"\\])*"` alternative skips string literals.
    text = _BLOCK_COMMENT_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        text,
    )
    text = _LINE_COMMENT_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        text,
    )
    # Drop trailing commas (JSONC allows them, standard JSON does not).
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text


#: Sensitive key names (exact match after normalize): the credential itself and common
#: concatenated variants (secretKey, and so on). An exact set avoids substring false
#: positives: max_tokens / token_count / tokenizer are not in the set.
_SENSITIVE_KEYS_EXACT = frozenset(
    {
        "token", "secret", "password", "passwd", "pwd",
        "auth", "authorization", "bearer",
        "credential", "credentials",
        "session_id", "sessionid", "cookie", "cookies", "set_cookie",
        "api_key", "access_key", "private_key", "secret_key",
        "apikey", "accesskey", "privatekey", "secretkey",
        "session_token", "access_token", "refresh_token", "id_token",
        "bearer_token",
    }
)
#: Sensitive key suffixes (endswith after normalize): compound credential keys such as
#: access_token / client_secret / api_keys. ``_token`` does not match ``_tokens``, so max_tokens is exempt.
_SENSITIVE_KEY_SUFFIXES = (
    "_key", "_keys", "_token", "_secret", "_secrets",
    "_password", "_passwd", "_credential", "_credentials",
    "_authorization", "_session_id", "_cookie",
)
#: Bare suffixes of concatenated compound keys (no separator): userpassword / proxyauthorization.
#: A bare ``token`` does not match max_tokens (tokens) or tokenizer.
_SENSITIVE_KEY_BARE_SUFFIXES = (
    "token", "secret", "password", "passwd",
    "authorization", "sessionid", "cookie", "credential",
)
#: Fast pre-check for sensitive words in non-JSON text (conservative: a hit logs a placeholder, not plaintext).
_SENSITIVE_HINT_RE = re.compile(
    r"token|secret|password|passwd|pwd|access_?key|private_?key|api_?key"
    r"|authorization|bearer|credential|session_?id|cookie",
    re.IGNORECASE,
)


def _is_sensitive_key(key: object) -> bool:
    """Whether a key name is sensitive, compared after normalize (strip quotes / lower / hyphen to underscore).

    Three levels: exact set, then a separator suffix, then a concatenated bare suffix.
    No substring match, so diagnostic keys such as max_tokens / token_count / tokenizer are not masked.
    """
    k = str(key).strip().strip("'\"").lower().replace("-", "_").replace(" ", "_")
    if k in _SENSITIVE_KEYS_EXACT:
        return True
    return k.endswith(_SENSITIVE_KEY_SUFFIXES) or k.endswith(_SENSITIVE_KEY_BARE_SUFFIXES)


def _mask_sensitive_obj(value: Any) -> Any:  # noqa: ANN401 — any JSON value
    """Recursively mask parsed JSON: sensitive values become ``***`` (keys stay, for debugging)."""
    if isinstance(value, dict):
        return {
            key: "***" if _is_sensitive_key(key) else _mask_sensitive_obj(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_mask_sensitive_obj(item) for item in value]
    return value


def _mask_sensitive_text(raw_text: str) -> str:
    """Mask config source text: parse and remask when possible, otherwise log a placeholder.

    Prefer ``json.loads`` and then :func:`_mask_sensitive_obj`, then re-serialize. That covers
    non-string values (numbers / arrays / nested objects) and escaped quotes that a regex
    would miss or truncate. When there is no sensitive key, the original text is returned
    verbatim (existing full debug logs stay the same). If parse fails but the text contains
    a sensitive key name (masking cannot be confirmed, and a regex replace on illegal JSON
    can truncate and leak plaintext), log only a length placeholder. Prefer logging less
    source over logging a token in plaintext.
    """
    try:
        data = json.loads(_strip_jsonc_comments(raw_text))
    except Exception:  # noqa: BLE001 — do not log the raw text when it is not JSON
        if _SENSITIVE_HINT_RE.search(raw_text):
            return f"<unparsed config with sensitive keys, {len(raw_text)} chars withheld>"
        return raw_text
    if isinstance(data, (dict, list)):
        masked_obj = _mask_sensitive_obj(data)
        if masked_obj != data:
            return json.dumps(masked_obj, ensure_ascii=False, default=str)
    return raw_text


try:
    from benchmark.libs.configstore import RemoteConfigClient
except Exception:  # pragma: no cover — environment-dependent; the local machine has no remote_config-client SDK
    RemoteConfigClient = None  # type: ignore[assignment]

logger = get_logger(__name__)

_REQUIRED_CLIENT_ATTRS = (
    "current_server",
    "default_timeout",
    "failover_base",
    "no_snapshot",
    "pulling_lock",
    "server_root_url",
    "snapshot_base",
    "tenant",
)


def _client_is_usable(client: Any) -> bool:  # noqa: ANN401 — RemoteConfigClient is an optional dynamic dependency
    """Whether the RemoteConfigClient instance finished initializing."""
    return all(hasattr(client, attr) for attr in _REQUIRED_CLIENT_ATTRS)


# ---------------------------------------------------------------------------
# RemoteConfig client singleton.
# ---------------------------------------------------------------------------

_remote_config_client: Any = None
_remote_config_client_lock = threading.Lock()


def _get_remote_config_client() -> Any:  # noqa: ANN401 — same as above; a dynamic dependency cannot be annotated statically
    """Return the singleton ``RemoteConfigClient``. ``None`` when init fails.

    A double-checked lock keeps it thread-safe. Credential rotation and tests can reset
    it with :func:`reset_remote_config_client`.
    """
    global _remote_config_client  # noqa: PLW0603 — standard module-level singleton with a double-checked lock

    if _remote_config_client is not None:
        return _remote_config_client
    with _remote_config_client_lock:
        if _remote_config_client is not None:
            return _remote_config_client
        if RemoteConfigClient is None:
            logger.warning("remote_config_client 未安装，RemoteConfig 配置将返回空 dict")
            _remote_config_client = None
            return _remote_config_client
        try:
            client = RemoteConfigClient()
            if not _client_is_usable(client):
                logger.warning("RemoteConfigClient 部分初始化，缺少必要属性，RemoteConfig 配置将返回空 dict")
                _remote_config_client = None
            else:
                _remote_config_client = client
        except Exception as exc:
            logger.warning("RemoteConfigClient 初始化失败，RemoteConfig 配置将返回空 dict: %s", exc)
            _remote_config_client = None
        return _remote_config_client


def reset_remote_config_client() -> None:
    """Clear the client singleton. Used for credential rotation and test fixtures. The next access re-inits."""
    global _remote_config_client  # noqa: PLW0603 — standard module-level singleton with a double-checked lock

    with _remote_config_client_lock:
        _remote_config_client = None
    clear_cache()


# ---------------------------------------------------------------------------
# Config cache (bucketed by (data_id, group); cachetools.TTLCache expires entries).
# ---------------------------------------------------------------------------

#: Cached RemoteConfig payloads (bucketed by (data_id, group) so modules do not share one source).
#: cachetools.TTLCache. Each caller ``ttl`` gets its own TTLCache, so a short TTL
#: (for example service_switches at 10s) can coexist with the default 300s.
_CACHE_MAXSIZE = 256
_settings_caches: dict[int, TTLCache[tuple[str, str], dict[str, Any]]] = {}
_settings_cache_lock = threading.RLock()


def _cache_timer() -> float:
    """TTLCache clock. Goes through the module ``time`` so tests can patch ``time.monotonic``."""
    return time.monotonic()


def _get_settings_cache(ttl: int) -> TTLCache[tuple[str, str], dict[str, Any]]:
    """Lazily build a TTLCache for a ttl (must be called under ``_settings_cache_lock``)."""
    cache = _settings_caches.get(ttl)
    if cache is None:
        cache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=ttl, timer=_cache_timer)
        _settings_caches[ttl] = cache
    return cache


def clear_cache() -> None:
    """Clear the RemoteConfig cache (every TTL bucket). Used on config change and in test fixtures."""
    with _settings_cache_lock:
        for cache in _settings_caches.values():
            cache.clear()
        _settings_caches.clear()


# ---------------------------------------------------------------------------
# Config fetch.
# ---------------------------------------------------------------------------


_DOMAIN_SECTION_KEYS = {
    "config.agent.registry": "agent_registry",
    "config.owner.mapping": "owner_mapping",
    "agent-eval.service.switches": "service_switches",
    "agent-eval.eval.plugins": "eval_plugins",
    "agent-eval.benchmark.oss": "object_store",
    "agent-eval.benchmark.streaming": "streaming",
    "agent-eval.benchmark.customization": "customization",
    "agent-eval.benchmark.report-providers": "report_providers",
    "agent-eval.odps.export": "export_jobs",
    "config.agent.eval.notification": "dispatch",
    "agent-eval.benchmark.report-site": "pages",
    "agent-eval.benchmark.trace-daily-report": "trace_report",
    "agent-eval.benchmark.external-daily": "external_daily",
}


def _domain_yaml_section(data_id: str) -> dict[str, Any] | None:
    """Domain settings come from conf/domain.local.yaml, not a remote store."""

    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "conf" / "domain.local.yaml"
    if not path.is_file():
        return None
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sections = document.get("sections") or {}
    key = _DOMAIN_SECTION_KEYS.get(data_id)
    if key is None:
        return None
    raw = sections.get(key)
    if isinstance(raw, dict):
        return dict(raw)
    return None


def fetch_remote_settings(data_id: str, group: str) -> dict[str, Any]:
    """Fetch config text from RemoteConfig and parse it into a JSON dict.

    Returns an empty dict, and does not interrupt the caller, when the client is
    unavailable or the fetch fails. Every call makes a RemoteConfig network request
    and does not use the cache. Use :func:`fetch_cached_settings` when a cache is wanted.

    Debugging: every real fetch logs ``data_id/group``, the raw text length, and the
    text at INFO (values of sensitive keys such as ``token`` are masked by
    :func:`_mask_sensitive_text` and never logged in plaintext). Empty content or an
    unavailable client is raised to WARNING (a silent ``{}`` used to make
    ``No eval config available`` hard to trace).
    """
    loaded = _domain_yaml_section(data_id)
    if loaded is not None:
        return loaded
    client = _get_remote_config_client()
    if client is None:
        logger.warning(
            "RemoteConfig client unavailable, skip get_config: data_id=%s group=%s",
            data_id,
            group,
        )
        return {}
    try:
        content = client.get_config(data_id=data_id, group=group)
    except Exception as exc:
        logger.warning("RemoteConfig get_config 失败 %s/%s: %s", data_id, group, exc)
        return {}

    raw_text = "" if content is None else str(content)
    logger.info(
        "RemoteConfig get_config raw: data_id=%s group=%s content_len=%d content=%s",
        data_id,
        group,
        len(raw_text),
        _mask_sensitive_text(raw_text),
    )

    if not raw_text.strip():
        logger.warning(
            "RemoteConfig get_config returned empty content: data_id=%s group=%s content_repr=%r",
            data_id,
            group,
            content,
        )
        return {}
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
# RemoteConfig may be JSONC (// line comments, /* */ block comments, trailing commas).
# Strip comments and trailing commas, then parse again.
        stripped = _strip_jsonc_comments(raw_text)
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning(
                "RemoteConfig 配置非合法 JSON/JSONC %s/%s: %s content_preview=%.500s",
                data_id,
                group,
                exc,
                _mask_sensitive_text(raw_text),
            )
            return {}
    if not isinstance(data, dict):
        logger.warning(
            "RemoteConfig 配置顶层非 JSON 对象 %s/%s type=%s",
            data_id,
            group,
            type(data).__name__,
        )
        return {}
    try:
        parsed_dump = json.dumps(_mask_sensitive_obj(data), ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001 — debug log; a serialize failure must not abort
        parsed_dump = f"<json.dumps failed: {exc}>"
    logger.info(
        "RemoteConfig get_config parsed: data_id=%s group=%s keys=%s parsed=%s",
        data_id,
        group,
        sorted(str(k) for k in data.keys()),
        parsed_dump,
    )
    return data


def fetch_cached_settings(
    data_id: str,
    group: str,
    *,
    use_cache: bool = True,
    ttl: int = CACHE_TTL_SEC,
) -> dict[str, Any]:
    """TTL-cached RemoteConfig fetch (bucketed by ``(data_id, group)``).

    A cache hit returns a shallow copy so the caller cannot mutate the cache.
    Pass ``use_cache=False`` to force a refresh.
    Each ``(data_id, group)`` is cached separately and does not overwrite another.
    Each ``ttl`` uses its own :class:`cachetools.TTLCache`, so a short TTL expires on its own.

    Args:
        data_id: RemoteConfig data_id.
        group: RemoteConfig group.
        use_cache: Whether to use the cache, default ``True``.
        ttl: Cache TTL in seconds, default :data:`CACHE_TTL_SEC`. ``0`` disables the cache (no write, no read).
    """
    if not use_cache or ttl <= 0:
        return fetch_remote_settings(data_id, group)
    key = (data_id, group)
    with _settings_cache_lock:
        cache = _get_settings_cache(ttl)
        cached = cache.get(key)
        if cached is not None:
            return dict(cached)
    result = fetch_remote_settings(data_id, group)
    with _settings_cache_lock:
        cache = _get_settings_cache(ttl)
        cache[key] = dict(result)
    return result


__all__ = [
    "clear_cache",
    "fetch_cached_settings",
    "fetch_remote_settings",
    "reset_remote_config_client",
]
