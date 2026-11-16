"""Parsed-result cache for the domain layer.

The raw RemoteConfig JSON dict is managed by the TTL cache in
:mod:`benchmark.settings.remote_config_store`. This module adds a **parsed-result
cache** on top so callers do not rebuild domain dataclasses on every call.

Two-layer cache layout::

    fetch_cached_settings()          # layer 1: raw RemoteConfig dict (TTL=300s)
        → get_or_compute()           # layer 2: parsed dataclass list (TTL=300s)
            → parse the raw dict into domain objects

The cache key is ``(data_id, group)``. TTL matches the RemoteConfig cache
(:data:`CACHE_TTL_SEC`). Thread-safe; :class:`cachetools.TTLCache` expires entries.

Usage::

    from benchmark.domain._cache import get_or_compute, clear_parsed_cache

    def fetch_my_configs(*, data_id, group, use_cache=True):
        def _parse():
            raw = fetch_cached_settings(data_id, group, use_cache=use_cache)
            return [MyConfig.from_dict(item) for item in raw.get("items", [])]
        return get_or_compute((data_id, group), _parse, use_cache=use_cache)

Conventions:
  - Cached objects are **immutable**; callers must not mutate the returned list or objects.
  - Tests should call :func:`clear_parsed_cache` in fixtures to avoid cross-case pollution.
  - On config change, call :func:`clear_parsed_cache` and :func:`clear_cache` to force a refresh.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from benchmark.libs.agenthost.observability.logger import get_logger
from cachetools import TTLCache

from benchmark.constant import CACHE_TTL_SEC

logger = get_logger(__name__)

T = TypeVar("T")

#: Parsed-result cache (bucketed by (data_id, group)).
_parsed_cache: TTLCache[tuple[str, str], object] = TTLCache(maxsize=256, ttl=CACHE_TTL_SEC)
_parsed_cache_lock = threading.RLock()


def get_or_compute(
    key: tuple[str, str],
    compute: Callable[[], T],
    *,
    use_cache: bool = True,
) -> T:
    """Return a cached parsed result; on a miss, call ``compute`` and store it.

    Args:
        key: Cache key, usually ``(data_id, group)``.
        compute: Function used on a miss (fetch from RemoteConfig and parse).
        use_cache: ``False`` skips the cache (no read, no write) and forces a recompute.
    """
    if not use_cache:
        return compute()
    with _parsed_cache_lock:
        cached = _parsed_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
    result = compute()
    with _parsed_cache_lock:
        _parsed_cache[key] = result
    return result


def clear_parsed_cache() -> None:
    """Clear every parsed-result cache entry; used on config change and in test fixtures."""
    with _parsed_cache_lock:
        _parsed_cache.clear()


# ---------------------------------------------------------------------------
# fetch_and_parse_list — RemoteConfig fetch + list parse + cache (DRY helper)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchListSpec(Generic[T]):
    """Behavior parameters for ``fetch_and_parse_list`` (FUNC-PARAMS-001 fold).

    ``fetcher`` loads the raw RemoteConfig dict (``fetch_cached_settings`` and the like).
    Signature: ``(data_id, group, *, use_cache) -> dict``.
    """

    list_key: str
    parse_item: Callable[[dict[str, Any]], T]
    fetcher: Callable[..., dict[str, Any]]
    use_cache: bool = True


def fetch_and_parse_list(
    data_id: str,
    group: str,
    spec: FetchListSpec[T],
) -> list[T]:
    """Fetch RemoteConfig config, extract list by key, parse each item with error tolerance.

    Encapsulates the common pattern across domain modules:

    1. Fetch raw dict from RemoteConfig (with TTL cache)
    2. Extract list by ``spec.list_key``
    3. Parse each item via ``spec.parse_item``, skipping invalid entries
    4. Cache the parsed result via :func:`get_or_compute`

    Usage::

        return fetch_and_parse_list(
            data_id, group,
            FetchListSpec(
                list_key="agents",
                parse_item=AgentConfig.from_dict,
                fetcher=fetch_cached_settings,
                use_cache=use_cache,
            ),
        )
    """

    list_key = spec.list_key
    parse_item = spec.parse_item
    fetcher = spec.fetcher
    use_cache = spec.use_cache

    def _parse() -> list[T]:
        raw = fetcher(data_id, group, use_cache=use_cache)
        raw_list = raw.get(list_key)
        if raw_list is None:
            return []
        if not isinstance(raw_list, list):
            logger.warning(
                "RemoteConfig %s '%s' 字段非 list: %s",
                data_id,
                list_key,
                type(raw_list).__name__,
            )
            return []
        result: list[T] = []
        for i, item in enumerate(raw_list):
            if not isinstance(item, dict):
                logger.warning(
                    "RemoteConfig %s '%s' 第 %d 项非 dict，跳过",
                    data_id,
                    list_key,
                    i,
                )
                continue
            try:
                result.append(parse_item(item))
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "RemoteConfig %s '%s' 第 %d 项解析失败: %s",
                    data_id,
                    list_key,
                    i,
                    exc,
                )
        return result

    return get_or_compute((data_id, group), _parse, use_cache=use_cache)


__all__ = ["FetchListSpec", "clear_parsed_cache", "fetch_and_parse_list", "get_or_compute"]
