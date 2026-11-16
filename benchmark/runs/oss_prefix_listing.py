"""List recent OSS entries under a dataset prefix ("latest 5 OSS rows" on the evaluation page).

Read-only: do not download object bodies. List directories / objects by prefix inside the
bucket configured in RemoteConfig.

Date macros (``${yyyymmdd±n}``) are stripped before listing: a macro means "resolve the
day at trigger time", while the list shows date partitions that **actually exist** in OSS,
so the real directory dates win.
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.report_storage_config import fetch_benchmark_storage_config
from benchmark.runs.common.oss_date_macro import DATE_MACRO

try:
    from benchmark.libs.storage import osscompat as oss2
except ImportError:  # pragma: no cover - depends on the deployment environment
    oss2 = None  # type: ignore[assignment]

try:
    # normandy-credential-sdk: installed only in the production deploy; None when the package is missing locally / in
    # CI.
    from benchmark.libs.credential import factory as _cred_factory
    from benchmark.libs.credential import names as _cred_names
except ImportError:  # pragma: no cover - depends on the deployment environment
    _cred_factory = _cred_names = None  # type: ignore[assignment]

logger = get_logger("agent_eval.benchmark.oss_prefix_listing")

#: Date-partition directory name (yyyymmdd)
_DATE_DIR_RE = re.compile(r"^\d{8}$")
#: Cap for one list call (shared by the directory and object modes)
_MAX_LIST_KEYS = 1000
#: Dedicated explicit thread pool for OSS listing (synchronous oss2): process-wide, fixed capacity.
#: Do not use asyncio.to_thread / run_in_executor(None, ...) (DEFAULT-EXECUTOR-001/002).
_OSS_LIST_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="oss-list")


class _OssBucket(Protocol):
    """Minimal list protocol for oss2.Bucket (only list_objects is used)."""

    def list_objects(self, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class OssPrefixEntry:
    """One OSS entry: ``path`` is an OSS prefix that can be shown as-is (bucket not included)."""

    path: str
    date: str
    modified_at: str | None


def strip_date_macros(prefix: str) -> str:
    """Strip date macros and normalize to a bare key path (drop scheme / bucket / surrounding slashes)."""
    text = (prefix or "").strip()
    if text.startswith("oss://"):
        without_scheme = text[len("oss://") :]
        slash = without_scheme.find("/")
        text = without_scheme[slash + 1 :] if slash >= 0 else ""
    text = DATE_MACRO.sub("", text)
    # A macro can leave consecutive slashes (for example ``a/${yyyymmdd-1}/b`` → ``a//b``)
    return re.sub(r"/{2,}", "/", text).strip("/")


def _storage_mode() -> str:
    return (fetch_benchmark_storage_config().credential_mode or "provider").strip() or "provider"


def _build_bucket() -> tuple[_OssBucket, str]:
    """Build an OSS Bucket (RemoteConfig bucket / credentials from the Normandy resource binding).

    Listing is a read-only console capability and only supports the production
    ``credential_mode=provider``. The local/CI ``env`` static-credential mode is not enabled
    (so long-lived AKs in environment variables are not read).
    """
    if oss2 is None:
        raise RuntimeError("oss2 is not installed")
    if _cred_factory is None or _cred_names is None:
        raise RuntimeError("normandy credential sdk is not installed")

    cfg = fetch_benchmark_storage_config()
    bucket_name = (cfg.bucket or "").strip()
    endpoint = (cfg.upload_endpoint or "").strip()
    if not bucket_name or not endpoint:
        raise RuntimeError(
            "RemoteConfig agent-eval.benchmark.oss incomplete (need bucket / upload_endpoint)",
        )
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    if _storage_mode() != "provider":
        raise RuntimeError("oss listing requires credential_mode=provider (Normandy)")

    provider = _cred_factory.get_default_credential_provider()
    credential: Any = provider.get_credential(
        _cred_names.of_aliyun_oss_bucket_name(bucket_name),
    )
    ak = getattr(credential, "access_key_id", "") or ""
    sk = getattr(credential, "access_key_secret", "") or ""
    token = getattr(credential, "security_token", None) or None
    if not ak or not sk:
        raise RuntimeError(f"Normandy returned empty credential bucket={bucket_name}")
    auth = oss2.StsAuth(ak, sk, token) if token else oss2.Auth(ak, sk)
    return oss2.Bucket(auth, endpoint, bucket_name), bucket_name


def _iso(value: object) -> str | None:
    """OSS ``last_modified`` is epoch seconds (int); datetime is also accepted."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, int | float) and value > 0:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    return None


def _date_entries(
    bucket: _OssBucket,
    list_prefix: str,
    base: str,
    limit: int,
) -> list[OssPrefixEntry]:
    """Prefer listing "date-partition directories" (real OSS dates).

    On oss2 ``ListObjectsResult`` the common-prefix attribute is ``prefix_list``
    (not ``prefixes``); the wrong name raises AttributeError and becomes a 503.
    """
    result = bucket.list_objects(prefix=list_prefix, delimiter="/", max_keys=_MAX_LIST_KEYS)
    prefixes = list(getattr(result, "prefix_list", None) or [])
    dates = sorted(
        {
            name
            for prefix in prefixes
            for name in [str(prefix).rstrip("/").rsplit("/", 1)[-1]]
            if _DATE_DIR_RE.match(name)
        },
        reverse=True,
    )
    paths = [f"{base}/{date}" if base else date for date in dates[:limit]]
    return [
        OssPrefixEntry(path=path, date=path.rsplit("/", 1)[-1], modified_at=None) for path in paths
    ]


def _object_entries(bucket: _OssBucket, list_prefix: str, limit: int) -> list[OssPrefixEntry]:
    """When there is no date directory, fall back to listing objects (LastModified descending).

    ``OssPrefixEntry.date`` is left empty: the object key itself no longer has a date segment to substitute.
    """
    result = bucket.list_objects(prefix=list_prefix, max_keys=_MAX_LIST_KEYS)
    objects = sorted(
        list(getattr(result, "object_list", None) or []),
        key=lambda item: getattr(item, "last_modified", 0) or 0,
        reverse=True,
    )[:limit]
    return [
        OssPrefixEntry(
            path=str(item.key),
            date="",
            modified_at=_iso(getattr(item, "last_modified", None)),
        )
        for item in objects
    ]


def _list_sync(prefix: str, limit: int) -> list[OssPrefixEntry]:
    bucket, bucket_name = _build_bucket()
    base = strip_date_macros(prefix)
    list_prefix = f"{base}/" if base else ""
    logger.info(
        "oss_prefix_listing.start bucket=%s prefix=%r base=%r limit=%d",
        bucket_name,
        prefix,
        base,
        limit,
    )
    entries = _date_entries(bucket, list_prefix, base, limit)
    if not entries:
        entries = _object_entries(bucket, list_prefix, limit)
    return entries


async def list_recent_oss_entries(prefix: str, *, limit: int = 5) -> list[OssPrefixEntry]:
    """List the latest ``limit`` OSS date partitions under the prefix (fall back to objects if there is no date directory).

    The synchronous oss2 call runs on this module's own ``_OSS_LIST_EXECUTOR`` (the caller's
    ContextVar is kept). It does not use the event-loop default thread pool.
    """
    ctx = copy_context()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_OSS_LIST_EXECUTOR, ctx.run, _list_sync, prefix, limit)


__all__ = ["OssPrefixEntry", "list_recent_oss_entries", "strip_date_macros"]
