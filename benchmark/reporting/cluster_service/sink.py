"""Report sinks persist artifacts locally and optionally publish them to object storage.

LocalReportSink writes the local copy first. CompositeReportSink keeps those paths
when a remote uploader fails. Remote URLs are returned as supplied by ObjectUploader.
OssArtifactUploader supplies configurable signed URLs for private OSS buckets;
StorageClient implementations that supply public URLs retain their own access limits.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from typing import Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

logger = get_logger(__name__)

_LOCAL_REPORT_EXECUTOR = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="cluster-report-local"
)

#: Artifact filenames (the Markdown / JSON / xlsx / HTML set of four).
REPORT_MD = "cluster_report.md"
REPORT_JSON = "report.json"
REPORT_SUMMARY_JSON = "summary.json"
REPORT_XLSX = "eval_summary.xlsx"
REPORT_CASE_EVAL_XLSX = "case_eval_detail.xlsx"
REPORT_HTML = "cluster_report.html"
LATEST_REPORT_JSON = "latest.json"


class ReportSink(Protocol):
    """Artifact publish target. ``publish`` returns {filename: accessible location}."""

    async def publish(self, batch_tag: str, artifacts: dict[str, bytes]) -> dict[str, str]: ...


class ObjectUploader(Protocol):
    """Object-storage upload interface (satisfied by ``StorageClient`` in libs/storage).

    Only the upload method is required: this module does not need download/delete, so a
    test fake implements one method. Returning None means failure (the libs/storage
    degradation convention).
    """

    async def upload(self, object_key: str, data: bytes) -> str | None: ...


class RemoteSink(Protocol):
    """Remote publish target (same shape as :class:`ReportSink`, named separately to mark the composition)."""

    async def publish(self, batch_tag: str, artifacts: dict[str, bytes]) -> dict[str, str]: ...

    async def publish_latest(self, evaluator_name: str, payload: bytes) -> str | None: ...


class LocalReportSink:
    """Local filesystem publish — always available, the default and the fallback when OSS fails."""

    def __init__(self, root: str) -> None:
        self._root = root

    @property
    def root(self) -> str:
        return self._root

    async def publish(self, batch_tag: str, artifacts: dict[str, bytes]) -> dict[str, str]:
        context = copy_context()
        return await asyncio.get_running_loop().run_in_executor(
            _LOCAL_REPORT_EXECUTOR, context.run, self._write_artifacts, batch_tag, artifacts
        )

    def _write_artifacts(self, batch_tag: str, artifacts: dict[str, bytes]) -> dict[str, str]:
        target = Path(self._root) / batch_tag
        target.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for name, payload in artifacts.items():
            path = target / name
            path.write_bytes(payload)
            written[name] = str(path)
        logger.info(
            "cluster_report.local_published batch=%s dir=%s files=%d",
            batch_tag,
            target,
            len(written),
        )
        return written


class OssReportSink:
    """OSS publish — a single file upload failure is logged and skipped, and does not affect the other files or the overall flow."""

    def __init__(self, client: ObjectUploader, *, key_prefix: str = "") -> None:
        self._client = client
        self._key_prefix = key_prefix.strip("/")

    def object_key(self, batch_tag: str, name: str) -> str:
        parts = [part for part in (self._key_prefix, batch_tag, name) if part]
        return "/".join(parts)

    def latest_object_key(self, evaluator_name: str) -> str:
        return self.object_key(evaluator_name, LATEST_REPORT_JSON)

    async def publish(self, batch_tag: str, artifacts: dict[str, bytes]) -> dict[str, str]:
        published: dict[str, str] = {}
        for name, payload in artifacts.items():
            key = self.object_key(batch_tag, name)
            try:
                url = await self._client.upload(key, payload)
            except Exception as exc:
                # Log only the exception type name, not the exception body (credentials may appear in SDK exceptions)
                logger.warning(
                    "cluster_report.oss_upload_failed key=%s err=%s",
                    key,
                    type(exc).__name__,
                )
                continue
            if not url:
                logger.warning("cluster_report.oss_upload_empty_url key=%s", key)
                continue
            published[name] = url
        return published

    async def publish_latest(self, evaluator_name: str, payload: bytes) -> str | None:
        """Upload the fixed latest pointer; failure does not affect artifacts of a finished batch."""

        key = self.latest_object_key(evaluator_name)
        try:
            return await self._client.upload(key, payload)
        except Exception as exc:
            logger.warning(
                "cluster_report.latest_upload_failed key=%s err=%s",
                key,
                type(exc).__name__,
            )
            return None


class CompositeReportSink:
    """Local first, then remote — **the order must not be reversed**.

    Writing locally is the only guarantee that the artifact exists. Uploading to OSS
    before the local write would let a slow or timed-out OSS stretch the whole report
    flow, and a remote failure might leave nothing written locally either.
    """

    def __init__(self, local: LocalReportSink, remote: RemoteSink | None = None) -> None:
        self._local = local
        self._remote = remote

    async def publish(self, batch_tag: str, artifacts: dict[str, bytes]) -> dict[str, str]:
        locations = await self._local.publish(batch_tag, artifacts)
        if self._remote is None:
            return locations
        try:
            remote_locations = await self._remote.publish(batch_tag, artifacts)
        except Exception as exc:
            logger.warning(
                "cluster_report.remote_sink_failed batch=%s err=%s",
                batch_tag,
                type(exc).__name__,
            )
            return locations
        # A remote URL overwrites the same-named local path (the remote one is shareable); the local path is kept under local_* for traceability
        merged = {f"local_{name}": path for name, path in locations.items()}
        merged.update(remote_locations)
        for name, path in locations.items():
            merged.setdefault(name, path)
        # Summary signal: remote < total means some files fell back to local (OSS upload partially or fully failed),
        # Per-file failure reasons are in the oss_upload_failed log of OssArtifactUploader.
        logger.info(
            "cluster_report.publish_summary batch=%s total=%d remote=%d local_fallback=%d",
            batch_tag,
            len(artifacts),
            len(remote_locations),
            len(artifacts) - len(remote_locations),
        )
        return merged

    async def publish_remote_only(
        self, batch_tag: str, artifacts: dict[str, bytes]
    ) -> dict[str, str]:
        """Upload only to remote object storage; do not create local artifacts or the latest pointer."""

        if self._remote is None:
            raise RuntimeError("cluster report OSS sink is not configured")
        locations = await self._remote.publish(batch_tag, artifacts)
        if len(locations) != len(artifacts):
            raise RuntimeError("cluster report OSS preview upload is incomplete")
        return locations

    async def publish_latest(self, evaluator_name: str, payload: bytes) -> str | None:
        """Publish latest only when the remote sink is available, so a local path is not mistaken for a machine-consumed address."""

        if self._remote is None:
            return None
        try:
            return await self._remote.publish_latest(evaluator_name, payload)
        except Exception as exc:
            logger.warning(
                "cluster_report.remote_latest_failed evaluator=%s err=%s",
                evaluator_name,
                type(exc).__name__,
            )
            return None


def build_sink(
    *,
    local_root: str,
    oss_client: ObjectUploader | None = None,
    oss_key_prefix: str = "",
) -> CompositeReportSink:
    """Assemble the sink from config. When ``oss_client`` is None, degrade to local only."""

    local = LocalReportSink(local_root)
    remote = (
        OssReportSink(oss_client, key_prefix=oss_key_prefix) if oss_client is not None else None
    )
    return CompositeReportSink(local, remote)


def default_local_root() -> str:
    """Default artifact root: ``tmp/cluster_reports`` under the service directory.

    This file lives in ``app/services/cluster_report/``, so the service root is ``parents[3]``.
    A custom path goes through the config layer (``Settings.cluster_report_local_root``);
    this function does not read environment variables directly.
    """

    return str(Path(__file__).resolve().parents[3] / "tmp" / "cluster_reports")


__all__ = [
    "LATEST_REPORT_JSON",
    "REPORT_CASE_EVAL_XLSX",
    "REPORT_HTML",
    "REPORT_JSON",
    "REPORT_MD",
    "REPORT_SUMMARY_JSON",
    "REPORT_XLSX",
    "CompositeReportSink",
    "LocalReportSink",
    "ObjectUploader",
    "OssReportSink",
    "RemoteSink",
    "ReportSink",
    "build_sink",
    "default_local_root",
]
