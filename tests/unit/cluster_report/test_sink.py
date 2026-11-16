"""T33 unit tests: ReportSink (local / OSS / combined degradation).

Core invariant: **local persistence must happen before the remote write and must not depend on it** — local storage is the only guarantee the artifact exists.
"""

import asyncio
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from benchmark.reporting.cluster_service.sink import (
    LATEST_REPORT_JSON,
    REPORT_HTML,
    REPORT_JSON,
    REPORT_MD,
    REPORT_XLSX,
    CompositeReportSink,
    LocalReportSink,
    OssReportSink,
    build_sink,
    default_local_root,
)

_BATCH = "run-20260729-153000"
_ARTIFACTS = {
    REPORT_MD: b"# report",
    REPORT_JSON: b'{"schema_version": "cluster_report.v2"}',
    REPORT_XLSX: b"PK\x03\x04fake-xlsx",
}


class _FakeOssClient:
    def __init__(
        self,
        *,
        fail_keys: set[str] | None = None,
        empty_keys: set[str] | None = None,
    ) -> None:
        self.uploaded: dict[str, bytes] = {}
        self._fail_keys = fail_keys or set()
        self._empty_keys = empty_keys or set()

    async def upload(self, object_key: str, data: bytes) -> str | None:
        if object_key in self._fail_keys:
            raise RuntimeError("oss boom")
        if object_key in self._empty_keys:
            return None
        self.uploaded[object_key] = data
        return f"https://cdn.example.com/{object_key}"


class TestLocalReportSink:
    async def test_writes_all_artifacts_under_batch_dir(self, tmp_path: Path) -> None:
        """Artifacts land in {root}/{batch_tag}/ (the path spec of Spec AC-14)."""
        sink = LocalReportSink(str(tmp_path))

        locations = await sink.publish(_BATCH, _ARTIFACTS)

        target = tmp_path / _BATCH
        assert target.is_dir()
        assert set(locations) == set(_ARTIFACTS)
        for name, payload in _ARTIFACTS.items():
            assert (target / name).read_bytes() == payload
            assert locations[name] == str(target / name)

    async def test_file_writes_run_off_event_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loop_thread = threading.get_ident()
        write_threads: list[int] = []
        original_write = Path.write_bytes
        loop = asyncio.get_running_loop()
        submit = Mock(wraps=loop.run_in_executor)
        monkeypatch.setattr(loop, "run_in_executor", submit)

        def record_write(path: Path, payload: bytes) -> int:
            write_threads.append(threading.get_ident())
            return original_write(path, payload)

        monkeypatch.setattr(Path, "write_bytes", record_write)
        await LocalReportSink(str(tmp_path)).publish(_BATCH, _ARTIFACTS)

        assert write_threads and all(t != loop_thread for t in write_threads)
        assert submit.call_args.args[0] is not None

    async def test_creates_nested_root(self, tmp_path: Path) -> None:
        sink = LocalReportSink(str(tmp_path / "a" / "b" / "c"))
        await sink.publish(_BATCH, {REPORT_MD: b"x"})
        assert (tmp_path / "a" / "b" / "c" / _BATCH / REPORT_MD).is_file()

    async def test_rewrites_existing_batch_dir(self, tmp_path: Path) -> None:
        """Rerunning the same batch overwrites the old artifacts and does not error (idempotent)."""
        sink = LocalReportSink(str(tmp_path))
        await sink.publish(_BATCH, {REPORT_MD: b"v1"})
        await sink.publish(_BATCH, {REPORT_MD: b"v2"})
        assert (tmp_path / _BATCH / REPORT_MD).read_bytes() == b"v2"

    async def test_empty_artifacts_still_creates_dir(self, tmp_path: Path) -> None:
        locations = await LocalReportSink(str(tmp_path)).publish(_BATCH, {})
        assert locations == {}
        assert (tmp_path / _BATCH).is_dir()


class TestOssReportSink:
    def test_object_key_layout(self) -> None:
        sink = OssReportSink(_FakeOssClient(), key_prefix="eval/cluster")
        assert sink.object_key(_BATCH, REPORT_MD) == f"eval/cluster/{_BATCH}/{REPORT_MD}"

    def test_object_key_without_prefix(self) -> None:
        sink = OssReportSink(_FakeOssClient())
        assert sink.object_key(_BATCH, REPORT_MD) == f"{_BATCH}/{REPORT_MD}"

    def test_object_key_strips_slashes(self) -> None:
        sink = OssReportSink(_FakeOssClient(), key_prefix="/eval/cluster/")
        assert sink.object_key(_BATCH, REPORT_MD) == f"eval/cluster/{_BATCH}/{REPORT_MD}"

    async def test_publishes_latest_under_evaluator_prefix(self) -> None:
        client = _FakeOssClient()
        sink = OssReportSink(client, key_prefix="eval")

        url = await sink.publish_latest("vui_agent", b'{"status":"success"}')

        key = f"eval/vui_agent/{LATEST_REPORT_JSON}"
        assert url == f"https://cdn.example.com/{key}"
        assert client.uploaded[key] == b'{"status":"success"}'

    async def test_uploads_and_returns_urls(self) -> None:
        client = _FakeOssClient()
        sink = OssReportSink(client, key_prefix="eval")

        locations = await sink.publish(_BATCH, _ARTIFACTS)

        assert set(locations) == set(_ARTIFACTS)
        assert locations[REPORT_MD] == f"https://cdn.example.com/eval/{_BATCH}/{REPORT_MD}"
        assert client.uploaded[f"eval/{_BATCH}/{REPORT_XLSX}"] == _ARTIFACTS[REPORT_XLSX]

    async def test_single_failure_does_not_block_others(self) -> None:
        """A single-file failure skips only that file (xlsx is large and times out easily, and must not take down md/json)."""
        client = _FakeOssClient(fail_keys={f"{_BATCH}/{REPORT_XLSX}"})

        locations = await OssReportSink(client).publish(_BATCH, _ARTIFACTS)

        assert REPORT_XLSX not in locations
        assert REPORT_MD in locations and REPORT_JSON in locations

    async def test_empty_url_treated_as_failure(self) -> None:
        """A provider returning None means the upload failed (the degradation convention of libs/storage)."""
        client = _FakeOssClient(empty_keys={f"{_BATCH}/{REPORT_MD}"})

        locations = await OssReportSink(client).publish(_BATCH, _ARTIFACTS)

        assert REPORT_MD not in locations
        assert REPORT_JSON in locations

    async def test_all_failures_return_empty(self) -> None:
        client = _FakeOssClient(fail_keys={f"{_BATCH}/{name}" for name in _ARTIFACTS})
        assert await OssReportSink(client).publish(_BATCH, _ARTIFACTS) == {}


class TestCompositeReportSink:
    async def test_local_written_before_remote(self, tmp_path: Path) -> None:
        """Key invariant: when uploading remotely, the local copy must already be on disk."""
        local_root = tmp_path / "reports"
        seen_local_exists: list[bool] = []

        class _CheckingRemote:
            async def publish(self, batch_tag: str, artifacts: dict) -> dict:
                seen_local_exists.append((local_root / batch_tag / REPORT_MD).is_file())
                return {REPORT_MD: "https://cdn/x"}

        sink = CompositeReportSink(LocalReportSink(str(local_root)), _CheckingRemote())
        await sink.publish(_BATCH, {REPORT_MD: b"x"})

        assert seen_local_exists == [True]

    async def test_remote_only_upload_does_not_write_local_files(self, tmp_path: Path) -> None:
        client = _FakeOssClient()
        sink = CompositeReportSink(
            LocalReportSink(str(tmp_path)),
            OssReportSink(client, key_prefix="eval"),
        )

        locations = await sink.publish_remote_only(_BATCH, {REPORT_HTML: b"<html></html>"})

        assert locations == {REPORT_HTML: f"https://cdn.example.com/eval/{_BATCH}/{REPORT_HTML}"}
        assert client.uploaded[f"eval/{_BATCH}/{REPORT_HTML}"] == b"<html></html>"
        assert not (tmp_path / _BATCH).exists()

    async def test_remote_only_upload_requires_configured_oss(self, tmp_path: Path) -> None:
        sink = CompositeReportSink(LocalReportSink(str(tmp_path)))

        with pytest.raises(RuntimeError, match="OSS sink is not configured"):
            await sink.publish_remote_only(_BATCH, {REPORT_HTML: b"<html></html>"})

    async def test_remote_urls_override_and_local_kept(self, tmp_path: Path) -> None:
        """The remote URL can be shared outward; the local path is kept under local_* so it stays traceable."""
        client = _FakeOssClient()
        sink = CompositeReportSink(
            LocalReportSink(str(tmp_path)),
            OssReportSink(client, key_prefix="eval"),
        )

        locations = await sink.publish(_BATCH, _ARTIFACTS)

        assert locations[REPORT_MD].startswith("https://cdn.example.com/")
        assert locations[f"local_{REPORT_MD}"].endswith(REPORT_MD)
        assert Path(locations[f"local_{REPORT_MD}"]).is_file()

    async def test_remote_partial_failure_falls_back_to_local_path(self, tmp_path: Path) -> None:
        """A file that did not upload successfully falls back to the local path, with no hole left behind."""
        client = _FakeOssClient(fail_keys={f"{_BATCH}/{REPORT_XLSX}"})
        sink = CompositeReportSink(LocalReportSink(str(tmp_path)), OssReportSink(client))

        locations = await sink.publish(_BATCH, _ARTIFACTS)

        assert locations[REPORT_MD].startswith("https://")
        assert locations[REPORT_XLSX].endswith(REPORT_XLSX)  # local path
        assert Path(locations[REPORT_XLSX]).is_file()

    async def test_remote_sink_exception_does_not_break_publish(self, tmp_path: Path) -> None:
        """The remote sink raising as a whole (for example failing to construct the client) does not affect the returned local artifacts."""

        class _BrokenRemote:
            async def publish(self, batch_tag: str, artifacts: dict) -> dict:
                raise RuntimeError("remote down")

        sink = CompositeReportSink(LocalReportSink(str(tmp_path)), _BrokenRemote())

        locations = await sink.publish(_BATCH, _ARTIFACTS)

        assert set(locations) == set(_ARTIFACTS)
        assert Path(locations[REPORT_MD]).is_file()

    async def test_latest_remote_exception_keeps_local_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import benchmark.reporting.cluster_service.sink as mod

        diagnostic = Mock()
        monkeypatch.setattr(mod, "logger", diagnostic)

        class BrokenLatestRemote:
            async def publish(self, batch_tag: str, artifacts: dict[str, bytes]) -> dict[str, str]:
                return {}

            async def publish_latest(self, evaluator_name: str, payload: bytes) -> str | None:
                raise RuntimeError("FAKE_SECRET_IN_REMOTE_EXCEPTION")

        sink = CompositeReportSink(LocalReportSink(str(tmp_path)), BrokenLatestRemote())
        locations = await sink.publish(_BATCH, {REPORT_MD: b"report"})

        assert await sink.publish_latest("vui_agent", b"{}") is None
        assert Path(locations[REPORT_MD]).read_bytes() == b"report"
        assert diagnostic.warning.call_count == 1
        assert "RuntimeError" in str(diagnostic.warning.call_args)
        assert "FAKE_SECRET_IN_REMOTE_EXCEPTION" not in str(diagnostic.warning.call_args)

    async def test_no_remote_returns_local_only(self, tmp_path: Path) -> None:
        sink = CompositeReportSink(LocalReportSink(str(tmp_path)))
        locations = await sink.publish(_BATCH, {REPORT_MD: b"x"})
        assert set(locations) == {REPORT_MD}


class TestBuildSink:
    async def test_local_only_without_oss_client(self, tmp_path: Path) -> None:
        sink = build_sink(local_root=str(tmp_path))
        locations = await sink.publish(_BATCH, {REPORT_MD: b"x"})
        assert locations[REPORT_MD] == str(tmp_path / _BATCH / REPORT_MD)

    async def test_with_oss_client(self, tmp_path: Path) -> None:
        client = _FakeOssClient()
        sink = build_sink(
            local_root=str(tmp_path),
            oss_client=client,
            oss_key_prefix="eval/cluster",
        )

        locations = await sink.publish(_BATCH, {REPORT_MD: b"x"})

        assert f"eval/cluster/{_BATCH}/{REPORT_MD}" in client.uploaded
        assert locations[REPORT_MD].startswith("https://")


class TestDefaultLocalRoot:
    """The default root directory. A custom path goes through the config layer (``Settings.cluster_report_local_root``).
    This function **does not read environment variables directly** — project rules forbid business code from reading env directly (ruff TID251).
    """

    def test_ignores_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression anchor: an early implementation read os.getenv directly, violating the config-layer constraint, and now ignores env."""
        monkeypatch.setenv("CLUSTER_REPORT_LOCAL_ROOT", "/tmp/should-be-ignored")
        assert default_local_root() != "/tmp/should-be-ignored"
        assert default_local_root().endswith("tmp/cluster_reports")

    def test_settings_field_exists_for_override(self) -> None:
        """Path override is provided by the config layer, and the field must exist."""
        from benchmark.settings.setting import Settings

        assert "cluster_report_local_root" in Settings.model_fields
        assert "cluster_report_project_id" in Settings.model_fields

    def test_default_points_to_service_root(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default root must land under the agent-eval service directory, not inside app/.

        Regression anchor: sink.py lives in app/services/cluster_report/, and the service root is parents[3].
        Writing parents[2] by mistake would put artifacts in app/tmp/ (polluting the code directory).
        """
        monkeypatch.delenv("CLUSTER_REPORT_LOCAL_ROOT", raising=False)

        root = Path(default_local_root())

        assert root.name == "cluster_reports"
        assert root.parent.name == "tmp"
        assert root.parent.parent.name == "agent-eval", root
