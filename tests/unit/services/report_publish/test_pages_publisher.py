"""Unit tests for GitPagesReportPublisher / build_pages_publisher / config fetch."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from benchmark.domain.report_site_config import (
    ReportSiteConfig,
    fetch_report_site_config,
)
from benchmark.reporting.publish.pages_publisher import (
    _API_CONTENT_LIMIT_BYTES,
    GitPagesReportPublisher,
    build_pages_publisher,
)

_FULL_CFG = ReportSiteConfig(
    enabled=True,
    site_base_url="https://eval-reports.io.alibaba-inc.com",
    git_api_base="https://code.alibaba-inc.com",
    repo="amap-llm/eval-report-pages",
    branch="master",
    token="test-token",
    git_username="evalbot",
    timeout_s=5,
)

_HTML = "<!DOCTYPE html><html><head></head><body>report</body></html>"


def _publisher(cfg: ReportSiteConfig = _FULL_CFG) -> GitPagesReportPublisher:
    return GitPagesReportPublisher(cfg)


def _mock_http(
    *,
    get_status: int | None = 404,
    post_status: int | None = None,
    put_status: int | None = None,
    post_text: str = "",
):
    """Build an AsyncClient mock; return (client_cls_mock, client_mock)."""
    client = AsyncMock()
    if get_status is not None:
        client.get = AsyncMock(
            return_value=MagicMock(status_code=get_status, text="", headers={}),
        )
    if post_status is not None:
        client.post = AsyncMock(
            return_value=MagicMock(status_code=post_status, text=post_text),
        )
    if put_status is not None:
        client.put = AsyncMock(return_value=MagicMock(status_code=put_status, text=""))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    cls = MagicMock(return_value=client)
    return cls, client


# ════════════════════════════════════════════════════════════════════════════
# _build_repo_path — path convention and sanitizing
# ════════════════════════════════════════════════════════════════════════════


class TestBuildRepoPath:
    def test_format_follows_month_run_id_convention(self) -> None:
        path = GitPagesReportPublisher._build_repo_path("run-123")
        assert re.fullmatch(r"reports/\d{4}-\d{2}/run-123/report\.html", path)

    def test_run_id_sanitized(self) -> None:
        path = GitPagesReportPublisher._build_repo_path("run id/with:bad*chars")
        assert " " not in path and ":" not in path and "*" not in path
        assert "/run_id_with_bad_chars/" in path

    def test_empty_run_id_falls_back_to_unknown(self) -> None:
        path = GitPagesReportPublisher._build_repo_path("   ")
        assert "/unknown/report.html" in path


# ════════════════════════════════════════════════════════════════════════════
# publish — the push flow
# ════════════════════════════════════════════════════════════════════════════


class TestPublish:
    @pytest.mark.asyncio
    async def test_empty_html_returns_none(self) -> None:
        assert await _publisher().publish("run-1", "") is None

    @pytest.mark.asyncio
    async def test_create_when_file_absent(self) -> None:
        """GET 404 → POST 201 creates it → return the derived URL (deploy-dir=reports, prefix stripped)."""
        cls, client = _mock_http(get_status=404, post_status=201)
        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            cls,
        ):
            url = await _publisher().publish("run-1", _HTML)

        assert url is not None
        assert re.fullmatch(
            r"https://eval-reports\.io\.alibaba-inc\.com/\d{4}-\d{2}/run-1/report\.html",
            url,
        )
        # alicode v3: the write-endpoint URL stops at the collection (joining the path into the URL returns 405)
        post_url = client.post.await_args.args[0]
        assert post_url.endswith("/api/v3/projects/amap-llm%2Feval-report-pages/repository/files")
        body = client.post.await_args.kwargs["json"]
        # The branch field is named branch_name (not GitLab's branch)
        assert body["branch_name"] == "master"
        # The file path is in the body, and it does not start with /
        assert re.fullmatch(r"reports/\d{4}-\d{2}/run-1/report\.html", body["file_path"])
        # base64 is disabled (if the server does not decode it, the repo stores garbage and cannot render):
        # the default is text encoding, content is the raw HTML, consistent with the bytes uploaded straight to OSS
        assert "encoding" not in body
        assert body["content"] == _HTML
        # The token appears only in the request header
        assert client.post.await_args.kwargs["headers"]["PRIVATE-TOKEN"] == "test-token"
        # The probe uses the query parameters file_path + ref (the v3 contract)
        assert client.get.await_args.kwargs["params"] == {
            "file_path": body["file_path"],
            "ref": "master",
        }

    @pytest.mark.asyncio
    async def test_content_not_base64_encoded(self) -> None:
        """Regression: content must be the raw HTML (so a mistaken base64 encoding does not leave Pages unable to render)."""
        cls, client = _mock_http(get_status=404, post_status=201)
        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            cls,
        ):
            await _publisher().publish("run-1", _HTML)

        body = client.post.await_args.kwargs["json"]
        assert body["content"].startswith("<!DOCTYPE html>")
        assert "encoding" not in body

    @pytest.mark.asyncio
    async def test_update_when_file_present(self) -> None:
        """GET 200 → PUT 200 updates successfully (a healer/regenerate replay is idempotent)."""
        cls, client = _mock_http(get_status=200, put_status=200)
        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            cls,
        ):
            url = await _publisher().publish("run-1", _HTML)

        assert url is not None
        client.put.assert_awaited_once()
        assert not client.post.called
        # A PUT update also hits the collection (file_path is in the body)
        put_url = client.put.await_args.args[0]
        assert put_url.endswith("/repository/files")

    @pytest.mark.asyncio
    async def test_post_conflict_falls_back_to_put(self) -> None:
        """A probe race: GET 404 → POST 400 already exists → rotate to PUT 200."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=MagicMock(status_code=404, text=""))
        client.post = AsyncMock(
            return_value=MagicMock(status_code=400, text='{"message":"File already exists"}'),
        )
        client.put = AsyncMock(return_value=MagicMock(status_code=200, text=""))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            MagicMock(return_value=client),
        ):
            with patch(
                "benchmark.reporting.publish.pages_publisher.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep:
                url = await _publisher().publish("run-1", _HTML)

        assert url is not None
        client.put.assert_awaited_once()
        assert mock_sleep.await_count == 1  # Exponential backoff after a conflict

    @pytest.mark.asyncio
    async def test_post_409_conflict_falls_back_to_put(self) -> None:
        """The remote already exists (POST 409 Conflict) → switch to PUT and force an overwrite update."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=MagicMock(status_code=404, text=""))
        client.post = AsyncMock(
            return_value=MagicMock(status_code=409, text='{"message":"Conflict"}'),
        )
        client.put = AsyncMock(return_value=MagicMock(status_code=200, text=""))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            MagicMock(return_value=client),
        ):
            with patch(
                "benchmark.reporting.publish.pages_publisher.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                url = await _publisher().publish("run-1", _HTML)

        assert url is not None
        client.put.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_post_400_without_exist_keyword_still_overwrites(self) -> None:
        """Forced overwrite: a POST 400 no longer depends on matching the error text, and always switches to PUT (the remote-already-exists semantics)."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=MagicMock(status_code=404, text=""))
        client.post = AsyncMock(
            return_value=MagicMock(status_code=400, text='{"message":"Bad Request"}'),
        )
        client.put = AsyncMock(return_value=MagicMock(status_code=200, text=""))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            MagicMock(return_value=client),
        ):
            with patch(
                "benchmark.reporting.publish.pages_publisher.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                url = await _publisher().publish("run-1", _HTML)

        assert url is not None
        client.put.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_conflict_retries_exhausted_returns_none(self) -> None:
        """POST 400 ↔ PUT 404 rotating past the limit → None (degrade in an extreme concurrency case)."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=MagicMock(status_code=404, text=""))
        client.post = AsyncMock(
            return_value=MagicMock(status_code=400, text="already exists"),
        )
        client.put = AsyncMock(return_value=MagicMock(status_code=404, text=""))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            MagicMock(return_value=client),
        ):
            with patch(
                "benchmark.reporting.publish.pages_publisher.asyncio.sleep",
                new_callable=AsyncMock,
            ):
                url = await _publisher().publish("run-1", _HTML)

        assert url is None
        # 3 attempts (POST → PUT → POST)
        assert client.post.await_count == 2
        assert client.put.await_count == 1

    @pytest.mark.asyncio
    async def test_probe_failure_returns_none_without_write(self) -> None:
        """A GET probe 401 (authentication failed) → give up the push, and POST/PUT are called zero times."""
        cls, client = _mock_http(get_status=401)
        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            cls,
        ):
            url = await _publisher().publish("run-1", _HTML)

        assert url is None
        assert not client.post.called
        assert not client.put.called

    @pytest.mark.asyncio
    async def test_probe_302_still_attempts_post_create(self) -> None:
        """GET 302 (an SSO/gateway redirect) → do not give up, and optimistically POST to create it."""
        client = AsyncMock()
        client.get = AsyncMock(
            return_value=MagicMock(
                status_code=302,
                text="",
                headers={"Location": "https://login.example/sso"},
            ),
        )
        client.post = AsyncMock(return_value=MagicMock(status_code=201, text=""))
        client.put = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            MagicMock(return_value=client),
        ):
            url = await _publisher().publish("run-1", _HTML)

        assert url is not None
        client.post.assert_awaited_once()
        assert not client.put.called

    @pytest.mark.asyncio
    async def test_probe_500_still_attempts_post_create(self) -> None:
        """GET 500 → optimistically try POST as if it does not exist (a successful write is enough; a failed write is then recorded as rejected)."""
        cls, client = _mock_http(get_status=500, post_status=201)
        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            cls,
        ):
            url = await _publisher().publish("run-1", _HTML)

        assert url is not None
        client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejected_status_returns_none(self) -> None:
        """POST 401 (the token is invalid) → do not retry, and return None directly."""
        cls, client = _mock_http(get_status=404, post_status=401)
        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            cls,
        ):
            url = await _publisher().publish("run-1", _HTML)

        assert url is None
        assert client.post.await_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_network_exception_returns_none_and_never_raises(self) -> None:
        """A network exception → None (the protocol contract: publish never raises)."""
        client = AsyncMock()
        client.get = AsyncMock(side_effect=ConnectionError("boom"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "benchmark.reporting.publish.pages_publisher.AsyncClient",
            MagicMock(return_value=client),
        ):
            url = await _publisher().publish("run-1", _HTML)

        assert url is None


# ════════════════════════════════════════════════════════════════════════════
# A large file (>20MB) switches to a git https push
# ════════════════════════════════════════════════════════════════════════════


def _big_html() -> str:
    """Build HTML over the Files API gateway limit (all ASCII, so the character count equals the byte count)."""
    return "<!DOCTYPE html>" + "a" * _API_CONTENT_LIMIT_BYTES


class TestGitPushFallback:
    """HTML over _API_CONTENT_LIMIT_BYTES → switch to a git smart-HTTP push."""

    @pytest.mark.asyncio
    async def test_oversized_html_switches_to_git(self) -> None:
        """Over the limit → do not use the Files API (a JSON body would hit the gateway limit), and use a git push."""
        publisher = _publisher()
        publisher._push_file = AsyncMock(return_value=False)
        publisher._push_via_git = AsyncMock(return_value=True)

        url = await publisher.publish("run-1", _big_html())

        assert url is not None
        publisher._push_via_git.assert_awaited_once()
        publisher._push_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_oversized_git_failure_returns_none(self) -> None:
        publisher = _publisher()
        publisher._push_via_git = AsyncMock(return_value=False)

        assert await publisher.publish("run-1", _big_html()) is None

    @pytest.mark.asyncio
    async def test_git_binary_missing_returns_none(self) -> None:
        """The image has no git → degrade gracefully to None (publish never raises)."""
        publisher = _publisher()
        with patch(
            "benchmark.reporting.publish.pages_publisher.shutil.which",
            return_value=None,
        ):
            assert await publisher.publish("run-1", _big_html()) is None

    @pytest.mark.asyncio
    async def test_git_username_missing_fails_fast(self) -> None:
        """git_username is not configured → do not start a clone (a PAT paired with oauth2 would be rejected by authentication)"""
        publisher = _publisher(
            ReportSiteConfig(
                enabled=True,
                site_base_url="https://eval-reports.io.alibaba-inc.com",
                git_api_base="https://code.alibaba-inc.com",
                repo="amap-llm/eval-report-pages",
                token="test-token",
                # git_username is missing
            ),
        )
        publisher._git_push_sync = MagicMock()
        with (
            patch(
                "benchmark.reporting.publish.pages_publisher.shutil.which",
                return_value="/usr/bin/git",
            ),
            patch("benchmark.reporting.publish.pages_publisher.logger") as mock_logger,
        ):
            assert await publisher._push_via_git("reports/x/report.html", "<html/>") is False

        publisher._git_push_sync.assert_not_called()
        assert any(
            "git_credentials_missing" in str(call) for call in mock_logger.warning.call_args_list
        )

    def test_git_push_sync_success(self, tmp_path: Path) -> None:
        """clone → add → commit → push all succeed; the remote URL is in the {domain account}:{token} form."""
        publisher = _publisher()
        calls: list[list[str]] = []

        def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN202
            calls.append(cmd)
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch(
                "benchmark.reporting.publish.pages_publisher.subprocess.run",
                side_effect=_fake_run,
            ),
            patch(
                "benchmark.reporting.publish.pages_publisher.tempfile.mkdtemp",
                return_value=str(tmp_path),
            ),
            patch("benchmark.reporting.publish.pages_publisher.shutil.rmtree") as rm,
        ):
            ok = publisher._git_push_sync("reports/2026-09/run-1/report.html", "<html>r</html>")

        assert ok is True
        assert len(calls) == 4  # clone / add / commit / push
        # The PAT is paired with the domain-account username (an oauth2 username must be paired with an OAuth2 token; the alicode auth matrix)
        assert calls[0][-2] == (
            "https://evalbot:test-token@code.alibaba-inc.com/amap-llm/eval-report-pages.git"
        )
        # The file on disk matches the content that was passed in (what git pushes is the raw HTML)
        written = tmp_path / "reports" / "2026-09" / "run-1" / "report.html"
        assert written.read_text(encoding="utf-8") == "<html>r</html>"
        rm.assert_called_once()  # The temp directory must be cleaned up (including the remote config that contains the token)

    def test_git_push_sync_clone_failure_sanitizes_token(self, tmp_path: Path) -> None:
        """clone fails → False; a token echoed in stderr must be redacted before it enters the log."""
        publisher = _publisher()

        def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN202
            return MagicMock(returncode=1, stdout="", stderr="auth failed for test-token")

        with (
            patch(
                "benchmark.reporting.publish.pages_publisher.subprocess.run",
                side_effect=_fake_run,
            ),
            patch(
                "benchmark.reporting.publish.pages_publisher.tempfile.mkdtemp",
                return_value=str(tmp_path),
            ),
            patch("benchmark.reporting.publish.pages_publisher.shutil.rmtree"),
            patch("benchmark.reporting.publish.pages_publisher.logger") as mock_logger,
        ):
            ok = publisher._git_push_sync("reports/x/report.html", "<html/>")

        assert ok is False
        for call in mock_logger.warning.call_args_list:
            assert "test-token" not in str(call)

    def test_git_push_sync_nothing_to_commit_is_idempotent(self, tmp_path: Path) -> None:
        """Replaying the same content: commit reports nothing to commit → treat it as success (idempotent)."""
        publisher = _publisher()

        def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN202
            if "commit" in cmd:
                return MagicMock(
                    returncode=1,
                    stdout="nothing to commit, working tree clean",
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch(
                "benchmark.reporting.publish.pages_publisher.subprocess.run",
                side_effect=_fake_run,
            ),
            patch(
                "benchmark.reporting.publish.pages_publisher.tempfile.mkdtemp",
                return_value=str(tmp_path),
            ),
            patch("benchmark.reporting.publish.pages_publisher.shutil.rmtree"),
        ):
            ok = publisher._git_push_sync("reports/x/report.html", "<html/>")

        assert ok is True

    def test_git_push_sync_non_fast_forward_recovers_via_rebase(self, tmp_path: Path) -> None:
        """A non-fast-forward push (a concurrency race) → pull --rebase → the retried push succeeds."""
        publisher = _publisher()
        push_calls = 0

        def _fake_run(cmd, **kwargs):  # noqa: ANN001, ANN202
            nonlocal push_calls
            if "push" in cmd:
                push_calls += 1
                if push_calls == 1:
                    return MagicMock(
                        returncode=1, stdout="", stderr="! [rejected] non-fast-forward"
                    )
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch(
                "benchmark.reporting.publish.pages_publisher.subprocess.run",
                side_effect=_fake_run,
            ),
            patch(
                "benchmark.reporting.publish.pages_publisher.tempfile.mkdtemp",
                return_value=str(tmp_path),
            ),
            patch("benchmark.reporting.publish.pages_publisher.shutil.rmtree"),
        ):
            ok = publisher._git_push_sync("reports/x/report.html", "<html/>")

        assert ok is True
        assert push_calls == 2


# ════════════════════════════════════════════════════════════════════════════
# build_pages_publisher — factory degrade semantics
# ════════════════════════════════════════════════════════════════════════════


class TestBuildPagesPublisher:
    def test_disabled_returns_none(self) -> None:
        with patch(
            "benchmark.reporting.publish.pages_publisher.fetch_report_site_config",
            return_value=ReportSiteConfig(),  # enabled=False by default
        ):
            assert build_pages_publisher() is None

    def test_incomplete_config_returns_none(self) -> None:
        cfg = ReportSiteConfig(
            enabled=True,
            site_base_url="https://eval-reports.io.alibaba-inc.com",
            # git_api_base / repo / token is missing
        )
        with patch(
            "benchmark.reporting.publish.pages_publisher.fetch_report_site_config",
            return_value=cfg,
        ):
            assert build_pages_publisher() is None

    def test_full_config_returns_publisher(self) -> None:
        with patch(
            "benchmark.reporting.publish.pages_publisher.fetch_report_site_config",
            return_value=_FULL_CFG,
        ):
            publisher = build_pages_publisher()

        assert isinstance(publisher, GitPagesReportPublisher)

    def test_http_scheme_git_api_base_warns_but_builds(self) -> None:
        """An http:// git_api_base is 302'd to https by the gateway: log a warning, but do not block assembly."""
        cfg = ReportSiteConfig(
            enabled=True,
            site_base_url="https://eval-reports.io.alibaba-inc.com",
            git_api_base="http://gitlab.alibaba-inc.com",
            repo="amap-llm/eval-report-pages",
            token="test-token",
        )
        with patch(
            "benchmark.reporting.publish.pages_publisher.fetch_report_site_config",
            return_value=cfg,
        ):
            with patch(
                "benchmark.reporting.publish.pages_publisher.logger",
            ) as mock_logger:
                publisher = build_pages_publisher()

        assert isinstance(publisher, GitPagesReportPublisher)
        assert any(
            "302" in str(call.args[0]) and "https" in str(call.args[0])
            for call in mock_logger.warning.call_args_list
        )


# ════════════════════════════════════════════════════════════════════════════
# fetch_report_site_config — RemoteConfig parsing and degrade
# ════════════════════════════════════════════════════════════════════════════


def _passthrough_cache(_key, compute, *, use_cache=True):  # noqa: ANN001, ANN202
    return compute()


class TestFetchReportSiteConfig:
    def test_empty_raw_returns_disabled_default(self) -> None:
        with patch(
            "benchmark.domain.report_site_config.fetch_cached_settings",
            return_value={},
        ):
            with patch(
                "benchmark.domain.report_site_config.get_or_compute",
                side_effect=_passthrough_cache,
            ):
                cfg = fetch_report_site_config(use_cache=False)

        assert cfg.enabled is False
        assert cfg.branch == "master"

    def test_full_raw_parsed(self) -> None:
        raw = {
            "enabled": True,
            "site_base_url": "https://eval-reports.io.alibaba-inc.com",
            "git_api_base": "https://code.alibaba-inc.com",
            "repo": "amap-llm/eval-report-pages",
            "branch": "master",
            "token": "secret-token",
            "git_username": "someone",
            "timeout_s": 15,
            "unexpected_field": "ignored",
        }
        with patch(
            "benchmark.domain.report_site_config.fetch_cached_settings",
            return_value=raw,
        ):
            with patch(
                "benchmark.domain.report_site_config.get_or_compute",
                side_effect=_passthrough_cache,
            ):
                cfg = fetch_report_site_config(use_cache=False)

        assert cfg.enabled is True
        assert cfg.repo == "amap-llm/eval-report-pages"
        assert cfg.git_username == "someone"
        assert cfg.timeout_s == 15

    def test_invalid_raw_falls_back_to_disabled(self) -> None:
        """timeout_s=0 violates ge=1 → parsing fails and degrades to the default (disabled)."""
        with patch(
            "benchmark.domain.report_site_config.fetch_cached_settings",
            return_value={"enabled": True, "timeout_s": 0},
        ):
            with patch(
                "benchmark.domain.report_site_config.get_or_compute",
                side_effect=_passthrough_cache,
            ):
                cfg = fetch_report_site_config(use_cache=False)

        assert cfg.enabled is False
