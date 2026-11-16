"""Aone Pages report publisher — Git platform HTTP API push plus local URL derivation.

Aone Pages has no runtime upload API: deployment is driven by "git push → deploy-pages CI".
This implementation pushes the report HTML as a commit into a dedicated reports repo through
the alicode Files API v3 (code.alibaba-inc.com / gitlab.alibaba-inc.com,
``POST/PUT /api/v3/projects/{repo}/repository/files``, file path in the body
``file_path`` field). The push triggers deployment automatically. The Pages URL is derived
locally from the path rule and does not wait for CI (optimistic return; a 404 during the
1-3min deploy window is expected).

Repo path convention (confirmed by the user): ``reports/{yyyy-mm}/{run_id}/report.html``
(``yyyy-mm`` is UTC at publish time; monthly directories support later archive cleanup).

Pages URL derivation: the deploy-pages pipeline is configured with ``deploy-dir=reports``
(skill red line: when deployable files sit in a clean subdirectory, point at that directory;
``deploy-dir=.`` is forbidden). The site root is the contents of ``reports/``, so the URL
strips the ``reports/`` prefix — ``{site_base_url}/{yyyy-mm}/{run_id}/report.html``.
That derivation follows the deterministic deploy-dir mapping, and is not a guess.
Protocol contract: ``publish`` **never raises** — any failure (network / auth / conflict
retries exhausted) only logs a warning and returns ``None``, and does not block run
completion (aligned with the existing OSS upload-failure semantics; see the storage
degradation in ``webdev_client.generate_report``).

Security red line: ``token`` (the Git platform PAT) is used only in the ``PRIVATE-TOKEN``
request header (Files API) or in the temporary clone's ``https://{git_username}:{token}@``
remote URL (git push; the username must be a domain account — a PAT with the ``oauth2``
username is rejected by auth; see the alicode "code push and download" auth matrix). It
must not be written to logs, exception messages, or persistent storage.

Large-file switch: the Files API sends the whole file in a JSON body and is bound by the
intranet gateway body limit (empirical 20MB; over the limit is 413 or a dropped connection).
Above ``_API_CONTENT_LIMIT_BYTES``, switch to git smart-HTTP push (alicode "code push and
download API": https-form push). A streaming pack has no body-size limit.
"""

import asyncio
import os
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import quote, urlparse

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.report_site_config import (
    ReportSiteConfig,
    fetch_report_site_config,
)
from benchmark.utils.http_transport import AsyncClient

logger = get_logger("agent_eval.benchmark.report_site_publisher")

#: Archive-root prefix inside the repo; the Pages site root (deploy-dir) points at this directory,
#: and URL derivation strips this prefix.
_REPO_ROOT_PREFIX = "reports/"

#: run_id path sanitizing: illegal characters are replaced with ``_`` (aligned with OSS object-key sanitizing,
#: see ``webdev_client._SAFE``).
_SAFE = re.compile(r"[^A-Za-z0-9_.\-]")


@dataclass(frozen=True)
class _GitCommitSpec:
    """Parameter bundle for ``_git_clone_and_commit`` (FUNC-PARAMS-001 parameter consolidation)."""

    tmpdir: str
    url: str
    env: dict[str, str]
    path: str
    content: str


@dataclass(frozen=True)
class _ProbeSpec:
    """Parameter bundle for ``_probe_exists`` (FUNC-PARAMS-001 parameter consolidation)."""

    files_url: str
    headers: dict[str, str]
    path: str


#: Cap on POST/PUT conflict-rotation retries (concurrent finalize competing on base-sha of the same repo)
_PUBLISH_MAX_ATTEMPTS = 3
_PUBLISH_BACKOFF_BASE_S = 0.5
_PUBLISH_BACKOFF_MAX_S = 5.0
_PUBLISH_JITTER_RATIO = 0.3

#: HTTP status codes (PLR2004)
_HTTP_OK = 200
_HTTP_CREATED = 201
_HTTP_BAD_REQUEST = 400
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_NOT_FOUND = 404
_HTTP_CONFLICT = 409
_HTTP_REDIRECT_MIN = 300
_HTTP_REDIRECT_MAX = 399

_SUCCESS_STATUSES = frozenset({_HTTP_OK, _HTTP_CREATED})
_AUTH_FAIL_STATUSES = frozenset({_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN})

#: Files API content limit: the whole file goes in a JSON body; above this threshold switch to git https push
#: (empirical intranet-gateway body limit 20MB; git smart-HTTP streaming pack has no such limit).
_API_CONTENT_LIMIT_BYTES = 20 * 1024 * 1024

#: Per-step timeout of the git subprocess (clone / commit / push / rebase), independent of API timeout_s.
_GIT_OP_TIMEOUT_S = 300.0


class ReportSiteReportPublisher(Protocol):
    """Aone Pages report publisher protocol."""

    async def publish(self, run_id: str, html: str) -> str | None:
        """Push the report HTML to the reports repo and return the derived Pages URL.

        Returns:
            ``https://<site>/reports/{yyyy-mm}/{run_id}/report.html``;
            on failure (including disabled / empty html) return ``None``. The implementation must not raise.
        """
        ...


class GitPagesReportPublisher:
    """alicode Files API v3 implementation (shared by code / gitlab.alibaba-inc.com).

    Push flow: ``GET`` probes whether the file exists (200 → PUT update; 404 → POST
    create; any other status gives up) → POST/PUT commit. When the remote already has
    a file at the same path, overwrite it: rotate retries as "PUT 404 → switch to POST"
    and "POST 400/409 already exists → switch to PUT update", capped at
    ``_PUBLISH_MAX_ATTEMPTS``.

    Contract points (alicode docs igv4fn / kib8qxwmkgadpopu):
    - The URL stops at ``/repository/files``; the file path is body ``file_path``.
      Putting the path in the URL the official GitLab v4 way (``/files/{path}``) has no
      such route → 405
    - The branch field is ``branch_name`` (not GitLab's ``branch``)
    - ``gitlab.alibaba-inc.com`` supports only api/v3; a v4 path is 302/405 from the gateway
    - Encoding uses the default ``text`` (base64 is forbidden): base64 depends on the
      server decoding by ``encoding`` before storing. If an implementation ignores that
      field, the repo stores the base64 text and Pages cannot render it. There is no
      successful online sample to prove the decode behavior, while HTML is plain text and
      a JSON body transports it losslessly, so text mode renders on any implementation
      (same as uploading bytes straight to OSS)
    - Above ``_API_CONTENT_LIMIT_BYTES`` (20MB), switch to git https push: a Files API
      whole-file JSON body hits the gateway body limit (413 / dropped connection), and
      git smart-HTTP streaming pack has no such limit (alicode "code push and download API")
    """

    def __init__(self, config: ReportSiteConfig) -> None:
        self._cfg = config
        self._api_base = config.git_api_base.rstrip("/")
        self._repo_enc = quote(config.repo, safe="")
        self._site_base = config.site_base_url.rstrip("/")

    async def publish(self, run_id: str, html: str) -> str | None:
        if not html:
            # The caller contract already gates on non-empty html; reaching here means the report artifact is abnormal and must be visible
            logger.warning("benchmark.report_site_publish_empty_html run_id=%s", run_id)
            return None
        path = self._build_repo_path(run_id)
        content_bytes = len(html.encode("utf-8"))
        try:
            if content_bytes > _API_CONTENT_LIMIT_BYTES:
                # Over the gateway body limit: switch to git https push (streaming pack)
                logger.info(
                    "benchmark.report_site_push_via_git run_id=%s path=%s bytes=%d",
                    run_id,
                    path,
                    content_bytes,
                )
                pushed = await self._push_via_git(path, html)
            else:
                pushed = await self._push_file(path, html)
        except Exception:
            # Defensive fallback: the publish protocol promises never to raise
            logger.warning(
                "benchmark.report_site_publish_failed run_id=%s path=%s",
                run_id,
                path,
                exc_info=True,
            )
            return None
        if not pushed:
            # Run-level terminal conclusion: the specific cause is already in an upstream warning (git_unavailable /
            # push_rejected / conflict_exhausted / git_timeout / git_error),
            # this line summarizes it so a run_id search finds "this report did not enter the Pages repo"
            logger.warning(
                "benchmark.report_site_push_incomplete run_id=%s path=%s bytes=%d",
                run_id,
                path,
                content_bytes,
            )
            return None
        # deploy-dir=reports: the site root maps to the contents of reports/, and the URL strips the prefix
        page_url = f"{self._site_base}/{path.removeprefix(_REPO_ROOT_PREFIX)}"
        logger.info(
            "benchmark.report_site_published run_id=%s path=%s url=%s",
            run_id,
            path,
            page_url,
        )
        return page_url

    @staticmethod
    def _build_repo_path(run_id: str) -> str:
        """``reports/{yyyy-mm}/{run_id}/report.html`` (UTC month bucket + run_id sanitizing)."""
        month = datetime.now(UTC).strftime("%Y-%m")
        safe_run = _SAFE.sub("_", run_id.strip()) or "unknown"
        return f"{_REPO_ROOT_PREFIX}{month}/{safe_run}/report.html"

    async def _push_file(self, path: str, content: str) -> bool:
        """Commit one file (GET probe → POST create / PUT update). Returns whether the push succeeded."""
        # The URL stops at the collection (alicode v3: file_path is in the body);
        # putting the path in the URL the GitLab v4 way is 405 on alicode (confirmed by two rounds of online logs)
        files_url = f"{self._api_base}/api/v3/projects/{self._repo_enc}/repository/files"
        headers = {"PRIVATE-TOKEN": self._cfg.token}
        body: dict[str, Any] = {
            "branch_name": self._cfg.branch,
            "file_path": path,
            # Default text encoding (base64 is forbidden; see the contract points in the class docstring)
            "content": content,
            "commit_message": f"feat(report): publish benchmark report {path}",
        }

        async with AsyncClient(timeout=self._cfg.timeout_s) as client:
            exists = await self._probe_exists(
                client,
                _ProbeSpec(files_url=files_url, headers=headers, path=path),
            )
            if exists is None:
                return False

            for attempt in range(1, _PUBLISH_MAX_ATTEMPTS + 1):
                if exists:
                    resp = await client.put(files_url, json=body, headers=headers)
                else:
                    resp = await client.post(files_url, json=body, headers=headers)

                if resp.status_code in _SUCCESS_STATUSES:
                    return True

                # Conflict rotation: PUT 404 (concurrent delete / probe race) → switch to POST;
                # POST 400/409 (probe race / optimistic create after 302, remote already exists) → switch to PUT and overwrite.
                # Under create semantics, 400/409 is always treated as "already exists"; do not match the error text:
                # if it is a real parameter error, PUT fails with the same status, and the final degradation semantics stay the same.
                if exists and resp.status_code == _HTTP_NOT_FOUND:
                    exists = False
                elif not exists and resp.status_code in {
                    _HTTP_BAD_REQUEST,
                    _HTTP_CONFLICT,
                }:
                    exists = True
                else:
                    # 401/403/422/5xx and similar: retrying is pointless, give up (do not leak the token; log only the status code)
                    logger.warning(
                        "benchmark.report_site_push_rejected path=%s status=%d body=%s",
                        path,
                        resp.status_code,
                        resp.text[:200],
                    )
                    return False

                if attempt < _PUBLISH_MAX_ATTEMPTS:
                    delay = min(
                        _PUBLISH_BACKOFF_BASE_S * (2 ** (attempt - 1)),
                        _PUBLISH_BACKOFF_MAX_S,
                    )
                    delay += random.uniform(0.0, delay * _PUBLISH_JITTER_RATIO)
                    logger.info(
                        "benchmark.report_site_push_conflict_retry path=%s attempt=%d delay_s=%.2f",
                        path,
                        attempt,
                        delay,
                    )
                    await asyncio.sleep(delay)

        logger.warning(
            "benchmark.report_site_push_conflict_exhausted path=%s attempts=%d",
            path,
            _PUBLISH_MAX_ATTEMPTS,
        )
        return False

    async def _push_via_git(self, path: str, content: str) -> bool:
        """Large files are pushed over git smart-HTTP (alicode "code push and download API").

        When the git binary or git_username is missing, degrade and return ``False``
        (same never-raise semantics as publish).
        """
        if shutil.which("git") is None:
            logger.warning(
                "benchmark.report_site_git_unavailable path=%s — git binary missing",
                path,
            )
            return False
        if not self._cfg.git_username:
            logger.warning(
                "benchmark.report_site_git_credentials_missing path=%s — "
                "RemoteConfig aone-pages 缺 git_username（域账号）；PAT 须配域账号"
                "用户名，oauth2 用户名须配 OAuth2 token（alicode《代码推送和下载》）",
                path,
            )
            return False
        return await asyncio.to_thread(self._git_push_sync, path, content)

    def _git_push_sync(self, path: str, content: str) -> bool:
        """Synchronous git push: shallow clone → write the file → commit → push.

        Auth: ``https://{git_username}:{token}@host``, username is a domain account
        (alicode auth matrix: a PAT pairs with a domain account; the oauth2 username
        pairs with an OAuth2 token). The token enters only the temporary clone's remote
        config, and the temp directory is deleted when the push ends (including on
        exception). Subprocess output may enter logs only after :meth:`_sanitize`.
        """
        host = urlparse(self._api_base).netloc
        url = (
            f"https://{quote(self._cfg.git_username, safe='')}:"
            f"{quote(self._cfg.token, safe='')}@{host}/{self._cfg.repo}.git"
        )
        # noqa: TID251 — the git subprocess must inherit PATH/HOME and similar; only GIT_TERMINAL_PROMPT is appended
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # noqa: TID251
        tmpdir = tempfile.mkdtemp(prefix="report_site_push_")
        try:
            if not self._git_clone_and_commit(
                _GitCommitSpec(tmpdir=tmpdir, url=url, env=env, path=path, content=content),
            ):
                return False
            return self._git_push_with_retry(tmpdir, env, path)
        except subprocess.TimeoutExpired:
            logger.warning("benchmark.report_site_git_timeout path=%s", path)
            return False
        except Exception:
            logger.warning("benchmark.report_site_git_error path=%s", path, exc_info=True)
            return False
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _git_clone_and_commit(
        self,
        spec: "_GitCommitSpec",
    ) -> bool:
        """clone → write the file → add → commit; nothing to commit counts as success (idempotent)."""
        tmpdir = spec.tmpdir
        url = spec.url
        env = spec.env
        path = spec.path
        content = spec.content
        clone = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                self._cfg.branch,
                url,
                tmpdir,
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_OP_TIMEOUT_S,
            env=env,
            check=False,
        )
        if clone.returncode != 0:
            logger.warning(
                "benchmark.report_site_git_clone_failed path=%s err=%s",
                path,
                self._sanitize(clone.stderr or "")[-500:],
            )
            return False

        target = os.path.join(tmpdir, path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as fp:
            fp.write(content)

        if not self._run_git(["git", "-C", tmpdir, "add", path], env, path):
            return False
        commit = subprocess.run(
            [
                "git",
                "-C",
                tmpdir,
                "-c",
                "user.name=agent-eval-bot",
                "-c",
                "user.email=agent-eval-bot@alibaba-inc.com",
                "commit",
                "-m",
                f"feat(report): publish benchmark report {path}",
            ],
            capture_output=True,
            text=True,
            timeout=_GIT_OP_TIMEOUT_S,
            env=env,
            check=False,
        )
        if commit.returncode == 0:
            return True
        combined = (commit.stdout or "") + (commit.stderr or "")
        if "nothing to commit" in combined:
            # Content unchanged (healer/regenerate replay is idempotent): the repo already has the target content
            return True
        logger.warning(
            "benchmark.report_site_git_commit_failed path=%s err=%s",
            path,
            self._sanitize(combined)[-500:],
        )
        return False

    def _git_push_with_retry(self, tmpdir: str, env: dict[str, str], path: str) -> bool:
        """push; on non-fast-forward (concurrent push race) ``pull --rebase`` and retry once."""
        for attempt in (1, 2):
            push = subprocess.run(
                ["git", "-C", tmpdir, "push", "origin", self._cfg.branch],
                capture_output=True,
                text=True,
                timeout=_GIT_OP_TIMEOUT_S,
                env=env,
                check=False,
            )
            if push.returncode == 0:
                return True
            err = self._sanitize((push.stderr or "") + (push.stdout or ""))
            recoverable = "non-fast-forward" in err or "fetch first" in err
            if (
                recoverable
                and attempt == 1
                and self._run_git(
                    ["git", "-C", tmpdir, "pull", "--rebase", "origin", self._cfg.branch],
                    env,
                    path,
                )
            ):
                continue
            logger.warning(
                "benchmark.report_site_git_push_failed path=%s err=%s",
                path,
                err[-500:],
            )
            return False
        return False

    def _run_git(self, cmd: list[str], env: dict[str, str], path: str) -> bool:
        """Run one git command; on failure log a sanitized message and return ``False``."""
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_GIT_OP_TIMEOUT_S,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning(
                "benchmark.report_site_git_step_failed path=%s cmd=%s err=%s",
                path,
                cmd[1] if len(cmd) > 1 else cmd,
                self._sanitize(proc.stderr or "")[-500:],
            )
            return False
        return True

    def _sanitize(self, text: str) -> str:
        """Log sanitizing: the token never lands in logs (git stderr may echo the remote URL)."""
        if not text:
            return ""
        return text.replace(self._cfg.token, "***").replace(quote(self._cfg.token, safe=""), "***")

    async def _probe_exists(
        self,
        client: AsyncClient,
        probe: "_ProbeSpec",
    ) -> bool | None:
        """Probe whether the file exists on the target branch.

        Returns:
            ``True`` / ``False``: definitely exists / does not exist;
            ``None``: auth failure and similar cases where the push should be given up.

        Note: Codeup / the intranet gateway often returns **302 (SSO/redirect)** for a
        Files API GET. Giving up on every non-200/404 means "publish was called but never
        POST/PUT". For 3xx, be optimistic: log Location and POST-create as if absent.
        If the file actually exists, the later ``POST 400 already exists → PUT`` rotation heals it.
        """
        files_url = probe.files_url
        headers = probe.headers
        path = probe.path
        resp = await client.get(
            files_url,
            # alicode v3: file_path is a query parameter (only official GitLab v4 puts it in the URL)
            params={"file_path": path, "ref": self._cfg.branch},
            headers=headers,
        )
        if resp.status_code == _HTTP_OK:
            return True
        if resp.status_code == _HTTP_NOT_FOUND:
            return False
        if resp.status_code in _AUTH_FAIL_STATUSES:
            logger.warning(
                "benchmark.report_site_probe_auth_failed path=%s status=%d "
                "(check RemoteConfig token / PRIVATE-TOKEN scope)",
                path,
                resp.status_code,
            )
            return None

        location = resp.headers.get("location") or resp.headers.get("Location") or ""
        if _HTTP_REDIRECT_MIN <= resp.status_code <= _HTTP_REDIRECT_MAX:
            # 302 is common on the SSO gateway; do not stop, optimistic POST (see the docstring)
            logger.warning(
                "benchmark.report_site_probe_redirect path=%s status=%d "
                "location=%s — treat as absent, try POST create",
                path,
                resp.status_code,
                location[:200],
            )
            return False

        logger.warning(
            "benchmark.report_site_probe_failed path=%s status=%d location=%s "
            "— treat as absent, try POST create",
            path,
            resp.status_code,
            location[:200],
        )
        return False


def build_pages_publisher() -> ReportSiteReportPublisher | None:
    """Assemble the Aone Pages publisher; return ``None`` when disabled, config is missing, or an exception occurs.

    Aligned with the graceful-degradation semantics of
    :func:`benchmark.reporting.framework.report_storage.build_report_storage_client` —
    the caller skips the second publish on ``None``, same behavior as before this was wired in.
    """
    cfg = fetch_report_site_config()

    if not cfg.enabled:
        return None

    if not (cfg.site_base_url and cfg.git_api_base and cfg.repo and cfg.token):
        logger.warning(
            "RemoteConfig aone-pages config: enabled=true but required fields "
            "incomplete (need site_base_url / git_api_base / repo / token); "
            "publisher disabled",
        )
        return None

    if cfg.git_api_base.startswith("http://"):
        # http is 302'd to https by the intranet gateway, and httpx does not follow redirects by default,
        # so probe/push would all fail (the 302 trap; see the _probe_exists docstring)
        logger.warning(
            "RemoteConfig aone-pages config: git_api_base=%s uses http:// scheme; "
            "gateway 302-redirect to https will break probe/push — "
            "switch to https://",
            cfg.git_api_base,
        )

    try:
        publisher: ReportSiteReportPublisher = GitPagesReportPublisher(cfg)
    except Exception:
        logger.warning(
            "benchmark.report_site_publisher init failed",
            exc_info=True,
        )
        return None

    logger.info(
        "benchmark.report_site_publisher ready site=%s repo=%s branch=%s",
        cfg.site_base_url,
        cfg.repo,
        cfg.branch,
    )
    return publisher


__all__ = [
    "ReportSiteReportPublisher",
    "GitPagesReportPublisher",
    "build_pages_publisher",
]
