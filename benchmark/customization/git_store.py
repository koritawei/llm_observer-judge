"""Git 内容仓库读写：一次 clone+push 原子提交 + 按 commit 读取。

写入对齐 ``report_publish.pages_publisher`` 的实证模式：浅克隆目标分支
→ 写全部文件 → commit → push；non-fast-forward 时 ``pull --rebase``
重试一次。读取优先 Files API v3（``file_path`` 走 query、content 按
encoding 解码），失败回退 ``git fetch --depth 1 origin <commit>`` +
``git archive``。token 永不入日志（stderr 统一脱敏）。
"""

import asyncio
import base64
import functools
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote, urlparse

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.customization_config import CustomizationConfig
from benchmark.customization.errors import CustomizationError
from benchmark.utils.http_transport import AsyncClient

logger = get_logger(__name__)

_GIT_AUTHOR_NAME = "agent-eval-bot"
_GIT_AUTHOR_EMAIL = "agent-eval-bot@alibaba-inc.com"

#: git 同步操作（subprocess）专用显式线程池：进程级持有、容量固定，
#: 禁止走事件循环默认线程池（DEFAULT-EXECUTOR-001）。
_GIT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="customization-git")

_T = TypeVar("_T")


async def _run_git_blocking(call: Callable[[], _T]) -> _T:
    """在显式 git 线程池执行同步调用，保留调用方 ContextVar 上下文。"""
    ctx = copy_context()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_GIT_EXECUTOR, ctx.run, call)


@dataclass(frozen=True)
class _CommitSpec:
    git_path: str
    files: dict[str, bytes]
    message: str


class GitContentStore:
    """受控 Git 内容仓库（业务 prompt/skill 专用，仓库定版 agent_eval_prompt_skill）。"""

    def __init__(self, config: CustomizationConfig) -> None:
        self._cfg = config
        self._api_base = config.git_api_base.rstrip("/")
        self._repo_enc = quote(config.repo, safe="")

    # ── 写入 ──────────────────────────────────────────────────────────

    async def commit_bundle(
        self,
        *,
        git_path: str,
        files: dict[str, bytes],
        message: str,
    ) -> str:
        """原子提交一个版本目录的全部文件，返回 commit sha。

        git 不可用 / 鉴权失败 / 超时 → :class:`CustomizationError` 503，
        调用方不分配后续状态（先提交 Git，再写注册）。
        """
        if shutil.which("git") is None:
            raise CustomizationError(503, "customization_git_unavailable", "git 二进制不可用")
        if not self._cfg.git_username or not self._cfg.token:
            raise CustomizationError(
                503,
                "customization_git_unavailable",
                "RemoteConfig customization 缺 git_username / token",
            )
        spec = _CommitSpec(git_path=git_path, files=files, message=message)
        try:
            return await _run_git_blocking(functools.partial(self._commit_sync, spec))
        except CustomizationError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise CustomizationError(503, "customization_git_unavailable", "git 提交超时") from exc
        except Exception as exc:
            raise CustomizationError(
                503, "customization_git_unavailable", f"git 提交失败：{type(exc).__name__}"
            ) from exc

    def _commit_sync(self, spec: _CommitSpec) -> str:
        host = urlparse(self._api_base).netloc
        url = (
            f"https://{quote(self._cfg.git_username, safe='')}:"
            f"{quote(self._cfg.token, safe='')}@{host}/{self._cfg.repo}.git"
        )
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # noqa: TID251
        tmpdir = tempfile.mkdtemp(prefix="customization_commit_")
        try:
            clone = self._run_git(
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
                env,
            )
            if clone.returncode != 0:
                raise CustomizationError(
                    503,
                    "customization_git_unavailable",
                    f"git clone 失败：{self._sanitize(clone.stderr)[-300:]}",
                )
            target_dir = Path(tmpdir, spec.git_path)
            target_dir.mkdir(parents=True, exist_ok=True)
            for rel_path, data in spec.files.items():
                destination = Path(target_dir, rel_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            self._git_step(["add", spec.git_path], tmpdir, env)
            commit = self._run_git(
                [
                    "git",
                    "-C",
                    tmpdir,
                    "-c",
                    f"user.name={_GIT_AUTHOR_NAME}",
                    "-c",
                    f"user.email={_GIT_AUTHOR_EMAIL}",
                    "commit",
                    "-m",
                    spec.message,
                ],
                env,
            )
            if commit.returncode != 0:
                combined = self._sanitize((commit.stdout or "") + (commit.stderr or ""))
                if "nothing to commit" not in combined:
                    raise CustomizationError(
                        503, "customization_git_unavailable", f"git commit 失败：{combined[-300:]}"
                    )
            self._push_with_retry(tmpdir, env)
            head = self._run_git(["git", "-C", tmpdir, "rev-parse", "HEAD"], env)
            return (head.stdout or "").strip()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _push_with_retry(self, tmpdir: str, env: dict[str, str]) -> None:
        for attempt in (1, 2):
            push = self._run_git(["git", "-C", tmpdir, "push", "origin", self._cfg.branch], env)
            if push.returncode == 0:
                return
            err = self._sanitize((push.stderr or "") + (push.stdout or ""))
            recoverable = "non-fast-forward" in err or "fetch first" in err
            if recoverable and attempt == 1:
                rebase = self._run_git(
                    ["git", "-C", tmpdir, "pull", "--rebase", "origin", self._cfg.branch],
                    env,
                )
                if rebase.returncode == 0:
                    continue
            raise CustomizationError(
                503, "customization_git_unavailable", f"git push 失败：{err[-300:]}"
            )

    def _git_step(self, args: list[str], cwd: str, env: dict[str, str]) -> None:
        proc = self._run_git(["git", "-C", cwd, *args], env)
        if proc.returncode != 0:
            raise CustomizationError(
                503,
                "customization_git_unavailable",
                f"git {' '.join(args[:1])} 失败：{self._sanitize(proc.stderr)[-300:]}",
            )

    def _run_git(self, cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self._cfg.timeout_s,
            env=env,
            check=False,
        )

    # ── 读取（按 commit，永不按 branch） ─────────────────────────────

    async def list_stage_versions(self, base_path: str) -> list[int]:
        """列举某 stage 根目录下的全部版本号（vN 目录），倒序；目录不存在返回空。

        404 → 空列表；其它 HTTP/网络失败 → 503（与 read_files / latest_commit 对齐）。
        显式分页 per_page=100，避免 GitLab 默认 20 条静默截断。
        """
        url = f"{self._api_base}/api/v3/projects/{self._repo_enc}/repository/tree"
        headers = {"PRIVATE-TOKEN": self._cfg.token}
        versions: list[int] = []
        page = 1
        per_page = 100
        try:
            async with AsyncClient(timeout=self._cfg.fetch_timeout_s) as client:
                while True:
                    response = await client.get(
                        url,
                        params={
                            "path": base_path,
                            "ref": self._cfg.branch,
                            "per_page": per_page,
                            "page": page,
                        },
                        headers=headers,
                    )
                    if response.status_code == HTTPStatus.NOT_FOUND:
                        return []
                    if response.status_code != HTTPStatus.OK:
                        raise CustomizationError(
                            503,
                            "customization_fetch_failed",
                            f"版本目录读取失败（{response.status_code}）：{base_path}",
                        )
                    entries = response.json()
                    if not isinstance(entries, list) or not entries:
                        break
                    for entry in entries:
                        name = entry.get("name", "") if isinstance(entry, dict) else ""
                        if name.startswith("v") and name[1:].isdigit():
                            versions.append(int(name[1:]))
                    if len(entries) < per_page:
                        break
                    page += 1
        except CustomizationError:
            raise
        except Exception as exc:
            raise CustomizationError(
                503,
                "customization_fetch_failed",
                f"版本目录读取网络失败：{base_path}",
            ) from exc
        return sorted(versions, reverse=True)

    async def latest_commit_for_path(self, path: str) -> str:
        """取某版本目录的最新 commit sha（版本不可变语义下的读取定位）。"""
        url = f"{self._api_base}/api/v3/projects/{self._repo_enc}/repository/commits"
        headers = {"PRIVATE-TOKEN": self._cfg.token}
        async with AsyncClient(timeout=self._cfg.fetch_timeout_s) as client:
            response = await client.get(
                url,
                params={"path": path, "ref_name": self._cfg.branch, "per_page": 1},
                headers=headers,
            )
            if response.status_code != HTTPStatus.OK:
                raise CustomizationError(
                    503,
                    "customization_fetch_failed",
                    f"提交历史读取失败（{response.status_code}）：{path}",
                )
            entries = response.json()
        if not entries:
            raise CustomizationError(503, "customization_fetch_failed", f"目录无提交记录：{path}")
        return str(entries[0].get("id") or entries[0].get("short_id") or "")

    async def read_files(self, *, git_commit: str, paths: list[str]) -> dict[str, bytes]:
        """按 commit 读取多个文件；任一失败整体抛 503（不部分回退）。"""
        try:
            return await self._read_via_files_api(git_commit, paths)
        except CustomizationError:
            raise
        except Exception as exc:
            logger.warning(
                "customization.files_api_read_failed commit=%s err=%s — fallback to git fetch",
                git_commit[:12],
                type(exc).__name__,
            )
        return await _run_git_blocking(
            functools.partial(self._read_via_git_sync, git_commit, paths)
        )

    async def _read_via_files_api(self, git_commit: str, paths: list[str]) -> dict[str, bytes]:
        files_url = f"{self._api_base}/api/v3/projects/{self._repo_enc}/repository/files"
        headers = {"PRIVATE-TOKEN": self._cfg.token}
        result: dict[str, bytes] = {}
        async with AsyncClient(timeout=self._cfg.fetch_timeout_s) as client:
            for path in paths:
                response = await client.get(
                    files_url,
                    params={"file_path": path, "ref": git_commit},
                    headers=headers,
                )
                if response.status_code == HTTPStatus.NOT_FOUND:
                    raise CustomizationError(
                        503,
                        "customization_fetch_failed",
                        f"commit {git_commit[:12]} 下文件不存在：{path}",
                    )
                if response.status_code != HTTPStatus.OK:
                    raise CustomizationError(
                        503,
                        "customization_fetch_failed",
                        f"Files API 读取失败（{response.status_code}）：{path}",
                    )
                payload = response.json()
                content = payload.get("content") or ""
                if payload.get("encoding") == "base64":
                    result[path] = base64.b64decode(content)
                else:
                    result[path] = str(content).encode("utf-8")
        return result

    def _read_via_git_sync(self, git_commit: str, paths: list[str]) -> dict[str, bytes]:
        host = urlparse(self._api_base).netloc
        url = (
            f"https://{quote(self._cfg.git_username, safe='')}:"
            f"{quote(self._cfg.token, safe='')}@{host}/{self._cfg.repo}.git"
        )
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # noqa: TID251
        tmpdir = tempfile.mkdtemp(prefix="customization_fetch_")
        try:
            init = self._run_git(["git", "-C", tmpdir, "init", "-q"], env)
            if init.returncode != 0:
                raise CustomizationError(503, "customization_fetch_failed", "git init 失败")
            self._git_step(["remote", "add", "origin", url], tmpdir, env)
            fetch = self._run_git(
                ["git", "-C", tmpdir, "fetch", "-q", "--depth", "1", "origin", git_commit],
                env,
            )
            if fetch.returncode != 0:
                raise CustomizationError(
                    503,
                    "customization_fetch_failed",
                    f"git fetch commit 失败：{self._sanitize(fetch.stderr)[-300:]}",
                )
            result: dict[str, bytes] = {}
            for path in paths:
                show = subprocess.run(
                    ["git", "-C", tmpdir, "show", f"FETCH_HEAD:{path}"],
                    capture_output=True,
                    timeout=self._cfg.fetch_timeout_s,
                    env=env,
                    check=False,
                )
                if show.returncode != 0:
                    raise CustomizationError(
                        503,
                        "customization_fetch_failed",
                        f"commit 下文件不存在：{path}",
                    )
                result[path] = show.stdout
            return result
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _sanitize(self, text: str) -> str:
        """日志脱敏：token 永不落日志（git stderr 可能回显 remote URL）。"""
        if not text:
            return ""
        return text.replace(self._cfg.token, "***").replace(quote(self._cfg.token, safe=""), "***")


__all__ = ["GitContentStore"]
