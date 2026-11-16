"""bundle 物化：按 commit 拉取 + sha256 校验 + 磁盘/内存两级缓存。

读取一律按 ``git_commit``（永不按 branch）；校验不一致 409、拉取失败
503，均显式失败不回退平台默认（设计 §4.7）。进程级单例经
:func:`get_bundle_store` 获取，单测可注入 fake。
"""

import asyncio
import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.customization_config import CustomizationConfig, fetch_customization_config
from benchmark.customization.errors import CustomizationError
from benchmark.customization.git_store import GitContentStore

logger = get_logger(__name__)


class BundleFetcher(Protocol):
    """按 commit 读取文件的最小协议（生产 = GitContentStore，单测 = fake）。"""

    async def read_files(self, *, git_commit: str, paths: list[str]) -> dict[str, bytes]:
        """读取指定文件；失败抛 :class:`CustomizationError`。"""
        ...


@dataclass(frozen=True)
class MaterializedStageBundle:
    """物化后的 stage 内容：prompt slot → 文本；skill 前置指令文本。"""

    prompt: Mapping[str, str] = field(default_factory=dict)
    skill_text: str | None = None


def _ref_files(ref: Mapping[str, Any]) -> list[str]:
    """ref 声明的全部文件相对路径（bundle 目录内）。"""
    paths: list[str] = []
    for kind in ("prompt", "skill"):
        section = ref.get(kind)
        if isinstance(section, Mapping):
            for entry in section.get("files") or []:
                if isinstance(entry, Mapping) and entry.get("path"):
                    paths.append(str(entry["path"]))
    return paths


def _verify_manifest(ref: Mapping[str, Any], contents: Mapping[str, bytes]) -> None:
    """manifest 存在/可解析/bundleSha256 与 ref 一致复核。不一致 409。"""
    git_path = str(ref.get("git_path") or "")
    manifest_raw = contents.get("manifest.json")
    if manifest_raw is None:
        raise CustomizationError(
            503, "customization_fetch_failed", f"bundle 缺 manifest.json：{git_path}"
        )
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CustomizationError(
            409, "checksum_mismatch", f"manifest 不可解析：{git_path}"
        ) from exc
    if manifest.get("bundleSha256") != ref.get("bundle_sha256"):
        raise CustomizationError(
            409, "checksum_mismatch", f"manifest bundleSha256 与 ref 不一致：{git_path}"
        )


def _verify_files(ref: Mapping[str, Any], contents: Mapping[str, bytes]) -> None:
    """逐文件 sha256 + 大小校验。不一致 409、缺失 503。"""
    git_path = str(ref.get("git_path") or "")
    for kind in ("prompt", "skill"):
        section = ref.get(kind)
        if not isinstance(section, Mapping):
            continue
        for entry in section.get("files") or []:
            path = str(entry["path"])
            data = contents.get(path)
            if data is None:
                raise CustomizationError(
                    503, "customization_fetch_failed", f"bundle 文件缺失：{git_path}/{path}"
                )
            if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                raise CustomizationError(
                    409, "checksum_mismatch", f"文件 sha256 不一致：{git_path}/{path}"
                )
            if len(data) != entry.get("bytes"):
                raise CustomizationError(
                    409, "checksum_mismatch", f"文件大小不一致：{git_path}/{path}"
                )


def _verify_and_build(
    ref: Mapping[str, Any],
    contents: Mapping[str, bytes],
) -> MaterializedStageBundle:
    """manifest 复核 + 逐文件校验 → 物化结果（prompt slot 文本 + skill 前置）。"""
    git_path = str(ref.get("git_path") or "")
    _verify_manifest(ref, contents)
    _verify_files(ref, contents)
    prompt: dict[str, str] = {}
    prompt_section = ref.get("prompt")
    if isinstance(prompt_section, Mapping):
        # 文件布局 prompt/prompts/<slot>.md：slot 即文件 stem
        for entry in prompt_section.get("files") or []:
            path = str(entry["path"])
            prompt[Path(path).stem] = contents[path].decode("utf-8")
    skill_text: str | None = None
    skill_section = ref.get("skill")
    if isinstance(skill_section, Mapping):
        entry_name = str(skill_section.get("entry") or "SKILL.md")
        entry_path = f"skill/{entry_name}"
        data = contents.get(entry_path)
        if data is None:
            raise CustomizationError(
                503, "customization_fetch_failed", f"skill 入口缺失：{git_path}/{entry_path}"
            )
        skill_text = data.decode("utf-8")
    return MaterializedStageBundle(prompt=prompt, skill_text=skill_text)


class BundleStore:
    """按 ref 物化 bundle（内存索引 + 磁盘缓存，键均为 commit）。"""

    def __init__(self, config: CustomizationConfig, fetcher: BundleFetcher) -> None:
        self._config = config
        self._fetcher = fetcher
        self._memory: dict[tuple[str, str], MaterializedStageBundle] = {}
        self._lock = asyncio.Lock()

    def _disk_dir(self, ref: Mapping[str, Any]) -> Path:
        repo_hash = hashlib.sha256(self._config.repo.encode("utf-8")).hexdigest()[:16]
        commit = str(ref.get("git_commit") or "")
        git_path = str(ref.get("git_path") or "")
        return Path(self._config.cache_dir, repo_hash, commit, git_path)

    async def ensure(self, ref: Mapping[str, Any]) -> MaterializedStageBundle:
        """物化一个 stage ref（已物化直接返回缓存）。拉取失败 503、校验失败 409。"""
        key = (str(ref.get("git_commit") or ""), str(ref.get("git_path") or ""))
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        async with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                return cached
            bundle = self._load_from_disk(ref)
            if bundle is None:
                bundle = await self._fetch_and_materialize(ref)
            self._memory[key] = bundle
            return bundle

    def load_materialized_sync(self, ref: Mapping[str, Any]) -> MaterializedStageBundle:
        """同步读已物化缓存（内存/磁盘）；未物化显式失败（不静默回退）。"""
        key = (str(ref.get("git_commit") or ""), str(ref.get("git_path") or ""))
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        bundle = self._load_from_disk(ref)
        if bundle is None:
            raise CustomizationError(
                503,
                "customization_fetch_failed",
                f"bundle 未物化（需先 ensure）：{ref.get('git_path')}",
            )
        self._memory[key] = bundle
        return bundle

    def _load_from_disk(self, ref: Mapping[str, Any]) -> MaterializedStageBundle | None:
        directory = self._disk_dir(ref)
        manifest_file = directory / "manifest.json"
        if not manifest_file.is_file():
            return None
        contents: dict[str, bytes] = {"manifest.json": manifest_file.read_bytes()}
        for rel_path in _ref_files(ref):
            target = directory / rel_path
            if not target.is_file():
                return None
            contents[rel_path] = target.read_bytes()
        try:
            return _verify_and_build(ref, contents)
        except CustomizationError:
            logger.warning("customization.disk_cache_invalid path=%s", ref.get("git_path"))
            return None

    async def _fetch_and_materialize(self, ref: Mapping[str, Any]) -> MaterializedStageBundle:
        git_path = str(ref.get("git_path") or "")
        commit = str(ref.get("git_commit") or "")
        rel_paths = ["manifest.json", *_ref_files(ref)]
        full_paths = [f"{git_path}/{rel}" for rel in rel_paths]
        raw = await self._fetcher.read_files(git_commit=commit, paths=full_paths)
        contents = {
            rel: raw[full] for rel, full in zip(rel_paths, full_paths, strict=True) if full in raw
        }
        if len(contents) != len(rel_paths):
            raise CustomizationError(
                503, "customization_fetch_failed", f"bundle 文件拉取不完整：{git_path}"
            )
        bundle = _verify_and_build(ref, contents)
        directory = self._disk_dir(ref)
        for rel_path, data in contents.items():
            target = directory / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        logger.info(
            "customization.bundle_materialized path=%s commit=%s files=%d",
            git_path,
            commit[:12],
            len(contents),
        )
        return bundle

    def prune_cache(self) -> None:
        """缓存总量超上限时按 commit 目录新旧清理（启动时调用）。"""
        root = Path(self._config.cache_dir)
        if not root.is_dir() or self._config.cache_max_bytes <= 0:
            return
        commit_dirs = [path for path in root.glob("*/*") if path.is_dir()]
        commit_dirs.sort(key=lambda path: path.stat().st_mtime)
        total = sum(
            item.stat().st_size
            for directory in commit_dirs
            for item in directory.rglob("*")
            if item.is_file()
        )
        for directory in commit_dirs:
            if total <= self._config.cache_max_bytes:
                break
            size = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
            shutil.rmtree(directory, ignore_errors=True)
            total -= size


_store: BundleStore | None = None


def get_bundle_store() -> BundleStore:
    """进程级 BundleStore 单例（按 RemoteConfig 配置构造；配置变不热更新）。"""
    global _store  # noqa: PLW0603 - 进程级单例（设计 §4.5.4）
    if _store is None:
        config = fetch_customization_config()
        _store = BundleStore(config, GitContentStore(config))
        _store.prune_cache()
    return _store


def set_bundle_store(store: BundleStore | None) -> None:
    """注入/重置单例（单测 fake 与应用装配用）。"""
    global _store  # noqa: PLW0603 - 进程级单例
    _store = store


__all__ = [
    "BundleFetcher",
    "BundleStore",
    "MaterializedStageBundle",
    "get_bundle_store",
    "set_bundle_store",
]
