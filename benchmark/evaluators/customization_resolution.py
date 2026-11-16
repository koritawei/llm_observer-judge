"""运行期定制解析：skill 前置 + prompt slot 覆盖（judge / attribution 共用）。

语义（设计 §4.6）：只要 ref 存在，skill 就参与拼接；prompt 只影响被
覆盖的 slot，未覆盖 slot 回退平台默认（唯一允许的回退）；无 ref 时
行为与现状完全一致。拉取/校验失败显式抛错，不静默回退默认。
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.evaluators.base import PromptBundleRef, SkillBundleRef
from benchmark.customization.bundle_store import (
    MaterializedStageBundle,
    get_bundle_store,
)

logger = get_logger(__name__)

#: 运行日志里 prompt 正文预览上限；超长截断并标注原文长度。
#: 运维要求「每次评测打印最新 prompt」，上限放宽到可覆盖常规定制模板。
_PROMPT_LOG_PREVIEW_CHARS = 20_000
#: git_path 末段 ``…/v{N}`` → contentVersion 回退推导。
_VERSION_SUFFIX_RE = re.compile(r"/v([1-9][0-9]*)$")


class _PromptManager(Protocol):
    async def load_prompt(self, name: str) -> str:
        """按名加载平台默认模板。"""
        ...


def bundle_refs_from_snapshot(
    snapshot: Mapping[str, Any] | None,
    stage: str,
) -> tuple[PromptBundleRef | None, SkillBundleRef | None]:
    """run 快照（config_json.agent_registration）→ 某 stage 的两个 ref。"""
    if not isinstance(snapshot, Mapping):
        return None, None
    customization = snapshot.get("customization")
    if not isinstance(customization, Mapping):
        return None, None
    stage_ref = customization.get(stage)
    if not isinstance(stage_ref, Mapping):
        return None, None
    git_path = str(stage_ref.get("git_path") or "")
    raw_version = stage_ref.get("content_version")
    content_version = (
        int(raw_version)
        if isinstance(raw_version, int) or (isinstance(raw_version, str) and raw_version.isdigit())
        else _content_version_from_path(git_path)
    )
    base = {
        "stage": stage,
        "git_path": git_path,
        "git_commit": str(stage_ref.get("git_commit") or ""),
        "bundle_sha256": str(stage_ref.get("bundle_sha256") or ""),
        "content_version": content_version,
    }
    prompt_ref: PromptBundleRef | None = None
    skill_ref: SkillBundleRef | None = None
    prompt_section = stage_ref.get("prompt")
    if isinstance(prompt_section, Mapping):
        prompt_ref = PromptBundleRef(
            **base,
            slots=[str(item) for item in prompt_section.get("slots") or []],
            files=[dict(item) for item in prompt_section.get("files") or []],
        )
    skill_section = stage_ref.get("skill")
    if isinstance(skill_section, Mapping):
        skill_ref = SkillBundleRef(
            **base,
            entry=str(skill_section.get("entry") or "SKILL.md"),
            files=[dict(item) for item in skill_section.get("files") or []],
        )
    return prompt_ref, skill_ref


def _content_version_from_path(git_path: str) -> int | None:
    """从 ``…/v{N}`` 路径段推导 contentVersion；无法识别返回 None。"""
    matched = _VERSION_SUFFIX_RE.search(git_path.strip())
    if matched is None:
        return None
    return int(matched.group(1))


def _preview_prompt(text: str, *, max_chars: int = _PROMPT_LOG_PREVIEW_CHARS) -> str:
    """日志预览：超长截断并标注原文长度。"""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}…[truncated len={len(text)}]"


def prompt_provenance(
    prompt_ref: PromptBundleRef | None,
    skill_ref: SkillBundleRef | None,
) -> dict[str, Any] | None:
    """从两个 ref 抽出可落库 / 可查询的定制溯源字段（无定制返回 None）。"""
    base = prompt_ref or skill_ref
    if base is None:
        return None
    version = base.content_version
    if version is None:
        version = _content_version_from_path(base.git_path)
    return {
        "stage": base.stage,
        "content_version": version,
        "git_commit": base.git_commit,
        "git_path": base.git_path,
        "bundle_sha256": base.bundle_sha256,
        "has_prompt": prompt_ref is not None,
        "has_skill": skill_ref is not None,
    }


@dataclass(frozen=True)
class _ResolvedPromptLog:
    """``_log_resolved_prompt`` 入参（FUNC-PARAMS-001 收敛）。"""

    source: str
    slot: str
    resolved: str
    prompt_ref: PromptBundleRef | None
    skill_ref: SkillBundleRef | None


def _log_resolved_prompt(log: _ResolvedPromptLog) -> None:
    """运行期 INFO：定制溯源元信息；仅命中定制版本时附截断 preview（运维核对）。"""
    provenance = prompt_provenance(log.prompt_ref, log.skill_ref) or {}
    commit = str(provenance.get("git_commit") or "")
    # 平台默认 prompt 不落正文；仅 content_version 命中定制时打印截断预览。
    preview = (
        _preview_prompt(log.resolved) if provenance.get("content_version") is not None else "-"
    )
    logger.info(
        "customization.prompt_resolved source=%s stage=%s slot=%s "
        "content_version=%s commit=%s sha=%s chars=%d preview=%s",
        log.source,
        provenance.get("stage") or "default",
        log.slot,
        provenance.get("content_version"),
        commit[:12] if commit else "-",
        str(provenance.get("bundle_sha256") or "-")[:12],
        len(log.resolved),
        preview,
    )


def bundle_refs_from_config(
    config_json: Mapping[str, Any] | None,
    stage: str,
) -> tuple[PromptBundleRef | None, SkillBundleRef | None]:
    """run config_json → 某 stage 的两个 ref（无注册快照时均为 None）。"""
    if not isinstance(config_json, Mapping):
        return None, None
    snapshot = config_json.get("agent_registration")
    return bundle_refs_from_snapshot(snapshot if isinstance(snapshot, Mapping) else None, stage)


def _stage_ref_dict(
    prompt_ref: PromptBundleRef | None,
    skill_ref: SkillBundleRef | None,
) -> dict[str, Any] | None:
    """两个形态 ref → BundleStore.ensure 需要的 stage ref（同 commit/同目录）。"""
    base = prompt_ref or skill_ref
    if base is None:
        return None
    return {
        "stage": base.stage,
        "git_path": base.git_path,
        "git_commit": base.git_commit,
        "bundle_sha256": base.bundle_sha256,
        "content_version": base.content_version,
        "prompt": (
            {"slots": list(prompt_ref.slots), "files": [dict(f) for f in prompt_ref.files]}
            if prompt_ref is not None
            else None
        ),
        "skill": (
            {"entry": skill_ref.entry, "files": [dict(f) for f in skill_ref.files]}
            if skill_ref is not None
            else None
        ),
    }


def _refs_of(
    source: object,
) -> tuple[PromptBundleRef | None, SkillBundleRef | None]:
    """从 EvaluatorRequest 或 metadata dict 提取两个 ref（dict 形态自动建模）。"""
    if isinstance(source, Mapping):
        prompt_raw = source.get("prompt_bundle_ref")
        skill_raw = source.get("skill_bundle_ref")
    else:
        prompt_raw = getattr(source, "prompt_bundle_ref", None)
        skill_raw = getattr(source, "skill_bundle_ref", None)
    prompt_ref = (
        prompt_raw
        if isinstance(prompt_raw, PromptBundleRef)
        else PromptBundleRef.model_validate(prompt_raw)
        if isinstance(prompt_raw, Mapping)
        else None
    )
    skill_ref = (
        skill_raw
        if isinstance(skill_raw, SkillBundleRef)
        else SkillBundleRef.model_validate(skill_raw)
        if isinstance(skill_raw, Mapping)
        else None
    )
    return prompt_ref, skill_ref


async def _ensure_bundle(
    prompt_ref: PromptBundleRef | None,
    skill_ref: SkillBundleRef | None,
) -> MaterializedStageBundle | None:
    stage_ref = _stage_ref_dict(prompt_ref, skill_ref)
    if stage_ref is None:
        return None
    return await get_bundle_store().ensure(stage_ref)


async def resolve_stage_prompt(
    request: object,
    prompt_manager: _PromptManager,
    slot: str,
    default_name: str,
) -> str:
    """skill 前置 +（bundle slot 覆盖或平台默认模板）。

    无 ref 时与现状一致（只加载平台默认）；bundle 未覆盖该 slot 时
    回退平台默认（部分覆盖语义），skill 仍前置。
    """
    prompt_ref, skill_ref = _refs_of(request)
    bundle = await _ensure_bundle(prompt_ref, skill_ref)
    skill = bundle.skill_text if bundle is not None else None
    text = bundle.prompt.get(slot) if bundle is not None else None
    if text is None:
        if bundle is not None and prompt_ref is not None:
            logger.debug("customization.slot_fallback slot=%s default=%s", slot, default_name)
        text = await prompt_manager.load_prompt(default_name)
    resolved = f"{skill}\n\n{text}" if skill else text
    # 每次评测都打印实际使用的 prompt（含平台默认 / 定制版本）。
    _log_resolved_prompt(
        _ResolvedPromptLog(
            source="prompt_manager",
            slot=slot,
            resolved=resolved,
            prompt_ref=prompt_ref,
            skill_ref=skill_ref,
        )
    )
    return resolved


async def resolve_stage_prompt_from_text(
    request: object,
    slot: str,
    default_text: str,
) -> str:
    """异步版：skill 前置 +（bundle slot 覆盖或给定的默认文本）。

    供不经 PromptManager 的 evaluator（如 search_ai_card 直读文件）使用。
    """
    prompt_ref, skill_ref = _refs_of(request)
    bundle = await _ensure_bundle(prompt_ref, skill_ref)
    if bundle is None:
        _log_resolved_prompt(
            _ResolvedPromptLog(
                source="text",
                slot=slot,
                resolved=default_text,
                prompt_ref=None,
                skill_ref=None,
            )
        )
        return default_text
    text = bundle.prompt.get(slot)
    if text is None:
        logger.debug("customization.slot_fallback slot=%s (text)", slot)
        text = default_text
    resolved = f"{bundle.skill_text}\n\n{text}" if bundle.skill_text else text
    _log_resolved_prompt(
        _ResolvedPromptLog(
            source="text",
            slot=slot,
            resolved=resolved,
            prompt_ref=prompt_ref,
            skill_ref=skill_ref,
        )
    )
    return resolved


def resolve_stage_prompt_text(
    request: object,
    slot: str,
    default_text: str,
) -> str:
    """不依赖 PromptManager 的同步版（归因 agent 用；bundle 须已物化）。

    ref 存在但未物化时显式抛错（CustomizationError 503），不静默回退。
    """
    prompt_ref, skill_ref = _refs_of(request)
    stage_ref = _stage_ref_dict(prompt_ref, skill_ref)
    if stage_ref is None:
        return default_text
    bundle = get_bundle_store().load_materialized_sync(stage_ref)
    skill = bundle.skill_text
    text = bundle.prompt.get(slot)
    if text is None:
        logger.debug("customization.slot_fallback slot=%s (sync)", slot)
        text = default_text
    resolved = f"{skill}\n\n{text}" if skill else text
    _log_resolved_prompt(
        _ResolvedPromptLog(
            source="sync",
            slot=slot,
            resolved=resolved,
            prompt_ref=prompt_ref,
            skill_ref=skill_ref,
        )
    )
    return resolved


async def ensure_request_bundles(request: object) -> None:
    """受理/worker 侧预物化（让同步解析路径可直接读缓存）。"""
    prompt_ref, skill_ref = _refs_of(request)
    await _ensure_bundle(prompt_ref, skill_ref)


__all__ = [
    "bundle_refs_from_config",
    "bundle_refs_from_snapshot",
    "ensure_request_bundles",
    "prompt_provenance",
    "resolve_stage_prompt",
    "resolve_stage_prompt_from_text",
    "resolve_stage_prompt_text",
]
