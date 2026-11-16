"""bundle 组装与校验：manifest 生成、slot 白名单、变量扫描、sha256。

布局（设计 §4.5.2）::

    {base_path}/{agent}/{business}/{variant}/{stage}/v{content_version}/
      manifest.json
      prompt/prompts/<slot>.md       # kind=prompt（可无）
      skill/SKILL.md                 # kind=skill（可无）
      skill/resources/…              # 仅 md/txt/json/yaml
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from benchmark.records.models.runtime_config_record import CustomizationKey
from benchmark.customization.errors import CustomizationError
from benchmark.reporting.framework.base import StageCustomizationPolicy

#: 路径段白名单（拒绝 ``..``、空段、绝对路径、反斜杠的统一闸口）
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SKILL_RESOURCE_SUFFIXES = (".md", ".txt", ".json", ".yaml", ".yml")
_PROMPT_SUFFIXES = (".md", ".prompt")


def slugify(value: str) -> str:
    """任意标识 → 路径安全 slug（小写、非字母数字归一为下划线）。"""
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_sha256(
    stage: str,
    prompt_ref: dict[str, Any] | None,
    skill_ref: dict[str, Any] | None,
) -> str:
    """stage 内容版本整体标识：对 ``{stage, prompt, skill}`` 规范化 JSON 做 sha256。"""
    canonical = json.dumps(
        {"stage": stage, "prompt": prompt_ref, "skill": skill_ref},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256(canonical.encode("utf-8"))


def _check_segment(value: str, label: str) -> str:
    cleaned = value.strip()
    if not SLUG_RE.fullmatch(cleaned):
        raise CustomizationError(
            400,
            "invalid_path_segment",
            f"{label} 只允许字母数字与 ._- 且长度 ≤64：{value!r}",
        )
    return cleaned


def _decode_text(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CustomizationError(400, "invalid_encoding", f"{label} 不是 UTF-8 文本") from exc


def _check_variables(text: str, slot: str, allowed: tuple[str, ...]) -> None:
    """校验模板 ``${var}``；``allowed`` 含 ``*`` 时放行任意标识符（可扩展宏）。"""

    if "*" in allowed:
        return
    unknown = sorted(set(_VARIABLE_RE.findall(text)) - set(allowed))
    if unknown:
        raise CustomizationError(
            400,
            "unknown_template_variable",
            f"slot {slot} 模板出现未声明变量：{', '.join(unknown)}",
        )


@dataclass(frozen=True)
class PreparedBundle:
    """校验完成、待提交的 bundle（文件相对 git_path 的路径 → 字节）。"""

    stage: str
    files: dict[str, bytes] = field(default_factory=dict)
    prompt_ref: dict[str, Any] | None = None
    skill_ref: dict[str, Any] | None = None
    bundle_sha256: str = ""


@dataclass(frozen=True)
class BundleLimits:
    """bundle 大小回退上限（policy 字段为 0 时用；来源 CustomizationConfig）。"""

    max_prompt_bytes: int
    max_skill_bytes: int
    max_files: int


@dataclass(frozen=True)
class BundleInput:
    """一次 bundle 组装入参：stage + 策略 + 内容 + 大小回退上限。"""

    stage: str
    stage_policy: StageCustomizationPolicy
    prompt_texts: dict[str, str]
    skill_files: dict[str, bytes]
    limits: BundleLimits


@dataclass
class _PartContext:
    """prepare 两半共用的执行上下文（files 是共享的输出字典，原地写入）。"""

    stage: str
    stage_policy: StageCustomizationPolicy
    limits: BundleLimits
    files: dict[str, bytes]


@dataclass(frozen=True)
class ManifestSpec:
    """一版定制内容的完整描述（build_manifest 的入参收拢）。

    key 是 DB 计数器与 Git 目录共用的四维定位键；created_at / created_by
    是审计字段，原样落进 manifest.json。
    """

    key: CustomizationKey
    content_version: int
    bundle: PreparedBundle
    created_at: str
    created_by: str


def prepare_bundle(bundle_input: BundleInput) -> PreparedBundle:
    """组装并校验一个 stage 的 bundle（prompt / skill 至少一项）。

    ``prompt_texts``: slot → 模板文本（已解码）；``skill_files``:
    bundle 内相对路径（如 ``SKILL.md`` / ``resources/x.md``）→ 字节。
    policy 声明的大小上限优先，为 0 时用 ``limits`` 的回退值。
    """
    if not bundle_input.prompt_texts and not bundle_input.skill_files:
        raise CustomizationError(
            400,
            "empty_stage_bundle",
            f"{bundle_input.stage} 阶段 prompt 与 skill 不能同时为空",
        )
    files: dict[str, bytes] = {}
    context = _PartContext(
        stage=bundle_input.stage,
        stage_policy=bundle_input.stage_policy,
        limits=bundle_input.limits,
        files=files,
    )
    prompt_ref = _prepare_prompt_part(context, bundle_input.prompt_texts)
    skill_ref = _prepare_skill_part(context, bundle_input.skill_files)
    max_files = bundle_input.stage_policy.max_files or bundle_input.limits.max_files
    if len(files) > max_files:
        raise CustomizationError(400, "too_many_files", f"bundle 文件数超过 {max_files}")

    return PreparedBundle(
        stage=bundle_input.stage,
        files=files,
        prompt_ref=prompt_ref,
        skill_ref=skill_ref,
        bundle_sha256=bundle_sha256(bundle_input.stage, prompt_ref, skill_ref),
    )


def _prepare_prompt_part(
    context: _PartContext,
    prompt_texts: dict[str, str],
) -> dict[str, Any] | None:
    """校验 prompt 部分（slot 白名单/变量/大小），文件写入 context.files。"""
    if not prompt_texts:
        return None
    stage_policy = context.stage_policy
    if not stage_policy.allow_prompt:
        raise CustomizationError(
            403, "customization_not_allowed", f"{context.stage} 阶段不允许业务定制 prompt"
        )
    slots_by_key = {slot.key: slot for slot in stage_policy.prompt_slots}
    max_bytes = stage_policy.max_prompt_bytes or context.limits.max_prompt_bytes
    total = 0
    entries: list[dict[str, Any]] = []
    for slot_key in sorted(prompt_texts):
        slot = slots_by_key.get(slot_key)
        if slot is None:
            raise CustomizationError(
                400, "slot_not_allowed", f"slot {slot_key} 不在 {context.stage} 阶段白名单内"
            )
        text = prompt_texts[slot_key]
        _check_variables(text, slot_key, slot.variables)
        data = text.encode("utf-8")
        if len(data) > (slot.max_bytes or max_bytes):
            raise CustomizationError(400, "prompt_too_large", f"slot {slot_key} 超过大小上限")
        total += len(data)
        path = f"prompt/prompts/{slot_key}.md"
        context.files[path] = data
        entries.append({"path": path, "sha256": _sha256(data), "bytes": len(data)})
    if total > max_bytes:
        raise CustomizationError(400, "prompt_too_large", "prompt 总大小超过上限")
    for slot in stage_policy.prompt_slots:
        if slot.required and slot.key not in prompt_texts:
            raise CustomizationError(400, "slot_required", f"slot {slot.key} 为必填槽位")
    return {"slots": sorted(prompt_texts), "files": entries}


def _prepare_skill_part(
    context: _PartContext,
    skill_files: dict[str, bytes],
) -> dict[str, Any] | None:
    """校验 skill 部分（入口/路径/类型/大小），文件写入 context.files。"""
    if not skill_files:
        return None
    stage_policy = context.stage_policy
    if not stage_policy.allow_skill:
        raise CustomizationError(
            403, "customization_not_allowed", f"{context.stage} 阶段不允许业务定制 skill"
        )
    entry_name = stage_policy.skill_entry or "SKILL.md"
    if entry_name not in skill_files:
        raise CustomizationError(400, "skill_entry_missing", f"skill 必须包含 {entry_name}")
    max_bytes = stage_policy.max_skill_bytes or context.limits.max_skill_bytes
    total = 0
    entries: list[dict[str, Any]] = []
    for rel_path in sorted(skill_files):
        _check_skill_path(rel_path)
        data = skill_files[rel_path]
        _decode_text(data, f"skill 文件 {rel_path}")
        total += len(data)
        path = f"skill/{rel_path}"
        context.files[path] = data
        entries.append({"path": path, "sha256": _sha256(data), "bytes": len(data)})
    if total > max_bytes:
        raise CustomizationError(400, "skill_too_large", "skill 总大小超过上限")
    return {"entry": entry_name, "files": entries}


def _check_skill_path(rel_path: str) -> None:
    if rel_path.startswith(("/", "\\")) or "\\" in rel_path:
        raise CustomizationError(400, "invalid_path", f"skill 文件路径非法：{rel_path!r}")
    segments = rel_path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise CustomizationError(400, "invalid_path", f"skill 文件路径非法：{rel_path!r}")
    if not rel_path.lower().endswith(_SKILL_RESOURCE_SUFFIXES):
        raise CustomizationError(
            400, "invalid_file_type", f"skill 文件仅支持 md/txt/json/yaml：{rel_path!r}"
        )


def prompt_slot_from_filename(filename: str) -> str:
    """上传文件名 → slot（``<slot>.md`` / ``<slot>.prompt``，其余拒绝）。"""
    name = filename.rsplit("/", 1)[-1]
    for suffix in _PROMPT_SUFFIXES:
        if name.lower().endswith(suffix):
            slot = name[: -len(suffix)]
            if SLUG_RE.fullmatch(slot):
                return slot
            break
    raise CustomizationError(
        400, "invalid_prompt_file", f"prompt 文件名须为 <slot>.md/.prompt：{filename!r}"
    )


def build_manifest(spec: ManifestSpec) -> dict[str, Any]:
    """manifest.json 内容（与 git_path 同目录落盘，读侧复核用）。"""
    return {
        "schemaVersion": 1,
        "agentName": spec.key.agent_name,
        "business": spec.key.business,
        "variantKey": spec.key.variant_key,
        "stage": spec.key.stage,
        "contentVersion": spec.content_version,
        "bundleSha256": spec.bundle.bundle_sha256,
        "prompt": spec.bundle.prompt_ref,
        "skill": spec.bundle.skill_ref,
        "createdAt": spec.created_at,
        "createdBy": spec.created_by,
    }


def stage_base_path(base_path: str, key: CustomizationKey) -> str:
    """stage 根目录。

    无 scope：``{base}/{agent}/{business}/{variant}/{stage}``（存量）。
    有 scope：``{base}/{agent}/{scope}/{business}/{variant}/{stage}``。
    """
    segments = [
        _check_segment(base_path.strip("/"), "base_path"),
        _check_segment(slugify(key.agent_name), "agent"),
    ]
    scope = (key.agent_scope or "").strip()
    if scope:
        segments.append(_check_segment(slugify(scope), "agentScope"))
    segments.extend(
        [
            _check_segment(key.business, "business"),
            _check_segment(key.variant_key, "variantKey"),
            _check_segment(key.stage, "stage"),
        ]
    )
    return "/".join(segments)


def git_path_for(base_path: str, key: CustomizationKey, content_version: int) -> str:
    """版本目录路径：``{stage_base}/v{n}``（段均校验）。"""
    return f"{stage_base_path(base_path, key)}/v{content_version}"


__all__ = [
    "SLUG_RE",
    "BundleInput",
    "BundleLimits",
    "ManifestSpec",
    "PreparedBundle",
    "build_manifest",
    "bundle_sha256",
    "git_path_for",
    "prepare_bundle",
    "prompt_slot_from_filename",
    "slugify",
    "stage_base_path",
]
