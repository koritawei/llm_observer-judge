"""Upload orchestration: policy check → allocate contentVersion → atomic Git commit → return the ref.

Order constraint: commit Git first, then write the counter ref back — a Git failure allocates no further state and
does not persist a registration (design §4.5.3). Re-uploading the same content idempotently reuses the latest version ref.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.records.models.runtime_config_record import CustomizationKey
from benchmark.domain.customization_config import CustomizationConfig, fetch_customization_config
from benchmark.domain.runtime_config_store import (
    next_customization_version,
    peek_customization_sequence,
    record_customization_ref,
)
from benchmark.runs.common.agent_capabilities import customization_policy_for_name
from benchmark.customization.errors import CustomizationError
from benchmark.customization.git_store import GitContentStore
from benchmark.customization.manifest import (
    BundleInput,
    BundleLimits,
    ManifestSpec,
    build_manifest,
    git_path_for,
    prepare_bundle,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class StageUploadRequest:
    """One upload of one stage (at least one of prompt / skill)."""

    agent_name: str
    business: str
    variant_key: str
    stage: str
    prompt_texts: dict[str, str] = field(default_factory=dict)
    skill_files: dict[str, bytes] = field(default_factory=dict)
    created_by: str = ""


async def upload_stage_customization(
    request: StageUploadRequest,
    *,
    config: CustomizationConfig | None = None,
    git_store: GitContentStore | None = None,
) -> dict[str, Any]:
    """Upload and commit one stage's customization and return a StageCustomizationRef (snake_case)."""
    cfg = config or fetch_customization_config()
    if not cfg.enabled:
        raise CustomizationError(
            503, "customization_disabled", "业务定制未启用（RemoteConfig customization.enabled=false）"
        )
    if request.stage not in ("judge", "attribution"):
        raise CustomizationError(400, "invalid_stage", f"未知 stage：{request.stage}")
    policy = customization_policy_for_name(request.agent_name)
    stage_policy = policy.judge if request.stage == "judge" else policy.attribution
    key = CustomizationKey(
        agent_name=request.agent_name,
        business=request.business,
        variant_key=request.variant_key,
        stage=request.stage,
    )
    bundle = prepare_bundle(
        BundleInput(
            stage=request.stage,
            stage_policy=stage_policy,
            prompt_texts=request.prompt_texts,
            skill_files=request.skill_files,
            limits=BundleLimits(
                max_prompt_bytes=cfg.max_prompt_bytes,
                max_skill_bytes=cfg.max_skill_bytes,
                max_files=cfg.max_files,
            ),
        )
    )

    # Idempotent: if the latest version for the same (agent, business, variant, stage) has the same content, reuse it
    sequence = await peek_customization_sequence(key)
    last_ref = sequence.get("last_ref")
    if isinstance(last_ref, dict) and last_ref.get("bundle_sha256") == bundle.bundle_sha256:
        logger.info(
            "customization.upload_idempotent agent=%s stage=%s version=%s",
            request.agent_name,
            request.stage,
            last_ref.get("content_version"),
        )
        return dict(last_ref)

    version = await next_customization_version(key)
    git_path = git_path_for(cfg.base_path, key, version)
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = build_manifest(
        ManifestSpec(
            key=key,
            content_version=version,
            bundle=bundle,
            created_at=created_at,
            created_by=request.created_by,
        )
    )
    files = dict(bundle.files)
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    store = git_store or GitContentStore(cfg)
    commit = await store.commit_bundle(
        git_path=git_path,
        files=files,
        message=(
            f"feat(benchmark-customization): {request.agent_name}/{request.business}/"
            f"{request.variant_key}/{request.stage}/v{version}"
        ),
    )
    ref: dict[str, Any] = {
        "stage": request.stage,
        "content_version": version,
        "git_branch": cfg.branch,
        "git_path": git_path,
        "git_commit": commit,
        "bundle_sha256": bundle.bundle_sha256,
        "prompt": bundle.prompt_ref,
        "skill": bundle.skill_ref,
        "created_at": created_at,
        "created_by": request.created_by,
        "status": "active",
    }
    await record_customization_ref(key, ref)
    logger.info(
        "customization.uploaded agent=%s business=%s variant=%s stage=%s version=%d "
        "commit=%s kinds=%s files=%d",
        request.agent_name,
        request.business,
        request.variant_key,
        request.stage,
        version,
        commit[:12],
        [kind for kind in ("prompt", "skill") if ref[kind]],
        len(files),
    )
    return ref


__all__ = ["StageUploadRequest", "upload_stage_customization"]
