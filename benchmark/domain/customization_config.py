"""Git content-repo config for business prompt/skill customization — loaded from RemoteConfig.

Same pattern as :mod:`benchmark.domain.report_site_config`: config lives in RemoteConfig
with a TTL cache. The sensitive ``token`` field lives only in RemoteConfig and must not
enter the repo, yaml, or logs.

RemoteConfig JSON shape::

    {
        "enabled": true,
        "git_api_base": "https://code.alibaba-inc.com",
        "repo": "agent_eval_prompt_skill",
        "branch": "master",
        "token": "<repo PAT>",
        "git_username": "<domain-account>",
        "base_path": "benchmark-customizations",
        "timeout_s": 30,
        "fetch_timeout_s": 15,
        "cache_dir": "/tmp/agent-eval-bundles",
        "cache_max_bytes": 268435456,
        "max_prompt_bytes": 65536,
        "max_skill_bytes": 262144,
        "max_files": 32
    }

``repo`` is fixed as ``agent_eval_prompt_skill`` (the content repo dedicated to business
prompt/skill customization; it stores only version directories and manifests, not
execution code). If the platform requires a full group path in RemoteConfig, prepend only
the group prefix and leave the repo name unchanged.

``enabled`` defaults to ``False``: when RemoteConfig is unset or unavailable, customization
is disabled (upload endpoints 503, registration save rejects a customization) and existing
flows are unaffected.
"""

from benchmark.libs.agenthost.observability.logger import get_logger
from pydantic import BaseModel, Field

from benchmark.settings.remote_config_store import fetch_cached_settings
from benchmark.domain._cache import get_or_compute

logger = get_logger(__name__)

#: RemoteConfig data_id for the business-customization Git repo config.
CUSTOMIZATION_DATA_ID = "agent-eval.benchmark.customization"
#: RemoteConfig group for the business-customization Git repo config.
CUSTOMIZATION_GROUP = "agent-eval.service.group"

#: Fixed repo name (content repo dedicated to business prompt/skill customization).
CUSTOMIZATION_REPO = "agent_eval_prompt_skill"


class CustomizationConfig(BaseModel):
    """Git content-repo config for business customization — loaded from RemoteConfig, Pydantic-validated.

    Every field has a default; ``enabled`` defaults to ``False``. ``token`` is a Git
    platform PAT (least privilege: repository read/write) and lives only in RemoteConfig.
    """

    model_config = {"extra": "ignore"}

    enabled: bool = False
    #: Git platform API base, for example ``https://code.alibaba-inc.com`` (must be https).
    git_api_base: str = "https://code.alibaba-inc.com"
    #: Content repo (fixed ``agent_eval_prompt_skill``; a group prefix is allowed).
    repo: str = CUSTOMIZATION_REPO
    #: Branch to write (reads always use the commit sha, not the branch).
    branch: str = "master"
    #: Git platform PAT (PRIVATE-TOKEN). Do not commit or log it (repr is off to avoid leaks).
    token: str = Field(default="", repr=False)
    #: git https push username (domain account; a PAT must use the domain account, not swapped).
    git_username: str = ""
    #: Root prefix of version directories inside the repo.
    base_path: str = "benchmark-customizations"
    #: Timeout for git writes (seconds).
    timeout_s: float = Field(default=30.0, ge=1)
    #: Timeout for fetching by commit (seconds).
    fetch_timeout_s: float = Field(default=15.0, ge=1)
    #: Directory for the materialized bundle cache.
    cache_dir: str = "/tmp/agent-eval-bundles"
    #: Total cap of the materialized cache (bytes); over-limit old dirs are cleaned at startup.
    cache_max_bytes: int = Field(default=268435456, ge=0)
    #: Default size cap for the prompt shape (used when policy does not declare one).
    max_prompt_bytes: int = Field(default=65536, ge=1)
    #: Default size cap for the skill shape (used when policy does not declare one).
    max_skill_bytes: int = Field(default=262144, ge=1)
    #: Max file count for one bundle (used when policy does not declare one).
    max_files: int = Field(default=32, ge=1)
    #: Allowlist of businesses that may register. Empty = unrestricted (tighten via RemoteConfig).
    #: When non-empty, business must hit the list to save a customized registration (§12.3 INV-5).
    business_allowlist: list[str] = Field(default_factory=list)
    #: Multi-instance quota (§12.3 INV-4): max active instances per component.
    max_instances_per_component: int = Field(default=20, ge=1)
    #: Multi-instance quota: max active instances per business.
    max_instances_per_business: int = Field(default=10, ge=1)
    #: Multi-instance quota: max active versions per instance.
    max_versions_per_instance: int = Field(default=10, ge=1)
    #: L2 process-isolation list (§12.7): case jobs for a matching business are routed
    #: to the benchmark-cases-l2 queue and consumed by a separate process. Empty = default queue.
    l2_businesses: list[str] = Field(default_factory=list)


def fetch_customization_config(
    *,
    data_id: str | None = None,
    group: str | None = None,
    use_cache: bool = True,
) -> CustomizationConfig:
    """Load and parse business-customization config from RemoteConfig; unavailable returns defaults (disabled)."""
    data_id = data_id or CUSTOMIZATION_DATA_ID
    group = group or CUSTOMIZATION_GROUP

    def _load_and_parse() -> CustomizationConfig:
        raw = fetch_cached_settings(data_id, group, use_cache=use_cache)
        if not raw:
            return CustomizationConfig()
        try:
            return CustomizationConfig.model_validate(raw)
        except Exception as exc:
            logger.warning(
                "RemoteConfig customization config parse failed (%s/%s): %s",
                data_id,
                group,
                exc,
            )
            return CustomizationConfig()

    return get_or_compute(
        (data_id, group),
        _load_and_parse,
        use_cache=use_cache,
    )


__all__ = [
    "CUSTOMIZATION_DATA_ID",
    "CUSTOMIZATION_GROUP",
    "CUSTOMIZATION_REPO",
    "CustomizationConfig",
    "fetch_customization_config",
]
