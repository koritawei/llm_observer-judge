"""BullMQ ``queue_prefix`` 归属策略（``envSign`` / ASI stage 驱动）。

标准环境（短 ``envSign``：``testing`` / ``staging`` / ``gray`` / ``production``，
或 ``SIGMA_APP_STAGE``：``DAILY`` / ``PRE_PUBLISH`` / … / ``PUBLISH``）：
强制使用固定裸前缀 ``daily`` / ``pre`` / ``prod``，并**覆盖**业务对
``EVAL_QUEUE_PREFIX`` 的修改；YAML ``node_slots`` 的 ``{forced}_*`` /
``{forced}-*`` 派生前缀仍保留（运维钉死节点）。

临时 / 项目环境（``envSign`` 长度 > 20，形如 ``aone-20221028-xxx``）：
**禁止**使用保留默认前缀，必须由业务设置非保留 ``EVAL_QUEUE_PREFIX``。

本地无 ``envSign`` 时：回退 ``AGENT_EVAL_ENV``（daily/pre/prod→标准强制；
其它→按临时环境要求自定义）。无平台信号却靠 profile 判标准泳道时，启动应
打 warning（``Settings.env`` 默认 ``daily``，易误写入共享日常 Redis）；本地
请显式 ``AGENT_EVAL_DEPLOY_KIND=local`` + 非保留 ``EVAL_QUEUE_PREFIX``。
``AGENT_EVAL_DEPLOY_KIND=personal|…`` 可强制按临时环境处理。
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

#: ``envSign`` 超过该长度视为临时/项目环境（平台约定）
TEMPORARY_ENV_SIGN_MIN_LEN = 20

#: 标准短 envSign → 固定 queue_prefix
STANDARD_ENV_SIGN_TO_PREFIX: Mapping[str, str] = {
    "testing": "daily",
    "staging": "pre",
    "gray": "pre",
    "production": "prod",
}

#: ASI ``SIGMA_APP_STAGE``（无 envSign 时）→ 固定 queue_prefix
SIGMA_APP_STAGE_TO_PREFIX: Mapping[str, str] = {
    "DAILY": "daily",
    "PRE_PUBLISH": "pre",
    "GRAY": "pre",
    "SMALLFLOW": "pre",
    "PUBLISH": "prod",
}

#: 标准部署 profile（``Settings.env`` / ``AGENT_EVAL_ENV``）回退用
STANDARD_DEPLOY_PROFILES: frozenset[str] = frozenset({"daily", "pre", "prod"})

#: 裸保留前缀：仅标准环境强制使用；临时环境禁止
RESERVED_QUEUE_PREFIXES: frozenset[str] = frozenset({"daily", "pre", "prod"})

#: profile → 入库 YAML / 标准强制前缀
STANDARD_PREFIX_BY_PROFILE: Mapping[str, str] = {
    "daily": "daily",
    "pre": "pre",
    "prod": "prod",
}

#: 显式声明为业务/本地/vibe → 按临时环境强制自定义前缀
PERSONAL_DEPLOY_KINDS: frozenset[str] = frozenset(
    {"personal", "business", "dev", "local", "vibe"},
)

#: BullMQ Redis 键空间世代（与裸泳道 ``daily`` / ``pre`` / ``prod`` 正交）。
#: 逻辑前缀 = ``{lane}:bullmq:{generation}``。递增后新旧节点不再共享
#: wait / active / inflight 键，历史 ``{lane}:bullmq``（无世代段）自动失效，
#: 避免未升级节点继续消费新入队 job。
#:
#: **不改变** ``resolve_queue_prefix`` / ``RESERVED_QUEUE_PREFIXES`` 业务限制：
#: 标准环境仍强制裸 ``daily``/``pre``/``prod``；临时/隔离环境仍禁止占用保留字、
#: 必须自定义非保留 ``EVAL_QUEUE_PREFIX``。世代只拼在 Redis 键上，不入库 YAML、
#: 不改 ``settings.eval_queue_prefix``。
BULLMQ_KEYSPACE_GENERATION = "v2"


class QueuePrefixPolicyError(ValueError):
    """``queue_prefix`` 违反归属策略（启动应 fail-fast，勿降级空跑 BullMQ）。"""


@dataclass(frozen=True, slots=True)
class QueuePrefixResolveInput:
    """``resolve_queue_prefix`` 入参（FUNC-PARAMS-001 收敛）。"""

    deploy_profile: str
    yaml_prefix: str
    env_prefix: str
    env_sign: str = ""
    sigma_app_stage: str = ""
    deploy_kind: str = ""


@dataclass(frozen=True, slots=True)
class RuntimeLaneDetectInput:
    """``detect_runtime_lane`` 入参（FUNC-PARAMS-001 收敛）。"""

    env_sign: str = ""
    sigma_app_stage: str = ""
    deploy_profile: str = ""
    deploy_kind: str = ""


def normalize_queue_prefix(raw: str | None) -> str:
    """去掉首尾空白；不做大小写折叠（前缀大小写敏感）。"""
    return (raw or "").strip()


def build_bullmq_logic_prefix(queue_prefix: str) -> str:
    """由**已解析的裸泳道**构造 BullMQ / inflight 逻辑前缀（含键空间世代）。

    入参须是 ``resolve_queue_prefix`` 的结果（或等价裸前缀），本函数**不做**
    标准/隔离归属判定，也不改写保留字集合——只追加 ``:bullmq:{generation}``。

    Examples:
        ``pre`` → ``pre:bullmq:v2``；``dev-123`` → ``dev-123:bullmq:v2``。
        历史无世代段的 ``pre:bullmq`` 不再被新代码使用。
    """
    lane = normalize_queue_prefix(queue_prefix)
    if not lane:
        raise QueuePrefixPolicyError(
            "eval_queue_prefix 为空：须先经 resolve_queue_prefix 写入，"
            "禁止回退 environment 以免污染 pre/prod 队列。"
        )
    return f"{lane}:bullmq:{BULLMQ_KEYSPACE_GENERATION}"


def normalize_deploy_profile(raw: str | None) -> str:
    """部署 profile 归一：去空白并小写。"""
    return (raw or "").strip().lower()


def normalize_deploy_kind(raw: str | None) -> str:
    """部署 kind 归一：去空白并小写；空表示未声明。"""
    return (raw or "").strip().lower()


def is_reserved_queue_prefix(prefix: str) -> bool:
    """是否为裸保留前缀。"""
    return normalize_queue_prefix(prefix) in RESERVED_QUEUE_PREFIXES


def is_temporary_env_sign(env_sign: str | None) -> bool:
    """``envSign`` 长度 > 20 → 临时/项目环境。"""
    return len((env_sign or "").strip()) > TEMPORARY_ENV_SIGN_MIN_LEN


def map_standard_env_sign_to_prefix(env_sign: str | None) -> str | None:
    """短标准 ``envSign`` → 固定前缀；未知返回 None。"""
    key = (env_sign or "").strip().lower()
    if not key:
        return None
    return STANDARD_ENV_SIGN_TO_PREFIX.get(key)


def map_sigma_stage_to_prefix(sigma_app_stage: str | None) -> str | None:
    """``SIGMA_APP_STAGE`` → 固定前缀；未知返回 None。"""
    key = (sigma_app_stage or "").strip().upper()
    if not key:
        return None
    return SIGMA_APP_STAGE_TO_PREFIX.get(key)


def is_standard_scoped_prefix(prefix: str, *, expected: str) -> bool:
    """标准环境下允许的 YAML 前缀：裸 ``expected``，或 ``expected_`` / ``expected-`` 派生。"""
    value = normalize_queue_prefix(prefix)
    if not value or not expected:
        return False
    if value == expected:
        return True
    return value.startswith(f"{expected}_") or value.startswith(f"{expected}-")


def has_platform_lane_signal(*, env_sign: str = "", sigma_app_stage: str = "") -> bool:
    """是否具备可判定泳道的平台信号（短/长 envSign 或已知 SIGMA stage）。"""
    sign = (env_sign or "").strip()
    if is_temporary_env_sign(sign):
        return True
    if map_standard_env_sign_to_prefix(sign) is not None:
        return True
    return map_sigma_stage_to_prefix(sigma_app_stage) is not None


def uses_ambiguous_standard_profile_fallback(request: RuntimeLaneDetectInput) -> bool:
    """无平台信号、未声明 personal kind，却靠 ``deploy_profile`` 判成标准泳道。

    典型风险：本地/CI 未设任何环境变量时 ``Settings.env`` 默认 ``daily``，
    会强制 ``daily`` 前缀并可能写入共享日常 Redis。
    """
    if normalize_deploy_kind(request.deploy_kind) in PERSONAL_DEPLOY_KINDS:
        return False
    if has_platform_lane_signal(
        env_sign=request.env_sign,
        sigma_app_stage=request.sigma_app_stage,
    ):
        return False
    profile = normalize_deploy_profile(request.deploy_profile)
    return profile in STANDARD_DEPLOY_PROFILES


def detect_runtime_lane(request: RuntimeLaneDetectInput) -> tuple[str, str | None]:
    """判定运行泳道。

    Returns:
        ``(lane, forced_prefix)``：
        - ``standard``：``forced_prefix`` 为必须使用的固定裸前缀
        - ``temporary``：``forced_prefix`` 为 None，必须业务自指定非保留前缀
    """
    if normalize_deploy_kind(request.deploy_kind) in PERSONAL_DEPLOY_KINDS:
        return "temporary", None

    sign = (request.env_sign or "").strip()
    if is_temporary_env_sign(sign):
        return "temporary", None

    forced = map_standard_env_sign_to_prefix(sign)
    if forced is not None:
        return "standard", forced

    forced = map_sigma_stage_to_prefix(request.sigma_app_stage)
    if forced is not None:
        return "standard", forced

    profile = normalize_deploy_profile(request.deploy_profile)
    if profile in STANDARD_DEPLOY_PROFILES:
        return "standard", STANDARD_PREFIX_BY_PROFILE[profile]
    return "temporary", None


def resolve_queue_prefix(request: QueuePrefixResolveInput) -> str:
    """解析生效 ``queue_prefix``（标准强制固定值；临时必须业务指定）。

    - **标准**：业务 ``EVAL_QUEUE_PREFIX`` 若与固定值不同则覆盖为固定值；
      YAML（含 ``node_slots``）若为 ``forced`` 或 ``forced_*`` / ``forced-*`` 则保留。
    - **临时**：必须非保留 ``EVAL_QUEUE_PREFIX``（或非保留 YAML）；禁止 daily/pre/prod。
    """
    lane, forced = detect_runtime_lane(
        RuntimeLaneDetectInput(
            env_sign=request.env_sign,
            sigma_app_stage=request.sigma_app_stage,
            deploy_profile=request.deploy_profile,
            deploy_kind=request.deploy_kind,
        )
    )
    env_p = normalize_queue_prefix(request.env_prefix)
    yaml_p = normalize_queue_prefix(request.yaml_prefix)

    if lane == "standard":
        if forced is None:
            raise QueuePrefixPolicyError("标准泳道缺少 forced prefix（内部错误）")
        # 运维 node_slots 钉死前缀优先于裸 forced（仍须落在 forced 命名空间内）
        if yaml_p and is_standard_scoped_prefix(yaml_p, expected=forced):
            return yaml_p
        return forced

    if env_p:
        if is_reserved_queue_prefix(env_p):
            raise QueuePrefixPolicyError(
                "临时/项目环境禁止使用保留前缀 "
                f"{env_p!r}（envSign={request.env_sign!r}）。"
                "必须设置非保留 EVAL_QUEUE_PREFIX，例如 dev-<工号>。"
            )
        return env_p

    if yaml_p and not is_reserved_queue_prefix(yaml_p):
        return yaml_p

    raise QueuePrefixPolicyError(
        "临时/项目环境禁止使用标准默认前缀 daily/pre/prod，"
        "必须设置非保留 EVAL_QUEUE_PREFIX"
        f"（envSign={request.env_sign!r} sigma={request.sigma_app_stage!r} "
        f"profile={request.deploy_profile!r} yaml={yaml_p!r}）。"
    )


def assert_committed_standard_yaml_prefixes(
    conf_dir: Path,
    *,
    profiles: Iterable[str] | None = None,
    missing: Literal["error", "warn"] = "error",
    warn: Callable[[str], None] | None = None,
) -> None:
    """锁死入库 ``app.{daily,pre,prod}.yaml`` 的裸 ``queue_prefix``（单测 / 启动）。

    只读通用 ``custom.bullmq.queue_prefix``，不解析 ``node_slots`` 覆盖。

    Args:
        conf_dir: ``conf/`` 目录。
        profiles: 仅校验这些 profile；``None`` 表示三份全检（单测默认）。
            启动建议只传当前 ``Settings.env``，避免镜像只挂载一份 yaml 时误杀。
        missing: 缺文件时 ``error`` 抛错，``warn`` 则回调 ``warn`` 后跳过。
        warn: ``missing="warn"`` 时的告警回调；未传则静默跳过缺文件。
    """
    if profiles is None:
        targets: Mapping[str, str] = STANDARD_PREFIX_BY_PROFILE
    else:
        targets = {
            normalize_deploy_profile(p): STANDARD_PREFIX_BY_PROFILE[normalize_deploy_profile(p)]
            for p in profiles
            if normalize_deploy_profile(p) in STANDARD_PREFIX_BY_PROFILE
        }
    for profile, expected in targets.items():
        path = conf_dir / f"app.{profile}.yaml"
        if not path.is_file():
            msg = f"缺少标准配置文件: {path}"
            if missing == "warn":
                if warn is not None:
                    warn(msg)
                continue
            raise QueuePrefixPolicyError(msg)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        custom = raw.get("custom") if isinstance(raw, dict) else None
        bullmq = custom.get("bullmq") if isinstance(custom, dict) else None
        prefix = ""
        if isinstance(bullmq, dict):
            prefix = normalize_queue_prefix(str(bullmq.get("queue_prefix") or ""))
        if prefix != expected:
            raise QueuePrefixPolicyError(
                f"{path.name} 的 custom.bullmq.queue_prefix 必须为 {expected!r}，"
                f"当前={prefix!r}。业务自定义前缀只能走临时环境的 EVAL_QUEUE_PREFIX，"
                "禁止污染标准 YAML。"
            )


__all__ = [
    "PERSONAL_DEPLOY_KINDS",
    "QueuePrefixPolicyError",
    "QueuePrefixResolveInput",
    "RESERVED_QUEUE_PREFIXES",
    "RuntimeLaneDetectInput",
    "SIGMA_APP_STAGE_TO_PREFIX",
    "STANDARD_DEPLOY_PROFILES",
    "STANDARD_ENV_SIGN_TO_PREFIX",
    "STANDARD_PREFIX_BY_PROFILE",
    "TEMPORARY_ENV_SIGN_MIN_LEN",
    "BULLMQ_KEYSPACE_GENERATION",
    "assert_committed_standard_yaml_prefixes",
    "build_bullmq_logic_prefix",
    "detect_runtime_lane",
    "has_platform_lane_signal",
    "is_reserved_queue_prefix",
    "is_standard_scoped_prefix",
    "is_temporary_env_sign",
    "map_sigma_stage_to_prefix",
    "map_standard_env_sign_to_prefix",
    "normalize_deploy_kind",
    "normalize_deploy_profile",
    "normalize_queue_prefix",
    "resolve_queue_prefix",
    "uses_ambiguous_standard_profile_fallback",
]
