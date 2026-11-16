"""归因触发判定策略 — 公共策略协议 + 默认实现 + 业务注册。

等级体系对照 ``services/agent-eval/docs/评测rubric_v2.md`` §2.3
（0–2 分制加权分）：EXCELLENT ≥ 1.8 / GOOD [1.4,1.8) /
ACCEPTABLE [1.0,1.4) / POOR < 1.0 / FAIL（门控覆写 0）。

使用方式：

- 公共策略协议 :class:`AttributionTriggerPolicy` 只约定
  ``should_trigger``，输入 fused 综合分与 per-dimension 原始分
  （``criterion_scores``，criterion id → score，可缺失）；
- 各业务（benchmark 组件名）经 :func:`register_trigger_policy` 注册
  自己的触发策略实现（同名覆盖，幂等）；
- RemoteConfig ``eval.plugins.attribution_trigger_policies.bindings`` 可按组件名
  改为 ``remote``（HTTP）或 ``class_ref``（免发版本地类），优先级高于注册表；
- 未注册业务回退默认实现 :class:`RubricV2TriggerPolicy`
  （fused < 1.4，即 FAIL ∪ POOR ∪ ACCEPTABLE 触发）；
- ``ieval_external`` 已内置注册 :class:`DimensionFullScoreTriggerPolicy`
  （D0–D4 任一维度不满分即触发，D0 满分 1、D1–D4 满分 2）；
- ``example_benchmark`` 已内置注册 :class:`RubricV2TriggerPolicy`
  （``max_exclusive=EXAMPLE_TRIGGER_MAX_EXCLUSIVE``，0–10 分制 fused < 3 触发）。

async 调用点用 :func:`should_trigger_attribution`（remote 策略走异步 HTTP）。

与 aone 派发策略（``dispatch_policy``）分层：本模块决定「是否归因」，
dispatch_policy 决定「归因后是否派工单」。
"""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.plugins.host import PluginResolution, call_plugin, resolve_plugin
from benchmark.plugins.remote_adapters import RemoteTriggerPolicy

logger = get_logger(__name__)

#: 默认触发阈值 — rubric_v2 GOOD 下界（= amap_chat RUBRIC_V2_GOOD_MIN），
#: fused < 1.4 ⇔ 等级 ∈ {FAIL, POOR, ACCEPTABLE}
RUBRIC_V2_TRIGGER_MAX_EXCLUSIVE = 1.4

#: example_benchmark 触发阈值（0–10 分制）— 与 ExampleRubric.pass_threshold /
#: judge_example prompt「score < 3 → 【失败摘要】」同口径；fused < 3 触发归因。
EXAMPLE_TRIGGER_MAX_EXCLUSIVE = 3.0

#: 维度满分表（ieval_external 触发用）— D0 满分 1，D1–D4 满分 2
DIMENSION_FULL_SCORES: dict[str, float] = {
    "D0": 1.0,
    "D1": 2.0,
    "D2": 2.0,
    "D3": 2.0,
    "D4": 2.0,
}


@runtime_checkable
class AttributionTriggerPolicy(Protocol):
    """归因触发策略协议 — 业务按综合分 / 维度分决定是否触发归因。

    ``fused_score`` 为加权融合分（rubric_v2 0–2 分制），``None`` 表示
    fused 行缺失 / 取分失败；``criterion_scores`` 为 judge 回调的
    per-criterion 原始分（criterion id → score，如 ``{"D0": 1.0}``），
    默认实现只看 fused，维度感知策略（D0–D4）依赖此入参。
    """

    def should_trigger(
        self,
        *,
        fused_score: float | None,
        criterion_scores: Mapping[str, float | None] | None = None,
    ) -> bool: ...


class RubricV2TriggerPolicy:
    """默认触发策略 — fused < 1.4（FAIL ∪ POOR ∪ ACCEPTABLE）。

    ``fused_score=None``（fused 行缺失 / 取分失败）触发 — 沿既有
    「不因缺分漏归因」语义（消费点 None → 0.0 判定等价）。
    """

    def __init__(
        self,
        max_exclusive: float = RUBRIC_V2_TRIGGER_MAX_EXCLUSIVE,
    ) -> None:
        self._max_exclusive = float(max_exclusive)

    @property
    def max_exclusive(self) -> float:
        """触发阈值上界（不含；日志/排查用）。"""
        return self._max_exclusive

    def should_trigger(
        self,
        *,
        fused_score: float | None,
        criterion_scores: Mapping[str, float | None] | None = None,
    ) -> bool:
        if fused_score is None:
            return True
        return fused_score < self._max_exclusive


class DimensionFullScoreTriggerPolicy:
    """维度满分触发策略（ieval_external）— D0–D4 任一不满分即触发。

    满分表：D0 = 1，D1–D4 = 2（评测rubric_v2.md §2.1 分制）。判定规则：

    - 维度 key 存在且 ``score < 满分`` → 触发；``score`` 为 ``None``
      视为不满分（保守触发，不因缺分漏归因）；
    - 维度 key 缺失 → 不参与判定（rubric 未覆盖该维度）；
    - D0–D4 全部缺失（非维度化回调）→ 回退 fallback 策略
      （默认 fused < 1.4），fused 亦为 ``None`` 时触发；
    - 维度 id 大小写容错（``"d0"`` ≡ ``"D0"``）；不在满分表中的维度
      （如 D5/D6）不参与判定。

    有维度参与判定且全部满分 → 不触发（即使 fused 综合分偏低也以
    维度判定为准 — fused 由维度加权而来，口径以维度原始分为权威）。
    """

    def __init__(
        self,
        *,
        fallback: "AttributionTriggerPolicy | None" = None,
        full_scores: Mapping[str, float] = DIMENSION_FULL_SCORES,
    ) -> None:
        self._fallback = fallback
        self._full_scores = {str(k).upper(): float(v) for k, v in full_scores.items()}

    @property
    def full_scores(self) -> dict[str, float]:
        """维度满分表（日志/排查用）。"""
        return dict(self._full_scores)

    def should_trigger(
        self,
        *,
        fused_score: float | None,
        criterion_scores: Mapping[str, float | None] | None = None,
    ) -> bool:
        if criterion_scores:
            normalized = {str(k).upper(): v for k, v in criterion_scores.items()}
            judged = 0
            for dim, full in self._full_scores.items():
                if dim not in normalized:
                    continue
                judged += 1
                score = normalized[dim]
                if score is None or float(score) < full:
                    return True
            if judged:
                return False
        fallback = self._fallback or get_default_trigger_policy()
        return fallback.should_trigger(fused_score=fused_score)


_DEFAULT_TRIGGER_POLICY = RubricV2TriggerPolicy()

#: 业务策略注册表 — benchmark 组件名 → policy（未注册回退默认实现）。
#: 以组件名（如 ``ieval_external``）为键而非 service_name — external
#: 与 inline 可能共享 service_name（ieval_external 的 service_name 为
#: amap_chat），按组件名注册才能只作用于 external 触发路径。
_TRIGGER_POLICY_REGISTRY: dict[str, AttributionTriggerPolicy] = {}


def get_default_trigger_policy() -> AttributionTriggerPolicy:
    """默认策略单例（rubric_v2：fused < 1.4 触发）。"""
    return _DEFAULT_TRIGGER_POLICY


def register_trigger_policy(
    benchmark_name: str,
    policy: AttributionTriggerPolicy,
) -> None:
    """业务注册自己的归因触发策略（同名覆盖，幂等）。"""
    _TRIGGER_POLICY_REGISTRY[benchmark_name] = policy
    logger.info(
        "attribution_trigger_policy.registered benchmark=%s policy=%s",
        benchmark_name,
        type(policy).__name__,
    )


def resolve_trigger_policy(benchmark_name: str | None) -> AttributionTriggerPolicy:
    """按 benchmark 组件名解析触发策略：RemoteConfig binding → 注册表 → 默认实现。"""
    return resolve_plugin(
        "attribution_trigger_policies",
        benchmark_name,
        resolution=PluginResolution(
            registry=_TRIGGER_POLICY_REGISTRY,
            default=_DEFAULT_TRIGGER_POLICY,
            remote_factory=lambda binding: RemoteTriggerPolicy(
                binding,
                fallback=_DEFAULT_TRIGGER_POLICY,
            ),
        ),
    )


async def should_trigger_attribution(
    policy: AttributionTriggerPolicy,
    *,
    fused_score: float | None,
    criterion_scores: Mapping[str, float | None] | None = None,
) -> bool:
    """async 调用点入口：remote 策略走异步 HTTP，本地策略直接同步判定。"""
    return bool(
        await call_plugin(
            policy,
            "should_trigger",
            fused_score=fused_score,
            criterion_scores=criterion_scores,
        )
    )


def clear_trigger_policies() -> None:
    """清空业务注册表并恢复内置注册（测试隔离用）。"""
    _TRIGGER_POLICY_REGISTRY.clear()
    _register_builtin_trigger_policies()


def _register_builtin_trigger_policies() -> None:
    """内置业务注册 — ieval_external（维度满分）+ example_benchmark（<3 分）。"""
    register_trigger_policy(
        "ieval_external",
        DimensionFullScoreTriggerPolicy(),
    )
    register_trigger_policy(
        "example_benchmark",
        RubricV2TriggerPolicy(max_exclusive=EXAMPLE_TRIGGER_MAX_EXCLUSIVE),
    )


_register_builtin_trigger_policies()


__all__ = [
    "AttributionTriggerPolicy",
    "DIMENSION_FULL_SCORES",
    "DimensionFullScoreTriggerPolicy",
    "EXAMPLE_TRIGGER_MAX_EXCLUSIVE",
    "RUBRIC_V2_TRIGGER_MAX_EXCLUSIVE",
    "RubricV2TriggerPolicy",
    "clear_trigger_policies",
    "get_default_trigger_policy",
    "register_trigger_policy",
    "resolve_trigger_policy",
    "should_trigger_attribution",
]
