"""ProfileResolver — 请求 profile → ResolvedProfile（设计文档 §5.2）。

职责：``rubric_ref`` 类加载（白名单）/ judge·归因 ``execution + target``
推导与校验 / 拦截器加载 / ``config_json`` 快照组装。

**快照即真相**：回调侧经 ``load_rubric_from_config`` 重建 rubric（F7 复用）；
worker / 归因侧经快照里的 ``judge.execution / target`` 路由执行方式 ——
进程重启 / 多节点 / 回调延迟场景下行为一致。快照**永不携带 api_key**
（§2 冻结项）。
"""

from dataclasses import dataclass
from typing import Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.eval_plugin_config import resolve_plugin_binding
from benchmark.runs.common.external_client_factory import (
    ExternalClientFactory,
    InvalidTargetUrlError,
)
from benchmark.runs.common.dimension_stats import serialize_stat_dimensions
from benchmark.runs.common.execution_classes import (
    ExecutionClassResolutionError,
    validate_execution_class_protocol,
)
from benchmark.runs.common.parse_interceptor import DataParseInterceptor
from benchmark.runs.common.rubric_base import BaseRubric, default_rubric_ref
from benchmark.runs.schema import BenchmarkRuntimeProfile
from benchmark.reporting.framework.base import BenchmarkComponentMetadata
from benchmark.reporting.framework.schema import AggregationConfig, Criterion
from benchmark.plugins.class_loader import PluginClassLoadError, load_plugin_class
from benchmark.plugins.remote_parse import RemoteParseInterceptor

logger = get_logger("agent_eval.benchmark.common.profile_resolver")

#: URL 形态前缀（execution 推导：http(s):// → async，否则 → sync）
_URL_PREFIXES = ("http://", "https://")


class ProfileResolutionError(ValueError):
    """profile 解析失败（→ API 400，不进流水线）。

    ``reason`` 为机器可读错误码（``invalid_rubric_ref`` /
    ``invalid_judge_target`` / ``invalid_attribution_target`` /
    ``invalid_parse_interceptor_ref``），观测日志 ``profile.resolve_failed``。
    """

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def _is_url(value: str) -> bool:
    return value.startswith(_URL_PREFIXES)


def load_class_ref(ref: str, *, what: str) -> type:
    """加载 ``module:Class`` 类引用（与插件宿主共用 importlib + 白名单前缀）。

    Args:
        ref: ``module:Class`` 引用串。
        what: 引用用途（错误码后缀：``invalid_{what}``）。

    Raises:
        ProfileResolutionError: 格式非法 / 越白名单 / 导入失败 / 目标非类。
    """
    try:
        cls = load_plugin_class(ref, what=what)
    except PluginClassLoadError as exc:
        raise ProfileResolutionError(f"invalid_{what}", str(exc)) from exc
    return cls


@dataclass(frozen=True)
class ResolvedJudge:
    """judge 解析结果（per-run）。

    ``execution=async`` 时 ``target`` 为完整 trigger URL；``sync`` 时
    ``target`` 为本地评测类名原文（``local_eval_class`` 为已加载类，
    ``None`` = 现网默认 EvaluatorService 路径）。
    """

    execution: str
    target: str | None
    provider: str
    dims: str
    prompt_version: str | None
    local_eval_class: type | None
    #: 执行类引用（judge_class 别名解析）；None = 现网默认实现
    class_ref: str | None = None


@dataclass(frozen=True)
class ResolvedAttribution:
    """归因解析结果（per-run）。``execution=none`` 时 target/provider 不生效。"""

    execution: str
    provider: str
    target: str | None
    do_cross_verify: bool
    local_attr_class: type | None
    #: 执行类引用（attribution_class 别名解析）；None = 现网默认实现
    class_ref: str | None = None


@dataclass(frozen=True)
class ResolvedProfile:
    """Resolver 输出：rubric 实例 + judge/归因配置 + 拦截器 + 快照组装。"""

    benchmark_name: str
    rubric: list[Criterion]
    rubric_ref: str | None
    aggregation: AggregationConfig
    parse_interceptor_ref: str | None
    parse_interceptor: DataParseInterceptor | None
    judge: ResolvedJudge
    attribution: ResolvedAttribution
    #: 评分规则文档链接（Rubric 类 ``rule_link_url`` 快照；None = 无链接）
    rule_link_url: str | None = None
    #: 统计维度快照（Rubric 类 ``stat_dimensions``；空 = 无维度展开）
    stat_dimensions: tuple[Any, ...] = ()

    def config_snapshot(self) -> dict[str, Any]:
        """config_json 快照片段（键名与 F7 ``load_*_from_config`` 对齐）。

        只含可序列化配置；**不含 api_key**（冻结项，§2 / §6.3）。
        """
        return {
            "rubric": [
                {
                    "id": c.id,
                    "name": c.name,
                    "weight": c.weight,
                    "evaluator_ref": c.evaluator_ref,
                    "description": c.description,
                }
                for c in self.rubric
            ],
            "aggregation": {
                "method": self.aggregation.method,
                "pass_threshold": self.aggregation.pass_threshold,
                "fusion_fn_ref": self.aggregation.fusion_fn_ref,
            },
            "rubric_ref": self.rubric_ref,
            "parse_interceptor_ref": self.parse_interceptor_ref,
            "rule_link_url": self.rule_link_url,
            # 统计维度快照：run 级配置真相（多节点一致），聚合/报告侧从
            # config_json 重建（优先于进程内注册表）
            "stat_dimensions": serialize_stat_dimensions(self.stat_dimensions),
            "judge": {
                "execution": self.judge.execution,
                "target": self.judge.target,
                "provider": self.judge.provider,
                "dims": self.judge.dims,
                "prompt_version": self.judge.prompt_version,
                "class_ref": self.judge.class_ref,
            },
            "attribution": {
                "execution": self.attribution.execution,
                "target": self.attribution.target,
                "provider": self.attribution.provider,
                "do_cross_verify": self.attribution.do_cross_verify,
                "class_ref": self.attribution.class_ref,
            },
        }


@dataclass(frozen=True)
class _AsyncTargetDefaults:
    """``_resolve_async_target`` 默认/回退配置（FUNC-PARAMS-001 参数收敛）。

    ``what`` 为错误归类前缀（``invalid_{what}``）；``fallback_url``
    为缺省回退 URL（async 归因缺省回退 judge.target，对齐现网 F3）。
    """

    default_url: str | None
    what: str
    fallback_url: str | None = None


class ProfileResolver:
    """请求 profile → ResolvedProfile（校验失败抛 ``ProfileResolutionError``）。

    Args:
        client_factory: 进程级 AoneClientFactory（``None`` = Aone 未配置，
            此时显式 async target 解析失败，缺省 judge 回退本地 sync —
            对齐现网 ``_resolve_judge_service`` 回退语义，F3）。
    """

    def __init__(self, *, client_factory: ExternalClientFactory | None = None) -> None:
        self._client_factory = client_factory

    def resolve(
        self,
        profile: BenchmarkRuntimeProfile,
        meta: BenchmarkComponentMetadata,
    ) -> ResolvedProfile:
        """解析并校验 profile，返回 ResolvedProfile（含 config_json 快照）。"""
        rubric, rubric_ref, rubric_cls = self._resolve_rubric(profile, meta)
        if profile.aggregation is not None:
            aggregation = profile.aggregation
        elif rubric_cls is not None:
            aggregation = rubric_cls.default_aggregation()
        else:
            aggregation = meta.aggregation
        interceptor_ref = profile.parse_interceptor_ref or (
            rubric_cls.parse_interceptor_ref if rubric_cls is not None else None
        )
        # component_rubric 跳过默认 Rubric 映射时 criteria 来自 meta，但会丢掉
        # Rubric 类上的 parse_interceptor_ref；对有默认映射的业务回退挂载，
        # 避免 OSS 宽表 passthrough 后 input 为空 → judge「query is required」。
        if interceptor_ref is None:
            interceptor_ref = self._default_parse_interceptor_ref(profile.benchmark_name)
        interceptor = self._load_interceptor(interceptor_ref)
        judge = self._resolve_judge(profile, rubric)
        self._enforce_judge_policy(rubric_cls, profile.benchmark_name, judge)
        attribution = self._resolve_attribution(profile, judge)
        resolved = ResolvedProfile(
            benchmark_name=profile.benchmark_name,
            rubric=rubric,
            rubric_ref=rubric_ref,
            aggregation=aggregation,
            parse_interceptor_ref=interceptor_ref,
            parse_interceptor=interceptor,
            judge=judge,
            attribution=attribution,
            rule_link_url=getattr(rubric_cls, "rule_link_url", None),
            stat_dimensions=(
                tuple(getattr(rubric_cls, "stat_dimensions", ()) or ())
                if rubric_cls is not None
                else ()
            ),
        )
        logger.info(
            "profile.resolved benchmark=%s rubric_ref=%s interceptor=%s "
            "judge=%s/%s attribution=%s/%s criteria=%d",
            profile.benchmark_name,
            rubric_ref,
            interceptor_ref,
            judge.execution,
            judge.provider,
            attribution.execution,
            attribution.provider,
            len(rubric),
        )
        return resolved

    # ── rubric（§6.1）────────────────────────────────────────────────────

    def _resolve_rubric(
        self,
        profile: BenchmarkRuntimeProfile,
        meta: BenchmarkComponentMetadata,
    ) -> tuple[list[Criterion], str | None, type[BaseRubric] | None]:
        """rubric_ref 类加载；缺省回退（§9）：默认映射 → component 元数据 rubric。

        ``component_rubric=True``（``rubric_class=component_rubric``）跳过默认映射，
        直接使用 component 元数据 rubric。
        """
        rubric_ref = profile.rubric_ref
        if not rubric_ref and not profile.component_rubric:
            rubric_ref = default_rubric_ref(profile.benchmark_name)
        if not rubric_ref:
            return list(meta.rubric), None, None
        cls = load_class_ref(rubric_ref, what="rubric_ref")
        criteria_fn = getattr(cls, "criteria", None)
        if not callable(criteria_fn):
            raise ProfileResolutionError(
                "invalid_rubric_ref",
                f"rubric class {rubric_ref!r} has no callable criteria()",
            )
        try:
            criteria = criteria_fn()
        except Exception as exc:
            raise ProfileResolutionError(
                "invalid_rubric_ref",
                f"rubric criteria() raised: {exc}",
            ) from exc
        if not isinstance(criteria, list) or not all(isinstance(c, Criterion) for c in criteria):
            raise ProfileResolutionError(
                "invalid_rubric_ref",
                f"rubric criteria() must return list[Criterion], got {type(criteria).__name__}",
            )
        if not criteria:
            raise ProfileResolutionError(
                "invalid_rubric_ref",
                f"rubric criteria() must be non-empty: {rubric_ref!r}",
            )
        rubric_cls = cls if issubclass(cls, BaseRubric) else None
        # 鸭子类型放行：非 BaseRubric 子类但 criteria() 合法时，类属性取默认
        if rubric_cls is None:
            logger.info(
                "profile.rubric_duck_typed ref=%s (not BaseRubric subclass; "
                "aggregation/interceptor class attrs ignored)",
                rubric_ref,
            )
        return list(criteria), rubric_ref, rubric_cls

    # ── 拦截器（§7.5）──────────────────────────────────────────────────

    @staticmethod
    def _default_parse_interceptor_ref(benchmark_name: str) -> str | None:
        """``component_rubric`` 等丢失 Rubric 类时，按默认映射回退拦截器。"""
        class_ref = default_rubric_ref(benchmark_name)
        if not class_ref:
            return None
        try:
            cls = load_class_ref(class_ref, what="rubric_ref")
        except ProfileResolutionError:
            return None
        interceptor = getattr(cls, "parse_interceptor_ref", None)
        return interceptor if isinstance(interceptor, str) and interceptor.strip() else None

    def _load_interceptor(self, ref: str | None) -> DataParseInterceptor | None:
        """加载拦截器实例（local 类 或 RemoteConfig remote）；``None`` = 透传。"""
        if not ref:
            return None
        binding = resolve_plugin_binding("parse_interceptors", ref)
        if binding is not None and not binding.enabled:
            raise ProfileResolutionError(
                "invalid_parse_interceptor_ref",
                f"parse interceptor {ref!r} is disabled by RemoteConfig eval plugins",
            )
        if binding is not None and binding.is_remote_active():
            return RemoteParseInterceptor(binding)

        cls = load_class_ref(ref, what="parse_interceptor_ref")
        try:
            instance = cls()
        except Exception as exc:
            raise ProfileResolutionError(
                "invalid_parse_interceptor_ref",
                f"parse interceptor {ref!r} instantiation failed: {exc}",
            ) from exc
        if not isinstance(instance, DataParseInterceptor):
            raise ProfileResolutionError(
                "invalid_parse_interceptor_ref",
                f"parse interceptor {ref!r} does not implement DataParseInterceptor",
            )
        return instance

    # ── judge（§6.3）────────────────────────────────────────────────

    @staticmethod
    def _enforce_judge_policy(
        rubric_cls: "type[BaseRubric] | None",
        benchmark_name: str,
        judge: "ResolvedJudge",
    ) -> None:
        """Rubric 类 judge_execution_policy 校验（fail-fast 防线，400）。

        声明 ``local`` 的 rubric（如 SearchEvalRubric：本地 evaluator 一次
        调用产出 0/1/2 总档 + 六维 sub_scores，维度体系与 Aone 通用管线
        不互通）不允许 async：显式 async 或缺省推导 async（RemoteConfig
        external-daily 项 judge 缺省 → 进程默认 pipeline）一律拒绝——
        20260923 线上事故：search_eval 缺省推导 async + e2e 管线，
        search_ai_card 维度被 ALL_DIMENSIONS=['D0'..'D6'] 白名单拒绝，
        全量 case 失败。``rubric_cls=None``（无 rubric_ref 的 meta 兜底路径）
        无政策可查，跳过。
        """
        policy = getattr(rubric_cls, "judge_execution_policy", None) if rubric_cls else None
        if policy == "local" and judge.execution == "async":
            raise ProfileResolutionError(
                "judge_execution_policy_violation",
                f"benchmark {benchmark_name!r} rubric declares "
                f"judge_execution_policy='local' (本地 sync 评测)，async judge "
                f"（显式或缺省推导）被拒绝；请在请求/RemoteConfig external-daily "
                f"项配置 sync 形态的 judge_class 别名",
            )

    def _resolve_judge(
        self,
        profile: BenchmarkRuntimeProfile,
        rubric: list[Criterion],
    ) -> ResolvedJudge:
        cfg = profile.judge
        target = (cfg.target or "").strip() or None
        # execution 推导（§6.3 约束 3）：显式优先；缺省按 target 形态
        execution = cfg.execution
        if execution is None:
            if target is not None:
                execution = "async" if _is_url(target) else "sync"
            else:
                execution = (
                    "async"
                    if self._client_factory is not None and self._client_factory.default_judge_url
                    else "sync"
                )

        local_eval_class: type | None = None
        resolved_target: str | None = target
        if execution == "async":
            resolved_target = self._resolve_async_target(
                target,
                _AsyncTargetDefaults(
                    default_url=(
                        self._client_factory.default_judge_url
                        if self._client_factory is not None
                        else None
                    ),
                    what="judge_target",
                ),
            )
        elif target is not None:
            local_eval_class = load_class_ref(target, what="judge_target")

        provider = cfg.provider or ("external" if execution == "async" else "local")
        dims = cfg.dims or ",".join(c.id for c in rubric)
        return ResolvedJudge(
            execution=execution,
            target=resolved_target,
            provider=provider,
            dims=dims,
            prompt_version=cfg.prompt_version,
            local_eval_class=local_eval_class,
            class_ref=self._validate_class_ref(cfg.class_ref, what="judge_class"),
        )

    @staticmethod
    def _validate_class_ref(class_ref: str | None, *, what: str) -> str | None:
        """执行类引用白名单 + 协议校验（进快照前 fail-fast，防 worker 侧才加载失败）。

        白名单前缀与可加载性由 ``load_class_ref`` 保证；此处进一步校验协议合规
        （execution/provider 声明合法、async 必需方法齐备），与别名解析路径
        （``execution_classes._resolve_execution_class``）同标准——API 调用方直传
        ``class_ref`` 时不会把不合规类写入快照。
        """
        ref = (class_ref or "").strip() or None
        if ref is not None:
            cls = load_class_ref(ref, what=what)
            kind = "judge" if what == "judge_class" else "attribution"
            try:
                validate_execution_class_protocol(cls, kind=kind, alias=ref)
            except ExecutionClassResolutionError as exc:
                raise ProfileResolutionError(f"invalid_{what}", str(exc)) from exc
        return ref

    # ── 归因（§6.4）────────────────────────────────────────────────────

    def _resolve_attribution(
        self,
        profile: BenchmarkRuntimeProfile,
        judge: ResolvedJudge,
    ) -> ResolvedAttribution:
        cfg = profile.attribution
        target = (cfg.target or "").strip() or None
        execution = cfg.execution
        provider = cfg.provider
        # provider / execution 联动（§5.1）：显式优先，缺省由 target 形态推导
        if execution is None:
            if provider == "none":
                execution = "none"
            elif target is not None:
                execution = "async" if _is_url(target) else "sync"
            elif provider == "external":
                execution = "async"
            else:
                execution = "sync"
        if provider is None:
            if execution == "none":
                provider = "none"
            elif execution == "async":
                provider = "external"
            else:
                # 缺省保持现网 pull 默认（example = 本地归因 example agent）
                provider = "local" if target is not None else "example"

        local_attr_class: type | None = None
        resolved_target: str | None = target
        if execution == "async":
            resolved_target = self._resolve_async_target(
                target,
                _AsyncTargetDefaults(
                    default_url=(
                        self._client_factory.default_attribution_url
                        if self._client_factory is not None
                        else None
                    ),
                    what="attribution_target",
                    # async 缺省回退 judge.target URL（对齐现网 F3 回退语义）
                    fallback_url=judge.target if judge.execution == "async" else None,
                ),
            )
        elif execution == "sync" and target is not None:
            local_attr_class = load_class_ref(target, what="attribution_target")
        elif execution == "none":
            resolved_target = None

        return ResolvedAttribution(
            execution=execution,
            provider=provider,
            target=resolved_target,
            do_cross_verify=(cfg.do_cross_verify if cfg.do_cross_verify is not None else True),
            local_attr_class=local_attr_class,
            class_ref=self._validate_class_ref(cfg.class_ref, what="attribution_class"),
        )

    # ── 共用：async target URL 解析 + 白名单校验 ─────────────────────────

    def _resolve_async_target(
        self,
        target: str | None,
        defaults: "_AsyncTargetDefaults",
    ) -> str:
        """async target 归一：显式 URL → 进程默认 → 回退 URL；全程白名单校验。"""
        default_url = defaults.default_url
        what = defaults.what
        fallback_url = defaults.fallback_url
        reason = f"invalid_{what}"
        url = target or default_url or fallback_url
        if not url:
            raise ProfileResolutionError(
                reason,
                f"{what}: async execution requires a trigger URL "
                "(explicit target, configured default, or judge fallback)",
            )
        if self._client_factory is None:
            raise ProfileResolutionError(
                reason,
                f"{what}: async execution requires Aone client factory configured",
            )
        try:
            self._client_factory.validate_target_url(url)
        except InvalidTargetUrlError as exc:
            raise ProfileResolutionError(reason, str(exc)) from exc
        return url


__all__ = [
    "ProfileResolutionError",
    "ProfileResolver",
    "ResolvedAttribution",
    "ResolvedJudge",
    "ResolvedProfile",
    "load_class_ref",
]
