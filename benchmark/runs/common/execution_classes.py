"""Benchmark Judge / Attribution 执行类别名注册表与执行协议。

外部配置只暴露稳定别名；具体类引用由代码白名单维护。实现类通过类属性
声明执行方式，并承载该执行方式下的业务逻辑：

- sync judge：``async judge_item(LocalJudgeRequest) -> dict``（可选，缺省
  = 现网 EvaluatorService 逐 criterion 评测）；
- sync attribution：``async attribute_item(LocalAttributionRequest)``（可选，
  缺省 = LocalAttributionRunner）；
- async judge：``async submit(AsyncJudgeSubmitRequest) -> job_id``（必需）+
  ``normalize_callback(JudgeCallback)`` classmethod（可选）；
- async attribution：``async submit(AttributionSubmitRequest)`` +
  ``async query_job_status(job_id)``（必需）+ ``normalize_callback``
  classmethod（可选）；构造参数为关键字 ``client_factory`` / ``aone_client``。

ProfileResolver 把解析结果（含 ``class_ref``）冻结为 run 级快照，worker /
回调 / healer 均按快照里的 ``class_ref`` 加载实现，因此 execution/provider
相同的多个实现类可同时配置、互不影响。

streaming 形态（Kafka 实时流）使用**独立别名子集**（§5.1）：批量专属别名
（如 ``ieval_external_judge`` / ``search_eval_judge``）依赖 run_id、回调与
``benchmark_run_items``，streaming 任务不得复用（校验拒绝）。

**统一 EvaluatorService 别名**（batch / streaming 共用，同源零代码路由）：

- ``evaluator_service_judge``：两侧注册表均有；marker 类、``target=None`` →
  按 ``evaluator_ref`` / ``evaluator_name`` 调 ``judge_<ref>``；
- ``evaluator_service_attribution``：两侧注册表均有；缺省同步归因链。

其余 streaming 子集：

- ``STREAMING_JUDGE_CLASS_REFS``：协议 ``StreamingJudgeImpl``；
  ``execution=sync`` 时可选实现 ``evaluate``（缺省 = EvaluatorService 路由）；
  ``execution=async`` 时必须实现 ``submit``（消息路径提交后不等待回调）；
- ``STREAMING_TRIAGE_CLASS_REFS``：triage level 分类器别名；``null`` = 按
  ``evaluator_name`` 查 ``classifiers_registry``；
- ``STREAMING_ATTRIBUTION_CLASS_REFS``：``null`` = 现状同步 triage 归因链；
  ``no_attribution`` = 关闭；显式别名可声明 ``sync`` / ``async``；
  任务级 ``judge_execution`` / ``attribution_execution`` 可覆盖类声明，
  支持 sync+sync / sync+async / async+sync / async+async 组合。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.plugins.class_loader import PluginClassLoadError, load_plugin_class

logger = get_logger("agent_eval.benchmark.common.execution_classes")

if TYPE_CHECKING:
    from benchmark.domain.eval_config import EvalConfigEntry
    from benchmark.evaluators.base import EvaluatorResult
    from benchmark.runs.common.external_client import (
        ExternalJobStatusResponse,
        ExternalJudgeClient,
    )
    from benchmark.runs.schema import AttributionCallback, JudgeCallback
    from benchmark.streaming.schemas import ObservationForEval

Execution = Literal["sync", "async", "none"]


@dataclass(frozen=True)
class AsyncJudgeSubmitRequest:
    """``AsyncJudgeImpl.submit`` 入参：单 case 提交所需的全部上下文。"""

    run_id: str
    item_id: str
    data: dict[str, Any]
    ds: str
    client: "ExternalJudgeClient"
    dims: str | None = None
    prompt_version: str | None = None


@dataclass(frozen=True)
class LocalAttributionRequest:
    """``LocalAttributionImpl.attribute_item`` 入参（FUNC-PARAMS-001 参数收敛）。"""

    run_id: str
    item_id: str
    fused_score: float
    provider: str


class LocalAttributionImpl(Protocol):
    """sync attribution 执行类协议（无参构造，结果自行写库）。"""

    async def attribute_item(self, req: LocalAttributionRequest) -> None:
        """单条本地归因。"""
        ...


class StreamingJudgeImpl(Protocol):
    """streaming judge 执行类协议（无参构造，§5.1）。

    业务专用 judge 类（类比 search_eval local judge）实现本协议并登记到
    ``STREAMING_JUDGE_CLASS_REFS``。实现类只接管**算分**环节——after_eval
    落库与 triage 归因仍走统一层（``EvaluatorService.after_eval``），
    streaming 与批量形态的执行与落库零分叉（§2 统一层原则）。

    ``execution=async`` 时实现 ``submit``（消息路径提交即返回）；``sync`` 时
    可选实现 ``evaluate``（缺省 = EvaluatorService 零代码路由）。
    """

    execution: Execution
    provider: str

    async def evaluate(
        self,
        trace: "ObservationForEval",
        config: "EvalConfigEntry",
    ) -> "EvaluatorResult":
        """单条 trace 评测，返回评分结果（不落库、不归因）。"""
        ...


class StreamingAsyncJudgeImpl(Protocol):
    """streaming async judge：提交后不等待回调（无 run 状态机）。"""

    execution: Execution
    provider: str

    async def submit(
        self,
        trace: "ObservationForEval",
        config: "EvalConfigEntry",
    ) -> str:
        """提交单条 judge，返回 correlator / job_id。"""
        ...


class AsyncJudgeImpl(Protocol):
    """async judge 执行类协议（无参构造）。"""

    async def submit(self, req: AsyncJudgeSubmitRequest) -> str:
        """提交单条 judge，返回平台 job_id（= 回调 job_id / source_trace_id）。"""
        ...


class AsyncAttributionImpl(Protocol):
    """async attribution 执行类协议（关键字构造 ``client_factory`` / ``aone_client``）。"""

    async def submit(self, request: "AttributionSubmitRequest") -> "AttributionSubmitResult":
        """提交单条归因，返回提交结果（含 job_id）。"""
        ...

    async def query_job_status(self, job_id: str) -> "ExternalJobStatusResponse":
        """查询归因 job 状态（healer 超时检测用）。"""
        ...


class LocalJudge:
    """兼容别名 ``local_judge``：语义同 :class:`EvaluatorServiceJudge`。

    新配置请用 ``evaluator_service_judge``；本类保留给 search_eval_v2 等
    存量批量触发与缺省链回退。
    """

    execution: Execution = "sync"
    provider = "local"


class NoAttribution:
    """显式关闭归因的执行类。"""

    execution: Execution = "none"
    provider = "none"


class EvaluatorServiceJudge:
    """EvaluatorService 零代码路由（``evaluator_ref`` → ``judge_<ref>``）。

    batch / streaming 共用别名 ``evaluator_service_judge``；不实现
    ``judge_item`` / ``evaluate`` → ``target=None`` → 现网 EvaluatorService。
    streaming 侧 ``judge_class=null`` 与本别名等价。
    """

    execution: Execution = "sync"
    provider = "local"


class EvaluatorServiceAttributionMarker:
    """批量侧 ``evaluator_service_attribution`` marker（同步默认归因链）。

    不实现 ``attribute_item`` → ``target=None`` → LocalAttributionRunner /
    attribution_services 路由；与 streaming 插件同别名、同源语义。
    """

    execution: Execution = "sync"
    provider = "local"


JUDGE_CLASS_REFS: dict[str, str] = {
    "local_judge": "benchmark.runs.common.execution_classes:LocalJudge",
    "evaluator_service_judge": (
        "benchmark.runs.common.execution_classes:EvaluatorServiceJudge"
    ),
}

ATTRIBUTION_CLASS_REFS: dict[str, str] = {
    "no_attribution": "benchmark.runs.common.execution_classes:NoAttribution",
    "evaluator_service_attribution": (
        "benchmark.runs.common.execution_classes:EvaluatorServiceAttributionMarker"
    ),
}

# ── streaming 形态别名子集（§5.1；与批量注册表互斥，批量别名在 streaming 被拒绝）──

STREAMING_JUDGE_CLASS_REFS: dict[str, str] = {
    "evaluator_service_judge": (
        "benchmark.runs.common.execution_classes:EvaluatorServiceJudge"
    ),
}

STREAMING_TRIAGE_CLASS_REFS: dict[str, str] = {}

STREAMING_ATTRIBUTION_CLASS_REFS: dict[str, str] = {
    "no_attribution": "benchmark.runs.common.execution_classes:NoAttribution",
    "evaluator_service_attribution": (
        "benchmark.streaming.execution_plugins:EvaluatorServiceAttribution"
    ),
}

#: 任务级 execution 覆盖合法取值
STREAMING_JUDGE_EXECUTIONS: frozenset[str] = frozenset({"sync", "async"})
STREAMING_ATTRIBUTION_EXECUTIONS: frozenset[str] = frozenset({"sync", "async", "none"})

#: async 执行类必须实现的方法（sync 方法可选：缺省走现网默认实现）
_ASYNC_REQUIRED_METHODS: dict[str, tuple[str, ...]] = {
    "judge": ("submit",),
    "attribution": ("submit", "query_job_status"),
}
_SYNC_PROTOCOL_METHOD: dict[str, str] = {
    "judge": "judge_item",
    "attribution": "attribute_item",
}


class ExecutionClassResolutionError(ValueError):
    """执行类别名不存在或实现声明非法。"""


@dataclass(frozen=True)
class ExecutionClassSpec:
    """类别名解析后的内部执行规格。"""

    alias: str
    class_ref: str
    execution: Execution
    provider: str
    target: str | None


def validate_execution_class_protocol(
    cls: type,
    *,
    kind: Literal["judge", "attribution"],
    alias: str,
) -> None:
    """执行类协议合规校验（execution/provider/target/async 必需方法）。

    别名解析（``_resolve_execution_class``）与 ``class_ref`` 直传校验
    （``ProfileResolver._validate_class_ref``）共用，保证进快照的类协议一致，
    避免不合规类直到 worker 执行才失败。
    """
    execution = getattr(cls, "execution", None)
    provider = getattr(cls, "provider", None)
    allowed = {"sync", "async"} if kind == "judge" else {"sync", "async", "none"}
    if execution not in allowed or not isinstance(provider, str) or not provider:
        raise ExecutionClassResolutionError(
            f"{kind}_class {alias!r} must declare valid execution/provider",
        )

    target = getattr(cls, "target", None)
    if target is not None and not isinstance(target, str):
        raise ExecutionClassResolutionError(
            f"{kind}_class {alias!r} target must be str or None",
        )
    if execution == "async":
        missing = [m for m in _ASYNC_REQUIRED_METHODS[kind] if not callable(getattr(cls, m, None))]
        if missing:
            raise ExecutionClassResolutionError(
                f"{kind}_class {alias!r} (async) must implement {missing}",
            )


def _resolve_execution_class(
    alias: str,
    *,
    registry: dict[str, str],
    kind: Literal["judge", "attribution"],
) -> ExecutionClassSpec:
    normalized = alias.strip()
    class_ref = registry.get(normalized)
    if class_ref is None:
        choices = ", ".join(sorted(registry))
        raise ExecutionClassResolutionError(
            f"unknown {kind}_class {alias!r}; available aliases: {choices}",
        )
    try:
        cls = load_plugin_class(class_ref, what=f"{kind}_class")
    except PluginClassLoadError as exc:
        raise ExecutionClassResolutionError(str(exc)) from exc

    validate_execution_class_protocol(cls, kind=kind, alias=normalized)

    target = getattr(cls, "target", None)
    execution = getattr(cls, "execution", None)
    provider = getattr(cls, "provider", None)
    sync_method = _SYNC_PROTOCOL_METHOD[kind]
    if execution == "sync" and target is None and callable(getattr(cls, sync_method, None)):
        target = class_ref

    return ExecutionClassSpec(
        alias=normalized,
        class_ref=class_ref,
        execution=execution,
        provider=provider,
        target=target,
    )


def resolve_judge_class(alias: str) -> ExecutionClassSpec:
    """解析 Judge 稳定别名。"""
    return _resolve_execution_class(alias, registry=JUDGE_CLASS_REFS, kind="judge")


def resolve_attribution_class(alias: str) -> ExecutionClassSpec:
    """解析 Attribution 稳定别名。"""
    return _resolve_execution_class(
        alias,
        registry=ATTRIBUTION_CLASS_REFS,
        kind="attribution",
    )


def validate_streaming_judge_protocol(cls: type, *, alias: str) -> None:
    """streaming judge 执行类协议合规校验（§5.1）。

    - ``execution='sync'`` + 非空 ``provider``：``evaluate`` 可选（缺省 =
      EvaluatorService 零代码路由）；
    - ``execution='async'`` + 非空 ``provider``：必须实现 ``submit``。
    """
    execution = getattr(cls, "execution", None)
    provider = getattr(cls, "provider", None)
    if execution not in {"sync", "async"} or not isinstance(provider, str) or not provider:
        raise ExecutionClassResolutionError(
            f"streaming judge_class {alias!r} must declare execution='sync'|'async' "
            "and a non-empty provider",
        )
    if execution == "async" and not callable(getattr(cls, "submit", None)):
        raise ExecutionClassResolutionError(
            f"streaming judge_class {alias!r} (async) must implement submit()",
        )


def resolve_streaming_judge_class(alias: str) -> ExecutionClassSpec:
    """解析 streaming judge 稳定别名（§5.1 streaming 专属子集）。

    与批量侧 :func:`resolve_judge_class` 同构（白名单 + ``load_plugin_class``
    前缀校验）；批量协议别名不在本注册表，传入即拒绝。``target`` 语义：
    sync 且实现 ``evaluate`` → ``class_ref``；async 且实现 ``submit`` →
    ``class_ref``；其余 → ``None``（缺省 = 现网 EvaluatorService 路由）。
    """
    normalized = alias.strip()
    class_ref = STREAMING_JUDGE_CLASS_REFS.get(normalized)
    if class_ref is None:
        choices = ", ".join(sorted(STREAMING_JUDGE_CLASS_REFS))
        raise ExecutionClassResolutionError(
            f"unknown streaming judge_class {alias!r}; available aliases: {choices}",
        )
    try:
        cls = load_plugin_class(class_ref, what="streaming judge_class")
    except PluginClassLoadError as exc:
        raise ExecutionClassResolutionError(str(exc)) from exc
    validate_streaming_judge_protocol(cls, alias=normalized)
    execution = getattr(cls, "execution", "sync")
    has_evaluate = callable(getattr(cls, "evaluate", None))
    has_submit = callable(getattr(cls, "submit", None))
    if execution == "async":
        target = class_ref if has_submit else None
    else:
        target = class_ref if has_evaluate else None
    return ExecutionClassSpec(
        alias=normalized,
        class_ref=class_ref,
        execution=execution,
        provider=cls.provider,
        target=target,
    )


def resolve_streaming_triage_class(alias: str) -> ExecutionClassSpec:
    """解析 streaming triage 稳定别名（``STREAMING_TRIAGE_CLASS_REFS``）。"""
    normalized = alias.strip()
    class_ref = STREAMING_TRIAGE_CLASS_REFS.get(normalized)
    if class_ref is None:
        choices = ", ".join(sorted(STREAMING_TRIAGE_CLASS_REFS))
        raise ExecutionClassResolutionError(
            f"unknown streaming triage_class {alias!r}; available aliases: {choices}",
        )
    try:
        cls = load_plugin_class(class_ref, what="streaming triage_class")
    except PluginClassLoadError as exc:
        raise ExecutionClassResolutionError(str(exc)) from exc
    if not callable(getattr(cls, "build_classifier", None)):
        raise ExecutionClassResolutionError(
            f"streaming triage_class {alias!r} must implement build_classifier()",
        )
    return ExecutionClassSpec(
        alias=normalized,
        class_ref=class_ref,
        execution="sync",
        provider=str(getattr(cls, "provider", "local") or "local"),
        target=class_ref,
    )


def resolve_streaming_attribution_class(alias: str) -> ExecutionClassSpec:
    """解析 streaming 归因稳定别名（``STREAMING_ATTRIBUTION_CLASS_REFS``）。

    ``null`` 不走本函数（= 现状同步链）；``no_attribution`` → execution=none。
    不复用批量 ``validate_execution_class_protocol``（批量 async 要求
    ``query_job_status`` / run 回调，streaming 单消息路径不适用）。
    """
    normalized = alias.strip()
    class_ref = STREAMING_ATTRIBUTION_CLASS_REFS.get(normalized)
    if class_ref is None:
        choices = ", ".join(sorted(STREAMING_ATTRIBUTION_CLASS_REFS))
        raise ExecutionClassResolutionError(
            f"unknown streaming attribution_class {alias!r}; available aliases: {choices}",
        )
    try:
        cls = load_plugin_class(class_ref, what="streaming attribution_class")
    except PluginClassLoadError as exc:
        raise ExecutionClassResolutionError(str(exc)) from exc
    execution = getattr(cls, "execution", None)
    provider = getattr(cls, "provider", None)
    if (
        execution not in STREAMING_ATTRIBUTION_EXECUTIONS
        or not isinstance(provider, str)
        or not provider
    ):
        raise ExecutionClassResolutionError(
            f"streaming attribution_class {alias!r} must declare "
            f"execution in {sorted(STREAMING_ATTRIBUTION_EXECUTIONS)} "
            "and a non-empty provider",
        )
    if execution == "async" and not callable(getattr(cls, "submit", None)):
        raise ExecutionClassResolutionError(
            f"streaming attribution_class {alias!r} (async) must implement submit()",
        )
    return ExecutionClassSpec(
        alias=normalized,
        class_ref=class_ref,
        execution=execution,
        provider=provider,
        target=class_ref if execution != "none" else None,
    )


def effective_streaming_judge_execution(
    *,
    class_execution: Execution | None,
    override: str | None,
) -> Execution:
    """任务级 ``judge_execution`` 覆盖优先，否则用类声明，缺省 ``sync``。"""
    if override is not None:
        normalized = override.strip().lower()
        if normalized not in STREAMING_JUDGE_EXECUTIONS:
            raise ExecutionClassResolutionError(
                f"invalid judge_execution {override!r}; "
                f"expected one of {sorted(STREAMING_JUDGE_EXECUTIONS)}",
            )
        return normalized  # type: ignore[return-value]
    if class_execution in STREAMING_JUDGE_EXECUTIONS:
        return class_execution  # type: ignore[return-value]
    return "sync"


def effective_streaming_attribution_execution(
    *,
    class_execution: Execution | None,
    override: str | None,
    attribution_class: str | None,
) -> Execution:
    """任务级 ``attribution_execution`` 覆盖优先；``no_attribution`` 强制 none。"""
    if attribution_class == "no_attribution":
        return "none"
    if override is not None:
        normalized = override.strip().lower()
        if normalized not in STREAMING_ATTRIBUTION_EXECUTIONS:
            raise ExecutionClassResolutionError(
                f"invalid attribution_execution {override!r}; "
                f"expected one of {sorted(STREAMING_ATTRIBUTION_EXECUTIONS)}",
            )
        return normalized  # type: ignore[return-value]
    if class_execution in STREAMING_ATTRIBUTION_EXECUTIONS:
        return class_execution  # type: ignore[return-value]
    return "sync"


def validate_streaming_execution_combo(
    *,
    judge_execution: Execution,
    attribution_execution: Execution,
    judge_class: str | None,
    attribution_class: str | None,
) -> None:
    """校验 judge/attribution sync·async·none 组合可达。

    - async judge 必须有可 submit 的 judge_class（不能是 null /
      evaluator_service_judge 等无 submit 标记类）；
    - attribution ``none`` 合法（关闭）；其余组合均允许（含 sync+async 等）。
    """
    if judge_execution == "async":
        if not judge_class:
            raise ExecutionClassResolutionError(
                "judge_execution=async requires an explicit judge_class that implements submit()",
            )
        resolved = resolve_streaming_judge_class(judge_class)
        if resolved.execution != "async" and resolved.target is None:
            # 类声明 sync 且无 evaluate/submit，但 override 为 async → 仍需 submit
            cls = load_plugin_class(resolved.class_ref, what="streaming judge_class")
            if not callable(getattr(cls, "submit", None)):
                raise ExecutionClassResolutionError(
                    f"judge_execution=async but judge_class {judge_class!r} "
                    "does not implement submit()",
                )
    if attribution_execution == "async" and attribution_class == "no_attribution":
        raise ExecutionClassResolutionError(
            "attribution_execution=async conflicts with attribution_class=no_attribution",
        )


#: 无参构造的执行类实例缓存（class_ref 来自白名单，集合有限）
_instance_cache: dict[str, object] = {}


def load_execution_instance(class_ref: str, *, what: str) -> object:
    """按 ``class_ref`` 加载并无参实例化执行类（进程内按 class_ref 缓存）。"""
    cached = _instance_cache.get(class_ref)
    if cached is not None:
        return cached
    cls = load_plugin_class(class_ref, what=what)
    instance = cls()
    _instance_cache[class_ref] = instance
    return instance


def snapshot_class_ref(cfg: object) -> str | None:
    """从 run 快照 ``judge`` / ``attribution`` 片段取 ``class_ref``（缺省 None）。"""
    if not isinstance(cfg, dict):
        return None
    ref = cfg.get("class_ref")
    return ref if isinstance(ref, str) and ref else None


def normalize_judge_callback(class_ref: str | None, payload: "JudgeCallback") -> "JudgeCallback":
    """按 judge 执行类的 ``normalize_callback`` 归一回调（无钩子 = 原样返回）。"""
    if not class_ref:
        return payload
    try:
        cls = load_plugin_class(class_ref, what="judge_class")
    except PluginClassLoadError:
        # 滚动部署窗口：快照冻结时类存在、回调到达时已移除——降级原样消费，
        # 不阻断回调主链路（对齐 _resolve_execution_class 的防御边界）。
        logger.warning(
            "normalize_judge_callback: judge_class class_ref=%s 加载失败，回调按原样消费",
            class_ref,
        )
        return payload
    hook = getattr(cls, "normalize_callback", None)
    return hook(payload) if callable(hook) else payload


def normalize_attribution_callback(
    class_ref: str | None,
    payload: "AttributionCallback",
) -> "AttributionCallback":
    """按 attribution 执行类的 ``normalize_callback`` 归一回调（无钩子 = 原样返回）。"""
    if not class_ref:
        return payload
    try:
        cls = load_plugin_class(class_ref, what="attribution_class")
    except PluginClassLoadError:
        # 滚动部署窗口：快照冻结时类存在、回调到达时已移除——降级原样消费。
        logger.warning(
            "normalize_attribution_callback: class_ref=%s 加载失败，回调按原样消费",
            class_ref,
        )
        return payload
    hook = getattr(cls, "normalize_callback", None)
    return hook(payload) if callable(hook) else payload


__all__ = [
    "ATTRIBUTION_CLASS_REFS",
    "AsyncAttributionImpl",
    "AsyncJudgeImpl",
    "AsyncJudgeSubmitRequest",
    "EvaluatorServiceAttributionMarker",
    "EvaluatorServiceJudge",
    "Execution",
    "ExecutionClassResolutionError",
    "ExecutionClassSpec",
    "JUDGE_CLASS_REFS",
    "LocalAttributionImpl",
    "LocalAttributionRequest",
    "LocalJudge",
    "NoAttribution",
    "STREAMING_ATTRIBUTION_CLASS_REFS",
    "STREAMING_ATTRIBUTION_EXECUTIONS",
    "STREAMING_JUDGE_CLASS_REFS",
    "STREAMING_JUDGE_EXECUTIONS",
    "STREAMING_TRIAGE_CLASS_REFS",
    "StreamingAsyncJudgeImpl",
    "StreamingJudgeImpl",
    "effective_streaming_attribution_execution",
    "effective_streaming_judge_execution",
    "load_execution_instance",
    "normalize_attribution_callback",
    "normalize_judge_callback",
    "resolve_attribution_class",
    "resolve_judge_class",
    "resolve_streaming_attribution_class",
    "resolve_streaming_judge_class",
    "resolve_streaming_triage_class",
    "snapshot_class_ref",
    "validate_execution_class_protocol",
    "validate_streaming_execution_combo",
    "validate_streaming_judge_protocol",
]
