"""策略 / 统计 / 提取 / 融合类插件的 remote 适配器。

每个适配器同时实现同步协议方法（与本地实现同签名）与 ``a<method>`` /
``acall`` 异步变体；async 调用点经 :func:`benchmark.plugins.host.call_plugin`
选择异步变体，避免 HTTP 阻塞事件循环。远程失败时：``binding.fallback_local=true`` →
使用注入的本地兜底实现；否则抛 :class:`RemotePluginError`。

各 action 响应约定见 :mod:`benchmark.plugins.remote_client`。
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from benchmark.libs.agenthost.observability.logger import get_logger

from benchmark.domain.plugin_binding import PluginBinding, PluginKind
from benchmark.plugins.remote_client import RemotePluginClient, RemotePluginError

if TYPE_CHECKING:
    # 仅类型引用：注册表模块 import 本模块，运行时反向 import 会成环
    from benchmark.attribution.badcase.attribution_trigger_policy import (
        AttributionTriggerPolicy,
    )
    from benchmark.attribution.badcase.dispatch_policy import NotificationDispatchPolicy

logger = get_logger(__name__)

TraceIdentity = tuple[str | None, str | None]


class _RemoteAdapter:
    """持有 binding + client，统一 remote 调用与失败日志。"""

    kind: PluginKind

    def __init__(
        self,
        binding: PluginBinding,
        *,
        client: RemotePluginClient | None = None,
    ) -> None:
        self._binding = binding
        self._client = client or RemotePluginClient()

    @property
    def fallback_local(self) -> bool:
        return self._binding.fallback_local

    def _invoke_sync(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._client.invoke_sync(
            binding=self._binding,
            kind=self.kind,
            action=action,
            payload=payload,
        )

    async def _invoke(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._client.invoke(
            binding=self._binding,
            kind=self.kind,
            action=action,
            payload=payload,
        )

    def _log_failure(self, exc: RemotePluginError) -> None:
        logger.warning(
            "remote_plugin.failed kind=%s name=%s err=%s fallback_local=%s",
            self.kind,
            self._binding.name,
            exc,
            self._binding.fallback_local,
        )


def _bool_result(data: dict[str, Any], *, default: bool) -> bool:
    if "result" in data:
        return bool(data["result"])
    if "allow" in data:
        return bool(data["allow"])
    return default


# ── 归因触发策略 ─────────────────────────────────────────────────────────────


class RemoteTriggerPolicy(_RemoteAdapter):
    """远程归因触发策略；``fallback`` 为本地兜底策略（缺省框架默认）。"""

    kind = "attribution_trigger_policy"

    def __init__(
        self,
        binding: PluginBinding,
        *,
        fallback: "AttributionTriggerPolicy",
        client: RemotePluginClient | None = None,
    ) -> None:
        super().__init__(binding, client=client)
        self._fallback = fallback

    @staticmethod
    def _payload(
        fused_score: float | None,
        criterion_scores: Mapping[str, float | None] | None,
    ) -> dict[str, Any]:
        return {
            "fused_score": fused_score,
            "criterion_scores": dict(criterion_scores or {}),
        }

    def _on_error(
        self,
        exc: RemotePluginError,
        fused_score: float | None,
        criterion_scores: Mapping[str, float | None] | None,
    ) -> bool:
        self._log_failure(exc)
        if self.fallback_local:
            return self._fallback.should_trigger(
                fused_score=fused_score,
                criterion_scores=criterion_scores,
            )
        raise exc

    def should_trigger(
        self,
        *,
        fused_score: float | None,
        criterion_scores: Mapping[str, float | None] | None = None,
    ) -> bool:
        payload = self._payload(fused_score, criterion_scores)
        try:
            data = self._invoke_sync("should_trigger", payload)
        except RemotePluginError as exc:
            return self._on_error(exc, fused_score, criterion_scores)
        return _bool_result(data, default=True)

    async def ashould_trigger(
        self,
        *,
        fused_score: float | None,
        criterion_scores: Mapping[str, float | None] | None = None,
    ) -> bool:
        payload = self._payload(fused_score, criterion_scores)
        try:
            data = await self._invoke("should_trigger", payload)
        except RemotePluginError as exc:
            return self._on_error(exc, fused_score, criterion_scores)
        return _bool_result(data, default=True)


# ── aone 派发策略 ─────────────────────────────────────────────────────────────


class RemoteDispatchPolicy(_RemoteAdapter):
    """远程 aone 派发策略；``fallback`` 为本地兜底策略。"""

    kind = "dispatch_policy"

    def __init__(
        self,
        binding: PluginBinding,
        *,
        fallback: "NotificationDispatchPolicy",
        client: RemotePluginClient | None = None,
    ) -> None:
        super().__init__(binding, client=client)
        self._fallback = fallback

    def _on_error(self, exc: RemotePluginError, grade: str | None, score: float | None) -> bool:
        self._log_failure(exc)
        if self.fallback_local:
            return self._fallback.should_dispatch(grade=grade, score=score)
        raise exc

    def should_dispatch(self, *, grade: str | None, score: float | None = None) -> bool:
        try:
            data = self._invoke_sync("should_dispatch", {"grade": grade, "score": score})
        except RemotePluginError as exc:
            return self._on_error(exc, grade, score)
        return _bool_result(data, default=True)

    async def ashould_dispatch(self, *, grade: str | None, score: float | None = None) -> bool:
        try:
            data = await self._invoke("should_dispatch", {"grade": grade, "score": score})
        except RemotePluginError as exc:
            return self._on_error(exc, grade, score)
        return _bool_result(data, default=True)


# ── 归因统计 ─────────────────────────────────────────────────────────────────


class RemoteAttributionStatsProvider(_RemoteAdapter):
    """远程归因统计（协议本身即 async）。失败且 fallback_local → 空统计。"""

    kind = "attribution_stats_provider"

    async def collect(self, run_id: str) -> dict[str, dict[str, Any]]:
        try:
            data = await self._invoke("collect", {"run_id": run_id})
        except RemotePluginError as exc:
            self._log_failure(exc)
            if self.fallback_local:
                return {}
            raise
        result = data.get("result", data.get("stats", {}))
        return result if isinstance(result, dict) else {}


# ── trace 身份提取（callback 形态 / item 形态）────────────────────────────────


def _to_plain(value: object) -> object:
    """pydantic / dataclass / Mapping → JSON 友好 dict；其余原样。"""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _parse_identity(data: dict[str, Any]) -> TraceIdentity:
    project_id = data.get("project_id")
    trace_id = data.get("trace_id")
    return (
        str(project_id) if project_id else None,
        str(trace_id) if trace_id else None,
    )


class RemoteTraceIdentityExtractor(_RemoteAdapter):
    """远程 trace 身份提取 → ``(project_id, trace_id)``。

    callback 形态（``kind=trace_identity_extractor``，payload 键 ``callback``）
    与 item 形态（``kind=item_trace_identity_extractor``，payload 键 ``item``）
    共用本类；失败一律回退 ``(None, None)``（trace_url 构建不可阻塞主流程）。
    """

    kind = "trace_identity_extractor"
    payload_key = "callback"

    def __init__(
        self,
        binding: PluginBinding,
        *,
        fallback: Callable[[object], TraceIdentity] | None = None,
        client: RemotePluginClient | None = None,
    ) -> None:
        super().__init__(binding, client=client)
        self._fallback = fallback

    def _on_error(self, exc: RemotePluginError, source: object) -> TraceIdentity:
        self._log_failure(exc)
        if self.fallback_local and self._fallback is not None:
            return self._fallback(source)
        return None, None

    def __call__(self, source: object) -> TraceIdentity:
        try:
            data = self._invoke_sync("extract", {self.payload_key: _to_plain(source)})
        except RemotePluginError as exc:
            return self._on_error(exc, source)
        return _parse_identity(data)

    async def acall(self, source: object) -> TraceIdentity:
        try:
            data = await self._invoke("extract", {self.payload_key: _to_plain(source)})
        except RemotePluginError as exc:
            return self._on_error(exc, source)
        return _parse_identity(data)


class RemoteItemTraceIdentityExtractor(RemoteTraceIdentityExtractor):
    """item 形态（sync judge 路径）远程身份提取。"""

    kind = "item_trace_identity_extractor"
    payload_key = "item"


# ── item 级分数融合 ───────────────────────────────────────────────────────────


class RemoteFusionFunction(_RemoteAdapter):
    """远程 item 级融合函数；签名与 ``fusion_fn_ref`` 本地函数一致。

    响应 ``{"score": <float>}``。失败时抛 :class:`RemotePluginError`，
    由 ``ScoreFuser`` 按 ``fallback_local`` 决定是否回退 method 融合。
    """

    kind = "fusion_function"

    @staticmethod
    def _payload(
        scores: Mapping[str, Any],
        rubric: list[Any],
        aggregation: object,
        score_of: Callable[[str], float],
    ) -> dict[str, Any]:
        return {
            "scores": {
                cid: {
                    "score": score_of(cid),
                    "reason": getattr(result, "reason", None),
                    "failed": not hasattr(result, "score"),
                }
                for cid, result in scores.items()
            },
            "rubric": [_to_plain(c) for c in rubric],
            "aggregation": _to_plain(aggregation),
        }

    @staticmethod
    def _parse_score(data: dict[str, Any]) -> float:
        raw = data.get("score", data.get("result"))
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise RemotePluginError(f"fusion response missing numeric score: {raw!r}") from exc

    def __call__(
        self,
        scores: Mapping[str, Any],
        rubric: list[Any],
        aggregation: object,
        score_of: Callable[[str], float],
    ) -> float:
        try:
            data = self._invoke_sync("fuse", self._payload(scores, rubric, aggregation, score_of))
        except RemotePluginError as exc:
            self._log_failure(exc)
            raise
        return self._parse_score(data)

    async def acall(
        self,
        scores: Mapping[str, Any],
        rubric: list[Any],
        aggregation: object,
        score_of: Callable[[str], float],
    ) -> float:
        try:
            data = await self._invoke("fuse", self._payload(scores, rubric, aggregation, score_of))
        except RemotePluginError as exc:
            self._log_failure(exc)
            raise
        return self._parse_score(data)


__all__ = [
    "RemoteAttributionStatsProvider",
    "RemoteDispatchPolicy",
    "RemoteFusionFunction",
    "RemoteItemTraceIdentityExtractor",
    "RemoteTraceIdentityExtractor",
    "RemoteTriggerPolicy",
    "TraceIdentity",
]
