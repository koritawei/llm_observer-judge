#!/usr/bin/env python3
"""为存量 benchmark-agent 注册回填 agentScope（默认 amap-agent）。

覆盖：普通注册 + 定制注册。缺 ``payload.agentScope`` 时写入默认评测对象，
并同步 ``scope_key=agent:{scope}``（与 ``save_agent`` 同口径）。

定制脏数据（有 customization 但缺 agentScope）在严格 ``AgentDocument``
校验下会失败：本脚本先注入 scope 再校验；仍失败则退回 raw payload 补丁，
避免回填被跳过、控制台列表继续 500。

用法（在 services/agent-eval 下，需可用的 AGENT_EVAL_DATABASE_URL）::

    bash scripts/backfill_agent_scope.sh
    bash scripts/backfill_agent_scope.sh --apply
    bash scripts/backfill_agent_scope.sh --apply --default-scope amap-agent
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any, NamedTuple

import _bootstrap  # noqa: F401 — sys.path / .env / MQ 关闭

DEFAULT_SCOPE = "amap-agent"
_MAX_UPDATES = 10_000


class BackfillCandidate(NamedTuple):
    """待回填的一条注册。"""

    document_id: str
    name: str
    kind: str  # plain | customized
    payload: dict[str, object]
    via: str  # validated | raw_patch


def missing_agent_scope(document: object) -> bool:
    """payload 未绑定评测对象（历史底版）。"""
    scope = getattr(document, "agent_scope", None)
    return not (scope and str(scope).strip())


def registration_kind(document: object) -> str:
    """区分普通 / 定制，便于 dry-run 核对。"""
    return "customized" if getattr(document, "customization", None) is not None else "plain"


def payload_missing_agent_scope(payload: dict[str, Any]) -> bool:
    """raw JSON 是否缺评测对象。"""
    scope = payload.get("agentScope")
    return not (isinstance(scope, str) and scope.strip())


def payload_registration_kind(payload: dict[str, Any]) -> str:
    """raw JSON 上的普通 / 定制标签。"""
    return "customized" if payload.get("customization") is not None else "plain"


def prepare_scope_backfill(
    payload: dict[str, Any],
    *,
    default_scope: str,
    validate: Any | None = None,
) -> BackfillCandidate | None:
    """缺 agentScope 时构造回填候选；已有 scope 返回 None。

    ``validate`` 默认 ``AgentDocument.model_validate``；注入 scope 后校验
    成功用 dump，失败则 raw_patch（只补 agentScope，保留其余字段）。
    """
    if not payload_missing_agent_scope(payload):
        return None
    patched: dict[str, object] = {**payload, "agentScope": default_scope}
    document_id = str(patched.get("id") or "").strip()
    name = str(patched.get("name") or document_id or "unknown").strip() or "unknown"
    kind = payload_registration_kind(patched)
    if not document_id:
        return None

    validator = validate
    if validator is None:
        from benchmark.domain.benchmark_console import AgentDocument

        validator = AgentDocument.model_validate

    try:
        document = validator(patched)
    except Exception:  # noqa: BLE001 — 脏数据回填允许降级
        return BackfillCandidate(
            document_id=document_id,
            name=name,
            kind=kind,
            payload=patched,
            via="raw_patch",
        )

    dumped = document.model_dump(by_alias=True, exclude_none=True)
    return BackfillCandidate(
        document_id=str(getattr(document, "id", document_id)),
        name=str(getattr(document, "name", name)),
        kind=registration_kind(document),
        payload=dumped,
        via="validated",
    )


async def _run(*, apply: bool, default_scope: str) -> int:
    from benchmark.settings.setting import settings
    from benchmark.records.database import create_database_config
    from benchmark.records.models.runtime_config_record import (
        KIND_BENCHMARK_AGENT,
        RuntimeConfigRecord,
        agent_scope_key,
        benchmark_agent_record_key,
    )
    from benchmark.domain.runtime_config_store import upsert_runtime_config
    from tortoise import Tortoise

    scope = default_scope.strip()
    if not scope:
        raise RuntimeError("default_scope must be non-empty")
    if not settings.database_url.startswith(("postgres", "asyncpg://", "psycopg://", "sqlite")):
        raise RuntimeError("refusing unrecognized database_url")
    config = create_database_config(db_url=settings.database_url, environment="production")
    await Tortoise.init(config=config.build_config())
    try:
        rows = await RuntimeConfigRecord.filter(
            kind=KIND_BENCHMARK_AGENT,
            deleted_at__isnull=True,
        )
        pending: list[BackfillCandidate] = []
        for row in rows:
            raw = dict(row.payload or {})
            candidate = prepare_scope_backfill(raw, default_scope=scope)
            if candidate is None:
                continue
            pending.append(candidate)

        plain_n = sum(1 for item in pending if item.kind == "plain")
        custom_n = sum(1 for item in pending if item.kind == "customized")
        raw_n = sum(1 for item in pending if item.via == "raw_patch")
        print(
            f"found {len(pending)} registration(s) missing agentScope "
            f"(plain={plain_n}, customized={custom_n}, raw_patch={raw_n}) -> {scope}"
        )
        if len(pending) >= _MAX_UPDATES:
            print(f"refusing: pending={len(pending)} >= {_MAX_UPDATES}; split batches")
            return 1

        for item in pending:
            print(
                f"  - {item.document_id} name={item.name} "
                f"kind={item.kind} via={item.via} -> {scope}"
            )
            if not apply:
                continue
            await upsert_runtime_config(
                kind=KIND_BENCHMARK_AGENT,
                record_key=benchmark_agent_record_key(item.document_id),
                payload=item.payload,
                scope_key=agent_scope_key(scope),
            )
        print(f"{'applied' if apply else 'dry-run'} {len(pending)} update(s)")
        return 0
    finally:
        await Tortoise.close_connections()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        "--execute",
        action="store_true",
        dest="apply",
        help="写库；默认 dry-run（亦接受 --execute）",
    )
    parser.add_argument("--default-scope", default=DEFAULT_SCOPE)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(apply=args.apply, default_scope=args.default_scope)))


if __name__ == "__main__":
    main()
