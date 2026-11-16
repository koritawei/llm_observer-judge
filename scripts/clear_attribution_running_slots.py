#!/usr/bin/env python3
"""运维：清除卡住的 attribution-running pending_callback 及其 inflight 槽位。

场景：Aone 归因已提交（``pending_callback`` 占 ``cluster:attr``），回调长期不到，
``attr_callback_timeout_ms`` 未生效或 Healer 未到点，后续 run 全部
``attribution.cluster_slot_busy``。

本脚本**只**操作 ``attribution-running`` 与 ``inflight:attr*`` / ``inflight:cluster:attr``，
**不**动 ``attribution-wait``（排队中的 case）。

默认 dry-run；必须显式 ``--job-id``；加 ``--execute`` 才真正删除。

用法（优先包装脚本，线上会拼 agentgraph PYTHONPATH）::

  bash scripts/clear_attribution_running_slots.sh \\
    --job-id 9f820d52696cf3b0 --job-id 15f795f942e9327a \\
    --job-id 00a8eef8714b45a4 --job-id bdfad4223ea67c9a

  bash scripts/clear_attribution_running_slots.sh \\
    --job-id 9f820d52696cf3b0 --job-id 15f795f942e9327a \\
    --job-id 00a8eef8714b45a4 --job-id bdfad4223ea67c9a \\
    --execute
"""

import argparse
import asyncio
import json
import sys
from typing import Any

import _bootstrap  # sys.path/.env/禁 MQ 副作用，须先于 app.* 执行
import redis.asyncio as aioredis
from benchmark.settings import Settings
from benchmark.libs.jobqueue.key_prefix import queue_key_prefix
from benchmark.paths import app_config_path
from benchmark.queue.inflight_slot import (
    release_cluster_slot,
    release_slot,
)
from benchmark.queue.queue_factory import (
    ATTRIBUTION_RUNNING,
    EvalQueueRegistry,
    load_queue_prefix_from_yaml,
)
from benchmark.queue.queue_prefix_policy import (
    QueuePrefixResolveInput,
    resolve_queue_prefix,
)

del _bootstrap  # 仅为副作用导入：已执行，释放模块名


def _job_hash_key(bullmq_prefix: str, job_id: str) -> str:
    return f"{bullmq_prefix}:{ATTRIBUTION_RUNNING}:{job_id}"


def _wait_list_key(bullmq_prefix: str) -> str:
    return f"{bullmq_prefix}:{ATTRIBUTION_RUNNING}:wait"


def _parse_job_data(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


async def _load_job_snapshot(
    r: aioredis.Redis,
    *,
    bullmq_prefix: str,
    job_id: str,
) -> dict[str, Any]:
    job_key = _job_hash_key(bullmq_prefix, job_id)
    name, timestamp, data_raw = await r.hmget(job_key, "name", "timestamp", "data")
    data = _parse_job_data(data_raw)
    return {
        "job_id": job_id,
        "job_key": job_key,
        "exists": name is not None or timestamp is not None or data_raw is not None,
        "name": name,
        "timestamp": timestamp,
        "run_id": data.get("run_id"),
        "item_id": data.get("item_id"),
        "slot_key": data.get("slot_key"),
        "attribution_job_id": data.get("attribution_job_id") or job_id,
        "submitter_node_id": data.get("submitter_node_id"),
    }


def _print_snapshot(snap: dict[str, Any]) -> None:
    print(
        f"  job_id={snap['job_id']} exists={snap['exists']} "
        f"name={snap['name']!r} timestamp={snap['timestamp']!r} "
        f"run_id={snap['run_id']!r} item_id={snap['item_id']!r} "
        f"slot_key={snap['slot_key']!r} "
        f"submitter={snap['submitter_node_id']!r}",
    )


async def _clear_jobs(
    *,
    job_ids: list[str],
    execute: bool,
    allow_non_pending: bool,
) -> int:
    settings = Settings()
    env_prefix_before_yaml = (settings.eval_queue_prefix or "").strip()
    yaml_prefix = load_queue_prefix_from_yaml(app_config_path(settings.env))
    settings.eval_queue_prefix = resolve_queue_prefix(
        QueuePrefixResolveInput(
            deploy_profile=settings.env,
            yaml_prefix=yaml_prefix,
            env_prefix=env_prefix_before_yaml,
            env_sign=settings.env_sign,
            sigma_app_stage=settings.sigma_app_stage,
            deploy_kind=settings.eval_queue_deploy_kind,
        )
    )

    qm = EvalQueueRegistry(settings)
    bullmq_prefix = queue_key_prefix(qm.prefix, ATTRIBUTION_RUNNING)
    wait_key = _wait_list_key(bullmq_prefix)
    cluster_key = f"{qm.prefix}:inflight:cluster:attr"

    r = aioredis.from_url(qm.redis_url, decode_responses=True)
    try:
        cluster_before = await r.get(cluster_key)
        wait_len_before = await r.llen(wait_key)
        print(f"prefix={qm.prefix} env={settings.env}")
        print(f"wait_key={wait_key}")
        print(f"cluster_key={cluster_key} value={cluster_before!r}")
        print(f"wait LLEN before={wait_len_before}")
        print(f"mode={'EXECUTE' if execute else 'DRY-RUN'}")
        print("Jobs:")

        snapshots: list[dict[str, Any]] = []
        for job_id in job_ids:
            snap = await _load_job_snapshot(
                r,
                bullmq_prefix=bullmq_prefix,
                job_id=job_id,
            )
            snapshots.append(snap)
            _print_snapshot(snap)

        missing = [s["job_id"] for s in snapshots if not s["exists"]]
        if missing:
            print(f"ERROR: job hash missing: {missing}", file=sys.stderr)
            return 2

        bad_name = [s["job_id"] for s in snapshots if (s["name"] or "") != "pending_callback"]
        if bad_name and not allow_non_pending:
            print(
                "ERROR: refusing non-pending_callback jobs "
                f"(use --allow-non-pending to override): {bad_name}",
                file=sys.stderr,
            )
            return 2

        if not execute:
            print(
                "[dry-run] would LREM wait + DEL job hash + "
                "release_slot(slot_key) + release_cluster_slot(attr) "
                f"for {len(snapshots)} job(s)",
            )
            print("[dry-run] attribution-wait is NOT touched")
            return 0

        for snap in snapshots:
            job_id = str(snap["job_id"])
            removed = await r.lrem(wait_key, 0, job_id)
            deleted = await r.delete(snap["job_key"])
            slot_key = snap.get("slot_key")
            if slot_key:
                await release_slot(str(slot_key))
            await release_cluster_slot("attr")
            print(
                f"  cleared job_id={job_id} lrem={removed} "
                f"del_hash={deleted} slot_key={slot_key!r}",
            )

        cluster_after = await r.get(cluster_key)
        wait_len_after = await r.llen(wait_key)
        print(f"cluster_key after={cluster_after!r}")
        print(f"wait LLEN after={wait_len_after}")
        print("done (attribution-wait untouched)")
        return 0
    finally:
        await r.aclose()
        await qm.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clear stuck attribution-running pending_callback jobs "
            "and release attr inflight slots (dry-run by default)"
        ),
    )
    parser.add_argument(
        "--job-id",
        action="append",
        dest="job_ids",
        required=True,
        help="BullMQ job id in attribution-running (repeatable, required)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually delete; default is dry-run",
    )
    parser.add_argument(
        "--allow-non-pending",
        action="store_true",
        help="allow clearing jobs whose name is not pending_callback",
    )
    args = parser.parse_args()
    # 去重但保序
    seen: set[str] = set()
    job_ids: list[str] = []
    for raw_jid in args.job_ids:
        jid = str(raw_jid).strip()
        if not jid or jid in seen:
            continue
        seen.add(jid)
        job_ids.append(jid)
    if not job_ids:
        raise SystemExit("at least one non-empty --job-id is required")

    raise SystemExit(
        asyncio.run(
            _clear_jobs(
                job_ids=job_ids,
                execute=bool(args.execute),
                allow_non_pending=bool(args.allow_non_pending),
            ),
        ),
    )


if __name__ == "__main__":
    main()
