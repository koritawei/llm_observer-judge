#!/usr/bin/env python3
"""Point-query one sidecar trace: Lindorm/OSS → thin filter → four trace tools (does not subscribe to MetaQ and does not run the judge).

bash scripts/sidecar.sh --trace-id <id>
bash scripts/sidecar.sh --trace-id <id> --project-id <pid>
bash scripts/sidecar.sh --trace-id <id> --project-id <pid> --print-notify
"""

import argparse
import asyncio
import json
import sys
from typing import Any

import _bootstrap  # sys.path/.env/disable MQ must run before app.*
import yaml
from benchmark.settings import Settings
from benchmark.domain.eval_config import EvalConfigStore
from benchmark.paths import app_config_path
from benchmark.trace.tools import TraceToolExt
from benchmark.ingest.sidecar.config import SidecarTraceConfig, load_sidecar_config
from benchmark.ingest.sidecar.converter import SidecarIdentityError, to_sidecar_evaluator_request
from benchmark.ingest.sidecar.filter import select_sampled_configs
from benchmark.ingest.sidecar.notify import build_notify, notify_console_fields, parse_notify
from benchmark.ingest.sidecar.observations import (
    _SIDECAR_OBSERVATIONS_KEY,
    _span_id,
    dict_observations,
    extract_project_id,
)
from benchmark.ingest.sidecar.oss_blob import OssBlob
from benchmark.ingest.sidecar.trace_reader import SidecarTraceReader, SidecarTraceRecord
from benchmark.ingest.sidecar.unpack import unpack_observations

del _bootstrap


class _InspectError(Exception):
    """User-facing inspect failure."""


class _UnexpectedTrace:
    async def fetch_skeleton(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected Langfuse fetch_skeleton")

    async def fetch_observation(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected Langfuse fetch_observation")

    async def find_observations(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected Langfuse find_observations")

    async def fetch_full_trace(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("unexpected Langfuse fetch_full_trace")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect sidecar trace tools without evaluating",
    )
    parser.add_argument("--trace-id", required=True)
    parser.add_argument(
        "--project-id",
        default="",
        help="票据 project_id；缺省则从树上取并告警",
    )
    parser.add_argument(
        "--print-notify",
        action="store_true",
        help="额外按 worker 规则打印 MetaQ 控制台信封（不发送）",
    )
    parser.add_argument(
        "--finish-reason",
        default="",
        choices=["", "finish", "timeout"],
        help="仅 --print-notify：覆盖 Lindorm 行上的 finish_reason",
    )
    parser.add_argument(
        "--max-observation-chars",
        type=int,
        default=0,
        help="覆盖 yaml tools.extensions.trace.max_observation_chars；0=用 yaml",
    )
    return parser.parse_args()


def _load_sidecar_runtime() -> tuple[Settings, SidecarTraceConfig]:
    settings = Settings()
    yaml_cfg = load_sidecar_config(app_config_path(settings.env))
    table = settings.sidecar_trace_lindorm_table or yaml_cfg.lindorm_table
    cfg = yaml_cfg.model_copy(update={"lindorm_table": table}) if table else yaml_cfg
    return settings, cfg


def _load_trace_tool_ext() -> TraceToolExt:
    settings = Settings()
    with app_config_path(settings.env).open() as handle:
        data = yaml.safe_load(handle) or {}
    raw = ((data.get("tools") or {}).get("extensions") or {}).get("trace") or {}
    return TraceToolExt.model_validate(raw)


async def _fetch_unpacked(
    trace_id: str,
) -> tuple[SidecarTraceRecord, list[Any], SidecarTraceConfig]:
    settings, cfg = _load_sidecar_runtime()
    dsn = settings.sidecar_trace_lindorm_dsn.strip()
    if not dsn:
        raise RuntimeError("SIDECAR_TRACE_LINDORM_DSN is empty")
    reader = SidecarTraceReader(
        dsn,
        cfg.lindorm_table,
        oss=OssBlob(
            endpoint=settings.sidecar_trace_oss_endpoint,
            access_key_id=settings.sidecar_trace_oss_access_key_id,
            access_key_secret=settings.sidecar_trace_oss_access_key_secret,
        ),
    )
    await reader.open()
    try:
        record = await reader.fetch_by_trace_id(trace_id)
    finally:
        await reader.close()
    if record is None:
        raise LookupError(f"Lindorm miss table={cfg.lindorm_table} trace_id={trace_id}")
    return record, unpack_observations(record.observations), cfg


def _ticket_project_id(args: argparse.Namespace, observations: list[object]) -> str:
    ticket = args.project_id.strip()
    if ticket:
        return ticket
    from_tree = extract_project_id(None, observations) or ""
    if from_tree:
        print(
            "WARN: --project-id omitted; using observation.project_id "
            f"{from_tree!r} (worker uses ticket id)",
            file=sys.stderr,
        )
    return from_tree


def _find_args(spans: list[object]) -> dict[str, str]:
    fallback_type = ""
    for span in spans:
        if not isinstance(span, dict):
            continue
        name = str(span.get("name") or "").strip()
        if name:
            return {"name": name}
        typ = str(span.get("type") or "").strip()
        if typ and not fallback_type:
            fallback_type = typ
    return {"type": fallback_type or "SPAN"}


def _print_console(console: dict[str, object]) -> None:
    print(f"topic: {console['topic']}")
    print(f"tag: {console['tag'] or ''}")
    print(f"key: {console['key']}")
    print("user_properties:")
    props = console["properties"]
    if isinstance(props, dict) and props:
        for key, value in props.items():
            print(f"  {key}={value}")
    else:
        print("  (none)")
    print("body:")
    print(json.dumps(console["body"], ensure_ascii=False, indent=2))


async def _run_tools(
    observations: list[object],
    *,
    trace_id: str,
    project_id: str,
    max_observation_chars: int,
) -> None:
    from benchmark.trace.tools import (  # noqa: PLC0415
        TraceToolProvider,
        bind_sidecar_observations,
        bind_trace_request,
    )

    provider = TraceToolProvider(  # type: ignore[arg-type]
        _UnexpectedTrace(),
        max_observation_chars=max_observation_chars,
    )
    with bind_sidecar_observations(list(observations)):
        bound = provider.create_bound(trace_id=trace_id, project_id=project_id)
        with bind_trace_request(bound):
            skeleton = await provider.call_tool("fetch_trace_skeleton", {})
            if not skeleton.success:
                raise RuntimeError(f"fetch_trace_skeleton failed: {skeleton.error}")
            body = json.loads(skeleton.to_text())
            spans = body.get("spans") if isinstance(body, dict) else None
            if not isinstance(spans, list) or not spans:
                raise RuntimeError("skeleton spans empty")
            first_id = spans[0].get("id") if isinstance(spans[0], dict) else None
            if not isinstance(first_id, str) or not first_id.strip():
                raise RuntimeError("skeleton span missing id")
            detail = await provider.call_tool(
                "fetch_observation_detail",
                {"observation_id": first_id},
            )
            if not detail.success:
                raise RuntimeError(f"fetch_observation_detail failed: {detail.error}")
            missing = await provider.call_tool(
                "fetch_observation_detail",
                {"observation_id": "sidecar-inspect-missing-id"},
            )
            if missing.success:
                raise RuntimeError("missing observation must fail closed")
            found = await provider.call_tool("find_observations", _find_args(spans))
            if not found.success:
                raise RuntimeError(f"find_observations failed: {found.error}")
            full = await provider.call_tool("fetch_full_trace", {})
            if not full.success:
                raise RuntimeError(f"fetch_full_trace failed: {full.error}")
            payload = json.loads(full.to_text())
            full_obs = payload.get("observations") if isinstance(payload, dict) else None
            if not isinstance(full_obs, list):
                raise RuntimeError("full_trace observations missing")
    n_obs = len(dict_observations(observations))
    body_truncated = bool(isinstance(body, dict) and body.get("body_truncated"))
    print(
        "ok tools "
        f"max_observation_chars={max_observation_chars} "
        f"skeleton_spans={len(spans)}/{n_obs} "
        f"body_truncated={body_truncated} "
        f"skeleton_chars={len(skeleton.to_text())} "
        f"detail_id={first_id!r} find_ok=True miss_closed=True "
        f"full_obs={len(full_obs)}"
    )
    if body_truncated or len(spans) != n_obs:
        print(
            "WARN: skeleton truncated vs unpacked tree; "
            "this is the live tool cap, not a Lindorm miss"
        )


def _print_notify_envelope(
    *,
    args: argparse.Namespace,
    record: SidecarTraceRecord,
    observations: list[object],
    cfg: SidecarTraceConfig,
    project_id: str,
) -> None:
    reason = (args.finish_reason or record.finish_reason or "").strip()
    if reason not in {"finish", "timeout"}:
        reason = "finish"
    payload = build_notify(
        args.trace_id,
        reason,
        observations,
        project_id=project_id,
    )
    if payload["project_id"] == "unknown":
        raise _InspectError(
            "project_id missing; pass --project-id (ticket id, not observation.project_id)"
        )
    if not payload.get("agent_name"):
        print(
            "WARN: agent_name empty; in-process filter will try RemoteConfig registry, "
            "then skip if still unknown",
            file=sys.stderr,
        )
    console = notify_console_fields(payload, topic_prefix=cfg.topic_prefix)
    print(f"# table={cfg.lindorm_table}")
    _print_console(console)


async def _inspect(args: argparse.Namespace) -> None:
    record, observations, cfg = await _fetch_unpacked(args.trace_id)
    items = dict_observations(observations)
    span_ids = [_span_id(item) for item in items if _span_id(item)]
    print(
        "ok lindorm+oss "
        f"table={cfg.lindorm_table} finish_reason={record.finish_reason!r} "
        f"n_obs={len(items)} n_span_id={len(span_ids)}"
    )
    if not items or not span_ids:
        raise _InspectError("unpacked observations empty or missing span_id")

    project_id = _ticket_project_id(args, observations)
    reason = record.finish_reason if record.finish_reason in {"finish", "timeout"} else "finish"
    payload = build_notify(
        args.trace_id,
        reason,
        observations,
        project_id=project_id,
    )
    if payload["project_id"] == "unknown":
        raise _InspectError("project_id missing; pass --project-id (ticket id, not span field)")
    notify = parse_notify(payload)
    print(
        f"ok identity project_id={payload['project_id']!r} agent_name={payload.get('agent_name')!r}"
    )

    store = EvalConfigStore()
    sidecar_cfgs = [
        entry for entry in store.get_configs(payload["project_id"]) if entry.is_sidecar_ingest
    ]
    hits = select_sampled_configs(notify, store)
    print(
        "ok remote_config "
        f"sidecar_entries={len(sidecar_cfgs)} "
        f"filter_hit={[entry.evaluator_name for entry in hits] or []}"
    )
    if hits:
        try:
            request = to_sidecar_evaluator_request(notify, observations, hits[0])
        except SidecarIdentityError as exc:
            raise _InspectError(str(exc)) from exc
        meta = request.metadata or {}
        print(
            "ok request "
            f"skip_trace_resolve={request.skip_trace_resolve} "
            f"pipeline_source={meta.get('pipeline_source')!r} "
            f"n_obs={len(items)}"
        )
        if (
            meta.get("pipeline_source") != "sidecar"
            or meta.get(_SIDECAR_OBSERVATIONS_KEY) is not None
            or not request.skip_trace_resolve
        ):
            raise _InspectError("EvaluatorRequest sidecar stamp mismatch")
    elif sidecar_cfgs:
        print("WARN: sidecar entries exist but thin filter missed")
    else:
        print("WARN: RemoteConfig has no ingest=sidecar entry")

    cap = args.max_observation_chars or _load_trace_tool_ext().max_observation_chars
    try:
        await _run_tools(
            observations,
            trace_id=args.trace_id,
            project_id=payload["project_id"],
            max_observation_chars=cap,
        )
    except RuntimeError as exc:
        raise _InspectError(str(exc)) from exc

    if args.print_notify:
        _print_notify_envelope(
            args=args,
            record=record,
            observations=observations,
            cfg=cfg,
            project_id=payload["project_id"],
        )


async def _amain(args: argparse.Namespace) -> int:
    try:
        await _inspect(args)
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except LookupError as exc:
        print(f"FAIL: {exc}")
        return 1
    except _InspectError as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain(_parse_args())))


if __name__ == "__main__":
    main()
