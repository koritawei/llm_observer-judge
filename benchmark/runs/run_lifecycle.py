"""Run 创建 / 状态机 / 心跳 / 完成失败 / 恢复

Mixin for :class:`~benchmark.runs.run_service.BenchmarkService`.
Symbols resolve through :mod:`run_helpers` so tests can patch
``benchmark.runs.run_helpers.*``.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from tortoise.functions import Count

from benchmark.records.enums import BenchmarkStatus, TraceStatus
from benchmark.runs import run_helpers as rh
from benchmark.runs.run_stop import (
    STOP_REASON,
    is_stop_requested,
    partial_report_meta,
)
from benchmark.lifecycle.run_lifecycle import (
    RUN_STALE_TIMEOUT_S,
    STALE_RECOVERY_REASON,
    FailRunSpec,
    RunLifecycleBinding,
    fail_inflight_runs,
    fail_run,
    heartbeat_run,
    recover_stale_runs,
)

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ListRunsQuery:
    """``list_runs`` 过滤与分页参数（FUNC-PARAMS-001 收敛）。"""

    benchmark_name: str | None = None
    task_id: str | None = None
    limit: int = 20
    offset: int = 0


def _normalize_count_key(name: str) -> str:
    """amap-agent / amap_agent 归一为下划线小写（与前端 normalizeRunCountKey 对齐）。"""
    return name.strip().lower().replace("-", "_")


def _count_name_aliases(name: str) -> list[str]:
    """计次键别名：evaluator 茎 ↔ 执行侧 benchmark 名（供卡片按任一形态命中）。

    主统计维度是 ``evaluator_name``；页面注册文档常用执行侧名
    （``amap_agent_benchmark``），因此响应里仍附带别名行，数值相同、
    不叠加。连字符 / 下划线两种形态都输出，避免前端茎命中失败。
    """
    try:
        from benchmark.runs.common.agent_catalog import (  # noqa: PLC0415
            canonical_agent_name,
        )
    except Exception:
        return [name] if name else []
    cleaned = name.strip()
    if not cleaned:
        return []
    stem = canonical_agent_name(cleaned)
    stem_norm = _normalize_count_key(stem) if stem else ""
    aliases = [cleaned, stem, stem_norm]
    if stem_norm and not stem_norm.endswith("_benchmark"):
        aliases.append(f"{stem_norm}_benchmark")
        # 连字符形态：amap-agent / amap-agent_benchmark（历史 scope id）
        hyphen = stem_norm.replace("_", "-")
        aliases.extend([hyphen, f"{hyphen}_benchmark"])
    # search_eval ↔ search_ai_card
    if cleaned == "search_eval" or stem_norm == "search_ai_card" or stem == "search_ai_card":
        aliases.extend(["search_eval", "search_ai_card"])
    seen: set[str] = set()
    ordered: list[str] = []
    for item in aliases:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


async def _benchmark_registered_at() -> dict[str, datetime]:
    """benchmark 名 → 注册时间锚点（计次只统计锚点之后完成的数据）。

    锚点取 agent 视图物化行（kind=agent_view）与页面 agent 文档
    （kind=benchmark_agent）中最早的落库 ``created_at``：页面注册的
    agent 以文档落库为准，静态 benchmark 以视图首次物化为准。查不
    到锚点的 benchmark 不设时间界（全量终态计次）。
    """
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        KIND_AGENT_VIEW,
        KIND_BENCHMARK_AGENT,
        RuntimeConfigRecord,
    )

    anchors: dict[str, datetime] = {}
    rows = await RuntimeConfigRecord.filter(
        kind__in=[KIND_AGENT_VIEW, KIND_BENCHMARK_AGENT],
    )
    for row in rows:
        if not isinstance(row.payload, dict) or row.created_at is None:
            continue
        name = str(row.payload.get("name") or "").strip()
        if not name:
            continue
        current = anchors.get(name)
        if current is None or row.created_at < current:
            anchors[name] = row.created_at
    return anchors


async def _registration_registered_at() -> dict[str, datetime]:
    """注册文档 id → 落库时间（隔离实例按注册计次的锚点）。

    页面注册的每个 benchmark agent 文档（含同名多实例）各一份，run 快照
    ``agent_registration.id`` 与之对应；已删除文档不作锚点。
    """
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        KIND_BENCHMARK_AGENT,
        RuntimeConfigRecord,
    )

    anchors: dict[str, datetime] = {}
    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        deleted_at__isnull=True,
    )
    for row in rows:
        if not isinstance(row.payload, dict) or row.created_at is None:
            continue
        registration_id = str(row.payload.get("id") or "").strip()
        if registration_id:
            anchors[registration_id] = row.created_at
    return anchors


async def _all_agent_registrations() -> dict[str, dict[str, Any]]:
    """注册 id → ``{name, created_at}``（实时流 ``evaluator_name=stem@reg`` 计次）。"""
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        KIND_BENCHMARK_AGENT,
        RuntimeConfigRecord,
    )

    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        deleted_at__isnull=True,
    )
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = row.payload if isinstance(row.payload, dict) else {}
        name = str(payload.get("name") or "").strip()
        registration_id = str(payload.get("id") or "").strip()
        if not name or not registration_id:
            continue
        out[registration_id] = {"name": name, "created_at": row.created_at}
    return out


async def _example_registrations_for_streaming_count() -> list[dict[str, Any]]:
    """example 双链注册清单：供**旧行** ``evaluator_name=example`` 按 content_version 归属。"""
    from benchmark.records.models.runtime_config_record import (  # noqa: PLC0415
        KIND_BENCHMARK_AGENT,
        RuntimeConfigRecord,
    )
    from benchmark.domain.benchmark_console import (  # noqa: PLC0415
        evaluator_name_candidates,
        registration_content_versions,
    )
    from benchmark.runs.common.agent_catalog import (  # noqa: PLC0415
        canonical_agent_name,
    )

    rows = await RuntimeConfigRecord.filter(
        kind=KIND_BENCHMARK_AGENT,
        deleted_at__isnull=True,
    )
    registrations: list[dict[str, Any]] = []
    for row in rows:
        payload = row.payload if isinstance(row.payload, dict) else {}
        name = str(payload.get("name") or "").strip()
        registration_id = str(payload.get("id") or "").strip()
        if not name or not registration_id:
            continue
        if _normalize_count_key(canonical_agent_name(name)) != "example":
            continue
        want_judge_cv, want_attr_cv = registration_content_versions(payload)
        registrations.append(
            {
                "id": registration_id,
                "name": name,
                "candidates": evaluator_name_candidates(name),
                "want_judge_cv": want_judge_cv,
                "want_attr_cv": want_attr_cv,
            }
        )
    return registrations


class BenchmarkRunLifecycle:
    """Run 创建 / 状态机 / 心跳 / 完成失败 / 恢复."""

    async def create_run(self, spec: rh.BenchmarkRunSpec) -> str:
        """Create a new benchmark run record, return the generated job ``run_id``.

        ``run_id`` 格式见 :func:`rh.generate_benchmark_job_id`（非裸 UUID）。

        Raises:
            RuntimeError: PG 表/列缺失（未执行 ``sql/schema.sql`` 或未
                ``RENAME business_agent_name → service_name``），或写入后读不到行。
        """
        benchmark_name = spec.benchmark_name
        run_id = rh.generate_benchmark_job_id(benchmark_name)
        stamped_config = (
            rh._stamp_bucket(dict(spec.config_json)) if spec.config_json is not None else None
        )
        now = datetime.now(UTC)
        try:
            await rh.BenchmarkRun.create(
                run_id=run_id,
                benchmark_name=benchmark_name,
                service_name=spec.service_name,
                dataset_name=spec.dataset_name or "",
                project_id=spec.project_id,
                status=BenchmarkStatus.INITIALIZING,
                owner=spec.owner or rh.current_worker_id(),
                # Stamp heartbeat at create so never-heartbeat stale recovery
                # does not treat a brand-new row as crash residue.
                heartbeat_at=now,
                total_samples=spec.total_samples,
                total_evaluators=spec.total_evaluators,
                completed_evaluators=0,
                config_json=stamped_config,
            )
        except Exception as exc:
            msg = str(exc).lower()
            schema_hint = ""
            if "benchmark_runs" in msg and (
                "does not exist" in msg
                or "no such table" in msg
                or "undefinedtable" in msg.replace(" ", "")
            ):
                schema_hint = (
                    " — apply services/agent-eval/sql/schema.sql (CREATE TABLE benchmark_runs)"
                )
            elif "service_name" in msg and (
                "does not exist" in msg or "no column" in msg or "no such column" in msg
            ):
                schema_hint = (
                    " — run ALTER TABLE benchmark_runs "
                    "RENAME COLUMN business_agent_name TO service_name"
                )
            rh.logger.exception(
                "benchmark.run_create_failed run_id=%s name=%s%s",
                run_id,
                benchmark_name,
                schema_hint,
            )
            raise RuntimeError(
                f"benchmark_runs insert failed for run_id={run_id}{schema_hint}: {exc}"
            ) from exc

        # Read-after-write: catch silent no-persist / wrong DB connection early.
        persisted = await rh.BenchmarkRun.filter(run_id=run_id).first()
        if persisted is None:
            raise RuntimeError(
                f"benchmark_runs insert reported success but row missing "
                f"(run_id={run_id}); check AGENT_EVAL_DATABASE_URL / search_path"
            )

        rh.logger.info(
            "benchmark.run_created run_id=%s name=%s agent=%s status=%s",
            run_id,
            benchmark_name,
            spec.service_name,
            persisted.status,
        )
        return run_id

    async def cas_finalize_run(
        self,
        *,
        run_id: str,
        total_samples: int,
        total_evaluators: int,
    ) -> int:
        """CAS: ``ACCEPTING → RUNNING`` in a single conditional UPDATE.

        Combines status transition + field updates to avoid a second
        unconditional ``update_run_status`` that could race with a
        concurrent worker already advancing the run past ``RUNNING``.

        Returns:
            Affected row count (0 = already finalized by another node).
        """
        updated = await rh.BenchmarkRun.filter(
            run_id=run_id,
            status=BenchmarkStatus.ACCEPTING.value,
        ).update(
            status=BenchmarkStatus.RUNNING.value,
            started_at=datetime.now(UTC),
            total_samples=total_samples,
            total_evaluators=total_evaluators,
        )
        return updated

    async def update_run_status(
        self,
        run_id: str,
        status: BenchmarkStatus,
        **fields: object,
    ) -> None:
        """Update run status and optional fields.

        若 ``fields`` 含 ``metrics_json`` / ``report_meta_json`` / ``config_json``，
        与已有 bucket **merge** 后写入，避免中途整桶覆盖抹掉增量字段。
        """
        update_fields: dict[str, Any] = {"status": status}
        needs_merge = any(k in rh._MERGE_BUCKET_FIELDS for k in fields)
        if needs_merge:
            run = await rh.BenchmarkRun.filter(run_id=run_id).first()
            if run is None:
                rh.logger.error(
                    "benchmark.update_status_missing_run run_id=%s",
                    run_id,
                )
                return
            for key, value in fields.items():
                if key == "metrics_json" and value is not None:
                    update_fields[key] = rh.merge_metrics_json(
                        run.metrics_json if isinstance(run.metrics_json, dict) else None,
                        dict(value),
                    )
                elif key == "report_meta_json" and value is not None:
                    update_fields[key] = rh.merge_report_meta_json(
                        run.report_meta_json if isinstance(run.report_meta_json, dict) else None,
                        dict(value),
                    )
                elif key == "config_json" and value is not None:
                    update_fields[key] = rh.merge_config_json(
                        run.config_json if isinstance(run.config_json, dict) else None,
                        dict(value),
                    )
                else:
                    update_fields[key] = value
        else:
            update_fields.update(fields)
        updated = await rh.BenchmarkRun.filter(run_id=run_id).update(**update_fields)
        if updated == 0:
            rh.logger.error(
                "benchmark.update_status_no_row run_id=%s status=%s "
                "(create_run missing or wrong DB?)",
                run_id,
                status,
            )

    async def heartbeat(self, run_id: str) -> None:
        """Update ``heartbeat_at`` to current UTC time."""
        await heartbeat_run(
            rh.BenchmarkRun,
            run_id,
            logger=rh.logger,
            no_row_log_message="benchmark.heartbeat_no_row run_id=%s",
        )

    async def update_evaluator_progress(self, run_id: str) -> None:
        """Atomically increment ``completed_evaluators`` on the run."""
        updated = await rh.BenchmarkRun.filter(run_id=run_id).update(
            completed_evaluators=rh.F("completed_evaluators") + 1,
        )
        if updated == 0:
            rh.logger.error(
                "benchmark.progress_no_row run_id=%s",
                run_id,
            )

    async def increment_progress(
        self,
        run_id: str,
        *,
        evaluated_delta: int = 0,
        failed_delta: int = 0,
    ) -> None:
        """原子递增 ``evaluated_items`` / ``failed_items`` 计数器。

        与 :meth:`DatasetEvalRunService.increment_progress` 对齐，
        供 BullMQ benchmark/dataset worker 在 case 评测完成后调用。
        """
        update_fields: dict[str, Any] = {}
        if evaluated_delta:
            update_fields["evaluated_items"] = rh.F("evaluated_items") + evaluated_delta
        if failed_delta:
            update_fields["failed_items"] = rh.F("failed_items") + failed_delta
        if not update_fields:
            return
        updated = await rh.BenchmarkRun.filter(run_id=run_id).update(**update_fields)
        if updated == 0:
            rh.logger.error("benchmark.progress_no_row run_id=%s", run_id)

    async def write_parse_failed_stubs(
        self,
        write: rh.ParseFailedStubWrite,
    ) -> int:
        """parse 失败 item 的 eval_failed stub 落库 + 进度推进（§7.6，进度列模型）。

        统一提交管线的 item 级失败语义：拦截器解析失败的 item 写 failed 行
        （报告可见）且不入队；fused 模式写 1 行（criterion_id=fused），
        per_eval 模式每 criterion 一行；随后 ``failed_items`` 推进对应数量，
        保证 ``(evaluated + failed) >= total`` 的 CAS 收敛。

        ``item`` 为原始 case dict（从中提取 ``input`` / ``output`` 落库）。

        Returns:
            实际写入的行数（供调用方对齐计数语义）。
        """
        run_id = write.run_id
        item_id = write.item_id
        reason = write.reason
        criterion_ids = write.criterion_ids
        err = reason[:2000]
        written = 0
        item = write.item if isinstance(write.item, dict) else {}
        item_input = item.get("input") if isinstance(item.get("input"), dict) else None
        item_output = item.get("output") if isinstance(item.get("output"), dict) else None
        for criterion_id in criterion_ids:
            await rh.BenchmarkRunItem.update_or_create(
                run_id=run_id,
                item_id=item_id,
                criterion_id=criterion_id,
                defaults={
                    "evaluator_ref": "parse",
                    "status": "failed",
                    "score": 0.0,
                    "error_message": err,
                    "input": item_input,
                    "output": item_output,
                },
            )
            written += 1
        if written:
            await self.increment_progress(run_id, failed_delta=written)
        return written

    # ═══════════════════════════════════════════════════════════════════════
    # Incremental JSON buckets
    # ═══════════════════════════════════════════════════════════════════════

    async def complete_run(
        self,
        write: rh.CompleteRunWrite,
    ) -> None:
        """标记 COMPLETED（或停止门闩下的 STOPPED）；metrics / report_meta merge。

        ``report_meta_json=None`` 时**保留**已有 report meta（不清空）。
        若 ``config_json.stop_requested_at`` 已设置，终态写 ``stopped`` 并
        【仅 stop】合并 ``partial_report_meta``（``completed_cases`` 等）；
        正常 completed/failed **禁止**写 partial / 压期望 / 改 ``total_samples``。
        """
        run_id = write.run_id
        final_score = write.final_score
        metrics_json = write.metrics_json
        report_meta_json = write.report_meta_json
        duration_ms = write.duration_ms
        run = await rh.BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            rh.logger.error(
                "benchmark.complete_missing_run run_id=%s",
                run_id,
            )
            return

        merged_metrics = rh.merge_metrics_json(
            run.metrics_json if isinstance(run.metrics_json, dict) else None,
            metrics_json,
        )
        config = run.config_json if isinstance(run.config_json, dict) else {}
        stopped = is_stop_requested(config)
        terminal = BenchmarkStatus.STOPPED if stopped else BenchmarkStatus.COMPLETED
        update_fields: dict[str, Any] = {
            "status": terminal,
            "final_score": final_score,
            "metrics_json": merged_metrics,
            "duration_ms": duration_ms,
            "finished_at": datetime.now(UTC),
        }
        if stopped:
            update_fields["error_message"] = STOP_REASON
        existing_meta = run.report_meta_json if isinstance(run.report_meta_json, dict) else None
        if report_meta_json is not None or stopped:
            base_meta = rh.merge_report_meta_json(existing_meta, report_meta_json or {})
            if stopped:
                completed_cases = int(run.evaluated_items or 0) + int(run.failed_items or 0)
                base_meta = partial_report_meta(
                    completed_cases=completed_cases,
                    stop_requested_at=(
                        str(config.get("stop_requested_at"))
                        if config.get("stop_requested_at")
                        else None
                    ),
                    existing=base_meta if isinstance(base_meta, dict) else {},
                )
            update_fields["report_meta_json"] = base_meta

        await rh.BenchmarkRun.filter(run_id=run_id).update(**update_fields)
        rh.logger.info(
            "benchmark.run_%s run_id=%s score=%.4f duration=%sms",
            "stopped" if stopped else "completed",
            run_id,
            final_score,
            duration_ms,
        )

    @property
    def _lifecycle_binding(self) -> RunLifecycleBinding:
        """benchmark 侧 run 生命周期绑定（fail / recover 共用）。"""
        return RunLifecycleBinding(
            model=rh.BenchmarkRun,
            failed_status=BenchmarkStatus.FAILED,
            non_terminal_statuses=list(self._NON_TERMINAL_STATUSES),
            logger=rh.logger,
        )

    async def fail_run(self, *, run_id: str, reason: str) -> None:
        """Mark a run as FAILED and cascade to orphaned tasks + run items.

        All non-completed ``benchmark_run_items`` for this run are also marked
        FAILED so that orphaned running items don't linger as
        false progress signals.
        """
        await fail_run(
            self._lifecycle_binding,
            run_id,
            FailRunSpec(
                reason=reason,
                no_row_log_message=(
                    "benchmark.fail_run_no_row run_id=%s reason=%s "
                    "(row was never created — check create_run / schema)"
                ),
                failed_log_message="benchmark.run_failed run_id=%s reason=%s",
            ),
        )
        items_failed = await self._fail_run_items(run_id, reason)
        if items_failed:
            rh.logger.info(
                "benchmark.run_items_failed run_id=%s count=%d",
                run_id,
                items_failed,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Query
    # ═══════════════════════════════════════════════════════════════════════

    async def get_run(self, run_id: str) -> rh.BenchmarkRun | None:
        """Fetch a single run by ``run_id``（已软删行视为不存在）。"""
        return await rh.BenchmarkRun.filter(
            run_id=run_id,
            deleted_at__isnull=True,
        ).first()

    async def get_run_by_pk(self, run_pk: object) -> rh.BenchmarkRun | None:
        """Fetch a single run by 行主键 ``id``（已软删行视为不存在）。"""
        return await rh.BenchmarkRun.filter(
            id=run_pk,
            deleted_at__isnull=True,
        ).first()

    async def get_run_by_id_or_pk(self, run_key: str) -> rh.BenchmarkRun | None:
        """解析路径键并始终按业务 ``run_id`` 取数。

        - 前端已有业务 ``run_id``：直接 ``get_run``。
        - 仅有行主键 UUID：先 ``get_run_by_pk`` 取出 ``run_id``，再 ``get_run(run_id)``。
        """
        key = str(run_key).strip()
        if not key:
            return None
        run = await self.get_run(key)
        if run is not None:
            return run
        try:
            pk = UUID(key)
        except (TypeError, ValueError, AttributeError):
            return None
        by_pk = await self.get_run_by_pk(pk)
        if by_pk is None:
            return None
        business_id = str(getattr(by_pk, "run_id", "") or "").strip()
        if not business_id:
            return by_pk
        # 明细表按业务 run_id 关联；pk 只做解析，拉数统一走 get_run。
        return await self.get_run(business_id) or by_pk

    async def build_run_progress(self, run: rh.BenchmarkRun) -> dict[str, Any]:
        """聚合 run 执行进度：per-criterion 样本统计。

        数据源全部实时查 PG——多节点部署下任意节点均可查询（分布式安全）：
        - per-criterion 统计：``metrics_json.criterion_metrics``
          （每个 criterion 完成即增量写入，中途轮询可见）
        - run 级进度：``evaluated_items`` / ``failed_items`` / ``total_samples``
          （BullMQ 迁移后由 CAS 计数器驱动，不再依赖 shard task 表）
        """
        criterion_metrics = (run.metrics_json or {}).get("criterion_metrics") or {}
        evaluators = [
            {
                "name": criterion_id,
                "total_cases": run.total_samples,
                "success": metric.get("success_count") or 0,
                "failed": metric.get("failed_count") or 0,
            }
            for criterion_id, metric in criterion_metrics.items()
        ]
        return {
            "total_items": run.total_samples,
            "evaluated_items": run.evaluated_items,
            "failed_items": run.failed_items,
            "sample_size": (run.config_json or {}).get("sample_size"),
            "evaluators": evaluators,
        }

    async def list_runs(self, query: "ListRunsQuery | None" = None) -> list[rh.BenchmarkRun]:
        """List runs, optionally filtered by benchmark name（跳过已软删行）。

        ``benchmark_name`` 可为逻辑名或 registrationId（列已优先落注册 id，
        逻辑名在 ``config_json.benchmark_services`` / 快照 name）。
        ``task_id`` 按 run 快照 ``config_json.eval_task_id`` 归属过滤
        （页面报告按任务隔离的数据源）：只返回该评测任务产生的 run。
        """
        from benchmark.domain.eval_config import run_matches_benchmark_name  # noqa: PLC0415

        q = query or ListRunsQuery()
        benchmark_name = q.benchmark_name
        task_id = q.task_id
        limit = q.limit
        offset = q.offset
        qs = rh.BenchmarkRun.filter(deleted_at__isnull=True)
        # config_json 过滤（任务归属 / 逻辑名）走 python 侧，PG/sqlite 一致。
        if benchmark_name or task_id:
            rows = await qs.order_by("-created_at")
            matched: list[rh.BenchmarkRun] = []
            for run in rows:
                cfg = run.config_json if isinstance(run.config_json, dict) else {}
                if task_id and cfg.get("eval_task_id") != task_id:
                    continue
                if benchmark_name and not run_matches_benchmark_name(
                    column_name=run.benchmark_name,
                    config=cfg,
                    query_name=benchmark_name,
                ):
                    continue
                matched.append(run)
            return matched[offset : offset + limit]
        return await qs.order_by("-created_at").limit(limit).offset(offset)

    async def _count_trace_completed_since(
        self,
        stem: str,
        total: int,
        anchor: datetime | None,
    ) -> int:
        """某 evaluator 茎在锚点之后完成评判的 trace 数（无锚点 = 全量）。"""
        if anchor is None:
            return total
        return (
            await rh.TraceRecord.filter(
                evaluator_name=stem,
                created_at__gte=anchor,
            )
            .exclude(status=TraceStatus.NEW.value)
            .count()
        )

    async def count_runs_by_benchmark(self) -> list[dict[str, Any]]:
        """按 ``evaluator_name`` / ``registrationId`` 统计评测次数，供代理卡片读取。

        计次口径（与页面「调用次数」语义对齐）：
        - **分链分量**：``batchCount`` = ``benchmark_run_items``（带注册
          快照的终态批量）；``streamingCount`` = ``trace_pipeline_records``
          （实时流）。``evaluationCount = batchCount + streamingCount``。
        - **实时流（新）**：``evaluator_name`` 为 ``stem@registrationId``
          或 ``stem@reg@stream``（兼收裸 registrationId 旧行）时按**注册**
          聚合——同 agent 多数据源（多 stream）在卡片上合计。
        - **实时流（共享茎）**：无注册落库的 ``amap_agent`` / ``amap_agent@stream``
          等仍按名聚合（``registrationId`` 空），供 streaming-only 卡片命中。
        - **实时流（example 旧行）**：``evaluator_name=example`` 仍按
          prompt ``content_version`` 归属到 ``registrationId``。
        - **批量**：仅计带 ``agent_registration.id`` 快照的终态 run。
          排除 ``trace_daily_report`` T+1 日报。
        - 响应里 ``registrationId`` 仍为**裸**注册文档 id（WebUI
          ``agent.id`` 精确匹配）；落库键是 ``stem@reg[@stream]``。
        """
        registered_at = await _benchmark_registered_at()
        registration_anchors = await _registration_registered_at()
        batch_counts = await self._count_batch_items_by_registration(
            registered_at,
            registration_anchors,
        )
        streaming_counts = await self._count_streaming_traces(
            registered_at,
            registration_anchors,
        )
        keys = set(batch_counts) | set(streaming_counts)
        result: list[dict[str, Any]] = []
        for name, registration_id in sorted(keys, key=lambda item: (item[0], item[1] or "")):
            batch = int(batch_counts.get((name, registration_id), 0))
            streaming = int(streaming_counts.get((name, registration_id), 0))
            total = batch + streaming
            if total <= 0:
                continue
            entry: dict[str, Any] = {
                "benchmarkName": name,
                "evaluationCount": total,
                "batchCount": batch,
                "streamingCount": streaming,
            }
            if registration_id:
                entry["registrationId"] = registration_id
            result.append(entry)
        return result

    async def _count_streaming_traces(  # noqa: PLR0912 — 复合键/裸 id/共享茎/example 分链计次
        self,
        registered_at: dict[str, datetime],
        registration_anchors: dict[str, datetime],
    ) -> dict[tuple[str, str | None], int]:
        """实时流计次：``stem@reg[@stream]`` 按注册合计；共享茎 / example 旧行兼容。"""
        from benchmark.domain.eval_config import parse_eval_identity_key  # noqa: PLC0415
        from benchmark.runs.common.agent_catalog import (  # noqa: PLC0415
            canonical_agent_name,
        )

        counts: dict[tuple[str, str | None], int] = {}
        registrations = await _all_agent_registrations()
        try:
            trace_rows = (
                await rh.TraceRecord.exclude(status=TraceStatus.NEW.value)
                .group_by("evaluator_name")
                .annotate(evaluation_count=Count("id"))
                .values("evaluator_name", "evaluation_count")
            )
        except Exception:
            rh.logger.warning("run_counts.trace_lookup_failed", exc_info=True)
            trace_rows = []
        for row in trace_rows:
            raw = str(row["evaluator_name"] or "").strip()
            if not raw:
                continue
            called = int(row["evaluation_count"])
            stem_part, parsed_reg, _stream = parse_eval_identity_key(raw)
            # 新落库：stem@reg 或 stem@reg@stream（兼收裸 registrationId）
            reg_key = (
                parsed_reg
                if (parsed_reg and parsed_reg in registrations)
                else (raw if raw in registrations else None)
            )
            if reg_key is not None:
                info = registrations[reg_key]
                name = str(info["name"])
                anchor = registration_anchors.get(reg_key) or info.get("created_at")
                stemmed = await self._count_trace_completed_since(raw, called, anchor)
                if stemmed > 0:
                    counts[(name, reg_key)] = counts.get((name, reg_key), 0) + stemmed
                continue
            # 无注册：取茎段（stem@stream → stem）；共享茎卡片合计各 stream
            raw_stem = stem_part or raw
            stem = canonical_agent_name(raw_stem)
            # 复合键（带 stream）时 raw != stem，仍按茎聚合
            if "@" not in raw and _normalize_count_key(raw) != _normalize_count_key(stem):
                continue
            if _normalize_count_key(stem) == "example":
                continue
            anchor = registered_at.get(stem)
            if anchor is None:
                for alias in _count_name_aliases(stem):
                    anchor = registered_at.get(alias)
                    if anchor is not None:
                        break
            stemmed = await self._count_trace_completed_since(raw, called, anchor)
            if stemmed <= 0:
                continue
            for alias in _count_name_aliases(stem):
                key = (alias, None)
                counts[key] = counts.get(key, 0) + stemmed
        example_counts = await self._count_example_streaming_by_registration(
            registration_anchors,
        )
        for key, value in example_counts.items():
            counts[key] = counts.get(key, 0) + value
        return counts

    async def _count_example_streaming_by_registration(
        self,
        registration_anchors: dict[str, datetime],
    ) -> dict[tuple[str, str | None], int]:
        """example 旧行：``evaluator_name=example`` → 唯一 registrationId。

        新落库已写 ``example@registrationId``，由 ``_count_streaming_traces``
        主路径计次；本方法只消化迁移前的茎名行。

        归属优先级（每行至多归属一个注册，避免多版本串数/重复计）：
        1. ``trace_meta_json.agent_registration_id``（若命中 example 注册）
        2. judge/attribution ``content_version`` 精确匹配且唯一
        3. 无版本戳行仅在「恰好一个无定制注册」时归入该注册；多无定制则丢弃
        """
        from benchmark.domain.benchmark_console import (  # noqa: PLC0415
            _ATTR_CONTENT_VERSION_PATHS,
            _JUDGE_CONTENT_VERSION_PATHS,
            AgentRegistrationMatch,
            content_version_from_bucket,
            row_matches_agent_registration,
        )
        from benchmark.runs.common.agent_catalog import (  # noqa: PLC0415
            canonical_agent_name,
        )

        registrations = await _example_registrations_for_streaming_count()
        if not registrations:
            return {}
        by_id = {str(reg["id"]): reg for reg in registrations}
        try:
            rows = (
                await rh.TraceRecord.filter(
                    evaluator_name="example",
                )
                .exclude(status=TraceStatus.NEW.value)
                .values(
                    "judge_json",
                    "attribution_detail_json",
                    "trace_meta_json",
                    "created_at",
                )
            )
        except Exception:
            rh.logger.warning(
                "run_counts.example_streaming_lookup_failed",
                exc_info=True,
            )
            return {}
        counts: dict[tuple[str, str | None], int] = {}
        for row in rows:
            created_at = row.get("created_at")
            chosen: dict[str, Any] | None = None
            meta = row.get("trace_meta_json")
            meta_reg = ""
            if isinstance(meta, dict):
                meta_reg = str(meta.get("agent_registration_id") or "").strip()
            if meta_reg and meta_reg in by_id:
                chosen = by_id[meta_reg]
            else:
                judge_cv = content_version_from_bucket(
                    row.get("judge_json"),
                    _JUDGE_CONTENT_VERSION_PATHS,
                )
                attr_cv = content_version_from_bucket(
                    row.get("attribution_detail_json"),
                    _ATTR_CONTENT_VERSION_PATHS,
                )
                matched = [
                    reg
                    for reg in registrations
                    if row_matches_agent_registration(
                        AgentRegistrationMatch(
                            evaluator_name="example",
                            candidates=reg["candidates"],
                            judge_cv=judge_cv,
                            attr_cv=attr_cv,
                            want_judge_cv=reg["want_judge_cv"],
                            want_attr_cv=reg["want_attr_cv"],
                        )
                    )
                ]
                if len(matched) == 1:
                    chosen = matched[0]
                elif len(matched) > 1 and (judge_cv is not None or attr_cv is not None):
                    # 多注册抢同一版本戳：取第一个带对应 want 的（稳定但不串无戳行）
                    versioned = [
                        reg
                        for reg in matched
                        if (judge_cv is not None and reg["want_judge_cv"] == judge_cv)
                        or (attr_cv is not None and reg["want_attr_cv"] == attr_cv)
                    ]
                    if len(versioned) == 1:
                        chosen = versioned[0]
                # len(matched)>1 且无版本戳 → 歧义，不计（多无定制注册不能共享）
            if chosen is None:
                continue
            anchor = registration_anchors.get(chosen["id"])
            if anchor is not None and created_at is not None and created_at < anchor:
                continue
            key = (str(chosen["name"]), str(chosen["id"]))
            counts[key] = counts.get(key, 0) + 1
        return {
            key: value
            for key, value in counts.items()
            if _normalize_count_key(canonical_agent_name(key[0])) == "example"
        }

    async def _count_batch_items_by_registration(
        self,
        registered_at: dict[str, datetime],
        registration_anchors: dict[str, datetime],
    ) -> dict[tuple[str, str | None], int]:
        """批量计次：仅带注册快照的终态 run，按 registrationId 聚合。

        无 ``agent_registration`` 快照的历史批量 **不计入** 按名总数
        （否则会把 ``example_benchmark`` 旧 case 算进新注册卡片）。
        锚点用注册文档创建时间；锚点之前完成的 run 不计。
        T+1 日报 run（``trace_daily_report``）不计。
        """
        counts: dict[tuple[str, str | None], int] = {}
        terminal = [
            BenchmarkStatus.COMPLETED.value,
            BenchmarkStatus.FAILED.value,
            BenchmarkStatus.STOPPED.value,
        ]
        runs = await rh.BenchmarkRun.filter(
            deleted_at__isnull=True,
            status__in=terminal,
        )
        eligible: dict[str, tuple[str, str]] = {}
        for run in runs:
            config = run.config_json if isinstance(run.config_json, dict) else {}
            if config.get("trace_daily_report") is True:
                continue
            snapshot = config.get("agent_registration")
            if not isinstance(snapshot, dict) or not snapshot.get("id"):
                continue
            registration_id = str(snapshot["id"])
            # 卡片名用快照/逻辑名（列可能已是 registrationId，不能当 name 键）
            card_name = (
                str(snapshot.get("name") or "").strip()
                or str(config.get("benchmark_services") or "").strip()
                or str(config.get("benchmark_component") or "").strip()
                or str(run.benchmark_name or "").strip()
            )
            if not card_name:
                continue
            anchor = registration_anchors.get(registration_id) or registered_at.get(card_name)
            if anchor is None:
                for alias in _count_name_aliases(card_name):
                    anchor = registered_at.get(alias)
                    if anchor is not None:
                        break
            if anchor is not None and (run.created_at is None or run.created_at < anchor):
                continue
            eligible[run.run_id] = (card_name, registration_id)
        if eligible:
            pairs = await rh.BenchmarkRunItem.filter(
                run_id__in=list(eligible),
            ).values("run_id", "item_id")
            seen: set[tuple[str, str]] = set()
            for pair in pairs:
                seen.add((str(pair["run_id"]), str(pair["item_id"])))
            for run_id, _item_id in seen:
                key = eligible[run_id]
                counts[key] = counts.get(key, 0) + 1
        return counts

    # ═══════════════════════════════════════════════════════════════════════
    # Delete
    # ═══════════════════════════════════════════════════════════════════════

    async def delete_run(self, run_pk: object) -> bool:
        """软删单次评测：按行主键 ``id`` 给该行打软删标记（is_deleted
        + deleted_at，与 BaseModelWithSoftDelete 语义一致）。

        不物理删行、不动 pipeline 明细；标记后 get/list/count/日报回填
        等查询全部跳过该行，页面不再展示。返回 False 表示行不存在或
        已被删除（重复删除按 404 处理，避免静默成功掩盖误操作）。
        """
        marked = await rh.BenchmarkRun.filter(
            id=run_pk,
            deleted_at__isnull=True,
        ).update(is_deleted=True, deleted_at=datetime.now(UTC))
        if marked:
            rh.logger.info("benchmark.run_soft_deleted id=%s", run_pk)
        return bool(marked)

    # ═══════════════════════════════════════════════════════════════════════
    # Stale run recovery
    # ═══════════════════════════════════════════════════════════════════════

    _NON_TERMINAL_STATUSES: tuple[str, ...] = (
        BenchmarkStatus.INITIALIZING,
        # 流式 ingest 中：append_items 会 heartbeat，但也属于非终态，
        # ingest 中断的残留由 recover_stale_runs 按超时清理。
        BenchmarkStatus.ACCEPTING,
        BenchmarkStatus.RUNNING,
        BenchmarkStatus.AGGREGATING,
        # GENERATING_REPORT 不在此列表中：该状态表示 CAS 已赢、
        # 聚合/报告阶段正在执行，进程崩溃后由 healer 的
        # _heal_stuck_generating_report 方法恢复（reset → re-finalize），
        # 而非被 recover_stale_runs 标记为 FAILED。
    )

    async def fail_inflight_runs_for_owner(
        self,
        *,
        reason: str,
        owner: str | None = None,
    ) -> int:
        """Immediately mark all non-terminal runs for ``owner`` as FAILED.

        Used on graceful shutdown / drain cancel so queued or mid-flight runs
        are not left as ``initializing`` / ``running`` after the process
        exits.  Orphaned tasks for each failed run are also marked FAILED.
        """
        current_owner = owner or rh.current_worker_id()
        failed_ids = await fail_inflight_runs(
            self._lifecycle_binding,
            owner=current_owner,
            reason=reason,
            log_message="benchmark.fail_inflight run_id=%s owner=%s reason=%s",
        )
        for run_id in failed_ids:
            items_failed = await self._fail_run_items(run_id, reason)
            if items_failed:
                rh.logger.info(
                    "benchmark.run_items_failed run_id=%s count=%d",
                    run_id,
                    items_failed,
                )
        return len(failed_ids)

    async def recover_stale_runs(
        self,
        *,
        stale_timeout_s: int = RUN_STALE_TIMEOUT_S,
    ) -> int:
        """Mark stale non-terminal runs as FAILED (timeout / crash / deploy).

        A run is considered stale if:
        - ``heartbeat_at`` older than ``stale_timeout_s``, OR
        - ``heartbeat_at`` is NULL and ``created_at`` older than cutoff

        Covers force-kill and rolling deploy: the old process dies, heartbeats
        stop, and the next startup recovers leftover rows. Does NOT re-run.

        Recovered runs' orphaned run_items are also marked
        FAILED to prevent false progress signals.
        """
        recovered_ids = await recover_stale_runs(
            self._lifecycle_binding,
            stale_timeout_s=stale_timeout_s,
            log_message=(
                "benchmark.recovered_stale_run run_id=%s owner=%s heartbeat_at=%s prev_status=%s"
            ),
        )
        for run_id in recovered_ids:
            items_failed = await self._fail_run_items(
                run_id,
                STALE_RECOVERY_REASON,
            )
            if items_failed:
                rh.logger.info(
                    "benchmark.run_items_failed run_id=%s count=%d",
                    run_id,
                    items_failed,
                )
        return len(recovered_ids)

    async def _fail_run_items(self, run_id: str, reason: str) -> int:
        """Mark all in-progress ``benchmark_run_items`` rows as FAILED.

        Called when the parent run is marked FAILED (timeout / crash /
        shutdown). Only items still in ``running`` state are cascaded —
        items already ``success`` or ``failed`` are preserved so that
        completed evaluations are not lost.

        Returns:
            Number of rows updated.
        """
        return (
            await rh.BenchmarkRunItem.filter(
                run_id=run_id,
            )
            .exclude(status__in=["failed", "success"])
            .update(
                status="failed",
                score=0.0,
                error_message=reason[:2000],
            )
        )

    # ═════════════════════════════════════════════════════════════════════════
    # External trigger methods (§5.5 / §5.6 / §12.1 / §15.3)
    # ═════════════════════════════════════════════════════════════════════════

    #: 外部触发 INITIALIZING 阶段独立短阈值（正常初始化分钟级完成，§5.6.4）
    EXTERNAL_INIT_STALE_TIMEOUT_S = 600  # 10 分钟

    #: 外部触发 RUNNING 阶段回调超时阈值（§14.2）
    EXTERNAL_CALLBACK_TIMEOUT_S = 4 * 3600  # 4 小时

    async def create_placeholder_run(
        self,
        run_id: str,
        spec: rh.BenchmarkRunSpec,
    ) -> bool:
        """创建占位 run（initializing），run_id 由调用方指定（§3.4/§5.1）。

        与 ``create_run`` 的区别：``run_id`` 由调用方按日期生成
        （``{benchmark_name}-{yyyymmdd}``），而非 ``rh.generate_benchmark_job_id``。

        Returns:
            True 表示创建成功；False 表示 run_id 已存在（幂等跳过）。
        """
        benchmark_name = spec.benchmark_name
        now = datetime.now(UTC)
        base_config: dict[str, Any] = {
            "external": True,
            "pull_attempt_count": 1,
        }
        if spec.config_json:
            base_config.update(spec.config_json)
        stamped_config = rh._stamp_bucket(base_config)
        try:
            await rh.BenchmarkRun.create(
                run_id=run_id,
                benchmark_name=benchmark_name,
                service_name=spec.service_name,
                dataset_name=spec.dataset_name or "",
                project_id=spec.project_id,
                status=BenchmarkStatus.INITIALIZING,
                owner=rh.current_worker_id(),
                heartbeat_at=now,
                total_samples=0,
                total_evaluators=0,
                completed_evaluators=0,
                config_json=stamped_config,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "unique" in msg or "duplicate" in msg or "conflict" in msg:
                rh.logger.info(
                    "benchmark.placeholder_skip run_id=%s (already exists)",
                    run_id,
                )
                return False
            raise
        rh.logger.info(
            "benchmark.placeholder_created run_id=%s name=%s",
            run_id,
            benchmark_name,
        )
        return True

    async def create_external_run(
        self,
        run_id: str,
        spec: rh.BenchmarkRunSpec,
    ) -> None:
        """补全外部触发 run 配置（merge 语义更新 config_json，§5.5）。

        以 **merge 语义**更新 config_json，不得覆盖 ``pull_attempt_count``
        （§5.6.1）。
        """
        run = await rh.BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            rh.logger.error(
                "benchmark.create_external_run_missing run_id=%s",
                run_id,
            )
            return

        existing_config = run.config_json if isinstance(run.config_json, dict) else {}
        merged = dict(existing_config)
        if spec.config_json:
            merged.update(spec.config_json)
        # 保留已有的 pull_attempt_count（merge 语义，§5.6.1）
        if "pull_attempt_count" not in merged:
            merged["pull_attempt_count"] = 1

        run.config_json = rh._stamp_bucket(merged)
        run.total_samples = spec.total_samples
        run.total_evaluators = spec.total_evaluators
        if spec.dataset_name is not None:
            run.dataset_name = spec.dataset_name
        if spec.project_id is not None:
            run.project_id = spec.project_id
        await run.save(
            update_fields=[
                "config_json",
                "total_samples",
                "total_evaluators",
                "dataset_name",
                "project_id",
            ],
        )
        rh.logger.info(
            "benchmark.external_run_updated run_id=%s total_samples=%d",
            run_id,
            spec.total_samples,
        )

    async def cas_reset_failed_run(
        self,
        run_id: str,
        *,
        max_attempts: int = 3,
    ) -> bool:
        """FAILED → INITIALIZING 条件重建（§5.6.2）。

        多节点并发仅一个影响行数=1。

        Returns:
            True 表示赢得 CAS（重建成功）；False 表示未赢得或已达上限。
        """
        async with rh.in_transaction() as conn:
            run = (
                await rh.BenchmarkRun.filter(
                    run_id=run_id,
                )
                .select_for_update()
                .using_db(conn)
                .first()
            )
            if run is None:
                return False
            if run.status != BenchmarkStatus.FAILED.value:
                return False

            config = run.config_json if isinstance(run.config_json, dict) else {}
            attempt = config.get("pull_attempt_count", 1)
            if attempt >= max_attempts:
                rh.logger.error(
                    "benchmark.cas_reset_max_attempts run_id=%s attempts=%d",
                    run_id,
                    attempt,
                )
                return False

            # 重置归因计数器 + pull_attempt_count + 1
            # judge 完成改由 get_pending_count==0 检测，不再需要 judge_received_count
            config["attribution_received_count"] = 0
            config["attribution_expected_count"] = 0
            config["pull_attempt_count"] = attempt + 1

            run.status = BenchmarkStatus.INITIALIZING.value
            run.config_json = rh._stamp_bucket(config)
            run.heartbeat_at = datetime.now(UTC)
            await run.save(
                using_db=conn,
                update_fields=["status", "config_json", "heartbeat_at"],
            )
        rh.logger.info(
            "benchmark.cas_reset_success run_id=%s attempt=%d",
            run_id,
            attempt + 1,
        )
        return True


__all__ = ["BenchmarkRunLifecycle"]
