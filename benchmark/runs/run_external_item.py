"""外部 judge/归因 item CAS 与查询

Mixin for :class:`~benchmark.runs.run_service.BenchmarkService`.
Symbols resolve through :mod:`run_helpers` so tests can patch
``benchmark.runs.run_helpers.*``.
"""

from typing import Any, Literal

from benchmark.runs import run_helpers as rh
from benchmark.runs.schema import FUSED_CRITERION_ID, ExternalJudgeItemResult


class BenchmarkRunExternalItem:
    """外部 judge/归因 item CAS 与查询."""

    async def _increment_attribution_counter_once(
        self,
        bump: rh._AttributionCounterBump,
    ) -> tuple[int, dict[str, Any], bool]:
        """按 item 幂等递增归因侧计数器（expected / received 共用）。

        Returns:
            ``(new_count, config, claimed)``。``claimed=True`` 表示本次首次计数。
        """
        run_id = bump.run_id
        item_id = bump.item_id
        count_key = bump.count_key
        ids_key = bump.ids_key
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
                raise RuntimeError(f"run not found: {run_id}")
            config = dict(run.config_json) if isinstance(run.config_json, dict) else {}
            seen_ids = [str(x) for x in (config.get(ids_key) or [])]
            item_key = str(item_id)
            if item_key in seen_ids:
                return int(config.get(count_key, 0) or 0), config, False
            seen_ids.append(item_key)
            config[ids_key] = seen_ids
            new_count = int(config.get(count_key, 0) or 0) + 1
            config[count_key] = new_count
            run.config_json = rh._stamp_bucket(config)
            await run.save(using_db=conn, update_fields=["config_json"])
            return new_count, config, True

    async def increment_attribution_expected_once(
        self,
        run_id: str,
        item_id: str,
    ) -> tuple[int, dict[str, Any], bool]:
        """按 item 幂等递增 ``attribution_expected_count``。

        与 ``increment_attribution_received_once`` 对称：重复入队 / healer
        回放 / status 重置后再 advanced 不得抬高 expected，否则
        ``received`` 永远追不上（ieval_external-20260909：136/101）。

        Returns:
            ``(new_count, config, claimed)``。``claimed=True`` 表示本次首次计数。
        """
        return await self._increment_attribution_counter_once(
            rh._AttributionCounterBump(
                run_id=run_id,
                item_id=item_id,
                count_key="attribution_expected_count",
                ids_key="attribution_expected_item_ids",
            ),
        )

    async def increment_attribution_received_once(
        self,
        run_id: str,
        item_id: str,
    ) -> tuple[int, dict[str, Any], bool]:
        """按 item 幂等递增 ``attribution_received_count``。

        回调与 healer timeout 可能并发触达同一 item；用
        ``attribution_received_item_ids`` 去重，避免多计导致提前 finalize。

        Returns:
            ``(new_count, config, claimed)``。``claimed=True`` 表示本次首次计数。
        """
        return await self._increment_attribution_counter_once(
            rh._AttributionCounterBump(
                run_id=run_id,
                item_id=item_id,
                count_key="attribution_received_count",
                ids_key="attribution_received_item_ids",
            ),
        )

    async def is_attribution_received(self, run_id: str, item_id: str) -> bool:
        """该 item 的归因 received 是否已计数（早到回调 / 终败去重）。"""
        run = await rh.BenchmarkRun.filter(run_id=run_id).first()
        if run is None:
            return False
        config = dict(run.config_json) if isinstance(run.config_json, dict) else {}
        received_ids = config.get("attribution_received_item_ids") or []
        return str(item_id) in {str(x) for x in received_ids}

    async def is_external_judge_item_terminal(
        self,
        run_id: str,
        item_id: str,
    ) -> bool:
        """judge 回调是否已把 fused item 推入终态（success/failed）。

        healer 用来区分「正常 completed（回调已落地）」与「丢回调」。
        """
        item = await self.get_run_item(run_id, item_id)
        return item is not None and item.status in {"success", "failed"}

    async def cas_advance_external_judge_item(
        self,
        *,
        result: ExternalJudgeItemResult,
    ) -> bool:
        """单事务条件推进：item 首次终态 → 进度列 +1 → judge_received +1。

        锁序：``benchmark_runs`` FOR UPDATE → ``benchmark_run_items`` FOR UPDATE，
        避免并发回调双计 / 进度与 finalize 计数分叉。

        Returns:
            True 表示本调用首次将 item 推入终态并递增计数；False 表示已终态（幂等）。
        """
        run_id = result.run_id
        item_id = result.item_id
        failed = result.failed
        source_trace_id = result.source_trace_id
        trace_id = result.trace_id
        trace_url = result.trace_url
        item_input = result.item_input
        item_output = result.item_output
        evaluator_ref = result.evaluator_ref

        status = "failed" if failed else "success"
        bucket = rh._build_judge_bucket(result)

        async with rh.in_transaction() as conn:
            run = (
                await rh.BenchmarkRun.filter(run_id=run_id)
                .select_for_update()
                .using_db(conn)
                .first()
            )
            if run is None:
                raise RuntimeError(f"run not found: {run_id}")

            item = (
                await rh.BenchmarkRunItem.filter(
                    run_id=run_id,
                    item_id=item_id,
                    criterion_id=result.criterion_id,
                )
                .select_for_update()
                .using_db(conn)
                .first()
            )
            if item is not None and item.status in {"success", "failed"}:
                return False

            err = (result.error_message or "")[:2000] or None
            if item is None:
                await rh.BenchmarkRunItem.create(
                    using_db=conn,
                    run_id=run_id,
                    item_id=item_id,
                    criterion_id=result.criterion_id,
                    evaluator_ref=evaluator_ref,
                    status=status,
                    score=result.score,
                    reason=bucket.get("reason"),
                    sub_scores=bucket.get("sub_scores"),
                    judge_json=bucket,
                    source_trace_id=source_trace_id,
                    trace_id=trace_id or None,
                    trace_url=trace_url or None,
                    input=item_input,
                    output=item_output,
                    error_message=err if failed else None,
                )
            else:
                item.evaluator_ref = evaluator_ref
                item.status = status
                item.score = result.score
                item.reason = bucket.get("reason")
                item.sub_scores = bucket.get("sub_scores")
                item.judge_json = bucket
                if source_trace_id:
                    item.source_trace_id = source_trace_id
                if trace_id:
                    item.trace_id = trace_id
                if trace_url:
                    item.trace_url = trace_url
                if item_input is not None:
                    item.input = item_input
                if item_output is not None:
                    item.output = item_output
                item.error_message = err if failed else None
                await item.save(
                    using_db=conn,
                    update_fields=[
                        "evaluator_ref",
                        "status",
                        "score",
                        "reason",
                        "sub_scores",
                        "judge_json",
                        "source_trace_id",
                        "trace_id",
                        "trace_url",
                        "input",
                        "output",
                        "error_message",
                    ],
                )

            config = dict(run.config_json) if isinstance(run.config_json, dict) else {}
            received = int(config.get("judge_received_count", 0) or 0)
            config["judge_received_count"] = received + 1
            run.config_json = rh._stamp_bucket(config)
            if failed:
                run.failed_items = int(run.failed_items or 0) + 1
            else:
                run.evaluated_items = int(run.evaluated_items or 0) + 1
            await run.save(
                using_db=conn,
                update_fields=[
                    "config_json",
                    "evaluated_items",
                    "failed_items",
                ],
            )
            return True

    async def get_run_item(
        self,
        run_id: str,
        item_id: str,
    ) -> rh.BenchmarkRunItem | None:
        """获取单个 benchmark_run_items 的 fused 行（本地归因重建 eval_request 用）。"""
        return await rh.BenchmarkRunItem.filter(
            run_id=run_id,
            item_id=item_id,
            criterion_id=FUSED_CRITERION_ID,
        ).first()

    async def resolve_item_id_by_job_id(
        self,
        run_id: str,
        job_id: str,
        *,
        kind: Literal["judge", "attribution"] = "judge",
    ) -> str | None:
        """回调解析：job_id → item_id（按回调类型决定查询优先级）。

        Judge 回调：job_id = judge job_id → 优先查 source_trace_id
        Attribution 回调：job_id = attribution task_id → 优先查
        attribution_job_id（避免 params.job_id 原样回传导致命中错误行）。

        旧实现不区分回调类型，始终先查 source_trace_id；但 attribution
        回调的 ``payload.job_id`` 是 ``params.job_id`` 原样回传 = judge
        source_trace_id，导致 step-1 永远命中、step-2（查
        attribution_job_id）成为死代码。若 Aone 回传了错误的
        ``params.job_id``（批量提交覆盖等），归因结果会写到错误 item 的行下。
        """
        if kind == "attribution":
            # 1. 优先查 attribution_job_id（归因 task_id）
            item = await rh.BenchmarkRunItem.filter(
                run_id=run_id,
                attribution_job_id=job_id,
            ).first()
            if item:
                return item.item_id
            # 2. 回退查 source_trace_id（早期回调竞态：PG 尚未写入
            #    attribution_job_id，但 source_trace_id 在 judge 阶段已写入）
            item = await rh.BenchmarkRunItem.filter(
                run_id=run_id,
                source_trace_id=job_id,
            ).first()
            return item.item_id if item else None
        # judge 回调：job_id = judge job_id → 优先查 source_trace_id
        item = await rh.BenchmarkRunItem.filter(
            run_id=run_id,
            source_trace_id=job_id,
        ).first()
        if item:
            return item.item_id
        # 回退查 attribution_job_id
        item = await rh.BenchmarkRunItem.filter(
            run_id=run_id,
            attribution_job_id=job_id,
        ).first()
        return item.item_id if item else None

    async def update_attribution_job_id(
        self,
        *,
        run_id: str,
        item_id: str,
        attribution_job_id: str,
    ) -> None:
        """归因提交后更新 attribution_job_id（回调匹配用）。"""
        await rh.BenchmarkRunItem.filter(
            run_id=run_id,
            item_id=item_id,
            criterion_id=FUSED_CRITERION_ID,
        ).update(attribution_job_id=attribution_job_id)

    async def update_notification_dispatch(
        self,
        *,
        run_id: str,
        item_id: str,
        dispatch: dict[str, Any],
    ) -> None:
        """归因回调推送 Aone 后更新 fused 行 notification_dispatch_json bucket。

        bucket 字段：notification_ref / notification_title / notification_dispatched_at /
        notification_skip_reason（validators.validate_notification_dispatch_json 已在模型层校验）。
        写失败仅记日志不抛出——推送结果落库不阻断回调主链路。
        """
        try:
            await rh.BenchmarkRunItem.filter(
                run_id=run_id,
                item_id=item_id,
                criterion_id=FUSED_CRITERION_ID,
            ).update(notification_dispatch_json=dispatch)
        except Exception:
            rh.logger.warning(
                "benchmark.update_aone_dispatch_failed run_id=%s item_id=%s",
                run_id,
                item_id,
                exc_info=True,
            )

    async def get_fused_score(self, run_id: str, item_id: str) -> float | None:
        """读取 fused 行评测分（Aone 派单分数门槛用）。

        行缺失 / score 为空 / 查询异常时返回 None —— 派单门槛对 None
        放行（交由 severity 规则判定），不因取分失败漏派。
        """
        try:
            row = await rh.BenchmarkRunItem.get_or_none(
                run_id=run_id,
                item_id=item_id,
                criterion_id=FUSED_CRITERION_ID,
            )
        except Exception:
            rh.logger.warning(
                "benchmark.get_fused_score_failed run_id=%s item_id=%s",
                run_id,
                item_id,
                exc_info=True,
            )
            return None
        if row is None or row.score is None:
            return None
        return float(row.score)

    async def get_fused_query(self, run_id: str, item_id: str) -> str:
        """读取 fused 行 ``input.query``（Benchmark ieval Aone 派单去重用）。

        行缺失 / input 非 dict / query 空 / 查询异常时返回空串 ——
        派单侧对空 query 跳过去重、照常建单。
        """
        try:
            row = await rh.BenchmarkRunItem.get_or_none(
                run_id=run_id,
                item_id=item_id,
                criterion_id=FUSED_CRITERION_ID,
            )
        except Exception:
            rh.logger.warning(
                "benchmark.get_fused_query_failed run_id=%s item_id=%s",
                run_id,
                item_id,
                exc_info=True,
            )
            return ""
        if row is None:
            return ""
        raw = row.input
        if not isinstance(raw, dict):
            return ""
        return str(raw.get("query") or "").strip()

    async def get_service_name(self, run_id: str) -> str | None:
        """读取 run 的 service_name（aone 下发策略按业务解析用）。

        行缺失 / 查询异常时返回 None —— 策略解析对 None 回退默认实现
        （rubric_v2：仅 POOR / FAIL 下发）。
        """
        try:
            run = await rh.BenchmarkRun.get_or_none(run_id=run_id)
        except Exception:
            rh.logger.warning(
                "benchmark.get_service_name_failed run_id=%s",
                run_id,
                exc_info=True,
            )
            return None
        return run.service_name if run else None

    async def get_trace_id(self, run_id: str, item_id: str) -> str | None:
        """读取 fused 行 trace_id（Aone 工单 Trace ID 用）。

        行缺失 / 列为空 / 查询异常时返回 None —— 派单侧对 None 回退
        judge job_id（source_trace_id），不因取数失败阻断派发。
        """
        try:
            row = await rh.BenchmarkRunItem.get_or_none(
                run_id=run_id,
                item_id=item_id,
                criterion_id=FUSED_CRITERION_ID,
            )
        except Exception:
            rh.logger.warning(
                "benchmark.get_trace_id_failed run_id=%s item_id=%s",
                run_id,
                item_id,
                exc_info=True,
            )
            return None
        return row.trace_id if row else None

    async def get_trace_url(self, run_id: str, item_id: str) -> str | None:
        """读取 fused 行 trace_url（Aone 工单 Trace 链接用）。

        行缺失 / 列为空 / 查询异常时返回 None —— 工单正文对空值
        显示 '-'，不因取数失败阻断派发。
        """
        try:
            row = await rh.BenchmarkRunItem.get_or_none(
                run_id=run_id,
                item_id=item_id,
                criterion_id=FUSED_CRITERION_ID,
            )
        except Exception:
            rh.logger.warning(
                "benchmark.get_trace_url_failed run_id=%s item_id=%s",
                run_id,
                item_id,
                exc_info=True,
            )
            return None
        return row.trace_url if row else None

    async def get_external_job_ids(self, run_id: str) -> dict[str, str]:
        """attribution 提交前读取 {item_id: source_trace_id}（§15.3）。

        source_trace_id 在 judge 提交后写入为 Aone job_id。
        """
        result: dict[str, str] = {}
        async for item in rh.BenchmarkRunItem.filter(
            run_id=run_id,
            criterion_id=FUSED_CRITERION_ID,
        ).exclude(source_trace_id__isnull=True):
            if item.source_trace_id:
                result[item.item_id] = item.source_trace_id
        return result


__all__ = ["BenchmarkRunExternalItem"]
