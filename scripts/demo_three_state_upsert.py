"""SQLite 模拟三态 upsert 操作演示脚本。

演示 TraceRecordService 的三态语义：
  1. 首次插入（upsert）
  2. 局部更新（update_attribution）
  3. 重新插入（upsert 同 key 再次调用）
  4. 再次局部更新（update_attribution + update_triage_level）

每一步后输出数据库完整状态。
"""

import asyncio
import uuid
from pathlib import Path

from benchmark.records.database import create_database_config
from benchmark.records.models.trace_record import TraceRecord
from benchmark.services import TraceRecordService
from benchmark.libs.models.database import register_database
from tortoise import Tortoise
from tortoise.backends.base.executor import EXECUTOR_CACHE


def _clear_observers() -> None:
    TraceRecord.clear_observer()


async def _dump_record(label: str) -> None:
    """从数据库重新查询记录并打印完整状态。"""
    record = await TraceRecord.filter(
        trace_id="t-demo",
        project_id="demo-project",
        agent_name="agent-a",
        evaluator_name="default",
    ).first()

    print(f"\n{'=' * 72}")
    print(f"  📊 {label}")
    print(f"{'=' * 72}")
    if record is None:
        print("  (记录不存在)")
        return

    # 标量列
    print("\n  ── 标量列 ──")
    print(f"  id (UUID):       {record.id}")
    print(f"  trace_id:        {record.trace_id}")
    print(f"  project_id:      {record.project_id}")
    print(f"  agent_name:      {record.agent_name}")
    print(f"  evaluator_name:  {record.evaluator_name}")
    print(f"  status:          {record.status}")
    print(f"  scene:           {record.scene}")
    print(f"  owner_group:     {record.owner_group}")
    print(f"  triage_level:    {record.triage_level}")
    print(f"  root_cause:      {record.root_cause}")
    print(f"  notification_job_status: {record.notification_job_status}")
    print(f"  created_at:      {record.created_at}")
    print(f"  updated_at:      {record.updated_at}")

    # JSON bucket 列
    print("\n  ── JSON Bucket 列 ──")
    buckets = {
        "judge_json": record.judge_json,
        "attribution_detail_json": record.attribution_detail_json,
        "notification_dispatch_json": record.notification_dispatch_json,
        "notification_job_json": record.notification_job_json,
        "label_studio_attribution_annotation_json": record.label_studio_attribution_annotation_json,
        "label_studio_badcase_judge_annotation_json": record.label_studio_badcase_judge_annotation_json,
        "trace_meta_json": record.trace_meta_json,
    }
    for name, value in buckets.items():
        if value is None:
            print(f"  {name}: None")
        else:
            print(f"  {name}:")
            for k, v in value.items():
                print(f"    {k}: {v}")

    print()


async def main() -> None:
    # ── 初始化 SQLite ──
    db_path = f"/tmp/demo_three_state_{uuid.uuid4().hex[:8]}.sqlite3"
    db_url = f"sqlite:///{db_path}"

    print(f"\n{'#' * 72}")
    print("  🔧 SQLite 模拟三态 Upsert 操作演示")
    print(f"  数据库: {db_path}")
    print(f"{'#' * 72}")

    _clear_observers()
    EXECUTOR_CACHE.clear()
    config = create_database_config(db_url=db_url, environment="test")
    await register_database(None, config)
    await Tortoise.generate_schemas(safe=True)

    repo = TraceRecordService()

    # ── 步骤 1：首次插入 ──
    print(f"\n{'─' * 72}")
    print("  ▶ 步骤 1：首次插入（upsert）")
    print("  传入: status='scored', judge_json={'score': '0.85', 'reason': 'good'},")
    print("        scene='voice', trace_meta_json={'langfuse_id': 'lf-123'}")
    print(f"{'─' * 72}")

    rec1 = await repo.upsert(
        trace_id="t-demo",
        project_id="demo-project",
        agent_name="agent-a",
        evaluator_name="default",
        status="scored",
        judge_json={"score": "0.85", "reason": "good"},
        scene="voice",
        trace_meta_json={"trace_id": "lf-123"},
    )
    print(f"\n  ✅ upsert 返回: id={rec1.id}, status={rec1.status}")
    await _dump_record("步骤 1 后数据库状态")
    step1_id = rec1.id
    step1_created = rec1.created_at
    step1_updated = rec1.updated_at

    # ── 步骤 2：局部更新（update_attribution）──
    print(f"\n{'─' * 72}")
    print("  ▶ 步骤 2：局部更新（update_attribution）")
    print("  传入: root_cause='模型幻觉', owner_group='algo-team', confidence=0.92")
    print("  预期: attribution_detail_json 被更新，其他域不变")
    print(f"{'─' * 72}")

    await asyncio.sleep(0.01)  # 确保 updated_at 有变化

    rec2 = await repo.update_attribution(
        trace_id="t-demo",
        project_id="demo-project",
        agent_name="agent-a",
        evaluator_name="default",
        root_cause="模型幻觉",
        owner_group="algo-team",
        confidence=0.92,
    )
    print(f"\n  ✅ update_attribution 返回: root_cause={rec2.root_cause}")
    await _dump_record("步骤 2 后数据库状态")
    step2_updated = rec2.updated_at

    # ── 步骤 3：重新插入（upsert 同 key 再次调用）──
    print(f"\n{'─' * 72}")
    print("  ▶ 步骤 3：重新插入（upsert 同 key 再次调用）")
    print("  传入: status='scored', judge_json={'score': '0.90', 'reason': 're-eval'}")
    print("  预期: judge_json 被覆盖，未传入的数据域被清空")
    print("        scene 重置为 non_voice，owner_group / root_cause 清空为 None")
    print(f"{'─' * 72}")

    await asyncio.sleep(0.01)  # 确保 updated_at 有变化

    rec3 = await repo.upsert(
        trace_id="t-demo",
        project_id="demo-project",
        agent_name="agent-a",
        evaluator_name="default",
        status="scored",
        judge_json={"score": "0.90", "reason": "re-eval"},
    )
    print(f"\n  ✅ upsert 返回: id={rec3.id}, status={rec3.status}")
    await _dump_record("步骤 3 后数据库状态")
    step3_updated = rec3.updated_at

    # ── 步骤 4：再次局部更新（update_attribution + update_triage_level）──
    print(f"\n{'─' * 72}")
    print("  ▶ 步骤 4：再次局部更新（update_attribution + update_triage_level）")
    print("  传入: root_cause='工具调用错误', owner_group='infra-team', confidence=0.78")
    print("        triage_level='SEVERE_BAD'")
    print("  预期: attribution_detail_json + triage_level 被写入，judge_json 保留步骤3的值")
    print(f"{'─' * 72}")

    await asyncio.sleep(0.01)  # 确保 updated_at 有变化

    rec4 = await repo.update_attribution(
        trace_id="t-demo",
        project_id="demo-project",
        agent_name="agent-a",
        evaluator_name="default",
        root_cause="工具调用错误",
        owner_group="infra-team",
        confidence=0.78,
    )
    print(f"\n  ✅ update_attribution 返回: root_cause={rec4.root_cause}")

    await asyncio.sleep(0.01)

    from benchmark.records.enums import TriageLevel

    rec4b = await repo.update_triage_level(
        trace_id="t-demo",
        project_id="demo-project",
        agent_name="agent-a",
        evaluator_name="default",
        triage_level=TriageLevel.SEVERE_BAD,
    )
    print(f"  ✅ update_triage_level 返回: triage_level={rec4b.triage_level}")
    await _dump_record("步骤 4 后数据库状态")
    step4_updated = rec4b.updated_at

    # ── 三列变化对比 ──
    print(f"\n{'#' * 72}")
    print("  📋 UUID / created_at / updated_at 三列变化对比")
    print(f"{'#' * 72}")
    print(f"\n  {'列名':<20} {'步骤1':<38} {'步骤2':<38} {'步骤3':<38} {'步骤4':<38}")
    print(f"  {'─' * 160}")
    print(
        f"  {'id (UUID)':<20} {str(step1_id):<38} {str(rec2.id):<38} {str(rec3.id):<38} {str(rec4b.id):<38}"
    )
    print(
        f"  {'created_at':<20} {str(step1_created):<38} {str(rec2.created_at):<38} {str(rec3.created_at):<38} {str(rec4b.created_at):<38}"
    )
    print(
        f"  {'updated_at':<20} {str(step1_updated):<38} {str(step2_updated):<38} {str(step3_updated):<38} {str(step4_updated):<38}"
    )
    print("\n  分析:")
    print(
        f"  - UUID:       {'不变 ✅' if step1_id == rec2.id == rec3.id == rec4b.id else '变了 ❌'}"
        "（UPDATE 操作，不触发 default=uuid.uuid4）"
    )
    print(
        f"  - created_at: {'不变 ✅' if step1_created == rec2.created_at == rec3.created_at == rec4b.created_at else '变了 ❌'}"
        "（auto_now_add 仅 INSERT 生效）"
    )
    print(
        f"  - updated_at: {'每次刷新 ✅' if step1_updated < step2_updated < step3_updated < step4_updated else '未刷新 ❌'}"
        "（auto_now 每次 save 生效）"
    )

    # 清理
    await Tortoise.close_connections()
    try:
        Path(db_path).unlink()
    except FileNotFoundError:
        pass

    print(f"\n{'#' * 72}")
    print("  🎉 演示完成")
    print(f"{'#' * 72}\n")


if __name__ == "__main__":
    asyncio.run(main())
