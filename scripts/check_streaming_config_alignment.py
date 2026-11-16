"""streaming ↔ trace-daily-report ↔ eval config 关联一致性校验（CI 配套，§5.1 / §7.2）。

校验项（全部读 RemoteConfig 实时配置，任一失败 exit 1）：

1. **group_id 唯一性**（§5.2）：``null`` 至多一个；显式值互不相同；
   同一 group 下同一 topic 不得被两个 ``running`` task 订阅；
2. **日报关联完整性**（§7.2）：``daily_report.enabled=true`` 的 task 必须
   在 trace-daily-report 配置存在同名 item；
3. **rubric_class 单真相源**（§7.2）：task 与同名 item 同时配置
   ``rubric_class`` 时必须相等（报告口径以 item 为准）；
4. **rubric evaluator_ref ↔ eval config scoreName 对齐**（§5.1 不变量）：
   task / item 的 rubric 类 ``criteria[].evaluator_ref`` 必须被该 item
   ``project_id`` 的 eval config ``evaluatorName`` 集合覆盖。

用法::

    python scripts/check_streaming_config_alignment.py
"""

import sys

import _bootstrap  # noqa: F401  (sys.path / .env / 禁 MQ 副作用)
from benchmark.domain.eval_config import EvalConfigStore
from benchmark.domain.streaming_config import StreamingTaskConfig, fetch_streaming_config
from benchmark.domain.trace_report_config import (
    TraceReportItemConfig,
    fetch_trace_report_config,
)
from benchmark.runs.common.rubric_classes import (
    RubricClassResolutionError,
    resolve_rubric_class,
)
from benchmark.plugins.class_loader import load_plugin_class


def _check_group_uniqueness(tasks: list[StreamingTaskConfig], errors: list[str]) -> None:
    """校验项 1：group_id 唯一性 + 同 group 同 topic 互斥（§5.2）。"""
    null_group_tasks: list[str] = []
    explicit_groups: dict[str, str] = {}
    running_topic_groups: dict[tuple[str, str], str] = {}
    for task in tasks:
        name = task.name.strip() or "<empty>"
        if task.group_id is None:
            null_group_tasks.append(name)
        elif task.group_id in explicit_groups:
            errors.append(
                f"group_id {task.group_id!r} 被多个 task 使用: "
                f"{explicit_groups[task.group_id]!r} 与 {name!r}"
            )
        else:
            explicit_groups[task.group_id] = name
        if task.state != "running":
            continue
        group_key = task.group_id or "<JUDGE_KAFKA_GROUP_ID>"
        for topic in task.topics:
            key = (group_key, topic)
            if key in running_topic_groups:
                errors.append(
                    f"topic {topic!r} 在同 group {group_key!r} 下被两个 running task "
                    f"订阅: {running_topic_groups[key]!r} 与 {name!r}"
                )
            else:
                running_topic_groups[key] = name
    if len(null_group_tasks) > 1:
        errors.append(
            f"group_id=null 的 task 至多一个（= JUDGE_KAFKA_GROUP_ID 迁移任务）: {null_group_tasks}"
        )


def _rubric_evaluator_refs(rubric_alias: str) -> set[str]:
    """加载 rubric 类并提取 ``criteria[].evaluator_ref`` 集合（component_rubric 返回空）。"""
    spec = resolve_rubric_class(rubric_alias)
    if spec.class_ref is None:
        return set()
    cls = load_plugin_class(spec.class_ref, what="rubric_class")
    return {c.evaluator_ref for c in cls.criteria() if c.evaluator_ref}


def _check_task_report_alignment(
    tasks: list[StreamingTaskConfig],
    report_items: dict[str, TraceReportItemConfig],
    errors: list[str],
) -> None:
    """校验项 2/3：task ↔ trace-daily-report item 关联 + rubric_class 相等。"""
    for task in tasks:
        if not task.daily_report.enabled:
            continue
        item = report_items.get(task.name)
        if item is None:
            errors.append(
                f"task {task.name!r} daily_report.enabled=true 但 trace-daily-report "
                "配置不存在同名 item"
            )
            continue
        if task.rubric_class and item.rubric_class and task.rubric_class != item.rubric_class:
            errors.append(
                f"task {task.name!r} rubric_class={task.rubric_class!r} 与 "
                f"trace-daily-report item rubric_class={item.rubric_class!r} 不一致"
                "（报告口径以 item 为准）"
            )


def _check_rubric_eval_config_alignment(
    tasks: list[StreamingTaskConfig],
    report_items: dict[str, TraceReportItemConfig],
    errors: list[str],
) -> None:
    """校验项 4：rubric evaluator_ref ↔ eval config scoreName 对齐（§5.1 不变量）。"""
    store = EvalConfigStore()
    for task in tasks:
        rubric_alias = task.rubric_class
        item = report_items.get(task.name)
        if rubric_alias is None and item is not None:
            rubric_alias = item.rubric_class
        if rubric_alias is None:
            continue
        try:
            refs = _rubric_evaluator_refs(rubric_alias)
        except RubricClassResolutionError as exc:
            errors.append(f"task {task.name!r} rubric_class 解析失败: {exc}")
            continue
        except Exception as exc:  # rubric 类 import / criteria() 异常
            errors.append(f"task {task.name!r} rubric 类加载失败: {exc}")
            continue
        if not refs:
            continue  # component_rubric 无 evaluator_ref 契约，跳过
        project_id = item.project_id if item is not None else None
        if not project_id:
            errors.append(
                f"task {task.name!r} 的 rubric {rubric_alias!r} 声明 evaluator_ref={sorted(refs)} "
                "但同名 trace-daily-report item 缺失或未配置 project_id，无法核对 scoreName"
            )
            continue
        score_names = {cfg.evaluator_name for cfg in store.get_configs(project_id)}
        missing = refs - score_names
        if missing:
            errors.append(
                f"task {task.name!r} rubric {rubric_alias!r} 的 evaluator_ref "
                f"{sorted(missing)} 未被 project_id={project_id!r} 的 eval config "
                f"scoreName 覆盖（现有 {sorted(score_names)}）"
            )


def main() -> int:
    errors: list[str] = []
    streaming_cfg = fetch_streaming_config(use_cache=False)
    if streaming_cfg is None:
        print("streaming 声明式配置不存在（RemoteConfig 未发布），跳过校验")
        return 0

    report_cfg = fetch_trace_report_config(use_cache=False)
    report_items = {item.name: item for item in report_cfg.reports}

    _check_group_uniqueness(streaming_cfg.tasks, errors)
    _check_task_report_alignment(streaming_cfg.tasks, report_items, errors)
    _check_rubric_eval_config_alignment(streaming_cfg.tasks, report_items, errors)

    if errors:
        print("streaming 配置关联一致性校验失败:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(
        f"streaming 配置关联一致性校验通过: tasks={len(streaming_cfg.tasks)} "
        f"report_items={len(report_items)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
