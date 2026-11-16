"""§8 prompts. Placeholders are replaced by the executor."""

from benchmark.attribution.code.models import TaskPack, label_catalog

_SYSTEM = """\
你是 {{REPO}} 的代码归因助手。当前是 plan / 只读 模式。

硬约束：
1. 只读仓库 {{REPO_ROOT}}。
   禁止编辑、创建、删除任何文件；禁止 git commit / push / 开分支。
2. 根因只落到模块级（目录/包/策略名），禁止文件行号级补丁。
3. 改法只给方案级：改哪类模块、方向、风险、验收描述。禁止大段补丁。
4. label_id 必须从闭集选一个；证据不足选门禁，不要硬猜。
5. 上游症状的 layer/category 只是症状提示，不是标签约束。
6. 先判门禁，再判模块标签。
7. module_root_cause、evidence_chain、fix_plan 必须用中文。模块名、工具名、路径保持原文，不要翻译。
8. 最终只输出一个 JSON 对象，不要围栏外解释。

闭集 label_id：
{{LABEL_CATALOG}}

门禁：
- gate.eval_misjudge：评测误判
- gate.non_agent：非 agent 可修

输出 schema：
{
  "label_id": "tool.execution_fault",
  "tool_name": "plan_route",
  "module_root_cause": "一句话，钉到模块",
  "evidence_chain": "症状 → 模块/行为 → 为何定此标签",
  "fix_plan": "1. 模块与方向\\n2. 风险\\n3. 验收描述",
  "gate": null
}

tool_name：仅 tool.* 填具体工具名，否则 ""。
gate：未命中为 null；命中时 label_id 写成对应 gate，fix_plan 写「不要改 agent 代码」。
"""

_USER = """\
case_id: {{CASE_ID}}
query: {{QUERY}}
仓库根目录: {{REPO_ROOT}}

【上游症状（仅供参考，勿当标签约束）】
layer: {{SYMPTOM_LAYER}}
category: {{SYMPTOM_CATEGORY}}
root_cause: {{SYMPTOM_ROOT_CAUSE}}
root_cause_chain: {{SYMPTOM_CHAIN}}
improvement_suggestion: {{SYMPTOM_SUGGESTION}}

【trace 摘要】
{{TRACE_SUMMARY}}

任务：
1. 先判断是否 gate.eval_misjudge 或 gate.non_agent。
2. 若非门禁：定位模块，选一个 label_id；tool 层补 tool_name。
3. 给出模块级根因与方案级改法。
4. 只输出 JSON。
"""


def render_system(*, repo_root: str, repo: str) -> str:
    return (
        _SYSTEM.replace("{{REPO_ROOT}}", repo_root)
        .replace("{{REPO}}", repo)
        .replace("{{LABEL_CATALOG}}", label_catalog())
    )


def render_user(task: TaskPack, *, repo_root: str) -> str:
    values = {
        "{{CASE_ID}}": task.case_id,
        "{{QUERY}}": task.query,
        "{{REPO_ROOT}}": repo_root,
        "{{SYMPTOM_LAYER}}": task.symptom_layer,
        "{{SYMPTOM_CATEGORY}}": task.symptom_category,
        "{{SYMPTOM_ROOT_CAUSE}}": task.symptom_root_cause,
        "{{SYMPTOM_CHAIN}}": task.symptom_chain,
        "{{SYMPTOM_SUGGESTION}}": task.symptom_suggestion,
        "{{TRACE_SUMMARY}}": task.trace_summary,
    }
    text = _USER
    for key, value in values.items():
        text = text.replace(key, value)
    return text
