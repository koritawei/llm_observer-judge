"""Input and output contract for the code-attribution module. Does not depend on host types."""

from dataclasses import dataclass


@dataclass
class TaskPack:
    case_id: str
    query: str
    symptom_layer: str
    symptom_category: str
    symptom_root_cause: str
    symptom_chain: str
    symptom_suggestion: str
    trace_summary: str
    repo: str = "amap-ai-agent"
    repo_root: str = ""


@dataclass
class ResultPack:
    status: str
    label_id: str = ""
    tool_name: str = ""
    module_root_cause: str = ""
    evidence_chain: str = ""
    fix_plan: str = ""
    commit_sha: str = ""
    gate: str | None = None
    error_message: str = ""


@dataclass
class RunTimings:
    total_sec: float = 0.0
    session_sec: float = 0.0
    poll_count: int = 0
    attempts: int = 0


@dataclass
class ModuleConfig:
    base_url: str = ""
    token: str = ""
    provider: str = "claude"
    project_path: str = ""
    poll_interval_sec: float = 2.0
    timeout_sec: float = 900.0
    live_enabled: bool = False


LABELS: tuple[tuple[str, str, str], ...] = (
    ("tool.capability_gap", "缺工具/可见性", "tools、discovery、visibility"),
    ("tool.execution_fault", "执行失败/超时/5xx", "工具实现、execution_policy"),
    ("tool.downstream_data", "工具成功但数据空", "数据装配；常在数据源"),
    ("tool.result_projection", "投影裁掉关键字段", "tool_output_projection"),
    ("plan.system_prompt", "系统提示/约束导致错路由", "prompts、prompt 节点"),
    ("plan.tool_orchestration", "选错工具/重试/预算", "orchestration_policy"),
    ("plan.context_assembly", "进模前上下文丢失", "context 节点、compaction"),
    ("plan.skill_routing", "Skill 未加载", "skill_load / retrieval"),
    ("plan.coordinator", "多/子 Agent 协调错", "coordinator、sub_agents"),
    ("gen.answer_synthesis", "终答合成/引用错", "reasoning/completion"),
    ("gen.output_protocol", "协议/结构化标签违规", "tool_cmd、finalize"),
    ("gen.render_pipeline", "渲染/流式管线错", "render、stream pipeline"),
    ("gate.eval_misjudge", "评测误判", "不改 agent"),
    ("gate.non_agent", "非 agent 可修", "不改 agent"),
)

LABEL_IDS: frozenset[str] = frozenset(item[0] for item in LABELS)
_BRIEFS: dict[str, str] = {label_id: brief for label_id, brief, _where in LABELS}


def label_brief(label_id: str) -> str:
    """Short display name for a label_id. Unknown ids return an empty string."""
    return _BRIEFS.get(label_id, "")


def label_catalog() -> str:
    """Short label table used by the prompt."""
    lines = [f"- {label_id}：{brief}。首查：{where}" for label_id, brief, where in LABELS]
    return "\n".join(lines)
