/** 状态 / 节点中文展示（与执行图同源，仅改展示层）。 */

const STATUS_ZH: Record<string, string> = {
  completed: "已完成",
  ok: "正常",
  scored: "已评分",
  bad_case: "坏样本",
  model_error: "模型错误",
  failed: "失败",
  infra_error: "基础设施错误",
  error: "错误",
  running: "运行中",
  pending: "等待中",
  skipped: "已跳过",
  connecting: "连接中",
  open: "已连接",
  closed: "已断开",
};

const NODE_ZH: Record<string, string> = {
  SamplingNode: "采样",
  IndexNode: "索引",
  WriteNode: "写入",
  DispatchPublish: "调度发布",
  EvalRouteNode: "评估路由",
  ScoreNode: "分数聚合",
  Attribution: "归因",
  Annotation: "标注",
  judge_quality: "质量评判",
  judge_fast_vllm: "快速评判",
  judge_safety: "安全评判",
  judge_code_eval: "代码评判",
};

const KIND_ZH: Record<string, string> = {
  span: "观测",
  eval: "评判",
  pipeline: "流水线",
  side: "旁路",
  LLM: "大模型",
};

const ANN_STATUS_ZH: Record<string, string> = {
  PENDING: "待处理",
  CLAIMED: "已认领",
  COMPLETED: "已完成",
  EXPIRED: "已过期",
};

export function nodeStatusClass(status: string): string {
  const s = status.toLowerCase();
  if (s === "completed" || s === "ok" || s === "scored") return "ok";
  if (s === "bad_case" || s === "model_error" || s === "warn") return "warn";
  if (s === "failed" || s === "infra_error" || s === "error") return "failed";
  if (s === "running") return "running";
  if (s === "skipped") return "pending";
  return "pending";
}

export function pillClassForStatus(status: string): string {
  return nodeStatusClass(status);
}

export function statusLabel(status?: string | null): string {
  if (!status) return "—";
  const key = status.toLowerCase();
  return STATUS_ZH[key] || STATUS_ZH[status] || status;
}

export function nodeLabel(id?: string | null, fallback?: string | null): string {
  if (!id) return fallback || "—";
  if (NODE_ZH[id]) return NODE_ZH[id];
  if (id.startsWith("judge_")) return id.replace(/^judge_/, "评判·");
  return fallback || id;
}

export function kindLabel(kind?: string | null): string {
  if (!kind) return "—";
  return KIND_ZH[kind] || kind;
}

export function skipReasonLabel(reason?: string | null): string {
  if (!reason) return "—";
  if (reason === "good_case") return "好样本跳过";
  if (reason === "not_triggered") return "未触发";
  if (reason.startsWith("aborted_after:")) {
    const who = reason.slice("aborted_after:".length);
    return `因 ${nodeLabel(who)} 硬错误后跳过`;
  }
  return reason;
}

export function annotationStatusLabel(status?: string | null): string {
  if (!status) return "—";
  return ANN_STATUS_ZH[status] || status;
}

export function wsStateLabel(state: string): string {
  return STATUS_ZH[state] || state;
}
