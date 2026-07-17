import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, scopeQuery, wsUrl } from "../api/client";
import { asText, formatMs, formatScore, formatTs, shortId } from "../lib/format";
import {
  annotationStatusLabel,
  kindLabel,
  nodeLabel,
  nodeStatusClass,
  pillClassForStatus,
  skipReasonLabel,
  statusLabel,
  wsStateLabel,
} from "../lib/status";

type NodeEv = {
  node: string;
  status: string;
  duration_ms?: number | null;
  detail?: Record<string, unknown>;
  ts?: string;
};

type Span = Record<string, unknown> & {
  span_id?: string;
  name?: string;
  kind?: string;
  status?: string;
  prompt?: string;
  response?: string;
  duration_ms?: number;
  timestamp?: string;
  tags?: string[];
  parent_span_id?: string | null;
};

type ExecNode = {
  id: string;
  label?: string;
  kind?: string;
  path?: "main" | "side" | string;
  status: string;
  order?: number;
  mode?: "serial" | "parallel" | string;
  stage?: number;
  stage_size?: number;
  error?: string | null;
  skip_reason?: string | null;
  score?: number | null;
  is_bad_case?: boolean | null;
};

type ExecStage = {
  id: number;
  mode: "serial" | "parallel" | string;
  nodes: ExecNode[];
};

type Summary = {
  trace_id: string;
  tenant_id?: string;
  spans: Span[];
  eval: {
    final_score: number;
    bad_case_detected: boolean;
    eval_scores: Record<string, unknown>[];
    eval_execution_layers: string[][];
    execution_graph?: ExecNode[];
  } | null;
  nodes: NodeEv[];
  eval_execution_layers: string[][];
  execution_graph?: ExecNode[];
  execution_stages?: ExecStage[];
  annotations: Record<string, unknown>[];
  attributions: Record<string, unknown>[];
};

type ObsNode = {
  id: string;
  kind: "span" | "eval" | "pipeline";
  name: string;
  status?: string;
  duration_ms?: number | null;
  ts?: string;
  span?: Span;
  score?: Record<string, unknown>;
  node?: NodeEv;
  exec?: ExecNode;
  depth: number;
};

type TabId = "io" | "scores" | "attribution" | "annotation" | "metadata";

export function TraceDetail() {
  const { traceId = "" } = useParams();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [live, setLive] = useState<NodeEv[]>([]);
  const [wsState, setWsState] = useState("connecting");
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tab, setTab] = useState<TabId>("io");

  const reloadSummary = async () => {
    if (!traceId) return null;
    const s = await api.get<Summary>(
      `/dashboard/api/v1/traces/${traceId}/summary?${scopeQuery()}`,
    );
    setSummary(s);
    return s;
  };

  useEffect(() => {
    if (!traceId) return;
    void reloadSummary()
      .then((s) => {
        if (!s) return;
        const firstSpan = s.spans?.[0];
        if (firstSpan?.span_id) setSelectedId(`span:${firstSpan.span_id}`);
        else if (s.execution_graph?.[0]) {
          const g0 = s.execution_graph[0];
          setSelectedId(g0.kind === "eval" ? `eval:${g0.id}` : `pipe:${g0.id}`);
        } else if (s.eval?.eval_scores?.[0]) {
          setSelectedId(`eval:${String(s.eval.eval_scores[0].agent_id || 0)}`);
        }
      })
      .catch((e) => setError(String(e)));
  }, [traceId]);

  useEffect(() => {
    if (!traceId) return;
    const ws = new WebSocket(wsUrl(`/dashboard/ws/traces/${traceId}`));
    ws.onopen = () => setWsState("open");
    ws.onclose = () => setWsState("closed");
    ws.onerror = () => setWsState("error");
    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data as string);
        if (data.type === "node") {
          setLive((prev) => [
            ...prev,
            {
              node: data.node,
              status: data.status,
              duration_ms: data.duration_ms,
              detail: data.detail,
              ts: data.ts,
            },
          ]);
        }
      } catch {
        /* ignore */
      }
    };
    return () => ws.close();
  }, [traceId]);

  useEffect(() => {
    const last = live[live.length - 1];
    if (!last) return;
    const refreshNodes = new Set([
      "ScoreNode",
      "Attribution",
      "Annotation",
      "EvalRouteNode",
    ]);
    if (!refreshNodes.has(last.node) && last.status !== "bad_case") return;
    const t = window.setTimeout(() => {
      void reloadSummary().catch(() => undefined);
    }, 250);
    return () => window.clearTimeout(t);
  }, [live.length, traceId]);

  const nodes = useMemo(() => {
    const base = summary?.nodes || [];
    const map = new Map<string, NodeEv>();
    for (const n of base) map.set(`${n.node}:${n.ts || ""}`, n);
    for (const n of live) map.set(`${n.node}:${n.ts || Date.now()}`, n);
    return [...map.values()];
  }, [summary, live]);

  const execGraph: ExecNode[] =
    summary?.execution_graph || summary?.eval?.execution_graph || [];

  const execStages: ExecStage[] = useMemo(() => {
    if (summary?.execution_stages?.length) return summary.execution_stages;
    const map = new Map<number, ExecNode[]>();
    const keys: number[] = [];
    for (const n of execGraph) {
      const sid = n.stage ?? n.order ?? 0;
      if (!map.has(sid)) {
        map.set(sid, []);
        keys.push(sid);
      }
      map.get(sid)!.push(n);
    }
    return keys.map((id) => {
      const stageNodes = map.get(id) || [];
      const mode =
        stageNodes.length > 1 || stageNodes[0]?.mode === "parallel"
          ? "parallel"
          : "serial";
      return { id, mode, nodes: stageNodes };
    });
  }, [summary?.execution_stages, execGraph]);

  const tree = useMemo((): ObsNode[] => {
    if (!summary) return [];
    const out: ObsNode[] = [];
    const scoreByAgent = new Map<string, Record<string, unknown>>();
    for (const sc of summary.eval?.eval_scores || []) {
      const agent = String(sc.agent_id || "judge");
      scoreByAgent.set(agent, sc);
    }
    const latestPipe = new Map<string, NodeEv>();
    for (const n of nodes) latestPipe.set(n.node, n);

    for (const sp of summary.spans || []) {
      const sid = String(sp.span_id || "");
      out.push({
        id: `span:${sid}`,
        kind: "span",
        name: String(sp.name || "观测"),
        status: String(sp.status || "OK"),
        duration_ms: typeof sp.duration_ms === "number" ? sp.duration_ms : null,
        ts: typeof sp.timestamp === "string" ? sp.timestamp : undefined,
        span: sp,
        depth: sp.parent_span_id ? 1 : 0,
      });
    }

    if (execGraph.length) {
      for (const n of execGraph) {
        const isEval = n.kind === "eval";
        const pipeEv = latestPipe.get(n.id);
        out.push({
          id: isEval ? `eval:${n.id}` : `pipe:${n.id}`,
          kind: isEval ? "eval" : "pipeline",
          name: nodeLabel(n.id, n.label),
          status: n.status,
          duration_ms: pipeEv?.duration_ms ?? null,
          ts: pipeEv?.ts,
          score: isEval ? scoreByAgent.get(n.id) : undefined,
          node: pipeEv,
          exec: n,
          depth: 1,
        });
      }
    } else {
      for (const n of latestPipe.values()) {
        out.push({
          id: `pipe:${n.node}`,
          kind: "pipeline",
          name: nodeLabel(n.node),
          status: n.status,
          duration_ms: n.duration_ms,
          ts: n.ts,
          node: n,
          depth: 1,
        });
      }
      for (const [agent, sc] of scoreByAgent) {
        out.push({
          id: `eval:${agent}`,
          kind: "eval",
          name: nodeLabel(agent),
          status: String(sc.score_status || "scored"),
          score: sc,
          depth: 1,
        });
      }
    }
    return out;
  }, [summary, nodes, execGraph]);

  const selected = tree.find((n) => n.id === selectedId) || tree[0] || null;

  const selectExecNode = (n: ExecNode) => {
    const id = n.kind === "eval" ? `eval:${n.id}` : `pipe:${n.id}`;
    setSelectedId(id);
    if (n.kind === "eval") setTab("scores");
    else if (n.id === "Attribution") setTab("attribution");
    else if (n.id === "Annotation") setTab("annotation");
    else setTab("io");
  };

  const inputText =
    selected?.kind === "span"
      ? asText(selected.span?.prompt)
      : selected?.kind === "eval"
        ? asText(selected.score?.reasoning || selected.score)
        : asText(selected?.node?.detail ?? selected?.exec);

  const outputText =
    selected?.kind === "span"
      ? asText(selected.span?.response)
      : selected?.kind === "eval"
        ? `分数=${formatScore(Number(selected.score?.normalized_score ?? selected.score?.raw_score ?? selected.exec?.score))}\n标签=${selected.score?.label ?? "—"}\n状态=${statusLabel(selected.status)}\n评分状态=${selected.score?.score_status ?? "—"}`
        : asText(statusLabel(selected?.status));

  const abortIdx = execGraph.findIndex((n) => n.status === "failed");

  // 时间线与执行图完全同源：同序、同状态；仅补耗时/时间戳
  const timelineRows = useMemo(() => {
    const latest = new Map<string, NodeEv>();
    for (const n of nodes) latest.set(n.node, n);

    if (execGraph.length) {
      return execGraph.map((n) => {
        const ev = latest.get(n.id);
        return {
          id: n.id,
          node: nodeLabel(n.id, n.label),
          status: n.status,
          duration_ms: ev?.duration_ms ?? null,
          ts: ev?.ts,
        };
      });
    }
    return [...latest.values()].map((n) => ({
      id: n.node,
      node: nodeLabel(n.node),
      status: n.status,
      duration_ms: n.duration_ms,
      ts: n.ts,
    }));
  }, [nodes, execGraph]);

  return (
    <div>
      <div className="header-bar">
        <div className="title-row">
          <Link to="/traces" className="muted" style={{ fontSize: "0.85rem" }}>
            ← 链路列表
          </Link>
          <h1 className="page-title" style={{ fontFamily: "var(--mono)", fontSize: "1.05rem" }}>
            {shortId(traceId, 16)}
          </h1>
          <span className={`pill ${wsState === "open" ? "ok" : "pending"}`}>
            实时 {wsStateLabel(wsState)}
          </span>
        </div>
        {summary?.eval && (
          <>
            <span
              className={`score-chip ${summary.eval.bad_case_detected ? "bad" : ""}`}
            >
              总分 {formatScore(summary.eval.final_score)}
            </span>
            {summary.eval.bad_case_detected && (
              <span className="pill failed">坏样本</span>
            )}
          </>
        )}
        {summary && <span className="pill pending">{tree.length} 条观测</span>}
      </div>

      {error && <p className="muted">{error}</p>}
      {!summary && !error && <p className="muted">加载链路中…</p>}

      {summary && (
        <>
          {!!execStages.length && (
            <section className="panel exec-graph-panel">
              <div className="obs-tree-head" style={{ border: "none", padding: "0 0 0.75rem" }}>
                流水线执行图
                <span
                  className="muted"
                  style={{ marginLeft: 8, fontWeight: 400, textTransform: "none" }}
                >
                  串行 → · 并行 ∥ · 与时间线 / 观测列表同源状态
                </span>
              </div>
              <div className="exec-legend">
                <span className="exec-legend-item">
                  <span className="exec-legend-swatch serial" /> 串行
                </span>
                <span className="exec-legend-item">
                  <span className="exec-legend-swatch parallel" /> 并行
                </span>
              </div>
              <div className="exec-flow">
                {execStages.map((stage, i) => {
                  const isParallel = stage.mode === "parallel" && stage.nodes.length > 1;
                  return (
                    <div key={stage.id} className="exec-step">
                      {i > 0 && <div className="exec-arrow">→</div>}
                      <div
                        className={`exec-stage ${isParallel ? "parallel" : "serial"}`}
                        data-mode={isParallel ? "parallel" : "serial"}
                      >
                        <div className="exec-stage-badge">
                          {isParallel ? `∥ 并行 ×${stage.nodes.length}` : "→ 串行"}
                        </div>
                        <div className={`exec-stage-body ${isParallel ? "stack" : ""}`}>
                          {stage.nodes.map((n) => {
                            const isAttr = n.id === "Attribution";
                            const isSide =
                              !isAttr && (n.kind === "side" || n.path === "side");
                            return (
                            <button
                              key={`${n.id}-${n.order}`}
                              type="button"
                              className={`exec-node ${nodeStatusClass(n.status)} ${isSide ? "side" : ""} ${isAttr ? "main-path" : ""}`}
                              title={
                                n.error ||
                                skipReasonLabel(n.skip_reason) ||
                                (n.score != null ? `分数=${n.score}` : statusLabel(n.status))
                              }
                              onClick={() => selectExecNode(n)}
                              style={{ cursor: "pointer", textAlign: "left" }}
                            >
                              <div className="exec-node-label">
                                {nodeLabel(n.id, n.label)}
                              </div>
                              <div className="exec-node-status">
                                {statusLabel(n.status)}
                                {n.is_bad_case ? " · 坏" : ""}
                                {isAttr ? " · 主链路" : isSide ? " · 旁路" : ""}
                              </div>
                            </button>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
              {abortIdx >= 0 && (
                <p className="muted" style={{ margin: "0.75rem 0 0", fontSize: "0.8rem" }}>
                  在{" "}
                  <code className="mono">
                    {nodeLabel(
                      execGraph[abortIdx]?.id,
                      execGraph[abortIdx]?.label,
                    )}
                  </code>{" "}
                  硬错误后，后续节点已跳过（失败即停）。
                </p>
              )}
            </section>
          )}

          <div className="obs-layout">
            <aside className="obs-tree">
              <div className="obs-tree-head">观测列表</div>
              {tree.map((n) => (
                <button
                  key={n.id}
                  type="button"
                  className={`obs-item ${selected?.id === n.id ? "active" : ""}`}
                  onClick={() => {
                    setSelectedId(n.id);
                    setTab("io");
                  }}
                  style={{ paddingLeft: `${0.85 + n.depth * 0.85}rem` }}
                >
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div className="obs-name">
                      {n.kind === "span" ? "◆ " : n.kind === "eval" ? "★ " : "○ "}
                      {n.name}
                    </div>
                    <div className="obs-meta">
                      {n.exec?.id === "Attribution" || n.name === "归因"
                        ? "主链路"
                        : n.exec?.kind === "side" ||
                            n.exec?.path === "side" ||
                            n.name === "标注"
                          ? "旁路"
                          : kindLabel(n.kind)}{" "}
                      · {statusLabel(n.status)}
                      {n.duration_ms != null ? ` · ${formatMs(n.duration_ms)}` : ""}
                      {n.kind === "eval" &&
                      (n.score?.normalized_score != null || n.exec?.score != null)
                        ? ` · ${formatScore(Number(n.score?.normalized_score ?? n.exec?.score))}`
                        : ""}
                    </div>
                  </div>
                  {n.status && (
                    <span className={`pill ${pillClassForStatus(n.status)}`}>
                      {statusLabel(n.status)}
                    </span>
                  )}
                </button>
              ))}
              {!tree.length && (
                <p className="muted" style={{ padding: "1rem" }}>
                  暂无观测。
                </p>
              )}
            </aside>

            <section className="obs-detail">
              <div className="obs-detail-head">
                {selected ? (
                  <>
                    {selected.name}{" "}
                    <span className="muted" style={{ fontWeight: 400 }}>
                      {kindLabel(selected.kind)}
                    </span>
                  </>
                ) : (
                  "详情"
                )}
              </div>
              <div className="tabs">
                {(
                  [
                    ["io", "输入/输出"],
                    ["scores", "评分"],
                    ["attribution", "归因"],
                    ["annotation", "标注"],
                    ["metadata", "元数据"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    className={`tab ${tab === id ? "active" : ""}`}
                    onClick={() => setTab(id)}
                  >
                    {label}
                    {id === "scores" && summary.eval?.eval_scores
                      ? ` (${summary.eval.eval_scores.length})`
                      : ""}
                    {id === "attribution" ? ` (${summary.attributions.length})` : ""}
                    {id === "annotation" ? ` (${summary.annotations.length})` : ""}
                  </button>
                ))}
              </div>
              <div className="obs-detail-body">
                {tab === "io" && selected && (
                  <>
                    <div className="io-block">
                      <div className="io-head">
                        <span>输入</span>
                        <span className="muted">
                          {inputText ? `${inputText.length} 字符` : "空"}
                        </span>
                      </div>
                      <pre className="io-body">{inputText || "—"}</pre>
                    </div>
                    <div className="io-block">
                      <div className="io-head">
                        <span>输出</span>
                        <span className="muted">
                          {outputText ? `${outputText.length} 字符` : "空"}
                        </span>
                      </div>
                      <pre className="io-body">{outputText || "—"}</pre>
                    </div>
                  </>
                )}

                {tab === "scores" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    {summary.eval ? (
                      <div className="panel" style={{ boxShadow: "none" }}>
                        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                          <span
                            className={`score-chip ${summary.eval.bad_case_detected ? "bad" : ""}`}
                          >
                            总分 {formatScore(summary.eval.final_score)}
                          </span>
                          {summary.eval.bad_case_detected && (
                            <span className="pill failed">坏样本</span>
                          )}
                        </div>
                      </div>
                    ) : (
                      <p className="muted">暂无评分。</p>
                    )}
                    {(summary.eval?.eval_scores || []).map((s, i) => {
                      const ui =
                        execGraph.find((n) => n.id === s.agent_id)?.status ||
                        String(s.score_status || "—");
                      return (
                        <div key={i} className="io-block">
                          <div className="io-head">
                            <span>{nodeLabel(String(s.agent_id || "judge"))}</span>
                            <span className={`pill ${pillClassForStatus(ui)}`}>
                              {statusLabel(ui)}
                            </span>
                            <span className="score-chip">
                              {formatScore(Number(s.normalized_score ?? s.raw_score))}
                            </span>
                          </div>
                          <pre className="io-body">
                            {`标签: ${s.label ?? "—"}\n评分状态: ${s.score_status ?? "—"}\n展示状态: ${statusLabel(ui)}\n理由:\n${s.reasoning ?? "—"}`}
                          </pre>
                        </div>
                      );
                    })}
                  </div>
                )}

                {tab === "attribution" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    {!summary.attributions.length && (
                      <p className="muted">暂无归因结果。</p>
                    )}
                    {summary.attributions.map((a, i) => (
                      <div key={i} className="io-block">
                        <div className="io-head">
                          <span>{String(a.root_cause || a.attribution_id || "归因")}</span>
                          <span className="pill ok">
                            置信度 {formatScore(Number(a.confidence))}
                          </span>
                        </div>
                        <pre className="io-body">{asText(a)}</pre>
                      </div>
                    ))}
                  </div>
                )}

                {tab === "annotation" && (
                  <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    {!summary.annotations.length && (
                      <p className="muted">暂无标注任务。</p>
                    )}
                    {summary.annotations.map((a, i) => (
                      <div key={i} className="io-block">
                        <div className="io-head">
                          <span>
                            {nodeLabel(String(a.eval_agent_id || a.annotation_id))}
                          </span>
                          <span className="pill pending">
                            {annotationStatusLabel(String(a.status))}
                          </span>
                        </div>
                        <pre className="io-body">{asText(a)}</pre>
                      </div>
                    ))}
                  </div>
                )}

                {tab === "metadata" && selected && (
                  <dl className="meta-grid">
                    <dt>链路 ID</dt>
                    <dd className="mono">{summary.trace_id}</dd>
                    <dt>租户</dt>
                    <dd>{summary.tenant_id || "—"}</dd>
                    <dt>观测</dt>
                    <dd>{selected.name}</dd>
                    <dt>类型</dt>
                    <dd>{kindLabel(selected.kind)}</dd>
                    <dt>状态</dt>
                    <dd>{statusLabel(selected.status)}</dd>
                    <dt>时间</dt>
                    <dd className="mono">{formatTs(selected.ts)}</dd>
                    <dt>耗时</dt>
                    <dd>{formatMs(selected.duration_ms)}</dd>
                    {selected.exec?.skip_reason && (
                      <>
                        <dt>跳过原因</dt>
                        <dd>{skipReasonLabel(selected.exec.skip_reason)}</dd>
                      </>
                    )}
                    {selected.span && (
                      <>
                        <dt>Span ID</dt>
                        <dd className="mono">{String(selected.span.span_id)}</dd>
                        <dt>种类</dt>
                        <dd>{kindLabel(String(selected.span.kind || ""))}</dd>
                        <dt>标签</dt>
                        <dd>
                          {Array.isArray(selected.span.tags)
                            ? selected.span.tags.join(", ") || "—"
                            : "—"}
                        </dd>
                      </>
                    )}
                  </dl>
                )}
              </div>
            </section>
          </div>

          <section className="panel" style={{ marginTop: "1rem", padding: 0, overflow: "auto" }}>
            <div className="obs-tree-head">
              流水线时间线
              <span
                className="muted"
                style={{ marginLeft: 8, fontWeight: 400, textTransform: "none" }}
              >
                与执行图同序同状态
              </span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>节点</th>
                  <th>状态</th>
                  <th>耗时</th>
                  <th>时间</th>
                </tr>
              </thead>
              <tbody>
                {timelineRows.map((n, i) => (
                  <tr key={`${n.id}-${i}`}>
                    <td>{n.node}</td>
                    <td>
                      <span className={`pill ${pillClassForStatus(n.status)}`}>
                        {statusLabel(n.status)}
                      </span>
                    </td>
                    <td className="mono muted">{formatMs(n.duration_ms)}</td>
                    <td className="muted mono">{formatTs(n.ts)}</td>
                  </tr>
                ))}
                {!timelineRows.length && (
                  <tr>
                    <td colSpan={4} className="muted">
                      等待流水线事件…
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  );
}
