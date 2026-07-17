import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { api, getAnnotatorId, scopeQuery } from "../api/client";
import { asText, formatMs, formatScore, formatTs, shortId } from "../lib/format";

type Ann = {
  annotation_id: string;
  trace_id: string;
  tenant_id: string;
  eval_agent_id: string;
  trigger_reason?: string;
  status: string;
  auto_score?: number | null;
  auto_label?: string | null;
  conflict_detected?: boolean;
  annotator_id?: string | null;
  human_score?: number | null;
  human_label?: string | null;
  human_root_cause?: string | null;
  score_delta?: number | null;
  label_correct?: boolean | null;
  evidence_sufficient?: boolean | null;
  fix_feasible?: boolean | null;
  created_at?: string | null;
  completed_at?: string | null;
  expires_at?: string | null;
};

type EvalScore = {
  agent_id?: string;
  score_status?: string;
  raw_score?: number | null;
  normalized_score?: number | null;
  is_bad_case?: boolean | null;
  label?: string | null;
  reasoning?: string | null;
  threshold?: number | null;
  weight?: number | null;
  conflict_detected?: boolean | null;
  error?: string | null;
};

type Attribution = {
  attribution_id?: string;
  eval_agent_id?: string;
  attribution_agent?: string;
  root_cause?: string;
  confidence?: number;
  evidence?: Record<string, unknown>[];
  tools_called?: string[];
  rounds?: number;
  early_stop?: boolean;
  status?: string;
  error?: string | null;
  duration_ms?: number;
};

type TraceSummary = {
  trace_id: string;
  tenant_id?: string;
  spans?: Record<string, unknown>[];
  eval?: {
    final_score?: number;
    bad_case_detected?: boolean;
    eval_scores?: EvalScore[];
  } | null;
  attributions?: Attribution[];
  annotations?: Record<string, unknown>[];
};

export function AnnotateDetail() {
  const { id = "" } = useParams();
  const [row, setRow] = useState<Ann | null>(null);
  const [trace, setTrace] = useState<TraceSummary | null>(null);
  const [humanScore, setHumanScore] = useState("");
  const [humanLabel, setHumanLabel] = useState("");
  const [rootCause, setRootCause] = useState("");
  const [evidenceOk, setEvidenceOk] = useState<boolean | null>(null);
  const [fixOk, setFixOk] = useState<boolean | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const load = async () => {
    setLoadErr(null);
    const ann = await api.get<Ann>(`/annotation/api/v1/annotations/${id}`);
    setRow(ann);
    if (ann.human_score != null) setHumanScore(String(ann.human_score));
    if (ann.human_label) setHumanLabel(ann.human_label);
    if (ann.human_root_cause) setRootCause(ann.human_root_cause);
    if (ann.evidence_sufficient != null) setEvidenceOk(ann.evidence_sufficient);
    if (ann.fix_feasible != null) setFixOk(ann.fix_feasible);

    try {
      const extra: Record<string, string> = {};
      if (ann.tenant_id) extra.tenant_id = ann.tenant_id;
      if ((ann as { project_id?: string }).project_id) {
        extra.project_id = (ann as { project_id?: string }).project_id!;
      }
      const summary = await api.get<TraceSummary>(
        `/dashboard/api/v1/traces/${ann.trace_id}/summary?${scopeQuery(extra)}`,
      );
      setTrace(summary);
    } catch (e) {
      setTrace(null);
      setLoadErr(`Trace context: ${String(e)}`);
    }
  };

  useEffect(() => {
    void load().catch((e) => setLoadErr(String(e)));
  }, [id]);

  const focusedJudge = useMemo(() => {
    if (!row || !trace?.eval?.eval_scores) return null;
    return (
      trace.eval.eval_scores.find((s) => s.agent_id === row.eval_agent_id) || null
    );
  }, [row, trace]);

  const relatedAttrs = useMemo(() => {
    const all = trace?.attributions || [];
    if (!row) return all;
    const matched = all.filter((a) => a.eval_agent_id === row.eval_agent_id);
    return matched.length ? matched : all;
  }, [row, trace]);

  const primarySpan = trace?.spans?.[0];

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setMsg(null);
    try {
      if (row?.status === "PENDING") {
        await api.post(`/annotation/api/v1/annotations/${id}/claim`, {
          annotator_id: getAnnotatorId(),
        });
      }
      await api.post(`/annotation/api/v1/annotations/${id}/complete`, {
        annotator_id: getAnnotatorId(),
        human_score: humanScore ? Number(humanScore) : null,
        human_label: humanLabel || null,
        human_root_cause: rootCause || null,
        evidence_sufficient: evidenceOk,
        fix_feasible: fixOk,
      });
      setMsg("已提交");
      await load();
    } catch (err) {
      setMsg(String(err));
    }
  };

  if (!row) return <p className="muted">{loadErr || msg || "加载中…"}</p>;

  return (
    <div>
      <div className="toolbar">
        <div className="toolbar-left">
          <h1 className="page-title">标注详情</h1>
          <span className={`pill ${row.status === "COMPLETED" ? "ok" : "pending"}`}>
            {row.status}
          </span>
        </div>
        <Link to={`/traces/${row.trace_id}`} className="muted" style={{ fontSize: "0.85rem" }}>
          打开 Trace →
        </Link>
      </div>
      <p className="page-desc" style={{ marginBottom: "1rem" }}>
        <Link to="/annotate">← 队列</Link>
        {" · "}
        <span className="mono">{shortId(row.annotation_id, 12)}</span>
        {" · agent "}
        <strong>{row.eval_agent_id}</strong>
        {" · trigger "}
        {row.trigger_reason || "—"}
      </p>
      {loadErr && <p className="muted">{loadErr}</p>}

      {/* Context: I/O */}
      {primarySpan && (
        <div className="grid-2" style={{ marginBottom: "1rem" }}>
          <div className="io-block">
            <div className="io-head">
              <span>Input</span>
              <span className="muted">{String(primarySpan.name || "span")}</span>
            </div>
            <pre className="io-body">{asText(primarySpan.prompt) || "—"}</pre>
          </div>
          <div className="io-block">
            <div className="io-head">
              <span>Output</span>
              <span className="muted">{String(primarySpan.status || "")}</span>
            </div>
            <pre className="io-body">{asText(primarySpan.response) || "—"}</pre>
          </div>
        </div>
      )}

      <div className="grid-2" style={{ marginBottom: "1rem" }}>
        {/* Annotation meta + form */}
        <section className="panel">
          <h3 style={{ marginTop: 0 }}>标注任务</h3>
          <dl className="meta-grid">
            <dt>Trace</dt>
            <dd className="mono">
              <Link to={`/traces/${row.trace_id}`}>{shortId(row.trace_id, 16)}</Link>
            </dd>
            <dt>Tenant</dt>
            <dd>{row.tenant_id}</dd>
            <dt>Eval agent</dt>
            <dd className="mono">{row.eval_agent_id}</dd>
            <dt>Trigger</dt>
            <dd>{row.trigger_reason || "—"}</dd>
            <dt>Auto score</dt>
            <dd>
              <span className={`score-chip ${Number(row.auto_score) < 0.5 ? "bad" : ""}`}>
                {formatScore(row.auto_score)}
              </span>
            </dd>
            <dt>Auto label</dt>
            <dd>{row.auto_label || "—"}</dd>
            <dt>Conflict</dt>
            <dd>{row.conflict_detected ? "yes" : "no"}</dd>
            <dt>Created</dt>
            <dd className="mono">{formatTs(row.created_at)}</dd>
            <dt>Expires</dt>
            <dd className="mono">{formatTs(row.expires_at)}</dd>
            {row.completed_at && (
              <>
                <dt>Completed</dt>
                <dd className="mono">{formatTs(row.completed_at)}</dd>
                <dt>Human score</dt>
                <dd>{formatScore(row.human_score)}</dd>
                <dt>Human label</dt>
                <dd>{row.human_label || "—"}</dd>
                <dt>Root cause</dt>
                <dd>{row.human_root_cause || "—"}</dd>
                <dt>Score Δ</dt>
                <dd>{row.score_delta != null ? formatScore(row.score_delta) : "—"}</dd>
              </>
            )}
          </dl>

          {row.status !== "COMPLETED" && (
            <>
              <h3>提交人工标注</h3>
              <form onSubmit={(e) => void onSubmit(e)} style={{ display: "grid", gap: "0.65rem" }}>
                <label>
                  human_score（0–1 或与 auto 同尺度）
                  <input
                    value={humanScore}
                    onChange={(e) => setHumanScore(e.target.value)}
                    placeholder={row.auto_score != null ? String(row.auto_score) : ""}
                  />
                </label>
                <label>
                  human_label
                  <input
                    value={humanLabel}
                    onChange={(e) => setHumanLabel(e.target.value)}
                    placeholder={row.auto_label || ""}
                  />
                </label>
                <label>
                  human_root_cause
                  <input
                    value={rootCause}
                    onChange={(e) => setRootCause(e.target.value)}
                    placeholder="根因（可参考下方归因）"
                  />
                </label>
                <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={evidenceOk === true}
                      onChange={(e) => setEvidenceOk(e.target.checked ? true : null)}
                    />
                    evidence_sufficient
                  </label>
                  <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={fixOk === true}
                      onChange={(e) => setFixOk(e.target.checked ? true : null)}
                    />
                    fix_feasible
                  </label>
                </div>
                <button className="primary" type="submit">
                  {row.status === "PENDING" ? "认领并完成" : "Complete"}
                </button>
              </form>
            </>
          )}
          {msg && <p className="muted">{msg}</p>}
        </section>

        {/* Focused judge */}
        <section className="panel">
          <h3 style={{ marginTop: 0 }}>
            Judge 详情
            {focusedJudge ? (
              <span className="muted" style={{ fontWeight: 400, marginLeft: 8 }}>
                · {row.eval_agent_id}
              </span>
            ) : null}
          </h3>
          {trace?.eval && (
            <p style={{ marginTop: 0 }}>
              Trace final{" "}
              <span
                className={`score-chip ${trace.eval.bad_case_detected ? "bad" : ""}`}
              >
                {formatScore(trace.eval.final_score)}
              </span>
              {trace.eval.bad_case_detected && (
                <span className="pill failed" style={{ marginLeft: 6 }}>
                  bad case
                </span>
              )}
            </p>
          )}
          {focusedJudge ? (
            <div className="io-block">
              <div className="io-head">
                <span>{focusedJudge.agent_id}</span>
                <span
                  className={`score-chip ${focusedJudge.is_bad_case ? "bad" : ""}`}
                >
                  {formatScore(focusedJudge.normalized_score ?? focusedJudge.raw_score)}
                </span>
              </div>
              <div className="obs-detail-body" style={{ padding: "0.75rem" }}>
                <dl className="meta-grid">
                  <dt>Status</dt>
                  <dd>{focusedJudge.score_status || "—"}</dd>
                  <dt>Label</dt>
                  <dd>{focusedJudge.label || "—"}</dd>
                  <dt>Bad case</dt>
                  <dd>{focusedJudge.is_bad_case ? "yes" : "no"}</dd>
                  <dt>Threshold</dt>
                  <dd>{focusedJudge.threshold ?? "—"}</dd>
                  <dt>Weight</dt>
                  <dd>{focusedJudge.weight ?? "—"}</dd>
                  <dt>Conflict</dt>
                  <dd>{focusedJudge.conflict_detected ? "yes" : "no"}</dd>
                </dl>
                <div className="io-block" style={{ marginTop: "0.75rem" }}>
                  <div className="io-head">
                    <span>Reasoning</span>
                  </div>
                  <pre className="io-body">
                    {focusedJudge.reasoning || focusedJudge.error || "—"}
                  </pre>
                </div>
              </div>
            </div>
          ) : (
            <p className="muted">未找到对应 Judge 结果（可能尚未完成 Eval）。</p>
          )}
        </section>
      </div>

      {/* All judges */}
      <section className="panel" style={{ marginBottom: "1rem", padding: 0 }}>
        <div className="obs-tree-head">全部 Judge scores</div>
        <div style={{ padding: "0.75rem 1rem", display: "flex", flexDirection: "column", gap: "0.65rem" }}>
          {!(trace?.eval?.eval_scores || []).length && (
            <p className="muted">暂无 Judge 分数。</p>
          )}
          {(trace?.eval?.eval_scores || []).map((s, i) => {
            const focus = s.agent_id === row.eval_agent_id;
            return (
              <div
                key={`${s.agent_id}-${i}`}
                className="io-block"
                style={
                  focus
                    ? { outline: "2px solid color-mix(in srgb, var(--accent) 45%, transparent)" }
                    : undefined
                }
              >
                <div className="io-head">
                  <span>
                    {s.agent_id || "judge"}
                    {focus ? " · 本标注" : ""}
                  </span>
                  <span className={`score-chip ${s.is_bad_case ? "bad" : ""}`}>
                    {formatScore(s.normalized_score ?? s.raw_score)}
                    {s.label ? ` · ${s.label}` : ""}
                  </span>
                </div>
                <pre className="io-body" style={{ maxHeight: 160 }}>
                  {`status: ${s.score_status ?? "—"}\nbad_case: ${s.is_bad_case ? "yes" : "no"}\nreasoning:\n${s.reasoning || s.error || "—"}`}
                </pre>
              </div>
            );
          })}
        </div>
      </section>

      {/* Attribution */}
      <section className="panel" style={{ padding: 0 }}>
        <div className="obs-tree-head">
          归因信息
          <span className="muted" style={{ marginLeft: 8, fontWeight: 400, textTransform: "none" }}>
            {relatedAttrs.length
              ? `${relatedAttrs.length} 条`
              : "暂无（good case 会跳过归因）"}
          </span>
        </div>
        <div style={{ padding: "0.75rem 1rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          {!relatedAttrs.length && <p className="muted">暂无归因结果。</p>}
          {relatedAttrs.map((a, i) => (
            <div key={a.attribution_id || i} className="io-block">
              <div className="io-head">
                <span>{a.root_cause || a.attribution_id || "attribution"}</span>
                <span className="pill ok">
                  conf {formatScore(a.confidence)}
                </span>
              </div>
              <div style={{ padding: "0.75rem" }}>
                <dl className="meta-grid">
                  <dt>Eval agent</dt>
                  <dd className="mono">{a.eval_agent_id || "—"}</dd>
                  <dt>Attributor</dt>
                  <dd>{a.attribution_agent || "—"}</dd>
                  <dt>Status</dt>
                  <dd>
                    <span className={`pill ${a.status === "COMPLETED" ? "ok" : "pending"}`}>
                      {a.status || "—"}
                    </span>
                  </dd>
                  <dt>Rounds</dt>
                  <dd>
                    {a.rounds ?? "—"}
                    {a.early_stop ? " · early_stop" : ""}
                  </dd>
                  <dt>Duration</dt>
                  <dd>{formatMs(a.duration_ms)}</dd>
                  <dt>Tools</dt>
                  <dd className="mono" style={{ fontSize: "0.75rem" }}>
                    {(a.tools_called || []).join(", ") || "—"}
                  </dd>
                </dl>
                {!!(a.evidence || []).length && (
                  <div className="io-block" style={{ marginTop: "0.65rem" }}>
                    <div className="io-head">
                      <span>Evidence ({a.evidence!.length})</span>
                    </div>
                    <pre className="io-body">{asText(a.evidence)}</pre>
                  </div>
                )}
                {a.error && (
                  <p className="muted" style={{ color: "var(--danger)" }}>
                    error: {a.error}
                  </p>
                )}
                {a.root_cause && row.status !== "COMPLETED" && !rootCause && (
                  <button
                    type="button"
                    style={{ marginTop: "0.5rem" }}
                    onClick={() => setRootCause(String(a.root_cause))}
                  >
                    填入 human_root_cause
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
