import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, getTenantId } from "../api/client";
import { formatMs, formatScore, formatTokens, formatTs, shortId } from "../lib/format";

type ScoreChip = {
  name: string;
  value?: number | null;
  label?: string | null;
  status?: string | null;
};

type TraceItem = {
  trace_id: string;
  tenant_id: string;
  span_count: number;
  status?: string;
  name?: string;
  kind?: string;
  final_score?: number;
  bad_case_detected?: boolean;
  updated_at?: string;
  duration_ms?: number | null;
  input_preview?: string;
  output_preview?: string;
  token_input?: number;
  token_output?: number;
  tags?: string[];
  scores?: ScoreChip[];
};

export function TraceList() {
  const [items, setItems] = useState<TraceItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const load = useCallback(() => {
    const tenant = getTenantId();
    setLoading(true);
    setError(null);
    api
      .get<{ items: TraceItem[]; count: number }>(
        `/dashboard/api/v1/traces?tenant_id=${encodeURIComponent(tenant)}&limit=100`,
      )
      .then((r) => setItems(r.items))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div>
      <div className="toolbar">
        <div className="toolbar-left">
          <h1 className="page-title">链路</h1>
          <span className="count-badge">{items.length} 条</span>
        </div>
        <button type="button" onClick={load} disabled={loading}>
          {loading ? "加载中…" : "刷新"}
        </button>
      </div>
      <p className="page-desc" style={{ marginBottom: "1rem" }}>
        观测列表 · Span 聚合 + 评判分数
      </p>
      {error && <p className="muted">{error}</p>}
      <section className="panel" style={{ padding: 0, overflow: "auto" }}>
        <table className="trace-table">
          <thead>
            <tr>
              <th>时间</th>
              <th>名称</th>
              <th>输入</th>
              <th>输出</th>
              <th>分数</th>
              <th>耗时</th>
              <th>Token</th>
              <th>观测数</th>
            </tr>
          </thead>
          <tbody>
            {items.map((t) => (
              <tr
                key={t.trace_id}
                onClick={() => navigate(`/traces/${t.trace_id}`)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") navigate(`/traces/${t.trace_id}`);
                }}
                tabIndex={0}
              >
                <td className="muted mono" style={{ whiteSpace: "nowrap" }}>
                  {formatTs(t.updated_at)}
                </td>
                <td>
                  <div>
                    <Link
                      to={`/traces/${t.trace_id}`}
                      onClick={(e) => e.stopPropagation()}
                    >
                      {t.name || "链路"}
                    </Link>
                  </div>
                  <div className="mono muted" style={{ fontSize: "0.72rem", marginTop: 2 }}>
                    {shortId(t.trace_id, 12)}
                    {t.status ? ` · ${t.status === "OK" ? "正常" : t.status === "ERROR" ? "错误" : t.status}` : ""}
                    {t.kind ? ` · ${t.kind === "LLM" ? "大模型" : t.kind}` : ""}
                  </div>
                  {!!t.tags?.length && (
                    <div style={{ marginTop: 4, display: "flex", gap: 4, flexWrap: "wrap" }}>
                      {t.tags.map((tag) => (
                        <span key={tag} className="pill pending">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </td>
                <td>
                  <div className="preview-cell">{t.input_preview || "—"}</div>
                </td>
                <td>
                  <div className="preview-cell">{t.output_preview || "—"}</div>
                </td>
                <td>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                    {t.final_score != null && (
                      <span className={`score-chip ${t.bad_case_detected ? "bad" : ""}`}>
                        总分 {formatScore(t.final_score)}
                      </span>
                    )}
                    {(t.scores || []).slice(0, 3).map((s) => (
                      <span
                        key={s.name}
                        className={`score-chip ${
                          typeof s.value === "number" && s.value < 0.5 ? "bad" : ""
                        }`}
                        title={s.label || undefined}
                      >
                        {s.name.replace(/^judge_/, "")}{" "}
                        {typeof s.value === "number" ? formatScore(s.value) : "—"}
                      </span>
                    ))}
                    {t.final_score == null && !(t.scores || []).length && (
                      <span className="muted">—</span>
                    )}
                  </div>
                </td>
                <td className="mono muted">{formatMs(t.duration_ms)}</td>
                <td className="mono muted">
                  {formatTokens(t.token_input, t.token_output)}
                </td>
                <td className="muted">{t.span_count}</td>
              </tr>
            ))}
            {!items.length && !loading && (
              <tr>
                <td colSpan={8} className="muted" style={{ padding: "2rem" }}>
                  暂无链路。可通过 SDK 或{" "}
                  <code className="mono">POST /api/v1/ingest</code> 写入后刷新。
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}
