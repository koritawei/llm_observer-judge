import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, getAnnotatorId, getTenantId } from "../api/client";

type Ann = {
  annotation_id: string;
  trace_id: string;
  tenant_id: string;
  status: string;
  eval_agent_id: string;
  auto_score?: number | null;
  trigger_reason?: string;
};

export function AnnotateQueue() {
  const [items, setItems] = useState<Ann[]>([]);
  const [backlog, setBacklog] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const tenant = getTenantId();

  const load = async () => {
    setError(null);
    try {
      const list = await api.get<{ items: Ann[] }>(
        `/annotation/api/v1/annotations?tenant_id=${encodeURIComponent(tenant)}&limit=100`,
      );
      setItems(list.items);
      setBacklog(
        await api.get(`/annotation/api/v1/tenants/${encodeURIComponent(tenant)}/annotation-backlog`),
      );
    } catch (e) {
      setError(String(e));
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const claim = async (id: string) => {
    await api.post(`/annotation/api/v1/annotations/${id}/claim`, {
      annotator_id: getAnnotatorId(),
    });
    await load();
  };

  return (
    <div>
      <h1 className="page-title">人工标注队列</h1>
      <p className="page-desc">认领 / 完成对接标注 API。</p>
      {error && <p className="muted">{error}</p>}
      {backlog && (
        <section className="panel" style={{ marginBottom: "1rem" }}>
          待处理={String(backlog.pending)} · 告警={String(backlog.alert)} · 阈值=
          {String(backlog.threshold)}
        </section>
      )}
      <section className="panel">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>链路</th>
              <th>评判 Agent</th>
              <th>状态</th>
              <th>自动分</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {items.map((a) => (
              <tr key={a.annotation_id}>
                <td className="mono">
                  <Link to={`/annotate/${a.annotation_id}`}>{a.annotation_id.slice(0, 8)}…</Link>
                </td>
                <td className="mono">
                  <Link to={`/traces/${a.trace_id}`}>{a.trace_id.slice(0, 12)}…</Link>
                </td>
                <td>{a.eval_agent_id}</td>
                <td>
                  <span className="pill pending">
                    {a.status === "PENDING"
                      ? "待处理"
                      : a.status === "CLAIMED"
                        ? "已认领"
                        : a.status === "COMPLETED"
                          ? "已完成"
                          : a.status}
                  </span>
                </td>
                <td>{a.auto_score ?? "—"}</td>
                <td>
                  {a.status === "PENDING" && (
                    <button onClick={() => void claim(a.annotation_id)}>认领</button>
                  )}
                </td>
              </tr>
            ))}
            {!items.length && (
              <tr>
                <td colSpan={6} className="muted">
                  暂无标注任务。
                </td>
              </tr>
            )}
          </tbody>
        </table>
        <button style={{ marginTop: "0.75rem" }} onClick={() => void load()}>
          刷新
        </button>
      </section>
    </div>
  );
}
