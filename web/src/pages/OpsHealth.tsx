import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";

type Check = { name: string; ok: boolean; detail?: unknown };

export function OpsHealth() {
  const [checks, setChecks] = useState<Check[]>([]);
  const [stackz, setStackz] = useState<Record<string, unknown> | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setErr(null);
    const paths: [string, string][] = [
      ["网关 /healthz", "/healthz"],
      ["网关 /readyz", "/readyz"],
      ["标注服务", "/annotation/healthz"],
      ["基准评测", "/benchmark/healthz"],
      ["看板", "/dashboard/healthz"],
    ];
    const results: Check[] = [];
    for (const [name, path] of paths) {
      try {
        const body = await api.get(path);
        results.push({ name, ok: true, detail: body });
      } catch (e) {
        results.push({ name, ok: false, detail: String(e) });
      }
    }
    setChecks(results);
    try {
      setStackz(await api.get("/stackz"));
    } catch (e) {
      setStackz(null);
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 5000);
    return () => clearInterval(t);
  }, [refresh]);

  const depths = (stackz?.mq_depths || {}) as Record<string, number>;

  return (
    <div>
      <h1 className="page-title">运维健康</h1>
      <p className="page-desc">轮询网关 / 标注 / 基准评测 / 看板与队列深度。</p>
      {err && <p className="muted">stackz: {err}</p>}
      <div className="grid-2">
        <section className="panel">
          <h3>服务探测</h3>
          <table>
            <thead>
              <tr>
                <th>服务</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {checks.map((c) => (
                <tr key={c.name}>
                  <td>{c.name}</td>
                  <td>
                    <span className={`pill ${c.ok ? "ok" : "failed"}`}>
                      {c.ok ? "正常" : "异常"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button onClick={() => void refresh()} style={{ marginTop: "0.75rem" }}>
            刷新
          </button>
        </section>
        <section className="panel">
          <h3>队列深度</h3>
          {stackz ? (
            <>
              <p className="muted mono">
                消息队列={String(stackz.mq_driver)} · 持久化={String(stackz.persistence)} ·
                已启动={String(stackz.started)}
              </p>
              <table>
                <thead>
                  <tr>
                    <th>主题</th>
                    <th>深度</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(depths).map(([k, v]) => (
                    <tr key={k}>
                      <td className="mono">{k}</td>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <p className="muted">split 模式可能无 stackz（仅 monolith DevStack）。</p>
          )}
        </section>
      </div>
    </div>
  );
}
