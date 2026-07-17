import { useEffect, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, scopeQuery } from "../api/client";

const COLORS = ["#0d9488", "#175cd3", "#b54708", "#b42318", "#667085", "#027a48"];

export function Analytics() {
  const [data, setData] = useState<Record<string, unknown>>({});
  const [error, setError] = useState<string | null>(null);
  const q = scopeQuery();

  useEffect(() => {
    const load = async () => {
      setError(null);
      try {
        const keys = [
          ["overview", `/dashboard/api/v1/stats/overview?${q}`],
          ["platform", `/dashboard/api/v1/stats/platform`],
          ["histogram", `/dashboard/api/v1/stats/score_histogram?${q}`],
          ["per_eval", `/dashboard/api/v1/stats/per_eval?${q}`],
          ["sampling", `/dashboard/api/v1/stats/sampling`],
          ["attribution", `/dashboard/api/v1/stats/attribution?${q}`],
          ["annotation_accuracy", `/dashboard/api/v1/stats/annotation_accuracy?${q}`],
          ["benchmark", `/dashboard/api/v1/stats/benchmark?${q}`],
          ["prompts", `/dashboard/api/v1/stats/prompts?${q}`],
        ] as const;
        const next: Record<string, unknown> = {};
        for (const [k, path] of keys) {
          next[k] = await api.get(path);
        }
        setData(next);
      } catch (e) {
        setError(String(e));
      }
    };
    void load();
  }, [q]);

  const overview = (data.overview || {}) as Record<string, number>;
  const hist = ((data.histogram as { buckets?: { lo: number; hi: number; count: number }[] })
    ?.buckets || []).map((b) => ({
    name: `${b.lo.toFixed(1)}-${b.hi.toFixed(1)}`,
    count: b.count,
  }));
  const perEval = ((data.per_eval as { agents?: { agent_id: string; mean_score: number; count: number }[] })
    ?.agents || []).map((a) => ({
    name: a.agent_id,
    mean: Number(a.mean_score.toFixed(3)),
    count: a.count,
  }));
  const attr = ((data.attribution as { categories?: { name: string; count: number }[] })
    ?.categories || []);
  const sampling = (data.sampling || {}) as Record<string, unknown>;
  const annAcc = (data.annotation_accuracy || {}) as Record<string, unknown>;
  const prompts = ((data.prompts as { versions?: { prompt_version_id: string; usage: number }[] })
    ?.versions || []);
  const bench = (data.benchmark || {}) as { count?: number; reports?: unknown[] };
  const platform = (data.platform || {}) as {
    mq_depths?: Record<string, number>;
    sampling?: Record<string, unknown>;
  };
  const mqBars = Object.entries(platform.mq_depths || {}).map(([name, count]) => ({
    name,
    count,
  }));

  return (
    <div>
      <h1 className="page-title">统计分析</h1>
      <p className="page-desc">设计 §10.2 九面板（进程内聚合，可换 CH）。</p>
      {error && <p className="muted">{error}</p>}

      <div className="grid-3" style={{ marginBottom: "1rem" }}>
        <StatCard label="链路数" value={overview.trace_count ?? "—"} />
        <StatCard label="评估数" value={overview.eval_count ?? "—"} />
        <StatCard
          label="坏样本率"
          value={
            overview.bad_case_rate != null
              ? `${(overview.bad_case_rate * 100).toFixed(1)}%`
              : "—"
          }
        />
        <StatCard
          label="平均分"
          value={overview.mean_score != null ? Number(overview.mean_score).toFixed(3) : "—"}
        />
        <StatCard
          label="有效采样率"
          value={
            sampling.effective_sample_rate != null
              ? `${(Number(sampling.effective_sample_rate) * 100).toFixed(1)}%`
              : "—"
          }
        />
        <StatCard
          label="标注一致率"
          value={
            annAcc.score_agree_rate != null
              ? `${(Number(annAcc.score_agree_rate) * 100).toFixed(1)}%`
              : "暂无"
          }
        />
      </div>

      <div className="grid-2">
        <ChartPanel title="评分分布">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={hist}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3b46" />
              <XAxis dataKey="name" stroke="#8aa0ad" tick={{ fontSize: 11 }} />
              <YAxis stroke="#8aa0ad" allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#3dbea0" />
            </BarChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel title="Per-Eval 均分">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={perEval}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3b46" />
              <XAxis dataKey="name" stroke="#8aa0ad" tick={{ fontSize: 11 }} />
              <YAxis stroke="#8aa0ad" domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="mean" fill="#5eb0e0" />
            </BarChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel title="归因分布">
          {attr.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={attr} dataKey="count" nameKey="name" outerRadius={80} label>
                  {attr.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">暂无归因结果</p>
          )}
        </ChartPanel>

        <ChartPanel title="平台健康 · 队列">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={mqBars}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a3b46" />
              <XAxis dataKey="name" stroke="#8aa0ad" tick={{ fontSize: 10 }} />
              <YAxis stroke="#8aa0ad" allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#e0a45c" />
            </BarChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel title="采样统计">
          <pre className="mono" style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(sampling, null, 2)}
          </pre>
        </ChartPanel>

        <ChartPanel title="人工标注准确率">
          <p>compared={String(annAcc.compared ?? 0)}</p>
          <p>score_agree_rate={String(annAcc.score_agree_rate ?? "n/a")}</p>
        </ChartPanel>

        <ChartPanel title="基准评测报告">
          <p>数量={String(bench.count ?? 0)}</p>
          <pre className="mono" style={{ whiteSpace: "pre-wrap", maxHeight: 160, overflow: "auto" }}>
            {JSON.stringify(bench.reports?.slice(0, 3) || [], null, 2)}
          </pre>
        </ChartPanel>

        <ChartPanel title="Prompt 版本用量">
          {prompts.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={prompts.map((p) => ({ name: p.prompt_version_id, usage: p.usage }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3b46" />
                <XAxis dataKey="name" stroke="#8aa0ad" tick={{ fontSize: 11 }} />
                <YAxis stroke="#8aa0ad" allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="usage" fill="#8aa0ad" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="muted">暂无 prompt 版本统计</p>
          )}
        </ChartPanel>

        <ChartPanel title="评估 DAG 拓扑">
          <p className="muted">
            拓扑在链路详情页按执行层渲染；此处汇总评判 agent 数=
            {perEval.length}
          </p>
        </ChartPanel>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="panel">
      <div className="muted" style={{ fontSize: "0.8rem" }}>
        {label}
      </div>
      <div style={{ fontSize: "1.6rem", fontWeight: 650, marginTop: "0.25rem" }}>{value}</div>
    </div>
  );
}

function ChartPanel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="panel">
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      {children}
    </section>
  );
}
