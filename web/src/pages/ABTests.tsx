import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, getTenantId } from "../api/client";
import { formatScore, shortId } from "../lib/format";

type PromptItem = {
  prompt_id: string;
  name: string;
  version_count: number;
  active_version_id?: string | null;
};

type Version = {
  version_id: string;
  prompt_id: string;
  version: number;
  status: string;
  content: string;
  changelog?: string | null;
};

type Experiment = {
  experiment_id: string;
  prompt_id: string;
  control_version_id: string;
  treatment_version_id: string;
  traffic_pct: number;
  status: string;
  metric_primary: string;
  started_at?: string;
  ended_at?: string | null;
  conclusion?: string | null;
  winner_version_id?: string | null;
};

type Arm = {
  prompt_version_id: string;
  count: number;
  mean_score: number | null;
  bad_case_rate: number | null;
  bad_case_count: number;
};

type Contrast = {
  experiment?: Experiment | null;
  control: Arm;
  treatment: Arm;
  delta_mean_score: number | null;
  sample_total: number;
  error?: string;
};

export function ABTests() {
  const tenant = getTenantId();
  const [prompts, setPrompts] = useState<PromptItem[]>([]);
  const [versions, setVersions] = useState<Version[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>("");
  const [contrast, setContrast] = useState<Contrast | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const [promptId, setPromptId] = useState("prompt_default_judge");
  const [controlId, setControlId] = useState("");
  const [treatmentId, setTreatmentId] = useState("");
  const [traffic, setTraffic] = useState("20");
  const [newContent, setNewContent] = useState("");

  const load = async () => {
    setError(null);
    const [p, e] = await Promise.all([
      api.get<{ items: PromptItem[] }>("/prompt/api/v1/prompts"),
      api.get<{ items: Experiment[] }>("/prompt/api/v1/experiments"),
    ]);
    setPrompts(p.items || []);
    setExperiments(e.items || []);
    if (!selectedExp && e.items?.length) {
      setSelectedExp(e.items[0].experiment_id);
    }
  };

  const loadVersions = async (pid: string) => {
    const v = await api.get<{ items: Version[] }>(
      `/prompt/api/v1/prompts/${encodeURIComponent(pid)}/versions`,
    );
    setVersions(v.items || []);
    const active = (v.items || []).find((x) => x.status === "ACTIVE");
    const draft = (v.items || []).find((x) => x.status === "DRAFT" || x.status === "EXPERIMENTAL");
    if (active) setControlId(active.version_id);
    if (draft) setTreatmentId(draft.version_id);
  };

  const loadContrast = async (expId: string) => {
    if (!expId) {
      setContrast(null);
      return;
    }
    const q = new URLSearchParams({
      experiment_id: expId,
      tenant_id: tenant,
    });
    const body = await api.get<Contrast>(`/dashboard/api/v1/stats/ab_contrast?${q}`);
    setContrast(body);
  };

  useEffect(() => {
    void load()
      .then(() => loadVersions(promptId))
      .catch((e) => setError(String(e)));
  }, [tenant]);

  useEffect(() => {
    void loadContrast(selectedExp).catch((e) => setError(String(e)));
  }, [selectedExp, tenant]);

  const chartData = useMemo(() => {
    if (!contrast || contrast.error) return [];
    return [
      {
        name: "Control (A)",
        mean: contrast.control.mean_score ?? 0,
        count: contrast.control.count,
        bad: (contrast.control.bad_case_rate ?? 0) * 100,
      },
      {
        name: "Treatment (B)",
        mean: contrast.treatment.mean_score ?? 0,
        count: contrast.treatment.count,
        bad: (contrast.treatment.bad_case_rate ?? 0) * 100,
      },
    ];
  }, [contrast]);

  const createTreatmentDraft = async () => {
    setMsg(null);
    try {
      const content =
        newContent.trim() ||
        versions.find((v) => v.version_id === controlId)?.content ||
        "You are a judge. Score the output.\nInput:\n{{ input_text }}\nOutput:\n{{ output_text }}\nJSON: {\"score\":0,\"label\":\"\",\"reasoning\":\"\"}";
      const ver = await api.post<Version>(
        `/prompt/api/v1/prompts/${encodeURIComponent(promptId)}/versions`,
        { content, changelog: "AB treatment draft" },
      );
      setTreatmentId(ver.version_id);
      setMsg(`已创建 DRAFT 版本 v${ver.version}`);
      await loadVersions(promptId);
      await load();
    } catch (err) {
      setError(String(err));
    }
  };

  const onCreateExperiment = async (e: FormEvent) => {
    e.preventDefault();
    setMsg(null);
    try {
      const exp = await api.post<Experiment>(
        `/prompt/api/v1/prompts/${encodeURIComponent(promptId)}/experiments`,
        {
          control_version_id: controlId,
          treatment_version_id: treatmentId,
          traffic_pct: Number(traffic) || 20,
          metric_primary: "mean_score",
        },
      );
      setSelectedExp(exp.experiment_id);
      setMsg(`实验已创建 · ${shortId(exp.experiment_id, 10)}`);
      await load();
    } catch (err) {
      setError(String(err));
    }
  };

  const onConclude = async (winner: "control" | "treatment") => {
    const exp = experiments.find((x) => x.experiment_id === selectedExp);
    if (!exp) return;
    setMsg(null);
    try {
      await api.post(
        `/prompt/api/v1/prompts/${encodeURIComponent(exp.prompt_id)}/experiments/${exp.experiment_id}/conclude`,
        { winner, conclusion: `UI conclude → ${winner}` },
      );
      setMsg(`已结束实验，胜者=${winner}`);
      await load();
      await loadContrast(selectedExp);
    } catch (err) {
      setError(String(err));
    }
  };

  const selected = experiments.find((x) => x.experiment_id === selectedExp);

  return (
    <div>
      <div className="toolbar">
        <div className="toolbar-left">
          <h1 className="page-title">A/B 实验</h1>
          <span className="pill ok">{experiments.length} 个实验</span>
        </div>
        <button type="button" onClick={() => void load().then(() => loadContrast(selectedExp))}>
          刷新
        </button>
      </div>
      <p className="page-desc">
        Prompt 版本对照实验（设计 §5.8）：按 trace_id hash 分流，对比 Control / Treatment 均分与 Bad Case 率。
      </p>
      {error && <p className="muted" style={{ color: "var(--danger)" }}>{error}</p>}
      {msg && <p className="muted">{msg}</p>}

      <div className="grid-2" style={{ marginBottom: "1rem" }}>
        <section className="panel" style={{ padding: 0 }}>
          <div className="obs-tree-head">实验列表</div>
          <div style={{ maxHeight: 360, overflow: "auto" }}>
            {!experiments.length && (
              <p className="muted" style={{ padding: "1rem" }}>
                暂无实验。右侧可创建 Treatment 版本并启动 A/B。
              </p>
            )}
            {experiments.map((exp) => (
              <button
                key={exp.experiment_id}
                type="button"
                onClick={() => setSelectedExp(exp.experiment_id)}
                style={{
                  display: "block",
                  width: "100%",
                  textAlign: "left",
                  border: "none",
                  borderBottom: "1px solid var(--line)",
                  borderRadius: 0,
                  boxShadow: "none",
                  background:
                    selectedExp === exp.experiment_id ? "var(--accent-soft)" : "transparent",
                  padding: "0.75rem 1rem",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                  <strong className="mono">{exp.prompt_id}</strong>
                  <span className={`pill ${exp.status === "RUNNING" ? "running" : "ok"}`}>
                    {exp.status}
                  </span>
                </div>
                <div className="muted" style={{ fontSize: "0.8rem", marginTop: 4 }}>
                  traffic {exp.traffic_pct}% · {shortId(exp.experiment_id, 12)}
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="panel">
          <h3 style={{ marginTop: 0 }}>创建 / 管理</h3>
          <form onSubmit={(e) => void onCreateExperiment(e)} style={{ display: "grid", gap: "0.55rem" }}>
            <label>
              Prompt
              <select
                value={promptId}
                onChange={(e) => {
                  setPromptId(e.target.value);
                  void loadVersions(e.target.value);
                }}
              >
                {(prompts.length ? prompts : [{ prompt_id: promptId, name: promptId } as PromptItem]).map(
                  (p) => (
                    <option key={p.prompt_id} value={p.prompt_id}>
                      {p.name || p.prompt_id}
                    </option>
                  ),
                )}
              </select>
            </label>
            <label>
              Control (A)
              <select value={controlId} onChange={(e) => setControlId(e.target.value)}>
                <option value="">—</option>
                {versions.map((v) => (
                  <option key={v.version_id} value={v.version_id}>
                    v{v.version} · {v.status} · {shortId(v.version_id, 8)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Treatment (B)
              <select value={treatmentId} onChange={(e) => setTreatmentId(e.target.value)}>
                <option value="">—</option>
                {versions.map((v) => (
                  <option key={v.version_id} value={v.version_id}>
                    v{v.version} · {v.status} · {shortId(v.version_id, 8)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Treatment 流量 %
              <input value={traffic} onChange={(e) => setTraffic(e.target.value)} />
            </label>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button className="primary" type="submit" disabled={!controlId || !treatmentId}>
                启动实验
              </button>
              <button type="button" onClick={() => void createTreatmentDraft()}>
                新建 Treatment DRAFT
              </button>
            </div>
          </form>
          <details style={{ marginTop: "0.75rem" }}>
            <summary className="muted">自定义 Treatment content（可选）</summary>
            <textarea
              rows={6}
              value={newContent}
              onChange={(e) => setNewContent(e.target.value)}
              placeholder="留空则复制 Control 内容"
              style={{ width: "100%", marginTop: 8, fontFamily: "var(--mono)", fontSize: "0.75rem" }}
            />
          </details>
        </section>
      </div>

      {selected && (
        <section className="panel" style={{ marginBottom: "1rem" }}>
          <div className="toolbar" style={{ marginBottom: "0.75rem" }}>
            <div className="toolbar-left">
              <h3 style={{ margin: 0 }}>对比结果</h3>
              <span className="muted mono">{shortId(selected.experiment_id, 12)}</span>
            </div>
            {selected.status === "RUNNING" && (
              <div style={{ display: "flex", gap: 8 }}>
                <button type="button" onClick={() => void onConclude("control")}>
                  Conclude → Control
                </button>
                <button type="button" className="primary" onClick={() => void onConclude("treatment")}>
                  Conclude → Treatment
                </button>
              </div>
            )}
          </div>

          <div className="grid-3" style={{ marginBottom: "1rem" }}>
            <div className="panel" style={{ boxShadow: "none", background: "var(--bg2)" }}>
              <div className="muted" style={{ fontSize: "0.8rem" }}>
                Δ mean score (B − A)
              </div>
              <div style={{ fontSize: "1.5rem", fontWeight: 650 }}>
                {contrast?.delta_mean_score != null
                  ? `${contrast.delta_mean_score >= 0 ? "+" : ""}${formatScore(contrast.delta_mean_score)}`
                  : "—"}
              </div>
            </div>
            <div className="panel" style={{ boxShadow: "none", background: "var(--bg2)" }}>
              <div className="muted" style={{ fontSize: "0.8rem" }}>
                Samples
              </div>
              <div style={{ fontSize: "1.5rem", fontWeight: 650 }}>
                {contrast?.sample_total ?? 0}
              </div>
            </div>
            <div className="panel" style={{ boxShadow: "none", background: "var(--bg2)" }}>
              <div className="muted" style={{ fontSize: "0.8rem" }}>
                Traffic / Status
              </div>
              <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>
                {selected.traffic_pct}% · {selected.status}
              </div>
            </div>
          </div>

          <div className="grid-2">
            <div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                  <XAxis dataKey="name" stroke="var(--muted)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--muted)" domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="mean" name="Mean score" fill="var(--accent)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
                  <XAxis dataKey="name" stroke="var(--muted)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--muted)" allowDecimals={false} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" name="N" fill="#175cd3" />
                  <Bar dataKey="bad" name="Bad case %" fill="#b42318" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid-2" style={{ marginTop: "1rem" }}>
            <div className="io-block">
              <div className="io-head">
                <span>Control (A)</span>
                <span className="score-chip">{formatScore(contrast?.control.mean_score)}</span>
              </div>
              <pre className="io-body" style={{ maxHeight: 120 }}>
                {`version: ${shortId(selected.control_version_id, 16)}\nn=${contrast?.control.count ?? 0}\nbad_case_rate=${formatScore(contrast?.control.bad_case_rate)}`}
              </pre>
            </div>
            <div className="io-block">
              <div className="io-head">
                <span>Treatment (B)</span>
                <span className="score-chip">{formatScore(contrast?.treatment.mean_score)}</span>
              </div>
              <pre className="io-body" style={{ maxHeight: 120 }}>
                {`version: ${shortId(selected.treatment_version_id, 16)}\nn=${contrast?.treatment.count ?? 0}\nbad_case_rate=${formatScore(contrast?.treatment.bad_case_rate)}`}
              </pre>
            </div>
          </div>
          {selected.conclusion && (
            <p className="muted" style={{ marginTop: "0.75rem" }}>
              Conclusion: {selected.conclusion}
              {selected.winner_version_id
                ? ` · winner ${shortId(selected.winner_version_id, 12)}`
                : ""}
            </p>
          )}
        </section>
      )}
    </div>
  );
}
