import { useState, type FormEvent } from "react";
import { api, getTenantId } from "../api/client";

export function Benchmarks() {
  const [reportId, setReportId] = useState("");
  const [batchId, setBatchId] = useState("");
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);

  const triggerBatch = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      const body = await api.post("/benchmark/api/v1/batch-eval/trigger", {
        tenant_id: getTenantId(),
        name: "dashboard-batch",
        traces: [
          {
            trace_id: "0123456789abcdef0123456789abcdef",
            trace_summary: {
              status: "OK",
              response: "a sufficiently long answer for quality judge",
            },
          },
        ],
        eval_configs: [{ agent_id: "judge_quality", model: "default-judge" }],
      });
      setResult(body);
      const rid = (body as { report_id?: string }).report_id;
      if (rid) setReportId(rid);
    } catch (err) {
      setError(String(err));
    }
  };

  const loadReport = async () => {
    setError(null);
    try {
      setResult(await api.get(`/benchmark/api/v1/benchmarks/${reportId}`));
    } catch (err) {
      setError(String(err));
    }
  };

  const loadBatch = async () => {
    setError(null);
    try {
      setResult(await api.get(`/benchmark/api/v1/batch-eval/${batchId}`));
    } catch (err) {
      setError(String(err));
    }
  };

  return (
    <div>
      <h1 className="page-title">基准评测</h1>
      <p className="page-desc">手动批次评估与报告查询。</p>
      {error && <p className="muted">{error}</p>}
      <div className="grid-2">
        <section className="panel">
          <h3>触发 Batch Eval</h3>
          <form onSubmit={(e) => void triggerBatch(e)}>
            <button className="primary" type="submit">
              Trigger demo batch
            </button>
          </form>
        </section>
        <section className="panel">
          <h3>查询</h3>
          <div style={{ display: "grid", gap: "0.5rem" }}>
            <input
              placeholder="report_id"
              value={reportId}
              onChange={(e) => setReportId(e.target.value)}
            />
            <button onClick={() => void loadReport()}>Get report</button>
            <input
              placeholder="batch_job_id"
              value={batchId}
              onChange={(e) => setBatchId(e.target.value)}
            />
            <button onClick={() => void loadBatch()}>Get batch</button>
          </div>
        </section>
      </div>
      {result != null && (
        <section className="panel" style={{ marginTop: "1rem" }}>
          <pre className="mono" style={{ whiteSpace: "pre-wrap", maxHeight: 420, overflow: "auto" }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}
