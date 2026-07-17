import { useState, type FormEvent } from "react";
import { api, getProjectId, getTenantId } from "../api/client";

type BatchTriggerResponse = {
  report_id?: string | null;
  batch_job_id?: string | null;
  batch_job?: { batch_job_id?: string };
};

export function Benchmarks() {
  const [reportId, setReportId] = useState("");
  const [batchId, setBatchId] = useState("");
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const triggerBatch = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const body = await api.post<BatchTriggerResponse>(
        "/benchmark/api/v1/batch-eval/trigger",
        {
          tenant_id: getTenantId(),
          project_id: getProjectId(),
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
        },
      );
      setResult(body);
      const rid = body.report_id ?? "";
      const bid =
        body.batch_job_id ||
        body.batch_job?.batch_job_id ||
        "";
      if (rid) setReportId(rid);
      if (bid) setBatchId(bid);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const loadReport = async () => {
    setError(null);
    const id = reportId.trim();
    if (!id) {
      setError("请先填写 report_id（Trigger 成功后会自动填入）");
      return;
    }
    setBusy(true);
    try {
      setResult(await api.get(`/benchmark/api/v1/benchmarks/${encodeURIComponent(id)}`));
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const loadBatch = async () => {
    setError(null);
    const id = batchId.trim();
    if (!id) {
      setError("请先填写 batch_job_id（Trigger 成功后会自动填入）");
      return;
    }
    setBusy(true);
    try {
      setResult(
        await api.get(`/benchmark/api/v1/batch-eval/${encodeURIComponent(id)}`),
      );
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
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
            <button className="primary" type="submit" disabled={busy}>
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
            <button type="button" onClick={() => void loadReport()} disabled={busy || !reportId.trim()}>
              Get report
            </button>
            <input
              placeholder="batch_job_id"
              value={batchId}
              onChange={(e) => setBatchId(e.target.value)}
            />
            <button type="button" onClick={() => void loadBatch()} disabled={busy || !batchId.trim()}>
              Get batch
            </button>
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
