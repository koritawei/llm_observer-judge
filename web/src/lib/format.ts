/** Display helpers aligned with Langfuse-style observability UI. */

export function shortId(id: string, n = 8): string {
  if (!id) return "—";
  return id.length <= n + 1 ? id : `${id.slice(0, n)}…`;
}

export function formatTs(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatScore(v?: number | null): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(3);
}

export function formatMs(ms?: number | null): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatTokens(input?: number, output?: number): string {
  const i = input || 0;
  const o = output || 0;
  if (!i && !o) return "—";
  return `${i} → ${o}`;
}

export function asText(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}
