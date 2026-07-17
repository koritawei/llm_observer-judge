const API_KEY_STORAGE = "llmj_api_key";
const TENANT_STORAGE = "llmj_tenant";
const PROJECT_STORAGE = "llmj_project";
const ANNOTATOR_STORAGE = "llmj_annotator";

export function getApiKey(): string {
  return localStorage.getItem(API_KEY_STORAGE) || "dev-key";
}

export function setApiKey(key: string) {
  localStorage.setItem(API_KEY_STORAGE, key);
}

export function getTenantId(): string {
  return localStorage.getItem(TENANT_STORAGE) || "tenant-dev";
}

export function setTenantId(id: string) {
  localStorage.setItem(TENANT_STORAGE, id);
}

export function getProjectId(): string {
  return localStorage.getItem(PROJECT_STORAGE) || "default";
}

export function setProjectId(id: string) {
  localStorage.setItem(PROJECT_STORAGE, id);
}

export function getAnnotatorId(): string {
  return localStorage.getItem(ANNOTATOR_STORAGE) || "annotator-1";
}

export function setAnnotatorId(id: string) {
  localStorage.setItem(ANNOTATOR_STORAGE, id);
}

/** Build tenant+project query string for dashboard / annotation APIs. */
export function scopeQuery(extra: Record<string, string> = {}): string {
  const q = new URLSearchParams({
    tenant_id: getTenantId(),
    project_id: getProjectId(),
    ...extra,
  });
  return q.toString();
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("X-API-Key", getApiKey());
  headers.set("X-Project-Id", getProjectId());
  const res = await fetch(path, { ...init, headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined }),
};

export type ProjectItem = {
  project_id: string;
  tenant_id: string;
  name: string;
};

export async function listProjects(): Promise<ProjectItem[]> {
  const tenant = getTenantId();
  const body = await api.get<{ items: ProjectItem[] }>(
    `/projects/api/v1/projects?tenant_id=${encodeURIComponent(tenant)}`,
  );
  return body.items || [];
}

export async function createProject(name: string): Promise<ProjectItem> {
  const tenant = getTenantId();
  return api.post<ProjectItem>(
    `/projects/api/v1/projects?tenant_id=${encodeURIComponent(tenant)}`,
    { name },
  );
}

export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}
