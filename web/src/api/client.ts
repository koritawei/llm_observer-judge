const API_KEY_STORAGE = "llmj_api_key";
const TENANT_STORAGE = "llmj_tenant";
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

export function getAnnotatorId(): string {
  return localStorage.getItem(ANNOTATOR_STORAGE) || "annotator-1";
}

export function setAnnotatorId(id: string) {
  localStorage.setItem(ANNOTATOR_STORAGE, id);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("X-API-Key", getApiKey());
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

export function wsUrl(path: string): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}
