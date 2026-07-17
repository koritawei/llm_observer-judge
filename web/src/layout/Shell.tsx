import { NavLink, Outlet } from "react-router-dom";
import {
  createProject,
  getAnnotatorId,
  getApiKey,
  getProjectId,
  getTenantId,
  listProjects,
  setAnnotatorId,
  setApiKey,
  setProjectId,
  setTenantId,
  type ProjectItem,
} from "../api/client";
import { useCallback, useEffect, useState } from "react";
import "./shell.css";

const linksUi = [
  { to: "/traces", label: "链路", hint: "观测链路", icon: "⧉", end: false },
  { to: "/annotate", label: "标注", hint: "人工标注", icon: "✎", end: false },
  { to: "/ab", label: "A/B 实验", hint: "Prompt 实验", icon: "⇄", end: false },
  { to: "/analytics", label: "统计", hint: "统计分析", icon: "▣", end: false },
  { to: "/benchmarks", label: "基准评测", hint: "评测集", icon: "▦", end: false },
  { to: "/ops", label: "运维健康", hint: "健康检查", icon: "●", end: false },
];

export function Shell() {
  const [apiKey, setKey] = useState(getApiKey());
  const [tenant, setTen] = useState(getTenantId());
  const [projectId, setProj] = useState(getProjectId());
  const [projects, setProjects] = useState<ProjectItem[]>([]);
  const [newProjectName, setNewProjectName] = useState("");
  const [annotator, setAnn] = useState(getAnnotatorId());
  const [projError, setProjError] = useState<string | null>(null);

  const refreshProjects = useCallback(async () => {
    setProjError(null);
    try {
      const items = await listProjects();
      setProjects(items);
      if (!items.find((p) => p.project_id === getProjectId()) && items.length) {
        setProjectId(items[0].project_id);
        setProj(items[0].project_id);
      }
    } catch (e) {
      setProjError(String(e));
    }
  }, []);

  useEffect(() => {
    void refreshProjects();
  }, [tenant, refreshProjects]);

  const onCreateProject = async () => {
    const name = newProjectName.trim();
    if (!name) return;
    setProjError(null);
    try {
      const created = await createProject(name);
      setNewProjectName("");
      await refreshProjects();
      setProjectId(created.project_id);
      setProj(created.project_id);
    } catch (e) {
      setProjError(String(e));
    }
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">LJ</div>
          <div>
            <div className="brand-title">LLMJudge</div>
            <div className="brand-sub">可观测 · 评判</div>
          </div>
        </div>
        <nav>
          {linksUi.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) => (isActive ? "nav active" : "nav")}
              title={l.hint}
            >
              <span className="nav-icon">{l.icon}</span>
              <span>{l.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="session">
          <label>
            API Key
            <input
              value={apiKey}
              onChange={(e) => {
                setKey(e.target.value);
                setApiKey(e.target.value);
              }}
            />
          </label>
          <label>
            Tenant
            <input
              value={tenant}
              onChange={(e) => {
                setTen(e.target.value);
                setTenantId(e.target.value);
              }}
            />
          </label>
          <label>
            Project
            <select
              value={projectId}
              onChange={(e) => {
                setProj(e.target.value);
                setProjectId(e.target.value);
              }}
            >
              {projects.map((p) => (
                <option key={p.project_id} value={p.project_id}>
                  {p.name}
                </option>
              ))}
              {!projects.length && <option value="default">default</option>}
            </select>
          </label>
          <label>
            新建项目
            <span style={{ display: "flex", gap: 6 }}>
              <input
                value={newProjectName}
                placeholder="项目名称"
                onChange={(e) => setNewProjectName(e.target.value)}
              />
              <button type="button" onClick={() => void onCreateProject()}>
                +
              </button>
            </span>
          </label>
          {projError && (
            <div className="session-error" title={projError}>
              项目加载失败
            </div>
          )}
          <label>
            Annotator
            <input
              value={annotator}
              onChange={(e) => {
                setAnn(e.target.value);
                setAnnotatorId(e.target.value);
              }}
            />
          </label>
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
