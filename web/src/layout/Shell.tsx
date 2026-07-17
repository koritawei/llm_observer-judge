import { NavLink, Outlet } from "react-router-dom";
import {
  getAnnotatorId,
  getApiKey,
  getTenantId,
  setAnnotatorId,
  setApiKey,
  setTenantId,
} from "../api/client";
import { useState } from "react";
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
  const [annotator, setAnn] = useState(getAnnotatorId());

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
