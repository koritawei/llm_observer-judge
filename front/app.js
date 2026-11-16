const stage = document.querySelector("#stage");
const notice = document.querySelector("#notice");
const tabs = document.querySelector("#tabs");
const TERMINAL = new Set(["completed", "stopped", "failed"]);
const VIEWS = ["import", "workflow", "runs", "improve", "compare", "book"];
const SAMPLE = [
  {"item_id": "c1", "input": {"text": "杭州明天会下雨吗？"}, "output": {"text": "会"}, "expected": {"text": "会"}},
  {"item_id": "c2", "input": {"text": "儿童退烧药一次吃多少？"}, "output": {"text": "不知道"}, "expected": {"text": "按说明书"}},
];
const DEFAULT_POLICY = {
  enabled_authorize_edges: ["manual_authorize"],
  isolation_requirement: "standard",
  required_check_ids: [],
  security_violations: 0,
  primary_metric_id: "repair",
  evolution_capabilities: ["evolution-kernel"],
  gate_receipt_trust: {host_build_hash: "local-console"},
};
const DEFAULT_CHECKS = [{check_id: "repair", applicability: "required", verdict: "pass", value: 1}];
const DEFAULT_DIFF = "--- a/subject.py\n+++ b/subject.py\n@@\n-return \"\"\n+return answer\n";
const NODE_LABELS = {
  build_context: "构建上下文",
  resolve_profile: "解析方案",
  collect_evidence: "收集证据",
  run_checkers: "检查",
  plan_judges: "规划评分",
  run_judges: "执行评分",
  collect_metrics: "汇总指标",
  aggregate_results: "聚合结果",
  cluster_results: "结果分群",
  emit_envelope: "输出信封",
};
const STATUS_LABELS = {
  executed: "已执行",
  skipped: "跳过",
  failed: "失败",
  not_reached: "未到达",
};

const state = {
  view: "import",
  profiles: [],
  domains: {},
  runs: [],
  books: {},
  readingId: "",
  baselineId: "",
  candidateId: "",
  caseId: "",
  selected: new Set(),
  campaign: null,
  query: "",
  workflowBusiness: "default",
};

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "on" && value) {
      for (const [event, handler] of Object.entries(value)) node.addEventListener(event, handler);
    } else if (value != null && value !== false) node.setAttribute(key, value);
  }
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"content-type": "application/json", ...(options.headers || {})},
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.message || response.statusText);
  return body;
}

function showError(message) {
  notice.textContent = message || "";
  notice.classList.toggle("show", Boolean(message));
}

function pretty(value) {
  if (value == null || value === "") return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function field(label, control) {
  return el("label", {}, [label, control]);
}

function input(name, value, attrs = {}) {
  return el("input", {name, value: value ?? "", ...attrs});
}

function textarea(name, value, rows) {
  const node = el("textarea", {name, rows: String(rows)});
  node.value = value ?? "";
  return node;
}

function select(name, options, selected) {
  const node = el("select", {name});
  for (const option of options) {
    const item = el("option", {value: option.value}, [option.label]);
    if (option.value === selected) item.selected = true;
    node.append(item);
  }
  return node;
}

function runOptions(placeholder) {
  return [{value: "", label: placeholder}, ...state.runs.map((run) => ({
    value: run.run_id,
    label: `${run.benchmark_name || run.profile_id || "run"} · ${run.status} · ${run.run_id}`,
  }))];
}

function button(className, text, onclick) {
  return el("button", {type: "button", class: className, on: {click: onclick}}, [text]);
}

async function refreshRuns() {
  state.runs = (await api("/v1/runs")).runs;
}

async function loadBook(runId, force = false) {
  if (!runId) return null;
  if (force || !state.books[runId]) state.books[runId] = await api(`/v1/runs/${runId}/book`);
  return state.books[runId];
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function follow(runId) {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    const run = await api(`/v1/runs/${runId}`);
    showError("");
    notice.textContent = `${run.run_id} · ${run.status} · 分数 ${run.final_score ?? "—"}`;
    notice.classList.add("show");
    if (TERMINAL.has(run.status)) {
      delete state.books[runId];
      await refreshRuns();
      if (state.view === "runs") paintRuns();
      return run;
    }
    await sleep(1000);
  }
  return null;
}

function setTab() {
  for (const tab of tabs.querySelectorAll("button")) {
    tab.classList.toggle("on", tab.dataset.view === state.view);
  }
}

async function show(view) {
  state.view = VIEWS.includes(view) ? view : "import";
  history.replaceState(null, "", `#${state.view}`);
  setTab();
  showError("");
  stage.replaceChildren(el("p", {class: "lead"}, ["正在读取…"]));
  try {
    if (state.view === "import") paintImport();
    else if (state.view === "workflow") await paintWorkflow();
    else if (state.view === "runs") paintRuns();
    else if (state.view === "improve") paintImprove();
    else if (state.view === "compare") await paintCompare();
    else await paintBook();
  } catch (error) {
    showError(error.message);
  }
}

function paintImport() {
  const domains = [{value: "", label: "不使用评测域，直接按 Profile 评测"}, ...Object.keys(state.domains).map((name) => ({value: name, label: name}))];
  const profiles = state.profiles.map((name) => ({value: name, label: name}));
  const preferred = state.profiles.includes("benchmark_sync") ? "benchmark_sync" : (profiles[0] && profiles[0].value);
  const form = el("form", {class: "grid", id: "import-form"}, [
    field("名称", input("benchmark_name", "demo", {required: "true"})),
    field("评测域", select("eval_domain_id", domains, "answer_quality")),
    field("Profile", select("profile_id", profiles, preferred)),
    field("Business", input("business", "default")),
    field("Cases JSON", textarea("cases", JSON.stringify(SAMPLE, null, 2), 12)),
  ]);
  const file = el("input", {type: "file", accept: "application/json,.json"});
  file.addEventListener("change", async () => {
    const picked = file.files && file.files[0];
    if (!picked) return;
    form.elements.cases.value = await picked.text();
  });
  stage.replaceChildren(
    el("section", {class: "panel"}, [
      el("h2", {}, ["导入数据集"]),
      el("p", {class: "lead"}, ["粘贴或选择 JSON。数组会当作 cases；对象可以自带 cases、名称和评测域。"]),
      form,
      el("div", {class: "actions"}, [
        el("button", {type: "submit", class: "primary", form: "import-form"}, ["导入并评测"]),
        button("ghost", "填入示例", () => {
          form.elements.cases.value = JSON.stringify(SAMPLE, null, 2);
        }),
        file,
      ]),
    ]),
  );
  form.elements.business.addEventListener("change", async () => {
    const business = form.elements.business.value || "default";
    const listed = await api(`/v1/profiles?business=${encodeURIComponent(business)}`);
    state.profiles = listed.profiles || [];
    const profile = form.elements.profile_id;
    const current = profile.value;
    profile.replaceChildren();
    for (const name of state.profiles) profile.append(el("option", {value: name}, [name]));
    if (state.profiles.includes(current)) profile.value = current;
    else if (state.profiles.includes("benchmark_sync")) profile.value = "benchmark_sync";
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");
    try {
      const data = new FormData(form);
      const parsed = JSON.parse(String(data.get("cases") || "[]"));
      const cases = Array.isArray(parsed) ? parsed : parsed.cases;
      if (!Array.isArray(cases) || !cases.length) throw new Error("cases 需要是非空数组");
      const domain = data.get("eval_domain_id") || (!Array.isArray(parsed) && parsed.eval_domain_id) || "";
      const body = {
        benchmark_name: data.get("benchmark_name") || (!Array.isArray(parsed) && parsed.benchmark_name) || "demo",
        business: data.get("business") || "default",
        cases,
      };
      const created = domain
        ? await api("/v1/datasets", {method: "POST", body: JSON.stringify({...body, eval_domain_id: domain})})
        : await api("/v1/runs", {method: "POST", body: JSON.stringify({...body, profile_id: data.get("profile_id")})});
      state.readingId = created.run_id;
      await refreshRuns();
      await show("runs");
      follow(created.run_id);
    } catch (error) {
      showError(error.message);
    }
  });
}

async function paintWorkflow() {
  const business = state.workflowBusiness || "default";
  const catalog = await api(`/v1/plugins?business=${encodeURIComponent(business)}`);
  state.workflowBusiness = catalog.business;
  const businesses = (catalog.businesses || ["default"]).map((name) => ({value: name, label: name}));
  const form = el("form", {id: "workflow-form"}, [
    el("div", {class: "grid two"}, [
      field("名称", input("benchmark_name", "composed", {required: "true"})),
      field("业务", select("business", businesses, catalog.business)),
    ]),
    el("p", {class: "meta"}, [`可见组 ${(catalog.groups || []).join("、")}`]),
  ]);
  form.elements.business.addEventListener("change", () => {
    state.workflowBusiness = form.elements.business.value || "default";
    paintWorkflow();
  });
  const board = el("div", {class: "workflow-board"});
  const flows = catalog.flows || [{id: "benchmark", title: "评测", nodes: catalog.stages || []}];
  for (const flow of flows) {
    board.append(flow.id === "evolution" ? drawEvolutionFlow(flow) : drawBenchmarkFlow(flow, catalog));
  }
  form.append(board);
  const execution = form.querySelector("select[data-slot='execution']");
  if (execution) syncTrialPlugin(execution);
  form.append(field("Cases JSON", textarea("cases", JSON.stringify(SAMPLE, null, 2), 8)));
  form.append(el("div", {class: "actions"}, [
    el("button", {type: "submit", class: "primary"}, ["按此编排评测"]),
  ]));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");
    try {
      const data = readForm(form);
      const parsed = JSON.parse(data.cases || "[]");
      const cases = Array.isArray(parsed) ? parsed : parsed.cases;
      if (!Array.isArray(cases) || !cases.length) throw new Error("cases 需要是非空数组");
      const plugins = {};
      for (const picker of form.querySelectorAll("select[data-slot]")) plugins[picker.dataset.slot] = picker.value;
      const created = await api("/v1/workflows", {
        method: "POST",
        body: JSON.stringify({
          benchmark_name: data.benchmark_name || "composed",
          business: data.business || catalog.business,
          base_profile: catalog.base_profile,
          plugins,
          cases,
        }),
      });
      state.readingId = created.run_id;
      await refreshRuns();
      await show("runs");
      follow(created.run_id);
    } catch (error) {
      showError(error.message);
    }
  });
  stage.replaceChildren(el("section", {class: "panel"}, [
    el("h2", {}, ["编排评测与自进化"]),
    el("p", {class: "lead"}, ["左边是评测的全部节点，每个节点只挂当前业务可见的插件。右边是自进化的全部节点，试跑评测沿用左边选中的评分插件。"]),
    form,
  ]));
}

function drawBenchmarkFlow(flow, catalog) {
  const column = el("section", {class: "flow composer"}, [
    el("h3", {}, [flow.title || "评测"]),
  ]);
  (flow.nodes || []).forEach((stageNode, index) => {
    const card = el("div", {class: "flow-node"}, [
      el("strong", {}, [stageNode.title]),
      el("span", {class: "mono"}, [stageNode.id]),
    ]);
    const slots = stageNode.slots || [];
    if (!slots.length) card.append(el("p", {}, ["按当前选择冻结评测方案"]));
    for (const slot of slots) {
      const plugins = slot.plugins || [];
      const fallback = plugins[0] && plugins[0].name;
      const chosen = plugins.some((plugin) => plugin.name === (catalog.defaults || {})[slot.slot])
        ? catalog.defaults[slot.slot]
        : fallback;
      const picker = plugins.length
        ? select(slot.slot, plugins.map((plugin) => ({
          value: plugin.name,
          label: `${plugin.title} · ${plugin.group}`,
        })), chosen)
        : el("p", {}, ["这个节点还没有可挂的插件"]);
      if (plugins.length) picker.dataset.slot = slot.slot;
      const note = el("em", {}, [pluginSummary(plugins, chosen)]);
      if (plugins.length) {
        picker.addEventListener("change", () => {
          note.textContent = pluginSummary(plugins, picker.value);
          if (slot.slot === "execution") syncTrialPlugin(picker);
        });
      }
      card.append(el("label", {class: "slot"}, [slot.title, picker, note]));
    }
    column.append(card);
    if (index < flow.nodes.length - 1) column.append(el("div", {class: "flow-arrow"}, ["↓"]));
  });
  return column;
}

function drawEvolutionFlow(flow) {
  const column = el("section", {class: "flow composer"}, [
    el("h3", {}, [flow.title || "自进化"]),
  ]);
  const nodes = flow.nodes || [];
  nodes.forEach((node, index) => {
    const card = el("div", {class: `flow-node ${node.lane || ""}`}, [
      el("strong", {}, [node.title]),
      el("span", {class: "mono"}, [node.id]),
      el("em", {}, [laneLabel(node.lane)]),
    ]);
    if (node.uses_slot) {
      card.append(el("p", {class: "trial-plugin", "data-uses": node.uses_slot}, ["沿用评测里的评分插件"]));
    }
    if (node.plugin) {
      const mounted = node.mounted;
      card.append(el("p", {}, [
        mounted ? `${mounted.title} · ${mounted.group}` : `${node.plugin} 不在当前可见组`,
      ]));
    }
    if (node.slot) card.append(el("p", {}, [`使用评测里的${node.slot_title || node.slot}`]));
    if (node.when) card.append(el("p", {}, [`继续：${node.when}`]));
    if (node.stop) card.append(el("p", {}, [node.stop]));
    column.append(card);
    const nxt = nodes[index + 1];
    if (nxt && node.next === nxt.id) column.append(el("div", {class: "flow-arrow"}, ["↓"]));
    else if (nxt && node.lane === "entry" && nxt.lane === "auto") {
      column.append(el("p", {class: "flow-note"}, ["冻结后进入自动推进"]));
    } else if (nxt && nxt.lane === "manual" && node.lane !== "manual") {
      column.append(el("p", {class: "flow-note"}, ["以下为手动步骤"]));
    }
  });
  return column;
}

function syncTrialPlugin(picker) {
  const label = picker.options[picker.selectedIndex] ? picker.options[picker.selectedIndex].text : picker.value;
  const root = picker.form || document;
  for (const node of root.querySelectorAll("[data-uses='execution']")) {
    node.textContent = `试跑使用 ${label}`;
  }
}

function laneLabel(lane) {
  if (lane === "manual") return "手动";
  if (lane === "entry") return "入口";
  return "自动";
}

function pluginSummary(plugins, name) {
  const found = plugins.find((plugin) => plugin.name === name);
  return found && found.summary ? found.summary : "";
}

function paintRuns() {
  const table = el("table", {}, [
    el("thead", {}, [el("tr", {}, ["名称", "Run", "Profile", "状态", "样本", "完成/失败", "分数", "父运行", ""].map((name) => el("th", {}, [name])))]),
  ]);
  const body = el("tbody");
  for (const run of state.runs) {
    const actions = el("div", {class: "row-actions"}, [
      button("ghost", "阅读", () => openBook(run.run_id)),
      button("ghost", "基线", () => { state.baselineId = run.run_id; show("compare"); }),
      button("ghost", "改进后", () => { state.candidateId = run.run_id; show("compare"); }),
      button("ghost", "停止", async () => {
        await api(`/v1/runs/${run.run_id}/stop`, {method: "POST"});
        await refreshRuns();
        paintRuns();
      }),
    ]);
    body.append(el("tr", {}, [
      el("td", {}, [run.benchmark_name || ""]),
      el("td", {class: "mono"}, [run.run_id]),
      el("td", {}, [run.profile_id || ""]),
      el("td", {}, [run.status || ""]),
      el("td", {}, [String(run.total_samples ?? "")]),
      el("td", {}, [`${run.evaluated_items ?? 0}/${run.failed_items ?? 0}`]),
      el("td", {}, [run.final_score ?? ""]),
      el("td", {}, [run.parent_run_id || "—"]),
      el("td", {}, [actions]),
    ]));
  }
  table.append(body);
  const replay = el("form", {class: "grid two", id: "replay-form"}, [
    field("父运行", select("parent_run_id", runOptions("选择已结束的运行"), state.readingId)),
    field("Agent", input("agent_id", "agent-1", {required: "true"})),
    field("操作者", input("triggered_by", "local-user", {required: "true"})),
    field("Bearer", input("token", "local-token")),
    field("要回放的 case，用逗号分隔", input("case_ids", [...state.selected].join(","))),
  ]);
  stage.replaceChildren(
    el("section", {class: "panel"}, [
      el("h2", {}, ["评测运行"]),
      el("div", {class: "actions"}, [button("ghost", "刷新", async () => { await refreshRuns(); paintRuns(); })]),
      table,
    ]),
    el("section", {class: "panel"}, [
      el("h2", {}, ["回放"]),
      el("p", {class: "lead"}, ["从已结束的运行里挑 case，按同一套内核再跑一遍。案例页勾选的条目会带到这里。"]),
      replay,
      el("div", {class: "actions"}, [el("button", {class: "primary", type: "submit", form: "replay-form"}, ["回放"])]),
    ]),
  );
  replay.addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");
    const data = new FormData(replay);
    const parent = String(data.get("parent_run_id") || "");
    const caseIds = String(data.get("case_ids") || "").split(/[\s,]+/).filter(Boolean);
    if (!parent || !caseIds.length) {
      showError("请选择父运行，并填写至少一个 case");
      return;
    }
    try {
      const created = await api(`/v1/benchmarks/runs/${parent}/playground-replay`, {
        method: "POST",
        headers: {authorization: `Bearer ${data.get("token") || ""}`},
        body: JSON.stringify({
          case_ids: caseIds,
          agent_id: data.get("agent_id"),
          triggered_by: data.get("triggered_by"),
          replay_agent_flow: true,
        }),
      });
      state.readingId = created.run_id;
      await refreshRuns();
      paintRuns();
      follow(created.run_id);
    } catch (error) {
      showError(error.message);
    }
  });
}

function paintImprove() {
  const form = el("form", {class: "grid", id: "improve-form"}, [
    field("基线运行", select("run_id", runOptions("选择已完成的评测"), state.baselineId || state.readingId)),
    field("再次评测名称", input("again_name", "demo-again")),
    field("差异文本", textarea("diff", DEFAULT_DIFF, 6)),
    field("允许修改的路径前缀", input("prefixes", "subject.py")),
    field("留出集 case，逗号分隔，可空", input("heldout", "")),
    field("改进后的 output，按 item_id", textarea("outputs", "{\n  \"c1\": {\"text\": \"会\"}\n}", 8)),
    field("门禁策略", textarea("policy", JSON.stringify(DEFAULT_POLICY, null, 2), 8)),
    field("检查项（不参与裁决）", textarea("checks", JSON.stringify(DEFAULT_CHECKS, null, 2), 5)),
  ]);
  const result = el("pre", {id: "improve-result"});
  stage.replaceChildren(
    el("section", {class: "panel"}, [
      el("h2", {}, ["改进后再评测"]),
      el("p", {class: "lead"}, ["开发集的每条 case 都要提交新 output，缺了不会沿用旧结果。第一次推进只发起复评。闸门读这次运行的分数，不采用检查项。留出集答案不能用来确认。"]),
      form,
      el("div", {class: "actions"}, [
        button("primary", "发起改进", () => submitImprovement(form, result)),
        button("ghost", "用新输出再次评测", () => submitAgain(form)),
      ]),
      result,
    ]),
  );
}

async function submitImprovement(form, result) {
  showError("");
  try {
    const data = readForm(form);
    const frozen = await api("/v1/evolution/gate-policy", {method: "POST", body: JSON.stringify(JSON.parse(data.policy))});
    const admitted = await api("/v1/evolution/events", {
      method: "POST",
      body: JSON.stringify({run_id: data.run_id, report_revision: 1}),
    });
    if (!admitted.campaign_id) throw new Error(admitted.reason || admitted.status || "未能发起改进");
    const advanced = await api(`/v1/evolution/campaigns/${admitted.campaign_id}/advance`, {
      method: "POST",
      body: JSON.stringify({
        gate_policy: frozen,
        heldout_ids: data.heldout.split(/[\s,]+/).filter(Boolean),
        subject_hash: "subject-local",
        repairer_hash: "repairer-local",
        evaluator_hash: "evaluator-local",
        diff_text: data.diff,
        allowed_prefixes: data.prefixes.split(/[\s,]+/).filter(Boolean),
        proposed_outputs: JSON.parse(data.outputs || "{}"),
        checks: JSON.parse(data.checks || "[]"),
      }),
    });
    state.campaign = {campaign_id: admitted.campaign_id, status: advanced.status, advance: advanced};
    state.baselineId = data.run_id;
    if (advanced.benchmark_run_id) state.candidateId = advanced.benchmark_run_id;
    result.textContent = JSON.stringify(state.campaign, null, 2);
    if (advanced.benchmark_run_id) follow(advanced.benchmark_run_id);
  } catch (error) {
    showError(error.message);
  }
}

async function submitAgain(form) {
  showError("");
  try {
    const data = readForm(form);
    const book = await loadBook(data.run_id);
    const outputs = JSON.parse(data.outputs || "{}");
    const cases = book.cases.map((item) => {
      const next = {
        item_id: item.item_id,
        input: item.input,
        output: Object.prototype.hasOwnProperty.call(outputs, item.item_id) ? outputs[item.item_id] : item.output,
      };
      if (item.expected != null) next.expected = item.expected;
      return next;
    });
    const body = {
      benchmark_name: data.again_name || `${book.run.benchmark_name || "demo"}-again`,
      business: book.run.business || "default",
      cases,
    };
    const created = book.run.eval_domain_id
      ? await api("/v1/datasets", {method: "POST", body: JSON.stringify({...body, eval_domain_id: book.run.eval_domain_id})})
      : await api("/v1/runs", {method: "POST", body: JSON.stringify({...body, profile_id: book.run.profile_id})});
    state.baselineId = data.run_id;
    state.candidateId = created.run_id;
    state.readingId = created.run_id;
    await refreshRuns();
    await show("compare");
    follow(created.run_id).then(() => {
      if (state.view === "compare") show("compare");
    });
  } catch (error) {
    showError(error.message);
  }
}

function readForm(form) {
  const data = new FormData(form);
  return Object.fromEntries([...data.entries()].map(([key, value]) => [key, String(value)]));
}

async function paintCompare() {
  await refreshRuns();
  const form = el("form", {class: "grid two"}, [
    field("基线", select("baseline", runOptions("选择基线"), state.baselineId)),
    field("改进后", select("candidate", runOptions("选择改进后的运行"), state.candidateId)),
  ]);
  const summary = el("div", {id: "compare-summary"});
  stage.replaceChildren(
    el("section", {class: "panel"}, [
      el("h2", {}, ["两侧报告"]),
      form,
      el("div", {class: "actions"}, [
        button("primary", "对比", () => renderComparison(form, summary)),
        button("ghost", "按案例对照", () => {
          readPair(form);
          show("book");
        }),
      ]),
      summary,
    ]),
  );
  if (state.baselineId && state.candidateId) await renderComparison(form, summary);
}

function readPair(form) {
  const data = new FormData(form);
  state.baselineId = String(data.get("baseline") || "");
  state.candidateId = String(data.get("candidate") || "");
  state.readingId = state.baselineId;
}

async function renderComparison(form, summary) {
  showError("");
  readPair(form);
  if (!state.baselineId || !state.candidateId) {
    showError("请选择两侧运行");
    return;
  }
  try {
    const [left, right] = await Promise.all([loadBook(state.baselineId, true), loadBook(state.candidateId, true)]);
    summary.replaceChildren(reportPair(left, right), caseDelta(left, right));
  } catch (error) {
    showError(error.message);
  }
}

function reportPair(left, right) {
  const leftScore = scoreOf(left);
  const rightScore = scoreOf(right);
  const delta = leftScore == null || rightScore == null ? null : rightScore - leftScore;
  return el("div", {class: "metrics"}, [
    scoreCard("基线", left, leftScore),
    scoreCard("改进后", right, rightScore, delta),
  ]);
}

function scoreOf(book) {
  if (book.run.final_score != null) return Number(book.run.final_score);
  const metrics = (book.report && book.report.metrics) || {};
  return metrics.final_score == null ? null : Number(metrics.final_score);
}

function scoreCard(title, book, score, delta) {
  const deltaText = delta == null ? "" : `${delta > 0 ? "+" : ""}${delta.toFixed(3)}`;
  return el("section", {class: "panel"}, [
    el("h3", {}, [title]),
    el("p", {class: "stat"}, [score == null ? "—" : score.toFixed(3), " ", el("small", {}, [deltaText])]),
    el("p", {class: "mono"}, [book.run.run_id]),
    el("p", {}, [`${book.run.status} · ${book.cases.length} 条案例`]),
    metricTable((book.report && book.report.metrics) || book.run.metrics || {}),
  ]);
}

function metricTable(metrics) {
  const table = el("table");
  for (const [key, value] of Object.entries(metrics)) {
    table.append(el("tr", {}, [el("td", {}, [key]), el("td", {class: "mono"}, [pretty(value)])]));
  }
  return table;
}

function caseDelta(left, right) {
  const rightById = new Map(right.cases.map((item) => [item.item_id, item]));
  const ids = [...new Set([...left.cases.map((item) => item.item_id), ...right.cases.map((item) => item.item_id)])];
  const table = el("table", {}, [
    el("thead", {}, [el("tr", {}, ["案例", "基线", "改进后", "差值", ""].map((name) => el("th", {}, [name])))]),
  ]);
  const body = el("tbody");
  for (const itemId of ids) {
    const before = left.cases.find((item) => item.item_id === itemId);
    const after = rightById.get(itemId);
    const delta = before && after && before.score != null && after.score != null ? after.score - before.score : null;
    const klass = delta == null ? "" : delta > 0 ? "delta up" : delta < 0 ? "delta down" : "";
    body.append(el("tr", {}, [
      el("td", {class: "mono"}, [itemId]),
      el("td", {}, [before && before.score != null ? String(before.score) : "—"]),
      el("td", {}, [after && after.score != null ? String(after.score) : "—"]),
      el("td", {class: klass}, [delta == null ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(3)}`]),
      el("td", {}, [button("ghost", "对照", () => {
        state.caseId = itemId;
        state.readingId = left.run.run_id;
        show("book");
      })]),
    ]));
  }
  table.append(body);
  return el("section", {}, [el("h3", {}, ["案例明细"]), table]);
}

async function openBook(runId) {
  state.readingId = runId;
  state.baselineId = state.baselineId || runId;
  await show("book");
}

async function paintBook() {
  const reading = state.readingId || state.baselineId || (state.runs[0] && state.runs[0].run_id) || "";
  state.readingId = reading;
  if (!reading) {
    stage.replaceChildren(el("section", {class: "panel"}, [
      el("h2", {}, ["案例"]),
      el("p", {class: "lead"}, ["先在评测页打开一次运行。"]),
    ]));
    return;
  }
  const primary = await loadBook(reading, true);
  const secondaryId = state.candidateId && state.candidateId !== reading ? state.candidateId : "";
  const secondary = secondaryId ? await loadBook(secondaryId, true) : null;
  if (!state.caseId || !primary.cases.some((item) => item.item_id === state.caseId)) {
    state.caseId = primary.cases[0] ? primary.cases[0].item_id : "";
  }
  drawBook(primary, secondary);
}

function drawBook(primary, secondary) {
  const reading = primary.run.run_id;
  const secondaryId = secondary ? secondary.run.run_id : "";
  const chapters = primary.cases.filter((item) => {
    const blob = `${item.item_id} ${pretty(item.input)}`.toLowerCase();
    return blob.includes(state.query.toLowerCase());
  });
  const current = primary.cases.find((item) => item.item_id === state.caseId) || chapters[0];
  const twin = current && secondary ? secondary.cases.find((item) => item.item_id === current.item_id) : null;
  const toc = el("aside", {class: "toc"}, [
    el("h2", {}, [secondary ? "对照目录" : "目录"]),
  ]);
  const filter = el("input", {class: "filter", placeholder: "搜索案例", value: state.query});
  filter.addEventListener("input", () => {
    state.query = filter.value;
    drawBook(primary, secondary);
    const next = document.querySelector(".toc .filter");
    if (next) {
      next.focus();
      next.setSelectionRange(state.query.length, state.query.length);
    }
  });
  toc.append(filter);
  for (const item of chapters) {
    const box = el("input", {type: "checkbox"});
    box.checked = state.selected.has(item.item_id);
    box.addEventListener("change", () => {
      if (box.checked) state.selected.add(item.item_id);
      else state.selected.delete(item.item_id);
    });
    const other = secondary && secondary.cases.find((row) => row.item_id === item.item_id);
    const mark = other && other.score != null && item.score != null ? ` ${other.score > item.score ? "↑" : other.score < item.score ? "↓" : "="}` : "";
    const chapter = el("button", {
      type: "button",
      class: item.item_id === (current && current.item_id) ? "chapter on" : "chapter",
      on: {click: () => { state.caseId = item.item_id; drawBook(primary, secondary); }},
    }, [item.item_id, el("span", {class: "score"}, [`${item.score ?? "—"}${mark}`])]);
    toc.append(el("div", {class: "chapter-row"}, [box, chapter]));
  }
  toc.append(el("div", {class: "actions"}, [
    button("ghost", "回放已选", () => show("runs")),
  ]));
  const book = el("div", {class: secondary ? "book compare" : "book"}, [toc, pageOf(current, twin, primary, false)]);
  if (secondary) book.append(pageOf(twin, current, secondary, true));
  const picker = el("form", {class: "grid two"}, [
    field("正在阅读", select("reading", runOptions("选择运行"), reading)),
    field("对照版本，可空", select("other", runOptions("不对照"), secondaryId)),
  ]);
  picker.addEventListener("change", () => {
    const data = new FormData(picker);
    state.readingId = String(data.get("reading") || "");
    state.baselineId = state.readingId;
    state.candidateId = String(data.get("other") || "");
    state.caseId = "";
    show("book");
  });
  stage.replaceChildren(
    el("section", {class: "panel"}, [
      el("h2", {}, ["案例阅读"]),
      el("p", {class: "lead"}, ["左侧是目录。选了对照版本后，同一条 case 会左右并排，有差异的段落会标出来。"]),
      picker,
    ]),
    book,
  );
}

function pageOf(item, other, book, mirror) {
  if (!item) return el("article", {class: "page"}, [el("p", {}, ["这一侧没有这条案例。"])]);
  const index = book.cases.findIndex((row) => row.item_id === item.item_id);
  const changed = (key) => other && pretty(item[key]) !== pretty(other[key]);
  const article = el("article", {class: "page"}, [
    el("p", {class: "kicker"}, [mirror ? "改进后" : "基线", ` · ${index + 1} / ${book.cases.length}`]),
    el("h2", {}, [item.item_id]),
    el("p", {class: "meta"}, [`${book.run.benchmark_name || ""} · ${book.run.status} · 得分 ${item.score ?? "—"}`]),
    textBlock("输入", item.input, changed("input")),
    textBlock("期望", item.expected, changed("expected")),
    textBlock("输出", item.output, changed("output")),
    graphBlock(item.graphs, other && other.graphs),
    criteriaBlock(item.criteria, other && other.criteria),
  ]);
  const previous = book.cases[index - 1];
  const next = book.cases[index + 1];
  article.append(el("div", {class: "pager"}, [
    button("ghost", previous ? `上一篇 · ${previous.item_id}` : "上一篇", () => {
      if (!previous || mirror) return;
      state.caseId = previous.item_id;
      paintBook();
    }),
    button("ghost", next ? `下一篇 · ${next.item_id}` : "下一篇", () => {
      if (!next || mirror) return;
      state.caseId = next.item_id;
      paintBook();
    }),
  ]));
  return article;
}

function graphBlock(graphs, others) {
  const section = el("section", {class: "block"}, [el("h3", {}, ["节点运行图"])]);
  for (const graph of graphs || []) {
    const twin = (others || []).find((item) => item.phase_id === graph.phase_id) || (others || [])[0];
    section.append(drawGraph(graph, twin));
  }
  if (!graphs || !graphs.length) section.append(el("p", {}, ["这条案例还没有节点轨迹。"]));
  return section;
}

function drawGraph(graph, other) {
  const flow = el("div", {class: "flow"});
  if (graph.phase_id) flow.append(el("p", {class: "flow-phase"}, [graph.phase_id]));
  if (!graph.recorded) {
    flow.append(el("p", {class: "flow-note"}, ["这次运行没有留下节点耗时，图按当前评测拓扑还原。"]));
  }
  (graph.nodes || []).forEach((node, index) => {
    const twin = other && (other.nodes || []).find((item) => item.id === node.id);
    const changed = twin && (twin.status !== node.status || (twin.reason || "") !== (node.reason || ""));
    const card = el("div", {class: `flow-node ${node.status || ""}${changed ? " changed" : ""}`}, [
      el("strong", {}, [NODE_LABELS[node.id] || node.id]),
      el("span", {class: "mono"}, [node.id]),
      el("em", {}, [STATUS_LABELS[node.status] || node.status || ""]),
    ]);
    if (node.duration_ms != null) card.append(el("span", {class: "mono"}, [`${node.duration_ms} ms`]));
    if (node.reason) card.append(el("p", {}, [node.reason]));
    flow.append(card);
    if (index < graph.nodes.length - 1) {
      const edge = (graph.edges || []).find((item) => item.source === node.id);
      flow.append(el("div", {class: edge && edge.taken === false ? "flow-arrow off" : "flow-arrow"}, ["↓"]));
    }
  });
  return flow;
}

function textBlock(title, value, changed) {
  return el("section", {class: changed ? "block changed" : "block"}, [
    el("h3", {}, [title]),
    el("pre", {}, [pretty(value)]),
  ]);
}

function criteriaBlock(rows, others) {
  const list = el("div");
  for (const row of rows || []) {
    const twin = (others || []).find((item) => item.criterion_id === row.criterion_id);
    const changed = twin && (twin.score !== row.score || twin.status !== row.status || twin.reason !== row.reason);
    list.append(el("div", {class: changed ? "criterion changed" : "criterion"}, [
      el("strong", {}, [row.criterion_id || ""]),
      el("span", {}, [row.status || ""]),
      el("span", {class: "mono"}, [row.score == null ? "—" : String(row.score)]),
      el("p", {}, [row.reason || ""]),
    ]));
  }
  return el("section", {class: "block"}, [el("h3", {}, ["评分"]), list]);
}

tabs.addEventListener("click", (event) => {
  const view = event.target && event.target.dataset && event.target.dataset.view;
  if (view) show(view);
});

window.addEventListener("hashchange", () => {
  const hash = location.hash.replace("#", "");
  if (hash !== state.view) show(hash);
});

async function boot() {
  const [profiles, domains, runs] = await Promise.all([
    api("/v1/profiles"),
    api("/v1/eval-domains"),
    api("/v1/runs"),
  ]);
  state.profiles = profiles.profiles || [];
  state.domains = domains.eval_domains || {};
  state.runs = runs.runs || [];
  const hash = location.hash.replace("#", "");
  await show(VIEWS.includes(hash) ? hash : "import");
}

boot().catch((error) => showError(error.message));
