# Triage + Attribution 开发指引

评测器（judge）跑完产出 `EvaluatorResult{score, reason}` 后，`EvaluatorService.after_eval`
会**同步 await** `TriageAttributionService.run()`，按 score 分流决定是否跑归因；
归因在 `eval_report_root` 关闭前完成，HTTP/Kafka 响应在归因结束后才返回。
`TriageOutcome.status` 为 `attribution_failed` / `dispatch_failed` 时由 `after_eval`
映射为 `EvaluatorError`（HTTP/MQ 失败）；`skipped` / `created` 仍成功返回。
模型服务端通过 QPS/QPM 限额和 429 退避重试控制流量。

```
EvaluatorService.after_eval(evaluator_name, request, result)
   │  report_child_scope(triage_attribution)  # 容器 span，嵌套 graph/LLM/tool
   │  同步 await（HTTP/Kafka 等归因完成后再返回；仍在 eval_report_root 内）
   ▼
TriageAttributionService.run()                     app/services/triage_attribution/service.py
   │  ① before_hook: TriageBeforeRunHook
   │     ├─ classifiers[evaluator_name].classify(TriageInput) → TriageVerdict
   │     └─ should_attribute=False → skipped（good / watch / no_decision）
   │  ② AttributionAgent.call()                     app/triage_attribution/example/attribution/
   │     └─ agent_manager.get("attribution_<name>")
   │  ③ after_hook: AoneAfterRunHook → create_aone_task()
   ▼
DB 写入（update_triage_level / update_attribution / update_dispatch / enqueue_aone_task）
```

## 核心概念

| 概念 | 文件 | 职责 |
|---|---|---|
| `TriageClassifier` | `app/services/triage_attribution/classifier.py` | ABC，纯 CPU 同步 `classify(TriageInput) → TriageVerdict` |
| `TriageInput` / `TriageVerdict` | `app/services/triage_attribution/schema.py` | 入参 / 出参契约 |
| `triage_decide` | `app/services/triage_attribution/triage.py` | 调 classifier + 兜底异常处理 → `TriageDecision` |
| `TriageAttributionService` | `app/services/triage_attribution/service.py` | 编排 before_hook → attribution agent → after_hook，含 DB 写入 |
| `BaseAttributionAgent` | `app/triage_attribution/business_components/agent.py` | 统一 `run/call`、结果序列化和异常响应；业务实现 `attribute()` |
| 公共归因契约 | `app/triage_attribution/business_components/contracts.py` | Context、规则信号、Evidence Gate 和六类结论的业务无关字段 |
| 公共归因 LLM 适配 | `app/triage_attribution/business_components/llm.py` | 懒加载 ChatModel 并适配双段 prompt；旧业务路径保留 re-export |
| `classifiers_registry` | `app/triage_attribution/classifiers_registry.py` | `triage_classifiers()` → `{logic_name: classifier}`，与 `evaluators/registry.py` 对称 |
| `AttributionAgent` | `app/triage_attribution/example/attribution/agent.py` | `@agent_component(name="attribution_<name>")`，经 `trace` 按需取证 |

## 归因公共组件与业务边界

`app/triage_attribution/business_components/` 只提供执行机制和稳定契约，不包含业务维度、
failure taxonomy、规则表或 prompt：

- 业务 Agent 继承 `BaseAttributionAgent`，实现 `attribute(request)`；公共基类保持
  `AgentResponse` 的成功/失败信封一致。
- Vehicle/VUI 的归因 Context、RuleSignal、GateVerdict 继承公共字段后继续收窄
  schema version、信号来源和 gate 枚举，序列化字段保持不变。
- `ChatModelAttributionLLM` 的实现统一位于公共目录；`example/_internal/llm.py` 和
  `vehicle_agent/_internal/llm.py` 作为兼容导出保留，旧 import 路径无需迁移。
- `contracts.py` 声明 `AttributionContextAdapter`、`RuleMatcher`、`EvidenceGate`、
  `InvestigationRunner`、`DecisionResolver` 与 `FinalValidationGate` 的阶段接口；
  `runtime.py` 固定推进 normalize envelope → match rules → evidence gate → optional
  investigation → decide and validate，支持终局结果短路，业务异常仍由
  `BaseAttributionAgent.call` 统一收敛。
- VUI 已接入公共 Runtime：规则直判不调查，其余路径选择 `single_llm`，原有 taxonomy、
  confidence、冲突、人审和 evidence ref 门禁保持在 VUI 插件内。
- Vehicle 已接入公共 Runtime：规则直判 / 不可达路径不调查（零 LLM、零 trace 工具），
  其余路径选择 `react_tools`（Final-Chat-First 分层取证的 `_investigation_react` 原样
  作为 Stage 4）；quote gate、多根因、人审、派单门禁与 Langfuse 阶段 span 保持在
  Vehicle 插件内（stages.py 纯函数不变）。

公共层禁止根据 `logic_name` 分支，也禁止反向 import 任一业务归因包。

## 分级规则（TriageLevel）

定义在 `app/data_model/enums.py`，与 `TraceRecord.triage_level` 标量列对齐：

| level | should_attribute | 后续 |
|---|---|---|
| `good` | False | 跳过 |
| `watch` | False | 跳过 |
| `mild_bad` | True | 跑 AttributionAgent + Aone |
| `severe_bad` | True | 跑 AttributionAgent + Aone |
| `no_decision` | False | 跳过 |
| `attribution_overridden` | — | **非 classifier 产出**：归因推翻 judge 后平台回写的标记值 |

> ⚠️ **阈值由各 classifier 自带**，平台层**不提供默认阈值策略**。下表是 example 参考实现的阈值，
> 仅作示例 —— 你的 classifier 应当根据业务自定义阈值或换用完全不同的策略（聚类 / LLM 二裁 / 规则 DSL）。

**example 参考阈值**（见 [`example/classifier.py`](./example/classifier.py)，**不要照抄**）：

| level | example 阈值 |
|---|---|
| `good` | score ≥ 8.0 |
| `watch` | 6.0 ≤ score < 8.0 |
| `mild_bad` | 3.0 ≤ score < 6.0 |
| `severe_bad` | score < 3.0 |
| `no_decision` | score = None |

## 配置

`conf/app.<env>.yaml` 的 `custom.triage_attribution` 段，由 `benchmark.settings.triage_loader.load_triage_configs` 解析：

```yaml
custom:
  triage_attribution:
    triage:
      badcase_score_threshold: 0.5    # 遗留字段，classifier 自带阈值，此处不影响决策
      policy_match_strict: true       # 遗留字段
      label_studio_dispatcher_enabled: false
    aone:
      enabled: false
      dry_run: true
      a1_bin: a1
      project: agent-eval
      category: badcase
      workitem_type: bug
```

`AppConfigManager` 通过 `register_section("custom", _CustomSectionSchema)` 放行此段。

> HTTP 立即返回 judge，后台归因不设置本地并发闸。模型限流由服务端 QPS/QPM
> 配额与 `RetryingChatModel` 的 429 退避重试处理。`triage` 段三个字段当前都是
> **遗留兼容字段，不影响决策** —— 阈值由各 classifier 自带。改阈值就改你的
> classifier 常量，**不要**改这里的 `badcase_score_threshold`。

## 接入新分流（3 步 checklist）

- [ ] **classifier** 在 `app/triage_attribution/<logic_name>/classifier.py` 实现
      `TriageClassifier.classify(TriageInput) → TriageVerdict`。阈值常量写在你的 classifier 里
      （参考 `example/classifier.py` 的 `GOOD_MIN_SCORE` 等，但值由你定）。
- [ ] **配对 agent** 在 `app/triage_attribution/<logic_name>/attribution/agent.py` 写归因 agent
      （`@agent_component(name="attribution_<logic_name>")`）。同步三份 `conf/app.<env>.yaml`：
      - `models.attribution_<logic_name>` —— 命名模型
      - `prompts.template_dir` —— **追加** `conf/agents/attribution_<logic_name>/prompts`
      - `agents.attribution_<logic_name>` —— `type: attribution_<logic_name>` + ext 字段
      prompt 模板**必须**放在 `conf/agents/attribution_<logic_name>/prompts/<prompt_name>.prompt`。
- [ ] **注册**（两个入口，**对称分离**）：
      - 在 [`classifiers_registry.py`](./classifiers_registry.py) 加 import + 在 `_CLASSIFIERS` dict
        加 `"<logic_name>": <Name>Classifier()`
      - 在 [`registry.py`](./registry.py) 加一行 `# noqa: F401` eager-import 归因 agent 模块触发
        `@agent_component` 注册

      > `classifiers_registry.py` 只登记 classifier 类（非 `@agent_component`），
      > `registry.py` 只登记 attribution agent（`@agent_component`）。两个文件不要混加。
- [ ] **测试**：在 `tests/unit/triage_attribution/<logic_name>/` 下加单测。最小模板见下方「测试模板」。
- [ ] **归因结论契约**：prompt / `AttributionResult` 覆盖「推翻 judge」「回写分级」「拒派 Aone」
      （见下方专节；参考 `conf/agents/attribution_example/prompts/badcase_attribution.prompt`）。

## 归因结论契约：推翻 judge / 回写分级 / 控制 Aone

业务归因 agent 产出 `AttributionResult`
（`app/services/badcase_attribution/schemas.py`）。对接入方而言，有两个抓手：

| 字段 | 谁填 | 平台行为 |
|---|---|---|
| `corrected_triage_level` | 归因 agent（可选） | 有合法值时：标量 `triage_level` → `attribution_overridden`；纠正分级写入 bucket；原分级写入 `original_triage_level`；强制 `should_dispatch=False` |
| `should_dispatch` | 归因 agent（默认 `true`） | `false` 时 `AoneAfterRunHook` 短路拒派，不调 a1 CLI |

> **一句话**：填 `corrected_triage_level` 表示推翻；平台用 `attribution_overridden` 标记标量、保留原/纠正分级到 bucket，并强制拒派 Aone。
> 填了前者时平台会强制拒派（即使 agent 忘了设 `should_dispatch=False`）。

### 决策矩阵

| 场景 | `corrected_triage_level` | `should_dispatch` | `triage_level` 标量 | bucket 额外字段 | Aone |
|---|---|---|---|---|---|
| 正常归因、要派单 | `null` | `true`（默认） | 保留 before 写入的分级 | — | 按灰度开关派 |
| 证据不足 / 人工复核 | `null` | `false` | **不改** | — | 拒派 |
| **推翻 judge（误判）** | 合法分级（常 `good`） | 平台强制 `false` | → **`attribution_overridden`** | `corrected_triage_level` + `original_triage_level`（classifier 原值） | 拒派 |
| escalate（低置信 / 缺证据 / 缺 refs） | 平台清成 `null` | `false` | **不回写**（避免半成品推翻） | — | 拒派 |

合法分级闭集（agent 可填入 `corrected_triage_level`）：`good` / `watch` / `mild_bad` / `severe_bad` / `no_decision`（非法值 schema 归一为 `null`）。
`attribution_overridden` **仅平台回写标量时使用**，agent **不要**把它填进 `corrected_triage_level`。

### 写库顺序

```
① before_hook（classifier）→ update_triage_level(judge 分级，如 mild_bad)
② AttributionAgent.call() → AttributionResult（corrected_triage_level=good 等）
③ _write_attribution
     └─ update_attribution：单次 UPDATE 同时写 bucket + 推翻标量
          root_cause / owner_domain / confidence / attribution_raw
          evidence_raw / reasoning_raw / action_hint / needs_human_review
          corrected_triage_level = agent 纠正分级（若有）
          original_triage_level  = ① 的 classifier 分级
          triage_level           = attribution_overridden（仅当 corrected 有值）
④ after_hook → 仅当 should_dispatch=true 且灰度允许才 create_aone_task
```

> 推翻分级**不再**单独调 `update_triage_level`。归因结论与最终分级是一次决策，
> 拆成两条写会在中途被取消或被并发覆盖时留下「bucket 已推翻、`triage_level`
> 仍是原分级」的不一致行。

实现落点：

- agent 侧强制拒派：`example/attribution/agent.py` → `_materialize_final`
- service 侧兜底拒派 + 回写：`TriageAttributionService._write_attribution`
- bucket + 推翻标量写入：`TraceRecordService.update_attribution`（`corrected_` / `original_triage_level` / `triage_level`）
- `TraceRecordService.update_triage_level` 仅用于 ① before_hook 的 classifier 分级

### Prompt / 产出约定（推翻 judge）

证据充分证明 judge 误判时：

1. 走 `decision_type=final`（**不要** `human_review`）
2. `root_cause` 写清误判原因（禁止落成「需要人工复核」）
3. 填 `corrected_triage_level`（你的最终分级，通常 `good`；**不要**填 `attribution_overridden`）
4. `needs_human_review=false`

参考模板：[`conf/agents/attribution_example/prompts/badcase_attribution.prompt`](../../conf/agents/attribution_example/prompts/badcase_attribution.prompt)。

仅拒派、不推翻分级时：只设 `should_dispatch=false`，`corrected_triage_level` 省略或 `null`。

### 验收单测（照抄断言）

- 推翻回写 + 拒派：`tests/unit/services/triage_attribution/test_service.py::test_run_rewrites_triage_and_skips_aone_when_corrected_triage_level`
  （断言 `triage_level == "attribution_overridden"`，且 bucket 含 `original_triage_level` / `corrected_triage_level`）
- agent 强制 `should_dispatch=false`：`tests/unit/triage_attribution/example/attribution/test_graph.py::test_corrected_triage_level_forces_should_dispatch_false`
- escalate 丢弃半成品：`.../test_graph.py::test_escalation_clears_corrected_triage_level`

字段落库说明见 [`docs/data-model.md`](../../docs/data-model.md) §`attribution_detail_json` / §`notification_dispatch_json`。

## 工具接入（可选）

平台公共取证已由 `tools.providers: [trace]` 提供（见
[`trace_resolver/README.md`](../services/trace_resolver/README.md)），归因 agent
直接 `get_provider("trace")` 即可，**不必**再挂白名单。

业务私有工具两条路：

- **A（推荐）**：自建 `BaseToolProvider` + `@tool_component`，YAML 追加 provider 名。
  tool metadata adapter 已由平台 `trace` 模块注册，**不需要** `libs/tools`。
- **B**：下方 `AgentTool` + `AgentToolProvider`（需依赖
  [`libs/tools`](../../../../libs/tools/README.md)）。

不要自建 runtime / 手写 `tool_manager.register_provider`——走 YAML +
`@tool_component`。

### 4 步接入（路径 B：AgentToolProvider）

- [ ] **写工具**：在 `app/triage_attribution/<logic_name>/tools/` 下写 `AgentTool`
      子类，`__init__.py` 导出类（`ToolLoader` 用 `inspect.getmembers` 扫顶层类）：

      ```python
      from typing import ClassVar
      from agentgraph.tools import AgentTool
      from agentgraph.tools.content import ToolResult
      from agentgraph.tools.context import ToolExecutionContext
      from pydantic import BaseModel, Field

      class BarArgs(BaseModel):
          trace_id: str = Field(..., description="待查 trace id")

      class BarTool(AgentTool):
          name: ClassVar[str] = "bar"
          description: ClassVar[str] = "查询 trace 原始数据"
          input_schema: ClassVar[type[BaseModel]] = BarArgs
          owner: ClassVar[str] = "<team>"
          stage: ClassVar[str] = "private"
          since: ClassVar[str] = "2026-07-08"

          async def run(self, args: BarArgs, ctx: ToolExecutionContext) -> ToolResult:
              return ToolResult.from_text(f"trace {args.trace_id} raw data ...")
      ```

- [ ] **框架装配入口**：在 `app/triage_attribution/<logic_name>/tools/providers.py`
      写一行 `@tool_component`。AppRuntime `walk_packages` 按模块名尾缀 `providers`
      自动发现：

      ```python
      from agentgraph.tools.component_metadata import tool_component
      from tools.agent_tool_provider import AgentToolProvider, AgentToolPackagesConfig

      @tool_component(name="<logic_name>_tools", ext_schema=AgentToolPackagesConfig)
      class <LogicName>ToolProvider(AgentToolProvider):
          pass
      ```

- [ ] **YAML**：平台已在 `tools.providers` 注册 `trace`；业务私有工具再追加自己的 provider：

      ```yaml
      tools:
        providers:
          - trace          # 平台公共（已有）
          - <logic_name>_tools      # 仅业务私有时追加
        extensions:
          <logic_name>_tools:
            packages: [benchmark.attribution.<logic_name>.tools]
            name: <logic_name>
      ```

- [ ] **在归因 agent 里调用**：通过 `runtime.tool_manager`（AppRuntime 已装好）。

      ```python
      result = await runtime.tool_manager.call_tool("bar", {"trace_id": tid}, context=ctx)
      text = result.to_text()
      ```

      > 框架 `AgentToolProvider.call_tool` 默认构造基类 `ToolExecutionContext`；
      > 若工具需要扩展字段（如自定义上下文），调用时显式传 `context=<your_ctx>`，
      > 该实例会透传给 `AgentTool.run(args, ctx)`。

### 不要做的事

- ❌ 不要自建 runtime / 手写 `tool_manager.register_provider(...)` —— YAML +
  `@tool_component` 自动完成（路径 A 的自建 `BaseToolProvider` 可以，但须挂装饰器）
- ❌ 不要在工具里反向 import `benchmark.services.*` 的业务编排 —— 工具是叶子能力，
  编排由归因 agent 的 ReAct 循环做
- ❌ 路径 B 不要漏依赖 `libs/tools`，否则 `AgentToolProvider` / scan `"tools"` 无效

### 与 judge 评测器共用工具

工具落点默认是 `app/triage_attribution/<logic_name>/tools/`（归因 agent 私有）。
若同一批工具也要被评测器复用，提到
[`libs/tools/src/tools/common/`](../../libs/tools/src/tools/common/)
（跨业务公共工具层），YAML 的 `packages` 加上 `tools.common`。完整接入范式见
[`app/evaluators/README.md`](../evaluators/README.md)「工具接入（可选）」。

## 测试模板

参考实现测试在 [`tests/unit/triage_attribution/example/`](../../tests/unit/triage_attribution/example/)：
- [`test_classifier.py`](../../tests/unit/triage_attribution/example/test_classifier.py) —— classifier 纯逻辑：表驱动测试 `classify` 的各级别阈值边界
- [`test_registry.py`](../../tests/unit/triage_attribution/example/test_registry.py) —— 注册表：`triage_classifiers()` 返回正确 logic_name

**classifier 最小单测骨架**（copy 后改 `<name>`）：

```python
from benchmark.attribution.service import TriageInput, TriageLevel
from benchmark.attribution.<name>.classifier import <Name>TriageClassifier


def _classify(score: float | None) -> tuple[TriageLevel, bool]:
    verdict = <Name>TriageClassifier().classify(
        TriageInput(trace_id="t", evaluator_name="<name>", project_id="p", score=score)
    )
    return verdict.level, verdict.should_attribute


def test_good_does_not_attribute():
    assert _classify(<your_good_threshold>) == (TriageLevel.GOOD, False)


def test_severe_bad_attributes():
    assert _classify(<your_severe_threshold>) == (TriageLevel.SEVERE_BAD, True)


def test_no_score_is_no_decision():
    assert _classify(None) == (TriageLevel.NO_DECISION, False)
```

> 跑测试：`cd services/agent-eval && .venv/bin/python -m pytest tests/unit/triage_attribution/<name>/ -v`

## 目录结构

```
app/
├── services/triage_attribution/          # 平台层（契约 + 编排）
│   ├── classifier.py                     # TriageClassifier ABC
│   ├── schema.py                         # TriageInput / TriageVerdict
│   ├── triage.py                         # triage_decide + TriageConfig
│   └── service.py                        # TriageAttributionService + hooks
└── triage_attribution/                   # 业务层（具体实现 + 注册入口）
    ├── classifiers_registry.py           # classifier 注册入口
    ├── registry.py                       # @agent_component 注册入口
    ├── example/                          # 参考实现（[example/README.md](./example/README.md)）
    ├── memory_pairwise/                  # 记忆增益归因（见下方「已接入业务」）
    ├── vehicle_agent/                    # vehicle-agent ReAct 取证归因
    └── vui_agent/                        # VUI 语音链路规则优先归因（见下方「已接入业务」）
```

## 参考实现

照抄模板见 [`example/README.md`](./example/README.md)——参考分类器 + 归因 agent 的最小装配。

## 已接入业务：memory_pairwise（记忆增益归因）

与 badcase 类评测器口径不同，是刻意的例外，改动前先读
[`../evaluators/README.md`](../evaluators/README.md) 的「记忆 Pairwise 消融 Judge」：

- **三个方向都归因**：judge 的 score 是相对胜负（1.0 记忆帮忙 / 0.5 无差异 / 0.0 记忆致损），
  不是绝对质量分。classifier（`memory_pairwise/classifier.py`）对三值全部
  `should_attribute=True`；level 只是标签——0.0→`mild_bad`（相对回归不占 severe_bad 口径）、
  1.0→`good`（good + 归因合法，level 与 should_attribute 解耦）、0.5→`watch`；
  三值之外→`no_decision` 弃权。
- **归因 agent 是单轮 LLM，不是 ReAct**（`memory_pairwise/attribution/agent.py`）：证据在评测
  阶段已齐（`judge_json.extra` 带 `mem_on_answer` / `mem_off_answer` / `rounds[].memory_signal`），
  预拼进 prompt 一次调用出结论，不依赖 `trace` 取证工具。
- **永不派 Aone 工单**：`should_dispatch` 在构造 `AttributionResult` 时**硬编码 False**
  （不由 LLM 输出决定），`service.py` 的 after_hook 会记 `agent_should_dispatch_false` 跳过派单。
  不要通过改 `custom.triage_attribution.aone` 来实现这一点——那是全局开关。
- **无信号早退**：tie 且 `memory_signal` 全为 `none` → 直接产出 `root_cause=no_signal_space`，
  不调 LLM。该判断放 agent 侧是因为 `TriageInput` 是 `extra="forbid"`、classifier 看不到
  `memory_signal`。
- **记忆因子闭集**写在 `memory_pairwise/constants.py::MEMORY_CAUSE_CATEGORIES`（10 个
  root_cause → 平台 7 类 category 映射）；`AttributionResult` 是 `extra="ignore"`，
  记忆语义只能承载在 `root_cause` 列。
- 批量读数：纯函数聚合见 `app/evaluators/memory_pairwise/aggregation.py`（胜/平/负、
  一致性率、`memory_signal` 分层、`query_category` 净口径交叉表）。

## 已接入业务：vui_agent（VUI 语音链路规则优先归因）

与 vehicle_agent 的 ReAct 取证路线不同，VUI 归因是「规则优先 + 单轮 LLM 兜底」，
设计依据 `docs/specs/vui-eval-closed-loop-design.md` §4.7 / §5.3：

- **classifier 0-2 分档**（`vui_agent/classifier.py`，口径版本 `vui-triage-v1`）：
  score ≥ 2 → good 不归因；≥ 1 → mild_bad；< 1 → severe_bad；None → no_decision。
- **规则优先零 LLM**（`vui_agent/attribution/rules.py`）：extra 信封
  （`vui_eval_extra.v1`）的 hard_rule_hits（`vui_max_loops_reached` / `vui_shape_cmd_missing`）
  置信度 ≥ 0.85 直通阈值时直接出结论；risk_flags 只产候选信号交 LLM 兜底；
  `source=sdm ∧ 形态失败` 走 SDM 误路由组合规则（0.80）。
- **checker/LLM 冲突强制人审**：`checker_llm_conflict` flag 不进规则表，gate 下发
  `judge_false_positive` 假设；LLM 仍判业务缺陷时保守转 `ambiguous_root_cause`，不派单。
- **taxonomy 闭集 + 根因指纹**（`vui_agent/taxonomy.py`，版本 `vui-taxonomy-v1`）：9 个
  二级分类，越界 LLM 输出归一为 unknown + 人审；指纹
  `sha256(taxonomy_version|agent|mode|category|subcategory|component)` 以
  `fingerprint:` 前缀进 `evidence_refs`（repair-handoff.v1 属二期）。
- **失败一律收敛为人审**：prompt 缺失 / LLM 异常 / 输出不可解析 / 信封降级
  均产 `needs_human_review=True, should_dispatch=False`，不抛穿。
- **模型装配**：`models.attribution_vui_agent` 与 `judge_vui_agent` 复用同一
  qwen3.7-max 网关密钥（`AGENT_EVAL_VUI_AGENT_MODEL_API_KEY`），三环境一致。
