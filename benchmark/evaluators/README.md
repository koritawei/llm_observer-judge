# 评测器（Evaluator）开发指引

一个评测器 = 一个 `agentgraph` agent。请求经 OpenAI 兼容入口
`POST /v1/chat/completions` 进来，用 `model` 字段（= 评测器名）选中对应评测器执行。

```
POST /v1/chat/completions                       app/api/eval.py
   │  body.model = 评测器名(logic_name);API 层解析请求 → EvaluatorRequest
   ▼
EvaluatorService.evaluate(name, request)        app/services/evaluation.py
   │  ① 按 judge_<name> 选中评测器（runtime.agent_manager）
   │  ② before_eval(request)：业务 metadata.trace_id → Langfuse 主键（一次）
   │  ③ agent.call() ──► 你的评测器.evaluate()（继承 BaseEvaluatorAgent）   app/evaluators/<name>/
   │  ④ after_eval(name, request, result)：落库 + 同步 await triage 分流/归因
   ▼                                             （HTTP 立即返回，triage 后台跑）
EvaluatorResult → OpenAI chat.completion 返回    app/api/eval.py

[后台] TriageAttributionService.run()
   │  classifier.classify(score) → level
   │  mild_bad / severe_bad → AttributionAgent → Aone 派单
   │  good / watch → 跳过
```

评测器由 `AppRuntime` 按 `conf/app.<env>.yaml` 的标准 `agents` / `models` / `prompts`
section 自动装配——**你不写注册代码**。

## 你只实现两个方法

继承 [`base.py`](./base.py) 的 `BaseEvaluatorAgent`；`call` / `run` / `get_agent_card`
由基类提供，**勿重写**。

| 方法 | 职责 |
|---|---|
| `from_config(cls, cfg, ctx)` | bootstrap 期调一次：`cfg` = 你的 ext_schema（已校验），用 `ctx.runtime.get_llm(cfg.default_model)` 取模型、`ctx.runtime.prompt_manager` 取 prompt，构造并返回实例 |
| `async evaluate(self, request) -> EvaluatorResult` | 每次请求调用，承载全部评测逻辑：入参 → 出参 |

## 出入参契约（见 [`base.py`](./base.py)）

- **入参 `EvaluatorRequest`**（API 层已解析好，agent 不再解析）：
  `input` / `output`（`JsonValue`，格式不固定——结构化对象或纯文本，自行解释）、
  `metadata` / `trace_id` / `trace_timestamp` / `agent_name` / `project_id`。其中 Kafka Judge
  路径会把源 TraceRecord 的原始请求 `timestamp` 映射为可选 `trace_timestamp`。
- **出参 `EvaluatorResult`**：`score`（必填）/ `reason` / `extra` / `sub_scores`（可选）。
  `after_eval` 落库到 `judge_json`：必写 `score`/`reason`；`sub_scores` 非 None、
  `extra` 非空时一并写入（缺省不写键）。
- **失败**：`raise EvaluatorError(code=<HTTP 状态 int>, error_type=<标识 str>, message=...)`。
  `error_type` 用 [`app/utils/exceptions.py`](../utils/exceptions.py) 的常量（→ envelope `type`）；
  `code` 是 HTTP 状态（`model/prompt_not_found`→400、`llm_*`→502、其它→500，可用
  `evaluator_status(error_type)` 推导）。统一错误基类 `BaseError(code, error_type, message)`，
  API 错误 `APIError` 同源；envelope 为 `{message, type=error_type, param, code=int}`。

请求 message content 是一个 JSON 信封（API 层用 pydantic 一次解析为 `EvaluatorRequest`，
见 [`app/api/schema.py`](../api/schema.py)）：

```json
{
  "input":    { "…agent 请求快照（结构化对象或纯文本）…": "" },
  "output":   { "…agent 响应快照（结构化对象或纯 markdown 文本）…": "" },
  "metadata": { "trace_id": "…" },
  "trace_timestamp": "2026-08-03T06:52:03.968Z",
  "agent_name": "…",
  "project_id": "…"
}
```

## 跨业务组件

`app/evaluators/business_components/` 提供不包含业务规则的公共能力：

- `signals.py`：统一 `EvaluationSignal`、`DimensionResult` 与 `JudgePlan`。
- `contracts.py`：统一 `EvidenceFact` 与归因可消费的 `EvaluationSnapshot`；业务 extra
  信封在迁移期保持不变。
- `checker.py`：Checker 协议、结果工厂和单 Checker 异常隔离。
- `checker_registry.py`：`ComponentDescriptor`（唯一能力真源，含 `component_kind` /
  `emitted_metrics` / `emitted_diagnostics`；`CheckerDescriptor` 为历史别名）、
  `CheckerSpec`、显式 registry，以及包装现有 `DEFAULT_CHECKERS` 的
  `LegacyCheckerSetAdapter`。
- `evidence.py`：`EvidenceProvider` 协议与 `EvidenceProviderRegistry`——中立证据
  产出组件（只观测、不判分），Vehicle 的 extractor 经它独立注册。
- `profiles.py`：业务 Profile 模型（`ProfileSpec`、`MetricBinding`）与 Loader——
  加载期校验未知插件、重复实例、非法维度与缺失 extension 声明，生成
  `config_digest`；RemoteConfig 运行参数覆盖仅接受 `component_enabled` 白名单开关。
- `envelope.py`：统一信封 `agent_eval_extra.v1`（`AgentEvalEnvelope`）及
  `BusinessExtension`、`MetricObservation`、`DiagnosticFinding`、`FailureRelation`；
  公共强类型核心承载跨业务稳定语义，业务差异字段进入 `business_extension`。
- `extensions.py`：Vehicle/VUI 强类型扩展模型（`VehicleEvalExtensionV3`、
  `VuiEvalExtensionV2`），作为扩展 payload 的序列化边界；新 Vehicle 结果仅经
  `AgentEvalEnvelope.business_extension` 输出。
- `normalize.py`：`normalize_vehicle_extra` 读取视图兼容层——把历史顶层
  `vehicle_eval_extra.v3` 或统一信封转换为旧读取视图；它只服务历史数据与尚未迁移的
  消费者，不是新评测结果的写出通道。
- `runner.py`：公共评测 Runner——固定九节点标准拓扑（`build_context` →
  `resolve_profile` → `collect_evidence` → `run_checkers` → `plan_judges` →
  `run_judges` → `collect_metrics` → `aggregate_results` → `emit_envelope`）、
  阶段异常隔离（收敛为 `error`/`error_code`）、显式 `skipped` 语义与
  `stage_report` 耗时观测、`EnvelopeIdentity` 信封身份贯通；业务逻辑经
  `EvaluationPipelineHooks` 注入，公共层不含任何业务分支。

### 业务 Profile 装配

Vehicle/VUI 的 `DEFAULT_CHECKERS`（及 Vehicle `EVIDENCE_EXTRACTORS`）不再手工维护
元组，而是由各业务目录下的 `profile.yaml` 经显式注册表装配：

- `vehicle_agent/plugin.py` + `profile.yaml`：5 Checker + 5 Evidence Provider 显式
  工厂注册；`checkers/__init__.py` 在模块加载时调用 `default_checkers()` /
  `default_evidence_providers()` 完成装配，顺序、参数与降级语义与历史元组逐项等价。
- `vui_agent/plugin.py` + `profile.yaml`：3 Checker 显式工厂注册，维度标识已切换
  D0-D6（D1 拒识路径、D4 输出形态与完成度）；延迟不占维度，经 `metrics` 声明由
  `collect_metrics` 阶段独立采集为 `MetricObservation`，不参与主分。
- 组件集合/顺序变更只改 `profile.yaml`；新增组件先在 plugin 注册工厂与 Descriptor，
  再在 Profile 引用注册名。YAML 不允许引用模块路径或写条件表达式。

### 公共评测 Runner 接入（九阶段）

Vehicle/VUI 的评测图不再各自手工拼装 `StateGraph`，而是由
`business_components/runner.py` 的 `build_evaluation_graph` 按固定九节点拓扑编译，
业务逻辑由各目录下的 `pipeline.py` 经 `EvaluationPipelineHooks` 注入：

- `vehicle_agent/pipeline.py`：`build_context` = `build_base_eval_context`（tag 无关
  基础上下文）；`resolve_profile` = `classify_tag` + `attach_scenario_context`（场景
  资产挂载）+ 写入信封身份；`collect_evidence` = Evidence Extractors；`run_checkers`
  = Base Checkers（`risk_flags` 顺序契约：checker flags 在前、evidence risk_hints
  在后）；`plan_judges` = `route_judge_plan`（`execution_mode="fan_out"`）；
  `run_judges` = `fanout_sub_judges` 并发（无待跑子 judge 时 `skipped`）；
  `collect_metrics` = `skipped`（Metric Collector 后续落地）；`aggregate_results` =
  `aggregate_judge_result`；`emit_envelope` = 最终结论校验。
- `vui_agent/pipeline.py`：`build_context` = `build_eval_context`；`resolve_profile`
  = 加载 Profile 写信封身份（scenario 固定 `default`）；`collect_evidence` =
  `skipped`；`run_checkers` = Base Checkers；`plan_judges` =
  `route_judge`；`run_judges` = `build_prompt` + LLM `evaluate` 串行（checker 高置信
  直判时 `skipped`）；`collect_metrics` = 按 Profile `metrics` 声明采集首末
  Markdown 延迟（`vui_latency` collector，阈值缺失时仅记录观测值）；
  `aggregate_results` = `aggregate_verdict`；`emit_envelope` = 最终结论校验。
- 新业务接入：只写 `pipeline.py` 提供 Hooks，不得自建/增删图节点；没有实际工作的
  可选阶段返回 `skipped(原因)`，由 Runner 记入 `stage_report`。
- 两侧 `state.py` 新增 Runner 观测通道 `stage_report` / `envelope_identity`；
  `JudgePlan.execution_mode` 仅表达执行调度（`sequential` / `fan_out`），
  `exclude=True` 不参与序列化，业务 extra 信封形状保持不变。

## LLM 输出 JSON 解析

amap-agent、example、competitor-pairwise evaluator 及 attribution agent 的原始 LLM
输出通过 `benchmark.utils.llm_json.parse_and_validate_llm_json` 解析，统一遵循以下边界：

1. 先去除 Markdown JSON fence，并以目标 Pydantic model 严格解析；
2. 仅当 Pydantic 报 `json_invalid`（JSON 语法错误）时，使用
   `json_repair.repair_json` 修复；
3. 有未闭合 JSON 字符串或对象/数组边界的截断文本直接失败，不允许 repair 补全；
4. 修复后的文本必须再由同一个 Pydantic model 校验后才可使用；成功 repair 只记录
   model 名称和原始文本长度。

因此未转义引号、尾逗号等轻微模型序列化错误可以恢复；合法 JSON 的缺字段、非法枚举、
错误分数或业务约束错误不会进入 repair。共享 helper 仍返回失败；`search_ai_card` 把这类
schema 不符标成 `llm_schema_invalid`（422，不再重试），截断和空回复仍是 502。badcase
attribution 的模型客户端在 helper 返回 `None` 时仍记录原有诊断并抛出 `ValidationError`；
triage attribution 则维持不可解析时返回 `None`。该 helper 只用于 LLM 原始输出，不能用于
HTTP request envelope、trace / Langfuse、RemoteConfig、Aone 等外部契约 JSON。

`judge_amap_agent` 的 expression prompt 还在生成侧约束字符串转义：多句 evidence 使用全角
分号连接，字段值内引用原文统一使用中文引号「」，避免模型生成相邻字符串或未转义 ASCII
双引号。该约束仅用于降低轻微序列化错误概率；遇到语义不明确、无法安全修复的输出仍必须
fail-closed 为 `llm_unparseable`，不得为了产出分数而接受语义已损坏的 repair 结果。

amap evaluator 的公共 `complete_and_parse` 在首次 parser 返回 `None` 时会追加 JSON 纠正指令
并重新请求一次。重试 prompt 保留原始评审任务，但不会回灌首次损坏响应；第二次仍不可解析时
返回原有 `llm_unparseable`。首次解析成功只调用一次，任一次模型调用异常都立即返回
`llm_failed`，不与传输层的限流、5xx、超时退避重复叠加。

## 接入一个新评测器（4 步 checklist）

- [ ] **建包** `app/evaluators/<name>/agent.py`：
      `@agent_component(name="judge_<name>", ext_schema=<Name>ConfigSchema)` + 继承
      `BaseEvaluatorAgent`，实现 `from_config` + `evaluate`。对称前缀约定：judge 评测器
      统一用 `judge_<logic_name>`（见 config-wiring 设计文档）。ext_schema 只承载**对标准
      section 的引用**（`default_model` → `models.<key>`、`prompt` → 模板名）。
- [ ] **三处配置同步**（`conf/app.<env>.yaml`，daily / pre / prod 三份）：
      - `models.judge_<name>` —— 命名模型，`api_key_env` 指向环境变量名
      - `prompts.template_dir` —— **追加** `conf/agents/judge_<name>/prompts`（不要覆盖已有项）
      - `agents.judge_<name>` —— `type: judge_<name>` + ext 字段（`default_model` / `prompt`）

      密钥由 `api_key_env` 命名的环境变量运行时提供，**不入仓**。
      prompt 模板文件**必须**放在 `conf/agents/judge_<name>/prompts/<prompt_name>.prompt`
      （`<prompt_name>` 与 `agents.judge_<name>.prompt` 字段一致）。
- [ ] **注册** 在 [`registry.py`](./registry.py) 加一行 `# noqa: F401` import：模块加载即触发
      `@agent_component` 注册。**注意**：`registry.py` 同时 eager-import judge 评测器与
      配对的 attribution agent（两者共用此入口），新增 judge 时只加 judge import，
      attribution 的 import 由 [`../triage_attribution/registry.py`](../triage_attribution/registry.py)
      负责 —— 二者对称分离，不要混加。
- [ ] **（可选）triage 分流**：若需要按 score 分流派单，到
      [`app/triage_attribution/README.md`](../triage_attribution/README.md) 走 3 步接入
      classifier + `attribution_<name>`。
- [ ] **测试**：在 `tests/unit/evaluators/<name>/` 下加单测。最小模板见下方「测试模板」。

## 工具接入（可选）

**平台公共 Langfuse 取数（框架级，不依赖 `libs/tools`）**：YAML 顶层

```yaml
tools:
  providers:
    - trace
```

装配依赖 `app/services/trace_resolver/tools.py` 的 import 副作用
（`register_metadata_adapter("tool", …)` + `@tool_component`）以及
`main.py` 对该模块的 eager import。有 `AGENT_EVAL_LANGFUSE_INSTANCES` 时
`from_config` 才会把实例挂进 `tool_manager`。judge / attribution 直接
`runtime.tool_manager.get_provider("trace")`，**业务 agent 不用再声明**。
三个 tool：`fetch_trace_skeleton` / `fetch_observation_detail` / `find_observations`。
平台不会通过 HTTP envelope 传递整包 `raw_trace`。业务 agent 只凭 `trace_id + project_id`
按需调用 Provider；`judge_amap_agent` 会在 preprocess 阶段自动调用一次
`fetch_trace_skeleton` 定位最新 generation，再调用一次 `fetch_observation_detail` 读取该条完整
input。消息归一化会过滤 `role=system`、content 含规范 `<system-reminder>` 标签的伪 user
运行时提醒，以及无 `tool_call_id`、content 为 function/tool schema 定义的伪 tool 消息；真实
user 消息和 tool result 保持完整，最后再按总长度上限交给 tool-call judge。detail 失败或不可
解析时仍使用 skeleton observation preview 合成 assistant/tool 消息，不阻断评测。
详见 [`trace_resolver/README.md`](../services/trace_resolver/README.md)。
控制台创建/运行 Evaluator 见 [langfuse-integration.md](../../docs/langfuse-integration.md)。

业务自有工具两条路（任选）：

- **A（推荐，与 `trace` 同模式）**：自建 `BaseToolProvider` + `@tool_component`，
  YAML `tools.providers` 追加名字。tool metadata adapter 已由 `trace` 模块注册，
  业务文件**不必**再拷贝 `libs/tools`。
- **B**：走下方 `AgentTool` + `AgentToolProvider`（需依赖
  [`libs/tools`](../../../libs/tools/README.md)，并保证 `scan_namespaces` 含 `"tools"`）。

### 4 步接入

- [ ] **写工具**：在 `app/evaluators/<name>/tools/` 下写 `AgentTool` 子类，
      `__init__.py` 导出类（`ToolLoader` 用 `inspect.getmembers` 扫顶层类）：

      ```python
      from typing import ClassVar
      from agentgraph.tools import AgentTool
      from agentgraph.tools.content import ToolResult
      from agentgraph.tools.context import ToolExecutionContext
      from pydantic import BaseModel, Field

      class FooArgs(BaseModel):
          query: str = Field(..., description="查询关键词")

      class FooTool(AgentTool):
          name: ClassVar[str] = "foo"
          description: ClassVar[str] = "查询示例工具"
          input_schema: ClassVar[type[BaseModel]] = FooArgs
          owner: ClassVar[str] = "<team>"
          stage: ClassVar[str] = "private"
          since: ClassVar[str] = "2026-07-08"

          async def run(self, args: FooArgs, ctx: ToolExecutionContext) -> ToolResult:
              return ToolResult.from_text(f"result for {args.query}")
      ```

- [ ] **框架装配入口**：在 `app/evaluators/<name>/tools/providers.py` 写一行
      `@tool_component`。AppRuntime `walk_packages` 按模块名尾缀 `providers` 自动发现：

      ```python
      from agentgraph.tools.component_metadata import tool_component
      from tools.agent_tool_provider import AgentToolProvider, AgentToolPackagesConfig

      @tool_component(name="<name>_tools", ext_schema=AgentToolPackagesConfig)
      class <Name>ToolProvider(AgentToolProvider):
          pass
      ```

- [ ] **YAML 填 packages**（`conf/app.<env>.yaml` 三份对称，把空 `tools` section 填实）：

      ```yaml
      tools:
        providers: [<name>_tools]
        extensions:
          <name>_tools:
            packages: [benchmark.evaluators.<name>.tools]
            name: <name>
      ```

- [ ] **在 agent 里调用**：通过 `runtime.tool_manager`（AppRuntime 已装好）。

      ```python
      result = await runtime.tool_manager.call_tool("foo", {"query": "x"}, context=ctx)
      text = result.to_text()
      ```

      > 框架 `AgentToolProvider.call_tool` 默认构造基类 `ToolExecutionContext`；
      > 若工具需要扩展字段（如自定义上下文），调用时显式传 `context=<your_ctx>`，
      > 该实例会透传给 `AgentTool.run(args, ctx)`。

### 不要做的事

- ❌ 不要自建 runtime / 手写 `tool_manager.register_provider(...)` —— YAML +
  `@tool_component` 自动完成（路径 A 的自建 `BaseToolProvider` 可以，但须挂装饰器）
- ❌ 不要在工具里反向 import `benchmark.services.*` 的业务编排 —— 工具是叶子能力，
  编排由 agent 的 graph 节点做
- ❌ 路径 B 不要漏依赖 `libs/tools`，否则 `AgentToolProvider` / scan `"tools"` 无效

### 多评测器共用工具

如果工具要被多个评测器（或归因 agent）复用，把它提到
[`libs/tools/src/tools/common/`](../../../libs/tools/src/tools/common/)
（跨业务公共工具层），YAML 的 `packages` 加上 `tools.common`。

## 参考实现

照抄模板见 [`example/README.md`](./example/README.md)——3 节点 LangGraph
（`build_prompt` → `evaluate` → `response`）+ `EvaluatorLLM` seam + 纯函数 `scoring.py`，
可直接 copy 后改 name。

> 多维度评测有两条并存契约（平台都不消费，仅落库/透传）：
> 1. **扁平分项**（example）：`sub_score_axes` + LLM 输出 `sub_scores: dict[str, float]`
>    → `judge_json.sub_scores`。
> 2. **嵌套明细**（如复合 judge）：业务把分项/checkpoints 放 `EvaluatorResult.extra`
>    → `judge_json.extra`（空 dict 不写键）。
>
> **triage / attribution 仍按总分走**。详见 spec
> `docs/superpowers/specs/2026-07-15-multi-score-evaluator-design.md`。

## amap-agent 复合 Judge

[`amap_agent/`](./amap_agent/) 是面向 amap-agent 的复合 judge evaluator：

- 请求 `model=amap_agent`，服务层会选择 `judge_amap_agent`。
- daily/pre/prod 的 `models.judge_amap_agent` 均使用 OpenAI-compatible
  `gpt-5.5-0424-global`，通过 `https://ai-llm-gateway.amap.com/open_api/v1` 调用；
  evaluator 名称、逻辑模型键、专用密钥环境变量和 prompt/权重保持不变。
- 共享前置节点先规整输入并做高德地图 query 分类（场景 + easy/medium/hard）。
- 对生产请求结构，前置节点从 `request_body.user.adiu/uid` 和
  `request_body.geography.city_code/latitude/longitude` 恢复用户与环境字段；input 未显式提供
  `env_time` 时，使用 Kafka 入口透传的源 trace `trace_timestamp` 原值。显式 `env_time` 始终优先。
- core_completion 使用的历史对话优先从 `history`、`eval_params.chat_history`、
  `request_body.history` 读取；这些显式字段均缺失且 query 非空时，从最新 generation 的
  `trace_messages` 中提取当前 query 之前的非空 user/assistant 消息，并排除 system、tool
  与本轮 query。query 为空时因无法确认当前轮边界，不执行 trace history fallback；query
  非空但与 trace 消息未精确匹配时，将最新 generation 的最后一条 user 消息视为当前轮
  边界。core_completion checkpoint 生成和最终 judge 共用归一化后的 `history` 变量，
  保证多轮追问的检查点与评分上下文一致。`context_info` 同样支持顶层、`eval_params`、
  `request_body` 和 `request_body.eval_params` 四种输入位置。最新 generation 完整输入若包含
  高德上下文 reminder，则从带稳定上下文 intro/suffix 的最后一个 user reminder 中恢复模型
  实际可见的 `# key` 与值文本，优先写入既有 `context_info`；不会把规则、工具预算、运行控制
  reminder 或 `<system-reminder>` wrapper 带入 prompt。trace detail 缺失、被截断或格式不匹配
  时回退显式请求值。恢复结果是 prompt-effective 文本，不保证可逆为原始 `context_info` dict；
  规范化后的 `context_info` 统一注入 Core Completion checkpoint 生成和最终 judge 两个 prompt。
- 前置节点从 runtime 懒取框架级 `trace` Provider，以请求的 `trace_id + project_id`
  调用 `fetch_trace_skeleton` 定位 `start_time` 最新的 generation，并通过
  `fetch_observation_detail` 获取该 observation 的完整 messages；随后过滤 `role=system`、
  content 含规范 `<system-reminder>` 标签的伪 user 运行时提醒，以及无 `tool_call_id` 的
  function/tool schema 定义消息，保留真实 user/assistant/tool 执行链。
  `trace_messages` 会优先直接读取顶层 `generation_detail.input`，避免大型 skeleton 先耗尽
  通用递归扫描的 2,000 节点预算；只有 detail 缺失、截断或无结构化 messages 时才执行受限
  的通用扫描，因此无需随 trace 体积调大扫描上限。
  detail 不可用或没有结构化 messages 时，回退为 skeleton tool span 合成的 assistant/tool
  消息。上下文恢复只使用完整 generation detail，不从 skeleton preview 猜测残缺 reminder。
  最终 `trace_messages` 仍受 6,000 字符总上限约束；当前图不再执行 tool-call judge，
  这些消息只用于恢复 core_completion/expression 的 history 上下文，全程不拉整包 trace。
- core_completion 与 expression 的“Agent 回复”优先使用同一次
  `fetch_observation_detail` 返回的最新 generation 完整 `output`，从而保留 `<tool_card>` 等
  模型真实输出；detail 缺失、`body_truncated=true` 或 output 为空时回退 evaluator 请求原始
  `output`。query 分类仍使用原始 `output`，避免控制标签影响 category/difficulty。
- Provider 未配置、请求缺少标识、trace 不存在或 Langfuse 请求失败时，仅将 trace 证据
  降级为空，core_completion/expression 评测链仍继续执行。
- preprocess 后的 `resolve_dataset_prior` 节点会一次性解析 dataset
  `expected_output + metadata`。仅当 `expected_output` 非空、metadata 同时存在
  `category`/`difficulty` 且二者不同时为空时，直接复用 metadata 值并跳过分类
  LLM。metadata 值不做枚举校验，允许其中一个值为空；`sub_scenario`
  固定补 `other`。分类 LLM 的输出仍由独立严格模型按现有枚举校验。
- 分类成功后 fan-out 并行执行两个分项；分类失败会直接进入 `aggregate` 返回结构化错误，
  不再调度 judge 节点。
  - `core_completion`：优先复用非空
    `expected_output.checkpoints.core_completion`；缺失或空数组时，按《基础满足度 Judge
    Agent Prompt》V2 生成 checkpoints，再按同文档 V2 judge prompt 判断任务核心完成度，
    输出 `0|1|eval_error`。旧的 `checkpoints.completion` 不会作为该分项的兼容别名。
  - `expression`：不使用 checkpoints，判断表达质量，输出 `1~5`。
- `aggregate` 只做确定性聚合：core_completion 为 0 时总分直接为 0；为 1 时先把
  expression 从 `1~5` 线性归一化到 `0~1`，再按 `core_completion=0.5`、
  `expression=0.5` 加权。`eval_error` 作为评测失败返回，不生成总分。
- 分项明细、检查点、分类结果保存在 `EvaluatorResult.extra`；
  `query_category_source` 为 `metadata|llm_generated`，core_completion 的
  `checkpoint_source` 为 `expected_output|llm_generated`；明细中不再包含旧 completion 和
  tool_call。当前 HTTP response 仍只返回 `score/reasoning`。
- dataset 评测必须保证 `output` 是被测 Agent 的真实回答、`expected_output` 是检查点先验。
  若 output provider 仍选 `expected_output`，会把检查点 JSON 当作真实回答；这类数据应使用
  `agent_chat` 或其他能返回真实 Agent 输出的 provider。
- 多个并行 judge 同时失败时，错误信息会合并保留各节点来源。
- prompt 模板变量必须全部由 evaluator 提供；未知 `${var}` 会返回 prompt 配置错误。

配置入口：

```yaml
agents:
  judge_amap_agent:
    type: judge_amap_agent
    default_model: judge_amap_agent
    classify_prompt: classify_query
    core_completion_checkpoint_prompt: core_completion_checkpoints
    core_completion_prompt: core_completion
    expression_prompt: expression
    max_parallel_llm_calls: 3
    weights:
      core_completion: 0.5
      expression: 0.5
```

prompt 文件位于 `conf/agents/judge_amap_agent/prompts/*.prompt`，模板变量使用 `${var}`。

## 竞品 Pairwise 横评 Judge

[`competitor_pairwise/`](./competitor_pairwise/) 用于高德与一个竞品的品牌盲化横评：

- HTTP 请求使用 `model=competitor_pairwise`；服务层选择 `judge_competitor_pairwise`。
- `output.eval_mode` 必须为 `horizontal_compare`，`output.candidates` 必须恰有两个动态候选，
  且包含配置里的 `focus_candidate`（daily/pre/prod 当前均为 `gaode`）。
- 候选可为直接字符串、带 `answer` / `response_text` / `component.text` 的结构化对象，或
  Langfuse trace skeleton。Skeleton preview 不是全文：完整 JSON preview 可直接使用；截断时按
  最新 `GENERATION` observation id 调公共 `trace.fetch_observation_detail`，并由 Provider 校验
  observation 归属。绑定使用 `EvaluatorService.before_eval` 已解析并校验的请求级 Langfuse
  主键与 `project_id`，不信任候选载荷自带的 `primary_id`。detail 单字段限制为 3000 字符，
  避免长 input 挤占 8K body；任何
  `body_truncated` / output 截断标记都会返回 `evaluation_failed`，不会猜截断文本。
- Judge 的环境上下文会携带两个候选各自的请求时间，统一按北京时间格式化为
  `YYYY-MM-DD HH:MM:SS`。竞品候选优先读取自身 `generate_time_b`，缺失时使用请求级
  `trace_timestamp`；高德候选依次读取 input / metadata / output 中的显式请求时间、EagleEye
  `request_id` 内嵌毫秒时间戳、`dialogId` 末尾时间戳，最后才以 `output.create_time` 兜底。
  时间缺失或非法时传空字符串，不阻断评测。
- 公共环境字段优先读取 `request_body.env_location` / `request_body.env_city`（或 `city`）；
  洛书请求使用的 `request_body.location` 会作为经纬度，`request_body.address` 会作为城市上下文
  兜底传入 prompt。
- Prompt 不注入候选名：query/context + 排序候选名的稳定哈希决定 Round 1 方向（不依赖 JSON
  key 顺序，也不固定偏向高德），Round 2 交换 A/B 并行执行；请求时间与答案使用同一轮
  order 映射并同步换位。映射后的 winner 不一致时才以 Round 1 方向执行 Round 3，多数决，
  无多数则 tie。
- score 面向 `focus_candidate`：胜 `1.0`、tie `0.5`、负 `0.0`。`reasoning` 以
  `winner=<candidate|tie>` 开头；逐轮 order/winner/hard-fail 仅放 `EvaluatorResult.extra`，
  不通过 HTTP 返回，但会由 `after_eval` 写入 `judge_json.extra`。
- daily/pre/prod 均复用 `qwen3.7-max` judge，并使用
  `AGENT_EVAL_JUDGE_AMAP_AGENT_MODEL_API_KEY` 注入模型密钥。
- LLM 调用/JSON 解析失败分别返回 `llm_failed` / `llm_unparseable`；winner 与 hard_fail
  自相矛盾同样视为不可解析，绝不折算为语义 tie。任一并发轮次失败会取消并回收兄弟任务。
- 该分数表示相对胜负，不是绝对质量分；当前不注册 classifier，后台 triage 会以
  `no_classifier` 安全跳过。

信封核心结构（OpenAI message `content` 中必须是这段对象的**合法 JSON 字符串**，不能直接传
Python `dict` repr）：

```json
{
  "input": {
    "request_body": {
      "query": "这附近挑个公园，坐公交怎么走",
      "env_city": "武汉"
    }
  },
  "output": {
    "eval_mode": "horizontal_compare",
    "candidates": {
      "baidu": {
        "answer": "竞品回答",
        "generate_time_b": "1789092215409"
      },
      "gaode": {
        "business_trace_id": "<gaode-business-trace-id>",
        "project_id": "<gaode-project-id>",
        "input": {
          "viewer_context": {
            "request_id": "21c5594817863461138625379d0aaa"
          }
        },
        "output": {"create_time": 1786346122557},
        "spans": [
          {"id": "<generation-id>", "type": "GENERATION"}
        ]
      }
    }
  },
  "metadata": {
    "trace_id": "<gaode-business-trace-id>",
    "coordinate": "30.5928,114.3055"
  },
  "agent_name": "luoshu",
  "project_id": "<gaode-project-id>"
}
```

顶层 `project_id` 是 detail 读取的权威项目；候选内同名字段仅作兼容 fallback，示例中二者应一致。

配置入口：

```yaml
agents:
  judge_competitor_pairwise:
    type: judge_competitor_pairwise
    default_model: judge_competitor_pairwise
    prompt: pairwise
    focus_candidate: gaode
    max_parallel_llm_calls: 10
```

prompt 位于 `conf/agents/judge_competitor_pairwise/prompts/pairwise.prompt`；单测位于
`tests/unit/evaluators/competitor_pairwise/`。

## 记忆 Pairwise 消融 Judge

[`memory_pairwise/`](./memory_pairwise/) 是 memory-online-eval（记忆在线评测）的盲化
消融 judge，度量「带记忆 vs 不带记忆」的记忆增益。本评测直接继承
`BaseEvaluatorAgent`，包内自带 `llm.py` / `schema.py` / `scoring.py` / `config.py`，
**不依赖任何其它评测器**（判分口径是本评测核心资产，跨评测器共享会让别处改动直接改变
我们的分数语义）：

- HTTP 请求使用 `model=memory_pairwise`；服务层选择 `judge_memory_pairwise`。
- **信封契约（设计 II，与竞品横评不同）**：不使用 `output.candidates`，而是
  - `input.request_body` —— 源 trace 的原始请求体，评测器据此**内部回放**一次 mem-off；
  - `output` —— 线上原始答案，直接作为 mem-on 一路（**不回放**）；
  - `input.context_info` / `input.memory_info` —— 可选的线上快照；缺省时评测器按
    `metadata.trace_id` 从该 trace 的 `build_prompt` observation 现捞。
  - Langfuse 变量映射会把这些字段以**字符串化 JSON** 传入，API 层已做还原
    （见 `app/api/schema.py::EvalEnvelope`）。
- **mem-off 回放**在评测器内部完成（见 [`memory_pairwise/replay.py`](./memory_pairwise/replay.py)）：
  删 `user_profile` / `app_use_context` / `short_term_memory`、mock adiu 隔离真实用户、
  `agent_strategy=ablation_no_context_memory`，打 amap-agent `/api/v1/agents/<id>/chat`。
  回放耗时/状态记录在评测树的 `memory_pairwise.replay_mem_off` span。
- 判分：盲化换序两轮投票，映射后 winner 不一致才跑第三轮，多数决、无多数则 tie。
  按团队规范经**单节点 StateGraph**（节点名 `judge_pairwise`）执行——判分是纯顺序管道、
  无状态机分支，单节点即满足 agentgraph 装载与节点级观测要求；节点内异常经 state
  带出图外原样重抛，`EvaluatorError` → HTTP 错误码映射不受图层影响。
- `focus_candidate` 默认 `with_memory`：胜 `1.0`（记忆有增益）、tie `0.5`、负 `0.0`
  （记忆无益/有损）。`reasoning` 以 `winner=<candidate|tie>` 开头。
- judge 输出除 `winner` / `hard_fail` / `reason` 外，还有 **`memory_signal`**
  （`used` / `unused` / `none`）标注本题是否存在可利用的个性化信号，以及
  **`query_category`**（navigation / poi_search / info_query / recommendation /
  chitchat / other，分层聚合用；脏值归 other、缺失为 None，不影响 winner 解析），
  随 `extra.rounds` 落库；prompt 明确要求「无信号空间 → tie」「仅详尽度差异 → tie」
  以压制噪声胜负。
- `extra` 还携带 **`mem_on_answer` / `mem_off_answer`**（各截断 2000 字符）与
  聚合后的 `query_category`——这是下游归因的对照组证据面，别删。
- **分维 GSB（产品口径对齐）**：judge 按《Memory_专项评估标准_V3.0》§6.1 的 4 个
  L1 维度（general / personalization / memory_use / safety）各判 A/B/tie，跨轮多数决后落
  `extra.dimension_winners`；聚合侧（`aggregation.py`）输出四维 GSB 表，
  `gsb = win_rate - loss_rate` 即产品 §6.4 的 GSB 差值（= 2×(均值分-0.5)），目标 GSB > 0。
  维度是附带产出：脏值宽松归一、老记录缺失不报错，总分 1/0.5/0 契约不变。
- daily/pre/prod 的 judge 模型为 Anthropic `claude-sonnet-4-6`（走 ai-llm-gateway），
  密钥用**专用** `AGENT_EVAL_JUDGE_MEMORY_PAIRWISE_MODEL_API_KEY`（分账）。
- 该分数表示相对胜负（记忆增益方向），不是绝对质量分。**已接入 triage/attribution**：
  三个方向（含变好）都归因、单轮 LLM、永不派 Aone 工单，口径与例外说明见
  [`../triage_attribution/README.md`](../triage_attribution/README.md) 的
  「已接入业务：memory_pairwise」；批量读数的纯函数聚合见
  `memory_pairwise/aggregation.py`。

配置入口：

```yaml
agents:
  judge_memory_pairwise:
    type: judge_memory_pairwise
    default_model: judge_memory_pairwise
    prompt: memory_pairwise
    focus_candidate: with_memory
    max_parallel_llm_calls: 10
    # mem-off 回放目标
    amap_base_url: https://pre-c4-amap-agent.amap-inc.com
    amap_agent_id: ai_chat
    mock_adiu_prefix: memeval-
    # 非空时带 Aone 项目环境隔离标头，路由到部署了 ablation 分支的项目环境
    replay_route_label: ""
    replay_timeout_ms: 30000
```

prompt 位于 `conf/agents/judge_memory_pairwise/prompts/memory_pairwise.prompt`；单测位于
`tests/unit/evaluators/memory_pairwise/`。整体由 Langfuse LLM-as-a-Judge 按 trace 触发
（Run on = Traces）。

### 线上回放隔离

prod 的 mem-off 回放直接打线上 amap-agent（`online-c4`），副作用按三层隔离，
证据均在 amap-agent 代码内可查：

- **session 写入隔离**：回放请求带 `eval_params.enable=true`，amap-agent 侧解析为
  `request.eval_mode`（`app/services/ai_chat/request_processor.py`）；
  `history_handler._is_yace()` 对 eval_mode 恒真（`handlers/history_handler.py`），
  会话网关写入按 `from=sdm_offline` 落**离线域**（`app/core/agent_session_service.py`，
  对齐 Java `TppSessionStoreService`），同口径下 `extra_info_server.writeSource` 也写
  `sdm_offline`（线上流量为 `amap-agent`），离线 Memory 抽取按 writeSource 白名单直接摘掉回放
  流量；ai_chat_kit 写路径同样按 `_is_yace` 跳过——即回放不会写生产 session / 会话历史。
- **身份隔离**：`user.adiu` / `device.adiu` 换成 `memeval-` 前缀 mock 值并抹 `uid`，
  即使有残余写动作也归属 mock 身份，不污染真实用户画像与记忆库。
- **容量控制**：每条被评测 trace 只产生一次回放，QPS 上限 = Langfuse 评测采样量；
  上线按低采样起步、观察后放量。

## 测试模板

参考实现测试在 [`tests/unit/evaluators/example/`](../../tests/unit/evaluators/example/)：
- [`test_agent.py`](../../tests/unit/evaluators/example/test_agent.py) —— 端到端：注入 `_FakeLLM`
  seam + 真实 `PromptManager`，构造 `AgentRequest` 调 `agent.call()`，断言 `EvaluatorResult` / `EvaluatorError`
- [`test_scoring.py`](../../tests/unit/evaluators/example/test_scoring.py) —— 纯函数：LLM 文本 → `{score, reason}` 解析
- [`test_nodes.py`](../../tests/unit/evaluators/example/test_nodes.py) —— 单节点：build_prompt / evaluate / response

**最小单测骨架**（copy 后改 `<name>`）：

```python
import pytest
from agentgraph.agents.request import AgentRequest, ChatMessage
from benchmark.evaluators.base import EvaluatorRequest, EvaluatorResult, decode_eval_text, encode_eval_json
from benchmark.evaluators.<name>.agent import <Name>EvaluatorAgent


class _FakeLLM:
    async def complete(self, prompt: str) -> str:
        return '{"score": 7, "reason": "ok"}'


async def test_call_returns_score():
    agent = <Name>EvaluatorAgent(
        name="<name>", description=None,
        llm=_FakeLLM(), prompt_manager=<your_prompt_manager>, prompt_name="<prompt>",
    )
    eval_request = EvaluatorRequest(input={...}, output={...}, metadata={"trace_id": "t-1"}, trace_id="t-1")
    request = AgentRequest(
        traceid="t-1", session_id="s-1",
        messages=[ChatMessage(role="user", content=encode_eval_json(eval_request))],
    )
    resp = await agent.call(request)
    result = EvaluatorResult.model_validate_json(decode_eval_text(resp.content_parts))
    assert result.score == 7
```

> 跑测试：`cd services/agent-eval && .venv/bin/python -m pytest tests/unit/evaluators/<name>/ -v`


### 裁决与 HTTP 报告元数据

评测器只产出 `EvaluatorResult` 的 score/reason/extra/sub_scores，不产出报告 trace ID。
`EvaluatorService.evaluate()` 将同一裁决交给 after_eval 和调用方。
HTTP 响应顶层 `ChatCompletion.langfuse_trace_id` 由 HTTP 层从独立的报告捕获上下文填入，
不可将其加回评测器结果或从被测请求的 trace ID 代填。
