# Example Triage + Attribution — 参考实现

照抄这个子包新建一组 triage + attribution 业务实现。两个东西：

- `ExampleTriageClassifier` —— 阈值法分级
- `AttributionAgent` —— `@agent_component(name="attribution_example")` 的归因 agent

## 子包布局

```
example/
├── classifier.py
├── constants.py
├── attribution/
│   ├── agent.py           # @agent_component — Langfuse tool ReAct（硬切主路径）
│   └── __init__.py
└── _internal/
    └── llm.py
```

## AttributionAgent

装配链：

```
AgentManager.get("attribution_<name>")
   │
   ▼
AttributionAgent.call(AgentRequest)
   │
   ├─ bind trace ToolProvider（metadata.trace_id / project_id）
   ├─ ReAct：LLM → tool_calls（skeleton/detail/find）或 final / human_review
   └─ AgentResponse(content=AttributionResult JSON)
```

平台 `TriageAttributionService` **零预拉**：metadata 只透传 `evaluator_request` / `evaluator_result` / `trace_id` / `project_id`，**无** `raw_trace`。

Tools（provider=`trace`）：

| name | 用途 |
|---|---|
| `fetch_trace_skeleton` | span 骨架（preview） |
| `fetch_observation_detail` | 单 observation 全文 |
| `find_observations` | 同 trace 内按 name/type 过滤 |

预算耗尽时：final pass 禁止再调 tool；若模型仍请求 tool，会**强制再归因一次**
（基于已有上下文出 `final`，略降 confidence），不再把 harness 文案写入 `root_cause`。

## 链路

`after_eval` → `ExampleTriageClassifier` → `attribution_example`。

## 分数带

阈值在 `classifier.py`，本说明只记录现有常量，不改分级：

| score | triage_level | 归因 |
|---|---|---|
| `>= 8` | `good` | 否 |
| `>= 6` | `watch` | 否 |
| `>= 3` | `mild_bad` | 是 |
| 否则 | `severe_bad` | 是 |

`missing_order_id` 若被误杀（轨迹已诚实澄清，judge 仍低分），靠归因
`corrected_triage_level=good` 推翻。不改上述阈值。

## 归因契约

归因 ReAct 的 trace 工具是 `fetch_trace_skeleton` / `fetch_observation_detail` /
`find_observations`。它们不是被测 agent 的业务工具（例如 `get_order`）。

| 场景 | 期望 |
|---|---|
| 空结果仍编造 | `category=tool` |
| 缺参诚实澄清被误杀 | `final`，且 `corrected_triage_level=good`，不要 `human_review` |
| 无 TOOL 证据 | `human_review` |

隔离：expected 的 `category` / `root_cause` 不进 ReAct 可见模板变量。fixture 里的
`expected` 只给测试断言，加载时剥掉，不写入 `metadata` 或 evidence prompt。

### 输出契约与解析降级

evidence prompt 的 Judge Failure Signal 以自然语言呈现（判分 / 判分器 / 判分理由），
末尾附 `## Output Contract`：`decision_type` 必需，`final` 必须带 `final` 且不得带
`tool_calls`，且禁止输出 `score` / `reason` 键——判分结论只是输入，不是输出模板。

LLM 输出未通过 `BadcaseAttributionDecision` 校验（典型形态：把 judge 的
`{"score", "reasoning"}` 抄回来、缺 `decision_type`）时，`AttributionAgent.call`
记 `attribution_agent.schema_fallback` 并降级为 `human_review`
（`needs_human_review=true`、`should_dispatch=false`、metadata `fallback=schema_validation`），
不再返回 FAILED——否则 `attribution_failed` 会被提升为 500，而 Kafka offset 已提前
提交，该条 trace 的归因永久缺失。

服务层 `_extract_attribution_result` 先读 `metadata.attribution_result`，再回退
`content_parts`；`attribution` 仍为空时，`TriageOutcome.error` 会带上
`status` / `error_code` / `error`，不再只有一句固定文案。

## prompt 优先级

归因 system 正文来自同一次评测请求的 `attribution_prompt_text`。`after_eval` 把整份
`EvaluatorRequest` 放进 `metadata.evaluator_request`，归因从这份快照读取。

优先级：

1. `evaluator_request.attribution_prompt_text` 非空字符串
2. `PromptManager.load_prompt(badcase_prompt_name)`
3. conf 文件 `conf/agents/attribution_example/prompts/{name}.prompt`

`judge_prompt_text` 和 `prompt_text` 是打分模板，归因忽略。请求没带
`attribution_prompt_text` 时走第 2、3 步。

上面只替换正文。工具说明在同目录 [`tool.prompt`](../../../conf/agents/attribution_example/prompts/tool.prompt)，不参与替换，正文选定后再拼上。

注入位置与 judge 相同，写在评测请求上：

```json
{
  "judge_prompt_text": "打分模板正文",
  "attribution_prompt_text": "归因 system 正文"
}
```
