# Example 评测器 — 参考实现

照抄这个子包新建一个 evaluator：`name="judge_<logic_name>"`，3 节点 LangGraph
（`build_prompt` → `evaluate` → `response`），各职责单一、可独立替换/测试。

## 文件清单（照抄什么）

| 文件 | 职责 | 改什么 |
|---|---|---|
| [`agent.py`](./agent.py) | `@agent_component(name="judge_<name>")` + 继承 `BaseEvaluatorAgent`；构造期组装 LLM + prompt 句柄；`from_config` 读 ext_schema；`evaluate` 是唯一业务方法（进图前可按 `tool_check` 做名称硬检）；`_build_state_graph` 装配 3 节点 | 改 `name`、`ExampleEvaluatorConfigSchema`；case 宏走 `build_case_variables`，一般不必再硬编码 `_VAR_*` |
| [`checks_tool.py`](./checks_tool.py) | `ToolNameCheck` + `check_tool_names`：只比较 `used_tool` 名称，不解析参数 | 改 `required` / `forbidden` / `must_be_empty` 的失败文案时保持 `硬检失败：` 前缀 |
| [`llm.py`](./llm.py) | `EvaluatorLLM` 抽象 + `ChatModelLLM` **惰性**实现（lambda 包 `runtime.get_llm(...)`，缺 `api_key` 到请求期才 fail，不阻塞 bootstrap） | 默认不动；想换 LangChain 直接 call 模型时改 `ChatModelLLM` |
| [`schema.py`](./schema.py) | input/output payload 模型 + `build_case_variables`（按存在字段建宏袋）+ `render_template` | 改 payload 模型；扩展宏优先靠 input 字段 + prompt，不必改渲染器 |
| [`scoring.py`](./scoring.py) | 把 LLM 文本解析成 `{score, reason}`（纯函数，方便单测） | 改解析规则（正则/JSON 抽取/容错） |
| [`state.py`](./state.py) | StateGraph 共享状态 TypedDict（`ExampleState`）+ `RENDER_CONTEXT` / `RESULT` / `ERROR` / `ERROR_CODE` 常量 | 一般不动；新增字段就在 TypedDict 加 key |
| [`nodes/build_prompt.py`](./nodes/build_prompt.py) | 加载 prompt 模板 → 渲染变量 → 写回 state | 不动 |
| [`nodes/evaluate.py`](./nodes/evaluate.py) | 调 LLM、塞消息、加 `EvaluatorError` 类型 guard | 不动 |
| [`nodes/response.py`](./nodes/response.py) | 从 state 抽 `{score, reason}` | 不动 |

## 状态机的 4 个节点字段

| key | 谁写 | 谁读 | 含义 |
|---|---|---|---|
| `RENDER_CONTEXT` | 入口 | `build_prompt` | `{name: value}` 模板变量（随意增减） |
| (内部) | `build_prompt` | `evaluate` | 渲染后的完整 prompt（参见 `state.py`） |
| `RESULT` | `response` | `evaluate` 顶层 | `{score: float, reason: str}` 终态 |
| `ERROR` / `ERROR_CODE` | 任一节点 | 路由 → END / 顶层 `evaluate` | 短路信号；任一节点 set 了直接 END，`evaluate` 翻成 `EvaluatorError` |

任一节点 set `ERROR` 即短路到 END——这是 LLM 失败 / prompt 缺失的统一退出路径。
名称硬检发生在进图之前，不走这四个字段，见下文「硬检与隔离」。

## prompt 模板

见 [`conf/agents/judge_example/prompts/evaluator.prompt`](../../../conf/agents/judge_example/prompts/evaluator.prompt)。
默认占位符 `${query}` / `${answer}` / `${used_tool}` / `${expected_output}` 等由
`build_case_variables` + `build_prompt` 运行时变量填充；新增宏优先改 prompt 与
数据集字段，而不是改 agent 常量。

## 多维度评测 / 动态 Rubric

`example` 用**单次** LLM + 单 prompt 产出总分、分项、权重与归因门限：

1. YAML `agents.judge_example`：
   - `sub_score_axes` / `sub_score_scale`：分项轴与取值区间；
   - `dimension_weights`：各轴权重（注入 prompt，LLM 回写 `weights`）；
   - `attribution_threshold`：归因门限（注入 prompt，LLM 回写同名字段）；
2. Prompt 格式 B 要求一次返回
   `{"score", "sub_scores", "weights", "attribution_threshold", "reasoning"}`；
3. `EvaluatorResult.extra` 携带 `weights` / `attribution_threshold`（LLM 缺省时
   用 YAML 回填）；框架在 fused worker / runner 中：
   - 有合法 `weights`+`sub_scores` → **覆盖** `fused_score`；
   - 有 `attribution_threshold` → `fused < threshold` 触发归因；
   - **二者皆缺** → 完全回退静态 Rubric / 注册的 AttributionTriggerPolicy
     （自定义 Rubric 与门限不受影响）。
4. `ExampleRubric` 只挂 **1 个** holistic criterion，分项经 `stat_dimensions`
   展开——不要把各轴写成多个 Criterion。

若 rubric 仍保留多个同 `evaluator_ref` 的 Criterion，且 `sub_scores` 键与
`criterion.id` 对齐，fused worker 会自动合并为一次 `evaluate` 再按维分发
（见 `criterion_eval` / `dynamic_rubric`）。

## 数据集 CSV / item 契约（与 WebUI 字段要求对齐）

| 列 / 字段 | 必填 | 建议格式（JSON 字符串与 dict 等价） |
|---|---|---|
| `input` | 是 | `{"request_body":{"query":"<用户问题>"}}`；列别名 `query` / `question`；纯文本会归一到该形状 |
| `output` | 是 | `{"component":{"type":"markdown","text":"<回答>"},"response_text":"<回答>"}`；列别名 `answer` / `response_text`；纯文本会归一 |
| `expected_output` | 否 | **原样**抬升（JSON 文本→dict，纯文本保持 str）；列别名 `expected` / `reference` |
| `trace_id` | 否 | Langfuse trace；也认 `traceid` / `source_trace_id` |
| `project` | 否 | Langfuse project；也认 `project_id` |

`ExampleParseInterceptor` 在提交前解码 JSON 并归一 input/output；`expected_output` 只做 JSON 解码、不改写结构。`parse_input` / `parse_output` 再兜底一次，避免 `${query}` / `${answer}` 静默为空。

## 标准 case

### Prompt 宏（可扩展）

`build_case_variables` 按 **input/output/expected 中实际存在的字段** 建袋；
prompt 里的 `${宏}` 有值则替换，无对应字段则替换为空串（`render_template`）。
运行时宏（`axes` / `expected_output` 等）仍可由 `build_prompt` 注入哨兵。
扩展方式：在 input（含 `request_body`）增加字段，并在 prompt 写同名
`${字段}`——**不必**改 `agent.py` 硬编码。`variableMapping` →
`request.variables` 覆盖同名键。

语义别名（始终写入，兼容默认 `evaluator.prompt`）：

| 模板变量 | 来源 |
|---|---|
| `query` | `input.request_body.query` |
| `answer` | `output.component.text`（空则回退 `output.response_text`） |
| `used_tool` | `output.content_desc.used_tool`（`list[str]`，仅名称） |
| `response_text` | `output.response_text` |
| `input` / `output` | 整包 JSON 文本（有值才写） |
| `expected_output` | `request.expected_output`（有值才注入；纯文本原样、dict JSON 序列化）。默认文件 prompt 含 `${expected_output}`；无值时渲染为 `(未配置期望输出)`，judge 按 query+answer 意图打分；有值时按 answer 对其满足程度打分 |

示例：input 带 `request_body.scene_id`，prompt 写 `${scene_id}` → 自动替换。
定制上传白名单含 `*`，允许此类扩展宏通过校验。

约定工具名全文 `get_order`。每份 fixture 是一份独立的 `EvaluatorRequest` JSON，互不引用，也不带 expected。

| case | 轨迹 | `tool_check` | 硬检 |
|---|---|---|---|
| `existing_order` | query 含订单号（如「查询订单 202410220001 的状态」）；`used_tool=["get_order"]`；答文含订单字段（状态、金额） | `required=["get_order"]` | 通过，然后进 LLM |
| `missing_order_id` | 见下方两条轨迹 | `must_be_empty=true` | 由 `used_tool` 决定 |
| `unknown_order` | 调了 `get_order`；答文是编造口吻（给出具体签收、地址、金额） | `required=["get_order"]` | 名称通过。低分以及失败摘要「空结果仍编造」是评分语义；本 agent 不会自动写入该摘要 |

`missing_order_id` 同一场景（query 无订单号，例如「帮我查一下订单状态」）有两条轨迹：

- 正面：`missing_order_id_clarify.json`。`used_tool=[]`，答文是澄清（请提供订单号）。硬检通过，分数交给 LLM。
- 失败：`missing_order_id.json`。误调 `get_order`（`used_tool=["get_order"]`）。硬检失败，`score` 为 `0`，`reason` 以 `硬检失败：` 开头并写明出现的工具名。不调 LLM，不拉 trace。

`unknown_order` 的硬检只保证名称在 `required` 里。工具返回空、回答仍编造具体结论时，评分上应低分，失败摘要用「空结果仍编造」。这是评分语义，不是 `evaluate` 的返回值；名称硬检通过后仍进 LLM，由 LLM 打分。

## 硬检与隔离

`tool_check` 仅 `judge_example` 消费。其它 evaluator 必须忽略。缺省 `None` 与请求未带该字段时一致。HTTP 信封 `EvalEnvelope.tool_check` 会原样透传到 `EvaluatorRequest`，不进入必填校验。

有 `tool_check` 时，`evaluate` 在 `_fetch_trace_evidence` 和图之前：

1. `ToolNameCheck.model_validate`。形状不合法抛 `EvaluatorError`（HTTP 400，`error_type=invalid_request_body`），不调 LLM，不拉 trace。
2. `used_tool = parse_output(output).content_desc.used_tool`，再 `check_tool_names`。顺序：`must_be_empty` 且列表非空 → 失败；`required` 缺任一 → 失败；`forbidden` 出现任一 → 失败。只比名称。
3. 失败：`EvaluatorResult(score=0.0, reason=硬检原因)`，不进图。
4. 通过：继续原 variable / prompt / LLM 路径。`tool_check` 不写入 `RENDER_CONTEXT`。
5. 空对象（`required` / `forbidden` 均为空且 `must_be_empty` 为 false）视为无约束，硬检通过后进 LLM。

隔离：checks 不进 judge 渲染变量。`expected_output` 仅当请求带值时注入；无值时 `build_prompt` 兜底为 `(未配置期望输出)`。可观察轨迹：`used_tool` 与答文作为 `${used_tool}` / `${answer}` 渲染。fixture 之间不互相引用。

Judge prompt 优先级：合法 judge-stage bundle > `judge_prompt_text` > `prompt_text` > YAML 里的 prompt 名 + `PromptManager` 文件模板。judge prompt/skill ref 任一存在时走 bundle 解析，不再消费请求正文覆盖；通用字段中的 attribution ref 会被过滤，不会串联进 judge prompt。无 bundle 时，前两种正文覆盖任一非空，`BuildPromptNode` 跳过文件并直接渲染这段正文。

工具说明是 `agent.py` 里的 `_JUDGE_TOOL_PROMPT`，由 `BuildPromptNode` 在正文渲染后再拼。未传入则不拼。照抄派生时改这段文案，不要在节点里写死目录。

同一次请求里的 `attribution_prompt_text` 只给归因用，不进 judge 渲染变量。

注入位置是评测请求本身（HTTP 信封或 `EvaluatorRequest`）：

```json
{
  "input": {"request_body": {"query": "..."}},
  "output": {"component": {"text": "..."}},
  "judge_prompt_text": "打分模板正文，可含 ${query} / ${answer}",
  "attribution_prompt_text": "归因 system 正文",
  "tool_check": {"required": ["get_order"]}
}
```
