# Search v2 评测提示词

来源：用户提供的 `docs_local/search_agent评测/Search Agent 评测.md`（2026-10-14）。
按文档章节原文拆成 `classification`、`foundation`、`style`、`tools` 四份 judge
和 `format_spec`、`tool_spec` 两份规范；只移除了章节外层的 Markdown fence。
完整文本随服务打包，不依赖本地参考文档或在线链接。

编排层明确两处配套协议：分类器的导航/噪声映射为报告「其他」，风格使用文档
已有「通用」章节；工具日志不足以判分时返回总分 null 和「不可评估」原因。
风格只注入共通规则及所选章节，工具 judge 不接收最终回答，任何 judge 都不接收
其他 judge 的评分与理由。最终分由代码计算。

阶段分数范围：基础 0/1；风格 0–12；工具 0–3。基础 0 不短路后续诊断。
基础 1 且任一后续分数 null 时记评测失败，不能折算成业务低分。

`format_spec` 需保留原有的信息查询、找点、通用章节和排列顺序；分界或所选章节
缺失/为空时返回 `evaluator_prompt_not_found`（400）。完整 JSON 不符合判分 schema
时返回 `llm_schema_invalid`（422），避免无效重试；残缺或无法解析的输出返回
`llm_unparseable`（502）。模型以 `length` / `max_tokens` 结束且判分无效时仍按
502 处理，保留截断重试能力；上述失败均不转成正常评分。

完整接入与回放说明见 `docs/search-eval-v2.md`。
