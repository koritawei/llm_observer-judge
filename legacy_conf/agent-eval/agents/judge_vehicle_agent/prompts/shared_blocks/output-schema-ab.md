## 输出要求

<output_format>
1. 严格要求 json 格式，不需要额外的 ```json``` 包裹。
2. 严格满足 json schema:
{
  "type": "object",
  "properties": {
    "A": {
      "type": "object",
      "properties": {
        "reasoning": {
          "type": "string",
          "description": "仅基于 A信息 的评判理由。必须先说明适用场景和预期输出形态，再按 D0-D6 中适用维度给出 pass/partial/fail/not_applicable 及证据。若存在扣分，必须对比列举 A信息 中 query 要求、实际输出、工具调用信息、已提供上下文或 DTO 标签中的差异点。"
        },
        "score": {
          "type": "number",
          "description": "A 的评分结果，取值 0、1 或 2。只输出单一总分，不输出 D0-D6 多维分。"
        }
      },
      "required": ["reasoning", "score"],
      "additionalProperties": false
    },
    "B": {
      "type": "object",
      "properties": {
        "reasoning": {
          "type": "string",
          "description": "仅基于 B信息 的评判理由。必须先说明适用场景和预期输出形态，再按 D0-D6 中适用维度给出 pass/partial/fail/not_applicable 及证据。若存在扣分，必须对比列举 B信息 中 query 要求、实际输出、工具调用信息、已提供上下文或 DTO 标签中的差异点。"
        },
        "score": {
          "type": "number",
          "description": "B 的评分结果，取值 0、1 或 2。只输出单一总分，不输出 D0-D6 多维分。"
        }
      },
      "required": ["reasoning", "score"],
      "additionalProperties": false
    }
  },
  "required": ["A", "B"],
  "additionalProperties": false
}
</output_format>
