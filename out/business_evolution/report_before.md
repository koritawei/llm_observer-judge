# 基线 report

- run_id: `shop-before-20261113-d8495a`
- status: `completed`
- final_score: 0.16666666666666666
- kernel_report: `/Users/koritafei/Desktop/code/llmobserverjudge/var/reports/shop-before-20261113-d8495a.json`

| 用例 | 角色 | 意图 | 工具 | 执行结果 | 动作 | 输出 | 期望 | 分数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greet | good | 你好→问候 | none | 空 | greet | 你好 | 你好 | 1.0 |
| refund | bad | 退→人工 | none | 空 | wrong_intent | 请联系人工 | 七天内可退 | 0.0 |
| shipping | bad | 运费→物流 | policy | 不知道 | wrong_route | 不知道 | 满99包邮 | 0.0 |
| price | bad | 鞋→价格 | price | 299元 | use_result | 299元 | 199元 | 0.0 |
| boil | bad | 沸点→知识 | knowledge | 100摄氏度 | drop_result | 空 | 100摄氏度 | 0.0 |
| photo | heldout | 未知 | none | 空 | no_tool | 不知道 | 用修复笔点掉红眼 | 0.0 |
