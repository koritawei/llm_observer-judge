# MR12 本地测试执行记录

2026-11-04 复跑，661 passed / 3 skipped。工作目录为 services/agent-eval。新增 test_nodes.py、test_vui_agent_kafka_rubric.py 均包含在下列目录范围中，另覆盖共享评测契约、VUI schema/scoring/checkers、报告与 Benchmark 公共契约。

执行源码基于 1ff1bc56adc6219e844dea9b118b06e164590179，仅修改 build_prompt/evaluate 的 VUI 日志命名。最终复跑提交号记录在 MR 说明中，下列源码 SHA256 可核对对应内容。

```sh
export PATH=/Users/shulong/.nvm/versions/node/v24.14.0/bin:$PATH
PYTHONPATH=.:../../libs/data_models/py:../../libs/storage/src:../../libs /Users/shulong/code/ai/amap-ai-agent/services/agent-eval/.venv/bin/python -m pytest tests/unit/evaluators/business_components tests/unit/evaluators/vui_agent tests/unit/cluster_report tests/unit/services/benchmark/common tests/unit/services/benchmark/vui_agent -q
```

当前验证为本地执行。.aoneci 中尚无 agent-eval 服务单测配置。CI 补齐安排为后续 VUI evaluator/pipeline 接线单元中接入本服务单测，并补充实际 pipeline/run 链接。当前远端通用门禁的结果单独回读。

三项跳过为真实 PostgreSQL 环境用例。输出中含既有 Tortoise 弃用提示及 RemoteConfig shutdown 的 pulling_lock 告警。以下完整输出仅将本机路径统一为 $WORKSPACE。

```text
........................................................................ [ 10%]
........................................................................ [ 21%]
........................................................................ [ 32%]
....................................sss................................. [ 43%]
........................................................................ [ 54%]
........................................................................ [ 65%]
........................................................................ [ 75%]
........................................................................ [ 86%]
........................................................................ [ 97%]
................                                                         [100%]
=============================== warnings summary ===============================
../../../../../services/agent-eval/.venv/lib/python3.12/site-packages/tortoise/fields/data.py:831
  $WORKSPACE/services/agent-eval/.venv/lib/python3.12/site-packages/tortoise/fields/data.py:831: DeprecationWarning: `pk` is deprecated, please use `primary_key` instead
    super().__init__(**kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
661 passed, 3 skipped, 1 warning in 188.81s (0:03:08)
[shutdown][default] error during shutdown: 'RemoteConfigClient' object has no attribute 'pulling_lock'
Traceback (most recent call last):
  File "$WORKSPACE/services/agent-eval/.venv/lib/python3.12/site-packages/remote_config_client/client.py", line 891, in shutdown
    with self.pulling_lock:
         ^^^^^^^^^^^^^^^^^
AttributeError: 'RemoteConfigClient' object has no attribute 'pulling_lock'

```

```text
718aaf1b7136da944693399a0768303e91055def3db12534f1ef99163e801938  services/agent-eval/app/evaluators/vui_agent/nodes/__init__.py
e84245a78cfacdc6c5c5283da8a7d46d2df60f87213cec181d54ac9b5bfc5002  services/agent-eval/app/evaluators/vui_agent/nodes/build_eval_context.py
4142f4b381262f611fea2b620a9ecf1ecf13d2f88fd81c5aa4474f74ec27fdb4  services/agent-eval/app/evaluators/vui_agent/nodes/build_prompt.py
d45caac3fd9c707d943b787b0e697055145d6864293669cb4b27ae9ad46b0baa  services/agent-eval/app/evaluators/vui_agent/nodes/evaluate.py
fba9c0f4e70da43078bb72872e725493fd705abca3836ef88ca23dbc0e07d1c4  services/agent-eval/app/evaluators/vui_agent/nodes/route_judge.py
9e38b50e0888d60d96a63be1cb8cea74312eb40d5adc99013f22ca833084c077  services/agent-eval/app/evaluators/vui_agent/nodes/run_checkers.py
d2281a7266058a439dea6a82d5a43567953998e3fad33627aabcf3aee0f65cac  services/agent-eval/app/services/benchmark/common/rubric_classes.py
16e6a171b8229bcd624e599218430653a5bd27a6ed00ebde5dbce1dfd9549298  services/agent-eval/app/services/benchmark/vui_agent/__init__.py
51238069726adb86ccaef0e9b9a30f3729cafe75a4f0063913d70d140a395fcb  services/agent-eval/app/services/benchmark/vui_agent/rubric.py
0566dacd0e0738d5c0abb0fa1f27891b402d09c407ddd7dec10fd55e5c53ceab  services/agent-eval/tests/unit/evaluators/vui_agent/test_nodes.py
4452e26d6abc6977dbc347a1729105645fa980d72a051a37cfd0fa9d1ccec259  services/agent-eval/tests/unit/services/benchmark/vui_agent/test_vui_agent_kafka_rubric.py
```
