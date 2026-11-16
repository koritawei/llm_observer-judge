"""Regression for a unique streaming assembly in main.py.

Historical defect: ``_ls_init_benchmark_streaming`` was defined twice (the full version plus
an old version that did not take settings). Python's later definition overrode the earlier one — the old version did not pass settings,
so the sidecar never started (``runner has no settings``), and the lifespan
called it twice, initializing twice. This test locks:
1. The function is defined only once in the module, and the lifespan calls it only once;
2. The definition that takes effect passes ``settings`` when constructing the runner (the precondition for starting the sidecar).
"""

import ast
import inspect
import textwrap

import benchmark.service.main as main_module


def test_single_definition_and_single_call() -> None:
    source = inspect.getsource(main_module)
    tree = ast.parse(source)
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_ls_init_benchmark_streaming"
    ]
    assert len(definitions) == 1, (
        "_ls_init_benchmark_streaming 必须只定义一次"
        "（历史缺陷：旧版覆盖完整版导致 sidecar 永不启动）"
    )
    calls = source.count("await _ls_init_benchmark_streaming(state)")
    assert calls == 1, f"lifespan 只应调用一次（实际 {calls} 次）"


def test_effective_definition_passes_settings() -> None:
    """The definition that takes effect must pass settings when constructing StreamingBenchmarkRunner.

    The old version passed only evaluator_service → runner._settings=None →
    ``_start_sidecar`` returns immediately (sidecar ingest never starts).
    """
    source = textwrap.dedent(inspect.getsource(main_module._ls_init_benchmark_streaming))
    assert "settings=settings" in source, (
        "runner 构造必须注入 settings，否则 sidecar 消费永远无法启动"
    )
    assert "kafka_enabled=" in source, "kafka 开关由 settings 派生并显式传入"
    assert 'ensure_application_logging(reason="after_sidecar_ingest")' in source, (
        "sidecar MetaQ 启动后必须 reclaim 日志（平台化回归：否则无处理日志）"
    )


__all__ = []
