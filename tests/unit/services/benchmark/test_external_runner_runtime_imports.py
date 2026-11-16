"""Regression test for where external_runner's runtime-constructed classes are imported.

Background: a pre-env ieval_external run once failed because ``BenchmarkRunSpec`` was imported only inside
a TYPE_CHECKING block, and constructing it at runtime in ``initialize_and_submit`` raised
``NameError`` (run failed: external init failed). That path depends on real OSS /
Aone resources, and the unit test is carried by an injected fake, so it cannot be guarded at the call site — therefore
the module namespace asserts that the runtime-constructed class is bound (a TYPE_CHECKING import does not bind it).
"""

from benchmark.workers import external_runner as external_runner_module

#: Classes constructed at runtime (not in an annotation position) that were historically placed inside a TYPE_CHECKING block.
_RUNTIME_CONSTRUCTED_SPECS = ("BenchmarkRunSpec", "SubmitBatchSpec")


def test_runtime_constructed_specs_bound_in_module_namespace() -> None:
    for name in _RUNTIME_CONSTRUCTED_SPECS:
        obj = getattr(external_runner_module, name, None)
        assert callable(obj), (
            f"{name} 未绑定在 external_runner 模块命名空间——"
            "请在运行时 import 区导入（TYPE_CHECKING 块 import 的名字"
            "运行时构造会触发 NameError）"
        )
