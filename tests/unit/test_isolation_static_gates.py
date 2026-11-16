"""Static gates for multi-instance isolation (design §12.5 D1/D2, always on in CI).

D1: evaluator packages forbid module-level mutable state — a module-level mutable literal may use only a constant
name (``__all__`` / UPPER_SNAKE / _UPPER_SNAKE) and must not be
written again or mutated in place anywhere in the file. State for each evaluation may travel only on the request,
and must not settle into a module-level writable variable (concurrent instances of the same code would cross-talk).

D2: a resolved prompt/model cache must not be keyed coarsely by component name — evaluator packages must not
use an ``lru_cache``/``cache`` decorator. A module-level cache variable must be registered on the allowlist below
with a note on where the key comes from (only a content key is allowed, such as a sha of the config content).
"""

import ast
from pathlib import Path

_EVAL_ROOT = Path(__file__).resolve().parents[2] / "app" / "evaluators"

#: D2 allowlist: module-level cache → where the key comes from. A new cache must be registered here first,
#: and the key must be a content key (a hash of config/prompt content). Component-name or instance-name keys are forbidden.
_ALLOWED_CACHES: dict[str, str] = {
    "app/evaluators/example/llm.py::_remote_config_llm_cache": (
        "key = RemoteConfigModelConfig 内容 sha256 前 16 位（内容键）"
    ),
}

_CONSTANT_NAME = ("__all__",)


def _is_constant_name(name: str) -> bool:
    return name in _CONSTANT_NAME or name.lstrip("_").isupper()


def _mutable_literal(node: ast.AST) -> bool:
    if isinstance(node, (ast.Dict, ast.List, ast.Set)):
        return True
    return isinstance(node, ast.Call) and getattr(node.func, "id", "") in {
        "dict",
        "list",
        "set",
        "OrderedDict",
    }


def _module_level_mutables(tree: ast.Module) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        if not _mutable_literal(node.value):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                found.append((target.id, node.lineno))
    return found


def _mutated_names(tree: ast.Module) -> set[str]:
    """Names written a second time or mutated in place anywhere in the file (subscript assignment, augmented assignment, in-place methods)."""
    mutated: set[str] = set()
    mutating_methods = {
        "append",
        "extend",
        "update",
        "pop",
        "popitem",
        "clear",
        "setdefault",
        "add",
        "discard",
        "remove",
        "move_to_end",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            mutated.add(node.target.id)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    mutated.add(target.value.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in mutating_methods and isinstance(node.func.value, ast.Name):
                mutated.add(node.func.value.id)
    return mutated


def _cache_decorators(tree: ast.Module) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = getattr(target, "id", "") or getattr(target, "attr", "")
            if name in {"lru_cache", "cache"}:
                lines.append(decorator.lineno)
    return lines


def _scan() -> tuple[list[str], list[str]]:
    d1_violations: list[str] = []
    d2_violations: list[str] = []
    for path in sorted(_EVAL_ROOT.rglob("*.py")):
        rel = f"app/evaluators/{path.relative_to(_EVAL_ROOT)}"
        tree = ast.parse(path.read_text())
        mutated = _mutated_names(tree)
        for name, lineno in _module_level_mutables(tree):
            key = f"{rel}::{name}"
            if key in _ALLOWED_CACHES:
                continue
            if "cache" in name.lower():
                d2_violations.append(f"{rel}:{lineno} 模块级缓存 {name} 未登记白名单（D2）")
                continue
            if not _is_constant_name(name):
                d1_violations.append(f"{rel}:{lineno} 模块级可变变量 {name} 非常量命名（D1）")
            elif name in mutated:
                d1_violations.append(f"{rel}:{lineno} 常量表 {name} 被二次写入/原地修改（D1）")
        for lineno in _cache_decorators(tree):
            d2_violations.append(f"{rel}:{lineno} 禁用 lru_cache/cache 装饰器（D2）")
    return d1_violations, d2_violations


def test_d1_no_module_level_mutable_state_in_evaluators() -> None:
    d1, _ = _scan()
    assert not d1, "\n".join(d1)


def test_d2_no_unregistered_caches_in_evaluators() -> None:
    _, d2 = _scan()
    assert not d2, "\n".join(d2)


def test_d2_allowlist_entries_still_exist() -> None:
    """The allowlist must not rot: a registered cache variable must really exist, and deleting it means deleting the registration too."""
    for key in _ALLOWED_CACHES:
        rel, name = key.split("::")
        path = Path(__file__).resolve().parents[2] / rel
        tree = ast.parse(path.read_text())
        names = {n for n, _ in _module_level_mutables(tree)}
        assert name in names, f"白名单缓存 {key} 已不存在，请同步移除登记"
