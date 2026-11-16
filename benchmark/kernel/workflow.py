# coding: utf-8
"""Plugin libraries grouped by business, and the workflow that mounts them.

A plugin becomes executable once its alias, binding, and enabled entry are in conf.
``group`` on the alias decides which businesses can see it. The workflow page
asks for one plugin per pipeline slot and freezes that selection into a profile.
"""

from __future__ import annotations

import copy
from typing import Any

from benchmark.evolution.graph import EVOLUTION_NODES
from benchmark.kernel.confloader import Conf
from benchmark.kernel.errors import EvalError
from benchmark.kernel.pluginhost import REQUIRED_SLOTS

PHASE_SLOTS = ("request_builder", "execution")
_ALLOWED_SLOTS = set(REQUIRED_SLOTS) | {"stream_filter", *PHASE_SLOTS}

WORKFLOW_STAGES: tuple[dict[str, Any], ...] = (
    {"id": "build_context", "title": "构建上下文", "slots": ("source", "normalizer")},
    {"id": "resolve_profile", "title": "解析方案", "slots": ()},
    {"id": "collect_evidence", "title": "收集证据", "slots": ("request_builder",)},
    {"id": "run_checkers", "title": "检查", "slots": ("admission_policy",)},
    {"id": "plan_judges", "title": "规划评分", "slots": ("trigger_policy",)},
    {"id": "run_judges", "title": "执行评分", "slots": ("execution", "retry_policy", "capacity_policy")},
    {"id": "collect_metrics", "title": "汇总指标", "slots": ("fusion",)},
    {"id": "aggregate_results", "title": "聚合结果", "slots": ("aggregation",)},
    {"id": "cluster_results", "title": "结果分群", "slots": ("attribution",)},
    {
        "id": "emit_envelope",
        "title": "输出信封",
        "slots": ("report", "dispatch_policy", "completion_extra", "callback_authenticator"),
    },
)

# One row per EVOLUTION_NODES entry. next is the success edge; an empty next ends the step.
_EVOLUTION_META: dict[str, dict[str, str]] = {
    "s01_admit": {"title": "准入活动", "lane": "entry", "next": "s02_qualify", "when": "活动已受理"},
    "s02_qualify": {"title": "资格检查", "lane": "entry", "next": "", "when": "", "stop": "本步结束，等待冻结"},
    "s03_freeze": {"title": "冻结基线", "lane": "auto", "next": "s04_diagnose", "when": "基线已冻结"},
    "s04_diagnose": {
        "title": "诊断失败",
        "lane": "auto",
        "next": "s05_plan",
        "when": "失败可行动",
        "stop": "不可行动则停止",
        "slot": "attribution",
    },
    "s05_plan": {"title": "制定变更", "lane": "auto", "next": "s06_lease", "when": "计划已授权", "stop": "未授权则停止"},
    "s06_lease": {"title": "领取工件租约", "lane": "auto", "next": "s07_propose", "when": "租约已领取", "stop": "租约拒绝则停止"},
    "s07_propose": {"title": "提交差异", "lane": "auto", "next": "s08_seal", "when": "差异已提交", "stop": "差异为空则停止"},
    "s08_seal": {"title": "封印候选", "lane": "auto", "next": "s09_skill", "when": "候选已封印", "stop": "未封印则停止"},
    "s09_skill": {"title": "技能生命周期", "lane": "auto", "next": "s10_tool", "when": "技能已随候选记录", "plugin": "skill_exec"},
    "s10_tool": {"title": "工具生命周期", "lane": "auto", "next": "s11_fast_check", "when": "工具步骤已记录", "plugin": "tool_exec"},
    "s11_fast_check": {"title": "快速检查", "lane": "auto", "next": "s12_trial", "when": "检查已通过", "stop": "检查未通过则停止"},
    "s12_trial": {
        "title": "试跑评测",
        "lane": "auto",
        "next": "s13_decide",
        "when": "试跑已完成",
        "stop": "试跑未完成则停止",
        "uses_slot": "execution",
    },
    "s13_decide": {"title": "闸门裁决", "lane": "auto", "next": "s14_confirm", "when": "裁决为硬通过", "stop": "未通过则停止"},
    "s14_confirm": {"title": "留出确认", "lane": "auto", "next": "", "when": "", "stop": "确认后停止自动推进"},
    "s15_archive": {"title": "归档", "lane": "manual", "next": "", "when": "", "stop": "手动入口，执行后结束"},
    "s16_repairer": {"title": "修复提案", "lane": "manual", "next": "s17_meta", "when": "提案已记录"},
    "s17_meta": {"title": "元验证", "lane": "manual", "next": "", "when": "", "stop": "手动链结束"},
    "s18_release": {"title": "发布", "lane": "manual", "next": "", "when": "", "stop": "手动入口，执行后结束"},
    "s19_rollback": {"title": "回滚", "lane": "manual", "next": "", "when": "", "stop": "手动入口，执行后结束"},
    "s20_stop": {"title": "停止", "lane": "manual", "next": "", "when": "", "stop": "手动入口，执行后结束"},
}

SLOT_TITLES = {
    "source": "数据来源",
    "normalizer": "归一化",
    "admission_policy": "准入",
    "retry_policy": "重试",
    "capacity_policy": "容量",
    "trigger_policy": "触发",
    "attribution": "归因",
    "dispatch_policy": "分发",
    "fusion": "融合",
    "aggregation": "聚合",
    "completion_extra": "完成条件",
    "report": "报告",
    "callback_authenticator": "回调鉴权",
    "stream_filter": "实时过滤",
    "request_builder": "请求构造",
    "execution": "评分执行",
}


def groups_for(conf: Conf, business: str) -> set[str]:
    spec = (conf.groups.get("businesses") or {}).get(business)
    if not isinstance(spec, dict):
        return {"kernel"}
    names = {str(item) for item in spec.get("groups") or [] if item}
    return names or {"kernel"}


def plugin_record(conf: Conf, name: str) -> dict[str, Any]:
    for table in (conf.aliases_batch, conf.aliases_streaming):
        found = (table.get("aliases") or {}).get(name)
        if isinstance(found, dict):
            return found
    return {}


def plugin_group(conf: Conf, name: str) -> str:
    return str(plugin_record(conf, name).get("group") or "kernel")


def evolution_nodes(plugins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """S01–S20 in graph order, with the success edge and any plugin the step mounts."""

    by_name = {item["name"]: item for item in plugins}
    nodes = []
    for name in EVOLUTION_NODES:
        meta = dict(_EVOLUTION_META[name])
        if meta.get("slot"):
            meta["slot_title"] = SLOT_TITLES.get(meta["slot"], meta["slot"])
        pinned = meta.get("plugin")
        if pinned:
            found = by_name.get(pinned)
            meta["mounted"] = (
                {"name": found["name"], "title": found["title"], "group": found["group"]}
                if found
                else None
            )
        nodes.append({"id": name, **meta})
    return nodes


def plugin_catalog(conf: Conf, business: str) -> dict[str, Any]:
    """Plugins and pipeline slots the business is allowed to mount."""

    groups = groups_for(conf, business)
    plugins = []
    for kind, items in (conf.bindings.get("bindings") or {}).items():
        if not isinstance(items, dict):
            continue
        for name in items:
            if name not in conf.enabled or name in conf.disabled:
                continue
            if plugin_group(conf, name) not in groups:
                continue
            record = plugin_record(conf, name)
            plugins.append({
                "name": name,
                "kind": kind,
                "group": plugin_group(conf, name),
                "title": str(record.get("title") or name),
                "summary": str(record.get("summary") or ""),
            })
    plugins.sort(key=lambda item: (item["kind"], item["name"]))
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in plugins:
        by_kind.setdefault(item["kind"], []).append(item)
    base = conf.profile("benchmark_sync")
    defaults = dict(base.get("plugins") or {})
    phase = (base.get("execution_plan") or [{}])[0]
    defaults["request_builder"] = phase.get("request_builder")
    defaults["execution"] = phase.get("execution")
    stages = []
    for stage in WORKFLOW_STAGES:
        slots = []
        for slot in stage["slots"]:
            slots.append({
                "slot": slot,
                "title": SLOT_TITLES.get(slot, slot),
                "plugins": by_kind.get(slot, []),
            })
        stages.append({"id": stage["id"], "title": stage["title"], "slots": slots})
    return {
        "business": business,
        "groups": sorted(groups),
        "flows": [
            {"id": "benchmark", "title": "评测", "nodes": stages},
            {"id": "evolution", "title": "自进化", "nodes": evolution_nodes(plugins)},
        ],
        "businesses": sorted(conf.groups.get("businesses") or {}),
        "base_profile": "benchmark_sync",
        "defaults": defaults,
        "stages": stages,
        "plugins": plugins,
    }


def profile_ids_for(conf: Conf, business: str) -> list[str]:
    """Profile ids whose every plugin sits in a group this business can see."""

    found = []
    for profile_id in conf.profiles.get("profiles") or {}:
        try:
            compose_profile(conf, str(profile_id), business=business, plugins={})
        except EvalError as exc:
            if exc.code != "plugin_group":
                raise
            continue
        found.append(str(profile_id))
    return sorted(found)


def compose_profile(
    conf: Conf,
    base_id: str,
    *,
    business: str,
    plugins: dict[str, Any],
) -> dict[str, Any]:
    """Copy a conf profile and replace slots with plugins this business can see."""

    if not isinstance(plugins, dict):
        raise EvalError("workflow plugins must be a mapping")
    profile = copy.deepcopy(conf.profile(base_id))
    chosen = dict(profile.get("plugins") or {})
    phase_choices: dict[str, str] = {}
    for slot, name in plugins.items():
        if slot not in _ALLOWED_SLOTS:
            raise EvalError(f"unknown plugin slot {slot}")
        if not name:
            continue
        _require_visible(conf, business, slot, str(name))
        if slot in PHASE_SLOTS:
            phase_choices[slot] = str(name)
        else:
            chosen[slot] = str(name)
    for slot, name in chosen.items():
        _require_visible(conf, business, slot, str(name))
    source_plan = list(profile.get("execution_plan") or [])
    score_index = next((index for index, phase in enumerate(source_plan) if phase.get("role") == "score"), 0)
    plan = []
    for index, phase in enumerate(source_plan):
        phase = dict(phase)
        if index == score_index:
            phase.update(phase_choices)
        _require_visible(conf, business, "execution", str(phase.get("execution") or ""))
        _require_visible(conf, business, "request_builder", str(phase.get("request_builder") or ""))
        plan.append(phase)
    if not plan:
        raise EvalError("execution_plan is empty")
    profile["plugins"] = chosen
    profile["execution_plan"] = plan
    return profile


def _require_visible(conf: Conf, business: str, kind: str, name: str) -> None:
    if plugin_group(conf, name) not in groups_for(conf, business):
        raise EvalError(
            f"插件 {name} 不在业务 {business} 可见的组里",
            code="plugin_group",
        )
    conf.binding(kind, name)
