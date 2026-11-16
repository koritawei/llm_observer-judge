from pathlib import Path

import pytest

from benchmark.kernel.confloader import load_conf
from benchmark.kernel.errors import EvalError
from benchmark.evolution.graph import EVOLUTION_NODES
from benchmark.kernel.workflow import WORKFLOW_STAGES, compose_profile, plugin_catalog, profile_ids_for

CONF = Path(__file__).resolve().parents[1] / "conf"


def test_default_business_does_not_see_shop_plugin():
    conf = load_conf(CONF, "local")
    catalog = plugin_catalog(conf, "default")
    names = {item["name"] for item in catalog["plugins"]}
    assert "shop_phrase_judge" not in names
    assert "exact_match_judge" in names
    assert "shop_phrase" not in profile_ids_for(conf, "default")


def test_shop_business_sees_its_plugin_and_profile():
    conf = load_conf(CONF, "local")
    catalog = plugin_catalog(conf, "shop")
    names = {item["name"] for item in catalog["plugins"]}
    assert "shop_phrase_judge" in names
    assert "kernel" in catalog["groups"]
    assert "shop" in catalog["groups"]
    judge = next(item for item in catalog["plugins"] if item["name"] == "shop_phrase_judge")
    assert judge["group"] == "shop"
    assert judge["kind"] == "execution"
    assert "shop_phrase" in profile_ids_for(conf, "shop")
    execution = next(stage for stage in catalog["stages"] if stage["id"] == "run_judges")
    slot_names = [slot["slot"] for slot in execution["slots"]]
    assert slot_names == ["execution", "retry_policy", "capacity_policy"]


def test_catalog_lists_every_benchmark_and_evolution_node():
    conf = load_conf(CONF, "local")
    catalog = plugin_catalog(conf, "shop")
    benchmark = next(item for item in catalog["flows"] if item["id"] == "benchmark")
    evolution = next(item for item in catalog["flows"] if item["id"] == "evolution")
    assert [node["id"] for node in benchmark["nodes"]] == [stage["id"] for stage in WORKFLOW_STAGES]
    assert [node["id"] for node in evolution["nodes"]] == list(EVOLUTION_NODES)
    trial = next(node for node in evolution["nodes"] if node["id"] == "s12_trial")
    assert trial["uses_slot"] == "execution"
    assert trial["next"] == "s13_decide"
    manual = [node["id"] for node in evolution["nodes"] if node["lane"] == "manual"]
    assert manual == ["s15_archive", "s16_repairer", "s17_meta", "s18_release", "s19_rollback", "s20_stop"]
    skill = next(node for node in evolution["nodes"] if node["id"] == "s09_skill")
    assert skill["mounted"]["name"] == "skill_exec"


def test_compose_rejects_plugin_outside_the_business_group():
    conf = load_conf(CONF, "local")
    with pytest.raises(EvalError) as raised:
        compose_profile(
            conf,
            "benchmark_sync",
            business="default",
            plugins={"execution": "shop_phrase_judge"},
        )
    assert raised.value.code == "plugin_group"


def test_compose_mounts_a_visible_plugin_on_the_score_phase():
    conf = load_conf(CONF, "local")
    profile = compose_profile(
        conf,
        "benchmark_sync",
        business="shop",
        plugins={"execution": "shop_phrase_judge"},
    )
    assert profile["execution_plan"][0]["execution"] == "shop_phrase_judge"
    assert profile["plugins"]["report"] == "report_metrics_only"
