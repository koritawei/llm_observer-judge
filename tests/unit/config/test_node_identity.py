"""Unit tests for node identity matching used by pre node_slots."""

from benchmark.settings.node_identity import (
    NodeIdentity,
    host_ip_from_service_hostname,
    idc_from_hostname,
    match_node_identity,
    resolve_host_ips,
)


def test_idc_from_hostname_uses_last_label() -> None:
    assert idc_from_hostname("agent-eval-service033084102185.pre.na610") == "na610"
    assert idc_from_hostname("localhost") == ""


def test_host_ip_from_service_hostname_decodes_amap_label() -> None:
    assert (
        host_ip_from_service_hostname("agent-eval-service033084102185.pre.na610") == "33.84.102.185"
    )
    assert host_ip_from_service_hostname("localhost") is None


def test_resolve_host_ips_includes_encoded_service_ip() -> None:
    ips = resolve_host_ips("agent-eval-service033084102185.pre.na610")
    assert "33.84.102.185" in ips


def test_match_node_identity_requires_all_provided_fields() -> None:
    identity = NodeIdentity(
        hostname="agent-eval-service033084102185.pre.na610",
        host_ips=frozenset({"33.84.102.185"}),
        idc="na610",
    )
    assert match_node_identity(
        {
            "hostname": "agent-eval-service033084102185.pre.na610",
            "host_ip": "33.84.102.185",
            "idc": "na610",
        },
        identity,
    )
    assert not match_node_identity({}, identity)
    assert not match_node_identity(
        {
            "hostname": "agent-eval-service033084102185.pre.na610",
            "host_ip": "33.84.102.185",
            "idc": "ea119",
        },
        identity,
    )
