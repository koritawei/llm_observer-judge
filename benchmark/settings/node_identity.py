"""Current process node identity for YAML slot overrides.

Used by optional ``custom.bullmq.node_slots`` so a matching instance can take
a per-node slot profile; other nodes keep the generic slots.
"""

import re
import socket
from collections.abc import Mapping
from dataclasses import dataclass

_SERVICE_HOST_IP_RE = re.compile(r"(\d{12})$")
_LOOPBACK_PREFIX = "127."


@dataclass(frozen=True)
class NodeIdentity:
    """Hostname, local IPv4 set, and IDC label of the running process."""

    hostname: str
    host_ips: frozenset[str]
    idc: str


def idc_from_hostname(hostname: str) -> str:
    """IDC is the last DNS label (``….pre.na610`` → ``na610``)."""
    parts = hostname.strip().split(".")
    if len(parts) < 2:
        return ""
    return parts[-1]


def host_ip_from_service_hostname(hostname: str) -> str | None:
    """Decode Amap ``service0XX0XX0XX0XX`` hostname suffix into dotted IPv4."""
    label = hostname.split(".", 1)[0]
    matched = _SERVICE_HOST_IP_RE.search(label)
    if matched is None:
        return None
    digits = matched.group(1)
    octets = [str(int(digits[index : index + 3])) for index in range(0, 12, 3)]
    return ".".join(octets)


def resolve_host_ips(hostname: str) -> frozenset[str]:
    """Collect IPv4s from hostname encoding plus local name resolution."""
    ips: set[str] = set()
    encoded = host_ip_from_service_hostname(hostname)
    if encoded:
        ips.add(encoded)
    try:
        ips.add(socket.gethostbyname(hostname))
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return frozenset(ip for ip in ips if ip and not ip.startswith(_LOOPBACK_PREFIX))


def current_node_identity() -> NodeIdentity:
    """Identity of this process; never raises on resolution failure."""
    hostname = socket.gethostname()
    return NodeIdentity(
        hostname=hostname,
        host_ips=resolve_host_ips(hostname),
        idc=idc_from_hostname(hostname),
    )


def match_node_identity(match: Mapping[str, object], identity: NodeIdentity) -> bool:
    """Return True only when every provided match field equals ``identity``."""
    if not match:
        return False
    expected_hostname = match.get("hostname")
    if expected_hostname is not None and str(expected_hostname) != identity.hostname:
        return False
    expected_ip = match.get("host_ip")
    if expected_ip is not None and str(expected_ip) not in identity.host_ips:
        return False
    expected_idc = match.get("idc")
    if expected_idc is not None and str(expected_idc) != identity.idc:
        return False
    return True
