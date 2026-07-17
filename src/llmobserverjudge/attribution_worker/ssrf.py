"""Custom tool endpoint SSRF protection (design §8.2)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class ToolEndpointValidator:
    """Whitelist + private-IP block + DNS rebinding check."""

    BLOCKED_HOSTNAMES = {
        "metadata.google.internal",
        "metadata",
        "localhost",
    }
    BLOCKED_IPS = {
        "169.254.169.254",
        "100.100.100.200",
        "0.0.0.0",
    }

    def __init__(self, allowed_hosts: list[str] | None = None) -> None:
        self.allowed_hosts = set(allowed_hosts or [])

    def validate(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("https", "http"):
            raise ValueError(f"Invalid scheme: {parsed.scheme}")
        host = parsed.hostname or ""

        if host in self.BLOCKED_HOSTNAMES or host in self.BLOCKED_IPS:
            raise ValueError(f"Blocked host: {host}")
        if host == "localhost" or host.endswith(".localhost"):
            raise ValueError(f"Blocked host: {host}")

        if host not in self.allowed_hosts:
            raise ValueError(f"Host not in whitelist: {host}")

        try:
            resolved = socket.getaddrinfo(host, None)
        except socket.gaierror as e:
            raise ValueError(f"DNS resolution failed for: {host}") from e

        for _family, _, _, _, sockaddr in resolved:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                raise ValueError(f"Blocked resolved IP: {host} -> {ip_str}")
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
                mapped = ip.ipv4_mapped
                if (
                    mapped.is_private
                    or mapped.is_loopback
                    or mapped.is_link_local
                    or mapped.is_reserved
                ):
                    raise ValueError(f"Blocked IPv4-mapped IPv6: {host} -> {ip_str}")

    def get_safe_client_kwargs(self) -> dict:
        return {"follow_redirects": False}
