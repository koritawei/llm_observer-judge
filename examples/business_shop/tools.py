# -*- coding: utf-8 -*-
"""Store tool catalog.

Policy, shipping, price, and knowledge tools are all executable. The selection layer decides which one is used.
"""


POLICY = {
    "退": "不可退",
}

SHIPPING = {
    "默认": "满99包邮",
}

PRICE = {
    "鞋": "299元",
}

KNOWLEDGE = {
    "沸点": "100摄氏度",
}


def lookup_policy(query: str) -> str:
    """Match a return policy from the question."""

    return _find(POLICY, query)


def quote_shipping(query: str) -> str:
    """Return the default shipping rule."""

    del query
    return SHIPPING["默认"]


def lookup_price(query: str) -> str:
    """Return the list price from a product keyword."""

    return _find(PRICE, query)


def lookup_knowledge(query: str) -> str:
    """Match a knowledge entry from the question."""

    return _find(KNOWLEDGE, query)


def run_catalog(query: str) -> dict[str, str]:
    """Run every tool in a fixed order."""

    ordered = (
        ("policy", lookup_policy),
        ("shipping", quote_shipping),
        ("price", lookup_price),
        ("knowledge", lookup_knowledge),
    )
    return {name: tool(query) for name, tool in ordered}


def _find(table: dict[str, str], query: str) -> str:
    for key, value in table.items():
        if key in query:
            return value
    return "不知道"
