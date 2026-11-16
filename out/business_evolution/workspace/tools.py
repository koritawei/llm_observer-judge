# -*- coding: utf-8 -*-
"""门店工具目录。

政策、运费、价格、知识四件工具都可执行。选择层决定最终采用哪一件。
"""


POLICY = {
    "退": "七天内可退",
}

SHIPPING = {
    "默认": "满99包邮",
}

PRICE = {
    "鞋": "199元",
}

KNOWLEDGE = {
    "沸点": "100摄氏度",
}


def lookup_policy(query: str) -> str:
    """按问题匹配退货政策。"""

    return _find(POLICY, query)


def quote_shipping(query: str) -> str:
    """返回默认运费口径。"""

    del query
    return SHIPPING["默认"]


def lookup_price(query: str) -> str:
    """按商品关键词返回标价。"""

    return _find(PRICE, query)


def lookup_knowledge(query: str) -> str:
    """按问题匹配知识条目。"""

    return _find(KNOWLEDGE, query)


def run_catalog(query: str) -> dict[str, str]:
    """按固定顺序执行全部工具。"""

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
