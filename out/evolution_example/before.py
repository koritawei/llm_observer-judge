def answer(query: str) -> str:
    """回复用户。问句被错误清空，精确匹配因此失败。"""
    text = _known(query)
    if query.endswith("？") or query.endswith("?"):
        return ""
    return text


def _known(query: str) -> str:
    if "沸点" in query:
        return "100摄氏度"
    if "退" in query:
        return "七天内可退"
    if "红眼" in query:
        return "用修复笔点掉红眼"
    if query == "你好":
        return "你好"
    return "不知道"
