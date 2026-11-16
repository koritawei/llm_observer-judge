"""工具调用摘要。给代码归因的 trace_summary 用。"""

import json
import re

REASONING_CAP = 600  # 非最终轮 reasoning 截断长度
FINAL_REASONING_CAP = 3000  # 最终轮 reasoning 上限
LGI_BUDGET = 10000  # 生成上下文总输出上限
TOOL_PREVIEW_CAP = 200  # lgi中真实工具返回preview长度

TOOL_RESULT_CAP = 500  # 单个工具返回摘要上限
TOOL_CALLS_BUDGET = 8000  # 工具调用记录总输出上限
EMPTY_RESULT_MARKERS = ("未查询到工具返回可读文本结果", "无可读文本结果")


def _parse_json_lenient(raw: str):
    """宽容JSON解析：先strict=False，失败后去除控制字符再解析，仍失败返回None。"""
    try:
        return json.loads(raw, strict=False)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _json_decode_error_text(raw: str) -> str:
    """JSON解析失败的标准降级输出：原文头2000字符 + 尾2000字符 + 失败标记。"""
    if len(raw) <= 4000:
        return f"[JSON解析失败，原文如下]\n{raw}"
    return (
        raw[:2000]
        + f"\n[JSON解析失败，已省略中间{len(raw) - 4000}字符，仅展示头尾各2000字符]\n"
        + raw[-2000:]
    )


def _tool_result_text(call: dict) -> str:
    """从工具调用记录中提取可读返回文本（兼容嵌套output.content.text与扁平结构）。"""
    output = call.get("output", call.get("result", ""))
    text = ""
    if isinstance(output, dict):
        content = output.get("content", "")
        if isinstance(content, dict):
            t = content.get("text", "")
            text = t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
        elif isinstance(content, str):
            text = content
        elif content is not None:
            text = json.dumps(content, ensure_ascii=False)
        if not text:
            for k in ("text", "summary", "data"):
                v = output.get(k)
                if v:
                    text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    break
    elif isinstance(output, str):
        text = output
    elif output is not None:
        text = json.dumps(output, ensure_ascii=False)
    # 若text本身是JSON字符串，尝试解析并提取其中的 text/summary 字段
    stripped = text.strip()
    if stripped.startswith("{"):
        obj = _parse_json_lenient(stripped)
        if isinstance(obj, dict):
            inner = obj.get("text") or obj.get("summary") or obj.get("content")
            if isinstance(inner, str) and inner:
                text = inner
    return text


def _tool_type_extra_line(tool_name: str, text: str) -> str:
    """按工具类型提取一行结构化摘要（在通用摘要基础上追加）。"""
    if not text:
        return ""
    name = str(tool_name).lower()
    if "search_place" in name:
        blocks = re.split(r"POI\s*(\d+)[::\uff1a]\s*", text)
        # blocks: [前缀, 编号1, 内容1, 编号2, 内容2, ...]
        parts = []
        for k in range(1, len(blocks) - 1, 2):
            num, body = blocks[k], blocks[k + 1]
            pname = re.split(r"[|｜]", body)[0].strip()
            addr_m = re.search(r"地址[::\uff1a]\s*([^|｜\n]+)", body)
            dist_m = re.search(r"(?:到当前位置)?距离[::\uff1a]\s*([^|｜\n]+)", body)
            p = f"POI {num}: {pname}"
            if addr_m:
                p += f" | 地址: {addr_m.group(1).strip()}"
            if dist_m:
                p += f" | 距离: {dist_m.group(1).strip()}"
            parts.append(p)
            if len(parts) >= 3:
                break
        if parts:
            return "[POI结果] " + " || ".join(parts)
    elif "get_weather" in name:
        dates = re.findall(r"\d{1,2}月\d{1,2}日|\d{4}[-/]\d{1,2}[-/]\d{1,2}", text)
        if dates:
            uniq = list(dict.fromkeys(dates))
            rng = uniq[0] if len(uniq) == 1 else f"{uniq[0]}~{uniq[-1]}"
            return f"[天气覆盖日期] {rng} (共{len(uniq)}天)"
    elif "plan_route" in name:
        dist_m = re.search(r"(\d+(?:\.\d+)?\s*(?:公里|千米|米|km))", text)
        dur_m = re.search(r"(\d+(?:\.\d+)?\s*小时(?:\s*\d+\s*分钟)?|\d+(?:\.\d+)?\s*分钟)", text)
        if dist_m or dur_m:
            seg = []
            if dist_m:
                seg.append(f"距离: {dist_m.group(1)}")
            if dur_m:
                seg.append(f"耗时: {dur_m.group(1)}")
            return "[路线摘要] " + " | ".join(seg)
    elif "search_web" in name:
        title_m = re.search(r"标题[::\uff1a]\s*([^\n]{1,80})", text)
        if title_m:
            body = text[title_m.end() : title_m.end() + 200].strip()
            return f"[网页文章] 标题: {title_m.group(1).strip()} | 正文前200字: {body}"
        return f"[网页文章摘录] {text[:240]}"
    return ""


def _extract_tool_calls(raw: str) -> str:
    """从tool_calls_json提取结构化工具调用摘要。

    含：返回内容摘要（单条上限TOOL_RESULT_CAP=500字符）、按工具类型结构化提取、
    失败模式特征（空结果占位符统计/失败比例/连续同名重试）。
    总输出上限TOOL_CALLS_BUDGET=8000字符，超出时优先保留失败调用和首尾调用。
    """
    if not raw:
        return "(无记录)"
    calls = _parse_json_lenient(raw)
    if not isinstance(calls, list):
        return _json_decode_error_text(raw)
    if not calls:
        return "(无工具调用)"

    records = []  # (block, is_fail, tool_name, is_empty)
    for idx, call in enumerate(calls, 1):
        if isinstance(call, dict) and "_omitted_marker" in call:
            records.append((str(call["_omitted_marker"]), False, "", False))
            continue
        if not isinstance(call, dict):
            continue
        tool_name = str(call.get("name", call.get("tool_name", call.get("function", "unknown"))))
        args = call.get(
            "args", call.get("arguments", call.get("input", call.get("parameters", {})))
        )
        if isinstance(args, dict) and isinstance(args.get("arguments"), dict | str):
            args = args["arguments"]
        if isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)[:200]
        else:
            args_str = str(args)[:200]

        output = call.get("output", call.get("result", ""))
        out_success = None
        error_info = str(call.get("error", "") or "")
        if isinstance(output, dict):
            out_success = output.get("success")
            if output.get("error"):
                error_info = str(output["error"])
        if "success" in call:
            success = bool(call["success"])
        elif out_success is not None:
            success = bool(out_success)
        elif "status" in call:
            success = call.get("status") == "success"
        else:
            success = not bool(error_info)
        latency = call.get("latency_ms", call.get("latency", "?"))

        text = _tool_result_text(call)
        is_empty = bool(text) and any(mk in text for mk in EMPTY_RESULT_MARKERS)
        has_output = ("output" in call) or ("result" in call)

        if success:
            status_str = "成功"
        else:
            status_str = f"失败 | 错误: {error_info[:TOOL_RESULT_CAP]}" if error_info else "失败"
        lines = [
            f"工具{idx}: {tool_name} | 参数: {args_str} | {status_str} | 返回{len(text)}字符 | {latency}ms"
        ]
        if not success:
            # 失败调用：完整保留error与返回占位文本
            if text:
                lines.append(f"  返回内容: {text[:TOOL_RESULT_CAP]}")
        elif text:
            lines.append(f"  返回摘录: {text[:TOOL_RESULT_CAP]}")
            extra = _tool_type_extra_line(tool_name, text)
            if extra:
                lines.append(f"  {extra[:300]}")
        elif not has_output:
            lines.append("  [返回缺失] 仅记录了调用入参，无返回内容")
        records.append(("\n".join(lines), (not success), tool_name, is_empty))

    # 失败模式特征
    names = [r[2] for r in records if r[2]]
    total_calls = len(names)
    fail_count = sum(1 for r in records if r[1] and r[2])
    empty_count = sum(1 for r in records if r[3])
    retry_flags = []
    run_name, run_len = None, 0
    for name in names + [None]:
        if name == run_name:
            run_len += 1
        else:
            if run_name is not None and run_len >= 3:
                retry_flags.append(f"{run_name} 连续调用{run_len}次")
            run_name, run_len = name, 1

    feat_lines = ["[失败模式特征]"]
    if empty_count:
        feat_lines.append(f"- 空结果占位(未查询到工具返回可读文本结果)出现次数: {empty_count}")
    if total_calls:
        feat_lines.append(f"- 失败调用比例: {fail_count}/{total_calls}")
    for rf in retry_flags:
        feat_lines.append(f"- [重复重试] {rf}")
    if len(feat_lines) == 1:
        feat_lines.append("- 无")
    feature_text = "\n".join(feat_lines)

    # 预算控制：优先保留失败调用和首尾调用
    n = len(records)
    budget = TOOL_CALLS_BUDGET - len(feature_text) - 20
    priority = {i for i, r in enumerate(records) if r[1]}
    if n:
        priority.add(0)
        priority.add(n - 1)
    used = sum(len(records[i][0]) + 1 for i in priority)
    if used > budget:
        per = max(300, budget // max(len(priority), 1))
        blocks = [records[i][0][:per] for i in sorted(priority)]
        omitted = n - len(priority)
    else:
        kept = set(priority)
        for i in range(n):
            if i in kept:
                continue
            blk = records[i][0]
            if used + len(blk) + 1 <= budget:
                kept.add(i)
                used += len(blk) + 1
        omitted = n - len(kept)
        blocks = [records[i][0] for i in sorted(kept)]
    if omitted > 0:
        blocks.append(
            f"[因长度限制省略了{omitted}条调用详情(优先保留失败与首尾调用)，失败模式见下方特征总结]"
        )
    return "\n".join(blocks) + "\n" + feature_text


def extract_tool_calls(raw: str) -> str:
    """压缩 tool_calls_json。空输入返回占位文本。"""
    return _extract_tool_calls(raw)


def extract_llm_calls(raw: str) -> str:
    """压缩 llm_calls_json。空输入返回占位文本。"""
    return _extract_llm_calls(raw)


def extract_generation_context(raw: str) -> str:
    """压缩 last_generation_input。空输入返回占位文本。"""
    return _extract_generation_context(raw)


def build_trace_summary(
    *,
    tool_calls_json: str,
    llm_calls_json: str,
    last_generation_input: str,
    response_text: str,
) -> str:
    """代码归因看到的证据：工具调用、LLM 链路、生成上下文、最终回复。"""
    reply = response_text.strip() or "(无记录)"
    return "\n\n".join(
        [
            "### 工具调用记录\n" + extract_tool_calls(tool_calls_json),
            "### LLM调用链路\n" + extract_llm_calls(llm_calls_json),
            "### 生成上下文\n" + extract_generation_context(last_generation_input),
            "### 模型最终回复\n" + reply,
        ]
    )


def _extract_llm_calls(raw: str) -> str:
    """从 llm_calls_json 提取结构化摘要。"""
    if not raw:
        return "(无记录)"
    calls = _parse_json_lenient(raw)
    if not isinstance(calls, list):
        return _json_decode_error_text(raw)
    if not calls:
        return "(无LLM调用记录)"

    n = len(calls)
    round_tool_names: list[tuple[str, ...]] = []
    finals: list[bool] = []
    for call in calls:
        names: list[str] = []
        has_final = False
        if isinstance(call, dict) and "_omitted_marker" not in call:
            tools_called = call.get("tools_called", call.get("tool_calls", []))
            if isinstance(tools_called, list):
                for tool_call in tools_called:
                    if isinstance(tool_call, dict):
                        names.append(_tool_call_name(tool_call))
            has_final = bool(call.get("has_final_response", call.get("final_response")))
        round_tool_names.append(tuple(names))
        finals.append(has_final)

    retry_rounds: set[int] = set()
    index = 0
    while index < n:
        end = index
        while (
            end + 1 < n
            and round_tool_names[end]
            and round_tool_names[end] == round_tool_names[end + 1]
            and not finals[end]
            and not finals[end + 1]
        ):
            end += 1
        if end - index + 1 >= 3 and round_tool_names[index]:
            retry_rounds.update(range(index, end + 1))
        index = end + 1

    lines: list[str] = []
    for idx, call in enumerate(calls, 1):
        if isinstance(call, dict) and "_omitted_marker" in call:
            lines.append(str(call["_omitted_marker"]))
            continue
        if not isinstance(call, dict):
            continue
        has_final = finals[idx - 1]
        cap = FINAL_REASONING_CAP if (has_final or idx == n) else REASONING_CAP
        reasoning = str(call.get("reasoning", call.get("content", "")))
        reasoning_show = reasoning[:cap]
        if len(reasoning) > cap:
            reasoning_show += "..."
        tool_parts = _llm_tool_parts(call.get("tools_called", call.get("tool_calls", [])))
        latency = call.get("latency_ms", call.get("latency", "?"))
        line = f"第{idx}轮: [推理] {reasoning_show}"
        if tool_parts:
            line += f" [工具调用] {'; '.join(tool_parts)}"
        if has_final:
            line += " [最终回复] 已生成"
        line += f" [耗时] {latency}ms"
        if (idx - 1) in retry_rounds:
            line += " [疑似重复无效重试]"
        lines.append(line)
    return "\n".join(lines)


def _tool_call_name(tool_call: dict) -> str:
    function = tool_call.get("function")
    name = tool_call.get("name") or (
        function.get("name", "unknown")
        if isinstance(function, dict)
        else str(function or "unknown")
    )
    return str(name)


def _llm_tool_parts(tools_called: object) -> list[str]:
    if not isinstance(tools_called, list):
        return []
    parts: list[str] = []
    for tool_call in tools_called:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        args = tool_call.get(
            "args",
            tool_call.get(
                "arguments",
                function.get("arguments", "") if isinstance(function, dict) else "",
            ),
        )
        if isinstance(args, dict):
            args_text = ", ".join(f"{key}={value}" for key, value in list(args.items())[:3])
        else:
            args_text = str(args)[:80]
        parts.append(f"{_tool_call_name(tool_call)}({args_text})")
    return parts


def _is_tool_schema_message(message: object) -> bool:
    """识别工具 schema 消息：role=tool/function 但无 tool_call_id。"""
    if not isinstance(message, dict):
        return False
    if message.get("role") not in ("tool", "function"):
        return False
    if message.get("tool_call_id"):
        return False
    content = message.get("content", "")
    if isinstance(content, dict):
        if content.get("type") == "function" or isinstance(content.get("function"), dict):
            return True
        content = json.dumps(content, ensure_ascii=False)
    text = str(content).lstrip()
    if text.startswith('{"type": "function"') or text.startswith('{"type":"function"'):
        return True
    head = text[:2000]
    return '"function"' in head and ('"parameters"' in head or '"description"' in head)


def _extract_generation_context(raw: str) -> str:
    """从 last_generation_input 提取上下文概要。"""
    if not raw:
        return "(无记录)"
    messages = _parse_json_lenient(raw)
    if isinstance(messages, dict) and "messages" in messages:
        messages = messages["messages"]
    if not isinstance(messages, list):
        return _json_decode_error_text(raw)
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages = messages[1:]

    schema_count = 0
    kept: list[object] = []
    for message in messages:
        if _is_tool_schema_message(message):
            schema_count += 1
            continue
        kept.append(message)
    messages = kept

    header_lines = [f"[上下文概要] 共{len(messages)}条消息(已排除首条system prompt)"]
    if schema_count:
        header_lines.append(f"[已过滤{schema_count}条工具schema消息]")
    tool_sec = _generation_tool_section(messages)
    conv_sec = _generation_conversation(messages)
    last_sec = _generation_last_messages(messages)
    header = "\n".join(header_lines)
    tool_fit = tool_sec[:3500] + ("\n[...工具返回摘要已截断...]" if len(tool_sec) > 3500 else "")
    remain = LGI_BUDGET - len(header) - len(last_sec) - len(tool_fit) - 10
    if remain < 0:
        last_sec = last_sec[: max(LGI_BUDGET - len(header) - 10, 500)] + "\n[...已截断...]"
        tool_fit = ""
        conv_fit = ""
    else:
        conv_fit = conv_sec[:remain] + (
            "\n[...会话消息因长度限制截断...]" if len(conv_sec) > remain else ""
        )
    return "\n".join(part for part in (header, tool_fit, conv_fit, last_sec) if part)


def _generation_tool_section(messages: list[object]) -> str:
    tool_messages = [
        message
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in ("tool", "function")
        and message.get("tool_call_id")
    ]
    if not tool_messages:
        return ""
    lines = ["[真实工具返回摘要]"]
    for index, message in enumerate(tool_messages, 1):
        content = str(message.get("content", ""))
        lines.append(f'#{index}({len(content)}字): "{content[:TOOL_PREVIEW_CAP]}"')
    return "\n".join(lines)


def _generation_conversation(messages: list[object]) -> str:
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            continue
        content = str(message.get("content", "") or "")
        if not content.strip():
            content = _tool_call_placeholder(message.get("tool_calls"))
        if content.strip():
            lines.append(f"[{message.get('role')}]: {content}")
    if not lines:
        return ""
    return "\n".join(["[会话消息(user/assistant)]", *lines])


def _generation_last_messages(messages: list[object]) -> str:
    lines = ["[最后3条消息完整内容]"]
    last_messages = messages[-3:] if len(messages) >= 3 else messages
    if not last_messages:
        lines.append("(无消息)")
    for message in last_messages:
        if not isinstance(message, dict):
            lines.append(str(message))
            continue
        content = str(message.get("content", "") or "")
        if not content.strip() and message.get("tool_calls"):
            content = "(内容为空，含工具调用)"
        lines.append(f"[{message.get('role', '?')}]: {content}")
    return "\n".join(lines)


def _tool_call_placeholder(tool_calls: object) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    names: list[str] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            names.append(_tool_call_name(tool_call))
    if not names:
        return ""
    return f"(发起工具调用: {', '.join(names)})"
