"""Fit complete recent dialogue turns to the local Qwen input budget."""
from __future__ import annotations

import json
from functools import lru_cache

from wearable_tasks.contract import canonical

from .model_server import encode_chat


@lru_cache(maxsize=2)
def local_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def count_input_tokens(tokenizer, messages, tools):
    # ChatEndpoint serializes with sorted keys. The template can tokenize schema keys differently in that order.
    wire = json.loads(canonical({"messages": messages, "tools": tools}))
    return len(encode_chat(tokenizer, wire["messages"], wire["tools"]))


def project_tool_results(messages, row_limit, focus_ids, focus_text):
    projected = []
    for message in messages:
        key = {"list_events": "events", "search_activity": "activities"}.get(message.get("name"))
        if message["role"] != "tool" or key is None:
            projected.append(message)
            continue
        try:
            result = json.loads(message["content"])
        except (ValueError, TypeError):
            projected.append(message)
            continue
        rows = result.get("data", {}).get(key)
        if result.get("status") != "success" or not isinstance(rows, list) or len(rows) <= row_limit:
            projected.append(message)
            continue

        def relevance(row):
            if row.get("event_id") in focus_ids:
                return 2
            if any(row.get(field) and row[field] in focus_text for field in ("title", "activity_id", "start_time")):
                return 1
            return 0

        ordered = sorted(rows, key=relevance, reverse=True)
        pinned = min(4, sum(relevance(row) > 0 for row in ordered))
        selected = ordered[:max(row_limit, pinned)]
        if len(selected) == len(rows):
            projected.append(message)
            continue
        view = {**result, "data": {**result["data"], key: selected},
                "context_projection": {"total_rows": len(rows), "shown_rows": len(selected), "omitted_rows": len(rows) - len(selected)}}
        projected.append({**message, "content": canonical(view)})
    return projected


def fit_context(messages, tools, tokenizer_path, input_budget, focus_ids=(), focus_text=""):
    tokenizer = local_tokenizer(tokenizer_path)
    turn_starts = [i for i, message in enumerate(messages) if message["role"] == "user"]
    count = 0
    # Keep the active turn and at most two preceding turns. Only remove whole turns.
    for start in turn_starts[-3:]:
        candidate = messages[:1] + messages[start:]
        count = count_input_tokens(tokenizer, candidate, tools)
        if count <= input_budget:
            return candidate, count
    if not turn_starts:
        return None, count
    # The active turn itself does not fit. Keep its protocol pairs and only project list rows.
    active = messages[turn_starts[-1]:]
    notice = "\n工具结果若标有 context_projection，未显示的条目未知，不能声称结果完整。必要时请缩小查询范围。"
    for row_limit in (16, 8, 4, 1, 0):
        projected = project_tool_results(active, row_limit, focus_ids, focus_text)
        if projected == active:
            continue
        candidate = [{**messages[0], "content": messages[0]["content"] + notice}, *projected]
        count = count_input_tokens(tokenizer, candidate, tools)
        if count <= input_budget:
            return candidate, count
    return None, count
