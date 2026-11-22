"""Strict model protocol and HTTP transport, independent of private tasks."""
from __future__ import annotations

import copy
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from wearable_tasks.contract import canonical

SLOTS = frozenset({"start_time", "end_time", "duration_minutes", "city", "date", "activity_id", "event_id", "minutes_before"})


class ProtocolError(ValueError):
    def __init__(self, message, diagnostic=None):
        super().__init__(message)
        self.diagnostic = diagnostic


class TransportError(RuntimeError):
    pass


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        return json.loads(text, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("Nonfinite JSON")))
    except (ValueError, TypeError) as exc:
        raise ProtocolError("Invalid JSON") from exc


def parse_message(message):
    """Eight native business tools; ask_user/final are control messages, not tools."""
    if not isinstance(message, dict):
        raise ProtocolError("Assistant message must be an object")
    if message.get("role", "assistant") != "assistant": raise ProtocolError("Completion role must be assistant")
    calls = message.get("tool_calls") or []
    if calls:
        if not isinstance(calls, list) or len(calls) != 1:
            raise ProtocolError("Exactly one tool call per turn; no partial parallel execution")
        try:
            f = calls[0]["function"]
            args = f["arguments"]
            args = strict_json(args) if isinstance(args, str) else args
            if not isinstance(f["name"], str) or not isinstance(args, dict):
                raise ProtocolError("Tool name/arguments have wrong types")
            if f["name"] == "final":
                # The local model may wrap the control response in its native tool envelope.
                value = {"type": "final", **args}
                if value["type"] != "final":
                    raise ProtocolError("Unexpected type in wrapped final response")
                if isinstance(value.get("result"), str):
                    value["result"] = strict_json(value["result"])
                return parse_message({"content": canonical(value)})
            canonical(args)
            return {"type": "tool_call", "name": f["name"], "arguments": copy.deepcopy(args)}
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("Malformed tool call") from exc
    text = message.get("content")
    if not isinstance(text, str) or len(text) > 65536:
        raise ProtocolError("Missing or oversized assistant content")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    if not text.startswith("{") and "<tool_call>" in text:
        raise ProtocolError("模型返回了未解析的工具调用标签")
    # Bounded question translation; structured control JSON remains the preferred protocol.
    if not text.startswith("{") and ("?" in text or "？" in text):
        slots = []
        if re.search(r"提前.*(?:分钟|多久)|提醒.*(?:分钟|多久)", text): slots.append("minutes_before")
        elif re.search(r"时长|持续|多少分钟|几分钟|多久", text): slots.append("duration_minutes")
        if re.search(r"开始时间|几点|何时|什么时候", text): slots.append("start_time")
        if re.search(r"城市|哪个地方", text): slots.append("city")
        if re.search(r"日期|哪天", text): slots.append("date")
        if slots:
            return {"type": "ask_user", "question": text, "requested_slots": slots}
    value = strict_json(text)
    if not isinstance(value, dict):
        raise ProtocolError("Control message must be an object")
    kind = value.get("type", "ask_user" if value.get("status") == "clarification" else "final")
    if kind == "ask_user":
        slots = value.get("requested_slots")
        question = value.get("question", value.get("message"))
        if (not isinstance(slots, list) or not slots or not all(isinstance(s, str) for s in slots) or len(set(slots)) != len(slots)
                or any(s not in SLOTS for s in slots)
                or not isinstance(question, str) or not question.strip()):
            raise ProtocolError("Clarification requires relevant supported slots and a question")
        return {"type": kind, "question": question, "requested_slots": slots}
    if kind != "final" or value.get("status") not in {"success", "failed"}:
        raise ProtocolError("Final status must be success or failed")
    if not isinstance(value.get("message"), str) or not isinstance(value.get("result"), dict):
        raise ProtocolError("Final message/result types are invalid")
    if set(value) - {"type", "status", "message", "result"}:
        raise ProtocolError("Unknown final fields")
    return {"type": "final", "status": value["status"], "message": value["message"], "result": value["result"]}


@dataclass
class Reply:
    message: dict
    usage: dict
    seconds: float


class ChatEndpoint:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        base = config["base_url"]
        from urllib.parse import urlsplit
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use a plain HTTP(S) base URL; credentials belong in an environment variable")
        self.url = base.rstrip("/") + "/chat/completions"

    def complete(self, messages, tools, seed):
        from time import perf_counter
        payload = {"model": self.config["model"], "messages": copy.deepcopy(messages),
                   "tools": copy.deepcopy(tools), "parallel_tool_calls": False,
                   "temperature": self.config.get("temperature", 0),
                   "max_tokens": self.config.get("max_tokens", 512), "stream": False, "seed": seed}
        if self.config.get("chat_template_kwargs"):
            payload["chat_template_kwargs"] = self.config["chat_template_kwargs"]
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.get("api_key_env", "MODEL_API_KEY"))
        if key: headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(self.url, data=canonical(payload).encode(), headers=headers)
        started = perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.config.get("timeout", 60)) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise TransportError("Model response exceeds two megabytes")
            data = strict_json(raw.decode("utf-8"))
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ProtocolError("Model response was truncated", {
                    "finish_reason": "length", "message": choice.get("message"), "usage": data.get("usage") or {}})
            message, usage = choice["message"], data.get("usage") or {}
            if not isinstance(usage, dict): raise ProtocolError("Invalid usage")
            return Reply(message, usage, perf_counter() - started)
        except urllib.error.HTTPError as exc:
            raise TransportError(f"Model HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise TransportError("Model connection failed or timed out") from None
        except (KeyError, IndexError, UnicodeError, TypeError) as exc:
            raise ProtocolError("Invalid completion envelope") from exc
