"""Streaming HTTP inference profiling, separate from complete task runtime measurements."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


def profile(base_url, payload, timeout=60):
    payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("MODEL_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["MODEL_API_KEY"]
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
    start = time.perf_counter()
    first, usage = None, {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and any(delta.get(k) for k in ("content", "tool_calls", "reasoning_content")):
                    first = time.perf_counter() - start
    return {"ttft_http_seconds": first, "model_http_seconds": time.perf_counter() - start,
            "generated_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
            "peak_model_memory_bytes": None, "tool_execution_included": False,
            "note": "含网络和服务排队，不能称为纯 GPU 前向耗时；显存需在模型服务端单独采集"}


def main():
    from .console import utf8_console
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--payload", required=True, help="包含 model/messages 以及可选 tools 的 JSON 请求体")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = profile(args.base_url, json.loads(Path(args.payload).read_text(encoding="utf-8-sig")))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
