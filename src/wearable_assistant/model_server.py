"""Small single-GPU Transformers inference endpoint. No weights loaded without --execute."""
from __future__ import annotations

import argparse
import copy
import hmac
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wearable_tasks.contract import canonical
from .protocol import ProtocolError, strict_json


def generation_stop_ids(tokenizer):
    # Qwen's model config can name endoftext while its chat template ends with im_end.
    # Explicit tokenizer EOS is authoritative for assistant turns; also accept padding/endoftext.
    values = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    result = list(dict.fromkeys(v for v in values if type(v) is int and v >= 0))
    if not result: raise ValueError("Tokenizer has no valid assistant stop token")
    return result


def normalize_messages(messages):
    result = copy.deepcopy(messages)
    if not isinstance(result, list) or not result: raise ValueError("messages must be a nonempty list")
    for message in result:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Invalid message role")
        for call in message.get("tool_calls", []):
            f = call["function"]
            if isinstance(f["arguments"], str): f["arguments"] = strict_json(f["arguments"])
            if not isinstance(f["arguments"], dict): raise ValueError("Tool arguments must be objects")
    return result


def parse_generated(text, tools):
    """Parse the pinned Qwen XML tool protocol (or JSON tool blocks), never eval."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "<tool_call>" not in clean:
        return {"role": "assistant", "content": clean}
    blocks = list(re.finditer(r"<tool_call>(.*?)</tool_call>", clean, re.S))
    if not blocks or clean[blocks[-1].end():].strip() or len(blocks) != clean.count("<tool_call>"):
        raise ProtocolError("Incomplete or trailing tool call output")
    calls = []
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    for i, block in enumerate(blocks):
        value = block[1].strip()
        if value.startswith("{"):
            obj = strict_json(value)
            name, arguments = obj["name"], obj["arguments"]
        else:
            f = re.fullmatch(r"<function=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</function>", value, re.S)
            if not f: raise ProtocolError("Malformed Qwen function")
            name, arguments = f[1], {}
            parameters = schemas.get(name, {}).get("properties", {})
            body, cursor = f[2], 0
            for p in re.finditer(r"<parameter=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</parameter>", body, re.S):
                if body[cursor:p.start()].strip() or p[1] in arguments: raise ProtocolError("Unexpected or duplicate Qwen parameter")
                kind = parameters.get(p[1], {}).get("type", "string")
                raw = p[2]
                # Qwen occasionally emits Python-style booleans inside XML parameter tags.
                if kind == "boolean" and raw in {"True", "False"}:
                    arguments[p[1]] = raw == "True"
                else:
                    arguments[p[1]] = raw if kind == "string" else strict_json(raw)
                cursor = p.end()
            if body[cursor:].strip(): raise ProtocolError("Unparsed Qwen parameter text")
        if not isinstance(name, str) or not isinstance(arguments, dict): raise ProtocolError("Invalid generated tool call")
        calls.append({"id": f"model_call_{i+1:04d}", "type": "function", "function": {"name": name, "arguments": canonical(arguments)}})
    return {"role": "assistant", "content": clean[:blocks[0].start()].strip(), "tool_calls": calls}


def encode_chat(tokenizer, messages, tools, return_tensors=None):
    return tokenizer.apply_chat_template(normalize_messages(messages), tools=tools or None, enable_thinking=False,
                                         tokenize=True, add_generation_prompt=True, return_tensors=return_tensors,
                                         return_dict=False)


class LocalModel:
    def __init__(self, args):
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForMultimodalLM
        if not torch.cuda.is_available(): raise ValueError("Local inference requires CUDA; no silent CPU fallback")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("Native BF16 unavailable; Pascal can explicitly use --dtype float32 --nf4 --attention eager")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        mc = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        loader = AutoModelForMultimodalLM if mc.model_type == "qwen3_5" else AutoModelForCausalLM
        kwargs = {"dtype": getattr(torch, args.dtype), "device_map": {"": 0}, "local_files_only": True, "attn_implementation": args.attention}
        if args.nf4:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=getattr(torch, args.dtype))
        self.model = loader.from_pretrained(args.model_path, **kwargs).eval()
        self.model_id, self.max_context = args.served_model_name, args.max_context
        self.stop_ids = generation_stop_ids(self.tokenizer)
        self.lock = threading.Lock()

    def complete(self, payload):
        import torch
        if payload.get("model") != self.model_id: raise ValueError("Unknown served model name")
        if payload.get("stream"): raise ValueError("Only non-streaming requests are supported")
        limit = payload.get("max_tokens", 512)
        temperature = payload.get("temperature", 0)
        if type(limit) is not int or not 1 <= limit <= 2048 or type(temperature) not in {int, float} or not 0 <= temperature <= 2:
            raise ValueError("Invalid generation budget")
        messages, tools = payload["messages"], payload.get("tools", [])
        with self.lock:
            ids = encode_chat(self.tokenizer, messages, tools, return_tensors="pt")
            # Explicitly prohibit truncating the conversation/tool schema to fit.
            if ids.shape[-1] + limit > self.max_context: raise ValueError("Full prompt plus completion exceeds max-context; no truncation")
            ids = ids.to("cuda")
            seed = payload.get("seed", 42)
            if type(seed) is not int: raise ValueError("seed must be an integer")
            torch.manual_seed(seed)
            with torch.inference_mode():
                generated = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=limit,
                    do_sample=temperature > 0, **({"temperature": temperature} if temperature > 0 else {}),
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.stop_ids, use_cache=True)
            new_ids = generated[0, ids.shape[-1]:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
            for token in {self.tokenizer.eos_token, self.tokenizer.pad_token} - {None}:
                while text.rstrip().endswith(token): text = text.rstrip()[:-len(token)]
            try: message = parse_generated(text, tools)
            except (ValueError, KeyError, TypeError): message = {"role": "assistant", "content": text}
            # Preserve literal XML tags; skip_special_tokens can remove only actual special tokens.
            length = len(new_ids)
            ended = length > 0 and int(new_ids[-1]) in self.stop_ids
            return {"model": self.model_id, "choices": [{"index": 0, "message": message,
                "finish_reason": "length" if length >= limit and not ended else "tool_calls" if message.get("tool_calls") else "stop"}],
                "usage": {"prompt_tokens": ids.shape[-1], "completion_tokens": length, "total_tokens": ids.shape[-1] + length}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="models/Qwen3.5-2B")
    p.add_argument("--served-model-name", default="Qwen/Qwen3.5-2B")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    p.add_argument("--nf4", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if not args.execute:
        print(json.dumps({"executed": False, "training": False, "model_path": args.model_path, "dtype": args.dtype,
                          "nf4": args.nf4, "max_context": args.max_context})); return
    token = os.environ.get("MODEL_API_KEY")
    if args.host not in {"127.0.0.1", "localhost"} and not token: raise ValueError("External model binding requires MODEL_API_KEY")
    model = LocalModel(args)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def reply(self, code, data):
            body = canonical(data).encode(); self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def allowed(self):
            from urllib.parse import urlsplit
            if self.headers.get("Origin") and urlsplit(self.headers["Origin"]).netloc != self.headers.get("Host"):
                self.reply(403, {"error": "Cross-origin request refused"}); return False
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "Unauthorized"}); return False
            return True
        def do_GET(self):
            if self.allowed():
                self.reply(200, {"data": [{"id": model.model_id, "object": "model", "eos_token_ids": model.stop_ids, "max_context": model.max_context}]} if self.path == "/v1/models" else {"status": "ok", "eos_token_ids": model.stop_ids})
        def do_POST(self):
            if not self.allowed(): return
            if self.path != "/v1/chat/completions": self.reply(404, {"error": "Not found"}); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 2_000_000: raise ValueError("Request size exceeds limit")
                payload = strict_json(self.rfile.read(size).decode("utf-8"))
                self.reply(200, model.complete(payload))
            except (ValueError, KeyError, TypeError, UnicodeError): self.reply(400, {"error": "Invalid generation request or full context exceeds configured budget"})
            except Exception:
                import traceback
                traceback.print_exc()
                self.reply(503, {"error": "Inference failed; check local model-server stderr/resources"})
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Local inference http://{args.host}:{args.port}/v1; {model.model_id}; EOS={model.stop_ids}; no training", flush=True)
    server.serve_forever()


if __name__ == "__main__": main()
