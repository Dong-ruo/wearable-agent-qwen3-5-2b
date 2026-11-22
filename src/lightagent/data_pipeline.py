"""Explicit conversion of user-supplied PUBLIC demonstrations. No synthetic task generation."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

VERSION = "public-tools-v1"
LICENSES = {"xlam": "CC-BY-4.0", "toolace": "Apache-2.0"}
SOURCES = {"xlam": "Salesforce/xlam-function-calling-60k", "toolace": "Team-ACE/ToolACE"}
PUBLIC_SYSTEM = "Use the provided tools when appropriate. Ask for missing required information. Read actual tool results before making claims."


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def json_value(value):
    return json.loads(value) if isinstance(value, str) else copy.deepcopy(value)


def normalize_schema(value):
    if isinstance(value, list):
        return [normalize_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: normalize_schema(v) for k, v in value.items() if not (k == "required" and v is None)}
    aliases = {"dict": "object", "list": "array", "int": "integer", "float": "number", "bool": "boolean"}
    if isinstance(result.get("type"), str):
        result["type"] = aliases.get(result["type"], result["type"])
    return result


def normalize_tools(raw):
    tools, aliases, used = [], {}, set()
    for definition in json_value(raw):
        f = definition.get("function", definition)
        original = f["name"]
        name = re.sub(r"[^a-zA-Z0-9_]", "_", original).strip("_") or "tool_" + digest(original)[:12]
        if name[0].isdigit():
            name = "tool_" + name
        if name in used or original in aliases:
            raise ValueError("工具名称重复或规范化后碰撞")
        used.add(name)
        aliases[original] = name
        parameters = normalize_schema(f.get("parameters", {"type": "object", "properties": {}}))
        Draft202012Validator.check_schema(parameters)
        if parameters.get("type") != "object":
            raise ValueError("工具参数根节点必须为 object")
        tools.append({"type": "function", "function": {"name": name, "description": f.get("description", ""), "parameters": parameters}})
    return tools, aliases


def replace_names_outside_strings(text, aliases):
    output, i, quote, escaped = [], 0, None, False
    names = sorted(aliases, key=len, reverse=True)
    while i < len(text):
        ch = text[i]
        if quote:
            output.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            i += 1
        elif ch in "\"'":
            quote = ch
            output.append(ch)
            i += 1
        else:
            found = next((name for name in names if text.startswith(name, i) and
                          (i == 0 or not (text[i-1].isalnum() or text[i-1] == "_")) and
                          text[i+len(name):].lstrip().startswith("(")), None)
            if found:
                output.append(aliases[found])
                i += len(found)
            else:
                output.append(ch)
                i += 1
    return "".join(output)


def literal(node, depth=0):
    if depth > 20:
        raise ValueError("参数嵌套过深")
    if isinstance(node, ast.Constant) and type(node.value) in {str, int, float, bool, type(None)}:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant) and type(node.operand.value) in {int, float}:
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if isinstance(node, ast.List):
        return [literal(n, depth+1) for n in node.elts]
    if isinstance(node, ast.Dict):
        pairs = [(literal(k, depth+1), literal(v, depth+1)) for k, v in zip(node.keys, node.values)]
        if any(not isinstance(k, str) for k, _ in pairs) or len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("对象键必须是无重复的字符串")
        return dict(pairs)
    raise ValueError("调用参数仅允许 JSON 字面量，不执行表达式")


def parse_toolace_calls(text, aliases):
    if len(text) > 100_000:
        raise ValueError("调用文本过长")
    rewritten = replace_names_outside_strings(text, aliases)
    tree = ast.parse(rewritten, mode="eval").body
    if not isinstance(tree, ast.List) or not tree.elts:
        raise ValueError("工具调用必须为非空列表")
    calls = []
    for node in tree.elts:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in aliases.values() or node.args:
            raise ValueError("只允许已提供工具的具名参数调用")
        if any(kw.arg is None for kw in node.keywords) or len({k.arg for k in node.keywords}) != len(node.keywords):
            raise ValueError("不允许参数展开或重复参数")
        arguments = {kw.arg: literal(kw.value) for kw in node.keywords}
        json.dumps(arguments, allow_nan=False)
        calls.append({"type": "function", "function": {"name": node.func.id, "arguments": arguments}})
    return calls


def validate_record(record):
    tools = {t["function"]["name"]: t["function"]["parameters"] for t in record["tools"]}
    if len(tools) != len(record["tools"]):
        raise ValueError("重复工具")
    pending, seen = {}, set()
    assistant_count = 0
    for m in record["messages"]:
        role = m["role"]
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("不合法的消息角色")
        if role == "tool":
            identity = m.get("tool_call_id")
            if identity not in pending or m.get("name", pending[identity]) != pending[identity]:
                raise ValueError("调用与工具响应无法对应")
            del pending[identity]
        elif pending:
            raise ValueError("工具响应未完成就进入下一消息")
        if role == "assistant":
            assistant_count += 1
            if not m.get("content") and not m.get("tool_calls"):
                raise ValueError("空的 assistant 监督目标")
            for call in m.get("tool_calls", []):
                name, args = call["function"]["name"], call["function"]["arguments"]
                if name not in tools or not isinstance(args, dict):
                    raise ValueError("未知工具或参数不是对象")
                Draft202012Validator(tools[name]).validate(args)
                identity = call["id"]
                if identity in seen:
                    raise ValueError("重复调用 ID")
                seen.add(identity)
                pending[identity] = name
    if not assistant_count:
        raise ValueError("没有 assistant 监督目标")
    # A final assistant tool-call is a valid SFT target even without executable observations.
    if pending and not record["messages"][-1].get("tool_calls"):
        raise ValueError("悬空工具响应")
    return record


def convert_xlam(row):
    tools, aliases = normalize_tools(row["tools"])
    calls = []
    for index, answer in enumerate(json_value(row["answers"])):
        calls.append({"id": f"call_{index}", "type": "function", "function": {
            "name": aliases[answer["name"]], "arguments": json_value(answer["arguments"])}})
    if not calls:
        raise ValueError("xLAM 空调用缺少可用的自然语言监督目标")
    return validate_record({"tools": tools, "messages": [{"role": "system", "content": PUBLIC_SYSTEM},
        {"role": "user", "content": row["query"]}, {"role": "assistant", "content": "", "tool_calls": calls}]})


def convert_toolace(row):
    system = row["system"]
    marker = system.find("JSON format")
    start = system.find("[", max(0, marker))
    if start < 0:
        raise ValueError("无法定位 ToolACE 的 JSON 工具定义")
    raw, _ = json.JSONDecoder().raw_decode(system[start:])
    tools, aliases = normalize_tools(raw)
    messages, pending, next_id = [{"role": "system", "content": PUBLIC_SYSTEM}], [], 0
    for m in row["conversations"]:
        role, text = m["from"], m["value"]
        if role == "assistant" and text.strip().startswith("["):
            calls = parse_toolace_calls(text.strip(), aliases)
            for call in calls:
                call["id"] = f"call_{next_id}"
                next_id += 1
            if len({c["function"]["name"] for c in calls}) != len(calls):
                raise ValueError("同一批同名工具响应无法安全消歧")
            pending = calls
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        elif role == "tool":
            if not pending:
                raise ValueError("工具返回没有对应调用")
            try:
                values = json.loads(text)
            except json.JSONDecodeError:
                values = None
            if isinstance(values, list) and values and all(isinstance(v, dict) and "name" in v for v in values):
                if len(values) != len(pending):
                    raise ValueError("工具返回数量不匹配")
                outputs = {aliases.get(v["name"], v["name"]): v.get("results", v) for v in values}
                if len(outputs) != len(values):
                    raise ValueError("工具返回名称重复")
                for call in pending:
                    name = call["function"]["name"]
                    messages.append({"role": "tool", "name": name, "tool_call_id": call["id"], "content": json.dumps(outputs[name], ensure_ascii=False)})
            elif len(pending) == 1:
                call = pending[0]
                messages.append({"role": "tool", "name": call["function"]["name"], "tool_call_id": call["id"], "content": text})
            else:
                raise ValueError("多工具返回无法安全配对")
            pending = []
        elif role in {"user", "assistant"}:
            messages.append({"role": role, "content": text})
        else:
            raise ValueError("未知 ToolACE 角色")
    return validate_record({"tools": tools, "messages": messages})


def signature_group(record):
    # Group by USED schema signatures, ignoring literal values, wording and distractor tools.
    used = {c["function"]["name"] for m in record["messages"] for c in m.get("tool_calls", [])}
    def semantic(value):
        if isinstance(value, dict):
            return {k: semantic(v) for k, v in value.items() if k not in {"description", "title", "examples", "default"}}
        return [semantic(v) for v in value] if isinstance(value, list) else value
    signatures = [semantic(t["function"]) for t in record["tools"] if not used or t["function"]["name"] in used]
    return digest(sorted(signatures, key=lambda s: s["name"]))


def split_for(group, seed):
    value = int(digest([seed, group])[:8], 16) / 2**32
    return "train" if value < .8 else "validation" if value < .9 else "test"


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        prefix = handle.read(4096).lstrip()
        handle.seek(0)
        if prefix.startswith("["):
            yield from json.load(handle)
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def prepare(manifest: dict, output: Path, tokenizer=None):
    if not manifest.get("inputs") or manifest.get("max_records", 1) < 1:
        raise ValueError("必须提供公开输入且 max_records 必须为正数")
    accepted, rejected, seen, groups = [], [], set(), {}
    for entry in manifest["inputs"]:
        source, revision = entry["source"], entry["revision"]
        if source not in LICENSES or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError("仅接受 xlam/toolace 和固定的 40 位数据 revision")
        converter = convert_xlam if source == "xlam" else convert_toolace
        for index, row in enumerate(read_rows(entry["path"])):
            try:
                record = converter(row)
                fingerprint = digest(record)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                group = signature_group(record)
                groups[group] = split_for(group, manifest.get("seed", 42))
                length = None
                if tokenizer is not None:
                    length = len(tokenizer.apply_chat_template(record["messages"], tools=record["tools"], tokenize=True, enable_thinking=False, add_generation_prompt=False))
                    if length > manifest.get("max_length", 4096):
                        raise ValueError("轨迹超长：整条剔除，禁止中途截断工具协议")
                metadata = {"source": SOURCES[source], "license": LICENSES[source], "revision": revision,
                    "source_id": str(row.get("id", index)), "transform_version": VERSION, "fingerprint": fingerprint,
                    "group_id": group, "split": groups[group], "token_length": length}
                turns = sum(bool(m.get("tool_calls")) for m in record["messages"])
                metadata["task_kind"] = "no_call" if not turns else "multi_step" if turns > 1 else "call_with_feedback" if any(m["role"] == "tool" for m in record["messages"]) else "call_only"
                metadata["language_bucket"] = "zh" if any(re.search(r"[\u4e00-\u9fff]", m.get("content") or "") for m in record["messages"] if m["role"] == "user") else "other"
                accepted.append({**record, "metadata": metadata})
            except (ValueError, KeyError, TypeError, SyntaxError, RecursionError, AttributeError, SchemaError, ValidationError) as exc:
                rejected.append({"source": source, "row": index, "reason": str(exc)[:500]})
    # Stable round-robin across signature groups avoids taking a file's first N rows.
    buckets = defaultdict(list)
    for record in sorted(accepted, key=lambda r: r["metadata"]["fingerprint"]):
        meta = record["metadata"]
        buckets[(meta["group_id"], meta["task_kind"], meta["language_bucket"])].append(record)
    selected = []
    keys = sorted(buckets, key=lambda k: digest([manifest.get("seed", 42), k]))
    limit = manifest.get("max_records", len(accepted))
    offset = 0
    while len(selected) < min(limit, len(accepted)):
        for key in keys:
            if offset < len(buckets[key]) and len(selected) < limit:
                selected.append(buckets[key][offset])
        offset += 1
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in selected:
                if row["metadata"]["split"] == split:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = {"version": VERSION, "accepted_before_sampling": len(accepted), "selected": len(selected),
              "counts": dict(Counter(r["metadata"]["split"] for r in selected)), "rejected": rejected,
              "task_kinds": dict(Counter(r["metadata"]["task_kind"] for r in selected)),
              "sources": dict(Counter(r["metadata"]["source"] for r in selected)),
              "length_checked": tokenizer is not None, "manifest": manifest}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    from .console import utf8_console
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", help="本地 tokenizer 路径；不指定则不声称完成长度检查")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = prepare(json.loads(Path(args.manifest).read_text(encoding="utf-8-sig")), Path(args.output), tokenizer)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rejected", "manifest"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
