"""Traceable SFT preparation. Commands are explicit; importing does not process data."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import unicodedata
import uuid
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError, ValidationError

from .data_pipeline import PUBLIC_SYSTEM, literal, normalize_schema, read_rows, replace_names_outside_strings

VERSION = "unified-sft-v1"
BEHAVIORS = ("selection", "feedback", "clarification", "no_call")
SPLITS = ("train", "validation", "test_id", "test_composition", "test_robustness")
SELF_SPLITS = {"train": "train", "val": "validation", "test_id": "test_id",
               "test_composition": "test_composition", "test_robustness": "test_robustness"}
PUBLIC_SOURCES = {"hermes_multi", "hermes_single", "glaive", "toolace"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate_object_key:" + key)
        out[key] = value
    return out


def safe_value(value):
    """JSON first, then a bounded AST literal whitelist; never execute expressions."""
    if not isinstance(value, str):
        canonical(value)
        return copy.deepcopy(value)
    value = re.sub(r"^(?:\s|\\n|\\r)+|(?:\s|\\n|\\r)+$", "", value)
    if len(value) > 1_000_000:
        raise ValueError("literal_too_large")
    try:
        result = json.loads(value, object_pairs_hook=unique_object,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite_json:" + x)))
    except json.JSONDecodeError:
        tree = ast.parse(value, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 50_000:
            raise ValueError("literal_too_many_nodes")
        result = literal(tree.body)
    canonical(result)
    return result


def tools_from(raw):
    raw = safe_value(raw)
    if not isinstance(raw, list):
        raise ValueError("tools_not_list")
    tools, aliases, used = [], {}, set()
    for item in raw:
        f = item.get("function", item)
        old = f["name"]
        if not isinstance(old, str) or not old:
            raise ValueError("invalid_tool_name")
        name = re.sub(r"[^A-Za-z0-9_]", "_", old).strip("_") or "tool_" + digest(old)[:12]
        if name[0].isdigit():
            name = "tool_" + name
        if len(name) > 64:
            name = name[:47] + "_" + digest(old)[:16]
        if name in used or old in aliases:
            raise ValueError("tool_name_collision")
        used.add(name)
        aliases[old] = name
        schema = normalize_schema(f.get("parameters", {"type": "object", "properties": {}}))
        if schema.get("type") != "object":
            raise ValueError("parameters_not_object")
        # Remote schema references would make an offline check depend on external services.
        def check_refs(v):
            if isinstance(v, dict):
                if "$ref" in v and not v["$ref"].startswith("#"):
                    raise ValueError("external_schema_reference")
                for child in v.values():
                    check_refs(child)
            elif isinstance(v, list):
                for child in v:
                    check_refs(child)
        check_refs(schema)
        Draft202012Validator.check_schema(schema)
        tools.append({"type": "function", "function": {"name": name,
                      "description": f.get("description", ""), "parameters": schema}})
    return sorted(tools, key=lambda t: t["function"]["name"]), aliases


def extract_tools(system):
    tagged = re.search(r"<tools>\s*(.*?)\s*</tools>", system, re.S)
    if tagged:
        return safe_value(tagged.group(1))
    marker = system.find("JSON format")
    if marker < 0:
        raise ValueError("missing_tool_definition_marker")
    start = system.find("[", marker)
    if start < 0:
        raise ValueError("missing_tool_definition_array")
    # A lexical scan isolates the literal, without accepting arbitrary trailing code.
    stack, quote, escaped = [], None, False
    for i in range(start, len(system)):
        ch = system[i]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack or (stack.pop(), ch) not in {("[", "]"), ("{", "}")}:
                raise ValueError("unbalanced_tools")
            if not stack:
                return safe_value(system[start:i+1])
    raise ValueError("unterminated_tools")


def call_list(text, aliases):
    try:
        value = safe_value(text)
    except (ValueError, SyntaxError):
        if len(text) > 100_000:
            raise ValueError("call_text_too_large")
        tree = ast.parse(replace_names_outside_strings(text, aliases), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 50_000:
            raise ValueError("call_ast_too_large")
        if not isinstance(tree.body, ast.List) or not tree.body.elts:
            raise ValueError("calls_not_nonempty_list")
        value = []
        for node in tree.body.elts:
            if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
                    or node.func.id not in aliases.values() or node.args):
                raise ValueError("unsafe_call_expression")
            keys = [kw.arg for kw in node.keywords]
            if None in keys or len(set(keys)) != len(keys):
                raise ValueError("expanded_or_duplicate_arguments")
            value.append({"name": node.func.id, "arguments": {kw.arg: literal(kw.value) for kw in node.keywords}})
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(v, dict) for v in value):
        raise ValueError("invalid_call_list")
    return value


class Conversation:
    def __init__(self, tools, aliases, system):
        self.tools, self.aliases = tools, aliases
        self.messages = [{"role": "system", "content": system}]
        self.pending, self.old_ids, self.id_map = {}, set(), {}
        self.sequence = 0

    def name(self, raw):
        name = self.aliases.get(raw, raw)
        if name not in self.aliases.values():
            raise ValueError("unknown_tool:" + str(raw))
        return name

    def calls(self, values, content=""):
        if self.pending:
            raise ValueError("calls_before_all_responses")
        calls = []
        for v in values:
            f = v.get("function", v)
            name, arguments = self.name(f["name"]), safe_value(f.get("arguments", {}))
            if not isinstance(arguments, dict):
                raise ValueError("arguments_not_object")
            self.sequence += 1
            identity = f"call_{self.sequence:04d}"
            original = v.get("id")
            if original is not None:
                if not isinstance(original, str) or original in self.old_ids:
                    raise ValueError("duplicate_or_invalid_source_call_id")
                self.old_ids.add(original)
                self.id_map[original] = identity
            calls.append({"id": identity, "type": "function", "function": {"name": name, "arguments": arguments}})
            self.pending[identity] = name
        self.messages.append({"role": "assistant", "content": content, "tool_calls": calls})

    def response(self, payload, name=None, identity=None):
        if identity is not None:
            identity = self.id_map.get(identity, identity)
            candidates = [identity] if identity in self.pending else []
            if name is not None and candidates and self.pending[identity] != self.name(name):
                raise ValueError("response_id_name_disagree")
        elif name is not None:
            name = self.name(name)
            candidates = [cid for cid, n in self.pending.items() if n == name]
        else:
            candidates = list(self.pending)
        if len(candidates) != 1:
            raise ValueError("ambiguous_or_orphan_response")
        cid = candidates[0]
        self.messages.append({"role": "tool", "tool_call_id": cid, "name": self.pending.pop(cid),
                              "content": payload if isinstance(payload, str) else canonical(payload)})

    def responses(self, text, name=None, identity=None):
        if name is not None or identity is not None:
            self.response(text, name, identity)
            return
        tagged = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, re.S)
        if tagged:
            residue = re.sub(r"<tool_response>.*?</tool_response>", "", text, flags=re.S).strip()
            if residue:
                raise ValueError("text_outside_tool_response")
            values = [safe_value(t) for t in tagged]
        else:
            try:
                v = safe_value(text)
            except (ValueError, SyntaxError):
                self.response(text)
                return
            values = v if (isinstance(v, list) and v and all(isinstance(x, dict) and
                         ("name" in x or "tool_call_id" in x) and ("results" in x or "content" in x) for x in v)) else [v]
        for value in values:
            if isinstance(value, dict) and ("name" in value or "tool_call_id" in value) and ("content" in value or "results" in value):
                self.response(value.get("content", value.get("results")), value.get("name"), value.get("tool_call_id"))
            else:
                self.response(value)

    def text(self, role, text):
        if self.pending:
            raise ValueError("next_turn_before_all_responses")
        self.messages.append({"role": role, "content": text})

    def record(self):
        return validate({"tools": self.tools, "messages": self.messages})


def convert_public(row, source):
    system = row.get("system", "")
    if not system:
        system = "\n".join(m.get("value", m.get("content", "")) for m in row.get("conversations", [])
                           if m.get("from", m.get("role")) == "system")
    tools, aliases = tools_from(row["tools"] if "tools" in row else extract_tools(system))
    conv = Conversation(tools, aliases, PUBLIC_SYSTEM)
    for m in row.get("conversations", row.get("messages", [])):
        role = {"human": "user", "gpt": "assistant"}.get(m.get("from", m.get("role")), m.get("from", m.get("role")))
        text = m.get("value", m.get("content", ""))
        if not isinstance(text, str):
            raise ValueError("message_content_not_string")
        if role == "system":
            continue  # Public protocol prompts are replaced; tools live in tools, not in duplicated text.
        if role == "assistant" and m.get("tool_calls"):
            conv.calls(m["tool_calls"], text)
        elif role == "assistant" and "<tool_call>" in text:
            pieces = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
            residue = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
            residue = re.sub(r"^(?:\s|\\n|\\r)+|(?:\s|\\n|\\r)+$", "", residue)
            if not pieces or "<tool_call" in residue or "</tool_call" in residue:
                raise ValueError("malformed_tool_call_tags")
            conv.calls([c for piece in pieces for c in call_list(piece, aliases)], residue)
        elif source == "toolace" and role == "assistant" and text.strip().startswith("["):
            # A bracketed natural-language answer is never silently treated as a no-call example.
            conv.calls(call_list(text.strip(), aliases))
        elif role == "tool":
            conv.responses(text, m.get("name"), m.get("tool_call_id"))
        elif role in {"user", "assistant"}:
            if "<tool_response>" in text or "<tool_call>" in text:
                raise ValueError("tool_protocol_in_wrong_role")
            conv.text(role, text)
        else:
            raise ValueError("unsupported_role:" + str(role))
    return conv.record(), {"name_mapping": aliases, "call_id_mapping": conv.id_map,
                           "source_system_sha256": digest(system), "system_policy": "canonical_public_protocol"}


def validate(record):
    if set(record) != {"tools", "messages"}:
        raise ValueError("model_input_fields_must_be_tools_messages")
    validators = {}
    for t in record["tools"]:
        if t.get("type") != "function":
            raise ValueError("invalid_tool_type")
        f = t["function"]
        if f["name"] in validators or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", f["name"]):
            raise ValueError("invalid_or_duplicate_tool_name")
        Draft202012Validator.check_schema(f["parameters"])
        validators[f["name"]] = Draft202012Validator(f["parameters"], format_checker=FormatChecker())
    seen, pending, users, targets = set(), {}, 0, 0
    messages = record["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("unfinished_demonstration")
    for i, m in enumerate(messages):
        role = m["role"]
        allowed = {"role", "content"} | ({"tool_calls"} if role == "assistant" else {"name", "tool_call_id"} if role == "tool" else set())
        if set(m) - allowed or not isinstance(m.get("content"), str):
            raise ValueError("invalid_message_fields_or_content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid_role")
        if role == "system" and i != 0:
            raise ValueError("noninitial_system")
        if role == "tool":
            cid = m.get("tool_call_id")
            if cid not in pending or m.get("name") != pending[cid]:
                raise ValueError("orphan_or_mismatched_response")
            del pending[cid]
        elif pending:
            raise ValueError("turn_with_pending_responses")
        if role == "user":
            users += 1
        if role == "assistant":
            if not users or (not m["content"].strip() and not m.get("tool_calls")):
                raise ValueError("empty_or_unprompted_assistant")
            targets += 1
            for c in m.get("tool_calls", []):
                cid, f = c["id"], c["function"]
                if c.get("type") != "function" or not isinstance(cid, str) or not cid or cid in seen:
                    raise ValueError("invalid_or_duplicate_call_id")
                if f["name"] not in validators or not isinstance(f["arguments"], dict):
                    raise ValueError("unknown_tool_or_nonobject_arguments")
                validators[f["name"]].validate(f["arguments"])
                seen.add(cid)
                pending[cid] = f["name"]
    if not targets:
        raise ValueError("no_supervision")
    if pending and not messages[-1].get("tool_calls"):
        raise ValueError("unresolved_calls")
    canonical(record)
    return record


def model_fingerprint(record):
    # Tool order, key order, original call IDs and JSON result whitespace are not identity.
    r = copy.deepcopy(record)
    r["tools"].sort(key=lambda t: t["function"]["name"])
    ids = {}
    for m in r["messages"]:
        for call in m.get("tool_calls", []):
            old = call["id"]
            ids[old] = f"call_{len(ids)+1:04d}"
            call["id"] = ids[old]
        if m["role"] == "tool":
            m["tool_call_id"] = ids.get(m["tool_call_id"], m["tool_call_id"])
            try:
                m["content"] = canonical(json.loads(m["content"], object_pairs_hook=unique_object))
            except ValueError:
                pass
    return digest(r)


def content_flags(record):
    """High-precision flags only. Natural-language factual equivalence remains unverified."""
    flags, results = [], []
    for i, m in enumerate(record["messages"]):
        if m["role"] != "tool":
            continue
        try:
            result = json.loads(m["content"])
        except ValueError:
            continue
        if isinstance(result, dict):
            results.append((i, result))
    final = record["messages"][-1]
    if final.get("tool_calls"):
        return flags
    try:
        reply = json.loads(final["content"])
    except ValueError:
        reply = None
    if results:
        last = results[-1][1]
        if last.get("status") in {"error", "failed", "failure"}:
            # Any explicit success claim immediately after an unrecovered failure needs review.
            success = re.search(r"\b(?:successfully|completed successfully)\b|操作成功|已成功|执行成功", final["content"], re.I)
            if success or (isinstance(reply, dict) and reply.get("status") == "success"):
                flags.append("success_claim_after_unrecovered_error")
        if isinstance(reply, dict) and isinstance(reply.get("result"), dict):
            data = last.get("data", last)
            if isinstance(data, dict) and any(k in data and data[k] != v for k, v in reply["result"].items()):
                flags.append("structured_final_contradicts_latest_result")
        observations = canonical([r.get("data", r) for _, r in results]).lower()
        # Narrow, explicit weather contradictions; this is not a general semantic verifier.
        for observed, contradictory in (("rain", "sunny"), ("sunny", "rain")):
            if re.search(r'"condition":"' + observed + r'"', observations) and not re.search(r'"condition":"' + contradictory + r'"', observations):
                if re.search(r"\b(?:weather is|will be|is currently) " + contradictory + r"\b", final["content"], re.I):
                    flags.append("explicit_weather_contradiction")
    return flags


def annotate(record, meta, review=None):
    messages = record["messages"]
    calls = [c for m in messages for c in m.get("tool_calls", [])]
    tags = set()
    if calls:
        tags.add("selection")
    if any(m["role"] == "tool" and any(n["role"] == "assistant" for n in messages[i+1:]) for i, m in enumerate(messages)):
        tags.add("feedback")
    if calls and not any(m["role"] == "tool" for m in messages):
        tags.add("call_generation_only")
    if any(m["role"] == "user" and any(n["role"] == "tool" for n in messages[:i]) for i, m in enumerate(messages)):
        tags.add("followup_user_message")
    for m in messages:
        if m["role"] == "tool":
            try:
                v = json.loads(m["content"])
            except ValueError:
                continue
            if isinstance(v, dict) and (v.get("status") in {"error", "failed", "failure"} or "error" in v or "error_code" in v):
                tags.add("tool_error_preserved")
    if meta.get("verified_missing_slots"):
        tags.add("clarification")
    primary = "clarification" if "clarification" in tags else "feedback" if "feedback" in tags else "selection" if calls else "unreviewed_no_call"
    flags = content_flags(record) if meta["source"] != "self" else []
    if review:
        if review.get("fingerprint") != model_fingerprint(record) or not review.get("reviewer") or not review.get("evidence"):
            raise ValueError("review_requires_matching_fingerprint_reviewer_evidence")
        target = review["primary_behavior"]
        if target not in BEHAVIORS:
            raise ValueError("invalid_review_behavior")
        evidence = review["evidence"]
        if target == "clarification":
            name = review["tool_name"]
            schema = next(t["function"]["parameters"] for t in record["tools"] if t["function"]["name"] == name)
            missing = review.get("missing_parameters", [])
            pos = review["question_message_index"]
            if (not missing or not set(missing) <= set(schema.get("required", [])) or not 0 <= pos < len(messages)
                    or messages[pos]["role"] != "assistant" or messages[pos].get("tool_calls")
                    or not evidence.get("request_lacks_required_values") or not evidence.get("question_is_relevant")):
                raise ValueError("unsupported_clarification_review")
            if any(c["function"]["name"] == name for m in messages[:pos+1] for c in m.get("tool_calls", [])):
                raise ValueError("guessed_tool_call_before_clarification")
        if target == "no_call" and (calls or evidence.get("reason") not in {"tool_not_applicable", "no_tool_needed", "reasonable_stop"}):
            raise ValueError("unsupported_no_call_review")
        if target in {"feedback", "selection"} and target not in tags:
            raise ValueError("review_disagrees_with_call_structure")
        if flags and evidence.get("content_consistency") != "passed":
            raise ValueError("content_flags_not_resolved_by_review")
        tags.add(target)
        primary = target
        meta["review"] = review
        flags = []
    meta.update(behavior_tags=sorted(tags), primary_behavior=primary, content_flags=flags,
                sampling_eligible=primary in BEHAVIORS and not flags,
                feedback_evidence="replayed_structured_facts" if meta["source"] == "self" else "structural_only_not_api_reexecuted",
                natural_language_consistency="reviewed" if review else "heuristics_only_not_proven")


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def semantic_schema(v):
    if isinstance(v, dict):
        return {k: semantic_schema(x) for k, x in v.items() if k not in {"description", "title", "examples", "default"}}
    return [semantic_schema(x) for x in v] if isinstance(v, list) else v


def request_key(record):
    text = next(m["content"] for m in record["messages"] if m["role"] == "user")
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', " SLOT ", text)
    text = re.sub(r"\d+(?:[.:/-]\d+)*", " NUMBER ", text)
    return " ".join(re.findall(r"\w+", text))


def build_groups(items, seed, validation_fraction):
    uf, relations = UnionFind(len(items)), Counter()
    exact, original_groups, prefix_index = {}, {}, {}
    signatures, shapes, requests = [], [], []
    for i, item in enumerate(items):
        record, meta = item["record"], item["meta"]
        fp = meta["fingerprint"]
        if fp in exact:
            uf.join(i, exact[fp])
            relations["exact"] += 1
        exact[fp] = i
        for prov in meta["provenance"]:
            if prov.get("original_group_id"):
                k = (prov["source"], prov["original_group_id"])
                if k in original_groups:
                    uf.join(i, original_groups[k])
                original_groups[k] = i
        used = {c["function"]["name"] for m in record["messages"] for c in m.get("tool_calls", [])}
        signatures.append(digest([semantic_schema(t["function"]) for t in record["tools"] if not used or t["function"]["name"] in used]))
        shapes.append(digest([[{ "name": c["function"]["name"], "arguments": sorted(c["function"]["arguments"])}
                              for c in m["tool_calls"]] for m in record["messages"] if m.get("tool_calls")]))
        requests.append(request_key(record))
        # Full message prefixes include canonical system, tool definitions, call IDs and responses.
        prefix_index[model_fingerprint(record)] = i
    for i, item in enumerate(items):
        r = item["record"]
        for end, message in enumerate(r["messages"][:-1], start=1):
            if message["role"] != "assistant":
                continue
            key = model_fingerprint({"tools": r["tools"], "messages": r["messages"][:end]})
            if key in prefix_index:
                j = prefix_index[key]
                uf.join(i, j)
                relations["dialogue_prefix"] += 1
                if {items[i]["meta"]["source"], items[j]["meta"]["source"]} == {"hermes_multi", "hermes_single"}:
                    relations["hermes_single_multi_prefix"] += 1
    # Private split families stay authoritative. Broad public API signatures must not collapse
    # every self trajectory using the same eight tools into one family.
    blocks, request_blocks, lsh = {}, {}, defaultdict(list)
    candidate_checks = 0
    for i, item in enumerate(items):
        if item["meta"]["source"] == "self":
            continue
        block = (signatures[i], shapes[i])
        # Conservatively keep the same used API schema and call pattern together,
        # even when literal parameter values or the request wording differ.
        if block in blocks:
            uf.join(i, blocks[block])
            relations["signature_structure"] += 1
        blocks[block] = i
        k = (signatures[i], requests[i])
        if k in request_blocks:
            uf.join(i, request_blocks[k])
            relations["signature_request"] += 1
        request_blocks[k] = i
        words = requests[i].split()
        shingles = set(zip(words, words[1:])) if len(words) > 1 else set(words)
        if len(shingles) < 3:
            continue
        # MinHash LSH proposes candidates; actual Jaccard and text similarity decide the edge.
        hashes = [min(int(digest([salt, s])[:16], 16) for s in shingles) for salt in range(12)]
        candidates = set()
        band_keys = [(signatures[i], b, tuple(hashes[b*3:b*3+3])) for b in range(4)]
        for key in band_keys:
            candidates.update(lsh[key])
        for j in sorted(candidates):
            candidate_checks += 1
            other_words = requests[j].split()
            other = set(zip(other_words, other_words[1:])) if len(other_words) > 1 else set(other_words)
            similarity = len(shingles & other) / len(shingles | other)
            if similarity >= .88 and SequenceMatcher(None, requests[i], requests[j], autojunk=False).ratio() >= .9:
                uf.join(i, j)
                relations["near_request"] += 1
        # One representative per already linked family is enough in each LSH bucket.
        for key in band_keys:
            if not any(uf.find(j) == uf.find(i) for j in lsh[key]):
                lsh[key].append(i)
    families = defaultdict(list)
    for i in range(len(items)):
        families[uf.find(i)].append(i)
    sizes = []
    for members in families.values():
        group = digest(sorted(items[i]["meta"]["fingerprint"] for i in members))
        fixed = {p["fixed_split"] for i in members for p in items[i]["meta"]["provenance"] if p.get("fixed_split")}
        if len(fixed) > 1:
            raise ValueError("existing_self_splits_conflict_in_family:" + group)
        split = next(iter(fixed)) if fixed else ("validation" if int(digest([seed, group])[:16], 16) / 2**64 < validation_fraction else "train")
        for i in members:
            items[i]["meta"].update(group_id=group, split=split)
        sizes.append(len(members))
    return {"relations": dict(relations), "family_count": len(families), "largest_family": max(sizes, default=0),
            "near_candidate_checks": candidate_checks, "cross_split_groups": 0,
            "cross_split_exact_duplicates": 0,
            "near_duplicate_scope": "public used-schema/call-pattern families plus lexical LSH within used-schema blocks; self fixed semantic families; not a semantic proof"}


def stats(items):
    lengths = sorted(i["meta"]["token_length"] for i in items)
    def percentile(q):
        return lengths[min(len(lengths)-1, math.ceil(q * len(lengths))-1)] if lengths else None
    return {"records": len(items), "sources": dict(Counter(i["meta"]["source"] for i in items)),
            "primary_behaviors": dict(Counter(i["meta"]["primary_behavior"] for i in items)),
            "behavior_tags": dict(Counter(t for i in items for t in i["meta"]["behavior_tags"])),
            "groups": len({i["meta"]["group_id"] for i in items}),
            "tokens": {"min": min(lengths, default=None), "p50": percentile(.5), "p90": percentile(.9),
                       "p95": percentile(.95), "p99": percentile(.99), "max": max(lengths, default=None),
                       "mean": sum(lengths)/len(lengths) if lengths else None,
                       "over_512": sum(n > 512 for n in lengths), "over_4096": sum(n > 4096 for n in lengths)}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def write_pair(root, data_path, meta_path, items):
    write_jsonl(root / data_path, (i["record"] for i in items))
    write_jsonl(root / meta_path, (dict(i["meta"], row_index=n) for n, i in enumerate(items)))


def load_pair(root, data_path, meta_path):
    from itertools import zip_longest
    items = []
    for n, (record, meta) in enumerate(zip_longest(read_rows(root / data_path), read_rows(root / meta_path))):
        if record is None or meta is None or meta.get("row_index") != n or meta.get("fingerprint") != model_fingerprint(record):
            raise ValueError("data_metadata_alignment_or_fingerprint_error:" + str(data_path))
        validate(record)
        items.append({"record": record, "meta": meta})
    return items


def verify_assets(root, report):
    root = Path(root).resolve()
    for relative, expected in report["asset_sha256"].items():
        path = (root / relative).resolve()
        path.relative_to(root)
        if file_hash(path) != expected:
            raise ValueError("dataset_asset_changed:" + relative)


def check_split_overlap(items):
    groups, fingerprints = {}, {}
    for item in items:
        meta = item["meta"]
        for value, index in ((meta["group_id"], groups), (meta["fingerprint"], fingerprints)):
            if value in index and index[value] != meta["split"]:
                raise ValueError("dataset_cross_split_overlap")
            index[value] = meta["split"]
        if any(p.get("fixed_split") and p["fixed_split"] != meta["split"] for p in meta["provenance"]):
            raise ValueError("self_original_split_changed")


def tokenizer_bundle(path):
    # Same loader/template contract as the training entrypoint; no model weights are loaded.
    from .training import load_tokenizer, require_stack
    require_stack()
    tokenizer, template = load_tokenizer({"model_name_or_path": str(path)})
    effective = template or tokenizer.chat_template
    if not isinstance(effective, str):
        raise ValueError("explicit_single_chat_template_required")
    assets = {p.name: file_hash(p) for p in Path(path).iterdir() if p.is_file() and
              (p.name.startswith(("tokenizer", "special_tokens", "vocab", "merges", "chat_template")) or p.name == "config.json")}
    if not assets:
        raise ValueError("missing_local_tokenizer_assets")
    info = {"local_path": str(Path(path).resolve()), "assets_sha256": assets,
            "chat_template_sha256": hashlib.sha256(effective.encode()).hexdigest(), "enable_thinking": False,
            "add_generation_prompt": False, "transformers": importlib.metadata.version("transformers"),
            "trl": importlib.metadata.version("trl")}
    return tokenizer, effective, info


def token_count(tokenizer, record, template):
    # Transformers 5 may return a BatchEncoding by default. Its length is the
    # number of fields, not the number of tokens. Always select input_ids.
    encoded = tokenizer.apply_chat_template(record["messages"], tools=record["tools"], chat_template=template,
        enable_thinking=False, add_generation_prompt=False, tokenize=True, truncation=False, return_dict=True)
    ids = encoded["input_ids"]
    if not isinstance(ids, list) or not ids or any(type(token) is not int for token in ids):
        raise ValueError("expected_one_unbatched_input_ids_sequence")
    return len(ids)


def verify_public_input(entry):
    manifest_path = Path(entry["download_manifest"])
    downloaded = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    path = Path(entry["path"])
    if not re.fullmatch(r"[0-9a-f]{40}", downloaded["revision"]):
        raise ValueError("unpinned_public_revision")
    asset = next(f for f in downloaded["files"] if f["filename"] == path.name)
    actual = file_hash(path)
    if asset["sha256"] != actual:
        raise ValueError("public_input_hash_mismatch:" + str(path))
    return {"source": entry["source"], "repo_id": downloaded["repo_id"], "revision": downloaded["revision"],
            "license": entry["license"], "path": str(path), "file_sha256": actual,
            "download_manifest_sha256": file_hash(manifest_path), "license_evidence": entry["license_evidence"],
            "verification_scope": "format_schema_consistency_only_no_real_api_reexecution"}


def self_items(config):
    from wearable_tasks.auditing import audit_privacy
    from wearable_tasks.exporting import assert_public, sft_record
    from wearable_tasks.pipeline import source_fingerprint
    from wearable_tasks.reference import replay
    root = Path(config["directory"])
    report = json.loads((root / "validation-report.json").read_text(encoding="utf-8-sig"))
    if report.get("status") != "passed" or report.get("source_fingerprint") != source_fingerprint():
        raise ValueError("self_validation_report_failed_or_stale_code")
    audit_hash = file_hash(root / "validation-report.json")
    seen_case_ids = set()
    for original, split in SELF_SPLITS.items():
        for relative in (f"private/{original}.jsonl", f"private/trajectories/{original}.jsonl"):
            if report.get("file_sha256", {}).get(relative) != file_hash(root / relative):
                raise ValueError("self_file_differs_from_validation_report:" + relative)
        tasks = list(read_rows(root / "private" / (original + ".jsonl")))
        trajectories = list(read_rows(root / "private" / "trajectories" / (original + ".jsonl")))
        expected = config["expected_counts"][split]
        if len(tasks) != expected or len(trajectories) != expected:
            raise ValueError("self_input_count_mismatch:" + split)
        by_id = {t["case_id"]: t for t in trajectories}
        if len(by_id) != len(trajectories) or len({t["case_id"] for t in tasks}) != len(tasks) or set(by_id) != {t["case_id"] for t in tasks}:
            raise ValueError("self_case_id_mismatch")
        source_hashes = {"tasks_sha256": file_hash(root / "private" / (original + ".jsonl")),
                         "trajectories_sha256": file_hash(root / "private" / "trajectories" / (original + ".jsonl"))}
        public_path = root / "public" / "sft" / (original + ".jsonl")
        if public_path.exists() and report.get("file_sha256", {}).get(public_path.relative_to(root).as_posix()) != file_hash(public_path):
            raise ValueError("self_public_file_differs_from_validation_report")
        published = list(read_rows(public_path)) if public_path.exists() else None
        if published is not None and len(published) != len(tasks):
            raise ValueError("self_public_export_count_mismatch")
        for index, task in enumerate(tasks):
            if task["case_id"] in seen_case_ids:
                raise ValueError("self_case_id_reused_across_splits")
            seen_case_ids.add(task["case_id"])
            if task["split"] != original:
                raise ValueError("self_original_split_mismatch")
            trajectory = by_id[task["case_id"]]
            env = replay(task, trajectory)
            verdict = env.evaluate()
            if not verdict.get("success") or verdict.get("reward") != 1:
                raise ValueError("self_replay_not_reward_one:" + task["case_id"])
            record = sft_record(task, trajectory, verified_env=env)
            assert_public(record)
            audit_privacy(task, env)
            if published is not None and canonical(record) != canonical(published[index]):
                raise ValueError("self_public_export_differs_from_replay")
            # Preserve verified messages/schema exactly; existing IDs already use the common format.
            record = validate(record)
            provenance = {"source": "self", "source_id": task["case_id"], "source_row": index,
                          "license": config["license"], "original_group_id": task["group_id"],
                          "fixed_split": split, "validation_report_sha256": audit_hash, **source_hashes,
                          "verification_scope": "local_simulator_replayed_reward_one_privacy_canary_checked"}
            meta = {"source": "self", "source_id": task["case_id"], "license": config["license"],
                    "provenance": [provenance], "replay_success": True, "terminal_reward": 1,
                    "verified_missing_slots": list(task["goal"]["required_asks"]),
                    "name_mapping": {t["function"]["name"]: t["function"]["name"] for t in record["tools"]},
                    "feedback_scope": verdict.get("feedback_scope", "structured_facts_only")}
            yield {"record": record, "meta": meta}


def write_views(root, items, max_tokens, report):
    check_split_overlap(items)
    selected = [i for i in items if i["meta"]["token_length"] <= max_tokens and not i["meta"]["content_flags"]]
    view_report, smoke_report = {}, {}
    for split in SPLITS:
        pool = [i for i in items if i["meta"]["split"] == split]
        full = [i for i in selected if i["meta"]["split"] == split]
        write_pair(root, f"pool/{split}.jsonl", f"metadata/pool-{split}.jsonl", pool)
        write_pair(root, f"{split}.jsonl", f"metadata/{split}.jsonl", full)
        view_report[split] = stats(full)
        if split in {"train", "validation"}:
            smoke = [i for i in full if i["meta"]["token_length"] <= 512]
            write_pair(root, f"local_smoke/{split}.jsonl", f"metadata/local_smoke-{split}.jsonl", smoke)
            smoke_report[split] = stats(smoke)
    over = sum(i["meta"]["token_length"] > max_tokens for i in items)
    report.update(full_max_tokens=max_tokens, full_views=view_report, local_smoke=smoke_report,
                  pool_by_split={s: stats([i for i in items if i["meta"]["split"] == s]) for s in SPLITS},
                  overlength_records=over, overlength_ratio=over/len(items) if items else 0,
                  content_quarantined=sum(bool(i["meta"]["content_flags"]) for i in items),
                  training_started=False, mask_status="pending_explicit_audit_command",
                  self_retained={s: sum(i["meta"]["source"] == "self" and i["meta"]["split"] == s for i in selected) for s in SPLITS})
    report["split_group_overlap"] = {a + "/" + b: len(
        {i["meta"]["group_id"] for i in items if i["meta"]["split"] == a} &
        {i["meta"]["group_id"] for i in items if i["meta"]["split"] == b})
        for n, a in enumerate(SPLITS) for b in SPLITS[n+1:]}
    report["asset_sha256"] = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.jsonl"))}
    write_json(root / "preparation-report.json", report)


def fresh_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("output_already_exists_choose_new_directory:" + str(path))
    temporary = path.with_name(path.name + ".preparing-" + uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True)
    return path, temporary


def prepare(manifest, output, tokenizer_path):
    if Path(output).exists():
        raise ValueError("output_already_exists_choose_new_directory")
    if not 0 < manifest["public_validation_fraction"] < 1:
        raise ValueError("validation_fraction_out_of_range")
    if type(manifest["full_max_tokens"]) is not int or manifest["full_max_tokens"] < 512:
        raise ValueError("full_max_tokens_must_be_at_least_512")
    tokenizer, template, tokenizer_info = tokenizer_bundle(tokenizer_path)
    print(canonical({"phase": "tokenizer_ready", "template_sha256": tokenizer_info["chat_template_sha256"]}), flush=True)
    reviews = {}
    for path in manifest.get("review_files", []):
        for row in read_rows(path):
            key = (row["source"], str(row["source_id"]))
            if key in reviews:
                raise ValueError("duplicate_review_key")
            reviews[key] = row
    items, rejected, raw_counts, input_info = [], [], Counter(), []
    for entry in manifest["public_inputs"]:
        source = entry["source"]
        if source not in PUBLIC_SOURCES:
            raise ValueError("unsupported_public_source")
        provenance = verify_public_input(entry)
        input_info.append(provenance)
        for index, row in enumerate(read_rows(entry["path"])):
            raw_counts[source] += 1
            identity = str(row.get("id", index))
            try:
                record, conversion = convert_public(row, source)
                meta = {"source": source, "source_id": identity, "license": entry["license"],
                        "provenance": [dict(provenance, source_id=identity, source_row=index, raw_record_sha256=digest(row),
                                             upstream_record_source=row.get("source"), category=row.get("category"),
                                             subcategory=row.get("subcategory"), task=row.get("task"), **conversion)], **conversion}
                annotate(record, meta, reviews.get((source, identity)))
                items.append({"record": record, "meta": meta})
            except (ValueError, KeyError, TypeError, SyntaxError, RecursionError, AttributeError, StopIteration, SchemaError, ValidationError) as exc:
                rejected.append({"source": source, "source_id": identity, "source_row": index,
                                 "stage": "conversion_validation_annotation", "reason_type": type(exc).__name__, "reason": str(exc)[:1000]})
            if (index + 1) % 500 == 0:
                print(canonical({"phase": "public_conversion", "source": source, "rows": index + 1,
                                 "accepted_total": len(items), "rejected_total": len(rejected)}), flush=True)
        print(canonical({"phase": "public_source_complete", "source": source, "rows": raw_counts[source]}), flush=True)
    for self_index, item in enumerate(self_items(manifest["self"]), start=1):
        raw_counts["self"] += 1
        annotate(item["record"], item["meta"])
        items.append(item)
        if self_index % 100 == 0:
            print(canonical({"phase": "self_replay", "completed": self_index}), flush=True)
    # Prefer the verified self transcript if a public item is an exact duplicate.
    priority = {"self": 0, "hermes_multi": 1, "glaive": 2, "toolace": 3, "hermes_single": 4}
    seen, deduplicated, duplicates = {}, [], []
    for item in sorted(items, key=lambda i: (priority[i["meta"]["source"]], i["meta"]["source_id"])):
        meta = item["meta"]
        meta.update(fingerprint=model_fingerprint(item["record"]), transform_version=VERSION)
        fp = meta["fingerprint"]
        if fp in seen:
            keeper = seen[fp]
            keeper["meta"]["provenance"].extend(meta["provenance"])
            duplicates.append({"fingerprint": fp, "retained_source": keeper["meta"]["source"],
                               "duplicate_source": meta["source"], "duplicate_source_id": meta["source_id"]})
        else:
            seen[fp] = item
            deduplicated.append(item)
    print(canonical({"phase": "grouping", "unique": len(deduplicated), "duplicates": len(duplicates)}), flush=True)
    grouping = build_groups(deduplicated, manifest["seed"], manifest["public_validation_fraction"])
    print(canonical({"phase": "grouping_complete", **grouping}), flush=True)
    for index, item in enumerate(deduplicated):
        record = item["record"]
        item["meta"]["token_length"] = token_count(tokenizer, record, template)
        if index and index % 500 == 0:
            print(canonical({"phase": "lengths", "completed": index, "total": len(deduplicated)}), flush=True)
    output, temporary = fresh_output(output)
    report = {"version": VERSION, "mode": "inventory_not_target_sampled", "seed": manifest["seed"],
              "raw_counts": dict(raw_counts), "input_files": input_info, "manifest": manifest,
              "tokenizer": tokenizer_info, "grouping": grouping,
              "stages": {"raw": sum(raw_counts.values()), "rejected": len(rejected),
                         "converted": len(items), "exact_duplicates_removed": len(duplicates), "unique": len(deduplicated)},
              "rejection_reasons": dict(Counter(r["reason_type"] + ":" + r["reason"].split(":")[0].split("\n")[0] for r in rejected)),
              "stage_counts_by_source": {s: {"raw": raw_counts[s],
                  "conversion_rejected": sum(r["source"] == s for r in rejected),
                  "converted": sum(i["meta"]["source"] == s for i in items),
                  "exact_duplicates_removed": sum(d["duplicate_source"] == s for d in duplicates),
                  "unique": sum(i["meta"]["source"] == s for i in deduplicated)} for s in raw_counts},
              "pool_by_split": {s: stats([i for i in deduplicated if i["meta"]["split"] == s]) for s in SPLITS},
              "public_api_reexecution": False,
              "annotation_limits": "feedback is structural; clarification/no-call require bound human review or self replay; natural-language correctness not proven",
              "short_smoke_limitations": "512-token full protocols may contain mostly simple calls; smoke success does not establish multi-turn quality",
              "self_expected_input_counts": manifest["self"]["expected_counts"]}
    try:
        write_jsonl(temporary / "metadata/rejected.jsonl", rejected)
        write_jsonl(temporary / "metadata/duplicates.jsonl", duplicates)
        write_json(temporary / "metadata/tokenizer.json", tokenizer_info)
        write_json(temporary / "metadata/review-schema.example.json", {
            "source": "glaive", "source_id": "REPLACE_WITH_SOURCE_ID", "fingerprint": "COPY_FROM_METADATA",
            "reviewer": "REVIEWER", "primary_behavior": "clarification", "tool_name": "NORMALIZED_TOOL_NAME",
            "missing_parameters": ["REQUIRED_PARAMETER"], "question_message_index": 2,
            "evidence": {"request_lacks_required_values": True, "question_is_relevant": True,
                         "content_consistency": "passed", "notes": "Record the actual missing value and relevant question here."}})
        write_views(temporary, deduplicated, manifest["full_max_tokens"], report)
        temporary.replace(output)
    except BaseException:
        # Retain a failed build for diagnosis; never publish a partially written dataset.
        write_json(temporary / "FAILED.json", {"status": "failed_unpublished"})
        raise
    return report


def ranked(items, seed, source_weights):
    # Weighted sampling without replacement, stable across input order.
    def score(item):
        u = (int(digest([seed, item["meta"]["fingerprint"]])[:16], 16) + 1) / (2**64 + 1)
        return -math.log(u) / source_weights.get(item["meta"]["source"], 1)
    return sorted(items, key=score)


def select_public(items, target, ratios, seed, weights, single_limit):
    if target < 0 or any(v < 0 for v in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9 or set(ratios) != set(BEHAVIORS):
        raise ValueError("invalid_mixture_targets")
    quotas = {b: int(target * ratios[b]) for b in BEHAVIORS}
    for b in sorted(BEHAVIORS, key=lambda b: (-(target * ratios[b] - quotas[b]), b))[:target-sum(quotas.values())]:
        quotas[b] += 1
    candidates = ranked([i for i in items if i["meta"]["sampling_eligible"]], seed, weights)
    chosen, picked, singles = [], set(), 0
    def take(item):
        nonlocal singles
        fp = item["meta"]["fingerprint"]
        if fp in picked or (item["meta"]["source"] == "hermes_single" and singles >= single_limit):
            return False
        picked.add(fp)
        chosen.append(item)
        singles += int(item["meta"]["source"] == "hermes_single")
        return True
    for behavior in BEHAVIORS:
        count = 0
        for item in candidates:
            if count >= quotas[behavior]:
                break
            if item["meta"]["primary_behavior"] == behavior and take(item):
                count += 1
    initial = Counter(i["meta"]["primary_behavior"] for i in chosen)
    for item in candidates:
        if len(chosen) >= target:
            break
        take(item)
    actual = Counter(i["meta"]["primary_behavior"] for i in chosen)
    return chosen, {"requested_total": target, "actual_total": len(chosen), "total_shortfall": max(0, target-len(chosen)),
                    "requested_behaviors": quotas, "available_behaviors": dict(Counter(i["meta"]["primary_behavior"] for i in candidates)),
                    "initial_behavior_shortfalls": {b: max(0, quotas[b]-initial[b]) for b in BEHAVIORS},
                    "actual_behaviors": dict(actual), "actual_ratios": {b: actual[b]/len(chosen) if chosen else 0 for b in BEHAVIORS},
                    "redistribution": "unfilled behavior quotas redistributed to eligible available behaviors without replacement",
                    "hermes_single_cap": single_limit, "source_weights": weights}


def mix(dataset, output, policy):
    if Path(output).exists():
        raise ValueError("output_already_exists_choose_new_directory")
    if not 0 <= policy["hermes_single_max_fraction"] <= 1:
        raise ValueError("invalid_singleturn_fraction")
    dataset = Path(dataset)
    prior = json.loads((dataset / "preparation-report.json").read_text(encoding="utf-8"))
    if prior["version"] != VERSION:
        raise ValueError("incompatible_dataset_version")
    if prior.get("status") == "invalid_do_not_use":
        raise ValueError("invalid_inventory_must_be_recounted_before_sampling")
    verify_assets(dataset, prior)
    items = [i for s in SPLITS for i in load_pair(dataset, f"pool/{s}.jsonl", f"metadata/pool-{s}.jsonl")]
    check_split_overlap(items)
    for source, weight in policy["source_weights"].items():
        if source not in PUBLIC_SOURCES or weight <= 0:
            raise ValueError("invalid_source_weight")
    selected = [i for i in items if i["meta"]["source"] == "self"]
    selections = {}
    for split, target in (("train", policy["public_train_target"]), ("validation", policy["public_validation_target"])):
        candidates = [i for i in items if i["meta"]["source"] != "self" and i["meta"]["split"] == split
                      and i["meta"]["token_length"] <= prior["full_max_tokens"] and not i["meta"]["content_flags"]]
        chosen, detail = select_public(candidates, target, policy["behavior_ratios"], policy["seed"],
                                       policy["source_weights"], math.floor(target * policy["hermes_single_max_fraction"]))
        selected.extend(chosen)
        selections[split] = detail
    selected.sort(key=lambda i: digest([policy["seed"], i["meta"]["fingerprint"]]))
    output, temporary = fresh_output(output)
    report = {**prior, "mode": "behavior_sampled", "parent_dataset": str(dataset.resolve()),
              "parent_pool_by_split": prior["pool_by_split"],
              "parent_report_sha256": file_hash(dataset / "preparation-report.json"), "mixture_policy": policy,
              "mixture_result": selections}
    try:
        shutil.copytree(dataset / "metadata", temporary / "metadata", dirs_exist_ok=True)
        write_views(temporary, selected, prior["full_max_tokens"], report)
        temporary.replace(output)
    except BaseException:
        write_json(temporary / "FAILED.json", {"status": "failed_unpublished"})
        raise
    return report


def recount(dataset, output, tokenizer_path):
    """Re-encode an intact pool without repeating conversion, replay or split assignment."""
    root = Path(dataset)
    if Path(output).exists():
        raise ValueError("output_already_exists_choose_new_directory")
    prior = json.loads((root / "preparation-report.json").read_text(encoding="utf-8"))
    if prior["version"] != VERSION:
        raise ValueError("incompatible_dataset_version")
    verify_assets(root, prior)
    items = [i for s in SPLITS for i in load_pair(root, f"pool/{s}.jsonl", f"metadata/pool-{s}.jsonl")]
    check_split_overlap(items)
    tokenizer, template, info = tokenizer_bundle(tokenizer_path)
    for index, item in enumerate(items, start=1):
        item["meta"]["token_length"] = token_count(tokenizer, item["record"], template)
        if index % 500 == 0:
            print(canonical({"phase": "recount", "completed": index, "total": len(items)}), flush=True)
    report = {**prior, "status": "prepared", "tokenizer": info,
              "recount_parent": str(root.resolve()), "recount_parent_report_sha256": file_hash(root / "preparation-report.json"),
              "length_method": "len(encoding.input_ids), explicit return_dict=True, no truncation"}
    report.pop("invalid_reason", None)
    report.pop("mask_summary", None)
    output, temporary = fresh_output(output)
    try:
        shutil.copytree(root / "metadata", temporary / "metadata", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("mask-audit.json"))
        write_json(temporary / "metadata/tokenizer.json", info)
        write_views(temporary, items, prior["full_max_tokens"], report)
        temporary.replace(output)
    except BaseException:
        write_json(temporary / "FAILED.json", {"status": "failed_unpublished"})
        raise
    return report


def audit(dataset, tokenizer_path, seed=42, random_count=32, per_behavior=4):
    from .training import encode_record, render_message_prefixes
    root = Path(dataset)
    report = json.loads((root / "preparation-report.json").read_text(encoding="utf-8"))
    verify_assets(root, report)
    tokenizer, template, info = tokenizer_bundle(tokenizer_path)
    for key in ("assets_sha256", "chat_template_sha256", "enable_thinking", "transformers", "trl"):
        if info[key] != report["tokenizer"][key]:
            raise ValueError("tokenizer_or_template_changed_reprepare:" + key)
    items, chosen, reasons = [], {}, defaultdict(set)
    for split in ("train", "validation"):
        split_items = load_pair(root, f"{split}.jsonl", f"metadata/{split}.jsonl")
        items.extend(split_items)
        buckets = defaultdict(list)
        for item in split_items:
            for tag in {item["meta"]["primary_behavior"], *item["meta"]["behavior_tags"]}:
                buckets[tag].append(item)
        for tag, bucket in buckets.items():
            for item in sorted(bucket, key=lambda i: digest([seed, i["meta"]["fingerprint"]]))[:per_behavior]:
                fp = item["meta"]["fingerprint"]
                chosen[fp] = item
                reasons[fp].add(split + ":" + tag)
    for item in sorted(items, key=lambda i: digest([seed, "random", i["meta"]["fingerprint"]]))[:random_count]:
        fp = item["meta"]["fingerprint"]
        chosen[fp] = item
        reasons[fp].add("random")
    # All available smoke examples have the same audit contract and include complete protocols.
    for split in ("train", "validation"):
        short = load_pair(root, f"local_smoke/{split}.jsonl", f"metadata/local_smoke-{split}.jsonl")
        for item in short:
            if item["meta"]["token_length"] > 512:
                raise ValueError("smoke_over_512")
            fp = item["meta"]["fingerprint"]
            chosen[fp] = item
            reasons[fp].add("local_smoke:" + split)
    inspections = []
    for fp, item in chosen.items():
        encoded, detail = encode_record(tokenizer, item["record"], template, report["full_max_tokens"])
        if len(encoded["input_ids"]) != item["meta"]["token_length"]:
            raise ValueError("token_length_drift")
        # Qwen's actual turn terminator must be trained in every assistant segment.
        eot = "<|im_end|>"
        eot_id = tokenizer.convert_tokens_to_ids(eot)
        if tokenizer.convert_ids_to_tokens(eot_id) != eot:
            raise ValueError("qwen_end_of_turn_token_missing")
        rendered = tokenizer.apply_chat_template(item["record"]["messages"], tools=item["record"]["tools"], chat_template=template,
                                                  tokenize=False, add_generation_prompt=False, enable_thinking=False)
        offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        previous = ""
        prefixes = render_message_prefixes(tokenizer, item["record"], template, rendered)
        for pos, (message, prefix) in enumerate(zip(item["record"]["messages"], prefixes)):
            if message["role"] == "assistant":
                supervised = detail["segments"][pos]["supervised_text"]
                # A mask containing only a terminator is insufficient. Check actual targets too.
                if message["content"].strip() and message["content"].strip() not in supervised:
                    raise ValueError("assistant_text_not_fully_supervised:" + fp)
                for call in message.get("tool_calls", []):
                    f = call["function"]
                    if f["name"] not in supervised or any(k not in supervised for k in f["arguments"]):
                        raise ValueError("assistant_call_payload_not_supervised:" + fp)
                if message.get("tool_calls"):
                    # Check every token of the rendered Qwen call payload, including values.
                    segment = rendered[len(previous):len(prefix)]
                    spans = list(re.finditer(r"<tool_call>.*?</tool_call>", segment, re.S))
                    if not spans:
                        raise ValueError("unsupported_qwen_call_rendering_cannot_audit_payload")
                    for span in spans:
                        lo, hi = len(previous) + span.start(), len(previous) + span.end()
                        payload_tokens = [j for j, (a, b) in enumerate(offsets) if b > lo and a < hi and b > a]
                        if not payload_tokens or any(not encoded["assistant_masks"][j] for j in payload_tokens):
                            raise ValueError("assistant_call_argument_values_not_supervised:" + fp)
                ends = [j for j, (lo, hi) in enumerate(offsets) if lo >= len(previous) and hi <= len(prefix)
                        and hi > lo and encoded["input_ids"][j] == eot_id]
                if not ends or not all(encoded["assistant_masks"][j] for j in ends):
                    raise ValueError("assistant_end_token_not_supervised:" + fp)
            previous = prefix
        inspections.append({"fingerprint": fp, "split": item["meta"]["split"], "reasons": sorted(reasons[fp]),
                            "tokens": detail["tokens"], "segments": detail["segments"], "all_assistant_end_tokens_supervised": True})
    if not inspections:
        raise ValueError("no_auditable_examples")
    report["mask_status"] = "passed"
    report["mask_summary"] = {"audited": len(inspections), "seed": seed,
                              "coverage": "random_and_each_available_behavior_and_all_smoke"}
    write_json(root / "preparation-report.json", report)
    result = {"status": "passed", "version": VERSION, "seed": seed, "tokenizer": info,
              "preparation_report_sha256": file_hash(root / "preparation-report.json"),
              "audited": len(inspections), "absent_behaviors": [b for b in BEHAVIORS if not any(b in i["meta"]["behavior_tags"] for i in items)],
              "coverage": "random plus every available behavior in both splits plus every local smoke example",
              "training_started": False, "examples": inspections}
    write_json(root / "metadata/mask-audit.json", result)
    return {k: v for k, v in result.items() if k != "examples"}


def training_items(config, split, tokenizer, template):
    """Read the new sidecar contract without injecting any metadata into model messages."""
    root = Path(config["preparation_report"]).resolve().parent
    report = json.loads((root / "preparation-report.json").read_text(encoding="utf-8"))
    if report["version"] != VERSION:
        raise ValueError("wrong_unified_dataset_version")
    if report.get("mode") != "behavior_sampled":
        raise ValueError("inventory_requires_review_and_explicit_mix_before_training")
    mask_path = root / "metadata/mask-audit.json"
    mask_audit = json.loads(mask_path.read_text(encoding="utf-8"))
    if mask_audit.get("status") != "passed" or mask_audit.get("preparation_report_sha256") != file_hash(root / "preparation-report.json"):
        raise ValueError("unified_dataset_requires_current_mask_audit")
    effective = template or tokenizer.chat_template
    if hashlib.sha256(effective.encode()).hexdigest() != report["tokenizer"]["chat_template_sha256"]:
        raise ValueError("training_template_differs_from_preparation")
    for filename, expected in report["tokenizer"]["assets_sha256"].items():
        if file_hash(Path(config["model_name_or_path"]) / filename) != expected:
            raise ValueError("training_tokenizer_differs_from_preparation:" + filename)
    for package in ("transformers", "trl"):
        if importlib.metadata.version(package) != report["tokenizer"][package]:
            raise ValueError("training_tokenizer_stack_changed:" + package)
    paths = (Path(config["train_file"] if split == "train" else config["validation_file"]).resolve(),
             Path(config["train_metadata_file"] if split == "train" else config["validation_metadata_file"]).resolve())
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if report["asset_sha256"].get(relative) != file_hash(path):
            raise ValueError("unified_training_asset_hash_mismatch:" + relative)
    items = load_pair(root, paths[0], paths[1])
    for item in items:
        meta = item["meta"]
        if (meta.get("transform_version") != VERSION or meta.get("split") != split or not meta.get("group_id")
                or not meta.get("provenance") or not meta.get("license") or meta.get("content_flags")):
            raise ValueError("invalid_unified_training_metadata")
        if meta["source"] == "self":
            from wearable_tasks.exporting import assert_public
            assert_public(item["record"])
            if not meta.get("replay_success") or meta.get("terminal_reward") != 1:
                raise ValueError("unverified_self_training_record")
        elif meta["source"] not in PUBLIC_SOURCES:
            raise ValueError("unknown_unified_source")
    return items


def main():
    from .console import utf8_console
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Convert, replay self data, deduplicate, group, split and count real tokens; no sampling")
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokenizer", required=True, help="Existing local Qwen tokenizer directory; never downloaded")
    p.add_argument("--output", required=True)
    p = sub.add_parser("mix", help="Optional sampling after reviewing inventory statistics")
    p.add_argument("--dataset", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("recount", help="Recount a hash-verified pool with real input_ids; preserve records and splits")
    p.add_argument("--dataset", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("audit-mask", help="Check real Qwen masks without loading model weights or training")
    p.add_argument("--dataset", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-count", type=int, default=32)
    p.add_argument("--per-behavior", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(json.loads(Path(args.manifest).read_text(encoding="utf-8-sig")), args.output, args.tokenizer)
    elif args.command == "mix":
        result = mix(args.dataset, args.output, json.loads(Path(args.policy).read_text(encoding="utf-8-sig")))
    elif args.command == "recount":
        result = recount(args.dataset, args.output, args.tokenizer)
    else:
        if args.random_count < 1 or args.per_behavior < 1:
            parser.error("audit sample counts must be positive")
        result = audit(args.dataset, args.tokenizer, args.seed, args.random_count, args.per_behavior)
    print(canonical({k: result[k] for k in ("version", "mode", "stages", "mixture_result", "local_smoke", "status", "audited", "training_started") if k in result}))


if __name__ == "__main__":
    main()
