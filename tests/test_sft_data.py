"""Prepared for manual execution. These do not claim to validate the actual Qwen tokenizer."""
import copy
import json

import pytest
from jsonschema.exceptions import ValidationError

from lightagent.sft_data import (Conversation, annotate, build_groups, call_list, convert_public,
                                 load_pair, model_fingerprint, safe_value, select_public,
                                 token_count, tools_from, validate, write_pair)


def tool(name="Weather API"):
    return {"name": name, "description": "weather", "parameters": {"type": "dict", "properties": {
        "city": {"type": "string"}, "days": {"type": "int"}}, "required": ["city"], "additionalProperties": False}}


def sample():
    return {"id": "one", "tools": json.dumps([tool()]), "conversations": [
        {"from": "system", "value": "source protocol"}, {"from": "human", "value": "Weather in Hangzhou?"},
        {"from": "gpt", "value": '<tool_call>{"name":"Weather API","arguments":{"city":"Hangzhou"}}</tool_call>'},
        {"from": "tool", "value": '<tool_response>{"name":"Weather API","content":{"condition":"rain"}}</tool_response>'},
        {"from": "gpt", "value": "It is raining."}]}


def item(record, source="glaive", fixed=None, original_group=None):
    meta = {"source": source, "source_id": "one", "fingerprint": model_fingerprint(record),
            "provenance": [{"source": source, "fixed_split": fixed, "original_group_id": original_group}]}
    annotate(record, meta)
    return {"record": record, "meta": meta}


def test_json_and_restricted_python_literals():
    assert safe_value('{"a":true,"b":null}') == {"a": True, "b": None}
    assert safe_value("{'a': True, 'b': [-2, None]}") == {"a": True, "b": [-2, None]}
    for value in ('{"x":1,"x":2}', "{'x':1,'x':2}", "__import__('os').getcwd()",
                  "[x for x in [1]]", "{'x': (1,2)}", "{'x': float('nan')}"):
        with pytest.raises((ValueError, SyntaxError)):
            safe_value(value)


def test_toolace_safe_calls_and_type_normalization():
    tools, aliases = tools_from([tool()])
    calls = call_list('[Weather API(city="Hangzhou", days=2)]', aliases)
    assert calls[0]["name"] == "Weather_API"
    assert calls[0]["arguments"]["days"] == 2
    assert tools[0]["function"]["parameters"]["properties"]["days"]["type"] == "integer"
    for text in ('[Weather API(city=__import__("os"))]', '[Weather API(**{})]', '[Weather API("Hangzhou")]'):
        with pytest.raises((ValueError, SyntaxError)):
            call_list(text, aliases)


def test_hermes_normalizes_definition_call_response_and_preserves_error_recovery():
    row = sample()
    row["conversations"][3]["value"] = '<tool_response>{"name":"Weather API","content":{"status":"error","error_code":"TEMPORARY"}}</tool_response>'
    row["conversations"][4:4] = copy.deepcopy(row["conversations"][2:4])
    row["conversations"][5]["value"] = '<tool_response>{"name":"Weather API","content":{"condition":"rain"}}</tool_response>'
    record, meta = convert_public(row, "hermes_multi")
    assert set(record) == {"tools", "messages"}
    assert meta["name_mapping"] == {"Weather API": "Weather_API"}
    assert record["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"city": "Hangzhou"}
    assert [m["name"] for m in record["messages"] if m["role"] == "tool"] == ["Weather_API", "Weather_API"]
    assert "TEMPORARY" in record["messages"][3]["content"]
    assert record["messages"][4]["tool_calls"][0]["id"] == "call_0002"


def test_parallel_same_name_needs_explicit_response_ids():
    tools, aliases = tools_from([tool()])
    conv = Conversation(tools, aliases, "system")
    conv.text("user", "Check two cities")
    conv.calls([{"id": "a", "name": "Weather API", "arguments": {"city": "A"}},
                {"id": "b", "name": "Weather API", "arguments": {"city": "B"}}])
    with pytest.raises(ValueError, match="ambiguous"):
        conv.response({}, name="Weather API")
    conv.response({"city": "B"}, name="Weather API", identity="b")
    conv.response({"city": "A"}, name="Weather API", identity="a")
    conv.text("assistant", "Done")
    assert conv.record()["messages"][3]["tool_call_id"] == "call_0002"


def test_do_not_pair_parallel_results_by_position():
    tools, aliases = tools_from([tool(), tool("Other API")])
    conv = Conversation(tools, aliases, "system")
    conv.text("user", "Check")
    conv.calls([{"name": name, "arguments": {"city": "A"}} for name in ("Weather API", "Other API")])
    with pytest.raises(ValueError, match="ambiguous"):
        conv.responses('[{"city":"A"},{"city":"B"}]')
    with pytest.raises(ValueError, match="before_all_responses"):
        conv.text("user", "Another request")


def test_call_only_is_valid_but_not_feedback():
    row = sample()
    row["conversations"] = row["conversations"][:3]
    record, _ = convert_public(row, "hermes_single")
    meta = item(record, "hermes_single")["meta"]
    assert meta["primary_behavior"] == "selection"
    assert "call_generation_only" in meta["behavior_tags"]
    assert "feedback" not in meta["behavior_tags"]


def test_schema_mismatch_is_rejected_without_coercion():
    row = sample()
    row["conversations"][2]["value"] = '<tool_call>{"name":"Weather API","arguments":{"city":"X","days":"2"}}</tool_call>'
    with pytest.raises(ValidationError, match="not of type"):
        convert_public(row, "hermes_multi")


def test_normalized_tool_name_collision_is_rejected():
    with pytest.raises(ValueError, match="collision"):
        tools_from([tool("A-B"), tool("A B")])


def test_single_multi_prefix_family_has_one_split():
    full, _ = convert_public(sample(), "hermes_multi")
    short = {"tools": copy.deepcopy(full["tools"]), "messages": copy.deepcopy(full["messages"][:3])}
    items = [item(full, "hermes_multi"), item(short, "hermes_single")]
    report = build_groups(items, 42, .1)
    assert items[0]["meta"]["group_id"] == items[1]["meta"]["group_id"]
    assert items[0]["meta"]["split"] == items[1]["meta"]["split"]
    assert report["relations"]["hermes_single_multi_prefix"] == 1


def test_private_groups_preserve_original_splits_with_shared_api():
    full, _ = convert_public(sample(), "hermes_multi")
    other = copy.deepcopy(full)
    other["messages"][1]["content"] = "A different request"
    items = [item(full, "self", "train", "family-a"), item(other, "self", "test_id", "family-b")]
    build_groups(items, 42, .1)
    assert [i["meta"]["split"] for i in items] == ["train", "test_id"]


def test_cross_split_exact_duplicate_fails_instead_of_moving_self_cases():
    full, _ = convert_public(sample(), "hermes_multi")
    items = [item(full, "self", "train", "family-a"), item(copy.deepcopy(full), "self", "validation", "family-b")]
    with pytest.raises(ValueError, match="self_splits_conflict"):
        build_groups(items, 42, .1)


def test_no_call_chat_is_not_automatically_a_clarification_or_refusal():
    tools, _ = tools_from([tool()])
    record = {"tools": tools, "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
    meta = item(record)["meta"]
    assert meta["primary_behavior"] == "unreviewed_no_call"
    assert not meta["sampling_eligible"]
    review = {"fingerprint": model_fingerprint(record), "reviewer": "human", "primary_behavior": "no_call",
              "evidence": {"reason": "no_tool_needed"}}
    annotate(record, meta, review)
    assert meta["primary_behavior"] == "no_call"


def test_clarification_review_requires_evidence_and_cannot_follow_guessed_call():
    record, _ = convert_public(sample(), "hermes_multi")
    review = {"fingerprint": model_fingerprint(record), "reviewer": "human", "primary_behavior": "clarification",
              "tool_name": "Weather_API", "missing_parameters": ["city"], "question_message_index": 4,
              "evidence": {"request_lacks_required_values": True, "question_is_relevant": True}}
    with pytest.raises(ValueError, match="guessed_tool_call"):
        annotate(record, {"source": "glaive"}, review)
    review["fingerprint"] = "stale"
    with pytest.raises(ValueError, match="matching_fingerprint"):
        annotate(record, {"source": "glaive"}, review)


def test_shortfalls_redistribute_without_copying_or_relabeling():
    record, _ = convert_public(sample(), "hermes_multi")
    items = []
    for n in range(3):
        r = copy.deepcopy(record)
        r["messages"][1]["content"] += str(n)
        items.append(item(r))
    chosen, report = select_public(items, 10, {"selection": .3, "feedback": .5, "clarification": .1, "no_call": .1},
                                   42, {"glaive": 3}, 1)
    assert len(chosen) == 3
    assert report["total_shortfall"] == 7
    assert report["initial_behavior_shortfalls"]["clarification"] == 1
    assert report["actual_behaviors"] == {"feedback": 3}
    assert len({i["meta"]["fingerprint"] for i in chosen}) == 3


def test_sidecars_are_aligned_and_cannot_silently_attach_to_another_row(tmp_path):
    record, _ = convert_public(sample(), "hermes_multi")
    original = item(record)
    write_pair(tmp_path, "train.jsonl", "metadata/train.jsonl", [original])
    assert load_pair(tmp_path, "train.jsonl", "metadata/train.jsonl")[0]["record"] == record
    record["messages"][-1]["content"] = "Changed"
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="alignment_or_fingerprint"):
        load_pair(tmp_path, "train.jsonl", "metadata/train.jsonl")


def test_metadata_cannot_be_written_inside_model_input():
    record, _ = convert_public(sample(), "hermes_multi")
    record["metadata"] = {"source": "glaive"}
    with pytest.raises(ValueError, match="model_input_fields"):
        validate(record)


def test_real_local_qwen_masks_system_parallel_same_name_responses_and_payloads():
    """Run when the actual tokenizer/TRL are available; never replace them with a fake."""
    from pathlib import Path
    path = Path("models/Qwen3.5-2B")
    if not path.exists():
        pytest.skip("actual local Qwen tokenizer not available")
    pytest.importorskip("trl")
    from lightagent.training import encode_record, load_tokenizer
    tools, aliases = tools_from([tool()])
    conv = Conversation(tools, aliases, "Use actual tool results.")
    conv.text("user", "Check Hangzhou and Beijing")
    conv.calls([{"id": "first", "name": "Weather API", "arguments": {"city": "Hangzhou"}},
                {"id": "second", "name": "Weather API", "arguments": {"city": "Beijing"}}])
    conv.response({"condition": "rain"}, name="Weather API", identity="second")
    conv.response({"condition": "sunny"}, name="Weather API", identity="first")
    conv.text("assistant", "Hangzhou is sunny; Beijing is rainy.")
    tokenizer, template = load_tokenizer({"model_name_or_path": str(path)})
    encoded, detail = encode_record(tokenizer, conv.record(), template, 32768)
    assert token_count(tokenizer, conv.record(), template) == len(encoded["input_ids"])
    assert "Hangzhou" in detail["segments"][2]["supervised_text"]
    assert "Beijing" in detail["segments"][2]["supervised_text"]
    assert all(s["supervised_tokens"] == 0 for s in detail["segments"] if s["role"] != "assistant")
    assert all("<|im_end|>" in s["supervised_text"] for s in detail["segments"] if s["role"] == "assistant")
    assert encoded["assistant_masks"]


def test_token_count_uses_input_ids_instead_of_number_of_encoding_fields():
    class MappingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is True
            assert kwargs["truncation"] is False
            return {"input_ids": [10, 11, 12, 13, 14], "attention_mask": [1, 1, 1, 1, 1]}
    assert token_count(MappingTokenizer(), {"tools": [], "messages": []}, "template") == 5
