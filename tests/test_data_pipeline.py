import copy
import json

import pytest
from jsonschema.exceptions import ValidationError

from lightagent.data_pipeline import (convert_toolace, convert_xlam, normalize_tools,
    parse_toolace_calls, prepare, signature_group, validate_record)


def definition(name="Get Weather"):
    return {"name": name, "parameters": {"type": "dict", "properties": {"city": {"type": "string"}, "days": {"type": "int"}}, "required": ["city"], "additionalProperties": False}}


def toolace():
    return {"system": "Here is a list of functions in JSON format:\n" + json.dumps([definition()]) + ". Only use listed tools.",
            "conversations": [{"from": "user", "value": "查杭州天气"},
             {"from": "assistant", "value": '[Get Weather(city="杭州", days=2)]'},
             {"from": "tool", "value": '[{"name":"Get Weather","results":{"condition":"rain"}}]'},
             {"from": "assistant", "value": "查询返回下雨。"}]}


def test_toolace_is_structured_and_tool_results_are_paired():
    record = convert_toolace(toolace())
    assert record["tools"][0]["function"]["name"] == "Get_Weather"
    call = record["messages"][2]["tool_calls"][0]
    assert call["function"]["arguments"] == {"city": "杭州", "days": 2}
    assert record["messages"][3]["tool_call_id"] == call["id"]
    assert record["messages"][3]["role"] == "tool"
    assert record["messages"][4]["content"] == "查询返回下雨。"


@pytest.mark.parametrize("text", [
    '[Get Weather(city=__import__("os").getcwd())]',
    '[Get Weather(city="杭州", **{"days":2})]',
    '[unknown(city="杭州")]',
    '[Get Weather("杭州")]',
    '[Get Weather(city=(1+2))]',
    '[Get Weather(city="杭州", days=float("nan"))]',
    '[Get Weather(city="杭州", city="上海")]',
])
def test_public_call_parser_never_executes_expressions(text):
    with pytest.raises((ValueError, SyntaxError)):
        parse_toolace_calls(text, {"Get Weather": "Get_Weather"})


def test_name_rewriting_does_not_change_argument_strings():
    calls = parse_toolace_calls('[Get Weather(city="Get Weather(x)")]', {"Get Weather": "Get_Weather"})
    assert calls[0]["function"]["arguments"]["city"] == "Get Weather(x)"


def test_type_errors_and_broken_tool_order_are_rejected():
    raw = toolace()
    raw["conversations"][1]["value"] = '[Get Weather(city="杭州", days="two")]'
    with pytest.raises(ValidationError):
        convert_toolace(raw)
    record = convert_toolace(toolace())
    record["messages"][3]["tool_call_id"] = "wrong"
    with pytest.raises(ValueError, match="对应"):
        validate_record(record)


def test_xlam_native_tool_call_and_conservative_grouping():
    a = convert_xlam({"query": "杭州天气", "tools": json.dumps([definition()]), "answers": '[{"name":"Get Weather","arguments":{"city":"杭州"}}]'})
    b = copy.deepcopy(a)
    b["messages"][1]["content"] = "换一种说法查询上海天气"
    b["messages"][2]["tool_calls"][0]["function"]["arguments"]["city"] = "上海"
    b["tools"].append({"type": "function", "function": {"name": "distractor", "parameters": {"type": "object"}}})
    assert signature_group(a) == signature_group(b)
    assert a["messages"][-1]["role"] == "assistant"
    assert isinstance(a["messages"][-1]["tool_calls"][0]["function"]["arguments"], dict)


def test_public_preparation_deduplicates_and_never_splits_a_signature_group(tmp_path):
    rows = [toolace(), toolace(), {"system": "bad", "conversations": []}]
    raw = tmp_path / "raw.json"
    raw.write_text("\n  " + json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    manifest = {"inputs": [{"source": "toolace", "path": str(raw), "revision": "a"*40}], "seed": 42}
    report = prepare(manifest, tmp_path / "out")
    assert report["accepted_before_sampling"] == 1 and len(report["rejected"]) == 1
    exported = [json.loads(line) for file in (tmp_path / "out").glob("*.jsonl") for line in file.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 1
    assert exported[0]["metadata"]["license"] == "Apache-2.0"
    assert exported[0]["metadata"]["revision"] == "a"*40
    assert not report["length_checked"]
    with pytest.raises(ValueError, match="revision"):
        prepare({"inputs": [{"source": "toolace", "path": str(raw), "revision": "main"}]}, tmp_path / "bad")
