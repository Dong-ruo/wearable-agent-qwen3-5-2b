"""Scripted-model integration fixtures, never model capability measurements."""
import copy
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from wearable_tasks.contract import canonical
from wearable_tasks.pipeline import source_fingerprint
from wearable_tasks.reference import run_reference
from wearable_tasks.tests.helpers import case
from wearable_assistant.backends import TaskBackend, LocalBackend
from wearable_assistant.evaluation import evaluate, load_config, replay, sha, write_json
from wearable_assistant.protocol import ChatEndpoint, ProtocolError, Reply, TransportError, parse_message
from wearable_assistant.runtime import Session
from wearable_assistant.service import SessionManager, make_handler
from wearable_assistant.model_server import normalize_messages, parse_generated


class Scripted:
    def __init__(self, actions):
        self.actions, self.payloads = copy.deepcopy(actions), []

    def complete(self, messages, tools, seed):
        self.payloads.append(copy.deepcopy({"messages": messages, "tools": tools, "seed": seed}))
        if not self.actions: raise AssertionError("Scripted fixture exhausted")
        action = self.actions.pop(0)
        if action["type"] == "tool_call":
            message = {"tool_calls": [{"function": {"name": action["name"], "arguments": canonical(action["arguments"])}}]}
        else: message = {"content": canonical(action)}
        return Reply(message, {"total_tokens": 10, "completion_tokens": 5}, .001)


def tool(name, **args): return {"type": "tool_call", "name": name, "arguments": args}
def final(result=None, status="success"): return {"type": "final", "status": status, "message": "完成", "result": result or {}}


@pytest.fixture
def config():
    config = load_config("configs/assistant.json")
    config["runtime"]["local_fast_path"] = False  # Existing fixtures exercise model proposals.
    return config


def dataset(tmp_path, tasks):
    root = tmp_path / "dataset"
    for split in ("train", "val", "test_id", "test_composition", "test_robustness"):
        selected = [t for t in tasks if t["split"] == split]
        for relative, rows in [(f"private/{split}.jsonl", selected),
                               (f"public/grpo/{split}.jsonl", [{"case_id": t["case_id"]} for t in selected])]:
            p = root / relative; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")
    write_json(root / "validation-report.json", {"status": "passed", "source_fingerprint": source_fingerprint(),
               "file_sha256": {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*.jsonl")}})
    return root


@pytest.mark.parametrize("content", ['{"status":"success","status":"failed","result":{},"message":"x"}',
                                      '{"status":"success","result":{"a":NaN},"message":"x"}',
                                      "__import__('os').system('bad')",
                                      '{"type":"ask_user","requested_slots":[{}],"question":"x"}'])
def test_unsafe_or_ambiguous_protocol_rejected(content):
    with pytest.raises(ProtocolError): parse_message({"content": content})


def test_parallel_rejected_before_any_execution():
    with pytest.raises(ProtocolError): parse_message({"tool_calls": [{}, {}]})
    assert parse_message({"content": "请问持续多少分钟？"})["requested_slots"] == ["duration_minutes"]


@pytest.mark.parametrize("kind", ["weather_query", "events_query", "create", "existing_reminder", "create_reminder", "update", "delete", "weather_activity", "preferences_activity", "clarify_time", "clarify_duration", "create_reschedule", "create_cancel", "conflict_fallback", "weather_reschedule_remind"])
def test_reference_action_fixture_through_public_runner(tmp_path, config, kind):
    task = case(kind)
    task["goal"]["private_canary"] = "DO_NOT_PASS_PRIVATE_GOAL_TO_MODEL"
    reference = run_reference(task)
    actions = [r["action"] for r in reference.trajectory()["actions"]]
    root = dataset(tmp_path, [task])
    backend = TaskBackend(root, task["case_id"])
    endpoint = Scripted(actions)
    session = Session(backend, endpoint, config["runtime"])
    session.run()
    assert backend.offline_evaluate()["success"], session.public()
    assert "DO_NOT_PASS_PRIVATE_GOAL_TO_MODEL" not in canonical(endpoint.payloads)
    assert set(endpoint.payloads[0]["tools"][0]["function"]["parameters"])
    assert len(endpoint.payloads[0]["tools"]) == 8
    snapshot = session.snapshot()
    restored = Session.restore(snapshot, Scripted([]))
    assert restored.public() == session.public()


def test_transient_read_retry_and_protocol_ids(tmp_path, config):
    task = case("retry_weather")
    root = dataset(tmp_path, [task]); args = task["goal"]["query"]["arguments"]
    expected = {**args, "condition": task["initial_state"]["weather"][args["city"]+"|"+args["date"]]}
    session = Session(TaskBackend(root, task["case_id"]), Scripted([tool("get_weather", **args), final(expected)]), config["runtime"])
    session.run()
    assert session.backend.offline_evaluate()["success"]
    assert session.counters["retries"] == 1 and session.counters["tool_calls"] == 2
    calls = [m["tool_calls"][0]["id"] for m in session.messages if m.get("tool_calls")]
    returns = [m["tool_call_id"] for m in session.messages if m["role"] == "tool"]
    assert calls == returns and len(set(calls)) == len(calls)


def test_weather_query_finishes_from_first_receipt_without_second_model_call(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    endpoint = Scripted([tool("get_weather", city="杭州", date="2026-11-19")])
    s = Session(LocalBackend(world, "查询杭州明天的天气。"), endpoint, config["runtime"])
    s.run()
    assert s.final == {"status": "success", "message": "本地模拟天气：杭州 2026-11-19 有雨。",
                       "result": {"city": "杭州", "date": "2026-11-19", "condition": "rain"}}
    assert s.counters["model_calls"] == s.counters["tool_calls"] == 1
    assert s.raw_final is None and s.final_origin == "controller_weather"
    assert s.public()["token_usage"]["reported_total_tokens"] == 10
    restored = Session.restore(s.snapshot(), Scripted([]))
    assert restored.public() == s.public()


def test_invented_activity_id_resolves_from_real_catalog_before_verified_create(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    world["state"]["activities"][0]["activity_id"] = "jog_772"
    action = tool("create_event", activity_id="running_outdoor", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)
    s = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步。"), Scripted([action]), config["runtime"])
    s.run()
    assert s.final["status"] == "success" and s.final_origin == "controller_verified_write"
    assert s.raw_final is None and s.counters["model_calls"] == 1
    assert [e["name"] for e in s.tool_events] == ["search_activity", "create_event", "list_events"]
    write = s.tool_events[1]
    assert write["arguments"]["activity_id"] == write["result"]["data"]["activity_id"] == "jog_772"
    assert write["arguments"]["duration_minutes"] == action["arguments"]["duration_minutes"]
    repair = next(e for e in s.timeline if e["type"] == "argument_repair")
    assert repair["proposed_action"]["arguments"]["activity_id"] == "running_outdoor"
    assert repair["executed_action"]["arguments"]["activity_id"] == "jog_772"
    assert Session.restore(s.snapshot(), Scripted([])).public() == s.public()


def test_ambiguous_activity_catalog_is_not_guessed(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    world["state"]["activities"].append({**world["state"]["activities"][0], "activity_id": "second_run"})
    action = tool("create_event", activity_id="running_outdoor", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)
    s = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步。"), Scripted([action] * 3), config["runtime"])
    s.run()
    assert s.final["status"] == "failed" and not s._has_writes()
    assert not any(e["type"] == "argument_repair" for e in s.timeline)
    assert s.state["consecutive_tool_failures"] == 3


def test_create_and_reminder_request_is_not_completed_after_create_only(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    action = tool("create_event", activity_id="running_outdoor", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)
    s = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步，并提前10分钟提醒我。"),
                Scripted([action, final(status="failed")]), config["runtime"])
    s.run()
    assert s.final["status"] == "partial" and s.final_origin == "model"
    assert len(s.final["result"]["executed_writes"]) == 1


@pytest.mark.parametrize("request_text,action,origin", [
    ("查看我的运动偏好", tool("get_preferences"), "controller_query"),
    ("查询明天的日程。", tool("list_events", start_time="2026-11-19T00:00:00+08:00", end_time="2026-11-20T00:00:00+08:00"), "controller_query")])
def test_simple_query_finishes_from_real_receipt(config, request_text, action, origin):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    s = Session(LocalBackend(world, request_text), Scripted([action]), config["runtime"])
    s.run()
    assert s.final["status"] == "success" and s.final_origin == origin
    assert s.counters["model_calls"] == s.counters["tool_calls"] == 1
    assert s.final["result"] == s.tool_events[0]["result"]["data"] and s.raw_final is None


def test_complex_preferences_request_is_not_completed_by_one_read(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    s = Session(LocalBackend(world, "查看我的运动偏好，再按偏好安排活动。"),
                Scripted([tool("get_preferences"), final(status="failed")]), config["runtime"])
    s.run()
    assert s.counters["model_calls"] == 2 and s.final["status"] == "failed"


def test_wrong_time_provider_receipt_cannot_be_directly_completed(config):
    from wearable_assistant.assistance import simple_create_matches
    messages = [{"role": "user", "content": "明天18:00安排30分钟户外跑步。"}]
    data = {"title": "户外跑步", "start_time": "2026-11-19T20:00:00+08:00", "duration_minutes": 30}
    assert not simple_create_matches(messages, data, "2026-11-18T09:00:00+08:00")
    data["start_time"] = "2026-11-19T18:00:00+08:00"
    assert simple_create_matches(messages, data, "2026-11-18T09:00:00+08:00")


@pytest.mark.parametrize("user_request", ["查询杭州明天和后天的天气。", "查询杭州明天的天气，然后安排30分钟跑步。",
                                     "查询杭州明天的天气，告诉我穿衣建议。", "查询杭州和上海明天的天气。"])
def test_compound_query_never_auto_completes_after_one_weather_receipt(config, user_request):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    s = Session(LocalBackend(world, user_request), Scripted([tool("get_weather", city="杭州", date="2026-11-19"), final(status="failed")]), config["runtime"])
    s.run()
    assert s.final["status"] == "failed" and s.final_origin == "model"
    assert s.counters["model_calls"] == 2


def test_weather_receipt_cannot_close_wrong_date_or_wrong_returned_facts(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    s = Session(LocalBackend(world, "查询杭州明天的天气。"), Scripted([tool("get_weather", city="杭州", date="2026-11-18"), final(status="failed")]), config["runtime"])
    s.run()
    assert s.final["status"] == "failed" and s.counters["tool_calls"] == 0
    assert s._direct_weather_final(tool("get_weather", city="杭州", date="2026-11-19"),
        {"status": "success", "data": {"city": "杭州", "date": "2026-11-18", "condition": "rain"}}) is None


def test_successful_read_stops_on_third_attempt_without_fourth_model_call(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    action = tool("get_weather", city="杭州", date="2026-11-19")
    s = Session(LocalBackend(world, "查询杭州明天和后天的天气。"), Scripted([action] * 3), config["runtime"])
    s.run()
    assert s.counters["tool_calls"] == 1 and s.counters["reused_reads"] == 2
    assert s.counters["model_calls"] == 3 and s.final["status"] == "failed"
    assert s.final_origin == "controller_stop" and s.raw_final is None
    assert "相同只读调用" in s.final["message"]
    assert s.messages[-1]["role"] == "assistant"
    assert json.loads(s.messages[-1]["content"]) == s.final
    calls = [m["tool_calls"][0]["id"] for m in s.messages if m.get("tool_calls")]
    returns = [m["tool_call_id"] for m in s.messages if m["role"] == "tool"]
    assert calls == returns and len(set(calls)) == 3
    restored = Session.restore(s.snapshot(), Scripted([]))
    assert restored.public() == s.public()


def test_failed_reads_are_not_cached_and_repetitions_are_bounded(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    action = tool("get_weather", city="北京", date="2026-11-19")
    s = Session(LocalBackend(world, "查询北京明天的天气。"), Scripted([action] * 3), config["runtime"])
    s.run()
    assert s.counters["tool_calls"] == 3 and s.counters["reused_reads"] == 0
    assert s.counters["model_calls"] == 3
    assert s.final["status"] == "failed"
    assert "未找到北京 2026-11-19" in s.final["message"]
    assert s.final["result"]["error_code"] == "CONSECUTIVE_TOOL_FAILURE_LIMIT"


def test_nanjing_changed_dates_stop_after_three_errors_including_guard_blocks(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    actions = [tool("get_weather", city="南京", date=f"2026-11-{day:02d}") for day in (19, 18, 17, 16)]
    endpoint = Scripted(actions)
    s = Session(LocalBackend(world, "查询南京明天天气"), endpoint, config["runtime"])
    s.run()
    assert s.counters["model_calls"] == 3 and len(endpoint.actions) == 1
    assert s.counters["tool_calls"] == 1 and s.counters["guard_blocks"] == 2
    assert s.final["status"] == "failed" and s.state["consecutive_tool_failures"] == 3
    assert "未找到南京 2026-11-19" in s.final["message"] and "连续 3 次" in s.final["message"]
    assert [e["error_code"] for e in s.final["result"]["errors"]] == ["NOT_FOUND", "DATE_NOT_PROVIDED_BY_USER", "DATE_NOT_PROVIDED_BY_USER"]
    assert json.loads(s.messages[-1]["content"]) == s.final
    calls = [m["tool_calls"][0]["id"] for m in s.messages if m.get("tool_calls")]
    assert calls == [m["tool_call_id"] for m in s.messages if m["role"] == "tool"]


def test_changed_tool_and_error_code_do_not_reset_consecutive_failures(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    actions = [tool("get_weather", city="南京", date="2026-11-19"),
               tool("get_weather", city="上海", date="2026-11-19"),
               tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)]
    s = Session(LocalBackend(world, "查询南京明天天气"), Scripted(actions), config["runtime"])
    s.run()
    assert s.final["status"] == "failed" and s.state["consecutive_tool_failures"] == 3
    assert s.counters["tool_calls"] == 1 and s.counters["guard_blocks"] == 2
    assert not s._has_writes()


def test_consecutive_failure_count_survives_restore_and_resets_on_new_user(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    s = Session(LocalBackend(world, "查询南京明天天气"), Scripted([]), config["runtime"])
    for day in (19, 18): s._tool(tool("get_weather", city="南京", date=f"2026-11-{day}"), lambda _: None)
    restored = Session.restore(s.snapshot(), Scripted([tool("get_weather", city="南京", date="2026-11-17")]))
    restored.run()
    assert restored.state["consecutive_tool_failures"] == 3 and restored.counters["model_calls"] == 1
    assert restored.final["result"]["attempts"] == 3
    again = Session.restore(restored.snapshot(), Scripted([tool("get_weather", city="杭州", date="2026-11-19"),
                                                          final({"city": "杭州", "date": "2026-11-19", "condition": "rain"})]))
    again.user_message("查询杭州明天的天气。")
    assert again.state["consecutive_tool_failures"] == 0 and again._failure_history == []
    again.run()
    assert again.final["status"] == "success"


def test_successful_tool_resets_failure_streak_and_allows_useful_progress(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    missing = [tool("get_weather", city="南京", date=f"2026-11-{day}") for day in (19, 20)]
    actions = missing + [tool("get_weather", city="杭州", date="2026-11-19")] + missing + [final(status="failed")]
    s = Session(LocalBackend(world, "查询南京和杭州明天及后天的天气，并汇总。"), Scripted(actions), config["runtime"])
    s.run()
    assert s.counters["model_calls"] == 6 and s.counters["tool_calls"] == 5
    assert s.state["consecutive_tool_failures"] == 2 and s.final_origin == "model"
    assert not any(e["type"] == "failure_limit" for e in s.timeline)


def test_third_failure_stops_before_an_automatic_fourth_tool_retry(config, monkeypatch):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    backend = LocalBackend(world, "查询南京明天天气")
    original = backend.step
    def temporarily_unavailable(action):
        response = original(action)
        response["observation"]["tool_result"]["error_code"] = "TEMPORARY_UNAVAILABLE"
        return response
    monkeypatch.setattr(backend, "step", temporarily_unavailable)
    action = tool("get_weather", city="南京", date="2026-11-19")
    s = Session(backend, Scripted([action, action]), config["runtime"])
    s.run()
    assert s.counters["tool_calls"] == 3 and s.counters["model_calls"] == 2
    assert s.counters["retries"] == 1 and s.final["result"]["attempts"] == 3
    assert "工具暂时不可用" in s.final["message"]


def test_failure_limit_preserves_existing_writes_as_partial_result(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    actions = [tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)]
    actions += [tool("get_weather", city="南京", date=f"2026-11-{day}") for day in (19, 18, 17)]
    s = Session(LocalBackend(world, "明天18:00安排30分钟跑步，再查询南京明天天气。"), Scripted(actions), config["runtime"])
    s.run()
    assert s.final["status"] == "partial" and "连续 3 次" in s.final["message"]
    assert len(s.final["result"]["executed_writes"]) == 1
    assert s.final["result"]["failure"]["attempts"] == 3
    assert json.loads(s.messages[-1]["content"]) == s.final


@pytest.mark.parametrize("value", [True, 0, 6, "3"])
def test_failure_limit_configuration_requires_bounded_integer(config, tmp_path, value):
    config["runtime"]["max_consecutive_tool_failures"] = value
    path = tmp_path / "config.json"
    write_json(path, config)
    with pytest.raises(ValueError, match="max_consecutive_tool_failures"): load_config(path)


def test_successful_write_invalidates_previous_read_cache(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    s = Session(LocalBackend(world, "明天18:00安排30分钟跑步。"), Scripted([]), config["runtime"])
    query = tool("list_events", start_time="2026-11-19T00:00:00+08:00", end_time="2026-11-20T00:00:00+08:00")
    _, before = s._tool(query, lambda _: None)
    assert before["data"]["events"] == []
    s._tool(tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30), lambda _: None)
    _, after = s._tool(query, lambda _: None)
    assert len(after["data"]["events"]) == 1 and s.counters["reused_reads"] == 0
    s.user_message("再查一次日程。")
    s._tool(query, lambda _: None)
    assert s.counters["reused_reads"] == 0 and s.counters["tool_calls"] == 4


@pytest.mark.parametrize("profile", ["baseline", "state"])
def test_query_shortcut_and_read_cache_do_not_change_control_profiles(config, profile):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    action = tool("get_weather", city="杭州", date="2026-11-19")
    s = Session(LocalBackend(world, "查询杭州明天的天气。"), Scripted([action, action, final({"city": "杭州", "date": "2026-11-19", "condition": "rain"})]), config["runtime"], profile)
    s.run()
    assert s.counters["tool_calls"] == 2 and s.counters["model_calls"] == 3
    assert s.final_origin == "model" and s.counters["reused_reads"] == 0


def test_guess_before_clarification_never_writes(tmp_path, config):
    task = case("clarify_duration"); root = dataset(tmp_path, [task])
    start = task["reference_intent"]["start_time"]
    session = Session(TaskBackend(root, task["case_id"]), Scripted([tool("create_event", activity_id="run", start_time=start, duration_minutes=30), final(status="failed")]), config["runtime"])
    session.run()
    assert session.counters["tool_calls"] == 0
    assert session.counters["guard_blocks"] == 1


def test_no_write_without_visible_authorization(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    request = "查询杭州明天的天气。"
    s = Session(LocalBackend(world, request), Scripted([tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30), final(status="failed")]), config["runtime"])
    s.run()
    assert s.counters["tool_calls"] == 0


def test_duplicate_write_is_blocked_but_receipt_kept(config):
    config["runtime"]["grounded_assistance"] = False  # Exercise model continuation after a write, not direct completion.
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    action = tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)
    s = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步。"), Scripted([action, action, final({"event_id": "event_00001"})]), config["runtime"])
    s.run()
    assert s.counters["duplicate_writes"] == 1
    assert sum(e["name"] == "create_event" for e in s.tool_events) == 1
    assert any(e["type"] == "write_verification" and e["passed"] for e in s.timeline)


def test_live_session_pause_restart_and_request_idempotency(tmp_path, config):
    endpoint = Scripted([{"type": "ask_user", "question": "请提供持续分钟数。", "requested_slots": ["duration_minutes"]}])
    db = tmp_path / "sessions.db"
    first = SessionManager(config, endpoint, db)
    result = first.create("明天18:00安排户外跑步，时长请先问我。", "new-1")
    sid = result["session_id"]
    assert result["state"]["phase"] == "waiting_for_user"
    first.store.close()
    endpoint = Scripted([tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30), final({"event_id": "event_00001"})])
    second = SessionManager(config, endpoint, db)
    done = second.message(sid, "持续30分钟。", "reply-1")
    assert done["final"]["status"] == "success"
    assert second.message(sid, "持续30分钟。", "reply-1") == done
    with pytest.raises(ValueError): second.message(sid, "持续40分钟。", "reply-1")
    assert second.create("明天18:00安排户外跑步，时长请先问我。", "new-1")["session_id"] == sid
    second.store.close()


def test_interrupted_write_is_not_resubmitted(tmp_path, config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    s = Session(LocalBackend(world, "明天18:00安排30分钟跑步。"), Scripted([]), config["runtime"])
    s.state["pending_call"] = tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)
    restored = Session.restore(s.snapshot(), Scripted([]))
    restored.run()
    assert restored.state["write_outcome_unknown"]
    assert restored.counters["model_calls"] == 0 and restored.counters["tool_calls"] == 0
    assert restored.final["status"] == "partial"


def test_evaluation_resume_and_receipt_replay(tmp_path, config, capsys):
    task = case("weather_query", split="test_id"); root = dataset(tmp_path, [task])
    config["dataset"] = str(root)
    endpoint = Scripted([r["action"] for r in run_reference(task).trajectory()["actions"]])
    out = tmp_path / "eval"
    result = evaluate(config, endpoint, out, splits=["test_id"])
    assert result["overall"]["successful"] == 1
    no_calls = Scripted([])
    resumed = evaluate(config, no_calls, out, splits=["test_id"], resume=True)
    assert resumed == result and not no_calls.payloads
    artifact = out / "cases" / (task["case_id"] + ".json")
    assert replay(artifact)["validation"]["success"]
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    saved["snapshot"]["backend"]["receipts"][0]["observation"]["tool_result"]["data"]["condition"] = "forged"
    write_json(artifact, saved)
    with pytest.raises(ValueError): replay(artifact)
    config["runtime"]["seed"] += 1
    with pytest.raises(ValueError): evaluate(config, no_calls, out, splits=["test_id"], resume=True)


def test_http_transport_native_calls_and_auth(tmp_path, config, monkeypatch):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            received.append((self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = canonical({"choices": [{"finish_reason": "stop", "message": {"content": canonical(final())}}], "usage": {"total_tokens": 10}}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        monkeypatch.setenv("MODEL_API_KEY", "fixture-secret")
        cfg = dict(config["endpoint"], base_url=f"http://127.0.0.1:{server.server_port}/v1")
        endpoint = ChatEndpoint(cfg)
        assert parse_message(endpoint.complete([{"role": "user", "content": "hi"}], [], 42).message)["type"] == "final"
        assert received[0][0] == "Bearer fixture-secret"
        assert received[0][1]["parallel_tool_calls"] is False
        assert "fixture-secret" not in canonical(received[0][1])
    finally: server.shutdown(); server.server_close(); thread.join()


def test_api_auth_ui_and_private_fields_not_exposed(tmp_path, config):
    manager = SessionManager(config, Scripted([final()]), tmp_path / "sessions.db")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(manager, "fixture-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert b'/web.js' in urllib.request.urlopen(base + "/").read()
        assert b'textContent' in urllib.request.urlopen(base + '/web.js').read()
        with pytest.raises(urllib.error.HTTPError) as exc: urllib.request.urlopen(base + "/health")
        assert exc.value.code == 401
        health_request = urllib.request.Request(base + "/health", headers={"Authorization": "Bearer fixture-token"})
        health = json.load(urllib.request.urlopen(health_request))
        assert health["controller_revision"] == "verified-completion-v7"
        assert health["runtime_limits"]["max_consecutive_tool_failures"] == 3
        assert health["runtime_limits"]["max_repeated_read_attempts"] == 3
        assert not manager.endpoint.payloads  # Checking the loaded policy must not request generation.
        request = urllib.request.Request(base + "/sessions", data=canonical({"request": "你好", "request_id": "new-1"}).encode(), headers={"Authorization": "Bearer fixture-token", "Content-Type": "application/json"})
        result = json.load(urllib.request.urlopen(request))
        assert "backend" not in result and "world" not in result and "goal" not in result
        assert result["final"]["status"] == "success"
    finally: server.shutdown(); server.server_close(); thread.join(); manager.store.close()


def test_qwen_native_xml_and_json_tools_use_schema_types():
    from wearable_tasks.contract import tool_schemas
    text = '<tool_call><function=create_event><parameter=activity_id>run</parameter><parameter=start_time>2026-11-19T18:00:00+08:00</parameter><parameter=duration_minutes>30</parameter></function></tool_call>'
    message = parse_generated(text, tool_schemas())
    assert parse_message(message)["arguments"]["duration_minutes"] == 30
    assert normalize_messages([message])[0]["tool_calls"][0]["function"]["arguments"]["activity_id"] == "run"
    with pytest.raises(ProtocolError): parse_generated(text + 'suffix', tool_schemas())
    bad = text.replace('</function>', '<parameter=duration_minutes>40</parameter></function>')
    with pytest.raises(ProtocolError): parse_generated(bad, tool_schemas())
    message = parse_generated('<tool_call>{"name":"get_preferences","arguments":{}}</tool_call>', tool_schemas())
    assert parse_message(message)["name"] == "get_preferences"


@pytest.mark.parametrize("literal,expected", [("False", False), ("True", True), ("false", False)])
def test_qwen_xml_accepts_python_boolean_literals_for_boolean_parameters(literal, expected):
    from wearable_tasks.contract import tool_schemas, validate_arguments
    raw = ("<tool_call>\n<function=search_activity>\n"
           f"<parameter=indoor>\n{literal}\n</parameter>\n"
           "<parameter=max_duration_minutes>\n30\n</parameter>\n</function>\n</tool_call>")
    action = parse_message(parse_generated(raw, tool_schemas()))
    assert action["arguments"]["indoor"] is expected
    assert action["arguments"]["max_duration_minutes"] == 30
    validate_arguments(action["name"], action["arguments"])
    with pytest.raises(ProtocolError):
        parse_generated(raw.replace(">\n30\n", ">\nFalse\n"), tool_schemas())


def test_cancel_during_model_call_prevents_proposed_write(tmp_path, config):
    started, proceed = threading.Event(), threading.Event()
    class Slow(Scripted):
        def complete(self, *args):
            started.set(); assert proceed.wait(5)
            return super().complete(*args)
    endpoint = Slow([tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)])
    manager = SessionManager(config, endpoint, tmp_path / 'cancel.db')
    result = manager.create('明天18:00安排30分钟跑步。', 'new-1', defer=True)
    results = []
    worker = threading.Thread(target=lambda: results.append(manager.run(result['session_id'], 'run-1')))
    worker.start()
    assert started.wait(5)
    assert manager.cancel(result['session_id'], 'cancel-1')['cancellation_requested']
    proceed.set(); worker.join(5)
    assert not worker.is_alive()
    assert results[0]['state']['phase'] == 'cancelled'
    assert results[0]['counters']['tool_calls'] == 0
    manager.store.close()


def test_catalog_does_not_override_missing_user_duration(tmp_path, config):
    task = case('clarify_duration'); root = dataset(tmp_path, [task])
    endpoint = Scripted([tool('search_activity', activity_id='run'), tool('create_event', activity_id='run',
                        start_time=task['reference_intent']['start_time'], duration_minutes=30), final(status='failed')])
    session = Session(TaskBackend(root, task['case_id']), endpoint, config['runtime'])
    session.run()
    assert not any(e['name']=='create_event' for e in session.tool_events)


def test_remote_write_unknown_not_replayed_and_has_stable_key(config):
    from wearable_assistant.backends import HTTPBackend, restore_backend
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            received.append((self.path, self.headers['Idempotency-Key']))
            self.send_response(503); self.end_headers()
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        backend = HTTPBackend({'base_url':f'http://127.0.0.1:{server.server_port}','now':'2026-11-18T09:00:00+08:00'},'明天18:00安排30分钟跑步。')
        endpoint = Scripted([tool('create_event',activity_id='run',start_time='2026-11-19T18:00:00+08:00',duration_minutes=30), final({'event_id':'invented'})])
        session = Session(backend,endpoint,config['runtime']);session.run()
        assert session.state['write_outcome_unknown'] and session.final['status']=='partial'
        writes = [key for path, key in received if path == '/tools/create_event']
        submitted_write = next(a for a in backend.actions if a.get('name') == 'create_event')
        assert writes == [submitted_write['request_id']]
        before_restore = list(received)
        restore_backend(session.snapshot()['backend'])
        assert received == before_restore
    finally: server.shutdown();server.server_close();thread.join()


def test_model_server_input_uses_actual_qwen_template_without_weights(config):
    pytest.importorskip('transformers')
    from transformers import AutoTokenizer
    from wearable_tasks.contract import tool_schemas
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3.5-2B',local_files_only=True)
    messages = [{'role':'user','content':'查询偏好。'},
        {'role':'assistant','content':'','tool_calls':[{'id':'call_1','type':'function','function':{'name':'get_preferences','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'call_1','name':'get_preferences','content':'{"preferences":{}}'}]
    rendered = tokenizer.apply_chat_template(normalize_messages(messages),tools=tool_schemas(),enable_thinking=False,tokenize=False,add_generation_prompt=True)
    assert '<function=get_preferences>' in rendered
    assert '"preferences":{}' in rendered


def test_empty_query_success_is_not_accepted(config):
    world = json.loads(Path(config['world_file']).read_text(encoding='utf-8'))
    session = Session(LocalBackend(world, '查询杭州明天天气。'), Scripted([final()]), config['runtime'])
    session.run()
    assert session.raw_final['status']=='success' and session.final['status']=='failed'
    assert any(e['type']=='final_block' for e in session.timeline)


def test_explicit_negation_blocks_even_well_formed_write(config):
    world = json.loads(Path(config['world_file']).read_text(encoding='utf-8'))
    endpoint = Scripted([tool('create_event',activity_id='run',start_time='2026-11-19T18:00:00+08:00',duration_minutes=30),final(status='failed')])
    session = Session(LocalBackend(world,'不要创建明天18:00的30分钟跑步日程，只查询天气。'),endpoint,config['runtime'])
    session.run()
    assert not session.tool_events


def test_city_clarification_in_interactive_session(config):
    world = json.loads(Path(config['world_file']).read_text(encoding='utf-8'))
    endpoint = Scripted([{'type':'ask_user','question':'请问查询哪个城市？','requested_slots':['city']},
        tool('get_weather',city='杭州',date='2026-11-19'),final({'city':'杭州','date':'2026-11-19','condition':'rain'})])
    session = Session(LocalBackend(world,'查询明天的天气。'),endpoint,config['runtime'])
    assert session.run()['state']['phase']=='waiting_for_user'
    session.user_message('city=杭州')
    assert session.run()['final']['status']=='success'


def test_partial_response_discloses_actual_writes_instead_of_model_prose(config):
    world=json.loads(Path(config['world_file']).read_text(encoding='utf-8'))
    bad=final(status='failed');bad['message']='本次什么都没有创建。'
    endpoint=Scripted([tool('create_event',activity_id='run',start_time='2026-11-19T18:00:00+08:00',duration_minutes=30),bad])
    session=Session(LocalBackend(world,'明天18:00安排30分钟跑步。'),endpoint,config['runtime'])
    session.run()
    assert session.final['status']=='partial'
    assert 'event_00001' in session.final['message'] and '什么都没有' not in session.final['message']
    assert session.final['result']['executed_writes'][0]['tool']=='create_event'
    assert session.raw_final['message']=='本次什么都没有创建。'


def test_readback_matching_wrong_provider_write_does_not_mean_success(config):
    class WrongProvider(LocalBackend):
        def step(self,action):
            action=copy.deepcopy(action)
            if action['type']=='tool_call' and action['name']=='create_event': action['arguments']['duration_minutes']=40
            return super().step(action)
    world=json.loads(Path(config['world_file']).read_text(encoding='utf-8'))
    endpoint=Scripted([tool('create_event',activity_id='run',start_time='2026-11-19T18:00:00+08:00',duration_minutes=30),final({'event_id':'event_00001'})])
    session=Session(WrongProvider(world,'明天18:00安排30分钟跑步。'),endpoint,config['runtime'])
    session.run()
    assert session.state['verification_failed'] and session.final['status']=='partial'
    assert session.final['result']['executed_writes'][0]['data']['duration_minutes']==40


def test_progress_read_does_not_wait_for_generation_or_replay_backend(tmp_path, config, monkeypatch):
    import time
    started, proceed = threading.Event(), threading.Event()
    class Slow(Scripted):
        def complete(self, *args):
            started.set(); assert proceed.wait(5)
            return super().complete(*args)
    manager = SessionManager(config, Slow([final()]), tmp_path/'progress.db')
    created = manager.create('你好', 'create-progress', defer=True)
    results=[]
    worker=threading.Thread(target=lambda:results.append(manager.run(created['session_id'],'run-progress')))
    worker.start(); assert started.wait(5)
    # get() must read the durable snapshot, not restore/replay an environment while executing.
    def forbidden(*a, **kw): raise AssertionError('Progress must not reconstruct a backend')
    with monkeypatch.context() as patch:
        patch.setattr(Session, 'restore', forbidden)
        before=time.monotonic(); current=manager.get(created['session_id'])
        assert time.monotonic()-before<.5
        assert current['state']['phase']=='waiting_for_model'
        assert current['state']['model_started_at'] and current['counters']['model_calls']==1
        assert 'backend' not in current and manager.store.recent(manager.endpoint_hash)[0]['session_id']==created['session_id']
    proceed.set();worker.join(5);assert not worker.is_alive()
    assert results[0]['final']['status']=='success'
    manager.store.close()


def test_local_model_generation_passes_chat_eos_and_accepts_eos_at_limit(monkeypatch):
    import types
    torch=pytest.importorskip('torch')
    from wearable_assistant.model_server import LocalModel,generation_stop_ids
    class Tokenizer:
        eos_token_id,pad_token_id=248046,248044
        eos_token,pad_token='<|im_end|>','<|endoftext|>'
        def apply_chat_template(self,*a,**kw): return torch.tensor([[10,11]])
        def decode(self,*a,**kw): return canonical(final())+'<|im_end|>'
    received=[]
    class FakeModel:
        def generate(self,**kw):
            received.append(kw)
            return torch.tensor([[10,11,55,248046]])
    local=LocalModel.__new__(LocalModel)
    local.tokenizer,local.model=Tokenizer(),FakeModel()
    local.model_id,local.max_context='fixture',100
    local.stop_ids=generation_stop_ids(local.tokenizer);local.lock=threading.Lock()
    monkeypatch.setattr(torch.Tensor,'to',lambda tensor,*a,**kw:tensor)
    result=local.complete({'model':'fixture','messages':[{'role':'user','content':'hi'}],'max_tokens':2})
    assert received[0]['eos_token_id']==[248046,248044]
    assert result['choices'][0]['finish_reason']=='stop'


def test_real_qwen_chat_stop_differs_from_model_default():
    from transformers import AutoTokenizer,AutoConfig,GenerationConfig
    from wearable_assistant.model_server import generation_stop_ids
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen3.5-2B',local_files_only=True)
    model_default=GenerationConfig.from_model_config(AutoConfig.from_pretrained('models/Qwen3.5-2B',local_files_only=True))
    assert tokenizer.eos_token_id==248046 and model_default.eos_token_id==248044
    assert 248046 in generation_stop_ids(tokenizer)


def test_truncation_keeps_diagnostic_without_executing_tools(config):
    class Truncated:
        def complete(self,*args):
            raise ProtocolError('Model response was truncated',{'message':{'content':'partial tool output'},'usage':{'completion_tokens':256,'total_tokens':2048},'finish_reason':'length'})
    cfg=dict(config['runtime'],max_protocol_repairs=0)
    world=json.loads(Path(config['world_file']).read_text(encoding='utf-8'))
    session=Session(LocalBackend(world,'查询杭州明天的天气。'),Truncated(),cfg)
    session.run()
    assert '截断' in session.final['message'] and session.counters['tool_calls']==0
    error=next(e for e in session.timeline if e['type']=='model_error')
    assert error['diagnostic']['message']['content']=='partial tool output'
    assert session.usages[0]['completion_tokens']==256


@pytest.mark.parametrize("connection_fails,expected", [
    (False, "模型响应格式错误：模型返回了未解析的工具调用标签"),
    (True, "模型服务请求失败：Model connection failed or timed out"),
])
def test_transport_and_malformed_tool_response_show_distinct_failures(config, connection_fails, expected):
    class Endpoint:
        def complete(self, *_):
            if connection_fails:
                raise TransportError("Model connection failed or timed out")
            content = "<tool_call>\n<function=get_preferences>\n</parameter>\n</function>\n</tool_call>"
            return Reply({"role": "assistant", "content": content}, {}, 0.0)

    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    runtime = dict(config["runtime"], max_protocol_repairs=0)
    session = Session(LocalBackend(world, "查看我的运动偏好。"), Endpoint(), runtime)
    session.run()
    assert session.final["status"] == "failed"
    assert session.final["message"] == expected
    assert session.counters["tool_calls"] == 0


def test_local_reminder_follows_original_event_across_long_conversation_and_restart(tmp_path):
    from wearable_assistant.context import local_tokenizer
    from wearable_assistant.model_server import encode_chat

    config = load_config("configs/assistant-p1000.json")
    config["runtime"]["local_fast_path"] = False
    create = tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)

    class Metered(Scripted):
        def complete(self, messages, tools, seed):
            reply = super().complete(messages, tools, seed)
            tokens = len(encode_chat(local_tokenizer(config["runtime"]["context_tokenizer"]), messages, tools))
            assert tokens <= 3840
            return Reply(reply.message, {"total_tokens": 3500, "prompt_tokens": tokens}, reply.seconds)

    endpoint = Metered([create] + [final() for _ in range(12)])
    database = tmp_path / "events.db"
    manager = SessionManager(config, endpoint, database)
    created = manager.create("明天18:00安排30分钟户外跑步。", "create")
    assert created["final"]["status"] == "success"
    sid = created["session_id"]
    event_id = created["final"]["result"]["event_id"]
    reminder = manager.message(sid, "给刚创建的活动设置提前10分钟的提醒。", "remind")
    receipt = reminder["final"]["result"]["reminder"]
    assert receipt["event_id"] == event_id and receipt["trigger_time"].endswith("17:50:00+08:00")
    assert reminder["state"]["known_objects"][event_id]["duration_minutes"] == 30

    for i in range(12):
        filler = f"第{i}轮普通对话。" + "话题仅用于检查历史窗口。" * 90
        last = manager.message(sid, filler, f"filler-{i}")
        assert last["final"]["status"] == "success"
    assert last["token_usage"]["reported_total_tokens"] > 32768
    assert any(e["type"] == "context_compacted" for e in last["timeline"])
    assert any("第0轮普通对话" in m["content"] for m in last["messages"] if m["role"] == "user")
    assert "第0轮普通对话" not in canonical(endpoint.payloads[-1]["messages"])
    assert event_id in endpoint.payloads[-1]["messages"][0]["content"]
    manager.store.close()

    endpoint = Scripted([])
    restarted = SessionManager(config, endpoint, database)
    changed = restarted.message(sid, "把刚创建的活动改到20:00，时长和活动不变。", "reschedule")
    verified = restarted.message(sid, "核对关联提醒。", "verify")
    for value in (changed, verified):
        assert value["final"]["status"] == "success"
        data = value["final"]["result"]
        assert data["event_id"] == event_id and data["start_time"].endswith("20:00:00+08:00")
        assert data["reminder_id"] == receipt["reminder_id"]
        assert data["reminder"]["trigger_time"].endswith("19:50:00+08:00")
        assert data["reminder"]["minutes_before"] == 10
    saved = restarted.store.load(sid, endpoint, restarted.endpoint_hash)
    assert saved.backend.read_event_bundle(event_id)["reminder"] == verified["final"]["result"]["reminder"]
    actions = [a["name"] for a in saved.backend.actions if a["type"] == "tool_call"]
    assert actions.count("create_event") == actions.count("set_reminder") == actions.count("update_event") == 1
    for hour, expected_count in ((18, 0), (20, 1)):
        listing = saved.backend.step(tool("list_events", start_time=f"2026-11-19T{hour}:00:00+08:00", end_time=f"2026-11-19T{hour}:30:00+08:00"))
        events = listing["observation"]["tool_result"]["data"]["events"]
        assert len(events) == expected_count
        if expected_count:
            assert events[0]["event_id"] == event_id and events[0]["duration_minutes"] == 30
    isolated = restarted.create("给刚创建的活动设置提前10分钟的提醒。", "different-session")
    assert isolated["state"]["phase"] == "waiting_for_user"
    assert "当前会话找不到" in isolated["messages"][-1]["content"]
    assert isolated["counters"]["tool_calls"] == 0 and not endpoint.payloads
    restarted.store.close()


@pytest.mark.parametrize("failure", ["missing", "stale", "unavailable"])
def test_confirmed_reschedule_discloses_unverified_reminder(config, monkeypatch, failure):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    endpoint = Scripted([tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)])
    backend = LocalBackend(world, "明天18:00安排30分钟户外跑步。")
    session = Session(backend, endpoint, config["runtime"])
    session.run()
    session.user_message("给刚创建的活动设置提前10分钟的提醒。")
    session.run()
    original = backend.read_event_bundle

    def failed_read(event_id):
        bundle = original(event_id)
        if bundle["event"]["start_time"].endswith("20:00:00+08:00"):
            if failure == "unavailable": raise OSError("fixture read failure")
            if failure == "missing":
                bundle["reminder"] = None
            else:
                bundle["reminder"]["trigger_time"] = "2026-11-19T17:50:00+08:00"
        return bundle

    monkeypatch.setattr(backend, "read_event_bundle", failed_read)
    session.user_message("把刚创建的活动改到明天20:00。")
    session.run()
    assert session.final["status"] == "partial"
    assert "活动已改期，提醒尚未核验" in session.final["message"]
    assert session.final["result"]["failure"]["event_verified"] is True
    assert session.final["result"]["failure"]["reminder_verified"] is False
    assert not session.state["known_reminders"]
    session.user_message("核对刚创建的活动的提醒。")
    monkeypatch.setattr(backend, "read_event_bundle", original)
    session.run()
    assert session.final["status"] == "success" and "19:50:00" in session.final["message"]


def test_ambiguous_activity_clarification_does_not_write_until_id_is_supplied(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    session = Session(LocalBackend(world, "给活动设置提前10分钟的提醒。"), Scripted([]), config["runtime"])
    for hour in (18, 19):
        receipt = session.backend.step(tool("create_event", activity_id="run", start_time=f"2026-11-19T{hour}:00:00+08:00", duration_minutes=30))
        session._remember_tool_result("create_event", receipt["observation"]["tool_result"]["data"])
    session.run()
    assert session.state["phase"] == "waiting_for_user" and session.counters["tool_calls"] == 0
    session = Session.restore(session.snapshot(), Scripted([]))
    session.user_message("event_id=event_00001")
    session.run()
    assert session.final["status"] == "success" and session.final["result"]["event_id"] == "event_00001"
    assert session.backend.read_event_bundle("event_00002")["reminder"] is None


@pytest.mark.parametrize("depth", [10, 20, 30])
@pytest.mark.parametrize("case_name", ["old_object", "active_constraint", "revoked_constraint", "ambiguous_target"])
def test_long_dialogue_memory_acceptance(depth, case_name):
    from wearable_assistant.context import count_input_tokens, local_tokenizer

    config = load_config("configs/assistant-p1000.json")
    config["runtime"]["local_fast_path"] = False
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    tokenizer = local_tokenizer(config["runtime"]["context_tokenizer"])
    prompt_lengths = []

    class Metered(Scripted):
        def complete(self, messages, tools, seed):
            prompt_lengths.append(count_input_tokens(tokenizer, messages, tools))
            assert prompt_lengths[-1] <= 3840
            return super().complete(messages, tools, seed)

    endpoint = Metered([final() for _ in range(depth)])
    initial = "这周只安排室内活动。" if case_name.endswith("constraint") else "已有活动。"
    session = Session(LocalBackend(world, initial), endpoint, config["runtime"])
    if case_name.endswith("constraint"):
        session.run()
        assert session.state["task_constraints"]["week"]["enabled"]
    else:
        for hour, activity in ((18, "run"), (19, "run"), (20, "walk"), (21, "reading")):
            receipt = session.backend.step(tool("create_event", activity_id=activity,
                                               start_time=f"2026-11-19T{hour}:00:00+08:00", duration_minutes=30))
            session._remember_tool_result("create_event", receipt["observation"]["tool_result"]["data"])
    for i in range(depth):
        session.user_message(f"第{i}轮普通话题。" + "请说明日常记录应如何整理。" * 35)
        session.run()
        assert session.final["status"] == "success"
    assert "第0轮普通话题" not in canonical(endpoint.payloads[-1]["messages"])
    session = Session.restore(session.snapshot(), endpoint)

    before = len(session.backend.actions)
    before_models = session.counters["model_calls"]
    expected = []
    if case_name == "old_object":
        session.user_message("把明天18:00的户外跑步改到明天22:00。"); session.run()
        assert session.final["result"]["event_id"] == "event_00001"
        assert session.backend.read_event_bundle("event_00001")["event"]["start_time"].endswith("22:00:00+08:00")
        assert session.backend.read_event_bundle("event_00002")["event"]["start_time"].endswith("19:00:00+08:00")
        expected = [("update_event", "event_00001")]
    elif case_name == "ambiguous_target":
        session.user_message("给户外跑步设置提前10分钟的提醒。"); session.run()
        assert session.state["phase"] == "waiting_for_user"
        question = session.messages[-1]["content"]
        assert "event_00001" in question and "event_00002" in question and "event_00003" not in question
        assert not any(a.get("name") == "set_reminder" for a in session.backend.actions[before:])
        session.user_message("event_id=event_00003"); session.run()
        assert session.state["phase"] == "waiting_for_user"
        session.user_message("event_id=event_00002"); session.run()
        assert session.final["result"]["event_id"] == "event_00002"
        expected = [("set_reminder", "event_00002")]
    else:
        if case_name == "revoked_constraint":
            session.user_message("取消本周室内活动限制。"); session.run()
            assert not session.state["task_constraints"]["week"]["enabled"]
        endpoint.actions.append(tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30))
        session.user_message("明天18:00安排30分钟户外跑步。"); session.run()
        if case_name == "active_constraint":
            assert session.final["status"] == "failed"
            assert session.final["result"]["error_code"] == "INDOOR_ONLY_CONSTRAINT"
            assert "未写入" in session.final["message"]
            assert any(e.get("error_code") == "INDOOR_ONLY_CONSTRAINT" for e in session.timeline)
            assert "这周只安排室内活动" in endpoint.payloads[-1]["messages"][0]["content"]
        else:
            assert session.final["status"] == "success"
            expected = [("create_event", None)]

    writes = [(a.get("name"), a["arguments"].get("event_id")) for a in session.backend.actions[before:]
              if a.get("name") in {"create_event", "update_event", "set_reminder", "delete_event"}]
    assert writes == expected
    print("MEMORY_CASE " + canonical({"distractor_turns": depth, "case": case_name, "met_expectation": True,
        "incorrect_writes": 0, "actual_writes": len(writes), "terminal_status": session.final["status"],
        "max_prompt_tokens": max(prompt_lengths), "target_model_calls": session.counters["model_calls"] - before_models}))


def test_event_field_protection_survives_detour_and_can_be_withdrawn(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    start = "2026-11-19T18:00:00+08:00"
    event_id = "event_00001"
    update = tool("update_event", event_id=event_id, duration_minutes=40)
    endpoint = Scripted([tool("create_event", activity_id="run", start_time=start, duration_minutes=30),
                         final(), update, update, final({"event_id": event_id})])
    session = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步。"), endpoint, config["runtime"])
    session.run()
    session.user_message("户外跑步后续只改开始时间。"); session.run()
    assert session.state["task_constraints"][f"event:{event_id}"]["enabled"]
    session.user_message("聊一个无关话题。"); session.run()
    session = Session.restore(session.snapshot(), endpoint)
    session.user_message("把户外跑步的时长改成40分钟。"); session.run()
    assert session.backend.read_event_bundle(event_id)["event"]["duration_minutes"] == 30
    assert any(e.get("error_code") == "PROTECTED_FIELDS_CHANGE" for e in session.timeline)
    session.user_message("取消户外跑步的只改开始时间限制。"); session.run()
    session.user_message("把户外跑步的时长改成40分钟。"); session.run()
    assert session.final["status"] == "success"
    assert session.backend.read_event_bundle(event_id)["event"]["duration_minutes"] == 40


def test_pending_reminder_parameters_survive_detour_and_can_be_cancelled(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    start = "2026-11-19T18:00:00+08:00"
    endpoint = Scripted([tool("create_event", activity_id="run", start_time=start, duration_minutes=30), final()])
    session = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步。"), endpoint, config["runtime"])
    session.run()
    session.user_message("给户外跑步设置提醒。"); session.run()
    assert session.state["waiting_slots"] == ["minutes_before"]
    assert session.backend.read_event_bundle("event_00001")["reminder"] is None
    session.user_message("先聊一个无关话题。"); session.run()
    session = Session.restore(session.snapshot(), endpoint)
    session.user_message("提前10分钟。"); session.run()
    assert session.final["result"]["reminder"]["minutes_before"] == 10
    session.user_message("给户外跑步设置提醒。"); session.run()
    session.user_message("取消刚才的请求。"); session.run()
    assert session.state["pending_event_followup"] is None
    assert len([a for a in session.backend.actions if a.get("name") == "set_reminder"]) == 1


def test_pending_creation_restores_only_its_own_slots(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    ask = {"type": "ask_user", "question": "跑步需要持续多久？", "requested_slots": ["duration_minutes"]}
    start = "2026-11-19T18:00:00+08:00"
    endpoint = Scripted([ask, final(), tool("create_event", activity_id="run", start_time=start, duration_minutes=30)])
    session = Session(LocalBackend(world, "明天18:00安排户外跑步，时长请先问我。"), endpoint, config["runtime"])
    session.run()
    assert session.state["pending_task"]["slots"]["start_time"] == start
    session.user_message("请解释一下为什么后天08:00适合阅读。"); session.run()
    session = Session.restore(session.snapshot(), endpoint)
    session.user_message("持续30分钟。"); session.run()
    assert session.final["status"] == "success"
    event_id = session.final["result"]["event_id"]
    assert session.backend.read_event_bundle(event_id)["event"]["start_time"] == start
    assert session.backend.read_event_bundle(event_id)["event"]["duration_minutes"] == 30


def test_weekly_constraint_does_not_leak_into_next_week(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    start = "2026-11-23T18:00:00+08:00"  # Monday after the simulated current week.
    endpoint = Scripted([tool("create_event", activity_id="run", start_time=start, duration_minutes=30)])
    session = Session(LocalBackend(world, "这周只安排室内活动。"), endpoint, config["runtime"])
    session.run()
    assert session.state["task_constraints"]["week"]["through_date"] == "2026-11-22"
    session.user_message(f"{start}安排30分钟户外跑步。"); session.run()
    assert session.final["status"] == "success"


def test_projected_tool_result_retains_focused_event_and_full_sqlite_record(tmp_path):
    from datetime import datetime, timedelta
    from wearable_assistant.context import count_input_tokens, local_tokenizer

    config = load_config("configs/assistant-p1000.json")
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    start = datetime.fromisoformat("2026-11-19T00:00:00+08:00")
    for index in range(140):
        at = start + timedelta(minutes=index * 10)
        world["state"]["events"].append({"event_id": f"event_{index+1:05d}", "activity_id": "run", "title": "户外跑步",
            "start_time": at.isoformat(), "end_time": (at + timedelta(minutes=5)).isoformat(), "duration_minutes": 5})
    world_path = tmp_path / "world.json"
    write_json(world_path, world)
    config["world_file"] = str(world_path)
    target = "event_00120"
    lookup = tool("list_events", start_time=start.isoformat(), end_time=(start + timedelta(days=1)).isoformat())
    target_row = world["state"]["events"][119]
    facts = {"event_id": target, "start_time": target_row["start_time"]}
    wrapped_final = tool("final", status="success", message="已找到目标活动的开始时间。", result=canonical(facts))
    endpoint = Scripted([lookup, wrapped_final])
    database = tmp_path / "memory.db"
    manager = SessionManager(config, endpoint, database)
    response = manager.create(f"请找到 {target} 并告知开始时间。", "lookup")
    assert response["final"]["status"] == "success"
    assert response["final"]["result"] == facts
    assert response["counters"]["tool_calls"] == 1
    sent = endpoint.payloads[-1]["messages"]
    projected = next(m for m in sent if m["role"] == "tool")
    data = json.loads(projected["content"])
    assert data["context_projection"]["total_rows"] == 140 and data["context_projection"]["omitted_rows"] > 0
    assert data["data"]["events"][0]["event_id"] == target
    assert sent[-2]["tool_calls"][0]["id"] == projected["tool_call_id"]
    tokenizer = local_tokenizer(config["runtime"]["context_tokenizer"])
    fitted = count_input_tokens(tokenizer, sent, endpoint.payloads[-1]["tools"])
    assert fitted <= 3840
    saved = manager.store.load(response["session_id"], endpoint, manager.endpoint_hash)
    original = next(m for m in saved.messages if m["role"] == "tool")
    assert len(json.loads(original["content"])["data"]["events"]) == 140
    assert len(saved.tool_events[0]["result"]["data"]["events"]) == 140
    assert "context_projection" not in original["content"]
    assert saved._final_error(final({**facts, "start_time": start.isoformat()})) == "UNSUPPORTED_QUERY_FACTS"
    full_messages = [original if m is projected else m for m in sent]
    full = count_input_tokens(tokenizer, full_messages, endpoint.payloads[-1]["tools"])
    assert full > 3840
    print("MEMORY_PROJECTION " + canonical({"raw_rows": 140, "shown_rows": len(data["data"]["events"]),
                                            "full_input_tokens": full, "fitted_input_tokens": fitted}))
    manager.store.close()


def test_create_success_requires_current_write_even_with_old_similar_event(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    claim = final({"event_id": "event_00001"})
    claim["message"] = "安排好了。"
    endpoint = Scripted([tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30), claim, claim])
    session = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步。"), endpoint, config["runtime"])
    session.run()
    assert session.final["status"] == "success"
    for request in ("明天20:00安排30分钟户外跑步。", "帮我明天安排跑步。"):
        session.user_message(request)
        session.run()
        assert session.final["status"] == "failed"
        assert session.timeline[-2]["error_code"] == "MISSING_VERIFIED_WRITE"
        assert session.backend.read_event_bundle("event_00001")["event_count"] == 1


def test_compound_create_and_reminder_needs_both_verified_writes(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    create = tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30)
    config["runtime"]["local_fast_path"] = True
    session = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步并提前10分钟提醒我。"),
                      Scripted([create, final({"event_id": "event_00001"})]), config["runtime"])
    session.run()
    assert session.final["status"] == "partial"
    assert session.counters["model_calls"] == 2
    assert any(e.get("error_code") == "MISSING_VERIFIED_WRITE" for e in session.timeline)
    assert session.backend.read_event_bundle("event_00001")["reminder"] is None


def test_update_cannot_substitute_duration_for_requested_time(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    endpoint = Scripted([tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30),
                         tool("update_event", event_id="event_00001", duration_minutes=45), final({"event_id": "event_00001"})])
    session = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步。"), endpoint, config["runtime"])
    session.run()
    session.user_message("请将 event_00001 的活动改到明天20:00。")
    session.run()
    assert session.final["status"] == "failed"
    assert any(e.get("error_code") == "UPDATE_TIME_CONTRADICTS_USER" for e in session.timeline)
    assert session.backend.read_event_bundle("event_00001")["event"]["duration_minutes"] == 30


def test_fast_path_checks_receipt_and_catalog_without_model(config):
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    cfg = {**config["runtime"], "local_fast_path": True}
    endpoint = Scripted([])
    session = Session(LocalBackend(world, "明天18:00安排30分钟户外跑步。"), endpoint, cfg)
    session.run()
    assert session.final["status"] == "success" and session.final_origin == "controller_fast_path"
    assert [event["name"] for event in session.tool_events] == ["search_activity", "create_event", "list_events"]
    assert session.tool_events[1]["verified"] is True
    session.user_message("查询明天的日程。")
    session.run()
    assert len(session.final["result"]["events"]) == 1 and not endpoint.payloads


def test_interactive_comparison_scores_current_state_and_discloses_false_success(tmp_path, config):
    from wearable_assistant.interactive_evaluation import evaluate_interactive

    cases = "examples/assistant-interactive-cases.json"
    expected = {"event_id": "event_00001"}
    false_claim = final(expected)
    likelier = [tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30), final(expected)]
    endpoints = iter([Scripted([false_claim]), Scripted(likelier), Scripted([false_claim]), Scripted([])])
    directory = tmp_path / "interactive"
    report = evaluate_interactive(config, lambda: next(endpoints), cases, directory,
                                  ["baseline", "memory", "full", "fast"], ["simple-create"])
    rows = {name: item["summary"] for name, item in report["variants"].items()}
    assert rows["baseline"]["false_success"]["count"] == 1
    assert rows["full"]["false_success"]["count"] == 0 and rows["full"]["raw_false_success"]["count"] == 1
    assert rows["memory"]["completed"]["count"] == rows["fast"]["completed"]["count"] == 1
    assert rows["fast"]["mean_model_calls_per_case"] == 0
    assert rows["fast"]["mean_input_tokens_per_measured_case"] == 0
    assert (directory / "full.sqlite3").exists() and (directory / "report.json").exists()


def test_daily_query_uses_full_day_and_interactive_scoring_checks_scope(tmp_path, config):
    from wearable_assistant.interactive_evaluation import evaluate_interactive

    partial = tool("list_events", start_time="2026-11-19T09:00:00+08:00", end_time="2026-11-19T17:00:00+08:00")
    claim = final({"events": []})
    endpoints = iter([Scripted([partial, claim]), Scripted([partial]), Scripted([])])
    report = evaluate_interactive(config, lambda: next(endpoints), "examples/assistant-interactive-cases.json",
                                  tmp_path / "daily", ["baseline", "full", "fast"], ["simple-list"])
    profiles = report["variants"]
    assert profiles["baseline"]["summary"]["false_success"]["count"] == 1
    assert profiles["full"]["summary"]["completed"]["count"] == profiles["fast"]["summary"]["completed"]["count"] == 1
    assert profiles["full"]["cases"][0]["model_calls"] == 1 and profiles["fast"]["cases"][0]["model_calls"] == 0

    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    session = Session(LocalBackend(world, "查询明天的日程。"), Scripted([]), config["runtime"])
    session._tool(partial, lambda _: None)
    assert session._final_error(claim) == "QUERY_EVIDENCE_REQUIRED"


def test_interactive_score_distinguishes_final_overblock_from_missing_execution(config):
    from wearable_assistant.interactive_evaluation import _score_turn, load_cases

    class MissingReadBack(LocalBackend):
        def step(self, action):
            response = super().step(action)
            if action.get("name") == "list_events":
                response["observation"]["tool_result"]["data"]["events"] = []
            return response

    case = load_cases("examples/assistant-interactive-cases.json", ["simple-create"])[0]["turns"][0]
    world = json.loads(Path(config["world_file"]).read_text(encoding="utf-8"))
    bounded = {**config["runtime"], "grounded_assistance": False}
    actions = [tool("create_event", activity_id="run", start_time="2026-11-19T18:00:00+08:00", duration_minutes=30),
               final({"event_id": "event_00001"})]
    session = Session(MissingReadBack(world, case["text"]), Scripted(actions), bounded)
    session.run()
    score = _score_turn(case["expect"], session, 0, 0, 0)
    assert score["valid_execution"] and score["overblock"]
    assert not score["passed"] and not score["false_success"]
