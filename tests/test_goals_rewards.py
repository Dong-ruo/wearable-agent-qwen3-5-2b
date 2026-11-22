import copy
import json
from datetime import datetime

import pytest

from lightagent.cases import REQUEST, scenario
from lightagent.environment import Environment
from lightagent.episode import run_episode
from lightagent.goals import EpisodeSpec, GoalSpec, evaluate_goal
from lightagent.integrations.trl_env import TaskToolEnv
from lightagent.models import Call, Final
from lightagent.policy import RulePolicy
from lightagent.replay import replay
from lightagent.rewards import RewardConfig, cost_metrics, group_statistics, score_episode, trl_reward
from lightagent.runtime import Runtime, Session
from test_environment import event_args
from test_runtime import ScriptedPolicy


def exercise_spec():
    return EpisodeSpec(request=REQUEST, scenario=scenario(), goal=GoalSpec.model_validate({
        "events": [{"activity": "outdoor_run", "city": "杭州", "start": "2026-11-17T18:00:00+08:00", "duration_minutes": 30}],
        "reminders": [{"event_target_index": 0, "minutes_before": 10}],
        "weather": [{"city": "杭州", "date": "2026-11-17", "condition": "sunny"}],
        "new_event_count": 1, "new_reminder_count": 1}))


def test_complete_goal_reward_and_deterministic_replay():
    spec = exercise_spec()
    runtime, reward = run_episode(spec, RulePolicy())
    assert reward["success"] and reward["reward"] == pytest.approx(1.1)
    assert replay(spec, runtime.artifact()) == reward
    tampered = copy.deepcopy(runtime.artifact())
    tampered["trace"][0]["result"]["data"]["now"] = "fake"
    with pytest.raises(ValueError, match="不一致"):
        replay(spec, tampered)


def test_reward_does_not_accept_runtime_repaired_false_success():
    spec = exercise_spec()
    runtime, report = run_episode(spec, ScriptedPolicy(Final(status="success", message="都完成了", event_id="fake")))
    assert runtime.last_final.status == "failed"
    assert not report["success"] and report["reward"] <= 0.1
    assert not report["checks"]["structured_feedback_consistent"]


def test_extra_writes_and_wrong_goal_cannot_outscore_success():
    spec = exercise_spec()
    runtime, good = run_episode(spec, RulePolicy())
    runtime.env.execute("calendar_create", event_args(start="2026-11-17T20:00:00+08:00", idempotency_key="extra"))
    bad = score_episode(spec.goal, runtime.env, runtime.last_final)
    assert not bad["success"]
    assert bad["reward"] < good["reward"] - .5
    assert not bad["checks"]["new_event_count"]


def test_result_only_ablation_and_constant_group_signal():
    spec = exercise_spec()
    runtime, _ = run_episode(spec, RulePolicy())
    assert score_episode(spec.goal, runtime.env, runtime.last_final, RewardConfig(mode="result_only"))["reward"] == 1
    stats = group_statistics([0, 0, 0, 0])
    assert stats["constant_group"] and stats["advantages"] == [0]*4
    assert not group_statistics([0, 1, 0, 1])["constant_group"]
    with pytest.raises(ValueError):
        RewardConfig(process_weight=.9)


def test_cost_allows_reasonable_retry_after_injected_write_failure():
    env = Environment(scenario(faults={"event_create_failures": 1}))
    env.execute("calendar_create", event_args())
    env.execute("calendar_create", event_args())
    assert cost_metrics(env.trace)["cost"] == 0
    env.execute("calendar_get", {"event_id": env.events[0]["id"]})
    env.execute("calendar_get", {"event_id": env.events[0]["id"]})
    assert cost_metrics(env.trace)["redundant_calls"] == 1


def test_l1_weather_and_l6_update_then_delete_use_private_goals():
    env = Environment(scenario())
    weather_goal = GoalSpec(weather=[{"city": "杭州", "date": "2026-11-17"}])
    runtime = Runtime(env, ScriptedPolicy(Call(name="weather_get", arguments={"city": "杭州", "date": "2026-11-17"}), Final(status="success", message="晴")), goal=weather_goal)
    assert runtime.run("查杭州明天天气").status == "success"
    e = env.execute("calendar_create", event_args())["data"]
    r = env.execute("reminder_create", {"event_id": e["id"], "minutes_before": 10, "idempotency_key": "r1"})["data"]
    goal = GoalSpec(events=[{"event_id": e["id"], "start": "2026-11-17T19:00:00+08:00"}], reminders=[{"event_id": e["id"], "minutes_before": 10}])
    update = [Call(name="calendar_update", arguments={"event_id": e["id"], "start": "2026-11-17T19:00:00+08:00", "idempotency_key": "u1"}),
        Call(name="calendar_get", arguments={"event_id": e["id"]}), Call(name="reminder_get", arguments={"reminder_id": r["id"]}), Final(status="success", message="已修改", event_id=e["id"], reminder_id=r["id"])]
    session = Session(env, lambda: ScriptedPolicy(*update))
    moved = session.submit("把刚才活动改到晚上七点", goal)
    assert moved.last_final.status == "success"
    assert env.reminders[0]["remind_at"] == "2026-11-17T18:50:00+08:00"
    session.policy_factory = lambda: ScriptedPolicy(Call(name="calendar_delete", arguments={"event_id": e["id"], "idempotency_key": "d1"}),
        Call(name="calendar_get", arguments={"event_id": e["id"]}), Final(status="success", message="已撤销", event_id=e["id"]))
    deleted = session.submit("撤销这个活动", GoalSpec(deleted_event_ids=[e["id"]]))
    assert deleted.last_final.status == "success" and not env.events and not env.reminders


def test_trl_reset_keeps_private_goal_and_user_answers_out_of_prompt():
    spec = exercise_spec()
    spec.missing_slots = {"city"}
    spec.user_answers = {"city": "PRIVATE_CITY"}
    spec.request = "查询天气，缺少城市"
    a, b = TaskToolEnv(), TaskToolEnv()
    assert a.reset(episode_json=spec.model_dump_json()) is None
    assert b.reset(episode_json=spec.model_dump_json()) is None
    assert a._goal == b._goal == spec.goal
    assert "PRIVATE_CITY" not in a.clock_now()
    assert "PRIVATE_CITY" not in a.user_ask("何时？", ["expression"])
    assert "PRIVATE_CITY" in a.user_ask("城市？", ["city"])
    a.calendar_create(**event_args())
    assert not b._env.events
    assert "get_reward" not in dir(a)


def test_trl_reward_reads_actual_environments_once():
    spec = exercise_spec()
    runtime, _ = run_episode(spec, RulePolicy())
    adapter = TaskToolEnv(spec.scenario, spec.request, spec.goal)
    adapter._env = runtime.env
    result = trl_reward([[{"role": "assistant", "content": runtime.last_final.model_dump_json()}]], [adapter])
    assert result == pytest.approx([1.1])
    assert adapter._last_reward["success"]


def test_unrelated_schedule_cannot_be_deleted_for_an_easier_reward():
    spec = exercise_spec()
    meeting = {"id": "protected", "title": "原有安排", "start": "2026-11-17T15:00:00+08:00", "end": "2026-11-17T16:00:00+08:00", "origin": "external"}
    spec.scenario.events.append(meeting)
    runtime, report = run_episode(spec, RulePolicy())
    assert report["success"]
    runtime.env.execute("calendar_delete", {"event_id": "protected", "idempotency_key": "bad"})
    hacked = score_episode(spec.goal, runtime.env, runtime.last_final)
    assert not hacked["success"] and not hacked["checks"]["protected_events"]


def test_modifying_event_requires_a_fresh_reminder_read():
    spec = exercise_spec()
    runtime, _ = run_episode(spec, RulePolicy())
    event = runtime.env.events[0]
    runtime.env.execute("calendar_update", {"event_id": event["id"], "start": "2026-11-17T19:00:00+08:00", "idempotency_key": "u1"})
    runtime.env.execute("calendar_get", {"event_id": event["id"]})
    goal = spec.goal.model_copy(deep=True)
    goal.events[0].start = datetime.fromisoformat("2026-11-17T19:00:00+08:00")
    checks = evaluate_goal(goal, runtime.env)
    assert not checks["reminder_0_verified"]
