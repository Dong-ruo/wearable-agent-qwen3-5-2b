from datetime import timedelta

import pytest

from lightagent.cases import scenario
from lightagent.environment import Environment, tool_specs


@pytest.fixture
def env():
    return Environment(scenario())


def event_args(**updates):
    args = {"title": "户外跑步", "activity": "outdoor_run", "city": "杭州",
            "start": "2026-11-17T18:00:00+08:00", "duration_minutes": 30, "idempotency_key": "task-1"}
    args.update(updates)
    return args


@pytest.mark.parametrize("name,args", [
    ("not_a_tool", {}),
    ("clock_now", {"unexpected": 1}),
    ("calendar_list", {"start": "2026-11-17T18:00:00", "end": "2026-11-17T19:00:00"}),
    ("calendar_list", {"start": "2026-11-17T19:00:00+08:00", "end": "2026-11-17T18:00:00+08:00"}),
    ("calendar_create", event_args(duration_minutes=True)),
    ("calendar_create", event_args(duration_minutes=0)),
    ("calendar_create", event_args(activity="swim")),
    ("calendar_create", event_args(start="2026-11-15T18:00:00+08:00")),
    ("reminder_create", {"event_id": "missing", "minutes_before": 10, "idempotency_key": "r1"}),
])
def test_bad_calls_never_mutate_state(env, name, args):
    before = env.snapshot()
    result = env.execute(name, args)
    assert not result["ok"]
    assert env.snapshot() == before
    assert env.trace[-1]["result"] == result


def test_idempotent_retries_and_changed_payload(env):
    first = env.execute("calendar_create", event_args())
    second = env.execute("calendar_create", event_args())
    assert first == second
    assert len(env.events) == 1
    mismatch = env.execute("calendar_create", event_args(title="修改标题"))
    assert mismatch["error"]["code"] == "idempotency_mismatch"
    reminder = {"event_id": first["data"]["id"], "minutes_before": 10, "idempotency_key": "r1"}
    assert env.execute("reminder_create", reminder) == env.execute("reminder_create", reminder)
    assert len(env.reminders) == 1
    assert env.execute("reminder_create", {**reminder, "idempotency_key": "r2"})["error"]["code"] == "duplicate_reminder"


def test_half_open_intervals_allow_adjacent_events(env):
    assert env.execute("calendar_create", event_args())["ok"]
    assert env.execute("calendar_create", event_args(start="2026-11-17T18:30:00+08:00", idempotency_key="task-2"))["ok"]
    conflict = env.execute("calendar_create", event_args(start="2026-11-17T18:15:00+08:00", idempotency_key="task-3"))
    assert conflict["error"]["code"] == "conflict"
    assert len(env.events) == 2


def test_fault_is_consumed_without_poisoning_idempotency():
    env = Environment(scenario(faults={"event_create_failures": 1}))
    assert not env.execute("calendar_create", event_args())["ok"]
    assert env.events == []
    assert env.execute("calendar_create", event_args())["ok"]
    assert len(env.events) == 1


def test_rollouts_are_isolated():
    original = scenario(faults={"event_create_failures": 1})
    a, b = Environment(original), Environment(original)
    a.execute("calendar_create", event_args())
    a.execute("calendar_create", event_args())
    assert b.events == [] and original.events == []
    assert b.faults["event_create_failures"] == 1
    assert original.faults["event_create_failures"] == 1


def test_weather_and_tool_schema(env):
    assert not env.execute("weather_get", {"city": "杭州", "date": "2026-11-18"})["ok"]
    schemas = tool_specs()
    assert len(schemas) == 14
    assert all(s["function"]["parameters"]["additionalProperties"] is False for s in schemas)
    assert not any("faults" in s["function"]["parameters"].get("properties", {}) for s in schemas)


def test_past_reminder_does_not_write(env):
    created = env.execute("calendar_create", event_args(start=(env.now + timedelta(minutes=5)).isoformat()))
    result = env.execute("reminder_create", {"event_id": created["data"]["id"], "minutes_before": 10, "idempotency_key": "r1"})
    assert result["error"]["code"] == "past_reminder"
    assert env.reminders == []

