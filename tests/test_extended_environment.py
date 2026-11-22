import copy

import pytest

from lightagent.cases import scenario
from lightagent.environment import Environment
from lightagent.models import Final
from lightagent.user_simulator import DeterministicUser
from test_environment import event_args


def scheduled():
    env = Environment(scenario(preferences={"preferred_city": "杭州"}))
    e = env.execute("calendar_create", event_args())["data"]
    r = env.execute("reminder_create", {"event_id": e["id"], "minutes_before": 10, "idempotency_key": "r1"})["data"]
    return env, e, r


def test_update_moves_linked_reminder_and_deletion_cascades():
    env, e, r = scheduled()
    args = {"event_id": e["id"], "start": "2026-11-17T19:00:00+08:00", "expected_revision": 0, "idempotency_key": "u1"}
    updated = env.execute("calendar_update", args)
    assert updated["ok"] and updated["data"]["revision"] == 1
    assert env.reminders[0]["remind_at"] == "2026-11-17T18:50:00+08:00"
    assert env.execute("calendar_update", args) == updated
    deletion = {"event_id": e["id"], "expected_revision": 1, "idempotency_key": "d1"}
    first = env.execute("calendar_delete", deletion)
    assert env.execute("calendar_delete", deletion) == first
    assert first["data"]["deleted_reminder_ids"] == [r["id"]]
    assert not env.events and not env.reminders
    newer = env.execute("calendar_create", event_args(idempotency_key="new-task"))
    assert newer["data"]["id"] != e["id"]


@pytest.mark.parametrize("fault", ["event_update_failures", "event_delete_failures"])
def test_failed_mutations_are_atomic(fault):
    env, e, _ = scheduled()
    env.faults[fault] = 1
    before = env.snapshot()
    name = "calendar_update" if "update" in fault else "calendar_delete"
    args = {"event_id": e["id"], "idempotency_key": "op1"}
    if name == "calendar_update":
        args["start"] = "2026-11-17T19:00:00+08:00"
    assert not env.execute(name, args)["ok"]
    assert env.snapshot() == before


def test_conflict_stale_revision_and_past_reminder_do_not_partially_move():
    env, e, _ = scheduled()
    env.execute("calendar_create", event_args(start="2026-11-17T19:00:00+08:00", idempotency_key="other"))
    before = env.snapshot()
    args = {"event_id": e["id"], "start": "2026-11-17T19:00:00+08:00", "idempotency_key": "u1"}
    assert env.execute("calendar_update", args)["error"]["code"] == "conflict"
    args.update(start="2026-11-17T20:00:00+08:00", expected_revision=5)
    assert env.execute("calendar_update", args)["error"]["code"] == "stale_revision"
    args.update(start="2026-11-16T10:05:00+08:00", expected_revision=0)
    assert env.execute("calendar_update", args)["error"]["code"] == "past_reminder"
    assert env.snapshot() == before


def test_activity_search_and_preferences_return_facts_without_planning():
    env, _, _ = scheduled()
    before = env.snapshot()
    assert env.execute("preferences_get", {})["data"]["preferences"] == {"preferred_city": "杭州"}
    results = env.execute("activity_search", {"indoor": True, "max_duration_minutes": 30})["data"]["activities"]
    assert len(results) == 1 and results[0]["id"] == "indoor_stretch"
    assert env.snapshot() == before


def test_deterministic_user_never_auto_supplies_unasked_answers():
    user = DeterministicUser({"city", "expression"}, {"city": "杭州", "expression": "明天下午六点"})
    assert user.respond(Final(status="success", message="完成", requested_slots=["city"])) is None
    assert user.respond(Final(status="clarification", message="喜欢什么颜色？", requested_slots=["color"])) is None
    assert user.respond(Final(status="clarification", message="哪个城市？", requested_slots=["city"])) == "杭州"
    assert user.respond(Final(status="clarification", message="哪个城市？", requested_slots=["city"])) is None
    assert user.respond(Final(status="clarification", message="何时？", requested_slots=["expression"])) == "明天下午六点"

