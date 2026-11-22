import copy
import unittest
from datetime import timedelta

from wearable_tasks.contract import schema_hash, stamp, tool_schemas
from wearable_tasks.environment import TaskEnvironment
from wearable_tasks.tests.helpers import case, empty_case


class CoreTests(unittest.TestCase):
    def create(self, env, start=None, duration=30, activity="run"):
        return env.execute("create_event", {"activity_id": activity,
            "start_time": start or env._task["reference_intent"]["start_time"], "duration_minutes": duration})

    def test_half_open_conflict_and_cross_midnight_query(self):
        env = TaskEnvironment(empty_case())
        start = stamp(env._task["reference_intent"]["start_time"])
        first = self.create(env)["data"]
        self.assertEqual(self.create(env, (start+timedelta(minutes=30)).isoformat())["status"], "success")
        before = copy.deepcopy(env._state)
        self.assertEqual(self.create(env, (start+timedelta(minutes=15)).isoformat())["error_code"], "TIME_CONFLICT")
        self.assertEqual(env._state, before)
        midnight = start.replace(hour=23, minute=50)
        event = self.create(env, midnight.isoformat(), 20)["data"]
        result = env.execute("list_events", {"start_time": (midnight+timedelta(minutes=10)).isoformat(), "end_time": (midnight+timedelta(minutes=30)).isoformat()})
        self.assertEqual(result["data"]["events"], [event])
        adjacent = env.execute("list_events", {"start_time": first["end_time"], "end_time": (start+timedelta(minutes=31)).isoformat()})
        self.assertNotIn(first, adjacent["data"]["events"])

    def test_update_conflict_and_invalid_reminder_are_atomic(self):
        env = TaskEnvironment(empty_case())
        start = stamp(env._task["reference_intent"]["start_time"])
        first = self.create(env)["data"]
        second = self.create(env, (start+timedelta(hours=1)).isoformat())["data"]
        env.execute("set_reminder", {"event_id": first["event_id"], "minutes_before": 10})
        before = copy.deepcopy(env._state)
        result = env.execute("update_event", {"event_id": first["event_id"], "start_time": second["start_time"]})
        self.assertEqual(result["error_code"], "TIME_CONFLICT")
        self.assertEqual(env._state, before)
        result = env.execute("update_event", {"event_id": first["event_id"], "start_time": (env._now+timedelta(minutes=5)).isoformat()})
        self.assertEqual(result["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(env._state, before)

    def test_reminder_idempotency_update_and_delete_cascade(self):
        env = TaskEnvironment(empty_case())
        event = self.create(env)["data"]
        args = {"event_id": event["event_id"], "minutes_before": 10}
        first = env.execute("set_reminder", args)
        self.assertEqual(env.execute("set_reminder", args), first)
        self.assertEqual(len(env._state["reminders"]), 1)
        new_start = stamp(event["start_time"])+timedelta(hours=1)
        self.assertEqual(env.execute("update_event", {"event_id": event["event_id"], "start_time": new_start.isoformat()})["status"], "success")
        self.assertEqual(env._state["reminders"][0]["trigger_time"], (new_start-timedelta(minutes=10)).isoformat())
        old_counter = env._state["next_event_id"]
        self.assertEqual(env.execute("delete_event", {"event_id": event["event_id"]})["status"], "success")
        self.assertEqual(env._state["reminders"], [])
        newer = self.create(env)["data"]
        self.assertNotEqual(newer["event_id"], event["event_id"])
        self.assertGreater(env._state["next_event_id"], old_counter)

    def test_schema_type_rules_and_failed_writes_never_allocate_ids(self):
        env = TaskEnvironment(empty_case())
        baseline = copy.deepcopy(env._state)
        bad = [
            ("create_event", {"activity_id": "missing", "start_time": env._task["reference_intent"]["start_time"], "duration_minutes": 30}),
            ("create_event", {"activity_id": "run", "start_time": env._task["reference_intent"]["start_time"], "duration_minutes": True}),
            ("create_event", {"activity_id": "run", "start_time": "2026-01-01T18:00:00", "duration_minutes": 30}),
            ("update_event", {"event_id": "missing", "duration_minutes": 20}),
            ("delete_event", {"event_id": "missing"}),
            ("set_reminder", {"event_id": "missing", "minutes_before": 10}),
            ("get_preferences", {"unknown": 1}),
            ("set_reminder", {"event_id": "missing", "minutes_before": 0}),
        ]
        for name, args in bad:
            with self.subTest(name=name, args=args):
                self.assertEqual(env.execute(name, args)["status"], "error")
                self.assertEqual(env._state, baseline)

    def test_four_environments_and_fault_counter_isolation(self):
        task = case("retry_weather")
        original = copy.deepcopy(task)
        envs = [TaskEnvironment(task) for _ in range(4)]
        snapshots = [e.private_snapshot() for e in envs]
        rule = task["fault_plan"][0]
        wrong = dict(rule["match"], city="没有配置的城市")
        self.assertEqual(envs[0].execute("get_weather", wrong)["error_code"], "NOT_FOUND")
        self.assertEqual(envs[0]._faults[0]["remaining"], 1)
        self.assertEqual(envs[0].execute("get_weather", rule["match"])["error_code"], "TEMPORARY_UNAVAILABLE")
        self.assertEqual(envs[0].execute("get_weather", rule["match"])["status"], "success")
        for i in (1, 2, 3):
            self.assertEqual(envs[i].private_snapshot(), snapshots[i])
        self.assertEqual(task, original)
        envs[0].reset()
        self.assertEqual(envs[0].private_snapshot(), snapshots[0])

    def test_write_faults_forbidden_and_budget_enforced(self):
        task = case()
        task["fault_plan"] = [{"tool": "create_event", "remaining": 1, "match": {}}]
        with self.assertRaises(ValueError):
            TaskEnvironment(task)
        task = case()
        task["max_tool_calls"] = 1
        env = TaskEnvironment(task)
        env.execute("get_preferences", {})
        result = env.step({"type": "tool_call", "name": "get_preferences", "arguments": {}})
        self.assertTrue(result["done"])
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["observation"]["tool_result"]["error_code"], "BUDGET_EXCEEDED")

    def test_schema_is_frozen_and_sorting_is_stable(self):
        original_hash = schema_hash()
        schemas = tool_schemas()
        self.assertEqual(len(schemas), 8)
        schemas[0]["function"]["parameters"]["properties"]["secret"] = {"type": "string"}
        self.assertEqual(schema_hash(), original_hash)
        task = case("events_query", ordinal=5)
        task["initial_state"]["events"].reverse()
        env = TaskEnvironment(task)
        events = env.execute("list_events", task["reference_intent"]["window"])["data"]["events"]
        self.assertEqual(events, sorted(events, key=lambda e: (e["start_time"], e["event_id"])))

    def test_reminder_may_equal_now_but_must_precede_event(self):
        env = TaskEnvironment(empty_case())
        start = env._now+timedelta(minutes=10)
        event = self.create(env, start.isoformat())["data"]
        reminder = env.execute("set_reminder", {"event_id": event["event_id"], "minutes_before": 10})
        self.assertEqual(reminder["data"]["trigger_time"], env._now.isoformat())
        before = copy.deepcopy(env._state)
        self.assertEqual(env.execute("set_reminder", {"event_id": event["event_id"], "minutes_before": 11})["error_code"], "INVALID_ARGUMENT")
        self.assertEqual(env._state, before)

    def test_malformed_actions_fail_without_crashing_terminal_validation(self):
        for action in ([], {"type": "final", "status": [], "result": {}, "message": "bad"},
                       {"type": "ask_user", "question": "bad", "requested_slots": [[1]]}):
            with self.subTest(action=action):
                env = TaskEnvironment(case("clarify_time"))
                result = env.step(action)
                self.assertTrue(result["done"])
                self.assertEqual(result["reward"], 0)
                self.assertFalse(env.evaluate()["success"])
