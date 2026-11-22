import copy
import unittest

from wearable_tasks.auditing import audit_negatives, audit_privacy
from wearable_tasks.environment import TaskEnvironment
from wearable_tasks.reference import ReferencePolicy, replay, run_reference
from wearable_tasks.templates import BASE_KINDS, LEVELS
from wearable_tasks.tests.helpers import case


class PublicStepOnly:
    def __init__(self, env):
        self.step = env.step


class L123Tests(unittest.TestCase):
    def test_all_lower_families_and_parameter_variants(self):
        for kind in BASE_KINDS:
            if LEVELS[kind] > 3:
                continue
            for ordinal in (0, 3, 5):
                for variant in range(4):
                    with self.subTest(kind=kind, ordinal=ordinal, variant=variant):
                        task = case(kind, ordinal, variant)
                        env = TaskEnvironment(task)
                        ReferencePolicy(task["reference_intent"]).solve(PublicStepOnly(env))
                        self.assertTrue(env.evaluate()["success"], env.evaluate())
                        self.assertEqual(replay(task, env.trajectory()).private_snapshot(), env.private_snapshot())
                        self.assertTrue(audit_privacy(task, env))

    def test_actual_weather_is_used_instead_of_private_goal(self):
        task = case("weather_activity", variant=0)
        key = next(iter(task["initial_state"]["weather"]))
        task["initial_state"]["weather"][key] = "rain"
        env = run_reference(task)
        creates = [r for r in env._log if r["name"] == "create_event"]
        self.assertEqual(creates[0]["arguments"]["activity_id"], "stretch")
        self.assertFalse(env.evaluate()["success"], "目标仍为晴天分支时，不能冒充参考策略读到了目标")

    def test_delete_other_then_create_target_is_rejected(self):
        task = case("create", ordinal=5)
        env = TaskEnvironment(task)
        wrong_id = task["initial_state"]["events"][0]["event_id"]
        env.execute("delete_event", {"event_id": wrong_id})
        ReferencePolicy(task["reference_intent"]).solve(PublicStepOnly(env))
        self.assertFalse(env.evaluate()["success"])
        self.assertFalse(env.evaluate()["checks"]["AUTHORIZED_WRITES"])


class L456Tests(unittest.TestCase):
    def test_all_upper_families_and_held_out_combination(self):
        kinds = [k for k in BASE_KINDS if LEVELS[k] >= 4] + ["weather_reschedule_remind"]
        for kind in kinds:
            for ordinal in (0, 3, 5):
                for variant in range(4):
                    with self.subTest(kind=kind, ordinal=ordinal, variant=variant):
                        task = case(kind, ordinal, variant)
                        env = TaskEnvironment(task)
                        ReferencePolicy(task["reference_intent"]).solve(PublicStepOnly(env))
                        self.assertTrue(env.evaluate()["success"], env.evaluate())
                        counts, _ = audit_negatives(task, env)
                        if LEVELS[kind] == 4:
                            self.assertGreater(counts["guessed_missing_slot"], 0)
                            self.assertGreater(counts["posthoc_clarification"], 0)
                        if LEVELS[kind] == 6:
                            self.assertGreater(counts["ignored_followup"], 0)
                        self.assertTrue(replay(task, env.trajectory()).evaluate()["success"])

    def test_noop_does_not_complete_cancel_task(self):
        task = case("create_cancel")
        env = TaskEnvironment(task)
        env.step({"type": "final", "status": "success", "result": {"event_id": "invented"}, "message": "什么都没做。"})
        self.assertFalse(env.evaluate()["success"])

    def test_followup_not_triggered_by_failed_create(self):
        task = case("create_reschedule")
        event = copy.deepcopy(task["initial_state"]["events"][0])
        spec = task["goal"]["create_spec"]
        event.update(event_id="blocking", start_time=spec["start_time"], duration_minutes=spec["duration_minutes"])
        from datetime import timedelta
        from wearable_tasks.contract import stamp
        event["end_time"] = (stamp(event["start_time"])+timedelta(minutes=event["duration_minutes"])).isoformat()
        task["initial_state"]["events"].append(event)
        env = TaskEnvironment(task)
        result = env.step({"type": "tool_call", "name": "create_event", "arguments": spec})
        self.assertEqual(result["observation"]["tool_result"]["error_code"], "TIME_CONFLICT")
        self.assertFalse(env._followup_fired)
        self.assertNotIn("user_reply", result["observation"])

    def test_unasked_and_unrelated_slots_are_not_supplied(self):
        task = case("clarify_time")
        env = TaskEnvironment(task)
        observation = env.observation()
        self.assertNotIn(task["user_simulator"]["answers"]["start_time"], observation["messages"][-1]["content"])
        result = env.step({"type": "ask_user", "question": "喜欢哪种颜色？", "requested_slots": ["color"]})
        self.assertNotIn("slots", result["observation"]["user_reply"])
        self.assertEqual(env._answered, {})
